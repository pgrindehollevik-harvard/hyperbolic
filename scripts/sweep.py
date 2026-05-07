"""
Grid-search runner.

Iterates a list of training configs, trains each, evaluates each, appends a row
to data/runs/sweep.csv. Resumable — re-running skips configs whose hash is
already in the CSV.

Each row stores the config + every metric returned by `eval.run_evaluation`.
For "winner" configs you can re-run `scripts/eval.py` to dump the per-run
artifacts (CSVs, plots) without redoing training.

Usage
-----
# Run the default Phase 1 grid (curvature x dim x seed):
python scripts/sweep.py --phase 1

# Run the Phase 2 hierarchy-aware-loss grid:
python scripts/sweep.py --phase 2

# Custom config: define a list of dicts in code and pass --custom <fn>.

The CSV columns will be a superset of all configs and metrics seen so far —
pandas handles the union when we rewrite it on each iteration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from itertools import product
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from eval import run_evaluation
from train import train_euclidean, train_hyperbolic


REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = REPO_ROOT / "data" / "runs" / "sweep"
SWEEP_CSV = REPO_ROOT / "data" / "runs" / "sweep.csv"


CONFIG_KEYS = (
    "geometry",
    "dim",
    "curvature",
    "epochs",
    "batch_size",
    "lr",
    "lr_proto",
    "weight_decay",
    "dropout",
    "seed",
    "tree_loss_weight",
    "tree_hierarchy",
)


def normalize_config(cfg: dict) -> dict:
    """Fill defaults; force curvature to None for euclidean (so it doesn't
    contribute to the hash and produce duplicate runs)."""
    out = {
        "geometry": cfg.get("geometry"),
        "dim": int(cfg.get("dim", 8)),
        "curvature": cfg.get("curvature", 1.0),
        "epochs": int(cfg.get("epochs", 30)),
        "batch_size": int(cfg.get("batch_size", 4096)),
        "lr": float(cfg.get("lr", 1e-3)),
        "lr_proto": float(cfg.get("lr_proto", 1e-2)),
        "weight_decay": float(cfg.get("weight_decay", 1e-4)),
        "dropout": float(cfg.get("dropout", 0.1)),
        "seed": int(cfg.get("seed", 0)),
        "tree_loss_weight": float(cfg.get("tree_loss_weight", 0.0)),
        "tree_hierarchy": str(cfg.get("tree_hierarchy", "default")),
    }
    if out["geometry"] == "euclidean":
        out["curvature"] = None
    return out


def config_hash(cfg: dict) -> str:
    cfg = normalize_config(cfg)
    serialized = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()[:10]


def grid(space: dict) -> list[dict]:
    keys = list(space.keys())
    values = [space[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def dedup(configs: list[dict]) -> list[dict]:
    seen, out = set(), []
    for cfg in configs:
        cfg = normalize_config(cfg)
        h = config_hash(cfg)
        if h in seen:
            continue
        seen.add(h)
        out.append(cfg)
    return out


def load_done(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return set(df["config_hash"].astype(str).tolist())


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        df = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)


def run_one(cfg: dict, device: str | None = None) -> dict:
    cfg = normalize_config(cfg)
    h = config_hash(cfg)
    run_dir = SWEEP_DIR / h
    run_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs = dict(
        dim=cfg["dim"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        lr_proto=cfg["lr_proto"],
        weight_decay=cfg["weight_decay"],
        dropout=cfg["dropout"],
        seed=cfg["seed"],
        tree_loss_weight=cfg["tree_loss_weight"],
        tree_hierarchy=cfg["tree_hierarchy"],
        run_dir=run_dir,
        device=device,
    )

    t0 = time.time()
    if cfg["geometry"] == "hyperbolic":
        train_kwargs["curvature"] = cfg["curvature"]
        out = train_hyperbolic(**train_kwargs)
    elif cfg["geometry"] == "euclidean":
        out = train_euclidean(**train_kwargs)
    else:
        raise ValueError(f"Unknown geometry: {cfg['geometry']}")
    train_seconds = time.time() - t0

    metrics = run_evaluation(
        ckpt_path=Path(out["ckpt_path"]),
        split="val",
        device=device,
        output_dir=run_dir / "eval_val",
        save_artifacts=False,
    )

    # Spread order: metrics first, then cfg, so the config we *requested* wins
    # on shared keys (e.g. curvature=None for euclidean, which eval.py coerces
    # to 1.0 for the model load). Otherwise we'd hash on None but log 1.0.
    row = {"config_hash": h, **metrics, **cfg, "train_seconds": train_seconds}
    return row


# ----------------------------------------------------------------------------
# Predefined sweeps
# ----------------------------------------------------------------------------

def phase1_space() -> list[dict]:
    """Phase 1: scale dim and curvature, no hierarchy-aware loss, three seeds."""
    eu = grid({
        "geometry": ["euclidean"],
        "dim": [2, 4, 8, 16, 32, 64],
        "curvature": [None],
        "epochs": [30],
        "batch_size": [4096],
        "lr": [1e-3],
        "lr_proto": [1e-2],
        "weight_decay": [1e-4],
        "dropout": [0.1],
        "seed": [0, 1, 2],
        "tree_loss_weight": [0.0],
        "tree_hierarchy": ["default"],
    })
    hy = grid({
        "geometry": ["hyperbolic"],
        "dim": [2, 4, 8, 16, 32, 64],
        "curvature": [0.1, 0.3, 1.0, 3.0],
        "epochs": [30],
        "batch_size": [4096],
        "lr": [1e-3],
        "lr_proto": [1e-2],
        "weight_decay": [1e-4],
        "dropout": [0.1],
        "seed": [0, 1, 2],
        "tree_loss_weight": [0.0],
        "tree_hierarchy": ["default"],
    })
    return dedup(eu + hy)


def phase2_space() -> list[dict]:
    """Phase 2: hierarchy-aware loss, sweep λ at the d=8 / c=1 sweet spot.

    We use d=8 because it's where the cross-entropy baseline reported results; if hyperbolic catches up
    here, that's the cleanest comparison. We also test d=16 and d=32 with the
    best λ later (planned in a separate small sweep).
    """
    space = {
        "geometry": ["euclidean", "hyperbolic"],
        "dim": [8, 16],
        "curvature": [1.0],
        "epochs": [30],
        "batch_size": [4096],
        "lr": [1e-3],
        "lr_proto": [1e-2],
        "weight_decay": [1e-4],
        "dropout": [0.1],
        "seed": [0, 1, 2],
        "tree_loss_weight": [0.0, 0.1, 0.3, 1.0, 3.0],
        "tree_hierarchy": ["default"],
    }
    return dedup(grid(space))


def phase3_space() -> list[dict]:
    """Phase 3: re-run best Phase 1 configs against alternative trees.

    The eval is hierarchy-agnostic at scoring time (it always uses 'default'),
    so this phase is about *training* against different trees with the
    hierarchy-aware loss and seeing whether tree choice changes which geometry
    wins. We keep λ fixed at a moderate value.
    """
    space = {
        "geometry": ["euclidean", "hyperbolic"],
        "dim": [8],
        "curvature": [1.0],
        "epochs": [30],
        "batch_size": [4096],
        "lr": [1e-3],
        "lr_proto": [1e-2],
        "weight_decay": [1e-4],
        "dropout": [0.1],
        "seed": [0, 1, 2],
        "tree_loss_weight": [1.0],
        "tree_hierarchy": ["default", "chronological", "flat"],
    }
    return dedup(grid(space))


def smoke_space() -> list[dict]:
    """Tiny 4-config grid for smoke-testing the runner end-to-end."""
    return dedup(grid({
        "geometry": ["euclidean", "hyperbolic"],
        "dim": [8],
        "curvature": [1.0],
        "epochs": [2],
        "batch_size": [4096],
        "lr": [1e-3],
        "lr_proto": [1e-2],
        "weight_decay": [1e-4],
        "dropout": [0.1],
        "seed": [0],
        "tree_loss_weight": [0.0, 1.0],
        "tree_hierarchy": ["default"],
    }))


PHASES = {
    "1": phase1_space,
    "2": phase2_space,
    "3": phase3_space,
    "smoke": smoke_space,
}


def run_sweep(configs: list[dict], device: str | None, csv_path: Path) -> None:
    configs = dedup(configs)
    done = load_done(csv_path)
    todo = [c for c in configs if config_hash(c) not in done]

    print(f"[sweep] {len(configs)} total configs, {len(done)} done, {len(todo)} remaining")
    print(f"[sweep] writing rows to {csv_path}")

    for i, cfg in enumerate(todo, 1):
        h = config_hash(cfg)
        print(f"\n[sweep] {i}/{len(todo)} hash={h}")
        print(f"[sweep] cfg={cfg}")
        try:
            row = run_one(cfg, device=device)
        except Exception as exc:  # noqa: BLE001
            print(f"[sweep] FAILED hash={h}: {exc}")
            traceback.print_exc()
            row = {"config_hash": h, **normalize_config(cfg), "error": str(exc)}
        append_row(csv_path, row)
        print(f"[sweep] appended hash={h}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=list(PHASES.keys()), default="smoke")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--csv", type=Path, default=SWEEP_CSV)
    args = parser.parse_args()

    configs = PHASES[args.phase]()
    run_sweep(configs, device=args.device, csv_path=args.csv)


if __name__ == "__main__":
    main()
