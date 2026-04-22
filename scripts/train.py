"""
Train functions for both euclidean and hyperbolic heads. 
"""

from __future__ import annotations
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import geoopt
except ModuleNotFoundError:
    geoopt = None

from dataset import FeatureDataset, NUM_CLASSES
from models import EuclideanHead, PoincareHead, ClassifierLayer


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "data" / "runs"


def _pick_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_loss_curves(losses: dict, run_dir: Path, title: str) -> Path:
    epochs = range(1, len(losses["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, losses["train_loss"], label="Train", marker="o")
    ax.plot(epochs, losses["val_loss"], label="Validation", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = run_dir / "loss.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def train_euclidean(
    dim: int = 8,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    lr_proto: float = 1e-2,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    seed: int = 2090,
    run_dir: Path | None = None,
    device: str | None = None,
) -> dict:
    _seed_all(seed)
    DEVICE = _pick_device(device)
    print(f"[euclidean d={dim}] device={DEVICE}")

    run_dir = Path(run_dir) if run_dir is not None else RUNS_DIR / f"euclidean_d{dim}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(FeatureDataset("train"), batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(FeatureDataset("val"), batch_size=batch_size,
                            shuffle=False)

    head = EuclideanHead(d_out=dim, dropout=dropout).to(DEVICE)
    clf = ClassifierLayer(num_classes=NUM_CLASSES, dim=dim, geometry="euclidean").to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    opt_head = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    opt_clf = torch.optim.Adam(clf.parameters(), lr=lr_proto, weight_decay=weight_decay)

    losses_path = run_dir / "losses.pkl"

    losses = {"train_loss": [], "val_loss": [], "val_top1": []}
    for epoch in range(1, epochs + 1):
        # --- Training ---
        head.train(); clf.train()
        epoch_loss, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt_head.zero_grad(set_to_none=True)
            opt_clf.zero_grad(set_to_none=True)
            logits = clf(head(xb))
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            opt_head.step()
            opt_clf.step()
            bs = yb.size(0)
            epoch_loss += loss.item() * bs
            n_seen += bs
        train_loss = epoch_loss / n_seen

        # --- Validation ---
        head.eval(); clf.eval()
        val_loss, correct, val_n = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = clf(head(xb))
                loss = criterion(logits, yb)
                bs = yb.size(0)
                val_loss += loss.item() * bs
                correct += (logits.argmax(dim=-1) == yb).sum().item()
                val_n += bs
        val_loss /= val_n
        val_top1 = correct / val_n

        losses["train_loss"].append(train_loss)
        losses["val_loss"].append(val_loss)
        losses["val_top1"].append(val_top1)
        with losses_path.open("wb") as f:
            pickle.dump(losses, f)
        print(f"Epoch {epoch}/{epochs} — train {train_loss:.4f} | val {val_loss:.4f} | top1 {val_top1:.4f}")

    ckpt_path = run_dir / "ckpt.pt"
    torch.save(
        {
            "head": head.state_dict(),
            "clf": clf.state_dict(),
            "epoch": epochs,
            "config": {
                "geometry": "euclidean",
                "dim": dim,
                "dropout": dropout,
                "curvature": None,
            },
        },
        ckpt_path,
    )
    print(f"[euclidean d={dim}] saved {ckpt_path}")

    plot_path = plot_loss_curves(losses, run_dir, title=f"Euclidean d={dim} — Training Loss")
    print(f"[euclidean d={dim}] saved {plot_path}")

    return {"losses": losses, "ckpt_path": str(ckpt_path), "run_dir": str(run_dir)}


def train_hyperbolic(
    dim: int = 8,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    lr_proto: float = 1e-2,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    curvature: float = 1.0,
    seed: int = 2090,
    run_dir: Path | None = None,
    device: str | None = None,
) -> dict:
    if geoopt is None:
        raise ImportError(
            "geoopt is required for hyperbolic training. Install it in the active environment first."
        )
    _seed_all(seed)
    DEVICE = _pick_device(device)
    print(f"[hyperbolic d={dim}] device={DEVICE}")

    run_dir = Path(run_dir) if run_dir is not None else RUNS_DIR / f"hyperbolic_d{dim}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(FeatureDataset("train"), batch_size=batch_size,
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(FeatureDataset("val"), batch_size=batch_size,
                            shuffle=False)

    head = PoincareHead(d_out=dim, dropout=dropout, curvature=curvature).to(DEVICE)
    clf = ClassifierLayer(num_classes=NUM_CLASSES, dim=dim,
                              geometry="hyperbolic", curvature=curvature).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    opt_head = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)

    opt_clf = geoopt.optim.RiemannianAdam(clf.parameters(), lr=lr_proto, stabilize=10)

    losses_path = run_dir / "losses.pkl"

    losses = {"train_loss": [], "val_loss": [], "val_top1": []}
    for epoch in range(1, epochs + 1):
        # --- Training ---
        head.train(); clf.train()
        epoch_loss, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt_head.zero_grad(set_to_none=True)
            opt_clf.zero_grad(set_to_none=True)
            logits = clf(head(xb))
            loss = criterion(logits, yb)
            loss.backward()
            # Clip head only; Riemannian optimizer handles manifold-safe steps itself.
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            opt_head.step()
            opt_clf.step()
            bs = yb.size(0)
            epoch_loss += loss.item() * bs
            n_seen += bs
        train_loss = epoch_loss / n_seen

        # --- Validation ---
        head.eval(); clf.eval()
        val_loss, correct, val_n = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = clf(head(xb))
                loss = criterion(logits, yb)
                bs = yb.size(0)
                val_loss += loss.item() * bs
                correct += (logits.argmax(dim=-1) == yb).sum().item()
                val_n += bs
        val_loss /= val_n
        val_top1 = correct / val_n

        losses["train_loss"].append(train_loss)
        losses["val_loss"].append(val_loss)
        losses["val_top1"].append(val_top1)
        with losses_path.open("wb") as f:
            pickle.dump(losses, f)
        print(f"Epoch {epoch}/{epochs} — train {train_loss:.4f} | val {val_loss:.4f} | top1 {val_top1:.4f}")

    ckpt_path = run_dir / "ckpt.pt"
    torch.save(
        {
            "head": head.state_dict(),
            "clf": clf.state_dict(),
            "epoch": epochs,
            "config": {
                "geometry": "hyperbolic",
                "dim": dim,
                "dropout": dropout,
                "curvature": curvature,
            },
        },
        ckpt_path,
    )
    print(f"[hyperbolic d={dim}] saved {ckpt_path}")

    plot_path = plot_loss_curves(losses, run_dir, title=f"Hyperbolic d={dim} — Training Loss")
    print(f"[hyperbolic d={dim}] saved {plot_path}")

    return {"losses": losses, "ckpt_path": str(ckpt_path), "run_dir": str(run_dir)}
