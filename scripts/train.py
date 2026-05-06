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
from hierarchy import distance_matrix, load_style_classes
from models import EuclideanHead, PoincareHead, ClassifierLayer, poincare_distance


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "data" / "runs"


def tree_regularizer(
    prototypes: torch.Tensor,
    tree_dists: torch.Tensor,
    geometry: str,
    curvature: float,
) -> torch.Tensor:
    """Scale-invariant MSE between prototype-prototype distances and tree distances.

    Each pair-distance vector is divided by its mean before the MSE so the
    regularizer enforces a *proportional* match (the geometry can pick its own
    scale) rather than fighting the classification loss for absolute size.
    """
    if geometry == "euclidean":
        d = torch.cdist(prototypes, prototypes)
    else:
        d = poincare_distance(
            prototypes.unsqueeze(1),
            prototypes.unsqueeze(0),
            curvature=curvature,
        )
    k = prototypes.shape[0]
    iu = torch.triu_indices(k, k, offset=1, device=prototypes.device)
    d_pairs = d[iu[0], iu[1]]
    t_pairs = tree_dists[iu[0], iu[1]].to(d_pairs.dtype)
    d_norm = d_pairs / d_pairs.mean().clamp_min(1e-12)
    t_norm = t_pairs / t_pairs.mean().clamp_min(1e-12)
    return ((d_norm - t_norm) ** 2).mean()


def _tree_dists_tensor(device: torch.device, hierarchy_name: str = "default") -> torch.Tensor:
    style_names = load_style_classes()
    t = distance_matrix(style_names, hierarchy_name=hierarchy_name)
    return torch.from_numpy(t).to(device).float()


def _class_weights_tensor(device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights, normalized so the mean weight is 1.

    Used when class_weighted=True so the cross-entropy loss treats rare
    leaf styles (Action_painting, Synthetic_Cubism, etc.) on par with
    common ones (Impressionism). Weights are derived from the training
    split labels; nothing else changes about the optimization.
    """
    train_labels = FeatureDataset("train").labels.numpy()
    counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(np.float64)
    counts = np.clip(counts, a_min=1.0, a_max=None)  # avoid div-by-zero on empties
    weights = (1.0 / counts) * (counts.sum() / NUM_CLASSES)
    return torch.from_numpy(weights).float().to(device)


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
    tree_loss_weight: float = 0.0,
    tree_hierarchy: str = "default",
    class_weighted: bool = False,
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
    cls_w = _class_weights_tensor(DEVICE) if class_weighted else None
    criterion = torch.nn.CrossEntropyLoss(weight=cls_w)
    opt_head = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    opt_clf = torch.optim.Adam(clf.parameters(), lr=lr_proto, weight_decay=weight_decay)

    tree_t = _tree_dists_tensor(DEVICE, tree_hierarchy) if tree_loss_weight > 0 else None

    losses_path = run_dir / "losses.pkl"

    losses = {"train_loss": [], "val_loss": [], "val_top1": [], "train_ce": [], "train_tree": []}
    for epoch in range(1, epochs + 1):
        # --- Training ---
        head.train(); clf.train()
        epoch_loss, epoch_ce, epoch_tree, n_seen = 0.0, 0.0, 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt_head.zero_grad(set_to_none=True)
            opt_clf.zero_grad(set_to_none=True)
            logits = clf(head(xb))
            ce = criterion(logits, yb)
            if tree_t is not None:
                tree_term = tree_regularizer(clf.prototypes, tree_t, "euclidean", 1.0)
                loss = ce + tree_loss_weight * tree_term
            else:
                tree_term = torch.tensor(0.0, device=DEVICE)
                loss = ce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            opt_head.step()
            opt_clf.step()
            bs = yb.size(0)
            epoch_loss += loss.item() * bs
            epoch_ce += ce.item() * bs
            epoch_tree += tree_term.item() * bs
            n_seen += bs
        train_loss = epoch_loss / n_seen
        train_ce = epoch_ce / n_seen
        train_tree = epoch_tree / n_seen

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
        losses["train_ce"].append(train_ce)
        losses["train_tree"].append(train_tree)
        losses["val_loss"].append(val_loss)
        losses["val_top1"].append(val_top1)
        with losses_path.open("wb") as f:
            pickle.dump(losses, f)
        if tree_t is not None:
            print(f"Epoch {epoch}/{epochs} — train {train_loss:.4f} (ce {train_ce:.4f} + λ·tree {tree_loss_weight}·{train_tree:.4f}) | val {val_loss:.4f} | top1 {val_top1:.4f}")
        else:
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
                "tree_loss_weight": tree_loss_weight,
                "tree_hierarchy": tree_hierarchy,
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
    tree_loss_weight: float = 0.0,
    tree_hierarchy: str = "default",
    class_weighted: bool = False,
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
    cls_w = _class_weights_tensor(DEVICE) if class_weighted else None
    criterion = torch.nn.CrossEntropyLoss(weight=cls_w)
    opt_head = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)

    opt_clf = geoopt.optim.RiemannianAdam(clf.parameters(), lr=lr_proto, stabilize=10)

    tree_t = _tree_dists_tensor(DEVICE, tree_hierarchy) if tree_loss_weight > 0 else None

    losses_path = run_dir / "losses.pkl"

    losses = {"train_loss": [], "val_loss": [], "val_top1": [], "train_ce": [], "train_tree": []}
    for epoch in range(1, epochs + 1):
        # --- Training ---
        head.train(); clf.train()
        epoch_loss, epoch_ce, epoch_tree, n_seen = 0.0, 0.0, 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt_head.zero_grad(set_to_none=True)
            opt_clf.zero_grad(set_to_none=True)
            logits = clf(head(xb))
            ce = criterion(logits, yb)
            if tree_t is not None:
                tree_term = tree_regularizer(clf.prototypes, tree_t, "hyperbolic", curvature)
                loss = ce + tree_loss_weight * tree_term
            else:
                tree_term = torch.tensor(0.0, device=DEVICE)
                loss = ce
            loss.backward()
            # Clip head only; Riemannian optimizer handles manifold-safe steps itself.
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            opt_head.step()
            opt_clf.step()
            bs = yb.size(0)
            epoch_loss += loss.item() * bs
            epoch_ce += ce.item() * bs
            epoch_tree += tree_term.item() * bs
            n_seen += bs
        train_loss = epoch_loss / n_seen
        train_ce = epoch_ce / n_seen
        train_tree = epoch_tree / n_seen

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
        losses["train_ce"].append(train_ce)
        losses["train_tree"].append(train_tree)
        losses["val_loss"].append(val_loss)
        losses["val_top1"].append(val_top1)
        with losses_path.open("wb") as f:
            pickle.dump(losses, f)
        if tree_t is not None:
            print(f"Epoch {epoch}/{epochs} — train {train_loss:.4f} (ce {train_ce:.4f} + λ·tree {tree_loss_weight}·{train_tree:.4f}) | val {val_loss:.4f} | top1 {val_top1:.4f}")
        else:
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
                "tree_loss_weight": tree_loss_weight,
                "tree_hierarchy": tree_hierarchy,
            },
        },
        ckpt_path,
    )
    print(f"[hyperbolic d={dim}] saved {ckpt_path}")

    plot_path = plot_loss_curves(losses, run_dir, title=f"Hyperbolic d={dim} — Training Loss")
    print(f"[hyperbolic d={dim}] saved {plot_path}")

    return {"losses": losses, "ckpt_path": str(ckpt_path), "run_dir": str(run_dir)}
