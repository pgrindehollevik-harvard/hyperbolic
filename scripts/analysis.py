"""
Helpers for analyzing sweep results.

Used by `notebooks/ms4_main.ipynb` and `scripts/make_figures.py`. Keeps the
notebook's analysis cells short and the figure-generation deterministic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_CSV = REPO_ROOT / "data" / "runs" / "sweep.csv"


HEADLINE_METRICS: dict[str, str] = {
    "top1_accuracy": "Top-1",
    "top5_accuracy": "Top-5",
    "balanced_accuracy": "Balanced acc.",
    "class_center_tree_spearman": "Class-center / tree Spearman",
    "tree_distortion_average": "Avg. tree distortion",
    "tree_distortion_worst_case": "Worst tree distortion",
    "dendrogram_cluster_f1": "Dendrogram F1",
    "knn_siblings_recall_at_5": "Sibling recall@5",
    "knn_cousins_recall_at_5": "Cousin recall@5",
    "frechet_nearest_prototype_accuracy": "Fréchet (Cubism) acc.",
}

HIGHER_IS_BETTER: set[str] = {
    "top1_accuracy", "top5_accuracy", "balanced_accuracy",
    "class_center_tree_spearman", "prototype_tree_spearman",
    "dendrogram_cluster_f1", "frechet_nearest_prototype_accuracy",
}
HIGHER_IS_BETTER.update({k for k in HEADLINE_METRICS if k.startswith("knn_")})

# Lower-is-better metrics (distortion, distance summaries, val loss).
LOWER_IS_BETTER: set[str] = {
    "tree_distortion_average", "tree_distortion_worst_case",
    "mean_tree_distance_all_predictions", "mean_tree_distance_mistakes",
    "loss",
}


def load_sweep(csv_path: Path | str = DEFAULT_SWEEP_CSV) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"sweep CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    # `tree_loss_weight` and `curvature` may have been written as strings in
    # rows where pandas mixed types — coerce.
    for col in ("tree_loss_weight", "curvature", "lr", "lr_proto", "weight_decay", "dropout"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def summarize_by_config(
    df: pd.DataFrame,
    group_keys: list[str],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Mean ± std across seeds, grouped by `group_keys`.

    Returns a flat-column DataFrame: each metric becomes `<metric>_mean` and
    `<metric>_std`. Adds an `n_seeds` column.
    """
    if metrics is None:
        metrics = list(HEADLINE_METRICS.keys())
    available = [m for m in metrics if m in df.columns]
    grouped = df.groupby(group_keys, dropna=False)[available].agg(["mean", "std", "count"])
    grouped.columns = [f"{m}_{stat}" for m, stat in grouped.columns]
    grouped = grouped.reset_index()
    # `count` will be the same across metrics within a group (assuming no
    # missing values), so use the first metric's count as n_seeds.
    if available:
        grouped["n_seeds"] = grouped[f"{available[0]}_count"]
        grouped = grouped.drop(columns=[c for c in grouped.columns if c.endswith("_count")])
    return grouped


def best_per_geometry_dim(
    df: pd.DataFrame,
    metric: str = "top1_accuracy",
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """For hyperbolic, pick the best curvature at each (dim, seed) before averaging.

    Returns one row per (geometry, dim) with mean ± std over seeds.
    """
    rows = []
    for geo in ("euclidean", "hyperbolic"):
        sub = df[df["geometry"] == geo]
        if sub.empty:
            continue
        if geo == "hyperbolic":
            picker = sub.loc[
                sub.groupby(["dim", "seed"])[metric].agg(
                    "idxmax" if higher_is_better else "idxmin"
                )
            ]
        else:
            picker = sub
        agg = picker.groupby("dim")[metric].agg(["mean", "std", "count"]).reset_index()
        agg["geometry"] = geo
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def plot_scaling(
    df: pd.DataFrame,
    metrics: Iterable[str] = ("top1_accuracy", "class_center_tree_spearman", "knn_siblings_recall_at_5"),
    figsize: tuple[float, float] = (15, 4.2),
) -> plt.Figure:
    """Phase-1-style scaling figure: mean ± std vs d, by geometry, hyperbolic
    using best curvature per d/seed."""
    metrics = list(metrics)
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        higher = metric in HIGHER_IS_BETTER
        best = best_per_geometry_dim(df, metric=metric, higher_is_better=higher)
        for geo, marker in [("euclidean", "o"), ("hyperbolic", "s")]:
            sub = best[best["geometry"] == geo]
            if sub.empty:
                continue
            ax.errorbar(sub["dim"], sub["mean"], yerr=sub["std"],
                        marker=marker, label=geo, capsize=3, linewidth=1.6)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df["dim"].unique()))
        ax.set_xticklabels([str(int(d)) for d in sorted(df["dim"].unique())])
        ax.set_xlabel("embedding dim d")
        label = HEADLINE_METRICS.get(metric, metric)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Scaling with d (hyperbolic = best curvature per d)")
    fig.tight_layout()
    return fig


def plot_lambda_sweep(
    df: pd.DataFrame,
    dim: int = 8,
    curvature: float = 1.0,
    metrics: Iterable[str] = ("top1_accuracy", "class_center_tree_spearman",
                              "tree_distortion_average", "knn_siblings_recall_at_5"),
    figsize: tuple[float, float] = (18, 4.2),
) -> plt.Figure:
    """Phase-2-style λ sweep at fixed d, c."""
    sub = df[(df["dim"] == dim) & (df["tree_hierarchy"] == "default")].copy()
    sub_hy = sub[(sub["geometry"] == "hyperbolic") & (sub["curvature"] == curvature)]
    sub_eu = sub[sub["geometry"] == "euclidean"]
    metrics = list(metrics)
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        for label, sub_geo, marker in [("euclidean", sub_eu, "o"),
                                       ("hyperbolic", sub_hy, "s")]:
            if sub_geo.empty:
                continue
            agg = sub_geo.groupby("tree_loss_weight")[metric].agg(["mean", "std"]).reset_index()
            ax.errorbar(agg["tree_loss_weight"], agg["mean"], yerr=agg["std"],
                        marker=marker, label=label, capsize=3, linewidth=1.6)
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_xlabel("λ (tree loss weight)")
        nice = HEADLINE_METRICS.get(metric, metric)
        ax.set_ylabel(nice)
        ax.set_title(nice)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Hierarchy-aware loss at d={dim}, c={curvature}")
    fig.tight_layout()
    return fig


def headline_table(
    df: pd.DataFrame,
    metric_for_selection: str = "top1_accuracy",
    metrics: Iterable[str] | None = None,
) -> pd.DataFrame:
    """For each geometry, pick the (d, c, λ, tree) cluster with the best
    `metric_for_selection` (mean over seeds) and report all `metrics`.
    """
    if metrics is None:
        metrics = list(HEADLINE_METRICS.keys())
    metrics = [m for m in metrics if m in df.columns]
    cluster_keys = ["dim", "curvature", "tree_loss_weight", "tree_hierarchy"]

    rows = []
    for geo in ("euclidean", "hyperbolic"):
        sub = df[df["geometry"] == geo]
        if sub.empty:
            continue
        agg = sub.groupby(cluster_keys, dropna=False)[metrics].agg(["mean", "std"])
        higher = metric_for_selection in HIGHER_IS_BETTER
        # Use positional indexing to avoid pandas MultiIndex .loc[] choking on
        # NaN curvature for euclidean configs.
        scores = agg[(metric_for_selection, "mean")].to_numpy()
        pos = int(np.nanargmax(scores)) if higher else int(np.nanargmin(scores))
        best = agg.iloc[pos]
        idx = agg.index[pos]
        config = dict(zip(cluster_keys, idx))
        flat = {f"{m}_mean": float(best[(m, "mean")]) for m in metrics}
        flat.update({f"{m}_std": float(best[(m, "std")]) for m in metrics})
        rows.append({"geometry": geo, **config, **flat})

    return pd.DataFrame(rows).set_index("geometry")


def _embed_split(ckpt_path: Path, split: str = "val", batch_size: int = 4096,
                 device: str | None = None, max_samples: int | None = None):
    """Load a checkpoint and return (embeddings, labels, prototypes, metadata)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from torch.utils.data import DataLoader

    from dataset import FeatureDataset
    from eval import evaluate, load_model, pick_device

    dev = pick_device(device)
    _, head, clf, metadata = load_model(
        ckpt_path=ckpt_path, geometry_override=None, curvature_override=None, device=dev,
    )
    dataset = FeatureDataset(split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    _, y_true, _, _, embeddings = evaluate(head, clf, loader, dev)
    if max_samples is not None and len(y_true) > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y_true), size=max_samples, replace=False)
        embeddings = embeddings[idx]
        y_true = y_true[idx]
    prototypes = clf.prototypes.detach().cpu().numpy()
    return embeddings, y_true, prototypes, metadata


def plot_poincare_disk(ckpt_path: Path, split: str = "val", max_samples: int = 800,
                       device: str | None = None,
                       figsize: tuple[float, float] = (8, 8)) -> plt.Figure:
    """2D scatter of embeddings + prototypes. Requires d=2 checkpoint.
    Draws the unit circle for hyperbolic; pure scatter for Euclidean.
    """
    embeddings, labels, prototypes, meta = _embed_split(
        ckpt_path, split=split, device=device, max_samples=max_samples,
    )
    if embeddings.shape[1] != 2:
        raise ValueError(f"plot_poincare_disk requires d=2, got d={embeddings.shape[1]}")

    from hierarchy import load_style_classes
    style_names = load_style_classes()
    n_classes = len(style_names)

    cmap = plt.get_cmap("tab20", n_classes)
    fig, ax = plt.subplots(figsize=figsize)
    if meta["geometry"] == "hyperbolic":
        c = float(meta["curvature"])
        radius = 1.0 / (c ** 0.5)
        circle = plt.Circle((0, 0), radius, fill=False, color="black",
                            linewidth=1.5, linestyle="--")
        ax.add_patch(circle)

    for k in range(n_classes):
        mask = labels == k
        if mask.any():
            ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                       s=6, alpha=0.35, color=cmap(k), label=None)
    for k in range(n_classes):
        ax.scatter(prototypes[k, 0], prototypes[k, 1],
                   s=110, color=cmap(k), edgecolor="black", linewidth=0.8, zorder=5)
        ax.annotate(style_names[k][:10], (prototypes[k, 0], prototypes[k, 1]),
                    fontsize=7, ha="center", va="center", zorder=6)

    ax.set_aspect("equal")
    ax.set_title(f"{meta['geometry']} d=2"
                 + (f", c={meta['curvature']}" if meta['geometry'] == 'hyperbolic' else ""))
    ax.set_xlabel("dim 0")
    ax.set_ylabel("dim 1")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def tree_leaf_order(hierarchy_name: str = "default") -> list[int]:
    """Return a label-index ordering produced by a depth-first traversal of
    the tree, so confusion matrices reordered by this list have block
    structure aligned with the hierarchy."""
    from hierarchy import get_hierarchy, load_style_classes
    style_names = load_style_classes()
    style_to_idx = {s: i for i, s in enumerate(style_names)}
    hierarchy = get_hierarchy(hierarchy_name)
    order: list[int] = []

    def dfs(node: str) -> None:
        if node in style_to_idx:
            order.append(style_to_idx[node])
        for child in hierarchy.get(node, []):
            dfs(child)

    dfs("Root")
    # Add any styles that weren't reached (shouldn't happen for default; will
    # for hierarchies that omit some styles).
    for i in range(len(style_names)):
        if i not in order:
            order.append(i)
    return order


def plot_confusion_block_tree(
    ckpt_path: Path, split: str = "val", device: str | None = None,
    hierarchy_name: str = "default", figsize: tuple[float, float] = (10, 9),
) -> plt.Figure:
    """Confusion matrix re-ordered by a tree-DFS leaf order, so on-tree
    mistakes cluster near the diagonal."""
    import torch
    from sklearn.metrics import confusion_matrix
    from torch.utils.data import DataLoader

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from dataset import FeatureDataset, NUM_CLASSES
    from eval import evaluate, load_model, pick_device
    from hierarchy import load_style_classes

    dev = pick_device(device)
    _, head, clf, _ = load_model(
        ckpt_path=ckpt_path, geometry_override=None, curvature_override=None, device=dev,
    )
    loader = DataLoader(FeatureDataset(split), batch_size=4096, shuffle=False)
    _, y_true, y_pred, _, _ = evaluate(head, clf, loader, dev)
    style_names = load_style_classes()
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    order = tree_leaf_order(hierarchy_name)
    cm_ord = cm[np.ix_(order, order)]
    labels_ord = [style_names[i] for i in order]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_ord, cmap="magma_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels_ord)))
    ax.set_yticks(range(len(labels_ord)))
    ax.set_xticklabels(labels_ord, rotation=90, fontsize=7)
    ax.set_yticklabels(labels_ord, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Row-normalized confusion, tree-ordered ({hierarchy_name})")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    return fig


def plot_knn_image_grid(
    ckpt_path: Path,
    query_indices: list[int] | None = None,
    n_queries: int = 4,
    k: int = 5,
    split: str = "val",
    device: str | None = None,
    seed: int = 0,
    wikiart_dir: Path | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """For each query, display the query painting next to its k nearest
    neighbors in embedding space. Requires data/wikiart/ for raw images.

    The figure has n_queries rows and k+1 columns: [query, NN-1, ..., NN-k].
    """
    from PIL import Image
    import sys as _sys
    import torch
    _sys.path.insert(0, str(Path(__file__).resolve().parent))

    from eval import load_split_metadata
    from hierarchy import load_style_classes
    from models import poincare_distance

    embeddings, labels, _, meta = _embed_split(ckpt_path, split=split, device=device)
    split_df = load_split_metadata(split)

    if wikiart_dir is None:
        # The download produces data/wikiart/wikiart/<style>/... — the outer
        # directory comes from the zip name, the inner one from the dataset.
        candidates = [REPO_ROOT / "data" / "wikiart" / "wikiart",
                      REPO_ROOT / "data" / "wikiart"]
        wikiart_dir = next((c for c in candidates if c.exists()), candidates[0])
    wikiart_dir = Path(wikiart_dir)

    rng = np.random.default_rng(seed)
    if query_indices is None:
        # Pick n_queries from distinct, well-populated styles for variety.
        query_indices = []
        for label in rng.permutation(int(labels.max()) + 1):
            mask = np.where(labels == label)[0]
            if len(mask) == 0:
                continue
            query_indices.append(int(rng.choice(mask)))
            if len(query_indices) >= n_queries:
                break
    query_indices = list(query_indices)
    n_queries = len(query_indices)

    embeddings_t = torch.from_numpy(embeddings)
    q_emb = embeddings_t[query_indices]
    if meta["geometry"] == "euclidean":
        dists = torch.cdist(q_emb, embeddings_t)
    else:
        c = float(meta["curvature"])
        dists = poincare_distance(
            q_emb.unsqueeze(1), embeddings_t.unsqueeze(0), curvature=c,
        )

    for row, q_idx in enumerate(query_indices):
        dists[row, q_idx] = float("inf")
    topk = dists.topk(k, dim=1, largest=False).indices.numpy()

    style_names = load_style_classes()

    def _load(path):
        img = Image.open(wikiart_dir / path).convert("RGB")
        img.thumbnail((180, 180))
        return img

    if figsize is None:
        figsize = (1.7 * (k + 1), 1.9 * n_queries)
    fig, axes = plt.subplots(n_queries, k + 1, figsize=figsize)
    if n_queries == 1:
        axes = axes[None, :]

    for row, q_idx in enumerate(query_indices):
        q_path = split_df.iloc[q_idx]["path"]
        axes[row, 0].imshow(_load(q_path))
        axes[row, 0].set_title(f"query · {style_names[int(labels[q_idx])]}", fontsize=7)
        axes[row, 0].axis("off")
        for col, n_idx in enumerate(topk[row]):
            n_path = split_df.iloc[int(n_idx)]["path"]
            axes[row, col + 1].imshow(_load(n_path))
            axes[row, col + 1].set_title(style_names[int(labels[int(n_idx)])], fontsize=7)
            axes[row, col + 1].axis("off")

    fig.suptitle(
        f"{meta['geometry']} d={meta['dim']}"
        + (f", c={meta['curvature']}" if meta["geometry"] == "hyperbolic" else "")
        + f" — kNN (k={k})",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def format_headline_for_report(table: pd.DataFrame) -> pd.DataFrame:
    """Convert the headline table to a "Eu vs Hy" mean±std string format."""
    metric_keys = list(HEADLINE_METRICS.keys())
    out_rows = []
    for metric in metric_keys:
        row = {"metric": HEADLINE_METRICS[metric]}
        for geo in table.index:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if mean_col in table.columns and not np.isnan(table.at[geo, mean_col]):
                row[geo] = f"{table.at[geo, mean_col]:.3f} ± {table.at[geo, std_col]:.3f}"
            else:
                row[geo] = "—"
        out_rows.append(row)
    return pd.DataFrame(out_rows).set_index("metric")
