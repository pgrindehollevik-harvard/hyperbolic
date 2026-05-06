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
        idx = (agg[(metric_for_selection, "mean")].idxmax()
               if higher else agg[(metric_for_selection, "mean")].idxmin())
        best = agg.loc[idx]
        config = dict(zip(cluster_keys, idx))
        flat = {f"{m}_mean": float(best[(m, "mean")]) for m in metrics}
        flat.update({f"{m}_std": float(best[(m, "std")]) for m in metrics})
        rows.append({"geometry": geo, **config, **flat})

    return pd.DataFrame(rows).set_index("geometry")


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
