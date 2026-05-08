"""Regenerate the SPIGM-submission body figures with ICML two-column layout.

Body figures in docs/figures/ were originally rendered for a single-column
1-inch-margin article (~6.5" wide). When dropped into ICML 2026's
two-column layout (3.25" per column), embedded fonts shrink below 4pt.
This script regenerates:

  - phase1_scaling.pdf   (3-panel, 7"x2.3", figure*)
  - phase2_lambda.pdf    (3-panel body version, drop tree-distortion)
  - phase2_lambda_full.pdf (4-panel for appendix)
  - phase3_tree_variants.pdf (3-panel body)
  - phase3_tree_variants_full.pdf (4-panel for appendix)
  - pareto.pdf           (1-panel, 3.25"x2.8", single column)

at display-matched dimensions with 7-9pt fonts. Output goes to
`docs_SPIGM/figures/`. Source data: `notebooks/sweep_results.csv`
(committed) plus two hardcoded baseline values for pareto (logreg top-1
0.641, kNN-5 (0.635, 0.160)) read off the existing pareto figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analysis import (  # noqa: E402
    HEADLINE_METRICS,
    plot_lambda_sweep,
    plot_scaling,
)

SWEEP_CSV = REPO_ROOT / "notebooks" / "sweep_results.csv"
FIG_DIR = REPO_ROOT / "docs_SPIGM" / "figures"

# Baseline numbers shown on pareto. Read off existing pareto.pdf legend so
# we don't need a fresh baselines.csv.
LOGREG_TOP1 = 0.641
KNN_TOP1 = 0.635
KNN_SIBLING_RECALL = 0.160


ICML_RC = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def _save(fig: plt.Figure, name: str) -> None:
    out = FIG_DIR / name
    fig.savefig(out)
    print(f"[figures-spigm] wrote {out.relative_to(REPO_ROOT)}")
    plt.close(fig)


def _load() -> pd.DataFrame:
    if not SWEEP_CSV.exists():
        raise SystemExit(f"sweep CSV not found at {SWEEP_CSV}")
    df = pd.read_csv(SWEEP_CSV)
    for col in ("tree_loss_weight", "curvature", "lr", "lr_proto",
                "weight_decay", "dropout"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def make_phase1(df: pd.DataFrame) -> None:
    sub = df[(df["tree_loss_weight"] == 0.0) & (df["tree_hierarchy"] == "default")]
    if sub.empty:
        print("[figures-spigm] phase1: no rows — skipping")
        return
    fig = plot_scaling(sub, figsize=(7.0, 2.3))
    _save(fig, "phase1_scaling.pdf")


def make_phase2(df: pd.DataFrame) -> None:
    sub = df[(df["tree_hierarchy"] == "default") & (df["dim"].isin([8, 16]))]
    sub = sub[sub["tree_loss_weight"].isin([0.0, 0.1, 0.3, 1.0, 3.0])]
    if sub.empty:
        print("[figures-spigm] phase2: no rows — skipping")
        return
    body_metrics = (
        "top1_accuracy",
        "class_center_tree_spearman",
        "knn_siblings_recall_at_5",
    )
    fig = plot_lambda_sweep(sub, dim=8, curvature=1.0,
                            metrics=body_metrics, figsize=(7.0, 2.3))
    _save(fig, "phase2_lambda.pdf")

    full_metrics = (
        "top1_accuracy",
        "class_center_tree_spearman",
        "tree_distortion_average",
        "knn_siblings_recall_at_5",
    )
    fig = plot_lambda_sweep(sub, dim=8, curvature=1.0,
                            metrics=full_metrics, figsize=(7.0, 2.3))
    _save(fig, "phase2_lambda_full.pdf")


def _phase3_panel(ax, agg: pd.DataFrame, metric: str) -> None:
    trees = ["default", "chronological", "flat"]
    width = 0.35
    x = list(range(len(trees)))
    for offset, geo in [(-width / 2, "euclidean"), (width / 2, "hyperbolic")]:
        means, stds = [], []
        for tree in trees:
            row = agg[(agg["tree_hierarchy"] == tree) & (agg["geometry"] == geo)]
            if row.empty:
                means.append(0.0)
                stds.append(0.0)
            else:
                means.append(float(row[(metric, "mean")].iloc[0]))
                stds.append(float(row[(metric, "std")].iloc[0]))
        ax.bar([xi + offset for xi in x], means, width=width, yerr=stds,
               capsize=2, label=geo)
    ax.set_xticks(x)
    ax.set_xticklabels(trees, rotation=15)
    ax.set_title(HEADLINE_METRICS.get(metric, metric))
    ax.set_xlabel("training tree")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()


def _phase3_render(df: pd.DataFrame, metrics, figsize) -> plt.Figure:
    sub = df[(df["tree_loss_weight"] == 1.0) & (df["dim"] == 8)]
    metrics = [m for m in metrics if m in sub.columns]
    agg = (
        sub.groupby(["tree_hierarchy", "geometry"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        _phase3_panel(ax, agg, metric)
    fig.suptitle("Sensitivity of geometry winner to training tree")
    fig.tight_layout()
    return fig


def make_phase3(df: pd.DataFrame) -> None:
    sub = df[(df["tree_loss_weight"] == 1.0) & (df["dim"] == 8)]
    if sub.empty:
        print("[figures-spigm] phase3: no rows — skipping")
        return
    body_metrics = [
        "top1_accuracy",
        "class_center_tree_spearman",
        "knn_siblings_recall_at_5",
    ]
    fig = _phase3_render(df, body_metrics, figsize=(7.0, 2.3))
    _save(fig, "phase3_tree_variants.pdf")

    full_metrics = [
        "top1_accuracy",
        "class_center_tree_spearman",
        "tree_distortion_average",
        "knn_siblings_recall_at_5",
    ]
    fig = _phase3_render(df, full_metrics, figsize=(7.0, 2.3))
    _save(fig, "phase3_tree_variants_full.pdf")


def make_pareto(df: pd.DataFrame) -> None:
    if "knn_siblings_recall_at_5" not in df.columns:
        print("[figures-spigm] pareto: no sibling recall column — skipping")
        return
    sub = df[df["tree_hierarchy"] == "default"].copy()
    sub = sub.dropna(subset=["top1_accuracy", "knn_siblings_recall_at_5",
                             "tree_loss_weight"])
    if sub.empty:
        print("[figures-spigm] pareto: no rows — skipping")
        return

    fig, ax = plt.subplots(figsize=(3.25, 2.8))
    # Marker size scaled by lambda (visualisation copy of original).
    base_size = 12.0
    sizes = base_size + 28.0 * sub["tree_loss_weight"].clip(upper=3.0)

    for geo, marker, color in [
        ("euclidean", "o", "tab:blue"),
        ("hyperbolic", "s", "tab:red"),
    ]:
        mask = sub["geometry"] == geo
        if mask.sum() == 0:
            continue
        ax.scatter(
            sub.loc[mask, "top1_accuracy"],
            sub.loc[mask, "knn_siblings_recall_at_5"],
            s=sizes[mask],
            marker=marker,
            label=geo,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.75,
            c=color,
        )

    ax.axvline(LOGREG_TOP1, linestyle=":", color="grey", linewidth=0.9,
               label=f"logreg top1={LOGREG_TOP1:.3f}")
    ax.scatter([KNN_TOP1], [KNN_SIBLING_RECALL], marker="*", s=90,
               color="black",
               label=f"kNN-5 ({KNN_TOP1:.3f}, {KNN_SIBLING_RECALL:.3f})")

    ax.set_xlabel("Top-1 accuracy (val)")
    ax.set_ylabel("Sibling recall@5 (default tree)")
    ax.set_title("Top-1 vs sibling recall (marker size $\\propto\\lambda$)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, framealpha=0.85)
    fig.tight_layout()
    _save(fig, "pareto.pdf")


def main() -> None:
    plt.rcParams.update(ICML_RC)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = _load()
    print(f"[figures-spigm] loaded {len(df)} rows from {SWEEP_CSV.relative_to(REPO_ROOT)}")

    make_phase1(df)
    make_phase2(df)
    make_phase3(df)
    make_pareto(df)


if __name__ == "__main__":
    main()
