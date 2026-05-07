"""Generate the figures used in the report (report/figures/*.pdf).

Run after `scripts/sweep.py --phase 1` (and 2, 3) finishes:

    python scripts/make_figures.py

Each figure is regenerated from `data/runs/sweep.csv`. Existing PDFs are
overwritten. Figures that need data we don't yet have are silently skipped
with a warning so this is safe to run early.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from analysis import (
    DEFAULT_SWEEP_CSV,
    HEADLINE_METRICS,
    format_headline_for_report,
    headline_table,
    load_sweep,
    plot_lambda_sweep,
    plot_scaling,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
# Figures live in docs/figures/ alongside the report sections; report/ is a
# legacy single-file draft kept around for reference.
FIG_DIR = REPO_ROOT / "docs" / "figures"


def _save(fig: plt.Figure, name: str) -> None:
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches="tight")
    print(f"[figures] wrote {out.relative_to(REPO_ROOT)}")
    plt.close(fig)


def make_phase1(df) -> None:
    sub = df[(df["tree_loss_weight"] == 0.0) & (df["tree_hierarchy"] == "default")]
    if sub.empty:
        print("[figures] phase1: no rows with tree_loss_weight=0 — skipping")
        return
    fig = plot_scaling(sub)
    _save(fig, "phase1_scaling.pdf")


def make_phase2(df) -> None:
    sub = df[(df["tree_hierarchy"] == "default") & (df["dim"].isin([8, 16]))]
    sub = sub[sub["tree_loss_weight"].isin([0.0, 0.1, 0.3, 1.0, 3.0])]
    if sub.empty:
        print("[figures] phase2: no λ-sweep rows — skipping")
        return
    fig = plot_lambda_sweep(sub, dim=8, curvature=1.0)
    _save(fig, "phase2_lambda.pdf")


def make_phase3(df) -> None:
    sub = df[(df["tree_loss_weight"] == 1.0) & (df["dim"] == 8)]
    if sub.empty:
        print("[figures] phase3: no tree-variant rows — skipping")
        return
    metrics = ["top1_accuracy", "class_center_tree_spearman",
               "tree_distortion_average", "knn_siblings_recall_at_5"]
    metrics = [m for m in metrics if m in sub.columns]

    agg = sub.groupby(["tree_hierarchy", "geometry"])[metrics].agg(["mean", "std"]).reset_index()
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.2))
    if len(metrics) == 1:
        axes = [axes]
    trees = ["default", "chronological", "flat"]
    width = 0.35
    x = range(len(trees))
    for ax, metric in zip(axes, metrics):
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
                   capsize=3, label=geo)
        ax.set_xticks(list(x))
        ax.set_xticklabels(trees)
        ax.set_title(HEADLINE_METRICS.get(metric, metric))
        ax.set_xlabel("training tree")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend()
    fig.suptitle("Phase 3 — sensitivity of geometry winner to ground-truth tree")
    fig.tight_layout()
    _save(fig, "phase3_tree_variants.pdf")


def make_headline_table(df) -> None:
    table = headline_table(df)
    if table.empty:
        print("[figures] headline: no rows — skipping")
        return
    formatted = format_headline_for_report(table)

    # Add baseline columns if available (logistic-on-CLIP, kNN-on-CLIP).
    baselines_path = REPO_ROOT / "data" / "runs" / "baselines.csv"
    if baselines_path.exists():
        import pandas as pd
        bl = pd.read_csv(baselines_path).set_index("method")
        # Map column names
        col_map = {
            "Top-1": "top1_accuracy",
            "Balanced acc.": "balanced_accuracy",
            "Sibling recall@5": "sibling_recall_at_5_default",
            "Cousin recall@5": "cousin_recall_at_5_default",
        }
        for label, source_key in col_map.items():
            if label in formatted.index:
                for method, col_name in [
                    ("logistic-regression-on-CLIP", "logistic-on-CLIP"),
                    ("kNN-5-on-CLIP", "kNN-5-on-CLIP"),
                ]:
                    if method in bl.index and source_key in bl.columns:
                        val = bl.loc[method, source_key]
                        if pd.notna(val):
                            formatted.at[label, col_name] = f"{float(val):.3f}"
                        else:
                            formatted.at[label, col_name] = "—"
        # Fill missing baseline cells with em-dash
        for col in ["logistic-on-CLIP", "kNN-5-on-CLIP"]:
            if col in formatted.columns:
                formatted[col] = formatted[col].fillna("—")

    csv_out = FIG_DIR / "headline_results.csv"
    formatted.to_csv(csv_out)
    print(f"[figures] wrote {csv_out.relative_to(REPO_ROOT)}")

    n_cols = len(formatted.columns)
    latex = formatted.to_latex(escape=True, column_format="l" + "c" * n_cols)
    tex_out = FIG_DIR / "headline_results.tex"
    tex_out.write_text(latex)
    print(f"[figures] wrote {tex_out.relative_to(REPO_ROOT)}")


def main() -> None:
    if not DEFAULT_SWEEP_CSV.exists():
        print(f"[figures] sweep CSV not found at {DEFAULT_SWEEP_CSV}; nothing to do.")
        return
    df = load_sweep(DEFAULT_SWEEP_CSV)
    print(f"[figures] loaded {len(df)} rows")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    make_phase1(df)
    make_phase2(df)
    make_phase3(df)
    make_headline_table(df)


if __name__ == "__main__":
    main()
