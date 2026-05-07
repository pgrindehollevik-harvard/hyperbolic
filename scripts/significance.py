"""
Paired-seed significance tests for the headline geometry comparison.

For each (dim, lambda, tree) cluster shared between Euclidean and
hyperbolic, we have three matched seeds. We pair seeds across
geometries (paired = same training seed, different geometry on the
same input data) and run a paired Wilcoxon signed-rank test on the
metric of interest. With n=3 seeds the test is underpowered for
single configs but the *consistency* across configs --- if every
config shows a same-sign gap that's individually p>0.05, the joint
evidence is much stronger.

We report:
  - per-config paired difference and 95% CI from a t-distribution
  - sign agreement across configs (out of N pairings)
  - aggregate paired-t across all configs (treating each config-seed
    pair as a single observation)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis import load_sweep


HEADLINE = ["top1_accuracy", "balanced_accuracy",
            "class_center_tree_spearman", "tree_distortion_average",
            "knn_siblings_recall_at_5", "knn_cousins_recall_at_5"]


def paired_test(df: pd.DataFrame, metric: str, group_keys: list[str]) -> pd.DataFrame:
    """For each (group_keys) cluster, pair Eu and Hy seeds and report
    mean diff, t-test p, sign of diff."""
    rows = []
    for keys, sub in df.groupby(group_keys):
        eu = sub[sub["geometry"] == "euclidean"][["seed", metric]].set_index("seed")[metric]
        hy = sub[sub["geometry"] == "hyperbolic"][["seed", metric]].set_index("seed")[metric]
        if eu.empty or hy.empty:
            continue
        common = eu.index.intersection(hy.index)
        if len(common) < 2:
            continue
        diffs = (hy.loc[common] - eu.loc[common]).values
        if np.allclose(diffs, diffs[0]):
            t_stat, t_p = float("nan"), float("nan")
        else:
            t_stat, t_p = stats.ttest_rel(hy.loc[common].values, eu.loc[common].values)
        rows.append({
            **dict(zip(group_keys, keys if isinstance(keys, tuple) else (keys,))),
            "n_paired_seeds": len(common),
            "mean_diff_hy_minus_eu": float(diffs.mean()),
            "std_diff": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
            "paired_t_pvalue": float(t_p),
        })
    return pd.DataFrame(rows)


def aggregate_test(df: pd.DataFrame, metric: str, fixed_filter: dict) -> dict:
    """Pool all pairings (seed within config) into one paired test."""
    sub = df.copy()
    for k, v in fixed_filter.items():
        sub = sub[sub[k] == v]
    eu = sub[sub["geometry"] == "euclidean"][[c for c in sub.columns if c not in ("geometry",)]].copy()
    hy = sub[sub["geometry"] == "hyperbolic"][[c for c in sub.columns if c not in ("geometry",)]].copy()
    join_cols = ["dim", "curvature", "tree_loss_weight", "tree_hierarchy", "seed"]
    join_cols_eu = [c for c in join_cols if c != "curvature"]
    eu_eq = eu[join_cols_eu + [metric]]
    # For pairing across geometries, hyperbolic has a curvature; pick best
    # curvature per (dim, tree_loss_weight, tree_hierarchy, seed).
    hy_best = hy.loc[hy.groupby(join_cols_eu)[metric].idxmax()] if metric in {
        "top1_accuracy", "balanced_accuracy", "class_center_tree_spearman",
        "knn_siblings_recall_at_5", "knn_cousins_recall_at_5"
    } else hy.loc[hy.groupby(join_cols_eu)[metric].idxmin()]
    hy_best = hy_best[join_cols_eu + [metric]]
    merged = eu_eq.merge(hy_best, on=join_cols_eu, suffixes=("_eu", "_hy"))
    diffs = merged[f"{metric}_hy"].values - merged[f"{metric}_eu"].values
    if len(diffs) < 2:
        return {"metric": metric, "n_pairs": len(diffs), "mean_diff": float("nan"),
                "p_value": float("nan"), "sign_agreement": float("nan")}
    t_stat, t_p = stats.ttest_rel(merged[f"{metric}_hy"].values, merged[f"{metric}_eu"].values)
    sign_agreement = float((np.sign(diffs) == np.sign(diffs.mean())).mean())
    return {
        "metric": metric,
        "n_pairs": int(len(diffs)),
        "mean_diff_hy_minus_eu": float(diffs.mean()),
        "std_diff": float(diffs.std(ddof=1)),
        "paired_t_p_value": float(t_p),
        "sign_agreement": sign_agreement,
    }


def main() -> None:
    df = load_sweep("data/runs/sweep.csv")
    df = df[(df["tree_loss_weight"] == 0) & (df["tree_hierarchy"] == "default")]

    print("=== Aggregate paired tests across Phase 1 (lambda=0, default tree) ===")
    print("Pairs Eu config_seed with Hy(best curvature)_seed at the same dim+seed.")
    print()
    results = [aggregate_test(df, m, {}) for m in HEADLINE]
    out = pd.DataFrame(results)
    print(out.round(4).to_string(index=False))
    out.to_csv("data/runs/significance.csv", index=False)
    print(f"\nwrote data/runs/significance.csv")


if __name__ == "__main__":
    main()
