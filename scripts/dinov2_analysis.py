"""
Cross-encoder check for Phase 4: build the empirical reference tree
from DINOv2 features rather than CLIP features, and re-evaluate the
Phase 1 winners against it.

If the geometry comparison on the DINOv2-derived empirical tree
matches the CLIP-derived empirical tree's, the Phase 4 finding
is not encoder-specific. If it diverges, the Phase 4 framing has to
be qualified.

Run once `data/features/dinov2_vitb14.npy` exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis import _embed_split, load_sweep
from empirical_recall import (
    empirical_relation_sets,
    linkage_relation_sets,
    recall_at_k,
)
from empirical_tree import (
    build_empirical_tree,
    build_empirical_tree_from_features,
    linkage_to_tree_distance_matrix,
)
from eval import collect_distance_statistics, distance_matrix_from_points, pairwise_values, spearman_corr, load_model
from hierarchy import distance_matrix as hierarchy_distance_matrix
from hierarchy import load_style_classes
from scipy.cluster.hierarchy import linkage


REPO_ROOT = Path(__file__).resolve().parents[1]
DINOV2_PATH = REPO_ROOT / "data" / "features" / "dinov2_vitb14.npy"


def _build_dinov2_tree_and_linkage() -> tuple[np.ndarray, np.ndarray]:
    """Return (T_dinov2_distance_matrix, dinov2_linkage_matrix)."""
    if not DINOV2_PATH.exists():
        raise FileNotFoundError(
            f"DINOv2 features not found at {DINOV2_PATH}. "
            f"Run scripts/extract_dinov2_features.py first."
        )
    # Compute centroids ourselves so we can also retain the linkage
    import re
    import pandas as pd

    INDEX_PATH = REPO_ROOT / "data" / "features" / "index.csv"
    SPLIT_DIR = REPO_ROOT / "data" / "wikiart_csvs"

    def _ascii_key(p: str) -> str:
        return re.sub(r"[^A-Za-z0-9/_.-]", "", p)

    features = np.load(DINOV2_PATH).astype(np.float32)
    index = pd.read_csv(INDEX_PATH)
    index["key"] = index["path"].map(_ascii_key)
    key_to_row = dict(zip(index["key"], index["row_idx"]))
    split_df = pd.read_csv(
        SPLIT_DIR / "style_train.csv", header=None, names=["path", "label"]
    )
    split_df["key"] = split_df["path"].map(_ascii_key)
    split_df["row_idx"] = split_df["key"].map(key_to_row)
    split_df = split_df.dropna(subset=["row_idx"]).reset_index(drop=True)
    rows = split_df["row_idx"].to_numpy(dtype=np.int64)
    labels = split_df["label"].to_numpy(dtype=np.int64)
    feats = features[rows]
    centers = np.stack([feats[labels == k].mean(axis=0) for k in range(27)])
    Z = linkage(centers, method="average", metric="euclidean")
    T = linkage_to_tree_distance_matrix(Z, n=27)
    return T, Z


def main() -> None:
    style_names = load_style_classes()
    T_default = hierarchy_distance_matrix(style_names, hierarchy_name="default")
    T_clip = build_empirical_tree(split="train", method="average")
    T_dinov2, Z_dinov2 = _build_dinov2_tree_and_linkage()
    Z_clip = None  # we don't need CLIP linkage here
    dinov2_relations = linkage_relation_sets(Z_dinov2, n=27)

    # Tree-vs-tree pairwise distance Spearman
    v_default = pairwise_values(T_default)
    v_clip = pairwise_values(T_clip)
    v_dinov2 = pairwise_values(T_dinov2)
    print("=== Tree-vs-tree Spearman on pairwise distances ===")
    print(f"  default vs CLIP-empirical    : {spearman_corr(v_default, v_clip):.3f}")
    print(f"  default vs DINOv2-empirical  : {spearman_corr(v_default, v_dinov2):.3f}")
    print(f"  CLIP-empirical vs DINOv2-empirical : {spearman_corr(v_clip, v_dinov2):.3f}")
    print()
    print(f"mean off-diag: default={T_default[T_default>0].mean():.2f}, "
          f"CLIP-emp={T_clip[T_clip>0].mean():.2f}, DINOv2-emp={T_dinov2[T_dinov2>0].mean():.2f}")
    print()

    # Re-evaluate Phase 1 winners against DINOv2 tree
    df = load_sweep("data/runs/sweep.csv")
    df = df[(df["tree_loss_weight"] == 0) & (df["tree_hierarchy"] == "default")]

    rows = []
    for geo in ["euclidean", "hyperbolic"]:
        sub = df[df["geometry"] == geo]
        for d in sorted(sub["dim"].unique()):
            sub_d = sub[sub["dim"] == d]
            best = sub_d.loc[sub_d["top1_accuracy"].idxmax()]
            ckpt = Path("data/runs/sweep") / best["config_hash"] / "ckpt.pt"
            if not ckpt.exists():
                continue
            embeddings, labels, _, meta = _embed_split(ckpt, split="val", device=None)
            _, head, clf, _ = load_model(
                ckpt_path=ckpt, geometry_override=None, curvature_override=None,
                device=torch.device("cpu"),
            )
            proto = clf.prototypes.detach().cpu()
            proto_dists = distance_matrix_from_points(proto, geometry=meta["geometry"], curvature=meta["curvature"])
            v_proto = pairwise_values(proto_dists)
            sp_default = spearman_corr(v_proto, v_default)
            sp_clip = spearman_corr(v_proto, v_clip)
            sp_dinov2 = spearman_corr(v_proto, v_dinov2)

            # Recall@5 against DINOv2-tree sibling/cousin sets
            _, neighbor_indices, _ = collect_distance_statistics(
                embeddings=embeddings, labels=labels,
                geometry=meta["geometry"], curvature=meta["curvature"],
                ks=[5], block_size=512,
            )
            neighbor_labels = labels[neighbor_indices]
            sib_recall, _ = recall_at_k(neighbor_labels, labels, dinov2_relations["siblings"], 5)
            cou_recall, _ = recall_at_k(neighbor_labels, labels, dinov2_relations["cousins"], 5)

            rows.append({
                "geometry": geo, "dim": int(d),
                "spearman_vs_default": sp_default,
                "spearman_vs_clip_empirical": sp_clip,
                "spearman_vs_dinov2_empirical": sp_dinov2,
                "sibling_recall_at_5_dinov2": sib_recall,
                "cousin_recall_at_5_dinov2": cou_recall,
            })
            print(rows[-1])

    out_df = pd.DataFrame(rows)
    out_df.to_csv("data/runs/dinov2_tree_eval.csv", index=False)
    print(f"\nwrote data/runs/dinov2_tree_eval.csv")


if __name__ == "__main__":
    main()
