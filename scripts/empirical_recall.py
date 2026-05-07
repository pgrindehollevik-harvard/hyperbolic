"""
Recompute sibling and cousin recall@k against the empirical tree.

The recall metrics in scripts/eval.py derive sibling and cousin sets
from STYLE_HIERARCHY (the hand-built default tree). This script
reconstructs those sets from the empirical tree's linkage matrix and
recomputes recall@5 on the same Phase 1 winner checkpoints, so the
"local advantage of hyperbolic is reference-tree-independent" claim
becomes a measured number rather than an assertion.

Sibling and cousin definitions on a binary linkage tree
-------------------------------------------------------
Each leaf $l$ has a unique merge sequence to the root.

- **Siblings of $l$**: leaves of the *other* branch at $l$'s lowest
  merge (the merge whose two children include $l$ as a leaf and a
  subtree $S$). Siblings = all leaves of $S$. For binary clustering on
  $n=27$ leaves, every leaf has at least one sibling (the other leaf
  or sub-tree merged with it first).
- **Cousins of $l$**: leaves of the *uncle* subtree at $l$'s
  grandparent merge --- i.e., the subtree merged with $l$'s
  immediate-parent subtree at the next-higher merge.

These match the spirit of the default tree's relations
(scripts/eval.py:style_relation_sets) which uses children-of-same-parent
and descendants-of-uncle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import linkage
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis import _embed_split, load_sweep
from dataset import FeatureDataset, NUM_CLASSES
from empirical_tree import class_centroids
from eval import collect_distance_statistics, load_model, pick_device


def _children_index_map(Z: np.ndarray, n: int) -> tuple[dict[int, list[int]], dict[int, int]]:
    """Return (children_of, parent_of) where keys are scipy node ids
    (leaves 0..n-1, internal nodes n..2n-2)."""
    children_of: dict[int, list[int]] = {}
    parent_of: dict[int, int] = {}
    for k, row in enumerate(Z):
        i, j = int(row[0]), int(row[1])
        new_node = n + k
        children_of[new_node] = [i, j]
        parent_of[i] = new_node
        parent_of[j] = new_node
    return children_of, parent_of


def _leaves_under(node: int, n: int, children_of: dict[int, list[int]]) -> set[int]:
    if node < n:
        return {node}
    a, b = children_of[node]
    return _leaves_under(a, n, children_of) | _leaves_under(b, n, children_of)


def linkage_relation_sets(Z: np.ndarray, n: int) -> dict[str, dict[int, set[int]]]:
    """Build sibling and cousin sets for each leaf of a SciPy linkage tree."""
    children_of, parent_of = _children_index_map(Z, n)

    siblings: dict[int, set[int]] = {}
    cousins: dict[int, set[int]] = {}

    for leaf in range(n):
        # Sibling subtree: the other child of leaf's immediate parent.
        p = parent_of.get(leaf)
        if p is None:
            siblings[leaf] = set()
            cousins[leaf] = set()
            continue
        a, b = children_of[p]
        sib_subtree = b if a == leaf else a
        siblings[leaf] = _leaves_under(sib_subtree, n, children_of) - {leaf}

        # Cousin subtree: the other child of grandparent.
        g = parent_of.get(p)
        if g is None:
            cousins[leaf] = set()
            continue
        ga, gb = children_of[g]
        uncle_subtree = gb if ga == p else ga
        cousins[leaf] = _leaves_under(uncle_subtree, n, children_of) - {leaf}

    return {"siblings": siblings, "cousins": cousins}


def empirical_relation_sets(method: str = "average") -> dict[str, dict[int, set[int]]]:
    """Build empirical sibling/cousin sets from CLIP train centroids."""
    train = FeatureDataset("train")
    centers = class_centroids(train.features.numpy(), train.labels.numpy(), n_classes=NUM_CLASSES)
    Z = linkage(centers, method=method, metric="euclidean")
    return linkage_relation_sets(Z, n=NUM_CLASSES)


def recall_at_k(neighbor_labels: np.ndarray, query_labels: np.ndarray,
                relation_sets: dict[int, set[int]], k: int) -> tuple[float, int]:
    """Mean recall@k: fraction of queries (with non-empty relation set)
    whose top-k neighbors contain at least one leaf from the relation
    set of the query's own class. Returns (mean, valid_query_count)."""
    valid_mask = np.array([bool(relation_sets[int(label)]) for label in query_labels], dtype=bool)
    if valid_mask.sum() == 0:
        return float("nan"), 0
    hits = []
    for idx in np.where(valid_mask)[0]:
        targets = relation_sets[int(query_labels[idx])]
        hits.append(np.isin(neighbor_labels[idx, :k], list(targets)).any())
    return float(np.mean(hits)), int(valid_mask.sum())


def evaluate_winner_against_empirical(ckpt_path: Path, k: int = 5,
                                       device: str | None = None) -> dict:
    """Compute sibling/cousin recall@k against the empirical tree's relations."""
    embeddings, labels, _, meta = _embed_split(ckpt_path, split="val", device=device)
    rel = empirical_relation_sets()

    # Reuse Phase 1's collect_distance_statistics for the kNN indices.
    _, neighbor_indices, _ = collect_distance_statistics(
        embeddings=embeddings,
        labels=labels,
        geometry=meta["geometry"],
        curvature=meta["curvature"],
        ks=[k],
        block_size=512,
    )
    neighbor_labels = labels[neighbor_indices]
    sib_recall, sib_n = recall_at_k(neighbor_labels, labels, rel["siblings"], k)
    cou_recall, cou_n = recall_at_k(neighbor_labels, labels, rel["cousins"], k)
    return {
        "geometry": meta["geometry"],
        "dim": meta["dim"],
        "curvature": meta["curvature"],
        f"sibling_recall_at_{k}_empirical": sib_recall,
        f"sibling_valid_queries_empirical": sib_n,
        f"cousin_recall_at_{k}_empirical": cou_recall,
        f"cousin_valid_queries_empirical": cou_n,
    }


def main() -> None:
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
            row = evaluate_winner_against_empirical(ckpt)
            row["default_sibling_recall_at_5"] = float(best["knn_siblings_recall_at_5"])
            row["default_cousin_recall_at_5"] = float(best["knn_cousins_recall_at_5"])
            rows.append(row)
            print(row)

    out = Path("data/runs/empirical_recall.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
