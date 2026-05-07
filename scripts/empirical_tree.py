"""
Build a data-driven reference tree from training-set CLIP feature centroids.

The hand-built tree in `scripts/hierarchy.py` is one art-historian's
view of WikiArt's 27 styles. To check whether our global-metric
results depend on that specific taxonomy, this module constructs a
purely data-driven alternative:

  1. Compute the per-class centroid of training-set CLIP ViT-B/16
     features (one 512-d vector per style).
  2. Run agglomerative clustering on those centroids in Euclidean
     space (average linkage on Euclidean distances). The result is a
     binary dendrogram.
  3. Convert the dendrogram to an integer tree-distance matrix by
     counting the number of edges on the path between each pair of
     leaves through their lowest common ancestor.

The output has the same shape and integer type as
`hierarchy.distance_matrix`, so it can be substituted into any
evaluation that consumes a tree-distance matrix.

Usage
-----
>>> import numpy as np
>>> from empirical_tree import build_empirical_tree
>>> T = build_empirical_tree()
>>> T.shape
(27, 27)
>>> T.diagonal().sum()
0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import FeatureDataset, NUM_CLASSES


def class_centroids(
    features: np.ndarray, labels: np.ndarray, n_classes: int = NUM_CLASSES
) -> np.ndarray:
    """Mean feature vector per class."""
    return np.stack([features[labels == k].mean(axis=0) for k in range(n_classes)])


def linkage_to_tree_distance_matrix(linkage_matrix: np.ndarray, n: int) -> np.ndarray:
    """Edge-count distances between every pair of leaves in the binary
    tree implied by a SciPy linkage matrix."""
    parent: dict[int, int] = {}
    for k, row in enumerate(linkage_matrix):
        i, j = int(row[0]), int(row[1])
        new_node = n + k
        parent[i] = new_node
        parent[j] = new_node

    def path_to_root(x: int) -> list[int]:
        path = [x]
        while x in parent:
            x = parent[x]
            path.append(x)
        return path

    T = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        pi = path_to_root(i)
        si = {x: idx for idx, x in enumerate(pi)}
        for j in range(i + 1, n):
            pj = path_to_root(j)
            for idx_j, anc in enumerate(pj):
                if anc in si:
                    T[i, j] = T[j, i] = si[anc] + idx_j
                    break
    return T


def build_empirical_tree(
    split: str = "train",
    method: str = "average",
) -> np.ndarray:
    """Construct an empirical tree-distance matrix from cached CLIP features.

    Parameters
    ----------
    split : str
        Which split to read centroids from. Default "train" (so the
        empirical tree is independent of validation labels).
    method : str
        Linkage method passed to `scipy.cluster.hierarchy.linkage`.
        "average" (UPGMA) is the most common default.
    """
    dataset = FeatureDataset(split)
    features = dataset.features.numpy()
    labels = dataset.labels.numpy()
    centers = class_centroids(features, labels, n_classes=NUM_CLASSES)
    Z = linkage(centers, method=method, metric="euclidean")
    return linkage_to_tree_distance_matrix(Z, n=NUM_CLASSES)


def build_empirical_tree_from_features(
    features_path: Path | str,
    method: str = "average",
) -> np.ndarray:
    """Build an empirical tree from a custom feature file.

    Used to construct a tree from a non-CLIP encoder (e.g.\\ DINOv2).
    Reuses the train-split labels from `data/features/index.csv` and
    the train-split CSV in `data/wikiart_csvs/style_train.csv` to map
    rows to class labels, then proceeds as `build_empirical_tree`.
    """
    import pandas as pd
    import re

    REPO_ROOT = Path(__file__).resolve().parents[1]
    INDEX_PATH = REPO_ROOT / "data" / "features" / "index.csv"
    SPLIT_DIR = REPO_ROOT / "data" / "wikiart_csvs"

    def _ascii_key(p: str) -> str:
        return re.sub(r"[^A-Za-z0-9/_.-]", "", p)

    features = np.load(features_path).astype(np.float32)
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

    centers = class_centroids(feats, labels, n_classes=NUM_CLASSES)
    Z = linkage(centers, method=method, metric="euclidean")
    return linkage_to_tree_distance_matrix(Z, n=NUM_CLASSES)


if __name__ == "__main__":
    T = build_empirical_tree()
    print("shape:", T.shape)
    print("max distance:", int(T.max()))
    print("mean off-diagonal:", float(T[T > 0].mean()))
    print("first row:", T[0])
