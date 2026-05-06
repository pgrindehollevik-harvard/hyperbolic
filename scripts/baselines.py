"""
Calibration baselines: how good are the prototype classifiers, in
absolute terms?

Trains two reference classifiers directly on frozen CLIP ViT-B/16
features and evaluates them on the val split:

  1. Linear (logistic regression) on the 512-d CLIP features.
  2. k-NN on the raw 512-d CLIP features (k=5, Euclidean).

These bracket the performance a non-prototype, geometry-agnostic
classifier achieves with the same input information our models see.
The number reported is top-1 accuracy plus sibling and cousin
recall@5 against both the default and empirical trees, so the same
metrics used in the main paper can be read on a baseline.

Run from repo root:
    python scripts/baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import FeatureDataset, NUM_CLASSES
from empirical_recall import (
    empirical_relation_sets,
    linkage_relation_sets,
    recall_at_k,
)
from eval import style_relation_sets
from hierarchy import load_style_classes
from sklearn.metrics import balanced_accuracy_score
from scipy.spatial.distance import cdist


def _knn_neighbor_labels(query: np.ndarray, train: np.ndarray, train_labels: np.ndarray, k: int) -> np.ndarray:
    """For each query row, return top-k train-label nearest neighbors by L2."""
    # Block to avoid quadratic memory on the full 24k x 57k.
    out = np.empty((query.shape[0], k), dtype=train_labels.dtype)
    block = 512
    for start in range(0, query.shape[0], block):
        end = min(start + block, query.shape[0])
        d = cdist(query[start:end], train)
        idx = np.argpartition(d, k, axis=1)[:, :k]
        # sort the partitioned set by actual distance
        rows = np.arange(end - start)[:, None]
        idx = idx[rows, np.argsort(d[rows, idx], axis=1)]
        out[start:end] = train_labels[idx]
    return out


def main() -> None:
    train = FeatureDataset("train")
    val = FeatureDataset("val")
    Xtr, ytr = train.features.numpy(), train.labels.numpy()
    Xvl, yvl = val.features.numpy(), val.labels.numpy()
    style_names = load_style_classes()

    rows = []

    # 1) Logistic regression on raw CLIP features.
    clf = LogisticRegression(max_iter=200, n_jobs=-1, C=1.0)
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xvl)
    rows.append({
        "method": "logistic-regression-on-CLIP",
        "top1_accuracy": accuracy_score(yvl, yhat),
        "balanced_accuracy": balanced_accuracy_score(yvl, yhat),
    })

    # 2) k-NN on raw CLIP features (k=5).
    # Use sklearn for top-1 accuracy
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(Xtr, ytr)
    yhat_knn = knn.predict(Xvl)

    # For sibling/cousin recall, we need val→val nearest neighbors (not val→train),
    # because the metric is "do my val nearest neighbors share my class's
    # sibling/cousin set". So compute val-to-val k=5 on raw CLIP, then use it.
    val_neighbors = _knn_neighbor_labels(Xvl, Xvl, yvl, k=6)  # k+1 because self appears
    # Drop the self-match (same row, same label at distance 0). Easiest: use
    # everything but the first column, since the self is always nearest.
    val_neighbors = val_neighbors[:, 1:]

    default_rel = style_relation_sets(style_names)
    empirical_rel = empirical_relation_sets()

    knn_row = {
        "method": "kNN-5-on-CLIP",
        "top1_accuracy": accuracy_score(yvl, yhat_knn),
        "balanced_accuracy": balanced_accuracy_score(yvl, yhat_knn),
    }
    for tree_name, rel in [("default", default_rel), ("empirical", empirical_rel)]:
        sib, _ = recall_at_k(val_neighbors, yvl, rel["siblings"], k=5)
        cou, _ = recall_at_k(val_neighbors, yvl, rel["cousins"], k=5)
        knn_row[f"sibling_recall_at_5_{tree_name}"] = sib
        knn_row[f"cousin_recall_at_5_{tree_name}"] = cou
    rows.append(knn_row)

    df = pd.DataFrame(rows)
    out = Path("data/runs/baselines.csv")
    df.to_csv(out, index=False)
    print(df.round(3).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
