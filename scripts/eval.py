#!/usr/bin/env python3
"""Evaluate Euclidean or hyperbolic style models on cached CLIP features."""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from dataset import FeatureDataset, INDEX_PATH, NUM_CLASSES, SPLIT_DIR, _ascii_key
from hierarchy import STYLE_HIERARCHY, distance_matrix, load_style_classes
from models import (
    ClassifierLayer,
    EuclideanHead,
    PoincareHead,
    poincare_distance,
    poincare_expmap0,
    poincare_logmap0,
    project_to_poincare_ball,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to ckpt.pt")
    parser.add_argument("--geometry", choices=["euclidean", "hyperbolic"])
    parser.add_argument("--curvature", type=float, default=None)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <checkpoint_dir>/eval_<split>",
    )
    parser.add_argument(
        "--distance-block-size",
        type=int,
        default=512,
        help="Block size for exact distance-based metrics such as tree distortion and kNN retrieval.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        nargs="+",
        default=[5, 10],
        help="k values for subtree retrieval consistency.",
    )
    parser.add_argument(
        "--frechet-style",
        type=str,
        default="Cubism",
        help="Style used for interpolation stability experiments.",
    )
    parser.add_argument(
        "--frechet-sample-size",
        type=int,
        default=8,
        help="Number of same-style embeddings averaged per Fréchet-mean trial.",
    )
    parser.add_argument(
        "--frechet-trials",
        type=int,
        default=100,
        help="Number of interpolation trials to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2090,
        help="Random seed for sampling-based metrics.",
    )
    return parser.parse_args()


def pick_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split_metadata(split: str) -> pd.DataFrame:
    index = pd.read_csv(INDEX_PATH)
    index["key"] = index["path"].map(_ascii_key)
    key_to_row = dict(zip(index["key"], index["row_idx"]))

    split_df = pd.read_csv(
        SPLIT_DIR / f"style_{split}.csv",
        header=None,
        names=["path", "label"],
    )
    split_df["key"] = split_df["path"].map(_ascii_key)
    split_df["row_idx"] = split_df["key"].map(key_to_row)
    split_df["dropped_missing_feature"] = split_df["row_idx"].isna()
    split_df = split_df.dropna(subset=["row_idx"]).reset_index(drop=True)
    split_df["row_idx"] = split_df["row_idx"].astype(np.int64)
    return split_df


def infer_geometry(ckpt_path: Path, config: dict, override: str | None) -> str:
    if override is not None:
        return override
    geometry = config.get("geometry")
    if geometry in {"euclidean", "hyperbolic"}:
        return geometry

    lowered = str(ckpt_path).lower()
    if "hyperbolic" in lowered:
        return "hyperbolic"
    if "euclidean" in lowered:
        return "euclidean"

    raise ValueError(
        "Could not infer geometry from checkpoint. Pass --geometry explicitly."
    )


def load_model(
    ckpt_path: Path,
    geometry_override: str | None,
    curvature_override: float | None,
    device: torch.device,
) -> tuple[dict, torch.nn.Module, torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    geometry = infer_geometry(ckpt_path, config, geometry_override)
    dim = int(config.get("dim", ckpt["clf"]["prototypes"].shape[1]))
    dropout = float(config.get("dropout", 0.1))
    curvature = config.get("curvature", 1.0)
    curvature = 1.0 if curvature is None else float(curvature)
    if curvature_override is not None:
        curvature = curvature_override

    if geometry == "euclidean":
        head = EuclideanHead(d_out=dim, dropout=dropout)
        clf = ClassifierLayer(
            num_classes=NUM_CLASSES,
            dim=dim,
            geometry="euclidean",
        )
    else:
        head = PoincareHead(d_out=dim, dropout=dropout, curvature=curvature)
        clf = ClassifierLayer(
            num_classes=NUM_CLASSES,
            dim=dim,
            geometry="hyperbolic",
            curvature=curvature,
        )

    head.load_state_dict(ckpt["head"])
    clf.load_state_dict(ckpt["clf"])
    head.to(device).eval()
    clf.to(device).eval()

    metadata = {
        "geometry": geometry,
        "dim": dim,
        "dropout": dropout,
        "curvature": curvature,
        "epoch": ckpt.get("epoch"),
    }
    return ckpt, head, clf, metadata


def pairwise_values(matrix: np.ndarray) -> np.ndarray:
    rows, cols = np.triu_indices_from(matrix, k=1)
    return matrix[rows, cols]


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(x.std(), 0.0) or np.allclose(y.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    return pearson_corr(x_rank, y_rank)


def distance_matrix_from_points(
    points: torch.Tensor,
    geometry: str,
    curvature: float,
) -> np.ndarray:
    if geometry == "euclidean":
        dists = torch.cdist(points, points)
    else:
        dists = poincare_distance(
            points.unsqueeze(1),
            points.unsqueeze(0),
            curvature=curvature,
        )
    return dists.detach().cpu().numpy()


def poincare_mobius_add(
    x: torch.Tensor,
    y: torch.Tensor,
    curvature: float,
) -> torch.Tensor:
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    c = curvature
    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    return project_to_poincare_ball(
        numerator / denominator.clamp_min(1e-12),
        curvature=curvature,
    )


def poincare_lambda_x(x: torch.Tensor, curvature: float) -> torch.Tensor:
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    return 2.0 / (1.0 - curvature * x2).clamp_min(1e-12)


def poincare_expmap(
    x: torch.Tensor,
    v: torch.Tensor,
    curvature: float,
) -> torch.Tensor:
    sqrt_c = torch.as_tensor(curvature, device=x.device, dtype=x.dtype).sqrt()
    lam = poincare_lambda_x(x, curvature)
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    second = torch.tanh(sqrt_c * lam * v_norm / 2.0) * v / (sqrt_c * v_norm)
    return poincare_mobius_add(x, second, curvature)


def poincare_logmap(
    x: torch.Tensor,
    y: torch.Tensor,
    curvature: float,
) -> torch.Tensor:
    sqrt_c = torch.as_tensor(curvature, device=x.device, dtype=x.dtype).sqrt()
    sub = poincare_mobius_add(-x, y, curvature)
    sub_norm = sub.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    lam = poincare_lambda_x(x, curvature)
    scale = (2.0 / (sqrt_c * lam)) * torch.atanh(
        (sqrt_c * sub_norm).clamp_max(1.0 - 1e-5)
    ) / sub_norm
    return scale * sub


def frechet_mean(
    points: torch.Tensor,
    geometry: str,
    curvature: float,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> torch.Tensor:
    if geometry == "euclidean":
        return points.mean(dim=0)

    mean = poincare_expmap0(
        poincare_logmap0(points, curvature=curvature).mean(dim=0, keepdim=True),
        curvature=curvature,
    )[0]

    for _ in range(max_iter):
        mean_batch = mean.unsqueeze(0).expand(points.shape[0], -1)
        tangent_mean = poincare_logmap(mean_batch, points, curvature).mean(dim=0)
        if tangent_mean.norm().item() < tol:
            break
        updated = poincare_expmap(
            mean.unsqueeze(0),
            tangent_mean.unsqueeze(0),
            curvature,
        )[0]
        if torch.dist(updated, mean).item() < tol:
            mean = updated
            break
        mean = updated

    return project_to_poincare_ball(mean, curvature=curvature)


def class_centers_from_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    geometry: str,
    curvature: float,
) -> np.ndarray:
    centers: list[np.ndarray] = []
    for label in range(NUM_CLASSES):
        class_points = torch.from_numpy(embeddings[labels == label])
        if class_points.numel() == 0:
            raise ValueError(f"split is missing class {label}, cannot compute class centers")
        center = frechet_mean(class_points, geometry=geometry, curvature=curvature)
        centers.append(center.numpy())
    return np.stack(centers)


def tree_distance_summary(
    tree_dists: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    pred_tree_dist = tree_dists[y_true, y_pred].astype(np.float64)
    mistakes = pred_tree_dist[y_true != y_pred]
    summary = {
        "mean_tree_distance_all_predictions": float(pred_tree_dist.mean()),
        "median_tree_distance_all_predictions": float(np.median(pred_tree_dist)),
    }
    if mistakes.size:
        summary.update(
            {
                "mean_tree_distance_mistakes": float(mistakes.mean()),
                "median_tree_distance_mistakes": float(np.median(mistakes)),
                "mistakes_within_tree_distance_2": float((mistakes <= 2).mean()),
            }
        )
    else:
        summary.update(
            {
                "mean_tree_distance_mistakes": 0.0,
                "median_tree_distance_mistakes": 0.0,
                "mistakes_within_tree_distance_2": 1.0,
            }
        )
    return summary


def save_confusion_plot(
    matrix: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_dendrogram_plot(
    children: np.ndarray,
    distances: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> None:
    try:
        from scipy.cluster.hierarchy import dendrogram
    except Exception:
        return

    n = len(labels)
    counts = np.zeros(children.shape[0], dtype=np.int64)
    for i, (left, right) in enumerate(children):
        left_count = 1 if left < n else counts[left - n]
        right_count = 1 if right < n else counts[right - n]
        counts[i] = left_count + right_count

    linkage = np.column_stack([children, distances, counts]).astype(float)
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(linkage, labels=labels, leaf_rotation=90, ax=ax)
    ax.set_title("Agglomerative Dendrogram on Style Centroids")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate(
    head: torch.nn.Module,
    clf: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct_top1 = 0
    correct_top5 = 0
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_topk: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            embeddings = head(xb)
            logits = clf(embeddings)
            loss = criterion(logits, yb)

            batch_size = yb.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

            preds = logits.argmax(dim=-1)
            topk = logits.topk(k=min(5, NUM_CLASSES), dim=-1).indices
            correct_top1 += (preds == yb).sum().item()
            correct_top5 += (topk == yb.unsqueeze(1)).any(dim=1).sum().item()

            all_labels.append(yb.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_topk.append(topk.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))

    metrics = {
        "loss": total_loss / total,
        "top1_accuracy": correct_top1 / total,
        "top5_accuracy": correct_top5 / total,
    }
    return (
        metrics,
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_topk),
        np.concatenate(all_embeddings),
    )


def pairwise_distance_block(
    query: torch.Tensor,
    reference: torch.Tensor,
    geometry: str,
    curvature: float,
) -> torch.Tensor:
    if geometry == "euclidean":
        return torch.cdist(query, reference)
    return poincare_distance(
        query.unsqueeze(1),
        reference.unsqueeze(0),
        curvature=curvature,
    )


def collect_distance_statistics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    geometry: str,
    curvature: float,
    ks: list[int],
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings_t = torch.from_numpy(embeddings)
    labels_np = np.asarray(labels)
    n = embeddings_t.shape[0]
    max_k = max(ks)
    style_indices = [np.where(labels_np == label)[0] for label in range(NUM_CLASSES)]

    neighbor_indices = np.empty((n, max_k), dtype=np.int64)
    neighbor_dists = np.empty((n, max_k), dtype=np.float32)

    pair_sums = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    pair_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        query = embeddings_t[start:end]
        block_dists = pairwise_distance_block(
            query,
            embeddings_t,
            geometry=geometry,
            curvature=curvature,
        )

        block_labels = labels_np[start:end]
        block_row_indices = {
            label: np.where(block_labels == label)[0]
            for label in np.unique(block_labels)
        }

        for label_j, column_indices in enumerate(style_indices):
            if column_indices.size == 0:
                continue
            dists_to_style = block_dists[:, column_indices]
            col_count = int(column_indices.size)
            for label_i, row_indices in block_row_indices.items():
                if label_i == label_j:
                    continue
                pair_sums[label_i, label_j] += dists_to_style[row_indices].sum().item()
                pair_counts[label_i, label_j] += int(row_indices.size) * col_count

        block_dists[torch.arange(end - start), torch.arange(start, end)] = float("inf")
        top_dists, top_indices = torch.topk(
            block_dists,
            k=max_k,
            dim=1,
            largest=False,
        )
        neighbor_indices[start:end] = top_indices.numpy()
        neighbor_dists[start:end] = top_dists.numpy()

    ordered_means = np.divide(
        pair_sums,
        pair_counts,
        out=np.full_like(pair_sums, np.nan, dtype=np.float64),
        where=pair_counts > 0,
    )

    sym_means = np.zeros_like(ordered_means)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i == j:
                continue
            sym_means[i, j] = np.nanmean([ordered_means[i, j], ordered_means[j, i]])

    return sym_means, neighbor_indices, neighbor_dists


def style_pair_distance_table(
    avg_style_dists: np.ndarray,
    tree_dists: np.ndarray,
    style_names: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    ratios = []
    for i in range(NUM_CLASSES):
        for j in range(i + 1, NUM_CLASSES):
            emb_dist = float(avg_style_dists[i, j])
            tree_dist = float(tree_dists[i, j])
            ratio = emb_dist / tree_dist
            ratios.append(ratio)
            rows.append(
                {
                    "style_a": style_names[i],
                    "style_b": style_names[j],
                    "tree_distance": tree_dist,
                    "avg_embedding_distance": emb_dist,
                    "raw_ratio": ratio,
                }
            )

    scale = float(np.exp(np.mean(np.log(np.asarray(ratios)))))
    for row in rows:
        scaled_tree = scale * row["tree_distance"]
        distortion = max(
            row["avg_embedding_distance"] / scaled_tree,
            scaled_tree / row["avg_embedding_distance"],
        )
        row["scaled_tree_distance"] = scaled_tree
        row["multiplicative_distortion"] = distortion

    pair_df = pd.DataFrame(rows).sort_values(
        "multiplicative_distortion",
        ascending=False,
    )
    metrics = {
        "tree_distortion_scale": scale,
        "tree_distortion_average": float(pair_df["multiplicative_distortion"].mean()),
        "tree_distortion_worst_case": float(pair_df["multiplicative_distortion"].max()),
    }
    return pair_df, metrics


@lru_cache(maxsize=1)
def hierarchy_helpers() -> tuple[dict[str, str], dict[str, set[str]]]:
    style_names = load_style_classes()
    style_set = set(style_names)
    parent = {child: node for node, children in STYLE_HIERARCHY.items() for child in children}

    @lru_cache(maxsize=None)
    def descendants(node: str) -> tuple[str, ...]:
        result: set[str] = set()
        if node in style_set and node != "Root":
            result.add(node)
        for child in STYLE_HIERARCHY.get(node, []):
            result.update(descendants(child))
        return tuple(sorted(result))

    descendants_map = {node: set(descendants(node)) for node in set(parent) | {"Root"}}
    return parent, descendants_map


def hierarchy_cluster_sets(style_names: list[str]) -> set[frozenset[int]]:
    name_to_idx = {name: i for i, name in enumerate(style_names)}
    _, descendants_map = hierarchy_helpers()
    clusters: set[frozenset[int]] = set()
    for node, members in descendants_map.items():
        if node == "Root":
            continue
        cluster = frozenset(name_to_idx[name] for name in members if name in name_to_idx)
        if 1 < len(cluster) < len(style_names):
            clusters.add(cluster)
    return clusters


def dendrogram_metrics(
    class_center_dists: np.ndarray,
    style_names: list[str],
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray, np.ndarray]:
    kwargs = {
        "linkage": "average",
        "distance_threshold": 0.0,
        "n_clusters": None,
        "compute_distances": True,
    }
    try:
        model = AgglomerativeClustering(metric="precomputed", **kwargs)
    except TypeError:
        model = AgglomerativeClustering(affinity="precomputed", **kwargs)

    model.fit(class_center_dists)
    n = len(style_names)
    clusters: dict[int, set[int]] = {i: {i} for i in range(n)}
    rows = []
    predicted_clusters: set[frozenset[int]] = set()
    for merge_idx, (left, right) in enumerate(model.children_):
        merged = clusters[int(left)] | clusters[int(right)]
        clusters[n + merge_idx] = merged
        if len(merged) < n:
            predicted_clusters.add(frozenset(merged))
        rows.append(
            {
                "merge_index": merge_idx,
                "left_child": int(left),
                "right_child": int(right),
                "distance": float(model.distances_[merge_idx]),
                "cluster_size": len(merged),
                "styles": "|".join(style_names[idx] for idx in sorted(merged)),
            }
        )

    gt_clusters = hierarchy_cluster_sets(style_names)
    overlap = predicted_clusters & gt_clusters
    precision = len(overlap) / len(predicted_clusters) if predicted_clusters else float("nan")
    recall = len(overlap) / len(gt_clusters) if gt_clusters else float("nan")
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    metrics = {
        "dendrogram_cluster_precision": float(precision),
        "dendrogram_cluster_recall": float(recall),
        "dendrogram_cluster_f1": float(f1),
        "dendrogram_exact_cluster_matches": int(len(overlap)),
        "dendrogram_num_predicted_clusters": int(len(predicted_clusters)),
        "dendrogram_num_ground_truth_clusters": int(len(gt_clusters)),
    }
    return metrics, pd.DataFrame(rows), model.children_, model.distances_


def style_relation_sets(style_names: list[str]) -> dict[str, dict[int, set[int]]]:
    parent, descendants_map = hierarchy_helpers()
    name_to_idx = {name: i for i, name in enumerate(style_names)}

    siblings: dict[int, set[int]] = {}
    cousins: dict[int, set[int]] = {}

    for style in style_names:
        idx = name_to_idx[style]
        direct_parent = parent.get(style)
        sibling_set: set[int] = set()
        cousin_set: set[int] = set()

        if direct_parent is not None:
            for sib in STYLE_HIERARCHY.get(direct_parent, []):
                if sib != style and sib in name_to_idx:
                    sibling_set.add(name_to_idx[sib])

            grandparent = parent.get(direct_parent)
            if grandparent is not None:
                for branch in STYLE_HIERARCHY.get(grandparent, []):
                    if branch == direct_parent:
                        continue
                    for related_style in descendants_map.get(branch, set()):
                        if related_style in name_to_idx:
                            cousin_set.add(name_to_idx[related_style])

        siblings[idx] = sibling_set
        cousins[idx] = cousin_set

    return {"siblings": siblings, "cousins": cousins}


def knn_retrieval_metrics(
    labels: np.ndarray,
    neighbor_indices: np.ndarray,
    style_names: list[str],
    ks: list[int],
) -> dict[str, float]:
    relations = style_relation_sets(style_names)
    neighbor_labels = labels[neighbor_indices]
    metrics: dict[str, float] = {}

    for relation_name, relation_sets in relations.items():
        valid = np.array([bool(relation_sets[int(label)]) for label in labels], dtype=bool)
        metrics[f"knn_{relation_name}_valid_queries"] = int(valid.sum())
        for k in ks:
            hits = []
            for idx in np.where(valid)[0]:
                targets = sorted(relation_sets[int(labels[idx])])
                hits.append(np.isin(neighbor_labels[idx, :k], targets).any())
            metrics[f"knn_{relation_name}_recall_at_{k}"] = (
                float(np.mean(hits)) if hits else float("nan")
            )

    return metrics


def point_to_set_distances(
    point: torch.Tensor,
    other: torch.Tensor,
    geometry: str,
    curvature: float,
) -> torch.Tensor:
    if geometry == "euclidean":
        return torch.cdist(point.unsqueeze(0), other).squeeze(0)
    return poincare_distance(
        point.view(1, 1, -1),
        other.unsqueeze(0),
        curvature=curvature,
    ).view(-1)


def frechet_interpolation_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    prototype_points: torch.Tensor,
    style_names: list[str],
    geometry: str,
    curvature: float,
    target_style: str,
    sample_size: int,
    trials: int,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    if target_style not in style_names:
        raise ValueError(f"Unknown style for Fréchet interpolation: {target_style}")

    target_idx = style_names.index(target_style)
    style_points = torch.from_numpy(embeddings[labels == target_idx])
    if style_points.shape[0] <= sample_size:
        raise ValueError(
            f"Style {target_style} only has {style_points.shape[0]} examples, which is not enough for sample size {sample_size}."
        )

    rng = np.random.default_rng(seed)
    trial_rows = []
    all_indices = np.arange(style_points.shape[0])

    for trial in range(trials):
        sample_idx = rng.choice(all_indices, size=sample_size, replace=False)
        holdout_mask = np.ones(style_points.shape[0], dtype=bool)
        holdout_mask[sample_idx] = False

        sample = style_points[sample_idx]
        holdout = style_points[holdout_mask]

        mean_point = frechet_mean(sample, geometry=geometry, curvature=curvature)
        proto_dists = point_to_set_distances(
            mean_point,
            prototype_points,
            geometry=geometry,
            curvature=curvature,
        )
        holdout_dist = point_to_set_distances(
            mean_point,
            holdout,
            geometry=geometry,
            curvature=curvature,
        ).mean()

        nearest_idx = int(proto_dists.argmin().item())
        competing = torch.cat([proto_dists[:target_idx], proto_dists[target_idx + 1 :]])
        margin = float(competing.min().item() - proto_dists[target_idx].item())

        trial_rows.append(
            {
                "trial": trial,
                "style": target_style,
                "sample_size": sample_size,
                "nearest_prototype_label": nearest_idx,
                "nearest_prototype_style": style_names[nearest_idx],
                "nearest_is_target_style": int(nearest_idx == target_idx),
                "target_prototype_distance": float(proto_dists[target_idx].item()),
                "best_other_prototype_margin": margin,
                "mean_holdout_same_style_distance": float(holdout_dist.item()),
            }
        )

    trial_df = pd.DataFrame(trial_rows)
    metrics = {
        "frechet_style": target_style,
        "frechet_sample_size": int(sample_size),
        "frechet_trials": int(trials),
        "frechet_nearest_prototype_accuracy": float(trial_df["nearest_is_target_style"].mean()),
        "frechet_target_prototype_distance_mean": float(trial_df["target_prototype_distance"].mean()),
        "frechet_target_prototype_distance_std": float(trial_df["target_prototype_distance"].std(ddof=0)),
        "frechet_best_other_margin_mean": float(trial_df["best_other_prototype_margin"].mean()),
        "frechet_holdout_same_style_distance_mean": float(trial_df["mean_holdout_same_style_distance"].mean()),
    }
    return metrics, trial_df


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    ckpt_path = args.ckpt.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else ckpt_path.parent / f"eval_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    _, head, clf, metadata = load_model(
        ckpt_path=ckpt_path,
        geometry_override=args.geometry,
        curvature_override=args.curvature,
        device=device,
    )

    raw_split_count = len(
        pd.read_csv(
            SPLIT_DIR / f"style_{args.split}.csv",
            header=None,
        )
    )
    split_df = load_split_metadata(args.split)
    dataset = FeatureDataset(args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    metrics, y_true, y_pred, topk_idx, embeddings = evaluate(head, clf, loader, device)

    style_names = load_style_classes()
    tree_dists = distance_matrix(style_names)
    conf_mat = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    per_class_total = conf_mat.sum(axis=1)
    per_class_correct = np.diag(conf_mat)
    per_class_acc = np.divide(
        per_class_correct,
        per_class_total,
        out=np.zeros_like(per_class_correct, dtype=np.float64),
        where=per_class_total > 0,
    )

    prototype_points = clf.prototypes.detach().cpu()
    prototype_dists = distance_matrix_from_points(
        prototype_points,
        geometry=metadata["geometry"],
        curvature=metadata["curvature"],
    )

    class_centers = class_centers_from_embeddings(
        embeddings=embeddings,
        labels=y_true,
        geometry=metadata["geometry"],
        curvature=metadata["curvature"],
    )
    class_center_dists = distance_matrix_from_points(
        torch.from_numpy(class_centers),
        geometry=metadata["geometry"],
        curvature=metadata["curvature"],
    )

    avg_style_dists, neighbor_indices, neighbor_dists = collect_distance_statistics(
        embeddings=embeddings,
        labels=y_true,
        geometry=metadata["geometry"],
        curvature=metadata["curvature"],
        ks=sorted(set(args.knn_k)),
        block_size=args.distance_block_size,
    )

    tree_vec = pairwise_values(tree_dists)
    proto_vec = pairwise_values(prototype_dists)
    center_vec = pairwise_values(class_center_dists)

    style_pair_df, distortion_metrics = style_pair_distance_table(
        avg_style_dists=avg_style_dists,
        tree_dists=tree_dists,
        style_names=style_names,
    )
    knn_metrics = knn_retrieval_metrics(
        labels=y_true,
        neighbor_indices=neighbor_indices,
        style_names=style_names,
        ks=sorted(set(args.knn_k)),
    )
    dendro_metrics, dendro_df, dendro_children, dendro_distances = dendrogram_metrics(
        class_center_dists=class_center_dists,
        style_names=style_names,
    )
    frechet_metrics, frechet_trials_df = frechet_interpolation_metrics(
        embeddings=embeddings,
        labels=y_true,
        prototype_points=prototype_points,
        style_names=style_names,
        geometry=metadata["geometry"],
        curvature=metadata["curvature"],
        target_style=args.frechet_style,
        sample_size=args.frechet_sample_size,
        trials=args.frechet_trials,
        seed=args.seed,
    )

    metrics.update(
        {
            "balanced_accuracy": float(per_class_acc.mean()),
            "num_samples": int(len(y_true)),
            "split": args.split,
            "checkpoint": str(ckpt_path),
            "device": str(device),
            "geometry": metadata["geometry"],
            "embedding_dim": metadata["dim"],
            "curvature": metadata["curvature"],
            "epoch": metadata["epoch"],
            "dropped_missing_feature_rows": int(raw_split_count - len(split_df)),
            "prototype_tree_pearson": pearson_corr(tree_vec, proto_vec),
            "prototype_tree_spearman": spearman_corr(tree_vec, proto_vec),
            "class_center_tree_pearson": pearson_corr(tree_vec, center_vec),
            "class_center_tree_spearman": spearman_corr(tree_vec, center_vec),
        }
    )
    metrics.update(tree_distance_summary(tree_dists, y_true, y_pred))
    metrics.update(distortion_metrics)
    metrics.update(dendro_metrics)
    metrics.update(knn_metrics)
    metrics.update(frechet_metrics)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    per_class_df = pd.DataFrame(
        {
            "label": np.arange(NUM_CLASSES),
            "style": style_names,
            "num_examples": per_class_total,
            "num_correct": per_class_correct,
            "accuracy": per_class_acc,
        }
    )
    per_class_df.to_csv(output_dir / "per_class_accuracy.csv", index=False)

    pd.DataFrame(conf_mat, index=style_names, columns=style_names).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    save_confusion_plot(conf_mat, style_names, output_dir / "confusion_matrix.png")

    pd.DataFrame(prototype_dists, index=style_names, columns=style_names).to_csv(
        output_dir / "prototype_distances.csv"
    )
    pd.DataFrame(class_center_dists, index=style_names, columns=style_names).to_csv(
        output_dir / "class_center_distances.csv"
    )
    pd.DataFrame(avg_style_dists, index=style_names, columns=style_names).to_csv(
        output_dir / "style_pair_mean_distances.csv"
    )
    style_pair_df.to_csv(output_dir / "tree_distortion_pairs.csv", index=False)
    dendro_df.to_csv(output_dir / "dendrogram_clusters.csv", index=False)
    save_dendrogram_plot(
        dendro_children,
        dendro_distances,
        style_names,
        output_dir / "dendrogram.png",
    )
    frechet_trials_df.to_csv(output_dir / "frechet_interpolation_trials.csv", index=False)

    predictions_df = pd.DataFrame(
        {
            "path": split_df["path"],
            "row_idx": split_df["row_idx"],
            "true_label": y_true,
            "true_style": [style_names[i] for i in y_true],
            "pred_label": y_pred,
            "pred_style": [style_names[i] for i in y_pred],
            "correct": y_true == y_pred,
            "tree_distance_true_pred": tree_dists[y_true, y_pred],
        }
    )
    for rank in range(topk_idx.shape[1]):
        predictions_df[f"top{rank + 1}_label"] = topk_idx[:, rank]
        predictions_df[f"top{rank + 1}_style"] = [
            style_names[i] for i in topk_idx[:, rank]
        ]
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    pd.DataFrame(
        neighbor_indices,
        columns=[f"neighbor_{rank + 1}_index" for rank in range(neighbor_indices.shape[1])],
    ).to_csv(output_dir / "knn_neighbor_indices.csv", index=False)
    pd.DataFrame(
        neighbor_dists,
        columns=[f"neighbor_{rank + 1}_distance" for rank in range(neighbor_dists.shape[1])],
    ).to_csv(output_dir / "knn_neighbor_distances.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"saved evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
