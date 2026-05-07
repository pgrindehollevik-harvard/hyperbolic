"""
WikiArt style hierarchies and tree-distance matrices.

Three hierarchies are provided:
- "default": the art-history lineage tree (Renaissance → Baroque → ...).
- "flat": every leaf is a direct child of Root. Acts as a null/control: the
  tree-distance matrix becomes constant off-diagonal, so no embedding can do
  better than chance on hierarchy-aware metrics. Useful for confirming our
  metrics actually depend on the tree.
- "chronological": a two-level era-based tree (Renaissance / Baroque-Rococo /
  19th century / Early 20th / Modern / Non-Western). Different pair-distances
  from "default" — used to test sensitivity of conclusions to the choice of
  ground-truth tree.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np


STYLE_HIERARCHY: dict[str, list[str]] = {
    "Root": ["Early_Renaissance", "Pop_Art", "Ukiyo_e", "Naive_Art_Primitivism"],
    "Early_Renaissance": ["Northern_Renaissance", "High_Renaissance"],
    "High_Renaissance": ["Mannerism_Late_Renaissance"],
    "Mannerism_Late_Renaissance": ["Baroque"],
    "Baroque": ["Rococo"],
    "Rococo": ["Romanticism"],
    "Romanticism": ["Realism", "Symbolism"],
    "Realism": ["Contemporary_Realism", "Impressionism"],
    "Impressionism": ["Post_Impressionism"],
    "Post_Impressionism": ["Pointillism", "Fauvism", "Cubism"],
    "Cubism": ["Analytical_Cubism", "Synthetic_Cubism"],
    "Symbolism": ["Art_Nouveau", "Expressionism"],
    "Expressionism": ["Abstract_Expressionism"],
    "Abstract_Expressionism": ["Action_painting", "Color_Field_Painting", "Minimalism"],
    "Pop_Art": ["New_Realism"],
}


CHRONOLOGICAL_HIERARCHY: dict[str, list[str]] = {
    "Root": ["Renaissance_Era", "Baroque_Era", "Long_19th_Century", "Early_20th_Century", "Modern_Postwar", "Non_Western"],
    "Renaissance_Era": [
        "Early_Renaissance", "Northern_Renaissance", "High_Renaissance", "Mannerism_Late_Renaissance",
    ],
    "Baroque_Era": ["Baroque", "Rococo"],
    "Long_19th_Century": [
        "Romanticism", "Realism", "Impressionism", "Post_Impressionism",
        "Pointillism", "Symbolism", "Art_Nouveau", "Contemporary_Realism",
    ],
    "Early_20th_Century": [
        "Fauvism", "Cubism", "Analytical_Cubism", "Synthetic_Cubism",
        "Expressionism", "Naive_Art_Primitivism",
    ],
    "Modern_Postwar": [
        "Abstract_Expressionism", "Action_painting", "Color_Field_Painting",
        "Minimalism", "Pop_Art", "New_Realism",
    ],
    "Non_Western": ["Ukiyo_e"],
}


def _flat_hierarchy(styles: list[str]) -> dict[str, list[str]]:
    return {"Root": list(styles)}


HIERARCHIES: dict[str, dict[str, list[str]]] = {
    "default": STYLE_HIERARCHY,
    "chronological": CHRONOLOGICAL_HIERARCHY,
}


REPO_ROOT = Path(__file__).resolve().parents[1]
STYLE_CLASSES_PATH = REPO_ROOT / "data" / "wikiart_csvs" / "style_class.txt"


def load_style_classes() -> list[str]:
    rows = []
    with open(STYLE_CLASSES_PATH) as f:
        for line in f:
            line = line.strip()
            idx_str, name = line.split(maxsplit=1)
            rows.append((int(idx_str), name))
    rows.sort()
    return [name for _, name in rows]


def get_hierarchy(name: str, styles: list[str] | None = None) -> dict[str, list[str]]:
    if name == "flat":
        if styles is None:
            styles = load_style_classes()
        return _flat_hierarchy(styles)
    if name not in HIERARCHIES:
        raise ValueError(f"Unknown hierarchy '{name}'. Options: default, chronological, flat.")
    return HIERARCHIES[name]


# Ack: Used the help of Claude Opus to build distance matrix here.
def distance_matrix(styles: list[str], hierarchy_name: str = "default") -> np.ndarray:
    hierarchy = get_hierarchy(hierarchy_name, styles)
    parent = {c: p for p, children in hierarchy.items() for c in children}

    depth = {"Root": 0}
    stack = ["Root"]
    while stack:
        n = stack.pop()
        for c in hierarchy.get(n, []):
            depth[c] = depth[n] + 1
            stack.append(c)

    def dist(a: str, b: str) -> int:
        steps = 0
        while depth[a] > depth[b]:
            a = parent[a]
            steps += 1
        while depth[b] > depth[a]:
            b = parent[b]
            steps += 1
        while a != b:
            a, b = parent[a], parent[b]
            steps += 2
        return steps

    n = len(styles)
    T = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            T[i, j] = T[j, i] = dist(styles[i], styles[j])
    return T


if __name__ == "__main__":
    styles = load_style_classes()
    for name in ["default", "chronological", "flat"]:
        T = distance_matrix(styles, hierarchy_name=name)
        print(f"--- {name} ---")
        print(f"shape={T.shape} mean={T[T>0].mean():.2f} max={T.max()}")
