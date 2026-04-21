"""
WikiArt style hierarchy and tree-distance matrix T.
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


# Ack: Used the help of Claude Opus to build distance matrix here. 
def distance_matrix(styles: list[str]) -> np.ndarray:
    parent = {c: p for p, children in STYLE_HIERARCHY.items() for c in children}

    depth = {"Root": 0}
    stack = ["Root"]
    while stack:
        n = stack.pop()
        for c in STYLE_HIERARCHY.get(n, []):
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
    T = distance_matrix(styles)
    print(T)
