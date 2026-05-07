"""Smoke test: train one Euclidean and one hyperbolic model at d=8.

Useful as a quick end-to-end check that the training pipeline works after
a fresh clone. For the full sweep see scripts/sweep.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train import train_euclidean, train_hyperbolic


if __name__ == "__main__":
    train_euclidean(dim=8, epochs=10)
    train_hyperbolic(dim=8, epochs=10)
