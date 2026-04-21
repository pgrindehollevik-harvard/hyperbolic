import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train import train_euclidean, train_hyperbolic


if __name__ == "__main__":
    train_euclidean(dim=8, epochs=10)
    train_hyperbolic(dim=8, epochs=10)
