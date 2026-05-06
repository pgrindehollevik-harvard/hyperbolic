#!/usr/bin/env python3
"""Extract DINOv2 ViT-B/14 features for all WikiArt images.

This mirrors scripts/extract_clip_features.py but with a non-CLIP
encoder, so the empirical reference tree built from these features
(scripts/empirical_tree.py with --features dinov2) is a genuine
out-of-distribution test of whether Phase~4's findings are
CLIP-specific.

Writes:
    data/features/dinov2_vitb14.npy   — (N, 768) float16
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "wikiart" / "wikiart"
OUT_DIR = ROOT / "data" / "features"


# DINOv2 uses ImageNet normalization (different from CLIP's).
DINOV2_TRANSFORM = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class WikiArtDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return DINOV2_TRANSFORM(Image.open(self.paths[i]).convert("RGB"))


def main() -> None:
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"Device: {device}", flush=True)

    paths = sorted(IMG_DIR.glob("*/*.jpg"))
    print(f"Images: {len(paths)}", flush=True)

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    model = model.to(device).eval()
    print("Model loaded", flush=True)

    # num_workers=0 avoids macOS multiprocessing fork issues that hang
    # silently with no traceback. Sequential preprocessing is the
    # bottleneck but only adds a few minutes for ~81k images.
    loader = DataLoader(
        WikiArtDataset(paths),
        batch_size=64,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )

    chunks = []
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader):
            feat = model(batch.to(device))  # CLS token, (B, 768)
            chunks.append(feat.cpu().numpy())
    features = np.concatenate(chunks).astype(np.float16)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "dinov2_vitb14.npy", features)
    print(f"Done in {(time.time()-t0)/60:.1f} min — {features.shape} features saved")


if __name__ == "__main__":
    main()
