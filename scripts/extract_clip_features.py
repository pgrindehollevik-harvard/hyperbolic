#!/usr/bin/env python3
"""Extract CLIP ViT-B/16 features for all WikiArt images.

Writes:
    data/features/clip_vitb16.npy   — (N, 512) float16
    data/features/index.parquet     — row_idx, path, style_name
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import open_clip

ROOT    = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "wikiart" / "wikiart"
OUT_DIR = ROOT / "data" / "features"


class WikiArtDataset(Dataset):
    def __init__(self, paths, preprocess):
        self.paths, self.preprocess = paths, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.preprocess(Image.open(self.paths[i]).convert("RGB"))


def main():
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"Device: {device}")

    paths = sorted(IMG_DIR.glob("*/*.jpg"))
    print(f"Images: {len(paths)}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai", device=device)
    model.eval()

    loader = DataLoader(WikiArtDataset(paths, preprocess),
                        batch_size=128, num_workers=8, shuffle=False)

    chunks = []
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader):
            chunks.append(model.encode_image(batch.to(device)).cpu().numpy())
    features = np.concatenate(chunks).astype(np.float16)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "clip_vitb16.npy", features)

    pd.DataFrame({
        "row_idx":    np.arange(len(paths), dtype=np.int64),
        "path":       [str(p.relative_to(IMG_DIR)) for p in paths],
        "style_name": [p.parent.name for p in paths],
    }).to_parquet(OUT_DIR / "index.parquet", index=False)

    print(f"Done in {(time.time()-t0)/60:.1f} min — {len(features)} features saved")


if __name__ == "__main__":
    main()
