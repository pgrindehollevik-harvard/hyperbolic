"""
FeatureDataset — cached CLIP features 

Inputs
------
data/features/clip_vitb16.npy        (N, 512) float16, produced by extract_clip_features.py
data/features/index.csv              row_idx, path, style_name
data/wikiart_csvs/style_train.csv    path,label   (no header)
data/wikiart_csvs/style_val.csv      path,label   (no header)

Yields `(feat_f32[512], label_int)`.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 27

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = REPO_ROOT / "data" / "features" / "clip_vitb16.npy"
INDEX_PATH    = REPO_ROOT / "data" / "features" / "index.csv"
SPLIT_DIR     = REPO_ROOT / "data" / "wikiart_csvs"


# Ack: Claude Code helped with handling an ASCII discrepancy that arose when I saved the data. 
def _ascii_key(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9/_.-]", "", path)


class FeatureDataset(Dataset):
    def __init__(self, split: Literal["train", "val"]):

        features = np.load(FEATURES_PATH).astype(np.float32)
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

        # * Counts corrupted rows. In this dataset there are 2 broken names which could not match. 
        # missing = split_df["row_idx"].isna().sum()
        # if missing:
        #     print(f"[FeatureDataset] warning: dropping {missing} {split} rows "
        #           f"(not in feature cache)")
        
        split_df = split_df.dropna(subset=["row_idx"]).reset_index(drop=True) # Drops corrupted rows

        rows = split_df["row_idx"].to_numpy(dtype=np.int64)
        self.features = torch.from_numpy(features[rows])
        self.labels = torch.tensor(split_df["label"].to_numpy(), dtype=torch.long)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return self.features[i], int(self.labels[i])
