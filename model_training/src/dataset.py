from __future__ import annotations

import os

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CardiomegalyDataset(Dataset):
    """Labelled chest X-ray dataset with clinical features.

    Each item returns:
        img      – transformed image tensor (C, H, W)
        tabular  – float32 tensor [age_norm, sex_bin]
        label    – float32 scalar (1 = Cardiomegaly, 0 = No Finding)
        name     – original filename string
    """

    def __init__(self, df: pd.DataFrame, transform=None) -> None:
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row     = self.df.iloc[idx]
        img     = Image.open(row["image_path"]).convert("L")
        if self.transform:
            img = self.transform(img)
        tabular = torch.tensor([row["age_norm"], row["sex_bin"]], dtype=torch.float32)
        label   = torch.tensor(float(row["label"]), dtype=torch.float32)
        return img, tabular, label, row["image_name"]


class SubmissionDataset(Dataset):
    """Unlabelled test images for final inference.

    Each item returns:
        img   – transformed image tensor
        fname – filename string
    """

    def __init__(self, image_dir: str, transform=None) -> None:
        self.image_dir   = image_dir
        self.transform   = transform
        self.image_files = sorted(
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        fname = self.image_files[idx]
        img   = Image.open(os.path.join(self.image_dir, fname)).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, fname
