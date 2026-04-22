from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

import torchxrayvision as xrv


# ---------------------------------------------------------------------------
# xrv normalisation
# ---------------------------------------------------------------------------
def xrv_normalize_np(pil_img: Image.Image) -> torch.Tensor:
    """PIL grayscale → (1, H, W) float tensor in [-1024, 1024]."""
    arr = np.array(pil_img, dtype=np.float32)          # (H, W) in [0, 255]
    arr = xrv.datasets.normalize(arr, 255)             # → [-1024, 1024]
    arr = arr[None, ...]                               # (1, H, W)
    return torch.from_numpy(arr).float()


# ---------------------------------------------------------------------------
# Labelled dataset (train / val / test)
# ---------------------------------------------------------------------------
class ChestXrayDataset(Dataset):
    """Returns (image_tensor, label, filename) triples.

    `image_tensor` is single-channel in the xrv-normalised range [-1024, 1024],
    ready to be fed directly to a torchxrayvision DenseNet.
    """

    def __init__(self, df: pd.DataFrame, pil_transform=None, use_erasing: bool = False) -> None:
        self.df = df.reset_index(drop=True)
        self.pil_transform = pil_transform
        self.use_erasing = use_erasing
        self._erasing = T.RandomErasing(
            p=0.5, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("L")
        if self.pil_transform is not None:
            img = self.pil_transform(img)
        tensor = xrv_normalize_np(img)
        if self.use_erasing:
            tensor = self._erasing(tensor)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return tensor, label, row["filename"]


# ---------------------------------------------------------------------------
# TTA dataset — flexible source (DataFrame with image_path, or external dir)
# ---------------------------------------------------------------------------
class TTADataset(Dataset):
    """Used by inference passes (one TTA transform per pass)."""

    def __init__(
        self,
        df: pd.DataFrame,
        pil_transform,
        image_dir: Optional[str] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.pil_transform = pil_transform
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        if "image_path" in row and pd.notna(row.get("image_path")):
            path = row["image_path"]
        else:
            path = os.path.join(self.image_dir, row["filename"])
        img = Image.open(path).convert("L")
        img = self.pil_transform(img)
        tensor = xrv_normalize_np(img)
        label = float(row["label"]) if "label" in row and not pd.isna(row.get("label", np.nan)) else 0.0
        name = row["filename"] if "filename" in row else os.path.basename(path)
        return tensor, torch.tensor(label, dtype=torch.float32), name


# ---------------------------------------------------------------------------
# Submission dataset (unlabelled images in a flat directory)
# ---------------------------------------------------------------------------
class SubmissionDataset(Dataset):
    """Unlabelled test images for final inference.

    Returns (image_tensor, filename).
    """

    def __init__(self, image_dir: str, pil_transform=None) -> None:
        self.image_dir = image_dir
        self.pil_transform = pil_transform
        self.image_files = sorted(
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        fname = self.image_files[idx]
        img = Image.open(os.path.join(self.image_dir, fname)).convert("L")
        if self.pil_transform is not None:
            img = self.pil_transform(img)
        tensor = xrv_normalize_np(img)
        return tensor, fname
