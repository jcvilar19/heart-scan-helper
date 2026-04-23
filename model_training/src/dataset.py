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
# Normalisation functions (one per backbone family)
# ---------------------------------------------------------------------------
def xrv_normalize_np(pil_img: Image.Image) -> torch.Tensor:
    """PIL grayscale → (1, H, W) float tensor in [-1024, 1024] (torchxrayvision)."""
    arr = np.array(pil_img, dtype=np.float32)          # (H, W) in [0, 255]
    arr = xrv.datasets.normalize(arr, 255)             # → [-1024, 1024]
    arr = arr[None, ...]                               # (1, H, W)
    return torch.from_numpy(arr).float()


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def imagenet_normalize_np(pil_img: Image.Image) -> torch.Tensor:
    """PIL grayscale → (3, H, W) float tensor normalized with ImageNet stats.

    The single grayscale channel is replicated to 3 channels so that ImageNet-
    pretrained backbones (MobileNet, EfficientNet) receive the expected input shape.
    """
    arr = np.array(pil_img, dtype=np.float32) / 255.0           # [0, 1]
    arr = np.stack([arr, arr, arr], axis=0)                      # (3, H, W)
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(arr).float()


# Module-level cache — processor is loaded once and reused across all calls.
_RAD_DINO_PROCESSOR = None


def _get_rad_dino_processor():
    global _RAD_DINO_PROCESSOR
    if _RAD_DINO_PROCESSOR is None:
        from transformers import AutoImageProcessor
        _RAD_DINO_PROCESSOR = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
    return _RAD_DINO_PROCESSOR


def rad_dino_normalize(pil_img: Image.Image) -> torch.Tensor:
    """PIL image → (3, H, W) tensor using RAD-DINO's official AutoImageProcessor.

    Applies the exact same MIMIC-CXR normalization stats used during rad-dino
    pretraining (mean/std provided by the HuggingFace processor config).
    Grayscale images are converted to RGB by replicating the single channel
    before passing to the processor.
    """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    proc = _get_rad_dino_processor()
    out  = proc(images=pil_img, return_tensors="pt")
    return out["pixel_values"][0]   # (3, H, W)


def get_normalize_fn(backbone: str):
    """Return the correct normalization callable for the given backbone name.

    "densenet121" / "densenet121-res224-all"
        → xrv_normalize_np      (1-ch grayscale, [-1024, 1024])
    "rad-dino"
        → rad_dino_normalize    (3-ch RGB via AutoImageProcessor, MIMIC-CXR stats)
          RAD-DINO is a ViT-B/14; feed at 518×518 for best accuracy.
    all other torchvision backbones
        → imagenet_normalize_np (3-ch RGB replicated, ImageNet stats)
    """
    if backbone in ("densenet121", "densenet121-res224-all"):
        return xrv_normalize_np
    if backbone == "rad-dino":
        return rad_dino_normalize
    return imagenet_normalize_np


# ---------------------------------------------------------------------------
# Labelled dataset (train / val / test)
# ---------------------------------------------------------------------------
class ChestXrayDataset(Dataset):
    """Returns (image_tensor, label, filename) triples.

    backbone controls the normalization applied after PIL transforms:
        "densenet121"         → single-channel tensor in [-1024, 1024] (xrv)
        any torchvision model → 3-channel tensor with ImageNet normalization

    When ``CFG.preprocessing_profile == \"model22\"`` and backbone is
    ``rad-dino``, the Model 22 path applies: optional thorax crop (cached
    bboxes), CLAHE, torchvision ``pil_transform``, optional albumentations
    (train only), then RAD-DINO processor + RandomErasing.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pil_transform=None,
        use_erasing: bool = False,
        backbone: str | None = None,
        preprocessing_profile: str | None = None,
    ) -> None:
        from src.config import CFG
        self.df = df.reset_index(drop=True)
        self.pil_transform = pil_transform
        self.use_erasing = use_erasing
        bb = backbone or CFG.backbone
        self._backbone = bb
        prof = preprocessing_profile or getattr(CFG, "preprocessing_profile", "default")
        self._model22_rad = prof == "model22" and bb == "rad-dino"
        self._crop_to_thorax = bool(getattr(CFG, "crop_to_thorax", False)) if self._model22_rad else False
        cache = getattr(CFG, "thorax_bbox_cache_path", "") or ""
        self._bbox_cache = (
            cache
            if cache
            else os.path.join(CFG.output_dir, "lung_bboxes.json")
        )
        self._normalize = get_normalize_fn(bb)
        self._erasing = T.RandomErasing(
            p=0.5, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["image_path"]
        img = Image.open(path).convert("L")

        if self._model22_rad:
            from src import model22_preprocess as m22

            if self._crop_to_thorax:
                img = m22.crop_thorax_pil(img, path, self._bbox_cache)
            img = m22.apply_clahe_pil(img)
            if self.pil_transform is not None:
                img = self.pil_transform(img)
            if self.use_erasing:
                img = m22.augment_medical_pil(img)
            normalize = getattr(self, "_normalize", rad_dino_normalize)
            tensor = normalize(img)
            if self.use_erasing:
                tensor = self._erasing(tensor)
        else:
            if self.pil_transform is not None:
                img = self.pil_transform(img)
            normalize = getattr(self, "_normalize", xrv_normalize_np)
            tensor = normalize(img)
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
        backbone: str | None = None,
        preprocessing_profile: str | None = None,
    ) -> None:
        from src.config import CFG
        self.df = df.reset_index(drop=True)
        self.pil_transform = pil_transform
        self.image_dir = image_dir
        bb = backbone or CFG.backbone
        self._backbone = bb
        prof = preprocessing_profile or getattr(CFG, "preprocessing_profile", "default")
        self._model22_rad = prof == "model22" and bb == "rad-dino"
        self._crop_to_thorax = bool(getattr(CFG, "crop_to_thorax", False)) if self._model22_rad else False
        cache = getattr(CFG, "thorax_bbox_cache_path", "") or ""
        self._bbox_cache = (
            cache
            if cache
            else os.path.join(CFG.output_dir, "lung_bboxes.json")
        )
        self._normalize = get_normalize_fn(bb)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        if "image_path" in row and pd.notna(row.get("image_path")):
            path = row["image_path"]
        else:
            path = os.path.join(self.image_dir or "", row["filename"])
        img = Image.open(path).convert("L")

        if self._model22_rad:
            from src import model22_preprocess as m22

            if self._crop_to_thorax:
                img = m22.crop_thorax_pil(img, path, self._bbox_cache)
            img = m22.apply_clahe_pil(img)
        img = self.pil_transform(img)
        normalize = getattr(self, "_normalize", xrv_normalize_np)
        tensor = normalize(img)
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

    def __init__(
        self,
        image_dir: str,
        pil_transform=None,
        backbone: str | None = None,
        preprocessing_profile: str | None = None,
    ) -> None:
        from src.config import CFG
        self.image_dir = image_dir
        self.pil_transform = pil_transform
        bb = backbone or CFG.backbone
        self._backbone = bb
        prof = preprocessing_profile or getattr(CFG, "preprocessing_profile", "default")
        self._model22_rad = prof == "model22" and bb == "rad-dino"
        self._crop_to_thorax = bool(getattr(CFG, "crop_to_thorax", False)) if self._model22_rad else False
        cache = getattr(CFG, "thorax_bbox_cache_path", "") or ""
        self._bbox_cache = (
            cache
            if cache
            else os.path.join(CFG.output_dir, "lung_bboxes.json")
        )
        self._normalize = get_normalize_fn(bb)
        self.image_files = sorted(
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        fname = self.image_files[idx]
        path = os.path.join(self.image_dir, fname)
        img = Image.open(path).convert("L")
        if self._model22_rad:
            from src import model22_preprocess as m22

            if self._crop_to_thorax:
                img = m22.crop_thorax_pil(img, path, self._bbox_cache)
            img = m22.apply_clahe_pil(img)
        if self.pil_transform is not None:
            img = self.pil_transform(img)
        normalize = getattr(self, "_normalize", xrv_normalize_np)
        tensor = normalize(img)
        return tensor, fname
