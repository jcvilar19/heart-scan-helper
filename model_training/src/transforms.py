from __future__ import annotations

from typing import List, Tuple

import torchvision.transforms as T
from PIL import Image

from src.config import CFG


# ---------------------------------------------------------------------------
# PIL helpers (TTA expects PIL → PIL transforms; xrv normalisation is applied
# downstream inside the Dataset).
# ---------------------------------------------------------------------------
def _pil_hflip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


# ---------------------------------------------------------------------------
# Training and evaluation transforms
# ---------------------------------------------------------------------------
def make_transforms(img_size: int | None = None) -> Tuple[T.Compose, T.Compose]:
    """Return (train_transform, eval_transform) PIL-space pipelines.

    All transforms produce a PIL grayscale image of size (img_size, img_size).
    The downstream Dataset converts it to a single-channel xrv-normalised
    tensor in [-1024, 1024].

    Train pipeline: small affine, mild jitter, light hflip; random erasing
                    happens after xrv normalisation inside the Dataset.
    Eval pipeline:  deterministic resize.
    """
    img_size = img_size if img_size is not None else CFG.img_size

    train_tf = T.Compose([
        T.Resize((img_size + 16, img_size + 16)),
        T.RandomCrop((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(
            degrees=8,
            translate=(0.04, 0.04),
            scale=(0.95, 1.05),
            fill=0,
        ),
        T.ColorJitter(brightness=0.15, contrast=0.15),
    ])

    eval_tf = T.Compose([
        T.Resize((img_size, img_size)),
    ])

    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# Test-time augmentation (TTA) transforms
# ---------------------------------------------------------------------------
def make_tta_transforms(img_size: int | None = None) -> List[T.Compose]:
    """Six deterministic PIL-space transforms.

    All end with a resized PIL image ready for xrv_normalize_np().
    Predictions are averaged across all passes (in logit space) inside
    `tta_predict` / `tta_predict_ensemble`.
    """
    img_size = img_size if img_size is not None else CFG.img_size
    size = (img_size, img_size)

    return [
        T.Compose([T.Resize(size)]),
        T.Compose([T.Resize(size), T.Lambda(_pil_hflip)]),
        T.Compose([T.Resize((img_size + 20, img_size + 20)), T.CenterCrop(size)]),
        T.Compose([T.Resize((img_size - 20, img_size - 20)),
                   T.Pad(10, fill=0), T.CenterCrop(size)]),
        T.Compose([T.Resize(size),
                   T.RandomAffine(degrees=(6, 6), fill=0)]),
        T.Compose([T.Resize(size),
                   T.RandomAffine(degrees=(-6, -6), fill=0)]),
    ]
