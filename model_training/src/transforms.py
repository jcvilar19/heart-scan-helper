from __future__ import annotations

from typing import List

import torchvision.transforms as T

from src.config import CFG


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _norm_params(mean: float, std: float):
    """Return (mean_list, std_list) for T.Normalize.

    Uses dataset-computed stats when CFG.use_dataset_stats is True,
    otherwise falls back to ImageNet defaults.
    """
    if CFG.use_dataset_stats:
        return [mean] * 3, [std] * 3
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Training and evaluation transforms
# ---------------------------------------------------------------------------

def make_transforms(train_mean: float, train_std: float):
    """Return (train_transform, eval_transform) torchvision pipelines.

    Train pipeline: resize + random crop, horizontal flip, affine, colour jitter,
                    Gaussian blur, random autocontrast, grayscale→3ch, normalise.
    Eval pipeline:  resize, grayscale→3ch, normalise.
    """
    mean, std = _norm_params(train_mean, train_std)

    train_tf = T.Compose([
        T.Resize((CFG.img_size + 20, CFG.img_size + 20)),
        T.RandomCrop((CFG.img_size, CFG.img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.4),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        T.RandomApply([T.RandomAutocontrast()], p=0.3),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    eval_tf = T.Compose([
        T.Resize((CFG.img_size, CFG.img_size)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# Test-time augmentation (TTA) transforms
# ---------------------------------------------------------------------------

def make_tta_transforms(train_mean: float, train_std: float) -> List:
    """Three-view TTA: centre-crop at three scales.

    Returns a list of three transforms (standard, slightly zoomed, more zoomed).
    Predictions are averaged across all three views in run_epoch_tta.
    """
    mean, std = _norm_params(train_mean, train_std)

    def _make_crop(resize_to: int):
        return T.Compose([
            T.Resize((resize_to, resize_to)),
            T.CenterCrop((CFG.img_size, CFG.img_size)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    return [
        T.Compose([
            T.Resize((CFG.img_size, CFG.img_size)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]),
        _make_crop(CFG.img_size + 12),
        _make_crop(CFG.img_size + 20),
    ]
