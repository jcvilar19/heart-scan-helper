from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class Config:
    # ── Data paths ────────────────────────────────────────────────────────
    csv_path:            str = "../../data/train_val.csv"
    image_dir:           str = "../../data/images"
    submission_test_dir: str = "../../data/test_images"
    output_dir:          str = "results"
    results_log_path:    str = "results_log.csv"   # global run log (one row per training run)

    # ── Reproducibility ──────────────────────────────────────────────────
    seed: int = 42

    # ── Image / DataLoader ───────────────────────────────────────────────
    img_size:    int = 224   # torchxrayvision DenseNet-121 native resolution
    batch_size:  int = 32
    num_workers: int = 4

    # ── Train / val / test split ─────────────────────────────────────────
    val_size:  float = 0.15
    test_size: float = 0.15

    # ── Training schedule (two-stage) ────────────────────────────────────
    frozen_epochs:   int = 3       # stage 1: head-only warmup
    finetune_epochs: int = 22      # stage 2: full unfreeze with cosine LR
    early_stop_patience: int = 6   # early stop on val AUC during stage 2
    # How many backbone blocks to keep frozen in stage 2 (0 = unfreeze all):
    #   DenseNet-121 : 0–4   dense block groups
    #   RAD-DINO ViT : 0–12  transformer blocks  (recommended: 8)
    frozen_blocks: int = 0

    # ── Optimiser ────────────────────────────────────────────────────────
    head_lr:      float = 3e-4     # classifier LR (both stages)
    backbone_lr:  float = 1e-4     # features LR (stage 2 only)
    weight_decay: float = 1e-4
    grad_clip:    float = 1.0

    # ── Data augmentation ────────────────────────────────────────────────
    # Mixup: interpolates two samples and their labels in every training batch.
    #   mixup_alpha > 0 enables it; λ ~ Beta(α, α).  0 = disabled.
    #   Typical range: 0.2 – 0.4.
    mixup_alpha:     float = 0.0
    # Label smoothing: prevents overconfidence by softening hard {0,1} targets.
    #   y_smooth = y*(1-ε) + 0.5*ε.  0 = disabled.  Typical range: 0.05 – 0.15.
    label_smoothing: float = 0.0

    # ── Architecture ─────────────────────────────────────────────────────
    # Options: "densenet121" | "rad-dino" | "mobilenet_v3_large" | "efficientnet_b0" | "efficientnet_b3"
    # densenet121        — torchxrayvision DenseNet-121, pretrained on ~1M chest X-rays (recommended)
    # rad-dino           — microsoft/rad-dino, DINOv2 ViT-B/14 pretrained on ~1M chest X-rays;
    #                      use img_size=518 (native: 37×37 patches at 14 px); 12 frozen_blocks max
    # mobilenet_v3_large — torchvision MobileNetV3-Large, pretrained on ImageNet (faster, lighter)
    # efficientnet_b0    — torchvision EfficientNet-B0,  pretrained on ImageNet (good accuracy/size trade-off)
    # efficientnet_b3    — torchvision EfficientNet-B3,  pretrained on ImageNet (higher accuracy, more params)
    backbone: str = "efficientnet_b0"

    # ── Ensemble ─────────────────────────────────────────────────────────
    # True:  train one model per entry in `seeds` and average predictions
    # False: train a single model using only `seed` (faster experimentation)
    use_ensemble: bool = True

    # ── Multi-seed ensemble ──────────────────────────────────────────────
    seeds: List[int] = field(default_factory=lambda: [42, 7, 2024])

    # ── Loss function ─────────────────────────────────────────────────────
    # False: standard BCE  |  True: 0.5*BCE + 0.5*(1 - soft_composite)
    use_composite_loss:    bool  = False
    # Blend weight α: α·BCE + (1-α)·(1-soft_composite).  0 = pure composite, 1 = pure BCE.
    composite_loss_alpha:  float = 0.5
    # Temperature for the pairwise-sigmoid soft-AUC term (higher → sharper ranking signal)
    composite_loss_gamma:  float = 1.0

    # ── Inference ────────────────────────────────────────────────────────
    tta_passes:  int = 6           # number of deterministic TTA transforms (max 6)
    n_bootstrap: int = 1000        # bootstrap iterations for threshold stabilisation

    # ── Device (auto-detected) ───────────────────────────────────────────
    device: str = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    def setup(self) -> "Config":
        """Create output directory and return self (for chaining)."""
        os.makedirs(self.output_dir, exist_ok=True)
        return self


# Global singleton — import and use directly, or override fields before training
CFG = Config().setup()
