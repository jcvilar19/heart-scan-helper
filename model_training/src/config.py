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
    finetune_epochs: int = 10      # stage 2: fine-tune with cosine LR
    early_stop_patience: int = 4   # early stop on val AUC during stage 2

    # How many feature blocks to keep frozen during stage 2.
    # DenseNet-121 has 4 dense blocks:
    #   0 → unfreeze everything             (most aggressive fine-tuning)
    #   1 → keep denseblock1 frozen
    #   2 → keep denseblock1–2 frozen       (recommended for small datasets)
    #   3 → keep denseblock1–3 frozen
    #   4 → keep all feature blocks frozen  (only classifier trains in stage 2)
    # For MobileNet / EfficientNet the number refers to sub-modules of model.features.
    frozen_blocks: int = 2

    # ── Optimiser ────────────────────────────────────────────────────────
    head_lr:      float = 3e-4     # classifier LR (both stages)
    backbone_lr:  float = 1e-4     # features LR (stage 2 only)
    weight_decay: float = 1e-4
    grad_clip:    float = 1.0

    # ── Architecture ─────────────────────────────────────────────────────
    # Options: "densenet121" | "mobilenet_v3_large" | "efficientnet_b0" | "efficientnet_b3"
    # densenet121       — torchxrayvision DenseNet-121, pretrained on ~1M chest X-rays (recommended)
    # mobilenet_v3_large — torchvision MobileNetV3-Large, pretrained on ImageNet (faster, lighter)
    # efficientnet_b0   — torchvision EfficientNet-B0,  pretrained on ImageNet (good accuracy/size trade-off)
    # efficientnet_b3   — torchvision EfficientNet-B3,  pretrained on ImageNet (higher accuracy, more params)
    backbone: str = "mobilenet_v3_large"

    # ── Ensemble ─────────────────────────────────────────────────────────
    # True:  train one model per entry in `seeds` and average predictions
    # False: train a single model using only `seed` (faster experimentation)
    use_ensemble: bool = False

    # ── Multi-seed ensemble ──────────────────────────────────────────────
    seeds: List[int] = field(default_factory=lambda: [42, 7, 2024])

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
