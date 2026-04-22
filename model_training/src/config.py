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

    # ── Optimiser ────────────────────────────────────────────────────────
    head_lr:      float = 3e-4     # classifier LR (both stages)
    backbone_lr:  float = 1e-4     # features LR (stage 2 only)
    weight_decay: float = 1e-4
    grad_clip:    float = 1.0

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
