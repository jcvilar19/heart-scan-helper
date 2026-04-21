from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

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
    img_size:    int = 300   # EfficientNet-B3 native resolution
    batch_size:  int = 32
    num_workers: int = 0     # keep 0 for macOS / Jupyter (no spawn issues)

    # ── Train / val / test split ─────────────────────────────────────────
    val_size:  float = 0.15
    test_size: float = 0.15

    # ── Training schedule ────────────────────────────────────────────────
    frozen_epochs:   int = 3
    finetune_epochs: int = 15

    # ── Optimiser ────────────────────────────────────────────────────────
    head_lr:             float = 5e-4
    backbone_lr:         float = 3e-5
    weight_decay:        float = 1e-4
    dropout:             float = 0.20
    early_stop_patience: int   = 6

    # ── Dataset normalisation ────────────────────────────────────────────
    use_dataset_stats: bool         = True
    stats_sample_size: Optional[int] = 1200

    # ── Clinical decision thresholds ────────────────────────────────────
    target_sensitivity: float = 0.90
    target_specificity: float = 0.90

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
