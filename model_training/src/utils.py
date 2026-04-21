from __future__ import annotations

import dataclasses
import os
import random
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from src.config import Config


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def free_device_cache(device: str) -> None:
    """Release unused memory on GPU / MPS (useful between Optuna trials)."""
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def log_run(
    model_name: str,
    val_metrics: dict,
    test_metrics: dict,
    config: "Config",
    n_blocks: int,
    log_path: str = "results_log.csv",
) -> pd.DataFrame:
    """Append one training run to the global results log CSV.

    Creates the file with a header if it does not exist yet, otherwise appends.

    Columns
    ───────
    run_id, model_name, created_at,
    <all Config fields except device/csv_path/image_dir/submission_test_dir>,
    n_blocks,
    val_auc, val_sensitivity, val_specificity, val_youden, val_accuracy, val_precision,
    val_tp, val_tn, val_fp, val_fn,
    test_auc, test_sensitivity, test_specificity, test_youden, test_accuracy, test_precision,
    test_tp, test_tn, test_fp, test_fn

    Parameters
    ──────────
    model_name   : human-readable name for this run (e.g. "efficientnet_b3_run1")
    val_metrics  : dict returned by compute_basic_metrics on the validation split
    test_metrics : dict returned by compute_basic_metrics on the test split
    config       : the Config instance used for this run
    n_blocks     : number of backbone blocks unfrozen in stage 2
    log_path     : path to the CSV results log (created if missing)

    Returns
    ───────
    Full DataFrame of all logged runs (including the new one).
    """
    cfg_dict = dataclasses.asdict(config)

    # exclude path fields and device — not meaningful for comparison
    skip = {"csv_path", "image_dir", "submission_test_dir", "output_dir", "device"}
    hyperparams = {k: v for k, v in cfg_dict.items() if k not in skip}

    row: dict = {
        "run_id":     datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model_name": model_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_blocks":   n_blocks,
        **hyperparams,
    }

    for prefix, metrics in [("val", val_metrics), ("test", test_metrics)]:
        for key, value in metrics.items():
            if key != "threshold":
                row[f"{prefix}_{key}"] = value
        row[f"{prefix}_threshold"] = metrics.get("threshold", float("nan"))

    new_row_df = pd.DataFrame([row])

    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        log_df = pd.concat([log_df, new_row_df], ignore_index=True)
    else:
        log_df = new_row_df

    log_df.to_csv(log_path, index=False)
    print(f"Run logged → {log_path}  ({len(log_df)} total runs)")
    return log_df
