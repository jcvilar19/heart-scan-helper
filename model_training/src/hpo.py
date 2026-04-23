"""Hyperparameter search with Optuna (composite score on validation, stage 2).

Tunes only:
    frozen_blocks, bce_pos_weight_scale, backbone_lr, label_smoothing,
    composite_loss_alpha, composite_loss_gamma

and forces ``use_composite_loss = True``.  All other fields are taken from a
frozen snapshot of ``CFG`` at search start.  Each trial runs ``train_one_seed``
with ``baseline.seed`` (single model per trial — ensemble flags are restored
for the optional final ``train()`` call).
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, Optional, Tuple

import optuna
from torch.utils.data import DataLoader

from src.config import CFG, Config
from src.train import train_one_seed, train
from src.utils import free_device_cache


def _max_frozen_blocks(backbone: str) -> int:
    if backbone == "rad-dino":
        return 12
    if backbone in ("densenet121", "densenet121-res224-all"):
        return 4
    return 8


def _trial_cfg(baseline: Config, trial: optuna.Trial) -> Config:
    """Build a Config for one Optuna trial (only the tuned fields differ)."""
    cfg = copy.deepcopy(baseline)
    cfg.use_composite_loss = True

    max_fb = _max_frozen_blocks(cfg.backbone)
    cfg.frozen_blocks = trial.suggest_int("frozen_blocks", 0, max_fb)

    cfg.bce_pos_weight_scale = trial.suggest_float("bce_pos_weight_scale", 0.0, 1.5)
    cfg.backbone_lr = trial.suggest_float("backbone_lr", 1e-6, 5e-4, log=True)
    cfg.label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.12)
    cfg.composite_loss_alpha = trial.suggest_float("composite_loss_alpha", 0.15, 0.95)
    cfg.composite_loss_gamma = trial.suggest_float("composite_loss_gamma", 0.5, 4.0)
    return cfg


def _objective(
    baseline: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    study_out_dir: str,
    trial: optuna.Trial,
) -> float:
    cfg = _trial_cfg(baseline, trial)
    trial_dir = os.path.join(study_out_dir, f"trial_{trial.number:04d}")
    os.makedirs(trial_dir, exist_ok=True)

    try:
        _, best_score, _, _ = train_one_seed(
            baseline.seed,
            train_loader,
            val_loader,
            output_dir=trial_dir,
            config=cfg,
        )
    finally:
        free_device_cache(baseline.device)

    if best_score is None or (isinstance(best_score, float) and best_score != best_score):
        return -1.0
    return float(best_score)


def run_hyperparameter_search(
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_trials: int = 15,
    output_dir: Optional[str] = None,
    baseline: Optional[Config] = None,
    run_final_train: bool = True,
    study_name: str = "cardiomegaly_hpo",
) -> Tuple[Dict[str, Any], float, Any, Any]:
    """Run Optuna TPE search, write best params to disk, apply to ``CFG``, optionally re-train.

    **Objective:** maximise the same score used for checkpointing in stage 2
    (``baseline.checkpoint_metric`` — default: validation composite).

    **Per trial:** one full ``train_one_seed`` with ``baseline.seed`` and
    ``use_composite_loss=True``; tuned hyperparameters are sampled by Optuna.
    ``baseline.use_ensemble`` is ignored during search (always one seed per
    trial); after search it is restored onto ``CFG`` before ``run_final_train``.

    Parameters
    ----------
    train_loader, val_loader
        Same loaders as in the training notebook.
    n_trials
        Number of Optuna trials (each runs a full two-stage train).
    output_dir
        Directory for per-trial checkpoints and ``hpo_best_params.json``.
        Default: ``baseline.output_dir / "hpo"``.
    baseline
        Frozen snapshot; default ``copy.deepcopy(CFG)`` at call time.
    run_final_train
        If True, after the study finishes, copy best params to global ``CFG``
        and call ``train()`` once with the original ``use_ensemble`` / ``seeds``.
    study_name
        Optuna study name (storage is in-memory).

    Returns
    -------
    (best_params, best_value, models, history)
        If ``run_final_train`` is False, ``models`` and ``history`` are ``None``.
    """
    baseline = baseline if baseline is not None else copy.deepcopy(CFG)
    study_out_dir = output_dir or os.path.join(baseline.output_dir, "hpo")
    os.makedirs(study_out_dir, exist_ok=True)

    # Persist baseline (everything except what Optuna overwrites) for reproducibility
    with open(os.path.join(study_out_dir, "hpo_baseline_config.json"), "w") as f:
        json.dump(
            {k: v for k, v in baseline.__dict__.items() if not k.startswith("_")},
            f,
            indent=2,
            default=str,
        )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=baseline.seed),
    )

    study.optimize(
        lambda tr: _objective(baseline, train_loader, val_loader, study_out_dir, tr),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_trial = study.best_trial
    best_value = float(best_trial.value) if best_trial.value is not None else float("-inf")

    best_params: Dict[str, Any] = dict(best_trial.params)
    best_params["use_composite_loss"] = True

    best_path = os.path.join(study_out_dir, "hpo_best_params.json")
    with open(best_path, "w") as f:
        json.dump({"best_value": best_value, "params": best_params}, f, indent=2)
    try:
        study.trials_dataframe().to_csv(
            os.path.join(study_out_dir, "hpo_trials_history.csv"), index=False,
        )
    except Exception:
        pass
    print(f"\n[HPO] Best value ({baseline.checkpoint_metric}) = {best_value:.6f}")
    print(f"[HPO] Best params → {best_path}")

    # Apply to global CFG (all tuned keys + use_composite_loss)
    for key, val in best_params.items():
        if hasattr(CFG, key):
            setattr(CFG, key, val)

    # Restore ensemble settings from baseline onto global CFG
    CFG.use_ensemble = baseline.use_ensemble
    CFG.seeds = copy.deepcopy(baseline.seeds)

    models, history = None, None
    if run_final_train:
        print("\n[HPO] Running final training with best hyperparameters ...")
        models, history = train(
            train_loader,
            val_loader,
            output_dir=baseline.output_dir,
            config=CFG,
        )
    return best_params, best_value, models, history
