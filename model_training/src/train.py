from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from src.config import CFG
from src.dataset import CardiomegalyDataset, SubmissionDataset
from src.model import (
    CardiomegalyModel,
    build_model,
    freeze_backbone,
    unfreeze_last_blocks,
)
from src.utils import free_device_cache, log_run


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------

def run_epoch(
    model: CardiomegalyModel,
    loader: DataLoader,
    criterion=None,
    optimizer=None,
) -> dict:
    """Single forward pass over *loader*.

    Pass ``optimizer=None`` for evaluation mode.
    Expects batches of ``(images, tabular, labels, names)``.
    """
    is_train = optimizer is not None
    model.train(is_train)

    losses, all_y_true, all_y_prob, all_names = [], [], [], []

    for images, tabular, labels, names in loader:
        images  = images.to(CFG.device)
        tabular = tabular.to(CFG.device)
        labels  = labels.to(CFG.device)

        with torch.set_grad_enabled(is_train):
            logits = model(images, tabular).squeeze(1)
            loss   = criterion(logits, labels) if criterion is not None else None
            probs  = torch.sigmoid(logits)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if loss is not None:
            losses.append(loss.item())
        all_y_true.extend(labels.detach().cpu().numpy())
        all_y_prob.extend(probs.detach().cpu().numpy())
        all_names.extend(list(names))

    y_true = np.array(all_y_true, dtype=np.float32)
    y_prob = np.array(all_y_prob, dtype=np.float32)
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "loss":   float(np.mean(losses)) if losses else np.nan,
        "auc":    float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "acc":    float((y_pred == y_true).mean()),
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "names":  all_names,
    }


def run_epoch_tta(
    model: CardiomegalyModel,
    df: pd.DataFrame,
    tta_transforms: List,
) -> dict:
    """Average predictions across all TTA transforms for labelled data."""
    all_probs, final_names, final_y_true = [], None, None

    for tf in tta_transforms:
        ds     = CardiomegalyDataset(df, transform=tf)
        loader = DataLoader(
            ds, batch_size=CFG.batch_size, shuffle=False,
            num_workers=CFG.num_workers, pin_memory=False,
        )
        out = run_epoch(model, loader)
        all_probs.append(out["y_prob"])
        if final_names is None:
            final_names, final_y_true = out["names"], out["y_true"]

    mean_prob = np.mean(np.stack(all_probs), axis=0)
    mean_pred = (mean_prob >= 0.5).astype(int)

    return {
        "loss":   np.nan,
        "auc":    float(roc_auc_score(final_y_true, mean_prob)) if len(np.unique(final_y_true)) > 1 else np.nan,
        "acc":    float((mean_pred == final_y_true).mean()),
        "y_true": final_y_true,
        "y_prob": mean_prob,
        "y_pred": mean_pred,
        "names":  final_names,
    }


def predict_submission_tta(
    model: CardiomegalyModel,
    submission_dir: str,
    tta_transforms: List,
) -> dict:
    """TTA inference on unlabelled submission images.

    Clinical metadata (age, sex) is unknown → neutral defaults (0.5).
    """
    all_probs, final_names = [], None

    for tf in tta_transforms:
        ds     = SubmissionDataset(submission_dir, transform=tf)
        loader = DataLoader(
            ds, batch_size=CFG.batch_size, shuffle=False,
            num_workers=CFG.num_workers, pin_memory=False,
        )
        model.eval()
        fold_probs, fold_names = [], []
        with torch.no_grad():
            for images, names in loader:
                images  = images.to(CFG.device)
                tabular = torch.full((images.size(0), 2), 0.5, device=CFG.device)
                probs   = torch.sigmoid(model(images, tabular).squeeze(1)).detach().cpu().numpy()
                fold_probs.extend(probs)
                fold_names.extend(list(names))

        all_probs.append(np.array(fold_probs))
        if final_names is None:
            final_names = fold_names

    return {
        "names":  final_names,
        "y_prob": np.mean(np.stack(all_probs), axis=0),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_basic_metrics(y_true, y_prob, threshold: float) -> dict:
    """Compute the full clinical metric set at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    youden      = sensitivity + specificity - 1.0
    auc         = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

    return dict(
        threshold=float(threshold),
        auc=float(auc),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        youden=float(youden),
        accuracy=float(accuracy),
        precision=float(precision),
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
    )


def find_best_threshold(
    y_true,
    y_prob,
    mode: str = "youden",
) -> Tuple[float, pd.DataFrame]:
    """Search all candidate thresholds and pick the best one for *mode*.

    mode options
    ─────────────
    'youden'              – maximises Youden index (sensitivity + specificity - 1)
    'target_sensitivity'  – highest specificity at ≥ CFG.target_sensitivity
    'target_specificity'  – highest sensitivity at ≥ CFG.target_specificity
    """
    candidates = np.concatenate(([0.0], np.unique(np.round(y_prob, 6)), [1.0]))
    tab = pd.DataFrame([compute_basic_metrics(y_true, y_prob, t) for t in candidates])

    if mode == "youden":
        best_row = tab.sort_values(["youden", "auc", "accuracy"], ascending=False).iloc[0]

    elif mode == "target_sensitivity":
        good = tab[tab["sensitivity"] >= CFG.target_sensitivity]
        best_row = (
            good.sort_values(["specificity", "youden"], ascending=False).iloc[0]
            if len(good)
            else tab.iloc[(tab["sensitivity"] - CFG.target_sensitivity).abs().argsort()].iloc[0]
        )

    elif mode == "target_specificity":
        good = tab[tab["specificity"] >= CFG.target_specificity]
        best_row = (
            good.sort_values(["sensitivity", "youden"], ascending=False).iloc[0]
            if len(good)
            else tab.iloc[(tab["specificity"] - CFG.target_specificity).abs().argsort()].iloc[0]
        )

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Choose: youden | target_sensitivity | target_specificity"
        )

    return float(best_row["threshold"]), tab


# ---------------------------------------------------------------------------
# Single training stage
# ---------------------------------------------------------------------------

def train_stage(
    model: CardiomegalyModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    scheduler,
    epochs: int,
    stage_name: str,
) -> Tuple[CardiomegalyModel, pd.DataFrame]:
    """Run one training stage with early stopping on val AUC."""
    best_score, best_state, patience_ctr = -np.inf, None, 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        tr  = run_epoch(model, train_loader, criterion, optimizer)
        val = run_epoch(model, val_loader,   criterion)

        score = val["auc"] if not np.isnan(val["auc"]) else val["acc"]
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()

        history.append({
            "stage":      stage_name,
            "epoch":      epoch,
            "train_loss": tr["loss"],  "train_auc": tr["auc"],
            "val_loss":   val["loss"], "val_auc":   val["auc"], "val_acc": val["acc"],
        })

        print(
            f"[{stage_name}] epoch {epoch:>2}/{epochs} | "
            f"train_loss={tr['loss']:.4f} | "
            f"val_loss={val['loss']:.4f} | "
            f"val_auc={val['auc']:.4f} | "
            f"val_acc={val['acc']:.4f}"
        )

        if score > best_score:
            best_score, best_state, patience_ctr = score, copy.deepcopy(model.state_dict()), 0
        else:
            patience_ctr += 1
            if patience_ctr >= CFG.early_stop_patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


# ---------------------------------------------------------------------------
# Full two-stage training pipeline
# ---------------------------------------------------------------------------

def train_model(
    model: CardiomegalyModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_df: pd.DataFrame,
    n_blocks: int = 7,
    config: Optional[CFG.__class__] = None,
) -> Tuple[CardiomegalyModel, pd.DataFrame]:
    """Run the full two-stage training schedule.

    Stage 1 — frozen backbone: only head + tabular branch are trained.
    Stage 2 — fine-tune:       last *n_blocks* backbone groups are unfrozen.

    Parameters
    ──────────
    model        : freshly built CardiomegalyModel (not yet to device)
    train_loader : DataLoader for training split
    val_loader   : DataLoader for validation split
    train_df     : training DataFrame (needed for pos_weight computation)
    n_blocks     : how many EfficientNet feature groups to unfreeze in stage 2
    config       : override global CFG (optional)

    Returns
    ───────
    (best_model, history_dataframe)
    """
    cfg = config or CFG

    n_neg   = int((train_df["label"] == 0).sum())
    n_pos   = int((train_df["label"] == 1).sum())
    pos_w   = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(cfg.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    print(f"pos_weight: {pos_w.item():.4f}  (neg={n_neg}, pos={n_pos})")
    print(f"Training on: {cfg.device}  |  frozen_epochs: {cfg.frozen_epochs}")

    # ── Stage 1: head only ────────────────────────────────────────────────
    model = freeze_backbone(model)
    opt1  = optim.AdamW(
        model.classifier.parameters(), lr=cfg.head_lr, weight_decay=cfg.weight_decay
    )
    sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=cfg.frozen_epochs, eta_min=1e-6)
    model, hist_frozen = train_stage(
        model, train_loader, val_loader,
        criterion, opt1, sch1,
        epochs=cfg.frozen_epochs, stage_name="frozen",
    )

    # ── Stage 2: fine-tune ────────────────────────────────────────────────
    print(f"\nFine-tuning last {n_blocks} EfficientNet blocks")
    model = unfreeze_last_blocks(model, n_blocks=n_blocks)
    backbone_params = [p for p in model.features.parameters()   if p.requires_grad]
    head_params     = [p for p in model.classifier.parameters() if p.requires_grad]
    opt2 = optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.backbone_lr},
            {"params": head_params,     "lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=cfg.finetune_epochs, eta_min=1e-7)
    model, hist_ft = train_stage(
        model, train_loader, val_loader,
        criterion, opt2, sch2,
        epochs=cfg.finetune_epochs, stage_name="finetune",
    )

    history = pd.concat([hist_frozen, hist_ft], ignore_index=True)
    return model, history


# ---------------------------------------------------------------------------
# Optuna hyperparameter search
# ---------------------------------------------------------------------------

def optuna_search(
    train_ds,
    val_ds,
    train_df: pd.DataFrame,
    n_trials: int = 20,
    config: Optional[CFG.__class__] = None,
):
    """Automatic hyperparameter search using Optuna TPE sampler.

    Each trial runs a short training (2 frozen + 4 fine-tune epochs) and
    returns the best val AUC. Per-epoch output is suppressed to keep logs clean.

    Parameters
    ──────────
    train_ds  : CardiomegalyDataset (training split)
    val_ds    : CardiomegalyDataset (validation split)
    train_df  : training DataFrame (for pos_weight)
    n_trials  : number of Optuna trials
    config    : optional Config override

    Returns
    ───────
    optuna.Study  (call .best_params to get the winner)
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cfg = config or CFG

    def objective(trial: optuna.Trial) -> float:
        head_lr      = trial.suggest_float("head_lr",      1e-4, 1e-2, log=True)
        backbone_lr  = trial.suggest_float("backbone_lr",  5e-6, 5e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        dropout      = trial.suggest_float("dropout",      0.10, 0.50)
        batch_size   = trial.suggest_categorical("batch_size", [16, 32, 64])
        n_blocks     = trial.suggest_int("n_blocks", 2, 8)   # 9 groups total (0–8)

        _kw      = dict(num_workers=cfg.num_workers, pin_memory=False)
        t_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **_kw)
        v_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **_kw)

        m     = build_model(dropout).to(cfg.device)
        n_neg = int((train_df["label"] == 0).sum())
        n_pos = int((train_df["label"] == 1).sum())
        pos_w = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(cfg.device)
        crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        _sink = io.StringIO()
        with contextlib.redirect_stdout(_sink):
            m    = freeze_backbone(m)
            opt1 = optim.AdamW(m.classifier.parameters(), lr=head_lr, weight_decay=weight_decay)
            sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=2, eta_min=1e-6)
            m, _ = train_stage(m, t_loader, v_loader, crit, opt1, sch1, epochs=2, stage_name="s1")

            m  = unfreeze_last_blocks(m, n_blocks=n_blocks)
            bp = [p for p in m.features.parameters()   if p.requires_grad]
            hp = [p for p in m.classifier.parameters() if p.requires_grad]
            opt2 = optim.AdamW(
                [{"params": bp, "lr": backbone_lr}, {"params": hp, "lr": head_lr}],
                weight_decay=weight_decay,
            )
            sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=4, eta_min=1e-7)
            m, hist = train_stage(m, t_loader, v_loader, crit, opt2, sch2, epochs=4, stage_name="s2")

        val_auc = float(hist["val_auc"].max()) if not hist["val_auc"].isna().all() else 0.0

        del m
        free_device_cache(cfg.device)
        return val_auc

    study = optuna.create_study(
        direction="maximize",
        study_name="cardiomegaly_hpo",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    print(f"Starting Optuna search — {n_trials} trials × 6 quick epochs each.")
    print(f"Device: {cfg.device}. Est. time: {n_trials * 1.5:.0f}–{n_trials * 2:.0f} min on MPS.\n")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\n{'='*55}")
    print(f"  Best val AUC : {best.value:.4f}  (trial #{best.number})")
    print(f"  Best params  :")
    for k, v in best.params.items():
        print(f"    {k:>15}: {v}")

    return study


def apply_best_params(study, train_ds, val_ds, test_ds, config=None):
    """Write Optuna best params back to CFG and rebuild DataLoaders.

    Returns
    ───────
    (train_loader, val_loader, test_loader, n_blocks)
    """
    cfg = config or CFG

    cfg.head_lr      = study.best_params["head_lr"]
    cfg.backbone_lr  = study.best_params["backbone_lr"]
    cfg.weight_decay = study.best_params["weight_decay"]
    cfg.dropout      = study.best_params["dropout"]
    cfg.batch_size   = study.best_params["batch_size"]
    n_blocks         = study.best_params["n_blocks"]

    _kw          = dict(num_workers=cfg.num_workers, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  **_kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, **_kw)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, **_kw)

    print(f"Config updated — batch_size={cfg.batch_size}, n_blocks={n_blocks}")
    return train_loader, val_loader, test_loader, n_blocks


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def save_results(
    model: CardiomegalyModel,
    history: pd.DataFrame,
    val_out: dict,
    test_out: dict,
    best_threshold: float,
    output_dir: str,
    model_name: str = "model",
    n_blocks: int = 7,
    config=None,
) -> None:
    """Persist model weights, history, metrics, per-image predictions, and global run log.

    Parameters
    ──────────
    model        : trained CardiomegalyModel
    history      : training history DataFrame
    val_out      : dict from run_epoch_tta on validation split
    test_out     : dict from run_epoch_tta on test split
    best_threshold: decision threshold chosen from validation set
    output_dir   : folder for this run's artefacts
    model_name   : human-readable name shown in the results log
    n_blocks     : number of backbone blocks fine-tuned (logged in table)
    config       : Config instance (defaults to global CFG)
    """
    from src.config import CFG
    cfg = config or CFG

    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
    history.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)

    val_metrics, test_metrics = None, None
    for split_name, out in [("val", val_out), ("test", test_out)]:
        metrics = compute_basic_metrics(out["y_true"], out["y_prob"], best_threshold)
        if split_name == "val":
            val_metrics = metrics
        else:
            test_metrics = metrics
        with open(os.path.join(output_dir, f"{split_name}_metrics_final.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        pd.DataFrame({
            "image_name": out["names"],
            "y_true":     out["y_true"].astype(int),
            "prob":       out["y_prob"],
            "pred":       (out["y_prob"] >= best_threshold).astype(int),
        }).to_csv(os.path.join(output_dir, f"{split_name}_predictions.csv"), index=False)

    print(f"Results saved → {output_dir}")

    # ── Append to global results log ─────────────────────────────────────
    log_run(
        model_name=model_name,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        config=cfg,
        n_blocks=n_blocks,
        log_path=cfg.results_log_path,
    )
