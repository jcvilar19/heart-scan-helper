from __future__ import annotations

import copy
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
from src.dataset import ChestXrayDataset, SubmissionDataset, TTADataset
from src.model import (
    build_model,
    cardio_logit,
    freeze_backbone,
    partial_unfreeze,
    trainable_params,
    unfreeze_all,
)
from src.transforms import make_tta_transforms
from src.utils import free_device_cache, log_run, set_seed


# ---------------------------------------------------------------------------
# Mixup helper
# ---------------------------------------------------------------------------
def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a randomly mixed batch and the corresponding soft labels.

    λ ~ Beta(α, α).  When α ≤ 0 the original batch is returned unchanged.

    Args:
        x:     Image tensor  (B, C, H, W)  on the training device.
        y:     Label tensor  (B,)  – may already be soft (e.g. after smoothing).
        alpha: Beta distribution parameter.  Typical: 0.2 – 0.4.
    """
    if alpha <= 0:
        return x, y
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[idx]
    mixed_y = lam * y + (1.0 - lam) * y[idx]
    return mixed_x, mixed_y


# ---------------------------------------------------------------------------
# Differentiable composite loss
# ---------------------------------------------------------------------------
class SoftCompositeLoss(nn.Module):
    """Differentiable approximation of composite = 0.5·AUC + 0.25·sens + 0.25·spec.

    Minimises ``1 - soft_composite``, blended with standard BCE for stability.

    Soft-AUC
        Pairwise sigmoid over all (positive, negative) logit pairs in the batch:
        ``soft_auc = mean( σ(γ · (logit⁺ − logit⁻)) )``
        where γ (``auc_gamma``) is a sharpness temperature.

    Soft-sens / soft-spec
        ``soft_sens = mean( σ(logit) | y=1 )``
        ``soft_spec = mean( 1 − σ(logit) | y=0 )``

    Total loss
        ``α · BCE  +  (1 − α) · (1 − soft_composite)``

    Args:
        alpha:     Weight of BCE in the blend (0 = pure composite, 1 = pure BCE).
        auc_gamma: Temperature for the pairwise sigmoid (higher → sharper AUC signal).
        eps:       Numerical stability floor.
    """

    def __init__(self, alpha: float = 0.5, auc_gamma: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.alpha     = alpha
        self.auc_gamma = auc_gamma
        self.eps       = eps
        self._bce      = nn.BCEWithLogitsLoss()

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self._bce(logit, target)

        prob     = torch.sigmoid(logit)
        # Use > 0.5 so the masks work correctly for both hard labels {0,1}
        # and soft targets produced by mixup or label smoothing.
        pos_mask = (target > 0.5)
        neg_mask = ~pos_mask
        n_pos    = pos_mask.sum()
        n_neg    = neg_mask.sum()

        # ── Soft AUC (pairwise) ──────────────────────────────────────────────
        if n_pos > 0 and n_neg > 0:
            pos_logits = logit[pos_mask]                                  # (n_pos,)
            neg_logits = logit[neg_mask]                                  # (n_neg,)
            diff       = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)  # (n_pos, n_neg)
            soft_auc   = torch.sigmoid(self.auc_gamma * diff).mean()
        else:
            soft_auc = torch.tensor(0.5, device=logit.device, dtype=logit.dtype)

        # ── Soft sensitivity / specificity ──────────────────────────────────
        soft_sens = prob[pos_mask].mean() if n_pos > 0 else torch.tensor(
            0.0, device=logit.device, dtype=logit.dtype)
        soft_spec = (1.0 - prob[neg_mask]).mean() if n_neg > 0 else torch.tensor(
            0.0, device=logit.device, dtype=logit.dtype)

        soft_composite  = 0.5 * soft_auc + 0.25 * soft_sens + 0.25 * soft_spec
        composite_loss  = 1.0 - soft_composite

        return self.alpha * bce_loss + (1.0 - self.alpha) * composite_loss


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------
def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip: Optional[float] = None,
    mixup_alpha: float = 0.0,
    label_smoothing: float = 0.0,
) -> dict:
    """Single forward pass over *loader*.

    Pass ``optimizer=None`` for evaluation (mixup and smoothing are skipped).
    Expects (image, label, name) batches. Uses CUDA AMP when available.

    Hard original labels are always accumulated for metric computation;
    the (potentially mixed + smoothed) soft labels are only used for the loss.
    """
    is_train = optimizer is not None
    model.train(is_train)

    losses, logits_all, labels_all, names_all = [], [], [], []
    pin = (CFG.device == "cuda")
    grad_clip = grad_clip if grad_clip is not None else CFG.grad_clip

    amp_ctx = torch.cuda.amp.autocast(enabled=(CFG.device == "cuda"))
    for x, y, names in loader:
        x = x.to(CFG.device, non_blocking=pin)
        y = y.to(CFG.device, non_blocking=pin)

        # Keep hard labels for metric accumulation (before any augmentation)
        y_hard = y.detach().clone()

        if is_train:
            # Mixup: interpolate two samples + their labels in-place
            if mixup_alpha > 0.0:
                x, y = mixup_data(x, y, alpha=mixup_alpha)
            # Label smoothing: y_smooth = y*(1-ε) + 0.5*ε
            if label_smoothing > 0.0:
                y = y * (1.0 - label_smoothing) + 0.5 * label_smoothing

        with torch.set_grad_enabled(is_train):
            with amp_ctx:
                logit = cardio_logit(model, x)
                loss = criterion(logit, y) if criterion is not None else None

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params(model), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params(model), grad_clip)
                    optimizer.step()

        if loss is not None:
            losses.append(loss.item())
        logits_all.append(logit.detach().float().cpu().numpy())
        labels_all.append(y_hard.float().cpu().numpy())   # always hard labels
        names_all.extend(list(names))

    y_true  = np.concatenate(labels_all)
    y_logit = np.concatenate(logits_all)
    y_prob  = 1.0 / (1.0 + np.exp(-y_logit))
    auc     = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    # Per-epoch metrics at threshold=0.5 (used for progress logging)
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc  = float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else float("nan")
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    composite = 0.5 * (auc if not np.isnan(auc) else 0.0) + 0.25 * sens + 0.25 * spec

    return {
        "loss":      float(np.mean(losses)) if losses else float("nan"),
        "auc":       float(auc),
        "acc":       acc,
        "sens":      sens,
        "spec":      spec,
        "composite": float(composite),
        "y_true":    y_true,
        "y_prob":    y_prob,
        "names":     names_all,
    }


# ---------------------------------------------------------------------------
# Single-seed two-stage training
# ---------------------------------------------------------------------------
def train_one_seed(
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Optional[str] = None,
    config=None,
) -> Tuple[nn.Module, float, str, list[dict]]:
    """Train ONE model end-to-end (frozen warmup → full fine-tune).

    Returns (best_model, best_val_auc, checkpoint_path, history).
    """
    cfg = config or CFG
    output_dir = output_dir or cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    set_seed(seed)

    model        = build_model(cfg.backbone).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())

    print("\n" + "=" * 80)
    print(f"  Seed    : {seed}")
    print(f"  Model   : {model.__class__.__name__}  ({total_params:,} total params)")
    print(f"  Backbone: {cfg.backbone}")
    print(f"  Device  : {cfg.device}")
    print("=" * 80)

    if cfg.use_composite_loss:
        criterion = SoftCompositeLoss(
            alpha=cfg.composite_loss_alpha,
            auc_gamma=cfg.composite_loss_gamma,
        )
        print(
            f"  Loss    : SoftCompositeLoss  "
            f"(α={cfg.composite_loss_alpha}, γ={cfg.composite_loss_gamma})"
        )
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("  Loss    : BCEWithLogitsLoss")

    mixup_alpha     = getattr(cfg, "mixup_alpha",     0.0)
    label_smoothing = getattr(cfg, "label_smoothing", 0.0)
    if mixup_alpha > 0:
        print(f"  Mixup   : α={mixup_alpha}")
    if label_smoothing > 0:
        print(f"  Smoothing: ε={label_smoothing}")

    scaler    = torch.cuda.amp.GradScaler(enabled=(cfg.device == "cuda"))
    history: list[dict] = []

    _aug_kw = dict(mixup_alpha=mixup_alpha, label_smoothing=label_smoothing)

    # ── Stage 1: frozen backbone, head-only warmup ─────────────────────────
    freeze_backbone(model)
    n_trainable = sum(p.numel() for p in trainable_params(model))
    print(f"\n  Stage 1 — all backbone blocks FROZEN  |  {n_trainable:,} trainable params")
    opt_frozen = optim.AdamW(
        trainable_params(model), lr=cfg.head_lr, weight_decay=cfg.weight_decay,
    )
    for ep in range(1, cfg.frozen_epochs + 1):
        t = run_one_epoch(model, train_loader, criterion, opt_frozen, scaler, **_aug_kw)
        v = run_one_epoch(model, val_loader, criterion)
        history.append({
            "seed": seed, "stage": "frozen", "epoch": ep,
            "train_loss": t["loss"], "train_auc": t["auc"],
            "train_acc": t["acc"], "train_composite": t["composite"],
            "val_loss":  v["loss"], "val_auc":   v["auc"],
            "val_acc":   v["acc"], "val_sens":   v["sens"],
            "val_spec":  v["spec"], "val_composite": v["composite"],
            "lr": opt_frozen.param_groups[0]["lr"],
        })
        print(
            f"  [frozen] {ep}/{cfg.frozen_epochs}  "
            f"loss={t['loss']:.4f}  train_acc={t['acc']*100:.1f}%  |  "
            f"val_auc={v['auc']:.4f}  val_acc={v['acc']*100:.1f}%  "
            f"sens={v['sens']:.3f}  spec={v['spec']:.3f}  comp={v['composite']:.4f}"
        )

    # ── Stage 2: partial or full fine-tune ───────────────────────────────
    frozen_blocks = getattr(cfg, "frozen_blocks", 0)
    partial_unfreeze(model, frozen_blocks)
    n_trainable = sum(p.numel() for p in trainable_params(model))
    if frozen_blocks == 0:
        stage2_label = "all blocks UNFROZEN"
    else:
        stage2_label = f"{frozen_blocks} block(s) still FROZEN"
    print(f"\n  Stage 2 — {stage2_label}  |  {n_trainable:,} trainable params")

    opt_ft = optim.AdamW(
        [
            {"params": model.features.parameters(),   "lr": cfg.backbone_lr},
            {"params": model.classifier.parameters(), "lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt_ft, T_max=cfg.finetune_epochs, eta_min=cfg.backbone_lr * 0.01,
    )

    best_auc, best_state, patience_ctr = -1.0, None, 0
    for ep in range(1, cfg.finetune_epochs + 1):
        t = run_one_epoch(model, train_loader, criterion, opt_ft, scaler, **_aug_kw)
        v = run_one_epoch(model, val_loader, criterion)
        sched.step()
        history.append({
            "seed": seed, "stage": "finetune", "epoch": ep,
            "train_loss": t["loss"], "train_auc": t["auc"],
            "train_acc": t["acc"], "train_composite": t["composite"],
            "val_loss":  v["loss"], "val_auc":   v["auc"],
            "val_acc":   v["acc"], "val_sens":   v["sens"],
            "val_spec":  v["spec"], "val_composite": v["composite"],
            "lr": opt_ft.param_groups[0]["lr"],
        })
        print(
            f"  [ft]     {ep}/{cfg.finetune_epochs}  "
            f"loss={t['loss']:.4f}  train_acc={t['acc']*100:.1f}%  |  "
            f"val_auc={v['auc']:.4f}  val_acc={v['acc']*100:.1f}%  "
            f"sens={v['sens']:.3f}  spec={v['spec']:.3f}  comp={v['composite']:.4f}  "
            f"lr={opt_ft.param_groups[0]['lr']:.2e}"
        )

        if v["auc"] > best_auc:
            best_auc, best_state, patience_ctr = (
                v["auc"], copy.deepcopy(model.state_dict()), 0
            )
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.early_stop_patience:
                print(f"  [ft]     early stop at epoch {ep} (best val AUC = {best_auc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = os.path.join(output_dir, f"model_seed{seed}.pth")
    torch.save(best_state if best_state is not None else model.state_dict(), ckpt_path)
    print(f"[seed={seed}] Best val AUC = {best_auc:.4f}   checkpoint → {ckpt_path}")

    return model, best_auc, ckpt_path, history


# ---------------------------------------------------------------------------
# Multi-seed ensemble training
# ---------------------------------------------------------------------------
def train_ensemble(
    train_loader: DataLoader,
    val_loader: DataLoader,
    seeds: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    config=None,
) -> Tuple[List[Tuple[int, nn.Module, float, str]], pd.DataFrame]:
    """Train one model per seed and return (models_list, full_history_df).

    `models_list` items: (seed, trained_model, best_val_auc, checkpoint_path).
    """
    cfg = config or CFG
    seeds = seeds if seeds is not None else cfg.seeds
    output_dir = output_dir or cfg.output_dir

    print(f"  ENSEMBLE TRAINING STARTED")

    models, all_history = [], []
    for seed in seeds:
        m, auc, ckpt, hist = train_one_seed(
            seed, train_loader, val_loader,
            output_dir=output_dir, config=cfg,
        )
        models.append((seed, m, auc, ckpt))
        all_history.extend(hist)
        free_device_cache(cfg.device)

    history_df = pd.DataFrame(all_history)
    history_df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)

    frozen_blocks = getattr(cfg, "frozen_blocks", 0)
    stage2_label  = "all blocks unfrozen" if frozen_blocks == 0 else f"{frozen_blocks} block(s) frozen"

    print("\n" + "═" * 80)
    print(f"  ENSEMBLE COMPLETE")
    print(f"  Backbone      : {cfg.backbone}")
    print(f"  Frozen blocks : {frozen_blocks}  ({stage2_label} in Stage 2)")
    print(f"  Seeds trained : {len(models)}")
    print("  Per-seed best val AUC:")
    for seed, _, auc, _ in models:
        print(f"    seed {seed:>5} : {auc:.4f}")
    best_seed = max(models, key=lambda x: x[2])
    print(f"  Best seed     : {best_seed[0]}  (AUC={best_seed[2]:.4f})")
    print("═" * 80)

    return models, history_df


# ---------------------------------------------------------------------------
# Unified entry point (respects CFG.use_ensemble)
# ---------------------------------------------------------------------------
def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Optional[str] = None,
    config=None,
) -> Tuple[List[Tuple[int, nn.Module, float, str]], pd.DataFrame]:
    """Train and return (models_list, history_df) — same format as train_ensemble.

    Behaviour is controlled by CFG.use_ensemble:
        True  → delegates to train_ensemble (one model per seed in CFG.seeds)
        False → trains a single model with CFG.seed and wraps result in the
                same list format so the rest of the pipeline works unchanged.
    """
    cfg = config or CFG
    if cfg.use_ensemble:
        return train_ensemble(train_loader, val_loader, output_dir=output_dir, config=cfg)

    m, auc, ckpt, hist = train_one_seed(
        cfg.seed, train_loader, val_loader, output_dir=output_dir, config=cfg,
    )
    history_df = pd.DataFrame(hist)
    history_df.to_csv(
        os.path.join(output_dir or cfg.output_dir, "training_history.csv"), index=False,
    )
    return [(cfg.seed, m, auc, ckpt)], history_df


# ---------------------------------------------------------------------------
# TTA inference
# ---------------------------------------------------------------------------
def tta_predict(
    model: nn.Module,
    df: pd.DataFrame,
    image_dir: Optional[str] = None,
    has_labels: bool = True,
    tta_transforms: Optional[List] = None,
    config=None,
) -> dict:
    """Run TTA inference for ONE model on a DataFrame.

    Predictions are averaged in **logit space** across all TTA passes.
    """
    cfg = config or CFG
    tta_transforms = tta_transforms or make_tta_transforms(cfg.img_size)
    tta_transforms = tta_transforms[:cfg.tta_passes]

    all_logits: list[np.ndarray] = []
    names_ref, labels_ref = None, None

    pin = (cfg.device == "cuda")
    amp_ctx = torch.cuda.amp.autocast(enabled=(cfg.device == "cuda"))

    for tf in tta_transforms:
        ds = TTADataset(df, tf, image_dir)
        loader = DataLoader(
            ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
            pin_memory=pin, shuffle=False,
        )
        pass_logits, pass_names, pass_labels = [], [], []
        model.eval()
        with torch.no_grad(), amp_ctx:
            for x, y, names in loader:
                x = x.to(cfg.device, non_blocking=pin)
                logit = cardio_logit(model, x).float().cpu().numpy()
                pass_logits.append(logit)
                pass_names.extend(list(names))
                if has_labels:
                    pass_labels.append(y.numpy())
        all_logits.append(np.concatenate(pass_logits))
        if names_ref is None:
            names_ref  = pass_names
            labels_ref = np.concatenate(pass_labels) if has_labels else None

    mean_logit = np.stack(all_logits, axis=0).mean(axis=0)
    mean_prob  = (1.0 / (1.0 + np.exp(-mean_logit))).astype(np.float32)
    return {
        "names":      names_ref,
        "y_prob":     mean_prob,
        "y_true":     labels_ref,
        "mean_logit": mean_logit,
    }


def tta_predict_ensemble(
    models_list: List[Tuple[int, nn.Module, float, str]],
    df: pd.DataFrame,
    image_dir: Optional[str] = None,
    has_labels: bool = True,
    tta_transforms: Optional[List] = None,
    config=None,
) -> dict:
    """Run TTA for every model in `models_list` and average in logit space."""
    cfg = config or CFG
    all_logits: list[np.ndarray] = []
    names_ref, labels_ref = None, None

    for (seed, model, _, _) in models_list:
        print(f"  TTA with seed={seed}...")
        pred = tta_predict(
            model, df, image_dir=image_dir, has_labels=has_labels,
            tta_transforms=tta_transforms, config=cfg,
        )
        all_logits.append(pred["mean_logit"])
        if names_ref is None:
            names_ref  = pred["names"]
            labels_ref = pred["y_true"]

    mean_logit = np.stack(all_logits, axis=0).mean(axis=0)
    mean_prob  = (1.0 / (1.0 + np.exp(-mean_logit))).astype(np.float32)
    return {"names": names_ref, "y_prob": mean_prob, "y_true": labels_ref}


# ---------------------------------------------------------------------------
# Submission inference
# ---------------------------------------------------------------------------
def predict_submission(
    models_list: List[Tuple[int, nn.Module, float, str]],
    submission_dir: str,
    tta_transforms: Optional[List] = None,
    config=None,
) -> dict:
    """TTA + ensemble inference on an unlabelled submission directory.

    Wraps the directory in a DataFrame so we can reuse `tta_predict_ensemble`.
    """
    cfg = config or CFG
    files = sorted(
        f for f in os.listdir(submission_dir)
        if os.path.isfile(os.path.join(submission_dir, f))
        and f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    sub_df = pd.DataFrame({"filename": files})
    return tta_predict_ensemble(
        models_list, sub_df,
        image_dir=submission_dir, has_labels=False,
        tta_transforms=tta_transforms, config=cfg,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics_at_threshold(y_true, y_prob, threshold: float) -> dict:
    """Composite-grading-aware metric set at a given threshold.

    composite = 0.5·AUC + 0.25·sensitivity + 0.25·specificity
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc  = (tp + tn) / (tp + tn + fp + fn)
    auc  = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    composite = 0.5 * auc + 0.25 * sens + 0.25 * spec
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "threshold":   float(threshold),
        "auc":         float(auc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "accuracy":    float(acc),
        "youden":      float(sens + spec - 1.0),
        "composite":   float(composite),
        "precision":   float(precision),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# Backwards-compatible alias (used by older notebook cells)
compute_basic_metrics = metrics_at_threshold


def find_best_threshold(y_true, y_prob) -> Tuple[float, dict]:
    """Pick the threshold that maximises sensitivity + specificity (Youden's J)."""
    candidates = np.unique(np.round(np.concatenate([[0.0], y_prob, [1.0]]), 6))
    best_score, best_row = -np.inf, None
    for thr in candidates:
        m = metrics_at_threshold(y_true, y_prob, thr)
        score = m["sensitivity"] + m["specificity"]
        if score > best_score:
            best_score, best_row = score, m
    return float(best_row["threshold"]), best_row


def bootstrap_threshold(
    y_true, y_prob,
    n_boot: Optional[int] = None,
    seed: Optional[int] = None,
) -> float:
    """Bootstrap-stabilised threshold (median across resamples).

    Generalises better than a single-shot pick on the raw val set.
    """
    n_boot = n_boot if n_boot is not None else CFG.n_bootstrap
    seed   = seed   if seed   is not None else CFG.seed

    rng = np.random.RandomState(seed)
    thrs: list[float] = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        thr, _ = find_best_threshold(y_true[idx], y_prob[idx])
        thrs.append(thr)
    return float(np.median(thrs)) if thrs else 0.5


def select_threshold(y_true, y_prob, config=None) -> Tuple[float, dict, dict]:
    """Pick the better of (single-shot) vs (bootstrap) thresholds on composite.

    Bootstrap is preferred unless its composite is clearly worse (margin 0.005).
    Returns (chosen_threshold, single_metrics, bootstrap_metrics).
    """
    cfg = config or CFG
    thr_single, _ = find_best_threshold(y_true, y_prob)
    thr_boot      = bootstrap_threshold(y_true, y_prob, n_boot=cfg.n_bootstrap, seed=cfg.seed)
    m_single = metrics_at_threshold(y_true, y_prob, thr_single)
    m_boot   = metrics_at_threshold(y_true, y_prob, thr_boot)
    chosen = thr_boot if m_boot["composite"] >= m_single["composite"] - 0.005 else thr_single
    return float(chosen), m_single, m_boot


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------
def save_results(
    models_list: List[Tuple[int, nn.Module, float, str]],
    history: pd.DataFrame,
    val_out: dict,
    test_out: dict,
    best_threshold: float,
    output_dir: str,
    model_name: str = "model",
    config=None,
) -> None:
    """Persist per-seed checkpoints, history, metrics, predictions, and global log.

    Per-seed `.pth` files are already written by `train_one_seed`; here we
    only re-save them under the conventional name and write the metrics +
    per-image prediction CSVs.
    """
    cfg = config or CFG
    os.makedirs(output_dir, exist_ok=True)

    # ── Metric files + per-image predictions ─────────────────────────────
    val_metrics  = metrics_at_threshold(val_out["y_true"],  val_out["y_prob"],  best_threshold)
    test_metrics = metrics_at_threshold(test_out["y_true"], test_out["y_prob"], best_threshold)

    for split_name, metrics in [("val", val_metrics), ("test", test_metrics)]:
        with open(os.path.join(output_dir, f"{split_name}_metrics_final.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    history.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)

    for split_name, out in [("val", val_out), ("test", test_out)]:
        y_true = out["y_true"].astype(int)
        y_pred = (out["y_prob"] >= best_threshold).astype(int)
        pd.DataFrame({
            "filename": out["names"],
            "y_true":   y_true,
            "prob":     out["y_prob"],
            "pred":     y_pred,
            "correct":  (y_pred == y_true).astype(int),
        }).to_csv(os.path.join(output_dir, f"{split_name}_predictions.csv"), index=False)

    # ── Ensemble manifest (which seeds + which checkpoints) ──────────────
    pd.DataFrame([
        {"seed": s, "best_val_auc": auc, "checkpoint": ckpt}
        for (s, _, auc, ckpt) in models_list
    ]).to_csv(os.path.join(output_dir, "ensemble_manifest.csv"), index=False)

    print(f"Results saved → {output_dir}")

    # ── Append to global results log ─────────────────────────────────────
    log_run(
        model_name=model_name,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        config=cfg,
        n_seeds=len(models_list),
        log_path=cfg.results_log_path,
    )
