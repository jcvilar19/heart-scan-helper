"""FastAPI inference server for the Cardiomegaly classifier.

Loads the multi-seed ensemble trained in ``model_training/`` and exposes a
single ``POST /predict`` endpoint that the frontend (`src/services/predict.ts`)
already knows how to consume.

Nothing inside ``model_training/`` is modified — we only *import* the model
factory (``src.model.build_model``) to rebuild the exact architecture that was
saved to disk, then load the weights on top.

Run locally
-----------
    cd inference_server
    pip install -r requirements.txt
    uvicorn server:app --host 0.0.0.0 --port 8000

Environment overrides (optional)
--------------------------------
    MODEL_BACKBONE        default: CFG.backbone   (e.g. "efficientnet_b0")
    MODEL_IMG_SIZE        default: CFG.img_size   (e.g. 224)
    MODEL_THRESHOLD       default: 0.5            (binary cut-off for the label)
    MODEL_USE_TTA         default: "false"        ("true" → 6-pass TTA per image)
    ALLOWED_ORIGINS       comma-separated CORS origins (exact match)
    ALLOWED_ORIGIN_REGEX  regex origin whitelist (e.g. Lovable preview URLs:
                          "https://.*\\.lovable\\.app")
    LOG_LEVEL             default: INFO
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from PIL import Image

# ---------------------------------------------------------------------------
# Paths — make `from src.model import ...` resolvable without touching
# `model_training/`. We prepend the training directory to sys.path so its
# internal `from src.config import CFG` style imports keep working.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = REPO_ROOT / "model_training"
NOTEBOOKS_DIR = TRAINING_DIR / "notebooks"
RESULTS_DIR = NOTEBOOKS_DIR / "results"
HF_MODEL_REPO_ID = os.environ.get("HF_MODEL_REPO_ID", "").strip()
HF_MODEL_REVISION = os.environ.get("HF_MODEL_REVISION", "main")
HF_HUB_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
HF_MODEL_CACHE_DIR = os.environ.get("HF_MODEL_CACHE_DIR", str(REPO_ROOT / ".hf-model-cache"))

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

# Point torch's hub cache to a writable in-project location so the server
# works in sandboxed environments where ``~/.cache`` is read-only. Setting
# this BEFORE importing torchvision is critical.
os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".torch-cache"))

# `build_model` in ``model_training/src/model.py`` constructs torchvision or
# torchxrayvision backbones WITH their pretrained weights. Those weights are
# irrelevant at inference time because we immediately overwrite them with the
# trained checkpoint from ``model_training/notebooks/results/``. We monkey-
# patch the constructors so the server skips every pretrained-weight
# download. This avoids needless bandwidth AND cache-dir permission errors
# when running in sandboxed environments.
import torchvision.models as _tvm  # noqa: E402  pylint: disable=wrong-import-position
import torchxrayvision as _xrv  # noqa: E402  pylint: disable=wrong-import-position

for _fn_name in ("efficientnet_b0", "efficientnet_b3", "mobilenet_v3_large"):
    _orig = getattr(_tvm, _fn_name, None)
    if _orig is None:
        continue

    def _no_download_builder(*args, __orig=_orig, **kwargs):
        kwargs["weights"] = None
        return __orig(*args, **kwargs)

    setattr(_tvm, _fn_name, _no_download_builder)

# torchxrayvision DenseNet also attempts a download when weights="..." is set.
# We wrap its __init__ so the caller's weights argument is remembered, but
# the actual download is skipped. We still restore the canonical label list
# (``self.pathologies`` / ``self.targets``) that downstream code in
# ``model_training/src/model.py::cardio_logit`` relies on to locate the
# Cardiomegaly output index.
_orig_xrv_densenet_init = _xrv.models.DenseNet.__init__


def _xrv_densenet_init_no_download(self, *args, **kwargs):
    requested_weights = kwargs.get("weights")
    kwargs["weights"] = None
    _orig_xrv_densenet_init(self, *args, **kwargs)
    if requested_weights and requested_weights in _xrv.models.model_urls:
        labels = _xrv.models.model_urls[requested_weights]["labels"]
        self.targets = labels
        self.pathologies = labels


_xrv.models.DenseNet.__init__ = _xrv_densenet_init_no_download

from src.config import CFG  # noqa: E402  pylint: disable=wrong-import-position
from src.model import build_model, cardio_logit  # noqa: E402  pylint: disable=wrong-import-position
from src.dataset import get_normalize_fn  # noqa: E402  pylint: disable=wrong-import-position


def _detect_backbone_from_checkpoint(ckpt_path: Path) -> str:
    """Inspect a saved state_dict and guess which backbone produced it.

    Rules:
      * torchxrayvision DenseNet-121  → has ``features.denseblockN.*`` keys
      * torchvision EfficientNet      → top-level ``features.0.0.weight`` (stem conv)
                                        and depth ≥ 9 feature groups
      * torchvision MobileNetV3-Large → ``features.0.0.weight`` with depth ~17
      * microsoft/rad-dino            → keys under ``features.embeddings`` /
                                        ``features.encoder.layer.``
    Defaults to ``CFG.backbone`` if no signature matches.
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    keys = list(state.keys())

    if any("denseblock" in k for k in keys):
        return "densenet121"
    if any(k.startswith("features.embeddings.") for k in keys) or any(
        k.startswith("features.encoder.layer.") for k in keys
    ):
        return "rad-dino"
    # torchvision feature indices
    feature_indices = {
        int(k.split(".")[1])
        for k in keys
        if k.startswith("features.") and k.split(".")[1].isdigit()
    }
    if feature_indices:
        # EfficientNet-B0 has 9 groups (features.0 … features.8)
        # MobileNetV3-Large has 17 groups (features.0 … features.16)
        if max(feature_indices) >= 12:
            return "mobilenet_v3_large"
        if max(feature_indices) >= 7:
            return "efficientnet_b0"
    return CFG.backbone


def _hf_download(filename: str) -> Path:
    """Download a file from HF model repo and return local cached path."""
    if not HF_MODEL_REPO_ID:
        raise FileNotFoundError(
            f"File {filename!r} not found locally and HF_MODEL_REPO_ID is not set."
        )
    path = hf_hub_download(
        repo_id=HF_MODEL_REPO_ID,
        filename=filename,
        revision=HF_MODEL_REVISION,
        token=HF_HUB_TOKEN,
        cache_dir=HF_MODEL_CACHE_DIR,
    )
    return Path(path)


def _resolve_manifest_path() -> Path:
    """Find `ensemble_manifest.csv` locally first, else download from HF."""
    local = RESULTS_DIR / "ensemble_manifest.csv"
    if local.exists():
        return local
    log.info(
        "Local ensemble_manifest.csv not found under %s; downloading from HF repo %s",
        RESULTS_DIR,
        HF_MODEL_REPO_ID or "<unset>",
    )
    return _hf_download("ensemble_manifest.csv")


def _resolve_optional_support_file(name: str) -> Path | None:
    """Find optional support file locally; if missing try HF model repo."""
    local = RESULTS_DIR / name
    if local.exists():
        return local
    try:
        return _hf_download(name)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Backbone + image size: auto-detected from the checkpoint so the server never
# runs with a mismatched architecture. Can still be forced via env vars.
# ---------------------------------------------------------------------------
def _first_checkpoint_path() -> Path:
    try:
        manifest = _resolve_manifest_path()
        df = pd.read_csv(manifest)
        first = df["checkpoint"].iloc[0]
        p = Path(first)
        if p.is_absolute() and p.exists():
            return p
        # Local resolution first.
        for candidate in (NOTEBOOKS_DIR / first, RESULTS_DIR / Path(first).name):
            if candidate.exists():
                return candidate
        # `_first_checkpoint_path` is executed during module import (before
        # `_resolve_checkpoint` is defined), so we do HF download inline here.
        for name in (first, Path(first).name):
            try:
                return _hf_download(name)
            except Exception:  # noqa: BLE001
                continue
        raise FileNotFoundError(f"Could not resolve first checkpoint from manifest entry: {first!r}")
    except Exception:  # noqa: BLE001
        pass
    fallback = RESULTS_DIR / "best_model.pth"
    if fallback.exists():
        return fallback
    try:
        return _hf_download("best_model.pth")
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            "No checkpoints found locally and could not download from HF. "
            "Set HF_MODEL_REPO_ID and upload ensemble_manifest.csv + *.pth."
        ) from exc


_DETECTED_BACKBONE = _detect_backbone_from_checkpoint(_first_checkpoint_path())
# DenseNet-121 (torchxrayvision) is trained on 224x224; ViT-B/14 needs 518.
_DEFAULT_IMG_SIZE = 518 if _DETECTED_BACKBONE == "rad-dino" else 224

BACKBONE: str = os.environ.get("MODEL_BACKBONE", _DETECTED_BACKBONE)
IMG_SIZE: int = int(os.environ.get("MODEL_IMG_SIZE", str(_DEFAULT_IMG_SIZE)))
USE_TTA: bool = os.environ.get("MODEL_USE_TTA", "true").lower() in {"1", "true", "yes"}


def _default_threshold() -> float:
    """Use the training-selected threshold when available."""
    metrics_path = _resolve_optional_support_file("val_metrics_final.json")
    if metrics_path is not None:
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            thr = float(data.get("threshold", 0.5))
            if 0.0 <= thr <= 1.0:
                return thr
        except Exception:  # noqa: BLE001
            pass
    return 0.5


DECISION_THRESHOLD: float = float(os.environ.get("MODEL_THRESHOLD", str(_default_threshold())))

_DEFAULT_ORIGINS = (
    "http://localhost:3000,"
    "http://localhost:5173,"
    "http://localhost:8080,"
    "http://127.0.0.1:3000,"
    "http://127.0.0.1:5173,"
    "http://127.0.0.1:8080"
)
ALLOWED_ORIGINS: list[str] = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if o.strip()
]
# Optional regex list — useful when the production frontend is served from a
# hash-based preview URL (e.g. Lovable / Vercel preview deployments).
# By default we allow:
#   * any *.lovable.app and *.lovableproject.com subdomain (deployed Lovable apps)
#   * any *.ngrok-free.app / *.ngrok.app / *.ngrok.io subdomain (when the user
#     forwards the dev server through ngrok and previews the app from anywhere)
# Override with `ALLOWED_ORIGIN_REGEX` to lock things down in production.
# Include common private LAN dev URLs (Vite "Network" URL is often
# `http://192.168.x.x:8080` — the Origin header is not localhost, so
# it must be accepted here or the browser will block with "Network Error").
_DEFAULT_ORIGIN_REGEX = (
    r"https://([a-z0-9-]+\.)*lovable\.app"
    r"|https://([a-z0-9-]+\.)*lovableproject\.com"
    r"|https://([a-z0-9-]+\.)*ngrok-free\.app"
    r"|https://([a-z0-9-]+\.)*ngrok\.app"
    r"|https://([a-z0-9-]+\.)*ngrok\.io"
    r"|http://(192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}):\d+"
)
_ORIGIN_REGEX: str | None = os.environ.get("ALLOWED_ORIGIN_REGEX", _DEFAULT_ORIGIN_REGEX) or None

DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

POSITIVE_LABEL = "Cardiomegaly"
NEGATIVE_LABEL = "No Cardiomegaly indication"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-5s  %(message)s",
)
log = logging.getLogger("inference")

# ---------------------------------------------------------------------------
# Preprocessing — delegate to the SAME normalization functions the training
# dataset uses (`xrv_normalize_np` for densenet121, `imagenet_normalize_np`
# for every other backbone). This guarantees byte-for-byte identical
# preprocessing between training and inference.
# ---------------------------------------------------------------------------
_normalize_fn = get_normalize_fn(BACKBONE)


def _pil_hflip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def _tta_pipelines(size: int) -> List[T.Compose]:
    """Match `src.transforms.make_tta_transforms` (6 deterministic passes)."""
    s = (size, size)
    return [
        T.Compose([T.Resize(s)]),
        T.Compose([T.Resize(s), T.Lambda(_pil_hflip)]),
        T.Compose([T.Resize((size + 20, size + 20)), T.CenterCrop(s)]),
        T.Compose([T.Resize((size - 20, size - 20)), T.Pad(10, fill=0), T.CenterCrop(s)]),
        T.Compose([T.Resize(s), T.RandomAffine(degrees=(6, 6), fill=0)]),
        T.Compose([T.Resize(s), T.RandomAffine(degrees=(-6, -6), fill=0)]),
    ]


def _single_eval_pipeline(size: int) -> T.Compose:
    return T.Compose([T.Resize((size, size))])


# ---------------------------------------------------------------------------
# Ensemble loading
# ---------------------------------------------------------------------------
def _resolve_checkpoint(p: str) -> Path:
    """Resolve checkpoint locally first, else download from HF model repo."""
    path = Path(p)
    if path.is_absolute() and path.exists():
        return path
    for candidate in (NOTEBOOKS_DIR / p, RESULTS_DIR / Path(p).name):
        if candidate.exists():
            return candidate
    # In model repos we usually store files flat, so try both raw entry and basename.
    tried = [p]
    if Path(p).name != p:
        tried.append(Path(p).name)
    for name in tried:
        try:
            downloaded = _hf_download(name)
            log.info("  → downloaded %s from HF repo %s", name, HF_MODEL_REPO_ID)
            return downloaded
        except Exception:  # noqa: BLE001
            continue
    raise FileNotFoundError(
        f"Checkpoint not found locally and not downloadable from HF repo: {p!r}"
    )


def _load_ensemble() -> List[nn.Module]:
    # Align CFG so build_model() reads the right backbone/size internally.
    CFG.backbone = BACKBONE
    CFG.img_size = IMG_SIZE

    try:
        manifest = _resolve_manifest_path()
        df = pd.read_csv(manifest)
        checkpoint_paths = [_resolve_checkpoint(p) for p in df["checkpoint"].tolist()]
        log.info(
            "Loading ensemble of %d models from %s",
            len(checkpoint_paths),
            manifest,
        )
    except Exception:
        best = RESULTS_DIR / "best_model.pth"
        if best.exists():
            checkpoint_paths = [best]
            log.info("No manifest found, falling back to local checkpoint: %s", best.name)
        else:
            checkpoint_paths = [_resolve_checkpoint("best_model.pth")]
            log.info("No manifest found, falling back to HF checkpoint: best_model.pth")

    models: list[nn.Module] = []
    for ckpt_path in checkpoint_paths:
        log.info("  → loading %s (%s)", ckpt_path.name, ckpt_path.resolve())
        model = build_model(BACKBONE)
        state = torch.load(ckpt_path, map_location=DEVICE)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint architecture mismatch. "
                f"backbone={BACKBONE!r}, checkpoint={ckpt_path.name!r}, "
                f"missing_keys={len(missing)}, unexpected_keys={len(unexpected)}. "
                "Use the correct MODEL_BACKBONE / MODEL_IMG_SIZE and ensure "
                "ensemble_manifest.csv points to checkpoints from that training run."
            )
        model.to(DEVICE).eval()
        models.append(model)

    log.info(
        "Ensemble ready — %d model(s) · device=%s · backbone=%s (detected=%s) · "
        "normalize=%s · img_size=%d · tta=%s · threshold=%.4f",
        len(models), DEVICE, BACKBONE, _DETECTED_BACKBONE,
        _normalize_fn.__name__, IMG_SIZE, USE_TTA, DECISION_THRESHOLD,
    )
    return models


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="CardioScan inference", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ensemble: list[nn.Module] = []
_loaded_checkpoints: list[str] = []


@app.on_event("startup")
def _startup() -> None:
    global _ensemble, _loaded_checkpoints
    try:
        manifest = _resolve_manifest_path()
        df = pd.read_csv(manifest)
        _loaded_checkpoints = [Path(p).name for p in df["checkpoint"].tolist()]
    except Exception:
        _loaded_checkpoints = ["best_model.pth"]
    _ensemble = _load_ensemble()


@app.get("/health")
def health() -> dict:
    return {
        "ok": bool(_ensemble),
        "models": len(_ensemble),
        "checkpoints": _loaded_checkpoints,
        "backbone": BACKBONE,
        "detected_backbone": _DETECTED_BACKBONE,
        "normalization": _normalize_fn.__name__,
        "img_size": IMG_SIZE,
        "device": str(DEVICE),
        "use_tta": USE_TTA,
        "threshold": DECISION_THRESHOLD,
    }


@torch.no_grad()
def _predict_probability_detailed(
    pil_gray: Image.Image,
    use_tta: bool,
    max_models: int | None = None,
) -> dict:
    """Run ensemble (+ optional TTA) on a single PIL image.

    Returns a dict with per-model / per-TTA logits for transparency.
    Matches `tta_predict` / `tta_predict_ensemble` in ``src.train`` exactly:
    average logits across TTA (per model), then average across models,
    then sigmoid.
    """
    pipelines = _tta_pipelines(IMG_SIZE) if use_tta else [_single_eval_pipeline(IMG_SIZE)]

    tensors = [_normalize_fn(pipeline(pil_gray)) for pipeline in pipelines]
    batch = torch.stack(tensors, dim=0).to(DEVICE)  # (num_tta, 3, H, W)

    active_model_count = len(_ensemble) if max_models is None else max(1, min(max_models, len(_ensemble)))
    active_models = _ensemble[:active_model_count]
    active_checkpoints = _loaded_checkpoints[:active_model_count]

    per_model_tta_logits: list[np.ndarray] = []
    per_model_mean_logit: list[float] = []
    for model in active_models:
        logit_vec = cardio_logit(model, batch).float().cpu().numpy()  # (num_tta,)
        per_model_tta_logits.append(logit_vec)
        per_model_mean_logit.append(float(np.mean(logit_vec)))

    ensemble_mean_logit = float(np.mean(per_model_mean_logit))
    probability = float(1.0 / (1.0 + np.exp(-ensemble_mean_logit)))

    return {
        "probability": probability,
        "ensemble_mean_logit": ensemble_mean_logit,
        "per_model_mean_logit": {
            name: lg for name, lg in zip(active_checkpoints, per_model_mean_logit)
        },
        "per_model_tta_logits": {
            name: lg.tolist() for name, lg in zip(active_checkpoints, per_model_tta_logits)
        },
        "num_tta_passes": batch.shape[0],
        "models_used": active_model_count,
        "checkpoints_used": active_checkpoints,
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    use_tta: bool | None = Query(default=None, description="Override TTA for this request."),
    max_models: int | None = Query(default=None, ge=1, description="Use only first N models for speed."),
) -> dict:
    if not _ensemble:
        raise HTTPException(status_code=503, detail="Model not ready")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        pil = Image.open(io.BytesIO(raw)).convert("L")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc

    effective_use_tta = USE_TTA if use_tta is None else use_tta

    try:
        details = _predict_probability_detailed(
            pil,
            use_tta=effective_use_tta,
            max_models=max_models,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    probability = details["probability"]
    is_positive = probability >= DECISION_THRESHOLD

    log.info(
        "/predict  file=%s  size=%d  prob=%.4f  thr=%.4f  -> %s  (per-model=%s, tta=%d)",
        image.filename,
        len(raw),
        probability,
        DECISION_THRESHOLD,
        "Cardiomegaly" if is_positive else "Negative",
        {k: round(v, 4) for k, v in details["per_model_mean_logit"].items()},
        details["num_tta_passes"],
    )

    return {
        "prediction": POSITIVE_LABEL if is_positive else NEGATIVE_LABEL,
        "prediction_binary": 1 if is_positive else 0,
        "confidence": probability,
        "heatmap_url": None,
        "source": "model",
        "threshold": DECISION_THRESHOLD,
        "ensemble_size": details["models_used"],
        "use_tta": effective_use_tta,
        "checkpoints": details["checkpoints_used"],
    }


@app.post("/debug/predict")
async def debug_predict(image: UploadFile = File(...)) -> dict:
    """Same as /predict but returns per-model and per-TTA raw logits for
    verification against the training notebook's val/test CSVs."""
    if not _ensemble:
        raise HTTPException(status_code=503, detail="Model not ready")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        pil = Image.open(io.BytesIO(raw)).convert("L")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc

    details = _predict_probability_detailed(pil, use_tta=USE_TTA)
    details["prediction"] = (
        POSITIVE_LABEL if details["probability"] >= DECISION_THRESHOLD else NEGATIVE_LABEL
    )
    details["prediction_binary"] = 1 if details["probability"] >= DECISION_THRESHOLD else 0
    details["threshold"] = DECISION_THRESHOLD
    details["use_tta"] = USE_TTA
    details["checkpoints"] = _loaded_checkpoints
    return details
