from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

APP_DIR = Path(__file__).resolve().parent

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    heatmap_url: str | None = None


def _looks_like_nn_state_dict(obj: dict[str, Any]) -> bool:
    if not obj:
        return False
    if "model_state_dict" in obj or "config" in obj:
        return False
    keys = list(obj.keys())[:5]
    return all(isinstance(k, str) and ("features." in k or "classifier." in k) for k in keys)


def _build_mobilenet_head(dropout: float) -> nn.Module:
    """Same head as Model3 / Model7 notebooks: Dropout + Linear(1280, 1)."""
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(1280, 1),
    )
    return model


def _resolve_checkpoint_path() -> Path:
    for key in ("MODEL_CHECKPOINT", "MODEL_BUNDLE_PATH", "MODEL3_BUNDLE_PATH", "MODEL7_BUNDLE_PATH"):
        raw = os.getenv(key)
        if raw:
            p = Path(raw)
            if p.exists():
                return p
            raise FileNotFoundError(f"Checkpoint path from {key} does not exist: {p}")

    # Model7.ipynb saves `best_model.pth` (state_dict only). `model7_meta.json` must live in model/.
    for candidate in (
        APP_DIR / "best_model.pth",
        APP_DIR / "model7_bundle.pth",
        APP_DIR / "model3_bundle.pth",
    ):
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No model weights found. For Model7 (notebook on Desktop → copied to model/Model7.ipynb):\n"
        "  1) After training in Colab, copy best_model.pth into model/best_model.pth\n"
        "  2) Keep model/model7_meta.json (stats + threshold from your run; edit if you retrain)\n"
        "Optional: model/model7_bundle.pth (full torch.save dict) or model/model3_bundle.pth\n"
        "Or set MODEL_CHECKPOINT to a .pth file."
    )


def _load_meta_json(weights_path: Path) -> dict[str, Any]:
    env_meta = os.getenv("MODEL_META_PATH")
    candidates = []
    if env_meta:
        candidates.append(Path(env_meta))
    candidates.append(weights_path.with_name(weights_path.stem + "_meta.json"))
    candidates.append(APP_DIR / "model7_meta.json")

    for path in candidates:
        if path.exists():
            with path.open(encoding="utf-8") as f:
                return json.load(f)

    raise FileNotFoundError(
        f"Raw weights '{weights_path.name}' require a sidecar meta JSON with preprocessing "
        f"and threshold. Tried: {[str(p) for p in candidates]}. "
        "See model/model7_meta.example.json or export a full bundle from the notebook."
    )


class CardiomegalyInferenceService:
    """
    Loads Model7-style bundles, Model3-style bundles, or raw state_dict + meta JSON.
    """

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = _resolve_checkpoint_path()
        self.model_version = self._infer_version_label(self.checkpoint_path)

        try:
            payload = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=False
            )
        except TypeError:
            payload = torch.load(self.checkpoint_path, map_location=self.device)

        if isinstance(payload, dict) and "model_state_dict" in payload:
            self._init_from_bundle(payload)
        elif isinstance(payload, dict) and _looks_like_nn_state_dict(payload):
            meta = _load_meta_json(self.checkpoint_path)
            self._init_from_state_dict(payload, meta)
        else:
            raise ValueError(
                f"Unrecognized checkpoint format: {self.checkpoint_path}. "
                "Expected a dict with 'model_state_dict' or a raw nn.Module state_dict."
            )

        self.pathology_label = "Cardiomegaly"
        self.model.eval()

    @staticmethod
    def _infer_version_label(path: Path) -> str:
        name = path.name.lower()
        if "model7" in name or name == "best_model.pth":
            return "model7"
        if "model3" in name:
            return "model3"
        return "custom"

    def _init_from_bundle(self, bundle: dict[str, Any]) -> None:
        config = bundle.get("config", {})
        if isinstance(config, dict):
            dropout = float(config.get("dropout", 0.20))
            img_size = int(config.get("img_size", 224))
            use_dataset_stats = bool(config.get("use_dataset_stats", True))
        else:
            dropout, img_size, use_dataset_stats = 0.20, 224, True

        mean, std = IMAGENET_MEAN, IMAGENET_STD
        if use_dataset_stats:
            mean_f = float(bundle.get("train_gray_mean", IMAGENET_MEAN[0]))
            std_f = float(bundle.get("train_gray_std", IMAGENET_STD[0]))
            mean = [mean_f, mean_f, mean_f]
            std = [std_f, std_f, std_f]

        self.threshold = float(bundle.get("chosen_threshold", 0.5))
        self.use_tta = bool(bundle.get("use_inference_tta", False))

        self.model = _build_mobilenet_head(dropout=dropout)
        self.model.load_state_dict(bundle["model_state_dict"])
        self.model.to(self.device)

        self.eval_transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Lambda(lambda x: x.repeat(3, 1, 1)),
                T.Normalize(mean=mean, std=std),
            ]
        )
        self.tta_transforms = self._make_tta_transforms(img_size, mean, std) if self.use_tta else []

    def _init_from_state_dict(self, state: dict[str, Any], meta: dict[str, Any]) -> None:
        dropout = float(meta.get("dropout", 0.20))
        img_size = int(meta.get("img_size", 224))
        use_dataset_stats = bool(meta.get("use_dataset_stats", True))
        self.threshold = float(meta.get("chosen_threshold", meta.get("threshold", 0.5)))
        self.use_tta = bool(meta.get("use_inference_tta", os.getenv("MODEL_INFERENCE_TTA", "0") == "1"))

        mean, std = IMAGENET_MEAN, IMAGENET_STD
        if use_dataset_stats:
            mean_f = float(meta["train_gray_mean"])
            std_f = float(meta["train_gray_std"])
            mean = [mean_f, mean_f, mean_f]
            std = [std_f, std_f, std_f]

        self.model = _build_mobilenet_head(dropout=dropout)
        self.model.load_state_dict(state)
        self.model.to(self.device)

        self.eval_transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Lambda(lambda x: x.repeat(3, 1, 1)),
                T.Normalize(mean=mean, std=std),
            ]
        )
        self.tta_transforms = self._make_tta_transforms(img_size, mean, std) if self.use_tta else []

    @staticmethod
    def _make_tta_transforms(img_size: int, mean: list[float], std: list[float]) -> list[T.Compose]:
        """Matches Model7.ipynb `make_tta_transforms` (3 views)."""
        return [
            T.Compose(
                [
                    T.Resize((img_size, img_size)),
                    T.ToTensor(),
                    T.Lambda(lambda x: x.repeat(3, 1, 1)),
                    T.Normalize(mean=mean, std=std),
                ]
            ),
            T.Compose(
                [
                    T.Resize((img_size + 12, img_size + 12)),
                    T.CenterCrop((img_size, img_size)),
                    T.ToTensor(),
                    T.Lambda(lambda x: x.repeat(3, 1, 1)),
                    T.Normalize(mean=mean, std=std),
                ]
            ),
            T.Compose(
                [
                    T.Resize((img_size + 20, img_size + 20)),
                    T.CenterCrop((img_size, img_size)),
                    T.ToTensor(),
                    T.Lambda(lambda x: x.repeat(3, 1, 1)),
                    T.Normalize(mean=mean, std=std),
                ]
            ),
        ]

    @torch.inference_mode()
    def predict(self, image_bytes: bytes) -> PredictResponse:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("L")
        except Exception as exc:
            raise ValueError("Invalid image file.") from exc

        transforms = self.tta_transforms if self.tta_transforms else [self.eval_transform]
        probs: list[float] = []
        for tf in transforms:
            tensor = tf(image).unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            p = float(torch.sigmoid(logits).squeeze().detach().cpu().item())
            probs.append(p)

        confidence = float(sum(probs) / len(probs))
        prediction = (
            self.pathology_label
            if confidence >= self.threshold
            else f"No clear {self.pathology_label} indication"
        )
        return PredictResponse(prediction=prediction, confidence=confidence, heatmap_url=None)


_inference_svc: CardiomegalyInferenceService | None = None
_inference_err: str | None = None


def _get_inference_service() -> CardiomegalyInferenceService:
    global _inference_svc, _inference_err
    if _inference_err is not None:
        raise HTTPException(status_code=503, detail=_inference_err)
    if _inference_svc is not None:
        return _inference_svc
    try:
        _inference_svc = CardiomegalyInferenceService()
    except FileNotFoundError as exc:
        _inference_err = str(exc)
        raise HTTPException(status_code=503, detail=_inference_err) from exc
    except Exception as exc:
        _inference_err = f"Inference init failed: {exc}"
        raise HTTPException(status_code=503, detail=_inference_err) from exc
    return _inference_svc


def create_app() -> FastAPI:
    app = FastAPI(
        title="Cardiomegaly predictor",
        version="2.1.0",
        description="Model7 (best_model.pth + model7_meta.json) or bundle checkpoints for med-image-clarity.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        try:
            s = _get_inference_service()
        except HTTPException as exc:
            return {"status": "no_weights", "detail": str(exc.detail)}
        return {
            "status": "ok",
            "checkpoint": s.checkpoint_path.name,
            "model_version": s.model_version,
            "tta": "on" if s.use_tta else "off",
        }

    @app.post("/predict", response_model=PredictResponse)
    async def predict(image: UploadFile = File(...)) -> PredictResponse:
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Use JPG, PNG or WebP.",
            )

        try:
            service = _get_inference_service()
        except HTTPException:
            raise

        try:
            image_bytes = await image.read()
            if not image_bytes:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            return service.predict(image_bytes=image_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return app


app = create_app()
