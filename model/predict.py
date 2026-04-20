from __future__ import annotations

import io
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
DEFAULT_BUNDLE_PATH = APP_DIR / "model3_bundle.pth"

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    heatmap_url: str | None = None


class Model3InferenceService:
    def __init__(self, bundle_path: Path):
        self.bundle_path = bundle_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bundle: dict[str, Any] | None = None
        self.model: nn.Module | None = None
        self.threshold = 0.5
        self.pathology_label = "Cardiomegaly"
        self.transforms: T.Compose | None = None
        self._load()

    def _build_model(self, dropout: float) -> nn.Module:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
        return model

    def _load(self) -> None:
        if not self.bundle_path.exists():
            raise FileNotFoundError(
                f"Model bundle not found at '{self.bundle_path}'. "
                "Export model3_bundle.pth from your notebook and copy it here."
            )

        bundle = torch.load(self.bundle_path, map_location=self.device)
        config = bundle.get("config", {})
        dropout = float(config.get("dropout", 0.30))
        img_size = int(config.get("img_size", 128))
        use_dataset_stats = bool(config.get("use_dataset_stats", False))

        mean = IMAGENET_MEAN
        std = IMAGENET_STD
        if use_dataset_stats:
            train_mean = float(bundle.get("train_gray_mean", IMAGENET_MEAN[0]))
            train_std = float(bundle.get("train_gray_std", IMAGENET_STD[0]))
            mean = [train_mean, train_mean, train_mean]
            std = [train_std, train_std, train_std]

        model = self._build_model(dropout=dropout)
        model.load_state_dict(bundle["model_state_dict"])
        model.to(self.device)
        model.eval()

        self.model = model
        self.bundle = bundle
        self.threshold = float(bundle.get("chosen_threshold", 0.5))
        self.transforms = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Lambda(lambda x: x.repeat(3, 1, 1)),
                T.Normalize(mean=mean, std=std),
            ]
        )

    @torch.inference_mode()
    def predict(self, image_bytes: bytes) -> PredictResponse:
        if self.model is None or self.transforms is None:
            raise RuntimeError("Model service is not initialized.")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("L")
        except Exception as exc:
            raise ValueError("Invalid image file.") from exc

        tensor = self.transforms(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        confidence = float(torch.sigmoid(logits).squeeze().detach().cpu().item())

        prediction = (
            self.pathology_label
            if confidence >= self.threshold
            else f"No clear {self.pathology_label} indication"
        )
        return PredictResponse(prediction=prediction, confidence=confidence, heatmap_url=None)


def create_app() -> FastAPI:
    bundle_path = Path(os.getenv("MODEL3_BUNDLE_PATH", str(DEFAULT_BUNDLE_PATH)))
    service = Model3InferenceService(bundle_path=bundle_path)

    app = FastAPI(title="Model3 Predictor", version="1.0.0")

    # Restrict in production to your frontend domain.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    async def predict(image: UploadFile = File(...)) -> PredictResponse:
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Use JPG, PNG or WebP.",
            )

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
