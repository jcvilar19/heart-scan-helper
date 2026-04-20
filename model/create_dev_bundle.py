"""
Create model/model7_bundle.pth for local development.

This is NOT your trained Model7 from the notebook. It uses ImageNet-pretrained
MobileNetV2 features with a freshly initialized classifier head so predict.py
and uvicorn can start without you copying weights yet.

Replace this file with your real export from Model7.ipynb when available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

APP_DIR = Path(__file__).resolve().parent
OUT = APP_DIR / "model7_bundle.pth"


def build_model(dropout: float = 0.2) -> nn.Module:
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(1280, 1),
    )
    return model


def main() -> None:
    model = build_model(0.2)
    bundle = {
        "model_state_dict": model.state_dict(),
        "config": {
            "img_size": 224,
            "dropout": 0.2,
            "use_dataset_stats": True,
            "seed": 42,
        },
        "chosen_threshold": 0.5,
        "train_gray_mean": 0.5,
        "train_gray_std": 0.25,
        "use_inference_tta": False,
    }
    torch.save(bundle, OUT)
    print(f"Wrote dev placeholder bundle: {OUT}")
    print("Replace with your trained model7_bundle.pth from Model7.ipynb for real predictions.")


if __name__ == "__main__":
    sys.exit(main() or 0)
