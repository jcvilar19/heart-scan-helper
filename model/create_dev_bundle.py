"""
Create model/best_model.pth for local smoke tests (Model7 notebook format).

Saves only state_dict (same as Model7.ipynb after training). Pair with the
committed model/model7_meta.json or edit meta after your own Colab run.

This is NOT your real trained Model7 unless you replace weights from Colab.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

APP_DIR = Path(__file__).resolve().parent
OUT = APP_DIR / "best_model.pth"


def build_model(dropout: float = 0.2) -> nn.Module:
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(1280, 1),
    )
    return model


def main() -> None:
    model = build_model(0.2)
    torch.save(model.state_dict(), OUT)
    print(f"Wrote dev weights (state_dict only): {OUT}")
    print("Uses model/model7_meta.json for preprocessing + threshold.")
    print("Replace OUT with your Colab best_model.pth for real Model7 predictions.")


if __name__ == "__main__":
    sys.exit(main() or 0)
