from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


class CardiomegalyModel(nn.Module):
    """EfficientNet-B3 backbone with late fusion of clinical tabular features.

    Architecture
    ────────────
    Image  → EfficientNet-B3 features → AdaptiveAvgPool → Flatten → 1536-dim
    Tabular [age_norm, sex_bin] → Linear(2, 16) → ReLU → 16-dim
    Concat(1536 + 16 = 1552) → Dropout → Linear(1552, 1)

    The tabular branch gives the model access to patient age and sex, which
    are clinically relevant for cardiomegaly detection.
    """

    def __init__(self, dropout: float = 0.20) -> None:
        super().__init__()
        base = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.features  = base.features    # 9 groups: features[0]..features[8]
        self.avgpool   = base.avgpool     # AdaptiveAvgPool2d(output_size=1)
        self.tab_embed = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1536 + 16, 1),
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        x = self.features(image)          # (B, 1536, H', W')
        x = self.avgpool(x)               # (B, 1536, 1, 1)
        x = torch.flatten(x, 1)           # (B, 1536)
        t = self.tab_embed(tabular)        # (B, 16)
        x = torch.cat([x, t], dim=1)      # (B, 1552)
        return self.classifier(x)         # (B, 1)


# ---------------------------------------------------------------------------
# Factory + backbone management helpers
# ---------------------------------------------------------------------------

def build_model(dropout: float = 0.20) -> CardiomegalyModel:
    """Instantiate a fresh CardiomegalyModel with ImageNet-pretrained weights."""
    return CardiomegalyModel(dropout)


def freeze_backbone(model: CardiomegalyModel) -> CardiomegalyModel:
    """Freeze the EfficientNet feature extractor; leave head + tabular branch trainable."""
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.tab_embed.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    return model


def unfreeze_last_blocks(model: CardiomegalyModel, n_blocks: int = 7) -> CardiomegalyModel:
    """Unfreeze the last *n_blocks* of the EfficientNet-B3 feature extractor.

    EfficientNet-B3 has 9 feature groups (features[0]..features[8]).
    Setting n_blocks=7 keeps only the stem (features[0]) frozen.
    """
    total = len(model.features)
    start = max(0, total - n_blocks)
    for i, block in enumerate(model.features):
        for p in block.parameters():
            p.requires_grad = (i >= start)
    for p in model.tab_embed.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    return model
