from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torchxrayvision as xrv


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------
def build_model(backbone: str | None = None) -> nn.Module:
    """Build a backbone model for Cardiomegaly classification.

    backbone options (also set via CFG.backbone):
        "densenet121"        — torchxrayvision DenseNet-121, pretrained on ~1M chest
                               X-rays; outputs raw Cardiomegaly logit via pathology index.
        "mobilenet_v3_large" — torchvision MobileNetV3-Large (ImageNet); final linear
                               replaced with a single-output head.
        "efficientnet_b0"    — torchvision EfficientNet-B0  (ImageNet); same replacement.
        "efficientnet_b3"    — torchvision EfficientNet-B3  (ImageNet); same replacement.

    All returned models expose .features and .classifier so that freeze_backbone()
    and the two-stage optimizer in train_one_seed() work unchanged.
    Input tensor format differs by backbone — use dataset.get_normalize_fn(backbone).
    """
    from src.config import CFG  # lazy to avoid circular import at module load
    backbone = backbone or CFG.backbone

    if backbone == "densenet121":
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        model.op_threshs = None      # raw logits at every output
        model.apply_sigmoid = False  # belt + suspenders
        return model

    import torchvision.models as tvm

    if backbone == "mobilenet_v3_large":
        model = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)
        return model

    if backbone in ("efficientnet_b0", "efficientnet_b3"):
        if backbone == "efficientnet_b0":
            model = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            model = tvm.efficientnet_b3(weights=tvm.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)
        return model

    raise ValueError(
        f"Unknown backbone: {backbone!r}. "
        "Choose from: densenet121, mobilenet_v3_large, efficientnet_b0, efficientnet_b3"
    )


def cardio_logit(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward pass returning a (B,) tensor of raw logits for Cardiomegaly.

    For torchxrayvision DenseNet the logit is extracted from the pathology head.
    For torchvision backbones (MobileNet, EfficientNet) the model directly outputs
    a (B, 1) tensor which is squeezed to (B,).
    """
    if isinstance(model, xrv.models.DenseNet):
        out = model(x)                                       # (B, num_pathologies)
        idx = model.pathologies.index("Cardiomegaly")
        return out[:, idx]
    return model(x).squeeze(1)                              # (B, 1) → (B,)


# ---------------------------------------------------------------------------
# Backbone management helpers
# ---------------------------------------------------------------------------
def freeze_backbone(model: nn.Module) -> nn.Module:
    """Freeze all params in .features; keep .classifier trainable."""
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    return model


# DenseNet-121 block groups (name in model.features → logical block index 1–4)
_DENSENET_BLOCK_GROUPS = [
    ("denseblock1", "transition1"),   # block 1
    ("denseblock2", "transition2"),   # block 2
    ("denseblock3", "transition3"),   # block 3
    ("denseblock4", "norm5"),         # block 4
]


def unfreeze_all(model: nn.Module) -> nn.Module:
    """Unfreeze every parameter. Kept for backwards compatibility; prefer partial_unfreeze."""
    for p in model.parameters():
        p.requires_grad = True
    return model


def partial_unfreeze(model: nn.Module, frozen_blocks: int = 0) -> nn.Module:
    """Selectively unfreeze the model for stage-2 fine-tuning.

    frozen_blocks — how many feature blocks to keep frozen:
        0  → unfreeze everything (same as unfreeze_all)
        1  → keep denseblock1 (+transition1) frozen
        2  → keep denseblock1–2 frozen
        3  → keep denseblock1–3 frozen
        4  → keep all dense blocks frozen (only classifier trains)

    For torchvision models (MobileNet, EfficientNet) the index refers to the
    numbered child modules of model.features (typically 16–18 entries).
    """
    # Start from fully unfrozen, then re-freeze the requested blocks.
    for p in model.parameters():
        p.requires_grad = True

    if frozen_blocks <= 0:
        return model

    if isinstance(model, xrv.models.DenseNet):
        frozen_names: set[str] = set()
        for i in range(min(frozen_blocks, len(_DENSENET_BLOCK_GROUPS))):
            frozen_names.update(_DENSENET_BLOCK_GROUPS[i])
        for name, module in model.features.named_children():
            if name in frozen_names:
                for p in module.parameters():
                    p.requires_grad = False
    else:
        # torchvision Sequential: freeze the first `frozen_blocks` sub-modules
        children = list(model.features.children())
        for module in children[:frozen_blocks]:
            for p in module.parameters():
                p.requires_grad = False

    return model


def trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """List of parameters with `requires_grad=True` (for optimiser construction)."""
    return [p for p in model.parameters() if p.requires_grad]
