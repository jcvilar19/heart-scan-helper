from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torchxrayvision as xrv


# ---------------------------------------------------------------------------
# RAD-DINO wrapper
# ---------------------------------------------------------------------------
class RadDinoWrapper(nn.Module):
    """microsoft/rad-dino — DINOv2 ViT-B/14 pretrained on ~1 M chest X-rays.

    Architecture matches **Model 22** (CLS token → MLP head 768→256→1).

    Wraps the HuggingFace model to expose the same ``.features`` / ``.classifier``
    contract used by every other backbone, so freeze helpers and the two-stage
    optimiser work without modification.

    Architecture
    ────────────
    .features   — the full Dinov2Model (embeddings + 12 transformer blocks + layernorm)
    .classifier — MLP head on **[CLS ∥ mean(patch tokens)]** (1536→256) → GELU →
                  Dropout(0.3) → Linear(256→1)

    Forward pass
    ────────────
    x : (B, 3, H, W) MIMIC-CXR-normalised tensor, any multiple of 14 px.
        Recommended resolution: 518 × 518 (native: 37 × 37 patches at 14 px).
    Pooling: CLS token concatenated with mean of patch tokens (excludes CLS).
    Returns (B, 1) logit tensor; ``cardio_logit`` squeezes to (B,).

    Freeze / unfreeze
    ─────────────────
    freeze_backbone()    → freezes .features; sets _backbone_frozen=True so
                           .train() keeps the backbone in eval() mode.
    partial_unfreeze(N)  → unfreeze last (12 − N) blocks + layernorm;
                           embeddings + first N blocks stay frozen.
    """

    def __init__(self) -> None:
        super().__init__()
        from transformers import AutoModel  # lazy — only loaded when this backbone is used
        dinov2 = AutoModel.from_pretrained("microsoft/rad-dino")
        self.features = dinov2
        hidden = dinov2.config.hidden_size  # 768 for ViT-B
        self._head_in = hidden * 2  # CLS + mean(patch tokens)
        self.classifier = nn.Sequential(
            nn.Linear(self._head_in, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        self._backbone_frozen: bool = False

    def train(self, mode: bool = True) -> "RadDinoWrapper":
        super().train(mode)
        # While the backbone is frozen keep it in eval() so its internal
        # Dropout / LayerScale layers don't change during head warmup.
        if mode and self._backbone_frozen:
            self.features.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(pixel_values=x)   # Dinov2ModelOutput
        h   = out.last_hidden_state           # (B, 1 + n_patches, 768)
        cls = h[:, 0]
        patch_mean = h[:, 1:].mean(dim=1)
        z   = torch.cat([cls, patch_mean], dim=-1)
        return self.classifier(z)             # (B, 1)


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------
def build_model(backbone: str | None = None) -> nn.Module:
    """Build a backbone model for Cardiomegaly classification.

    backbone options (also set via CFG.backbone):
        "densenet121"        — torchxrayvision DenseNet-121, pretrained on ~1M chest
                               X-rays; outputs raw Cardiomegaly logit via pathology index.
        "rad-dino"           — microsoft/rad-dino, DINOv2 ViT-B/14 pretrained on ~1M
                               chest X-rays (HuggingFace); 518×518 recommended input.
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

    if backbone in ("densenet121", "densenet121-res224-all"):
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        model.op_threshs = None      # raw logits at every output
        model.apply_sigmoid = False  # belt + suspenders
        return model

    if backbone == "rad-dino":
        return RadDinoWrapper()

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
        "Choose from: densenet121, rad-dino, mobilenet_v3_large, efficientnet_b0, efficientnet_b3"
    )


def cardio_logit(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward pass returning a (B,) tensor of raw logits for Cardiomegaly.

    For torchxrayvision DenseNet the logit is extracted from the pathology head.
    For all other backbones (MobileNet, EfficientNet, RadDinoWrapper) the model
    outputs (B, 1) which is squeezed to (B,).
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
    if isinstance(model, RadDinoWrapper):
        model._backbone_frozen = True
        model.features.eval()   # prevent LayerScale/Dropout updates while frozen
    return model


def unfreeze_all(model: nn.Module) -> nn.Module:
    """Unfreeze every parameter. Kept for backwards compatibility; prefer partial_unfreeze."""
    for p in model.parameters():
        p.requires_grad = True
    return model


# DenseNet-121 block groups: (block_name, transition_name) for blocks 1–4
_DENSENET_BLOCK_GROUPS = [
    ("denseblock1", "transition1"),
    ("denseblock2", "transition2"),
    ("denseblock3", "transition3"),
    ("denseblock4", "norm5"),
]


def partial_unfreeze(model: nn.Module, frozen_blocks: int = 0) -> nn.Module:
    """Selectively unfreeze the model for stage-2 fine-tuning.

    frozen_blocks — how many feature blocks to keep frozen:
        0  → unfreeze everything (same as unfreeze_all)

    DenseNet-121 (4 dense block groups):
        1  → keep denseblock1 (+transition1) frozen
        2  → keep denseblock1–2 frozen
        3  → keep denseblock1–3 frozen
        4  → keep all dense blocks frozen (only classifier trains)

    RAD-DINO / ViT-B (12 transformer blocks):
        1–12 → keep embeddings + first N transformer blocks frozen
               (last 12−N blocks + layernorm are unfrozen)
        ≥12  → keep all transformer blocks frozen (only classifier trains)

    torchvision models (MobileNet, EfficientNet):
        N    → freeze first N indexed children of model.features.
    """
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

    elif isinstance(model, RadDinoWrapper):
        # Always freeze the patch/position embeddings
        for p in model.features.embeddings.parameters():
            p.requires_grad = False
        # Freeze the first `frozen_blocks` transformer blocks
        encoder_layers = model.features.encoder.layer
        for block in encoder_layers[:frozen_blocks]:
            for p in block.parameters():
                p.requires_grad = False
        # Some blocks are now trainable — allow backbone to go back into train()
        model._backbone_frozen = False

    else:
        for module in list(model.features.children())[:frozen_blocks]:
            for p in module.parameters():
                p.requires_grad = False

    return model


def trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """List of parameters with `requires_grad=True` (for optimiser construction)."""
    return [p for p in model.parameters() if p.requires_grad]
