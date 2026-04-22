from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torchxrayvision as xrv


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------
def build_model(weights: str = "densenet121-res224-all") -> nn.Module:
    """Load a torchxrayvision DenseNet-121 pretrained on ~1M chest X-rays.

    The model is configured to output **raw logits** at every pathology head
    (op_threshs disabled, sigmoid disabled). Use `cardio_logit()` to extract
    the Cardiomegaly logit from the multi-pathology output.
    """
    model = xrv.models.DenseNet(weights=weights)
    model.op_threshs = None          # raw logits at every output
    model.apply_sigmoid = False      # belt + suspenders
    return model


def cardio_logit(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward pass returning a (B,) tensor of raw logits for Cardiomegaly."""
    out = model(x)                                       # (B, num_pathologies)
    idx = model.pathologies.index("Cardiomegaly")
    return out[:, idx]


# ---------------------------------------------------------------------------
# Backbone management helpers
# ---------------------------------------------------------------------------
def freeze_backbone(model: nn.Module) -> nn.Module:
    """Freeze all conv + BN params in `features`; keep the classifier trainable."""
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    return model


def unfreeze_all(model: nn.Module) -> nn.Module:
    """Unfreeze every parameter (used in stage 2)."""
    for p in model.parameters():
        p.requires_grad = True
    return model


def trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """List of parameters with `requires_grad=True` (for optimiser construction)."""
    return [p for p in model.parameters() if p.requires_grad]
