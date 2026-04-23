from __future__ import annotations

from typing import List, Tuple

import torchvision.transforms as T
from PIL import Image

from src.config import CFG


# ---------------------------------------------------------------------------
# PIL helpers (TTA expects PIL → PIL transforms; xrv normalisation is applied
# downstream inside the Dataset).
# ---------------------------------------------------------------------------
def _pil_hflip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


# ---------------------------------------------------------------------------
# Training and evaluation transforms
# ---------------------------------------------------------------------------
def make_transforms(
    img_size: int | None = None,
    *,
    style: str | None = None,
    backbone: str | None = None,
) -> Tuple[T.Compose, T.Compose]:
    """Return (train_transform, eval_transform) PIL-space pipelines.

    ``style``:
        ``\"default\"`` — Resize+RandomCrop margin, light hflip (general backbones).
        ``\"model22\"`` — only used when ``backbone == \"rad-dino\"`` (Model 22:
        larger margin resize + RandomCrop, **no** horizontal flip).

    All transforms produce a PIL grayscale image of size ``(img_size, img_size)``
    (except intermediate sizes before the final crop on train).
    """
    from src.config import CFG as _CFG

    img_size = img_size if img_size is not None else _CFG.img_size
    bb = backbone or _CFG.backbone
    st = style if style is not None else getattr(_CFG, "transform_style", "default")
    use_m22 = st == "model22" and bb == "rad-dino"

    if use_m22:
        train_tf = T.Compose(
            [
                T.Resize((img_size + 32, img_size + 32)),
                T.RandomCrop((img_size, img_size)),
                T.RandomAffine(
                    degrees=8,
                    translate=(0.04, 0.04),
                    scale=(0.95, 1.05),
                    fill=0,
                ),
                T.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        )
        eval_tf = T.Compose([T.Resize((img_size, img_size))])
        return train_tf, eval_tf

    train_tf = T.Compose(
        [
            T.Resize((img_size + 16, img_size + 16)),
            T.RandomCrop((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(
                degrees=8,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
                fill=0,
            ),
            T.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    )

    eval_tf = T.Compose([T.Resize((img_size, img_size))])

    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# Test-time augmentation (TTA) transforms
# ---------------------------------------------------------------------------
def make_tta_transforms(
    img_size: int | None = None,
    *,
    style: str | None = None,
) -> List[T.Compose]:
    """Deterministic PIL-space TTA pipelines (no horizontal flip for *model22*).

    ``style``:
        ``\"default\"`` — up to 6 passes (includes one horizontal flip).
        ``\"model22\"`` — 7 passes: plain, zoom-in, zoom-out, ±6° rotation,
        brightness, contrast (matches ``Model22_improved.ipynb``).
    """
    from src.config import CFG as _CFG

    img_size = img_size if img_size is not None else _CFG.img_size
    st = style if style is not None else getattr(_CFG, "tta_style", "default")
    size = (img_size, img_size)

    if st == "model22":
        return [
            T.Compose([T.Resize(size)]),
            T.Compose([T.Resize((img_size + 20, img_size + 20)), T.CenterCrop(size)]),
            T.Compose(
                [
                    T.Resize((img_size - 20, img_size - 20)),
                    T.Pad(10, fill=0),
                    T.CenterCrop(size),
                ]
            ),
            T.Compose([T.Resize(size), T.RandomAffine(degrees=(6, 6), fill=0)]),
            T.Compose([T.Resize(size), T.RandomAffine(degrees=(-6, -6), fill=0)]),
            T.Compose([T.Resize(size), T.ColorJitter(brightness=0.1)]),
            T.Compose([T.Resize(size), T.ColorJitter(contrast=0.15)]),
        ]

    return [
        T.Compose([T.Resize(size)]),
        T.Compose([T.Resize(size), T.Lambda(_pil_hflip)]),
        T.Compose([T.Resize((img_size + 20, img_size + 20)), T.CenterCrop(size)]),
        T.Compose(
            [
                T.Resize((img_size - 20, img_size - 20)),
                T.Pad(10, fill=0),
                T.CenterCrop(size),
            ]
        ),
        T.Compose([T.Resize(size), T.RandomAffine(degrees=(6, 6), fill=0)]),
        T.Compose([T.Resize(size), T.RandomAffine(degrees=(-6, -6), fill=0)]),
    ]
