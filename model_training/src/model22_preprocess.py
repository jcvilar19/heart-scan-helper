"""Optional preprocessing from Model 22 (RAD-DINO notebook).

Thorax crop (ChestX-Det PSPNet), CLAHE, and light albumentations on PIL images.
Only imported when ``CFG.preprocessing_profile == \"model22\"`` and backbone is
``rad-dino`` — keeps default runs free of extra dependencies until enabled.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image

try:
    import cv2
except ImportError as e:  # pragma: no cover
    cv2 = None  # type: ignore[misc, assignment]

try:
    import albumentations as A
except ImportError as e:  # pragma: no cover
    A = None  # type: ignore[misc, assignment]

_SEG_MODEL: Optional[torch.nn.Module] = None
_BBOX_CACHE: Optional[Dict[str, Any]] = None
_BBOX_CACHE_PATH: Optional[str] = None


def _require_cv2() -> Any:
    if cv2 is None:
        raise ImportError(
            "opencv-python-headless is required for Model 22 preprocessing (CLAHE). "
            "Install with: pip install opencv-python-headless"
        )
    return cv2


def _require_albumentations() -> Any:
    if A is None:
        raise ImportError(
            "albumentations is required for Model 22 medical augmentations. "
            "Install with: pip install albumentations"
        )
    return A


def apply_clahe_pil(pil_img: Image.Image) -> Image.Image:
    """CLAHE on luminance (Model 22: clipLimit=2.0, 8×8 tiles)."""
    cv = _require_cv2()
    arr = np.array(pil_img.convert("L"), dtype=np.uint8)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return Image.fromarray(clahe.apply(arr))


_medical_aug = None


def _get_medical_aug():
    global _medical_aug
    if _medical_aug is None:
        Alb = _require_albumentations()
        _medical_aug = Alb.Compose(
            [
                Alb.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
                Alb.ElasticTransform(alpha=30, sigma=5, p=0.3),
                Alb.GaussNoise(var_limit=(5, 20), p=0.3),
                Alb.Sharpen(alpha=(0.1, 0.3), p=0.3),
            ]
        )
    return _medical_aug


def augment_medical_pil(pil_img: Image.Image) -> Image.Image:
    """Albumentations on grayscale PIL (train only)."""
    aug = _get_medical_aug()
    arr = np.array(pil_img.convert("L"))
    return Image.fromarray(aug(image=arr)["image"])


def _get_seg_model(device: torch.device) -> torch.nn.Module:
    global _SEG_MODEL
    if _SEG_MODEL is None:
        print("Loading ChestX-Det PSPNet for thorax bounding boxes (Model 22)...")
        _SEG_MODEL = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
    return _SEG_MODEL


def _load_bbox_cache(cache_path: str) -> Dict[str, Any]:
    global _BBOX_CACHE, _BBOX_CACHE_PATH
    if _BBOX_CACHE is not None and _BBOX_CACHE_PATH == cache_path:
        return _BBOX_CACHE
    _BBOX_CACHE_PATH = cache_path
    if os.path.isfile(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            _BBOX_CACHE = json.load(f)
    else:
        _BBOX_CACHE = {}
    return _BBOX_CACHE


def _save_bbox_cache(cache_path: str) -> None:
    if _BBOX_CACHE is None:
        return
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(_BBOX_CACHE, f)


def _compute_one_bbox(pil_img: Image.Image, device: torch.device, thorax_pad: float) -> List[int]:
    W, H = pil_img.size
    arr = np.array(pil_img, dtype=np.float32)
    arr = xrv.datasets.normalize(arr, 255)
    tensor = torch.from_numpy(arr[None, None, ...]).float().to(device)
    seg_model = _get_seg_model(device)
    with torch.no_grad():
        out = seg_model(tensor)
    seg = torch.sigmoid(out)[0]
    mask = (seg[[4, 5, 8]].max(0).values > 0.5).cpu().numpy()
    if not mask.any():
        return [0, 0, W, H]
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    sx, sy = W / 512.0, H / 512.0
    x0, x1 = int(round(x0 * sx)), int(round(x1 * sx))
    y0, y1 = int(round(y0 * sy)), int(round(y1 * sy))
    pad_x = int(round(thorax_pad * (x1 - x0)))
    pad_y = int(round(thorax_pad * (y1 - y0)))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(W, x1 + pad_x)
    y1 = min(H, y1 + pad_y)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return [0, 0, W, H]
    return [x0, y0, x1, y1]


def ensure_thorax_bboxes(
    image_paths: List[str],
    cache_path: str,
    device: str | torch.device,
    thorax_pad: float = 0.05,
    save_every: int = 64,
) -> None:
    """Populate JSON cache of thorax bboxes (absolute paths as keys)."""
    if not image_paths:
        return
    dev = torch.device(device) if isinstance(device, str) else device
    # PSPNet is small; CPU avoids competing with training on MPS/CUDA.
    seg_dev = torch.device("cuda") if dev.type == "cuda" else torch.device("cpu")
    cache = _load_bbox_cache(cache_path)
    todo = [p for p in image_paths if p not in cache]
    if not todo:
        print(f"Thorax bbox cache up to date ({len(cache)} entries): {cache_path}")
        return
    print(f"Segmenting {len(todo)} image(s) for thorax crop (cache: {len(cache)})...")
    for i, p in enumerate(todo, 1):
        try:
            img = Image.open(p).convert("L")
            cache[p] = _compute_one_bbox(img, seg_dev, thorax_pad)
        except Exception as e:  # noqa: BLE001
            print(f"  bbox failed for {os.path.basename(p)}: {e!r} -> full image")
            cache[p] = None
        if i % save_every == 0:
            _save_bbox_cache(cache_path)
            print(f"  flushed cache  {i}/{len(todo)}")
    global _BBOX_CACHE
    _BBOX_CACHE = cache
    _save_bbox_cache(cache_path)
    print(f"Thorax bbox cache saved ({len(cache)} paths) → {cache_path}")


def crop_thorax_pil(pil_img: Image.Image, image_path: str, cache_path: str) -> Image.Image:
    """Crop PIL using cached bbox; full image if missing or invalid."""
    cache = _load_bbox_cache(cache_path)
    bbox = cache.get(image_path)
    if bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return pil_img
    x0, y0, x1, y1 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    W, H = pil_img.size
    if x1 <= x0 or y1 <= y0 or x0 >= W or y0 >= H:
        return pil_img
    return pil_img.crop((x0, y0, x1, y1))
