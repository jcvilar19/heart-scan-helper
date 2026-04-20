# Cardiomegaly inference (Model7 + Model3)

FastAPI service used by the web UI: `POST /predict` (multipart field `image`) and `GET /health`.

## Development placeholder weights

If you do not have a trained bundle yet, generate a local `model7_bundle.pth` (ImageNet backbone + random head, **not clinically valid**):

```bash
py model/create_dev_bundle.py
```

Weights under `model/*.pth` are gitignored so they are not pushed to GitHub.

The loader prefers **Model7** artifacts when present, then falls back to Model3.

## Checkpoint resolution order

1. `MODEL_CHECKPOINT` or `MODEL_BUNDLE_PATH` or `MODEL7_BUNDLE_PATH` or `MODEL3_BUNDLE_PATH` (first set wins)
2. Otherwise, the first existing file under `model/`:
   - `model7_bundle.pth` (recommended for Model7)
   - `model3_bundle.pth`
   - `best_model.pth` (requires a meta JSON; see below)

## Option A — Model7 full bundle (recommended)

After training in `Model7.ipynb`, run a **new cell** once you have `train_mean`, `train_std`, and `best_threshold` (your notebook uses `best_threshold = thr_youden` after TTA):

```python
import torch, os, json

bundle = {
    "model_state_dict": model.state_dict(),
    "config": {k: v for k, v in CFG.__dict__.items() if isinstance(v, (int, float, bool, str))},
    "chosen_threshold": float(best_threshold),
    "train_gray_mean": float(train_mean),
    "train_gray_std": float(train_std),
    "use_inference_tta": True,   # matches validation TTA; set False for faster CPU inference
}

out_path = os.path.join(CFG.output_dir, "model7_bundle.pth")
torch.save(bundle, out_path)
print("Saved:", out_path)
```

Copy `model7_bundle.pth` into this repo folder:

- `model/model7_bundle.pth`

## Option B — Notebook output `best_model.pth` + meta JSON

The notebook saves weights only:

```python
torch.save(model.state_dict(), os.path.join(CFG.output_dir, "best_model.pth"))
```

Copy:

- `best_model.pth` → `model/best_model.pth`
- Create `model/model7_meta.json` (see `model/model7_meta.example.json`)

Required meta fields:

- `img_size`, `dropout`, `use_dataset_stats`
- `train_gray_mean`, `train_gray_std` (from your notebook’s `estimate_gray_mean_std`)
- `chosen_threshold` (e.g. `best_threshold` / `thr_youden` after TTA)
- Optional: `use_inference_tta` (`true` to average the same 3 TTA crops as in the notebook)

You can point to a custom meta path with `MODEL_META_PATH`.

## Install and run

```bash
pip install -r model/requirements.txt
uvicorn model.predict:app --host 0.0.0.0 --port 8000 --reload
```

`GET /health` returns `checkpoint`, `model_version` (`model7` | `model3` | `custom`), and whether TTA is on.

## Frontend

In `.env` (local, not committed):

```env
VITE_PREDICT_API_URL="http://localhost:8000"
```

## API response

```json
{
  "prediction": "Cardiomegaly",
  "confidence": 0.87,
  "heatmap_url": null
}
```

`heatmap_url` is reserved for future Grad-CAM support.
