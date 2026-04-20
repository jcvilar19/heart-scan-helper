# Cardiomegaly inference (Model7 + Model3)

FastAPI: `POST /predict` (multipart field `image`) and `GET /health`.

## Model7 in this repo

- **`model/Model7.ipynb`** — copy of your training notebook (reference / re-run in Colab).
- **`model/model7_meta.json`** — preprocessing + threshold aligned with a Colab run of that notebook (edit if you retrain).
- **`model/best_model.pth`** — **not in Git** (too large). After training, copy Colab’s `best_model.pth` here, or generate a local dev file (below).

### Default checkpoint order (`predict.py`)

1. Environment override (first that is set and exists):  
   `MODEL_CHECKPOINT`, `MODEL_BUNDLE_PATH`, `MODEL3_BUNDLE_PATH`, `MODEL7_BUNDLE_PATH`
2. Otherwise the first file that exists under `model/`:
   - **`best_model.pth`** (Model7 notebook export — **recommended**)
   - `model7_bundle.pth` (optional full `torch.save` dict)
   - `model3_bundle.pth` (legacy)

For `best_model.pth`, the server loads **`model/model7_meta.json`** (or `MODEL_META_PATH`, or `best_model_meta.json` next to the weights).

### Git / large files

`model/*.pth` is **gitignored**. Only the notebook, `model7_meta.json`, and scripts are versioned.

### Dev placeholder (not clinically valid)

To create a local `best_model.pth` so the API runs before you copy Colab weights:

```bash
py model/create_dev_bundle.py
```

That writes **`model/best_model.pth`** (state_dict only) using ImageNet backbone + random head. **Replace** it with your real Colab `best_model.pth` when you have it.

If no weights exist yet, `GET /health` returns `status: "no_weights"` and `POST /predict` returns **503** until a `.pth` is present.

---

## Option A — Model7 as in the notebook (recommended)

After training in Colab (`Model7.ipynb`), the notebook saves:

```python
torch.save(model.state_dict(), os.path.join(CFG.output_dir, "best_model.pth"))
```

1. Download **`best_model.pth`** from your Colab / Drive output folder.  
2. Copy it to **`model/best_model.pth`** in this project.  
3. Keep or edit **`model/model7_meta.json`** so `train_gray_mean`, `train_gray_std`, and `chosen_threshold` match **your** notebook printouts after TTA.

---

## Option B — Full bundle (optional)

If you prefer a single file with weights + config inside:

```python
bundle = {
    "model_state_dict": model.state_dict(),
    "config": {k: v for k, v in CFG.__dict__.items() if isinstance(v, (int, float, bool, str))},
    "chosen_threshold": float(best_threshold),
    "train_gray_mean": float(train_mean),
    "train_gray_std": float(train_std),
    "use_inference_tta": True,
}
torch.save(bundle, os.path.join(CFG.output_dir, "model7_bundle.pth"))
```

Save as **`model/model7_bundle.pth`** (still gitignored locally).

---

## Install and run

```bash
pip install -r model/requirements.txt
uvicorn model.predict:app --host 0.0.0.0 --port 8000 --reload
```

`GET /health` returns `checkpoint`, `model_version`, `tta`, or `status: "no_weights"` if no `.pth` is found.

## Frontend

In `.env` (local):

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
