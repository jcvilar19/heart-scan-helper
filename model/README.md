# Cardiomegaly Model (Model3 Integration)

This folder now contains a FastAPI inference service that loads the `Model3`
bundle exported from your notebook and exposes:

- `POST /predict` (multipart/form-data with `image`)
- `GET /health`

## 1) Export bundle from notebook

Your notebook already saves this artifact in Cell 20:

- `model3_bundle.pth`

Copy that file into this folder:

- `model/model3_bundle.pth`

You can also keep it somewhere else and set `MODEL3_BUNDLE_PATH`.

## 2) Install Python dependencies

```bash
pip install -r model/requirements.txt
```

## 3) Run inference API

```bash
uvicorn model.predict:app --host 0.0.0.0 --port 8000 --reload
```

## 4) Connect frontend to backend

In your frontend `.env` add:

```env
VITE_PREDICT_API_URL="http://localhost:8000"
```

Then restart Vite (`npm run dev`).

## API response format

```json
{
  "prediction": "Cardiomegaly",
  "confidence": 0.87,
  "heatmap_url": null
}
```

`heatmap_url` is currently optional and returned as `null` until Grad-CAM
generation is added on the backend.
