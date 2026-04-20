# Med Image Clarity (CardioScan)

Web app for AI-assisted chest X-ray screening, with optional Supabase auth/history and a local Python inference API.

## Model weights (where they live)

Trained PyTorch files **are not committed** to Git (they are large). They must live on your machine here:

| What | Path in this repo |
|------|-------------------|
| **Recommended (Model7)** | `model/model7_bundle.pth` |
| Legacy Model3 | `model/model3_bundle.pth` |
| Weights only + meta | `model/best_model.pth` + `model/model7_meta.json` |

Example on Windows:

`med-image-clarity\model\model7_bundle.pth`

### Quick dev placeholder

If you need a file so the API starts before you copy real weights from Colab:

```bash
py model/create_dev_bundle.py
```

That creates `model/model7_bundle.pth` locally (ImageNet backbone + untrained head — **not for clinical use**). Replace it with your real bundle from `Model7.ipynb` when ready.

Full instructions, export cell, and API usage: **[model/README.md](model/README.md)**.

## Web app

```bash
npm install
npm run dev
```

Set `VITE_PREDICT_API_URL=http://localhost:8000` in a local `.env` file (do not commit secrets) to use the Python backend.

## Inference API

```bash
pip install -r model/requirements.txt
uvicorn model.predict:app --host 0.0.0.0 --port 8000 --reload
```

See [model/README.md](model/README.md) for `POST /predict`, health checks, and environment variables.
