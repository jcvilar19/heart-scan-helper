# Med Image Clarity (CardioScan)

Web app for AI-assisted chest X-ray screening, with optional Supabase auth/history and a local Python inference API.

## Model weights (Model7)

The training notebook lives in the repo as **`model/Model7.ipynb`**. Preprocessing and threshold for inference are in **`model/model7_meta.json`** (committed).

Trained weights **are not in Git** (large). Put Colab’s export here:

| What | Path |
|------|------|
| **Model7 weights (notebook default)** | `model/best_model.pth` |
| Meta (stats + threshold) | `model/model7_meta.json` (already in repo; edit after retrain) |
| Optional full bundle | `model/model7_bundle.pth` (local only) |
| Legacy | `model/model3_bundle.pth` |

Example: `med-image-clarity\model\best_model.pth`

### Quick dev placeholder

```bash
py model/create_dev_bundle.py
```

Creates **`model/best_model.pth`** locally (not clinically valid). Replace with your real **`best_model.pth`** from Colab.

Details: **[model/README.md](model/README.md)**.

## Web app

```bash
npm install
npm run dev
```

Create a local `.env` from the template (this file is gitignored):

```bash
copy .env.example .env
```

Then edit `.env` and add your **Supabase** URL and anon key (Dashboard → Project Settings → API):

- `VITE_SUPABASE_URL` / `VITE_SUPABASE_PUBLISHABLE_KEY` (browser)
- `SUPABASE_URL` / `SUPABASE_PUBLISHABLE_KEY` (same values for SSR)

Optional: `VITE_PREDICT_API_URL=http://localhost:8000` for the Python backend.

## Inference API

```bash
pip install -r model/requirements.txt
uvicorn model.predict:app --host 0.0.0.0 --port 8000 --reload
```

See [model/README.md](model/README.md) for `POST /predict`, health checks, and environment variables.
