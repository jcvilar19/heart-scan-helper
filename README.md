# CardioScan — chest X-ray cardiomegaly screening

A two-piece application:

| Piece | Where it lives | What it is |
|---|---|---|
| **Frontend** | `src/` (React + Vite + TanStack Router) | Drag-and-drop UI, Supabase auth & history, talks to the inference API over HTTPS |
| **Inference API** | `inference_server/server.py` (FastAPI + PyTorch) | Loads the multi-seed DenseNet-121 ensemble trained in `model_training/` and serves `POST /predict` |

The frontend is deployed to **Lovable**, the inference API is deployed to **Hugging Face Spaces** (free CPU tier). Both are described below.

---

## 1. Production setup — Lovable + Hugging Face Spaces (recommended)

This is the path you should follow. No laptop, no ngrok, no port games. The Space gives you a permanent URL like `https://<user>-<space>.hf.space` that Lovable calls forever.

### A. Deploy the inference API to Hugging Face Spaces

One-time:

1. Sign up at <https://huggingface.co> (free).
2. Create a write token: <https://huggingface.co/settings/tokens>.
3. Install the CLI and log in:
   ```bash
   pip install -U "huggingface_hub[cli]"
   huggingface-cli login           # paste the token
   brew install git-lfs && git lfs install
   ```
4. Create the Space (web UI): <https://huggingface.co/new-space>
   - **SDK**: Docker
   - **Hardware**: CPU basic (free)
   - Name it e.g. `cardio-scan-api`

Each deploy:

```bash
HF_USER=<your-handle> HF_SPACE=cardio-scan-api ./scripts/deploy-space.sh
```

The script copies `space/Dockerfile`, `space/README.md`, `inference_server/`, `model_training/src/`, and `model_training/notebooks/results/*.{pth,csv,json}` into the Space repo and pushes. HF rebuilds the image automatically (~5–8 min).

When the Space shows **Running**:

```bash
curl https://<your-handle>-cardio-scan-api.hf.space/health
# → {"ok":true,"models":3,"backbone":"densenet121", ...}
```

### B. Deploy the frontend to Lovable

1. Push this repo to GitHub.
2. In Lovable, connect the repo and add **Environment variables** (same names as `.env.example`):
   - `VITE_SUPABASE_URL`, `VITE_SUPABASE_PUBLISHABLE_KEY`, `VITE_SUPABASE_PROJECT_ID`
   - `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`
   - `VITE_PREDICT_API_URL=https://<your-handle>-cardio-scan-api.hf.space`
3. Redeploy. Open the Lovable URL and upload an X-ray.

CORS in `inference_server/server.py` already accepts `*.lovable.app`, `*.lovableproject.com`, and `*.hf.space` out of the box — no additional config needed.

---

## 2. Local development (everything on your laptop)

Use this while iterating; once a feature is ready, push to Lovable + deploy the Space.

### Frontend

```bash
npm install
cp .env.example .env             # then fill in your Supabase values
npm run dev                      # http://localhost:8080
```

`.env` should keep:

```
VITE_PREDICT_API_URL=http://127.0.0.1:8000
```

> Use `127.0.0.1`, not `localhost` — macOS browsers resolve `localhost` to IPv6 first, which uvicorn isn’t bound to. The frontend also auto-rewrites this URL when you open the app via Vite’s **Network** link (192.168.x).

### Inference API

```bash
cd inference_server
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run.sh                          # = uvicorn server:app --host 0.0.0.0 --port 8000
# To use a different port: PORT=8001 ./run.sh   (then update VITE_PREDICT_API_URL)
```

Always use the venv. Running plain `uvicorn` from a global install gives `ModuleNotFoundError: No module named 'torch'`.

The server auto-detects the model architecture from the checkpoints in `model_training/notebooks/results/` and reads the optimal threshold from `val_metrics_final.json`. See `inference_server/README.md` for env vars and the `/debug/predict` endpoint.

---

## 3. Public preview via ngrok (optional — quick share before deploying the Space)

Useful if you want to demo the local model from another device or a draft Lovable URL without redeploying the Space:

```bash
./scripts/dev-public.sh
```

This boots uvicorn + ngrok, writes the public HTTPS URL into `.env`, and prints instructions. Restart `npm run dev` to pick it up. The URL changes each session unless you upgrade ngrok to a static domain — long-term, Spaces (Section 1) is the better answer.

---

## Repo layout

```
src/                       # React frontend (Vite + TanStack Router)
inference_server/          # FastAPI predictor (loads model_training checkpoints)
  ├─ server.py             # the app
  ├─ requirements.txt
  ├─ run.sh                # local launcher (uses inference_server/.venv)
  └─ README.md             # endpoints + env vars
model_training/            # PyTorch training pipeline (untouched at inference time)
  └─ notebooks/results/    # *.pth checkpoints + threshold metrics
space/                     # Hugging Face Spaces deploy assets
  ├─ Dockerfile            # multi-stage image used by the Space
  └─ README.md             # Spaces metadata (sdk: docker, app_port: 7860)
scripts/
  ├─ deploy-space.sh       # push inference server to HF Spaces (production)
  └─ dev-public.sh         # uvicorn + ngrok helper (quick demo)
.env.example               # template for frontend env vars
```
