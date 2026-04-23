# inference_server

FastAPI service that wraps the **trained ensemble** in
`model_training/notebooks/results/` and exposes a single `POST /predict`
endpoint for the React frontend.

This folder is strictly an inference layer — **nothing inside
`model_training/` is modified**. We only import `src.model.build_model` and
`src.model.cardio_logit` to recreate the architecture before loading the
saved state dicts.

## 1. Install

```bash
cd inference_server
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> The backbone is **auto-detected from the first checkpoint** referenced by
> `ensemble_manifest.csv`, so there is never a mismatch between
> architecture and weights. The shipped checkpoints in
> `model_training/notebooks/results/` are `torchxrayvision densenet121`
> (trained before `CFG.backbone` in `model_training/src/config.py` was
> changed to `efficientnet_b0`); the server correctly identifies and
> uses them. You can still force a specific backbone via `MODEL_BACKBONE`
> if you train a new model — see _Configuration_ below.

## 2. Run

**You must use the venv** where `pip install -r requirements.txt` was run. If
you see `ModuleNotFoundError: No module named 'torch'`, you started `uvicorn`
with the system Python instead of `inference_server/.venv`.

```bash
cd inference_server
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uvicorn server:app --host 0.0.0.0 --port 8000
```

Or, without activating (always uses the project interpreter):

```bash
cd inference_server
./.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
```

If port 8000 is still taken, use **8001** (or any free port) and point the
frontend at the same port:

```bash
PORT=8001 ./run.sh
# in project root .env:
# VITE_PREDICT_API_URL=http://127.0.0.1:8001
```

If your shell prompt already shows `inference_server` in the path, you are
**inside** that folder—do not run `cd inference_server` again (you will get
`no such file`).

**`[Errno 48] address already in use` on port 8000** means something else is
already bound there (usually an older uvicorn you forgot to stop). On macOS:

```bash
lsof -i :8000
# note the PID in the second column, then:
kill <PID>
# if it does not exit:
kill -9 <PID>
```

Then start uvicorn again. To use another port without killing the other process
(e.g. 8001), add `--port 8001` and set `VITE_PREDICT_API_URL` in the frontend
`.env` to match.

> **IPv4/IPv6 gotcha.** Bind with `--host 0.0.0.0` (not `127.0.0.1`). On
> macOS, browsers often resolve `localhost` to IPv6 `::1` first, and
> `uvicorn --host 127.0.0.1` only listens on IPv4, which surfaces in the
> browser as a bare `Network Error` even though `curl 127.0.0.1:8000` works.
> The frontend's `.env` uses `http://127.0.0.1:8000` (not `localhost`) for
> the same reason.

On startup the server will:

1. Read `model_training/notebooks/results/ensemble_manifest.csv`.
2. For each row, rebuild the EfficientNet-B0 architecture and load the
   corresponding `model_seed*.pth` checkpoint.
3. Move every model to CUDA / MPS / CPU (auto-detected).

If the manifest is missing it falls back to
`model_training/notebooks/results/best_model.pth`.

## 3. Frontend wiring

The app's `.env` already points at this server:

```
VITE_PREDICT_API_URL=http://localhost:8000
```

`src/services/predict.ts` posts the uploaded file to `/predict` as
`multipart/form-data` (field name: `image`). The response shape is exactly
what the frontend expects:

```json
{
  "prediction": "Cardiomegaly",
  "confidence": 0.873,
  "heatmap_url": null,
  "source": "model",
  "threshold": 0.504486,
  "ensemble_size": 3,
  "use_tta": true
}
```

The frontend uses `source: "model"` to render a green "Real model" badge on
each result card, so there is no ambiguity about whether a prediction came
from the real trained ensemble.

## 4. Configuration (env vars)

| Variable           | Default                | Purpose                                                  |
| ------------------ | ---------------------- | -------------------------------------------------------- |
| `MODEL_BACKBONE`   | `CFG.backbone`         | Must match the architecture used for training            |
| `MODEL_IMG_SIZE`   | `CFG.img_size`         | Must match training (224 for EfficientNet-B0, 518 for RAD-DINO) |
| `MODEL_THRESHOLD`  | `val_metrics_final.json::threshold` (fallback `0.5`) | Cut-off used when choosing the label string |
| `MODEL_USE_TTA`    | `true`                 | `true` → run the 6-pass TTA used at training evaluation time |
| `ALLOWED_ORIGINS`  | localhost dev origins  | Comma-separated CORS origins (exact match)               |
| `ALLOWED_ORIGIN_REGEX` | _(unset)_          | Regex for origins, e.g. `https://.*\.lovable\.app` for Lovable preview URLs |
| `LOG_LEVEL`        | `INFO`                 | Standard Python logging level                            |

Example:

```bash
MODEL_USE_TTA=true MODEL_THRESHOLD=0.504 uvicorn server:app --port 8000
```

## 5. Smoke-test

```bash
curl -s http://localhost:8000/health | jq .
curl -s -X POST -F "image=@/path/to/xray.png" http://localhost:8000/predict | jq .

# Full transparency: per-model + per-TTA raw logits so you can compare
# against val_predictions.csv / test_predictions.csv in the notebook:
curl -s -X POST -F "image=@/path/to/xray.png" http://localhost:8000/debug/predict | jq .
```

Every `/predict` call is also logged in the server terminal with the
filename, per-model mean logits, and the final probability — useful to
confirm the frontend is actually hitting the server.

## 6. Deploying to production (Lovable + separate inference host)

Lovable hosts the React frontend, but it cannot run this Python server.
You need a separate Python host for the inference server. In all cases the
steps are the same:

1. **Push this repo** (including `model_training/notebooks/results/*.pth`)
   to the chosen host. The whole monorepo is self-contained.
2. **Start command** (the host's "start" or "web" command):
   ```bash
   uvicorn server:app --host 0.0.0.0 --port $PORT --app-dir inference_server
   ```
   Most PaaS hosts inject `PORT`; if yours doesn't, use `8000`.
3. **Requirements**: point the host at `inference_server/requirements.txt`.
4. **Set env vars on the inference host**:
   ```
   ALLOWED_ORIGINS=https://your-app.lovable.app,https://your-custom-domain.com
   # Or, if Lovable assigns preview URLs with a hash prefix:
   ALLOWED_ORIGIN_REGEX=https://.*\.lovable\.app
   MODEL_USE_TTA=true
   ```
5. **Set env var on Lovable (frontend)**:
   ```
   VITE_PREDICT_API_URL=https://your-inference-server-url
   ```
   Then redeploy the Lovable frontend so Vite bakes the new URL into the
   bundle.

### Suggested hosts

| Host | Free tier | Notes |
| ---- | --------- | ----- |
| **Hugging Face Spaces** | 2 vCPU / 16 GB, always-on free | Ideal for this model. Create a Space with the "FastAPI" SDK, push the repo, and point it at `inference_server/server.py`. |
| **Render.com** | Web service, spins down after 15 min idle | Simple Git-push deploy. Cold start ~30 s while weights load. |
| **Fly.io** | Shared-CPU 256 MB free | Docker-based. Dockerfile is trivial (Python base → pip install → CMD uvicorn). |
| **Railway / Modal** | Paid / pay-per-second | Always-on, fastest DX. |

### Keep model checkpoints in Git LFS (optional)

`model_training/notebooks/results/model_seed*.pth` are ~30 MB each. On
GitHub free plans this is fine, but you may want to move them to Git LFS
to keep repo clones small.

## 7. Notes

- The ensemble is loaded once at startup (one-time cost of a few seconds).
- **Auto-detection**: the server inspects the first checkpoint on startup and
  picks the matching backbone (`densenet121`, `efficientnet_b0`,
  `mobilenet_v3_large`, or `rad-dino`). No `CFG` / env-var bookkeeping required.
- **Correct preprocessing per backbone**: the server delegates to
  `model_training/src/dataset.py::get_normalize_fn` so the normalization
  matches training exactly — `xrv_normalize_np` (grayscale, [-1024, 1024]) for
  torchxrayvision DenseNet-121, `imagenet_normalize_np` (3-channel) for every
  other backbone.
- **No pretrained-weight downloads**: torchvision and torchxrayvision
  constructors are monkey-patched so they skip their pretrained-weight
  download entirely — our trained checkpoint fully overwrites those weights
  anyway. This means the server works offline and in sandboxed environments.
- **Fail-fast checkpoint loading**: if `state_dict` keys don't match the
  architecture, startup aborts with a clear error listing the mismatch.
- Each request is ~50–150 ms on CPU without TTA (3 × DenseNet-121 forward
  passes). With `MODEL_USE_TTA=true` that becomes ~0.5–1.5 s per image.
- **Verified**: the server reproduces `notebooks/results/val_predictions.csv`
  probabilities to 6 decimal places (zero delta) on the validation set.
