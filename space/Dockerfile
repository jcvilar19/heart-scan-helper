# syntax=docker/dockerfile:1
#
# Hugging Face Spaces (Docker SDK) image for the CardioScan inference server.
#
# Spaces conventions:
#   * Container must listen on the port declared in `README.md` (`app_port`).
#   * Container is started as user `user` (UID 1000), not root.
#   * /data is the only writable location persisted across restarts (we don't
#     need persistence — checkpoints ship inside the image).
#
# Build context expected by `scripts/deploy-space.sh`:
#   /
#   ├── Dockerfile                       (this file)
#   ├── README.md                        (HF Spaces metadata)
#   ├── inference_server/                (FastAPI app + requirements.txt)
#   └── model_training/
#       ├── src/                         (only the package; notebooks excluded)
#       └── notebooks/results/           (.pth checkpoints + manifest + metrics)

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    XDG_CACHE_HOME=/app/.cache

# System deps for Pillow / scientific Python wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires the runtime user to be UID 1000.
RUN useradd --create-home --uid 1000 user
WORKDIR /app

# Install Python deps first so layer caches when only code/weights change.
COPY --chown=user:user inference_server/requirements.txt /app/inference_server/requirements.txt
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1 torchvision==0.19.1 && \
    pip install -r /app/inference_server/requirements.txt

# Copy the actual code and weights.
COPY --chown=user:user inference_server/      /app/inference_server/
COPY --chown=user:user model_training/src/    /app/model_training/src/
COPY --chown=user:user model_training/notebooks/results/ /app/model_training/notebooks/results/

# Pre-create writable cache dirs owned by `user`.
RUN mkdir -p /app/.cache/torch /app/.cache/huggingface && chown -R user:user /app

USER user

# Spaces routes traffic to whatever app_port we declared in README.md (7860).
ENV PORT=7860 \
    LOG_LEVEL=INFO \
    MODEL_USE_TTA=true \
    ALLOWED_ORIGIN_REGEX="https://([a-z0-9-]+\\.)*lovable\\.app|https://([a-z0-9-]+\\.)*lovableproject\\.com|https://([a-z0-9-]+\\.)*hf\\.space|http://localhost(:\\d+)?|http://127\\.0\\.0\\.1(:\\d+)?"

EXPOSE 7860

# server.py expects to be importable from /app/inference_server.
WORKDIR /app/inference_server
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
