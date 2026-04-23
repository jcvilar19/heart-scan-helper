#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# deploy-space.sh
#
# Push the inference server + model checkpoints to a Hugging Face Space using
# the Docker SDK. Result: a permanent public HTTPS URL that the Lovable
# frontend can call (no laptop, no ngrok, no port games).
#
# One-time setup
#   1. Make a free account at https://huggingface.co
#   2. Create a token (Settings → Access Tokens → "write")
#   3. `pip install -U "huggingface_hub[cli]"` then `hf auth login`
#      (older versions used `huggingface-cli login` — same thing)
#   4. Create the Space in the UI:
#        https://huggingface.co/new-space
#        SDK: Docker, Hardware: CPU basic (free)
#        Name it whatever, e.g. `cardio-scan-api`
#
# Usage
#   HF_USER=<your-handle> HF_SPACE=cardio-scan-api ./scripts/deploy-space.sh
#
# What it does
#   * Clones https://huggingface.co/spaces/$HF_USER/$HF_SPACE into a temp dir
#   * Copies Dockerfile, README, inference_server/, model_training/src/,
#     model_training/notebooks/results/ into it
#   * Stages .pth checkpoints with git-lfs
#   * Commits and pushes — HF rebuilds the image automatically (~5-8 min)
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPACE_SRC_DIR="$ROOT_DIR/space"

: "${HF_USER:?Set HF_USER (your Hugging Face username) — e.g. HF_USER=jcvilar}"
: "${HF_SPACE:?Set HF_SPACE (the Space name you created) — e.g. HF_SPACE=cardio-scan-api}"
HF_BRANCH="${HF_BRANCH:-main}"
COMMIT_MSG="${COMMIT_MSG:-Deploy CardioScan inference $(date -u +%Y-%m-%dT%H:%M:%SZ)}"
HF_MODEL_REPO_ID="${HF_MODEL_REPO_ID:-}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2; exit 1
fi
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs is required (brew install git-lfs && git lfs install)" >&2; exit 1
fi
if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Install huggingface_hub: pip install -U 'huggingface_hub[cli]' && hf auth login" >&2; exit 1
fi

WORK_DIR="$(mktemp -d -t cardio-space.XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

# Resolve the stored HF token using whichever CLI is available.
HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" ]] && command -v hf >/dev/null 2>&1; then
  HF_TOKEN="$(hf auth token 2>/dev/null || true)"
fi
if [[ -z "$HF_TOKEN" ]]; then
  HF_TOKEN="$(python3 -c 'from huggingface_hub import HfFolder; print(HfFolder.get_token() or "")' 2>/dev/null || true)"
fi
if [[ -z "$HF_TOKEN" ]]; then
  echo "Could not find an HF token. Run: hf auth login   (or: huggingface-cli login)" >&2
  exit 1
fi

REMOTE_URL="https://user:${HF_TOKEN}@huggingface.co/spaces/${HF_USER}/${HF_SPACE}"
echo "[deploy-space] cloning https://huggingface.co/spaces/${HF_USER}/${HF_SPACE}"
git clone --depth=1 --branch "$HF_BRANCH" "$REMOTE_URL" "$WORK_DIR/space"

cd "$WORK_DIR/space"
git lfs install --local

# Track .pth (and other large files) with LFS BEFORE we copy them in.
cat > .gitattributes <<'EOF'
*.pth filter=lfs diff=lfs merge=lfs -text
*.pt  filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
EOF

# Wipe the existing tree (except .git / .gitattributes) so removed files are
# actually deleted on HF — keeps each deploy reproducible.
find . -mindepth 1 -maxdepth 1 \
  ! -name '.git' \
  ! -name '.gitattributes' \
  -exec rm -rf {} +

echo "[deploy-space] copying files"
cp "$SPACE_SRC_DIR/Dockerfile" ./Dockerfile
cp "$SPACE_SRC_DIR/README.md"  ./README.md

mkdir -p inference_server model_training/src model_training/notebooks/results
rsync -a --delete \
      --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
      "$ROOT_DIR/inference_server/"             ./inference_server/
rsync -a --delete \
      --exclude '__pycache__' --exclude '*.pyc' \
      "$ROOT_DIR/model_training/src/"           ./model_training/src/
if [[ -n "$HF_MODEL_REPO_ID" ]]; then
  echo "[deploy-space] HF_MODEL_REPO_ID is set ($HF_MODEL_REPO_ID) — skipping local .pth copy"
  # Keep only lightweight metadata in the Space image. The server downloads
  # manifest/checkpoints at runtime from the HF model repo.
  rsync -a --delete \
        --include '*/' \
        --include '*.csv' --include '*.json' \
        --exclude '*.pth' \
        --exclude '*' \
        "$ROOT_DIR/model_training/notebooks/results/" ./model_training/notebooks/results/
else
  rsync -a --delete \
        --include '*/' \
        --include '*.pth' --include '*.csv' --include '*.json' \
        --exclude '*' \
        "$ROOT_DIR/model_training/notebooks/results/" ./model_training/notebooks/results/
fi

# Sanity check
test -f ./Dockerfile
test -f ./inference_server/server.py
test -f ./model_training/src/model.py
if [[ -z "$HF_MODEL_REPO_ID" ]]; then
  ls ./model_training/notebooks/results/*.pth >/dev/null
fi

git add -A
if git diff --cached --quiet; then
  echo "[deploy-space] nothing to commit; Space is already up to date"
  exit 0
fi
git -c user.email="deploy@cardio.local" \
    -c user.name="cardio-deploy" \
    commit -m "$COMMIT_MSG"

echo "[deploy-space] pushing to $HF_BRANCH (this will trigger a rebuild)"
git push origin "$HF_BRANCH"

cat <<EOF

  ============================================================================
   Pushed to Hugging Face. Build will take ~5-8 min.

   Watch logs:
     https://huggingface.co/spaces/${HF_USER}/${HF_SPACE}/logs

   Public URL (set this in Lovable as VITE_PREDICT_API_URL):
     https://${HF_USER}-${HF_SPACE}.hf.space

   Test once it is "Running":
     curl https://${HF_USER}-${HF_SPACE}.hf.space/health
  ============================================================================

EOF
