#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# dev-public.sh
#
# One command to expose the local CardioScan inference server to the public
# internet via ngrok, so a hosted frontend (Lovable, Vercel, etc.) can talk to
# the model running on your laptop.
#
# What it does:
#   1. Boots `uvicorn` against `inference_server/server.py` on port 8000
#      (only if it isn't already running).
#   2. Boots `ngrok http 8000` and captures the public https URL it prints.
#   3. Writes that URL into the project `.env` as `VITE_PREDICT_API_URL=...`
#      so `npm run dev` and `npm run build` automatically pick it up.
#   4. Streams logs from both processes; Ctrl+C cleans them up.
#
# Prerequisites:
#   * `ngrok` installed and `ngrok config add-authtoken <token>` already run
#     (free account works; sign up at https://dashboard.ngrok.com).
#   * `inference_server/.venv` set up per inference_server/README.md (or
#     activate your own python env before running this script).
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_DIR="$ROOT_DIR/inference_server"
ENV_FILE="$ROOT_DIR/.env"
PORT="${PORT:-8000}"

PIDS=()
cleanup() {
  echo
  echo "[dev-public] shutting down…"
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# 1. uvicorn
# ---------------------------------------------------------------------------
if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "[dev-public] starting uvicorn on :${PORT}"
  pushd "$SERVER_DIR" >/dev/null
  if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  fi
  uvicorn server:app --host 0.0.0.0 --port "$PORT" --log-level info &
  PIDS+=("$!")
  popd >/dev/null

  echo "[dev-public] waiting for /health…"
  for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "[dev-public] uvicorn is ready"
      break
    fi
    sleep 1
  done
else
  echo "[dev-public] uvicorn already running on :${PORT}, reusing it"
fi

# ---------------------------------------------------------------------------
# 2. ngrok
# ---------------------------------------------------------------------------
if ! command -v ngrok >/dev/null 2>&1; then
  echo "[dev-public] ERROR: 'ngrok' not found in PATH." >&2
  echo "             Install from https://ngrok.com/download and run:" >&2
  echo "               ngrok config add-authtoken <YOUR_TOKEN>" >&2
  exit 1
fi

echo "[dev-public] starting ngrok http ${PORT}"
ngrok http "$PORT" --log=stdout --log-format=json >"/tmp/ngrok-cardio.log" 2>&1 &
PIDS+=("$!")

# ---------------------------------------------------------------------------
# 3. Discover the public URL via the ngrok local API and write it to .env
# ---------------------------------------------------------------------------
read_public_url() {
  curl -fsS http://127.0.0.1:4040/api/tunnels 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(next((t["public_url"] for t in d.get("tunnels", []) if t.get("public_url","").startswith("https://")), ""))'
}

PUBLIC_URL=""
for _ in $(seq 1 30); do
  PUBLIC_URL="$(read_public_url || true)"
  if [[ -n "$PUBLIC_URL" ]]; then
    break
  fi
  sleep 1
done

if [[ -z "$PUBLIC_URL" ]]; then
  echo "[dev-public] ERROR: could not read ngrok public URL from http://127.0.0.1:4040" >&2
  exit 1
fi

echo "[dev-public] public URL: $PUBLIC_URL"

# Update / add VITE_PREDICT_API_URL in .env (in-place) ----------------------
touch "$ENV_FILE"
if grep -qE '^VITE_PREDICT_API_URL=' "$ENV_FILE"; then
  # macOS sed needs an empty -i suffix
  sed -i.bak -E "s|^VITE_PREDICT_API_URL=.*$|VITE_PREDICT_API_URL=${PUBLIC_URL}|" "$ENV_FILE"
  rm -f "${ENV_FILE}.bak"
else
  printf '\nVITE_PREDICT_API_URL=%s\n' "$PUBLIC_URL" >>"$ENV_FILE"
fi
echo "[dev-public] updated .env → VITE_PREDICT_API_URL=${PUBLIC_URL}"

cat <<EOF

  ============================================================================
   Inference server is now reachable from the public internet.

   Local frontend (npm run dev) will use it automatically after a restart.

   To deploy on Lovable:
     1. Push this repo to GitHub.
     2. In Lovable → Project → Environment variables, set:
          VITE_PREDICT_API_URL = ${PUBLIC_URL}
        (re-run this script when the URL changes; or pay for a static
         ngrok domain so it never changes.)
     3. Redeploy.

   Press Ctrl+C to tear everything down.
  ============================================================================

EOF

# Hold the script open so the trap fires on Ctrl+C.
wait
