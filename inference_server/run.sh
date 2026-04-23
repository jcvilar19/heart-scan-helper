#!/usr/bin/env bash
# Run the API with the project venv (avoids: ModuleNotFoundError: No module named 'torch')
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -d .venv ]]; then
  echo "No .venv here. First run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi
PORT="${PORT:-8000}"
exec .venv/bin/uvicorn server:app --host 0.0.0.0 --port "$PORT" "$@"
