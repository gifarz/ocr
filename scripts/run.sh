#!/usr/bin/env bash
# One-command local runner for the CompliFi OCR service.
#
# Usage:
#   ./scripts/run.sh          # dev: auto-reload on code changes
#   ./scripts/run.sh prod     # prod: no reload, binds 0.0.0.0 (same as the Dockerfile's CMD)
#
# What this does, in order - each step is skipped if it's already done, so
# re-running this after the first time is fast and safe:
#   1. Create a venv at .venv/ if one doesn't exist yet.
#   2. Install/update requirements.txt into it.
#   3. Check tesseract is actually on PATH (the #1 first-run failure - a
#      missing system package, not a Python one - so this fails with a
#      clear message instead of a confusing traceback from pytesseract).
#   4. Start uvicorn.
#
# .env is picked up automatically by app/config.py's own load_dotenv()
# call - nothing to source here.
set -euo pipefail

# Resolve paths relative to the repo root regardless of the caller's
# current directory, so `./scripts/run.sh` works the same from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

VENV_DIR=".venv"
MODE="${1:-dev}"

if ! command -v tesseract >/dev/null 2>&1; then
    echo "tesseract is not on PATH - install it first:" >&2
    echo "  sudo apt-get install -y tesseract-ocr tesseract-ocr-ind" >&2
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "No venv found at $VENV_DIR - creating one..."
    python3 -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "Installing/updating dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

if [ ! -f .env ] && [ -f .env.example ]; then
    echo "No .env found - copying .env.example (fine for local dev; fill in" \
         "OCR_SERVICE_API_KEY before exposing this beyond localhost)."
    cp .env.example .env
fi

PORT="${PORT:-8089}"

case "$MODE" in
    dev)
        echo "Starting in dev mode (auto-reload) on port $PORT..."
        exec uvicorn app.main:app --reload --port "$PORT"
        ;;
    prod)
        echo "Starting in prod mode on 0.0.0.0:$PORT..."
        exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
        ;;
    *)
        echo "Unknown mode '$MODE' - use 'dev' (default) or 'prod'." >&2
        exit 1
        ;;
esac
