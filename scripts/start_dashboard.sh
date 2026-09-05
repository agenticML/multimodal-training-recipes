#!/usr/bin/env sh
set -eu
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PORT=${1:-8000}
ASSET_DIR=${AGENTICML_ASSET_DIR:-$PROJECT_ROOT/agenticml-assets}
mkdir -p "$ASSET_DIR"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${PYTHON:-}" ]; then
    PYTHON_BIN=$PYTHON
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON_BIN=python3
fi
cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m training_recipes.dashboard --host 127.0.0.1 --port "$PORT" --asset-dir "$ASSET_DIR"
