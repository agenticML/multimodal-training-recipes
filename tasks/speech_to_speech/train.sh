#!/usr/bin/env sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$TASK_ROOT/../.." && pwd)
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${PYTHON:-}" ]; then
    PYTHON_BIN=$PYTHON
elif [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON_BIN=python3
fi
exec "$PYTHON_BIN" -m training_recipes.cli --task speech_to_speech "$@"
