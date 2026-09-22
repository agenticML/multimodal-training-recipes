#!/usr/bin/env sh
set -eu

usage() {
    cat <<'EOF'
Usage:
  AGENTICML_API_KEY=... ./scripts/run-online.sh \
    [--agenticml-host URL] --task TASK [training-recipe options]

Example:
  AGENTICML_API_KEY=... ./scripts/run-online.sh \
    --task asr --architecture wav2vec2 --profile smoke

The API key must be supplied through AGENTICML_API_KEY. The launcher forces
AgenticML tracking to online mode and forwards all remaining arguments to the
standard training-recipe CLI. AGENTICML_HOST defaults to
https://api.agenticml.xyz and can be overridden with --agenticml-host URL.
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi

AGENTICML_HOST=https://api.agenticml.xyz
if [ "${1:-}" = "--agenticml-host" ]; then
    if [ -z "${2:-}" ]; then
        echo "error: --agenticml-host requires a URL" >&2
        usage >&2
        exit 2
    fi
    AGENTICML_HOST=$2
    shift 2
fi

if [ -z "${AGENTICML_API_KEY:-}" ]; then
    echo "error: AGENTICML_API_KEY is not set" >&2
    exit 2
fi

for argument in "$@"; do
    case "$argument" in
        --tracker-mode|--tracker-mode=*|--no-track)
            echo "error: tracking mode is managed by run-online.sh" >&2
            exit 2
            ;;
    esac
done

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export AGENTICML_HOST
export AGENTICML_MODE=online
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
exec "$PYTHON_BIN" -m training_recipes.cli --tracker-mode online "$@"
