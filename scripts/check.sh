#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

"$PYTHON_BIN" -m compileall -q src custom_nodes tests
PYTHONPATH="$ROOT/src" "$PYTHON_BIN" -m unittest discover -s tests -v
bash -n bin/comfycolab
sh -n install.sh
COMFYCOLAB_PYTHON="$PYTHON_BIN" "$ROOT/bin/comfycolab" --help \
  | grep -Fq "Run pinned ComfyUI"
git diff --check
