#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m compileall -q src custom_nodes tests
PYTHONPATH="$ROOT/src" python -m unittest discover -s tests -v
git diff --check
