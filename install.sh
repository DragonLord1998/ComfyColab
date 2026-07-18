#!/bin/sh
set -eu

PYTHON_BIN="${COMFYCOLAB_PYTHON:-}"
REPOSITORY_URL="${COMFYCOLAB_REPO_URL:-https://github.com/DragonLord1998/ComfyColab.git}"
REPOSITORY_REF="${COMFYCOLAB_REPO_REF:-main}"
VENV_DIR="${COMFYCOLAB_VENV_DIR:-${HOME}/.local/share/comfycolab/venv}"
BIN_DIR="${COMFYCOLAB_BIN_DIR:-${HOME}/.local/bin}"

if [ -z "$PYTHON_BIN" ]; then
  for candidate in python3.13 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1 \
      && "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))' \
        >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [ -z "$PYTHON_BIN" ] \
  || ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))' \
    >/dev/null 2>&1; then
  echo "ComfyColab requires Python 3.12 or newer." >&2
  echo "Install a supported Python or set COMFYCOLAB_PYTHON to its executable." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install --upgrade \
  "git+${REPOSITORY_URL}@${REPOSITORY_REF}"
mkdir -p "$BIN_DIR"
ln -sf "$VENV_DIR/bin/comfycolab" "$BIN_DIR/comfycolab"

echo "Installed: $BIN_DIR/comfycolab"
echo "Start with: comfycolab start"

case ":${PATH}:" in
  *":${BIN_DIR}:"*) ;;
  *)
    echo "Note: add $BIN_DIR to PATH before using the command." >&2
    ;;
esac
