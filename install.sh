#!/bin/sh
set -eu

BIN_DIR="${HOME}/.local/bin"
DESTINATION="${BIN_DIR}/comfycolab"
SOURCE_URL="https://raw.githubusercontent.com/DragonLord1998/ComfyColab/main/bin/comfycolab"
TEMPORARY="$(mktemp -t comfycolab-install.XXXXXX)"

cleanup() {
  rm -f "$TEMPORARY"
}
trap cleanup EXIT

mkdir -p "$BIN_DIR"
curl -fsSL "$SOURCE_URL" -o "$TEMPORARY"
chmod 755 "$TEMPORARY"
mv "$TEMPORARY" "$DESTINATION"
trap - EXIT

echo "Installed: $DESTINATION"
echo "Start with: comfycolab start"

case ":${PATH}:" in
  *":${BIN_DIR}:"*) ;;
  *)
    echo "Note: add $BIN_DIR to PATH before using the command." >&2
    ;;
esac
