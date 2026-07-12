#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m compileall -q src custom_nodes tests
PYTHONPATH="$ROOT/src" python -m unittest discover -s tests -v
bash -n bin/comfycolab
sh -n install.sh
launcher_tmp="$(mktemp -d -t comfycolab-launcher.XXXXXX)"
cleanup_launcher_test() {
  rm -rf "$launcher_tmp"
}
trap cleanup_launcher_test EXIT
COMFYCOLAB_COLAB_BIN="$ROOT/tests/fixtures/fake_colab.sh" \
COMFYCOLAB_CONFIG="$launcher_tmp/sessions.json" \
COMFYCOLAB_STATE_DIR="$launcher_tmp/state" \
COMFYCOLAB_BOOTSTRAP_URL="file://$ROOT/src/comfycolab/remote_bootstrap.py" \
  "$ROOT/bin/comfycolab" start > "$launcher_tmp/output.txt"
grep -Fq "ComfyUI: https://fake.trycloudflare.com" "$launcher_tmp/output.txt"
grep -Fq "https://fake.trycloudflare.com" "$launcher_tmp/state/comfy-url"
EXPECT_REFRESH=1 \
COMFYCOLAB_COLAB_BIN="$ROOT/tests/fixtures/fake_colab.sh" \
COMFYCOLAB_CONFIG="$launcher_tmp/sessions.json" \
COMFYCOLAB_STATE_DIR="$launcher_tmp/state" \
COMFYCOLAB_BOOTSTRAP_URL="file://$ROOT/src/comfycolab/remote_bootstrap.py" \
  "$ROOT/bin/comfycolab" start --refresh > "$launcher_tmp/refresh-output.txt"
grep -Fq "ComfyUI: https://fake.trycloudflare.com" "$launcher_tmp/refresh-output.txt"
git diff --check
