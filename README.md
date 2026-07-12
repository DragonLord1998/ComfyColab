# ComfyColab

ComfyColab starts a temporary GPU-backed Google Colab session from the command
line, installs ComfyUI and its GGUF loader, exposes the ComfyUI interface through
a Cloudflare quick tunnel, and adds a curated Z-Image Turbo bundle-loader node.

There is no Google Drive integration. Models are downloaded inside Colab and
disappear when the runtime is released.

## What `comfycolab start` does

1. Creates or reuses a named Colab session through `google-colab-cli`.
2. Clones or updates ComfyUI in `/content/ComfyUI`.
3. Installs `city96/ComfyUI-GGUF`.
4. Clones this repository and links its node pack into ComfyUI.
5. Starts ComfyUI and `cloudflared` as detached Colab processes.
6. Prints the public URL for the ComfyUI interface.

Cloudflare only exposes the ComfyUI interface. Model downloads run directly
inside the Colab VM.

## Install

Requirements:

- Python 3.10 or newer
- Git
- A Google account with Colab access

```bash
git clone <this-repository-url> ComfyColab
cd ComfyColab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The Python installation brings in the official `google-colab-cli` package.

## Start

The default authentication strategy is Application Default Credentials (ADC),
matching the current `google-colab-cli` recommendation. The default runtime is a
G4 with an NVIDIA RTX PRO 6000 GPU:

```bash
comfycolab start
```

The launcher forwards `GOOGLE_APPLICATION_CREDENTIALS` when it is set. If your
existing Colab CLI setup uses its OAuth flow instead:

```bash
comfycolab start --auth oauth2
```

Choose a different supported accelerator when available:

```bash
comfycolab start --gpu T4
```

When running from a Git clone, the launcher reads the public repository URL from
the `origin` remote. For an exported source archive or wheel, pass it explicitly:

```bash
comfycolab start --repo-url https://github.com/OWNER/ComfyColab.git
```

The final output includes the URL to open:

```text
ComfyUI: https://example.trycloudflare.com
Session: comfycolab
```

Lifecycle commands:

```bash
comfycolab status
comfycolab url
comfycolab stop
```

An already-running ComfyUI process is reused by default. After publishing a
catalog or node-pack update, refresh the managed repositories and restart the UI:

```bash
comfycolab start --refresh
```

ComfyUI and ComfyUI-GGUF are pinned to revisions tested with this release. A
ComfyColab release advances those pins intentionally instead of silently taking
breaking upstream changes.

## Z-Image Turbo node

Add **Z-Image Turbo Bundle Loader** under `ComfyColab/loaders`.

Inputs:

- `quantization`: `Q4_K_M`, `Q3_K_M`, `Q5_K_M`, or `Q8_0`
- `force_redownload`: discard and fetch the temporary files again

Outputs:

- `MODEL`: Z-Image Turbo diffusion model loaded by ComfyUI-GGUF
- `CLIP`: Qwen3-4B Q4 text encoder loaded as `lumina2`
- `VAE`: the official Z-Image `ae.safetensors`

The first Q4 execution downloads roughly 7.8 GB. Later executions in the same
Colab session reuse the verified files. Downloads use partial files, resume when
the server supports byte ranges, and are promoted into the model folders only
after their SHA-256 checksum matches the catalog.

## Catalog

The initial catalog is
[`custom_nodes/ComfyColab-ZImage/catalog/z_image_turbo.json`](custom_nodes/ComfyColab-ZImage/catalog/z_image_turbo.json).
It is deliberately curated rather than accepting arbitrary user URLs. Each
entry records the filename, Hugging Face download URL, checksum, and display
size.

The initial sources are:

- Z-Image Turbo GGUF models: `jayn7/Z-Image-Turbo-GGUF`
- Qwen3-4B GGUF text encoder: `unsloth/Qwen3-4B-GGUF`
- VAE: `Comfy-Org/z_image_turbo`

Model files retain their upstream licenses. This repository does not redistribute
the weights.

## Verify locally

The local checks do not allocate a Colab runtime or download model weights:

```bash
bash scripts/check.sh
```

They validate CLI command construction, bootstrap rendering, catalog integrity,
atomic downloads, and delegation to the existing ComfyUI loaders.
