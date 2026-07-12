# ComfyColab

ComfyColab starts a temporary GPU-backed Google Colab session from the command
line, installs ComfyUI and its GGUF loader, exposes the ComfyUI interface through
a Cloudflare quick tunnel, and adds curated bundle-loader nodes for Z-Image
Turbo, Qwen Image Edit 2511, and Krea 2.

There is no Google Drive integration. Models are downloaded inside Colab and
disappear when the runtime is released.

## Mac quick start

ComfyColab uses an already-working `colab` CLI. Install the lightweight shell
command once—no additional local Python environment is required:

```bash
curl -fsSL https://raw.githubusercontent.com/DragonLord1998/ComfyColab/main/install.sh | sh
```

After that, the complete startup flow is one command:

```bash
comfycolab start
```

It requests a G4 runtime, runs the public bootstrap inside Colab, and prints the
Cloudflare URL for ComfyUI. The existing Colab CLI remains responsible for Google
authentication and runtime allocation.

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

## Bundle-loader nodes

All three nodes live under `ComfyColab/loaders`, download only when executed,
and return standard `MODEL`, `CLIP`, and `VAE` outputs.

### Z-Image Turbo

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

### Qwen Image Edit 2511

Add **Qwen Image Edit 2511 Bundle Loader** and choose `Q4_K_M`, `Q3_K_M`,
`Q5_K_M`, or `Q8_0`.

- `MODEL`: selected Qwen Image Edit 2511 GGUF, loaded by ComfyUI-GGUF
- `CLIP`: official Qwen 2.5 VL 7B FP8 text/vision encoder, loaded as `qwen_image`
- `VAE`: official Qwen Image VAE

The Q4 bundle downloads roughly 22.9 GB. Connect its outputs to ComfyUI's
native Qwen Image Edit conditioning nodes. The model weights are Apache-2.0.

### Krea 2

Add **Krea 2 Bundle Loader** and choose `Turbo FP8` (default) or
`Raw FP8 (training/base)`. Krea does not recommend the Raw checkpoint for
normal inference.

- `MODEL`: official Krea 2 FP8 diffusion model, loaded by ComfyUI's native loader
- `CLIP`: official Qwen3-VL 4B FP8 encoder, loaded as `krea2`
- `VAE`: official Qwen Image VAE

Each Krea 2 bundle downloads roughly 18.6 GB. Krea 2 uses its own community
license; review the license in the
[`Comfy-Org/Krea-2`](https://huggingface.co/Comfy-Org/Krea-2) repository before
using the weights. Downloading or using the weights constitutes acceptance of
that license. Its terms include commercial-use and deployment obligations.

For generation, Krea recommends 8 steps with no CFG for Turbo. Qwen Image Edit
2511 workflows typically use ComfyUI's `TextEncodeQwenImageEditPlus` node and
the model's `index_timestep_zero` reference method.

## Catalog

The catalogs are under
[`custom_nodes/ComfyColab-ZImage/catalog`](custom_nodes/ComfyColab-ZImage/catalog).
They are deliberately curated rather than accepting arbitrary user URLs. Each
entry records a revision-pinned Hugging Face URL, filename, checksum, and
display size.

The initial sources are:

- Z-Image Turbo GGUF models: `jayn7/Z-Image-Turbo-GGUF`
- Qwen3-4B GGUF text encoder: `unsloth/Qwen3-4B-GGUF`
- VAE: `Comfy-Org/z_image_turbo`
- Qwen Image Edit GGUF: `unsloth/Qwen-Image-Edit-2511-GGUF`
- Qwen encoder and VAE: `Comfy-Org/Qwen-Image_ComfyUI`
- Krea 2 bundle: `Comfy-Org/Krea-2`

Model files retain their upstream licenses. This repository does not redistribute
the weights.

## Verify locally

The local checks do not allocate a Colab runtime or download model weights:

```bash
bash scripts/check.sh
```

They validate CLI command construction, bootstrap rendering, catalog integrity,
atomic downloads, and delegation to the existing ComfyUI loaders.

Before publishing catalog changes, verify the pinned remote metadata without
downloading the model payloads:

```bash
python scripts/verify_catalogs.py
```
