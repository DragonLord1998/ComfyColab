# ComfyColab

Run ComfyUI on a temporary Google Colab GPU from your Mac with one command.

ComfyColab creates a Colab session, installs ComfyUI and ComfyUI-GGUF, adds a
curated model-loader node pack, and prints a Cloudflare URL you can open in any
browser. No Google Drive is mounted and no model setup is required in the Colab
terminal.

## What you get

- A temporary **G4 / NVIDIA RTX PRO 6000** Colab runtime by default
- ComfyUI and `city96/ComfyUI-GGUF`, pinned to tested revisions
- Bundle-loader nodes for Z-Image, Qwen Image Edit, Krea 2, and FLUX.2
- Two simple 3D nodes for TRELLIS.2 generation and UltraShape refinement
- The complete advanced TRELLIS.2 and GeometryPack node suites
- On-demand model downloads with checksums and resume support
- A public Cloudflare URL for the ComfyUI interface
- No Google Drive and no permanent cloud storage

Everything inside `/content` disappears when the Colab runtime is released.

## Quick start on Mac

Before installing ComfyColab, make sure the official `colab` command already
works on your Mac and can create a session. ComfyColab reuses that CLI's Google
authentication; the installer below adds the friendly wrapper and does not
replace your Colab login setup.

Install the command once:

```bash
curl -fsSL https://raw.githubusercontent.com/DragonLord1998/ComfyColab/main/install.sh | sh
```

Start ComfyUI:

```bash
comfycolab start
```

The first launch downloads ComfyUI, its Python dependencies, and a prebuilt
TRELLIS.2 environment into the new runtime. On the default G4, the reusable
TRELLIS cache avoids resolving and rebuilding that environment from scratch.
When everything is ready, the terminal prints:

```text
ComfyUI: https://example.trycloudflare.com
Session: comfycolab
```

Open the `ComfyUI` link in Safari, Chrome, or another browser.

## Everyday commands

```bash
# Start or reuse the runtime
comfycolab start

# Show whether the runtime is active
comfycolab status

# Print the saved ComfyUI URL again
comfycolab url

# Pull the newest node-pack changes and restart ComfyUI
comfycolab start --refresh

# Release the Colab runtime
comfycolab stop
```

Always run `comfycolab stop` when you are finished so the Colab runtime is not
left consuming compute units.

## What happens when you start it?

1. The launcher creates or reuses a named Colab session.
2. It requests the `G4` accelerator unless you configured another one.
3. It clones tested versions of ComfyUI and ComfyUI-GGUF into `/content`.
4. On a matching G4 runtime, it restores the checksum-verified combined 3D
   cache when available, with the existing TRELLIS.2 cache as a rollback path.
5. It clones pinned TRELLIS.2, GeometryPack, and UltraShape sources, applies
   revision-checked compatibility patches, and links both ComfyColab node packs.
6. It starts ComfyUI and a Cloudflare quick tunnel.
7. It prints the browser URL in your Mac terminal.

Cloudflare only carries the browser traffic. Model files are downloaded directly
from Hugging Face to the Colab VM.

## Using the model nodes

In ComfyUI, right-click the canvas and look under:

```text
ComfyColab / loaders
```

Every bundle loader returns the same three standard outputs:

- `MODEL` — the diffusion model
- `CLIP` — the text encoder; some models also use it for image understanding
- `VAE` — the image encoder/decoder

The files are downloaded only when you run the node for the first time. Running
the same node again in the same Colab session reuses the verified files.

| Loader node | Default/typical bundle | Approximate download | Notes |
| --- | --- | ---: | --- |
| Z-Image Turbo | Q4 | 7.8 GB | Fast image generation |
| Qwen Image Edit 2511 | Q4 | 22.9 GB | Image editing and reference conditioning |
| Krea 2 Turbo | FP8 | 18.6 GB | Krea recommends 8 steps with no CFG |
| FLUX.2 Klein 4B | Q4 | 11.0 GB | 4 steps, guidance 1 |
| FLUX.2 Klein 9B | Q4 | 14.9 GB | 4 steps, guidance 1 |
| FLUX.2 Dev | Q4 | 38.5 GB | Higher quality, much larger download |

### Z-Image Turbo

Choose `Q4_K_M`, `Q3_K_M`, `Q5_K_M`, or `Q8_0`. The default Q4 bundle uses a
Qwen3-4B text encoder and the official Z-Image VAE.

Enable `force_redownload` only when you want to discard the cached temporary
files and download them again.

### Qwen Image Edit 2511

Choose a GGUF quantization and connect the outputs to ComfyUI's native Qwen Image
Edit conditioning nodes. Typical workflows use `TextEncodeQwenImageEditPlus`
with the model's `index_timestep_zero` reference method. The model weights are
Apache-2.0.

### Krea 2

Use `Turbo FP8` for normal generation. `Raw FP8` is the training/base checkpoint
and is not Krea's recommended inference model.

Krea 2 uses the
[`Comfy-Org/Krea-2` community license](https://huggingface.co/Comfy-Org/Krea-2).
Review it before downloading or deploying the weights; it includes commercial-use
and deployment obligations.

### FLUX.2 Klein 4B

The official small model is **Klein 4B**—there is no released Klein 3B. Choose a
GGUF quantization and use 4 steps with guidance/CFG 1. The 4B weights are
Apache-2.0.

### FLUX.2 Klein 9B

Choose a GGUF quantization and use 4 steps with guidance/CFG 1. The 9B weights
use the BFL FLUX Non-Commercial License and its acceptable-use requirements.

### FLUX.2 Dev

This is the largest bundle. BFL recommends guidance 4 and 50 steps; 28 steps is
a practical speed/quality compromise. FLUX.2 Dev uses the BFL FLUX
Non-Commercial License.

## Simple 3D nodes

After updating an existing runtime, start with:

```bash
comfycolab start --refresh
```

In ComfyUI, search for:

- **ComfyColab TRELLIS.2 — Image to 3D**
- **ComfyColab UltraShape — Refine Geometry**

Both output a native GLB `File3D`, so their result connects directly to
**Preview 3D & Animation** and **Save 3D Model**. The two facades are the only
normal-search ComfyColab 3D nodes; their adapters remain development-only.

### TRELLIS.2 — Image to 3D

Connect an image, choose a quality preset, and run the workflow. `1024 —
Quality` is the default. `512 — Fast` is the safest first test. `1536 —
Maximum` is expensive and never silently falls back to a smaller shape.

When strict 1536 needs more than `max_tokens`, the node stops with the required
token count. Raise the cap if the runtime has room, or manually choose 1024;
ComfyColab does not rerun at 1408, 1280, 1152, or 1024 behind your back.

`remove_background=Auto` uses the pinned TRELLIS BiRefNet node. `Off` supplies
an all-foreground mask. The advanced TRELLIS inputs use `0` for their preset
value, except `max_tokens`, whose visible default is `49152`.

### UltraShape — Refine Geometry

Connect a native GLB from TRELLIS (or another File3D-producing node), connect
the original reference image, and choose `Fast`, `Detailed`, or `Ultra`.
UltraShape runs in its own process group but deliberately reuses the cached
`trellis2-nodes` Python/CUDA environment. Cancellation terminates that process
group and removes incomplete outputs.

The first UltraShape run downloads two temporary, revision-pinned model sets:

| Artifact | Approximate size | Verification |
| --- | ---: | --- |
| UltraShape `ultrashape_v1.pt` | 7.37 GB | exact size + SHA-256 |
| DINOv2 Large inference files | 1.22 GB | exact size + per-file SHA-256 |

Downloads resume `.partial` files, retry stalls, and report percentage, speed,
and ETA through ComfyUI progress. They live under
`/content/.comfycolab/models/3d` and disappear with the Colab runtime.

With `retexture=true`, the refined geometry is normalized for TRELLIS encoding,
textured and rasterized in that same normalized space, then restored to the
input GLB's orientation, position, and scale. With `retexture=false`, the node
returns restored geometry with a neutral material and records it as
geometry-only in its sidecar.

`Detailed` and `Ultra` use provisional 1024 octree presets. Their release gate
is two successful full G4 runs without OOM; current evidence is tracked in
[`docs/3d-validation.md`](docs/3d-validation.md). Hostile local scenarios and
failure-cleanup evidence are tracked separately in
[`docs/3d-ultraqa.md`](docs/3d-ultraqa.md).

### Temporary 3D cache

Pipeline results are stored under:

```text
/content/.comfycolab/cache/3d/
  trellis/<key>/model.glb
  ultrashape/<key>/geometry.glb + transform.json + record.json
  texture/<key>/model.glb
```

`Use cache` reuses validated artifacts, `Refresh this node` recomputes all
stages owned by that facade, and `Disable cache` performs no result-cache reads
or writes. Final GLBs are published under `/content/ComfyUI/output/3d` for the
normal ComfyUI preview/download path.

## Advanced TRELLIS.2 nodes

ComfyColab installs the pinned
[`PozzettiAndrea/ComfyUI-TRELLIS2`](https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2)
node suite in an isolated `comfy-env` environment. TRELLIS.2 is a separate 3D
pipeline, so it does not appear under `ComfyColab / loaders` and does not return
the usual `MODEL`, `CLIP`, and `VAE` outputs.

In ComfyUI, search for the `TRELLIS2` category and add **(Down)Load TRELLIS.2
Models**. The upstream `geometry_texture.json` workflow included with the node
shows the full path from a single input image to a PBR-textured mesh and GLB
export.

Important notes:

- The first model-node execution downloads roughly 17–18 GB into temporary
  Colab storage. Later runs in the same session reuse those files.
- The default G4 bootstrap downloads a separate 5.02 GB prebuilt environment
  cache in three parallel parts. This replaces the much slower dependency
  resolution step; it does not contain model weights. Bootstrap prints the
  aggregate percentage, rolling download speed, and ETA every five seconds.
  A part that receives no data for 30 seconds is retried automatically up to
  five times, resuming the verified partial download when GitHub supports it.
- Start at `512` resolution. Test `1024_cascade` only after the 512 workflow is
  stable. The simple facade does not remove or replace these advanced nodes.
- Microsoft officially requires Linux, CUDA, and at least 24 GB VRAM. The G4's
  96 GB is sufficient, but its Blackwell CUDA architecture is outside
  Microsoft's officially tested A100/H100 set. All cached native extensions and
  the ComfyUI nodes pass a live G4 import/CUDA smoke test; full model generation
  remains experimental until the 512-to-GLB workflow is exercised end to end.
- Microsoft's official pipeline is single-image. The selected wrapper also
  provides an experimental `TRELLIS.2 Multi-View Image to Shape` node for up to
  six views; that spatial-blending implementation is a wrapper extension, not an
  official Microsoft capability claim.
- TRELLIS.2-4B is MIT-licensed. DINOv3 conditioning has a separate upstream
  license; the selected wrapper currently uses a public compatible mirror for
  that encoder.

Generated `.glb` files are written under `/content/ComfyUI/output` and disappear
with the rest of the runtime unless you download them.

### 3D environment cache

The rollback G4 cache profile is recorded in
[`cache/trellis2-g4-v1.json`](cache/trellis2-g4-v1.json). The combined profile
state is recorded in [`cache/3d-g4-v2.json`](cache/3d-g4-v2.json). They are pinned to Linux
x86-64, Python 3.12.13, PyTorch 2.11.0 + CUDA 12.8, Blackwell compute capability
12.0, exact source revisions, and exact patch IDs. Bootstrap verifies every
archive part before extraction, imports all native CUDA modules, and runs a CUDA
tensor probe. A combined-cache mismatch falls back to the existing TRELLIS.2
cache plus the minimal UltraShape inference overlay; a base-cache mismatch
falls back to the normal upstream installer.

No archive or model weights are committed to Git. Release parts are created on
the matching G4 with `scripts/build_3d_cache.py` only after the live validation
gates pass. Third-party licensing and territory notices are summarized in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Temporary storage

ComfyColab deliberately does not mount Google Drive.

- Models live under `/content/ComfyUI/models`.
- Generated images live under `/content/ComfyUI/output`.
- Both disappear when the runtime is stopped or reclaimed by Colab.

Download any generated images you want to keep before running
`comfycolab stop`.

## About the Google Colab proxy option

The current `google-colab-cli` executes Python through a separate Jupyter
connection. That connection cannot complete Colab frontend-only `eval_js()` /
`proxyPort()` requests, so a terminal-only Google proxy URL is not reliable.

For now, use the Cloudflare URL printed by `comfycolab start`. The experimental
`--colab-proxy` flag may fall back to Cloudflare and should not be relied upon.
This limitation is unrelated to whether you use Safari or Chrome.

## Configuration

The lightweight Mac launcher uses environment variables:

```bash
# Use a different Colab accelerator
COMFYCOLAB_GPU=T4 comfycolab start

# Use a different session name
COMFYCOLAB_SESSION=my-comfy comfycolab start

# Use the Colab CLI OAuth flow instead of ADC
COMFYCOLAB_AUTH=oauth2 comfycolab start

# Point to an existing colab executable
COMFYCOLAB_COLAB_BIN=/full/path/to/colab comfycolab start
```

Supported accelerator names are `G4`, `T4`, `L4`, `A100`, and `H100`. Colab may
still refuse an accelerator when it is unavailable or your account lacks access.

## Python development installation

The one-line Mac installer is recommended for normal use. For repository
development, install the Python package in a virtual environment:

```bash
git clone https://github.com/DragonLord1998/ComfyColab.git
cd ComfyColab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Requirements are Python 3.10 or newer, Git, and a Google account with Colab
access. The package installs the official `google-colab-cli` dependency.

## Model catalog

The curated catalogs live in
[`custom_nodes/ComfyColab-ZImage/catalog`](custom_nodes/ComfyColab-ZImage/catalog).
Each entry records a revision-pinned Hugging Face URL, filename, expected size,
and SHA-256 checksum.

Current upstream sources include:

- `jayn7/Z-Image-Turbo-GGUF`
- `unsloth/Qwen3-4B-GGUF`
- `Comfy-Org/z_image_turbo`
- `unsloth/Qwen-Image-Edit-2511-GGUF`
- `Comfy-Org/Qwen-Image_ComfyUI`
- `Comfy-Org/Krea-2`
- `unsloth/FLUX.2-klein-4B-GGUF`
- `unsloth/FLUX.2-klein-9B-GGUF`
- `Comfy-Org/vae-text-encorder-for-flux-klein-*`
- `city96/FLUX.2-dev-gguf`
- `Comfy-Org/flux2-dev`

The repository does not redistribute model weights. Downloaded files retain
their upstream licenses.

## Verify locally

Run the local test suite without allocating a Colab runtime or downloading model
weights:

```bash
bash scripts/check.sh
```

Before publishing catalog changes, verify the pinned Hugging Face metadata:

```bash
python scripts/verify_catalogs.py
```

## Troubleshooting

### `comfycolab: command not found`

Add the local binary directory to your shell path and open a new terminal:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### The URL stopped working

Cloudflare quick-tunnel URLs are temporary. Check whether the runtime still
exists:

```bash
comfycolab status
```

If the session was reclaimed, run `comfycolab start` again. If it is still
active but the UI needs rebuilding, use `comfycolab start --refresh`.

### A model download is slow

Large bundles can take time, especially FLUX.2 Dev. The model download happens
inside Colab and does not travel through Cloudflare. Keep the terminal and
runtime alive while the node is downloading.

### A generated image downloads slowly

Browser downloads do pass through the Cloudflare quick tunnel. The native Colab
proxy cannot currently be obtained reliably through `google-colab-cli`; this is
a known architectural limitation of the terminal-only workflow.
