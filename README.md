# ComfyColab

ComfyColab is the Colab engine for pinned ComfyUI installations. Output-domain
nodes, workflows, model catalogs, environments, and optimizations are owned by
independently versioned daughter repositories:

- `ComfyColab-Image`
- `ComfyColab-Video`
- `ComfyColab-3D`
- `ComfyColab-3DGS`
- `ComfyColab-WM` (World Model)

Core owns Colab session transport, authenticated bootstrap, ComfyUI lifecycle,
immutable pack resolution, runtime state, endpoints, and model-agnostic engine
optimizations. It defaults to installing ComfyUI with no daughter pack.

The five daughter repositories are public and their exact commits and manifest
digests are recorded in `registry/published-packs.json`. This core branch remains
an unreleased `0.2.0.dev1` candidate; GitHub `main` and its installer
still represent the legacy release. The user-selectable official pack registry
remains empty until each pack passes its clean-lock Colab gate. Legacy
all-in-one sources remain in this checkout until the matching daughter is proven
installable and rollback-safe. The public full-node notebook uses the explicit
`legacy-full` compatibility runtime in the meantime: it verifies and links the
exact Image, Video, 3D, and 3DGS daughter commits while reusing the previously
working full dependency bootstrap. See
[the migration status](docs/modularization-status.md).

Everything inside `/content` disappears when the Colab runtime is released.

## Quick start on Mac after 0.2 publication

Prerequisites:

- Python 3.12 or newer (`python3 --version`)
- the official `colab` command, authenticated and able to create a session

ComfyColab reuses the Colab CLI's Google authentication. The installer creates
an isolated environment under `~/.local/share/comfycolab/venv`, links the
command into `~/.local/bin`, and does not replace your Colab login setup. If
your supported Python has a versioned command, set `COMFYCOLAB_PYTHON` to it.

Install the command once:

```bash
curl -fsSL https://raw.githubusercontent.com/DragonLord1998/ComfyColab/main/install.sh | sh
```

Start ComfyUI:

```bash
comfycolab start
```

The launcher resolves the selected profile into an immutable lock before it
allocates or mutates a Colab runtime. Authenticated stage 0 verifies the exact
core commit and stage-1 digest; stage 1 installs the locked ComfyUI revision.
When everything is ready, the terminal prints:

```text
ComfyUI: https://example.trycloudflare.com
Session: comfycolab
```

Open the `ComfyUI` link in Safari, Chrome, or another browser.

## Core and pack commands

Core-only start becomes the public default with the 0.2 release. Pack aliases
and the generic `legacy-full` start remain unavailable until the corresponding
published commits pass their runtime gates and are promoted into the
authenticated official registry. The explicit notebook-only `--legacy-full`
compatibility path is available now.

```bash
# Core-only
comfycolab start

# After runtime promotion, official pack aliases can be composed
comfycolab start --pack image --pack video

# Current full-node notebook compatibility path
comfycolab notebook --profile legacy-full --legacy-full \
  --accept-license accept_research_license \
  --output ComfyColab-Full.ipynb

# After all five packs pass their generic-runtime gates
comfycolab start --profile legacy-full
comfycolab pack resolve --pack image --pack video
comfycolab pack doctor
comfycolab pack rollback

# Deterministic two-cell notebook with the same embedded lock
comfycolab notebook --pack image --output ComfyColab-Image.ipynb

# Lifecycle
comfycolab status
comfycolab url
comfycolab start --refresh
comfycolab stop
```

`--refresh` reuses the saved lock and never silently advances core, ComfyUI, or
pack versions. `comfycolab pack update` is the explicit version-resolution
operation. When an update changes the lock, the prior canonical lock is retained
for `comfycolab pack rollback`. That command restores lock selection only; until
fresh-environment rollback is proven, stop and recreate the Colab runtime before
applying the restored lock.

`registry/published-packs.json` authenticates the five public daughter commits
for contract CI. `registry/official-packs.json` is intentionally empty until a
pack also passes clean dependency installation, ComfyUI startup, node discovery,
its accelerator smoke test, and rollback. This prevents an apparently valid
alias from resolving to a pack that is public but not yet runnable.

`--legacy-full` is deliberately limited to notebook rendering. It does not
promote the daughter refs into `registry/official-packs.json`; it selects the
immutable refs embedded in `profiles/legacy-full.json`, verifies their manifest
digests inside Colab, and links their declared node roots. `ComfyColab-WM` is
excluded because it does not contain nodes yet. CubePart's source and weights
carry research-only terms. The published notebook is rendered with
`--accept-license accept_research_license` at the repository owner's explicit
direction; review and accept those upstream terms before running Cell 2. The
CubePart node still requires its per-request acceptance checkbox.

Always run `comfycolab stop` when you are finished so the Colab runtime is not
left consuming compute units.

## Authenticated launch flow

1. Resolve the authenticated core profile and selected pack manifests.
2. Reject dependency, environment, runtime-variable, destination, and patch
   conflicts before runtime mutation.
3. Persist canonical lock bytes under
   `~/.config/comfycolab/locks/<session>.lock.json`.
4. Create or reuse the requested Colab session.
5. Stage 0 clones the exact core commit and verifies the stage-1 file digest.
6. Stage 1 installs exact locked sources, links declared node roots, runs
   offline hooks and health checks, and starts ComfyUI.
7. The CLI verifies that the readiness payload reports the same lock digest.
8. ComfyColab prints the browser URL.

Cloudflare only carries the browser traffic. Model files are downloaded directly
from Hugging Face to the Colab VM.

## Legacy domain documentation

The sections below document the pre-split node packs. They remain temporarily
for behavioral parity and will be removed from core one pack at a time only
after exact-lock installation, rollback, and live validation succeed.

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
Transient Hugging Face failures (including stale signed-link `403` responses,
rate limits, and server errors) are retried up to five times with backoff. A
partial file is kept and resumed; if a stale Colab `HF_TOKEN` causes `401` or
`403`, the public bundle is retried anonymously. `force_redownload` is the only
option that deliberately discards resumable partial data.

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

## LTX-2.3 video

After updating an existing runtime, run:

```bash
comfycolab start --refresh
```

In ComfyUI, search for:

```text
ComfyColab LTX-2.3 — Text/Image to Video
```

The facade follows Lighttricks' current direct DistilledPipeline two-stage
design and returns a native `VIDEO`, decoded `IMAGE` frames, and synchronized
`AUDIO`. It generates the audio/video latent together at 24 FPS. If spatial
upscaling is enabled, it upsamples the video latent and runs the short
refinement stage. Choosing 48 FPS then applies the official temporal x2 latent
upscaler before decoding; audio does not pass through the temporal video
upscaler.

Inputs:

- `prompt`; the direct distilled pipeline is positive-only and has no
  negative-prompt branch
- optional first-frame `image`; leave it disconnected for text-to-video
- `gguf_model`: `Q3_K_S`, `Q4_K_S`, or `Q4_K_M`
- `fps`: `24` or temporally upscaled `48`
- `spatial_upscaler`: `None`, `1.5x`, or `2x`; exact 1.5x output requires
  base width and height divisible by 64
- base `width`, `height`, `frame_count`, `seed`, and `image_strength`

`Q3_K_S` is the lowest-memory default. Because the GGUF quantizes the direct
LTX-2.3 Distilled 1.1 checkpoint, the node uses Lighttricks' current
positive-only Euler schedules and does not stack the older ComfyUI workflow's
distilled LoRA or CFG++ path on top. `2x` selects the latest v1.1
spatial-upscaler hotfix; `1.5x` and temporal `2x` select their latest v1.0
releases.

The selected assets download on the first queue and are checksum verified.
The default Q3 / 24 FPS / 2x bundle is approximately 22.3 GB. Selecting 48 FPS
adds about 0.26 GB; the Q4 diffusion models add roughly 3.2–4.5 GB over Q3.
All files remain in temporary Colab storage.

An importable text/image-to-video starter is included at
[`workflows/comfycolab_ltx23_text_image_to_video.json`](workflows/comfycolab_ltx23_text_image_to_video.json).
It queues as text-to-video by default; connect its **Load Image** output to the
facade's optional `image` input for image-to-video.

Lighttricks lists 32 GB+ VRAM and 100 GB+ free disk as prerequisites for its
ComfyUI workflows. Local tests verify the facade schema, selected downloads,
graph branches, bootstrap pin, and workflow wiring; a real live Colab run is
still required to establish runtime memory use, audiovisual synchronization,
and output quality on the chosen accelerator.

## Simple 3D nodes

After updating an existing runtime, start with:

```bash
comfycolab start --refresh
```

In ComfyUI, right-click the canvas and look under:

```text
ComfyColab / 3D
```

or search for:

- **ComfyColab TRELLIS.2 — Image to 3D**
- **ComfyColab TRELLIS2MV — Multi-View to 3D**
- **ComfyColab UltraShape — Refine Geometry**
- **ComfyColab TripoSplat — Image to Gaussian Splat**
- **ComfyColab Pixal3D — Image to 3D**
- **ComfyColab Pixal3DMV (Experimental) — Multi-View to 3D**
- **ComfyColab SkinTokens — Auto Rig 3D**
- **ComfyColab CubePart — Segment 3D Parts**

The mesh nodes output native GLB `File3D` results, so they connect directly to
**Preview 3D & Animation** and **Save 3D Model**. TripoSplat outputs a native
ComfyUI `SPLAT` plus a splat `FILE_3D`. Worker and graph adapters remain
development-only.

### TRELLIS.2 — Image to 3D

Connect an image, choose a quality preset, and run the workflow. `1024 —
Quality` is the default. `512 — Fast` is the safest first test. `1536 —
Maximum` is expensive and never silently falls back to a smaller shape.

The visible facade reports coarse, truthful stages while its expanded TRELLIS
nodes execute: preparing models and input, generating shape, building the
geometry preview, generating texture, and baking the final GLB. When its output
is connected to **Preview 3D & Animation** or **Preview 3D (Advanced)**, that
viewer receives an early neutral-gray geometry preview as soon as the processed
shape exists. The final textured model replaces it after PBR baking completes.
The early preview is not a continuously updating mesh; it is the first valid
geometry checkpoint.

Generated geometry is checked after raw shape decoding, after the pinned
upstream-parity remesh pass, and again after GLB export. A structurally valid
but planar/collapsed mesh now stops with its stage, PCA rank, and singular-value
ratio instead of reporting `Complete` or entering the result cache. GLB export
also applies the upstream Z-up-to-Y-up and texture-V conversion together.

When strict 1536 needs more than `max_tokens`, the node stops with the required
token count. Raise the cap if the runtime has room, or manually choose 1024;
ComfyColab does not rerun at 1408, 1280, 1152, or 1024 behind your back.

`remove_background=Auto` uses the pinned TRELLIS BiRefNet node. `Off` supplies
an all-foreground mask. The advanced TRELLIS inputs use `0` for their preset
value, except `max_tokens`, whose visible default is `49152`.

### TRELLIS2MV — Multi-View to 3D

Connect four labeled horizontal views in this exact order: front, back, left,
and right. Top and bottom are an optional pair, so the node accepts either four
or six views. Geometry uses the pinned community
`Trellis2MultiViewImageToShape` implementation; PBR texturing uses the front
reference because the wrapper's released texture stage remains single-view.

ComfyColab applies a revision-checked efficiency patch that computes the
directional spatial blend weights once per diffusion run instead of rebuilding
the same 3D weights at every step. The active cameras, softmax math, and model
predictions are unchanged. This multiview sampler is a community extension, not
an official Microsoft TRELLIS.2 capability claim.

### TripoSplat — Image to Gaussian Splat

Connect one ComfyUI `IMAGE`, choose a quality preset, and run the node. The
presets are `Fast — 65K`, `Balanced — 131K`, and `Quality — 262K`. The default
is `Quality — 262K`; `Fast — 65K` is the safest first live test.

The `SPLAT` output is ComfyUI's native in-memory Gaussian splat output. The
`model_3d` output is a native splat `FILE_3D` written as `ply`, `spz`, or
`ksplat`. Use `ply` for the broadest Gaussian-splat interoperability and full
spherical-harmonic data, `spz` for a compact splat file, and `ksplat` for
viewers that expect the KIRI/ksplat format.

TripoSplat uses native ComfyUI TripoSplat support from ComfyUI v0.23.0+ and the
ComfyUI revision pinned by ComfyColab. The first uncached run downloads about
3.78 GB of public model files from the official `VAST-AI/TripoSplat` Hugging
Face repository into the runtime's normal ComfyUI model folders; no API token
is required for those public assets. The official project is
[`VAST-AI-Research/TripoSplat`](https://github.com/VAST-AI-Research/TripoSplat)
and the official model repository is
[`VAST-AI/TripoSplat`](https://huggingface.co/VAST-AI/TripoSplat). The official
TripoSplat source and weights are MIT licensed.

Local contract and workflow tests can verify the node schema, graph expansion,
download logic, presets, and file-format choices. They do not prove live Colab
GPU execution, runtime memory behavior, or output quality; those claims require
a real live G4 run that produces and validates a non-empty splat artifact.

### Pixal3D — Image to 3D

Connect one image and run the public facade. This is intentionally
single-image-only: there is no mode selector and no `num_views` control. The
quality choices are exactly `1024 — Stable` and `1536 — Experimental`; the 1536
tier is a live-GPU release gate and is not locally proven.

Pixal3D runs in an isolated hidden worker environment instead of importing its
runtime into the main ComfyUI process. The official source is
`TencentARC/Pixal3D` pinned to
`cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af`, with pinned runtime companions for
DINOv3, MoGe, NAF, `utils3d`, and NATTEN. The first uncached run can download
and build several multi-GB model/runtime components, so expect a long first
download, transient Hugging Face failures, and source-build risk until a ready
Pixal3D worker cache is published and validated.

`keep_worker_loaded=true` keeps the hidden Pixal3D process resident between
requests in the same runtime, which speeds repeated prompts but retains more
GPU/CPU memory. Turn it off for one-shot validation or after memory pressure.
The result cache follows the normal 3D cache controls: `Use cache` reuses a
validated GLB for the same image/settings/source revisions, `Refresh this node`
recomputes and overwrites the Pixal3D result, and `Disable cache` avoids result
cache reads and writes. Every Pixal3D live G4 gate is still pending; the local
workflow and contract tests do not prove model execution or output quality.

### Pixal3DMV — Experimental Multi-View to 3D

Pixal3DMV accepts the same four required labeled views and optional top/bottom
pair as TRELLIS2MV. It is an explicit experimental ComfyColab adapter, not an
official Pixal3D mode. Inspired by ReconViaGen's multiview reconstruction
strategy, it runs Pixal3D's own projection conditioners for each canonical
camera and fuses view-aligned projected features before the existing Pixal3D
samplers. It does not make a contact sheet, average output meshes, or pretend a
batch of unrelated single-image runs is multiview inference.

`Directional projection` uses spatial softmax weights so each 3D location
favors the nearest labeled camera; `Average projection` is a diagnostic equal
blend. Both modes average global image features and preserve the official
single-view tensor contracts downstream. This zero-shot adapter has local
contract tests only and remains pending a real G4 quality/VRAM validation.

### Pixal3DMV — Advanced Weighted Multi-View to 3D

The advanced Pixal3DMV facade keeps the same six-view labeled contract but adds
per-view quality weights so stronger views can dominate the fusion pass. Use it
when the source set is uneven, such as a Flux-generated multiview batch where
some angles are noticeably cleaner than others. The node is still an
experimental adapter, not official Pixal3D multiview support, and it shares the
same live G4 validation requirement as the base multiview node.

### SkinTokens — Auto Rig 3D

Connect a GLB and run **ComfyColab SkinTokens — Auto Rig 3D** to generate a
skeleton hierarchy and dense per-vertex skin weights with the pinned
SkinTokens/TokenRig release. `preserve_texture=true` enables the upstream
transfer path, while `use_postprocess` opts into its voxel skin cleanup. The
first run provisions a separate CUDA worker environment and model artifacts;
the documented upstream minimum is an NVIDIA GPU with at least 14 GB VRAM.

### CubePart — Segment 3D Parts

CubePart is schema-conditioned decomposition: provide an ordered comma- or
newline-separated list such as `body, wheel, handle`. It returns a combined
colored GLB, a persistent directory containing one GLB per generated part, and
a JSON manifest. It does not perform unlabeled segment-anything inference.

The source and weights carry research-oriented RAIL terms, so the node refuses
to provision artifacts or run until `accept_research_license=true`. Review the
upstream terms first; that switch records acceptance for the request but does
not change or bypass the license.

### UltraShape — Refine Geometry

Connect a native GLB from TRELLIS (or another File3D-producing node), connect
the original reference image, and choose `Fast`, `Conservative`, `Detailed`,
or `Ultra`. `Fast` keeps its existing quick 512 settings. The public
`Conservative` default uses 24 steps at 512. `Detailed` and `Ultra` preserve
their existing 1024 behavior and are explicitly experimental; the required
live 512 and 1024 release runs remain pending.
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

Planar input is rejected before model provisioning or worker launch. If the
adaptive decoder finds no candidate surface points, the worker returns a
`NoDecodableSurface` error with the requested resolution, decode-stage
resolution, preceding active-point count, and seed; it does not continue with
an empty grid or leave a partial/cache artifact.

`Conservative` is the public 512 octree preset. `Detailed` and `Ultra` retain
provisional 1024 behavior for saved-workflow compatibility; their release gate
is two successful full G4 runs without OOM. Current evidence is tracked in
[`docs/3d-validation.md`](docs/3d-validation.md). Hostile local scenarios and
failure-cleanup evidence are tracked separately in
[`docs/3d-ultraqa.md`](docs/3d-ultraqa.md).

### Temporary 3D cache

Pipeline results are stored under:

```text
/content/.comfycolab/cache/3d/
  trellis/<key>/model.glb
  trellis-multiview/<key>/model.glb
  pixal3d/<key>/model.glb
  skintokens/<key>/model.glb + metadata.json
  cubepart/<key>/parts.glb + part GLBs + manifest.json
  ultrashape/<key>/geometry.glb + transform.json + record.json
  texture/<key>/model.glb
```

`Use cache` reuses structurally and volumetrically validated artifacts,
`Refresh this node` recomputes all
stages owned by that facade, and `Disable cache` performs no result-cache reads
or writes. Final GLBs are published under `/content/ComfyUI/output/3d` for the
normal ComfyUI preview/download path.

The geometry/export repair uses new TRELLIS, UltraShape, and texture result
schema versions in every key/record. Pre-fix cache entries therefore cannot be
returned; a planar entry encountered under a current key is deleted and
regenerated.

## Advanced TRELLIS.2 nodes

ComfyColab installs the pinned
[`PozzettiAndrea/ComfyUI-TRELLIS2`](https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2)
node suite in an isolated `comfy-env` environment. TRELLIS.2 is a separate 3D
pipeline, so it does not appear under `ComfyColab / loaders` and does not return
the usual `MODEL`, `CLIP`, and `VAE` outputs.

In ComfyUI, search the `TRELLIS2 / Advanced` category and add **(Down)Load
TRELLIS.2 Models**. The upstream `geometry_texture.json` workflow included with
the node shows the full path from a single input image to a PBR-textured mesh
and GLB export. Existing node IDs and workflow contracts are unchanged; only
their search category moved.

Important notes:

- The first model-node execution downloads roughly 17–18 GB into temporary
  Colab storage. Later runs in the same session reuse those files. ComfyColab
  raises the pinned `comfy-env` isolated-call limit from 10 minutes to 2 hours
  for its server process, so a healthy first download is not killed mid-transfer.
  The patch is version/source checked and does not enable force-download or
  delete Hugging Face's resumable cache.
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
  Microsoft's officially tested A100/H100 set. The facade has produced and
  validated textured GLBs at genuine 512 and 1024-cascade resolutions on a live
  G4, and the preserved advanced workflow also passes. A genuine 1536 run is
  still a release gate; the recorded attempt lost its Colab backend during the
  high-density tiled stage before a GLB was produced.
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

Generated two-cell notebooks use the authenticated Colab proxy as the primary
access path. Cell 1 reserves the session-bound proxy and Cell 2 starts ComfyUI
with the matching CORS origin, probes the `/system_stats` transport, and
requires a successful `/ws` handshake. Cell 2 prints the Colab proxy URL first,
then embeds ComfyUI with `serve_kernel_port_as_iframe()`. The raw proxy URL is
valid only for the current signed-in user while that notebook/runtime remains
open; browser security may still reject it in a separate tab, so the embedded
view is shown immediately afterward. A Cloudflare URL is printed last as the
fallback.

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
python -m pip install -e ".[test]"
```

Requirements are Python 3.12 or newer, Git, and a Google account with Colab
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
