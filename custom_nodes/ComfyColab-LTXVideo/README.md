# ComfyColab Video

## MiniMax H3

`MiniMax H3 Bundle Loader` downloads the selected FL2VA or Ref2VA model and
the shared text encoder and VAEs. The loaded H3 diffusion model is cloned and
patched with ComfyUI's registered SageAttention backend, so the optimization
applies only to H3 sampling. The runtime uses CUDA 13 with PyTorch 2.11 and
builds SageAttention 2.2.0 from an immutable commit in the official repository.

The loader asks only for MiniMax H3 Community License acknowledgement. Regional
availability is verified by the user; the node performs no region, country,
IP, or geolocation check.

`ComfyColab MiniMax H3 — Prompt Enhancer` accepts a plain-language prompt and
returns one `STRING` for the prompt input of either H3 generator. Select T2VA,
I2VA, FL2VA, L2VA, or Ref2VA and use the same duration on the enhancer and H3
node. The included workflows place the enhancer before the H3 prompt input.

The enhancer runs the official `Qwen/Qwen3.8-27B` model through Unsloth's
immutable `Qwen3.8-27B-Q4_K_M.gguf` (17.1 GB) and pinned llama.cpp `b10437`.
Thinking is enabled and returned separately; only the schema-constrained final
rewrite reaches H3. The rewrite is checked against MiniMax's exact Base or
six-section Ref2VA contract and retried once when validation fails. The isolated
llama.cpp process then exits so its GPU allocation is released before H3
sampling. Required thinking is bounded inside the total completion budget so it
cannot consume the final structured answer. A fresh G4 restores the pinned
CUDA 12.8/SM120 llama.cpp server from the checksum-verified
[GitHub cache release](https://github.com/DragonLord1998/ComfyColab/releases/tag/llama-cpp-b10437-cu128-sm120-v1);
only a failed or incompatible restore falls back to building the immutable
llama.cpp source. The GGUF remains separately downloaded and verified.

The compact system policy is pinned to MiniMax's official guide commit
`d21241f0a4b3acbb34c97dae47fa417b7065e438`, including the exact field names,
mode headers, retention markers, timing notation, and structural rules from its
[H3 prompt-writing skill](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing),
Base reference, and Ref2VA reference. User text is treated only as creative
source material and cannot replace the system policy.

## LTX-2.3

`ComfyColab LTX-2.3 — Text/Image to Video` is a single public facade over the
latest LTX-2.3 Distilled 1.1 pipeline. It downloads the selected community GGUF
plus pinned text, VAE, and upscaler assets into the temporary Colab runtime.

Inputs:

- `prompt` and optional first-frame `image`
- `gguf_model`: `Q3_K_S`, `Q4_K_S`, or `Q4_K_M`
- `fps`: `24` or temporally upscaled `48`
- `spatial_upscaler`: `None`, `1.5x`, or latest `2x` v1.1
- base width, base height, frame count, seed, and image-conditioning strength

The selected GGUFs quantize the direct Distilled 1.1 checkpoint, so the graph
uses Lighttricks' current positive-only Euler schedules rather than the older
development-checkpoint plus distilled-LoRA ComfyUI recipe. For exact 1.5x
output dimensions, both base dimensions must be divisible by 64.

The node returns a native ComfyUI `VIDEO`, decoded frames, and synchronized
audio. The GGUF conversion is community supplied; the spatial and temporal
upscalers are official Lighttricks assets. Local tests validate the graph and
download contracts, but live Colab inference is still required to prove runtime
memory use, audiovisual synchronization, and output quality.
