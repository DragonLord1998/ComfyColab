# ComfyColab LTX Video

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
