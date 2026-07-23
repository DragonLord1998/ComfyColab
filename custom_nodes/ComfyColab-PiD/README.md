# ComfyColab PiD

`ComfyColab PiD — Image Upscaler` is one public ComfyUI facade over the native
PiD/PixelDiT nodes in ComfyColab's pinned ComfyUI build.

- 4x: one distilled PiD pass.
- Experimental 16x (tiled): two 4x passes, with context-window sampling and
  tiled VAE encoding on the second pass.
- VAE families: FLUX.1, FLUX.2, and Qwen Image.

The selected VAE and PiD decoder are a matched pair. The node also uses the
PixelDiT Gemma text encoder and ComfyUI's `pixel_space` VAE for output decoding.
Model downloads remain blocked until the NVIDIA noncommercial license checkbox
is explicitly enabled.

See [`docs/pid-upscaler.md`](../../docs/pid-upscaler.md) for model sources,
limits, and licensing.
