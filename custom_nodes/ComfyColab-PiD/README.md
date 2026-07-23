# ComfyColab PiD

`ComfyColab PiD — Image Upscaler` is one public ComfyUI facade over the native
PiD/PixelDiT nodes in ComfyColab's pinned ComfyUI build.

- 4x: one distilled PiD pass.
- Experimental 16x (tiled): two 4x passes, with context-window sampling and
  tiled VAE encoding on the second pass.
- VAE families: FLUX.1, FLUX.2, Qwen Image, and experimental Mage-VAE.

FLUX.1, FLUX.2, and Qwen Image use matched VAE/PiD pairs. Mage-VAE is an
experimental bridge: its unscaled 128-channel, 16x-downsampled latent is sent to
the FLUX.2 PiD checkpoint because NVIDIA has not published a Mage-VAE-specific
PiD checkpoint. Mage-VAE encoding runs in the isolated Mage worker and supports
the node's tiled 16x path. The node also uses the PixelDiT Gemma text encoder and
ComfyUI's `pixel_space` VAE for output decoding. Model downloads remain blocked
until the NVIDIA noncommercial license checkbox is explicitly enabled.

See [`docs/pid-upscaler.md`](../../docs/pid-upscaler.md) for model sources,
limits, and licensing.
