# NVIDIA PiD upscaler

`ComfyColab PiD — Image Upscaler` wraps the native NVIDIA PiD support in the
pinned ComfyUI build. It accepts a ComfyUI `IMAGE`, downloads the selected
compatible VAE, PiD decoder, and PixelDiT text encoder on first use, and returns
an `IMAGE`.

## Inputs

- `image`: any still image. PiD runs at aligned internal dimensions and the
  facade resizes back to an exact 4x or 16x result for unusual source sizes.
- `vae_family`: `FLUX.1`, `FLUX.2`, or `Qwen Image`. This chooses a
  matched VAE and PiD checkpoint; arbitrary VAE files are not interchangeable
  because PiD is trained for a specific latent format.
- `prompt`: a short description of the source image. PiD uses it while
  synthesizing high-resolution detail.
- `scale`: `4x` or `Experimental 16x (tiled)`.
- `seed` and `degrade_sigma`: PiD sampling controls. Keep `degrade_sigma=0`
  for a clean source image.
- `tile_size` and `tile_overlap`: advanced controls for the second PiD pass in
  experimental 16x mode.
- `accept_nvidia_noncommercial_license`: must be enabled before the node
  downloads or runs the NVIDIA PiD weights.

## How 16x works

PiD checkpoints are native 4x decoders. The experimental mode therefore
cascades two 4x passes. The second pass uses ComfyUI context-window sampling
with overlap and tiled VAE encoding to control peak VRAM. This is much slower
than 4x, may create seams or invented detail, and can produce extremely large
images. For example, a 512x512 input becomes 8192x8192.

The default four-step distilled schedule is:

```text
0.999, 0.866, 0.634, 0.342, 0
```

## Model and license sources

- PiD code and research: <https://github.com/nv-tlabs/PiD>
- ComfyUI-native PiD weights: <https://huggingface.co/Comfy-Org/PixelDiT>
- Original NVIDIA weights and terms: <https://huggingface.co/nvidia/PiD>

The ComfyUI repackaged PiD weights are labeled `NSCLv1`. Review the upstream
license before enabling the acceptance input. ComfyColab does not change those
terms.
