# NVIDIA PiD upscaler

`ComfyColab PiD — Image Upscaler` wraps the native NVIDIA PiD support in the
pinned ComfyUI build. It accepts a ComfyUI `IMAGE`, downloads the selected
compatible VAE, PiD decoder, and PixelDiT text encoder on first use, and returns
an `IMAGE`.

## Inputs

- `image`: any still image. PiD runs on overlapping source-space tiles and
  merges them back to the exact requested output size.
- `vae_family`: `FLUX.1`, `FLUX.2`, `Qwen Image`, or
  `Mage-VAE (experimental)`. The first three choose a matched VAE and PiD
  checkpoint. The Mage option encodes with Microsoft Mage-VAE and feeds its
  128-channel, 16x-downsampled latent to the FLUX.2 PiD checkpoint.
- `prompt`: a short description of the source image. PiD uses it while
  synthesizing high-resolution detail.
- `scale`: `2x`, `4x`, `8x`, or `16x`.
- `seed` and `degrade_sigma`: PiD sampling controls. Keep `degrade_sigma=0`
  for a clean source image.
- `tile_size` and `tile_overlap`: control both the bounded PiD output tiles and
  the final Z-Image Turbo cleanup tiles.
- `accept_nvidia_noncommercial_license`: must be enabled before the node
  downloads or runs the NVIDIA PiD weights.

## Tiled scale pipeline

PiD checkpoints are native 4x decoders. The node splits the source into an
overlapped tile list before PiD sampling, processes every tile at a bounded
resolution, and merges the list with feathered weights. `2x` and `4x` use one
native PiD pass; `8x` and `16x` use two tiled 4x passes. The `2x` and `8x`
paths Lanczos-reduce each native PiD tile before merging, so they never assemble
an unnecessary full 4x or 16x intermediate canvas.

After the requested size is reached, the node splits the image into a second
tile list and applies a deliberately light Z-Image Turbo image-to-image pass:
AuraFlow shift 3, 5 steps, CFG 1, `dpmpp_2m_sde`, `beta`, and denoise 0.33.
This pass is intended to remove seams, ringing, and compression artifacts
without redesigning the source. Z-Image uses its own matched VAE for cleanup;
the default PiD conditioning VAE remains FLUX.2.

The old facade caps of 4096 px for 4x and 8192 px for 16x are removed. Each PiD
latent stays tile-sized; only the merged output image occupies the final canvas.
The pinned ComfyUI `ImageMergeTileList` node accepts a final canvas up to 32768
px per side, and the facade checks this before any model download. Very large
results still require enough system memory for that final image.

Mage-VAE uses the same isolated worker as the Mage-Flow nodes. Each tile is
encoded from deterministic posterior means before PiD conditioning.

## Mage-VAE compatibility

Mage-VAE is regularized toward FLUX.2 VAE latents and has the same 128-channel,
16x-downsampled tensor shape, so the native PiD conditioning node can consume
it through the FLUX.2 latent path. NVIDIA does not currently publish a PiD
checkpoint trained specifically for Mage-VAE, however. The option is therefore
an experimental cross-VAE bridge rather than a matched pair; output fidelity
may be lower than the standard FLUX.2 selection.

The default four-step distilled schedule is:

```text
0.999, 0.866, 0.634, 0.342, 0
```

## Model and license sources

- PiD code and research: <https://github.com/nv-tlabs/PiD>
- ComfyUI-native PiD weights: <https://huggingface.co/Comfy-Org/PixelDiT>
- Original NVIDIA weights and terms: <https://huggingface.co/nvidia/PiD>
- Microsoft Mage-Flow and Mage-VAE weights: <https://huggingface.co/microsoft/Mage-Flow>

The ComfyUI repackaged PiD weights are labeled `NSCLv1`. Review the upstream
license before enabling the acceptance input. ComfyColab does not change those
terms.
