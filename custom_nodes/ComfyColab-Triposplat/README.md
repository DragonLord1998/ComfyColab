# ComfyColab TripoSplat

Lazy ComfyUI V3 node package for generating Gaussian splats from a single image
through the TripoSplat core nodes shipped in ComfyUI commit
`8b099de36acd81acd1afa3b5442951dc847e0a52`.

## Public Node

- `ComfyColabTripoSplatImageToGaussianSplat`
- Display name: `ComfyColab TripoSplat — Image to Gaussian Splat`
- Input: `IMAGE`
- Outputs: native `SPLAT` and splat `File3D` (`ply`, `spz`, or `ksplat`)

The package import path is intentionally light: importing `ComfyColab-Triposplat`
does not import ComfyUI, Torch, CUDA, or model code. ComfyUI materializes the V3
node class from `comfy_entrypoint`.

## Models

The node downloads the five official public `VAST-AI/TripoSplat` files on first
execution, pinned to immutable Hugging Face revision
`de3b99ab2627d565a8d5fc40f2db52557b82b974`:

- `background_removal/birefnet.safetensors`
- `clip_vision/dino_v3_vit_h.safetensors`
- `diffusion_models/triposplat_fp16.safetensors`
- `vae/flux2-vae.safetensors`
- `vae/triposplat_vae_decoder_fp16.safetensors`

Downloads are resumable, checksum-verified, and atomically promoted from
`.part` files only after size and SHA-256 validation.

## Controls

- Quality presets: `Fast — 65K`, `Balanced — 131K`, `Quality — 262K`
- Deterministic `seed`
- Background removal toggle
- Sampling steps and guidance scale
- Live sampling preview toggle
- File export format: `ply`, `spz`, or `ksplat`

PLY is the default because it preserves full spherical-harmonic splat data.
