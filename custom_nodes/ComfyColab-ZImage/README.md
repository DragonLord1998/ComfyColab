# ComfyColab model bundle nodes

This pack provides six curated, checksum-verified bundle loaders:

- `Z-Image Turbo Bundle Loader`
- `Qwen Image Edit 2511 Bundle Loader`
- `Krea 2 Bundle Loader`
- `FLUX.2 Klein 4B Bundle Loader`
- `FLUX.2 Klein 9B Bundle Loader`
- `FLUX.2 Dev Bundle Loader`

Each returns standard `MODEL`, `CLIP`, and `VAE` outputs. Z-Image, Qwen Edit,
and FLUX.2 use ComfyUI-GGUF for their diffusion models. Krea 2 uses ComfyUI's
native FP8 loader because the standard GGUF loader does not yet support the
Krea 2 architecture.

The node requires [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF). The
ComfyColab bootstrap installs that dependency automatically.

All files are stored under the active ComfyUI `models` folders. In a Colab
runtime these files are temporary and disappear when the runtime is released.
Downloads resume when supported and are installed only after SHA-256
verification succeeds. Transient `401`, `403`, `408`, `416`, `425`, `429`, and `5xx`
responses are retried up to five times with backoff while preserving partial
data. Because these curated artifacts are public, a `401`/`403` received with a
configured Hugging Face token is retried anonymously in case that token is
stale. `force_redownload` remains the explicit opt-in that removes cached and
partial files.
