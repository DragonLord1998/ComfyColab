# ComfyColab Z-Image node

`Z-Image Turbo Bundle Loader` downloads a curated, checksum-verified bundle into
the current ComfyUI runtime and returns three standard outputs:

- `MODEL`: selected Z-Image Turbo GGUF quantization
- `CLIP`: fixed Qwen3-4B Q4 text encoder, loaded as `lumina2`
- `VAE`: official `ae.safetensors`

The node requires [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF). The
ComfyColab bootstrap installs that dependency automatically.

All files are stored under the active ComfyUI `models` folders. In a Colab
runtime these files are temporary and disappear when the runtime is released.
