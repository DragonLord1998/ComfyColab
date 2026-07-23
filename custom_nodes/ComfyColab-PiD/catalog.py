from __future__ import annotations

from typing import Any


PIXELDIT_REVISION = "70e1298e4e1798c6af6393153f145366fec7bf35"
PIXELDIT_BASE = f"https://huggingface.co/Comfy-Org/PixelDiT/resolve/{PIXELDIT_REVISION}"


class CatalogError(RuntimeError):
    pass


BACKBONES: dict[str, dict[str, dict[str, Any]]] = {
    "FLUX.1": {
        "model": {
            "filename": "pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors",
            "folder_key": "diffusion_models",
            "url": (
                f"{PIXELDIT_BASE}/diffusion_models/"
                "pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors?download=true"
            ),
            "sha256": "18931256e97822dc31db10b1e7399c73e7ee2c897f6d461eb1d1cf5e1d2de049",
            "size_bytes": 2800450070,
            "display_size": "2.80 GB",
        },
        "vae": {
            "filename": "ae.safetensors",
            "folder_key": "vae",
            "url": (
                "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/"
                "d24c4cf2a0cd98a42f23467e27e3d76ee9438b8e/split_files/vae/"
                "ae.safetensors?download=true"
            ),
            "sha256": "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
            "size_bytes": 335304388,
            "display_size": "335 MB",
        },
    },
    "FLUX.2": {
        "model": {
            "filename": "pid_1.5_flux2_1024_to_4096_4step_bf16.safetensors",
            "folder_key": "diffusion_models",
            "url": (
                f"{PIXELDIT_BASE}/diffusion_models/"
                "pid_1.5_flux2_1024_to_4096_4step_bf16.safetensors?download=true"
            ),
            "sha256": "dddaf186c7bb9f73b873a7f2b59235eaea5eae9ec7731c0ec40d49d3287eee14",
            "size_bytes": 2800744982,
            "display_size": "2.80 GB",
        },
        "vae": {
            "filename": "flux2-dev-vae.safetensors",
            "folder_key": "vae",
            "url": (
                "https://huggingface.co/Comfy-Org/flux2-dev/resolve/"
                "03d6521e6f6a47396b3f951cbea50f7e6c2f482e/split_files/vae/"
                "flux2-vae.safetensors?download=true"
            ),
            "sha256": "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5",
            "size_bytes": 336213556,
            "display_size": "336 MB",
        },
    },
    "Qwen Image": {
        "model": {
            "filename": "pid_1.5_qwenimage_1024_to_4096_4step_bf16.safetensors",
            "folder_key": "diffusion_models",
            "url": (
                f"{PIXELDIT_BASE}/diffusion_models/"
                "pid_1.5_qwenimage_1024_to_4096_4step_bf16.safetensors?download=true"
            ),
            "sha256": "6abdd5377db01d4a9d8b013a2853c72d1673c6a6316a2a821345d92800277325",
            "size_bytes": 2800450070,
            "display_size": "2.80 GB",
        },
        "vae": {
            "filename": "qwen_image_vae.safetensors",
            "folder_key": "vae",
            "url": (
                "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/"
                "46839d338df81ce625d5fae27d7e370314c0fbc9/split_files/vae/"
                "qwen_image_vae.safetensors?download=true"
            ),
            "sha256": "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
            "size_bytes": 253806246,
            "display_size": "254 MB",
        },
    },
}

TEXT_ENCODER = {
    "filename": "gemma_2_2b_it_elm_bf16.safetensors",
    "folder_key": "text_encoders",
    "url": (
        f"{PIXELDIT_BASE}/text_encoders/"
        "gemma_2_2b_it_elm_bf16.safetensors?download=true"
    ),
    "sha256": "e7ae59c203c392db4aa4e27783e924ec3225eb563392260cf747e1130ffcdb88",
    "size_bytes": 5232958571,
    "display_size": "5.23 GB",
}


def backbone_names() -> list[str]:
    return list(BACKBONES)


def selected_assets(backbone: str) -> dict[str, dict[str, Any]]:
    try:
        assets = BACKBONES[backbone]
    except KeyError as error:
        raise CatalogError(f"Unknown PiD VAE/backbone selection: {backbone}") from error
    return {
        "model": assets["model"],
        "text_encoder": TEXT_ENCODER,
        "vae": assets["vae"],
    }
