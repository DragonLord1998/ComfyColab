from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .catalog import bundle_for, quantization_names
from .download import download_file


class _ComfyProgress:
    def __init__(self) -> None:
        self._bar: Any = None
        self._total: int | None = None

    def __call__(self, completed: int, total: int | None) -> None:
        if not total:
            return
        if self._bar is None or self._total != total:
            try:
                comfy_utils = importlib.import_module("comfy.utils")
                self._bar = comfy_utils.ProgressBar(total)
                self._total = total
            except (ImportError, AttributeError):
                return
        self._bar.update_absolute(completed, total)


def _first_model_path(folder_paths: Any, key: str) -> Path:
    paths = folder_paths.get_folder_paths(key)
    if not paths:
        raise RuntimeError(f"ComfyUI has no configured model folder for '{key}'.")
    destination = Path(paths[0])
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _loader(comfy_nodes: Any, name: str) -> Any:
    loader_class = comfy_nodes.NODE_CLASS_MAPPINGS.get(name)
    if loader_class is None:
        raise RuntimeError(
            f"Required loader '{name}' is unavailable. Ensure ComfyUI-GGUF was installed "
            "before starting ComfyUI."
        )
    return loader_class()


class ZImageTurboBundleLoader:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        return {
            "required": {
                "quantization": (quantization_names(), {"default": "Q4_K_M"}),
                "force_redownload": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "text_encoder", "vae")
    FUNCTION = "load_bundle"
    CATEGORY = "ComfyColab/loaders"
    DESCRIPTION = (
        "Downloads a verified Z-Image Turbo GGUF bundle into temporary storage and "
        "returns the model, Qwen3 text encoder, and VAE."
    )

    @classmethod
    def IS_CHANGED(cls, quantization: str, force_redownload: bool = False):
        return float("nan") if force_redownload else quantization

    def load_bundle(self, quantization: str, force_redownload: bool = False):
        folder_paths = importlib.import_module("folder_paths")
        comfy_nodes = importlib.import_module("nodes")
        bundle = bundle_for(quantization)

        destinations = {
            "model": _first_model_path(folder_paths, "unet_gguf"),
            "text_encoder": _first_model_path(folder_paths, "clip_gguf"),
            "vae": _first_model_path(folder_paths, "vae"),
        }

        for component, specification in bundle.items():
            destination = destinations[component] / specification["filename"]
            download_file(
                url=specification["url"],
                destination=destination,
                expected_sha256=specification["sha256"],
                force=force_redownload,
                progress=_ComfyProgress(),
            )

        model = _loader(comfy_nodes, "UnetLoaderGGUF").load_unet(
            bundle["model"]["filename"]
        )[0]
        text_encoder = _loader(comfy_nodes, "CLIPLoaderGGUF").load_clip(
            bundle["text_encoder"]["filename"],
            type="lumina2",
        )[0]
        vae = _loader(comfy_nodes, "VAELoader").load_vae(bundle["vae"]["filename"])[0]
        return model, text_encoder, vae
