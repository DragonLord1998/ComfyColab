from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .catalog import (
    bundle_for,
    flux_2_dev_bundle_for,
    flux_2_dev_variants,
    flux_2_klein_4b_bundle_for,
    flux_2_klein_4b_variants,
    flux_2_klein_9b_bundle_for,
    flux_2_klein_9b_variants,
    krea_2_bundle_for,
    krea_2_variants,
    quantization_names,
    qwen_edit_bundle_for,
    qwen_edit_quantization_names,
)
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
            f"Required loader '{name}' is unavailable. Start ComfyUI through the latest "
            "ComfyColab bootstrap so its pinned core and GGUF loaders are installed."
        )
    return loader_class()


def _download_bundle(
    folder_paths: Any,
    bundle: dict[str, dict[str, Any]],
    folder_keys: dict[str, str],
    force_redownload: bool,
) -> None:
    destinations = {
        component: _first_model_path(folder_paths, folder_key)
        for component, folder_key in folder_keys.items()
    }
    for component, specification in bundle.items():
        download_file(
            url=specification["url"],
            destination=destinations[component] / specification["filename"],
            expected_sha256=specification["sha256"],
            force=force_redownload,
            progress=_ComfyProgress(),
        )


def _load_vae(comfy_nodes: Any, filename: str) -> Any:
    return _loader(comfy_nodes, "VAELoader").load_vae(filename)[0]


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

        _download_bundle(
            folder_paths,
            bundle,
            {"model": "unet_gguf", "text_encoder": "clip_gguf", "vae": "vae"},
            force_redownload,
        )

        model = _loader(comfy_nodes, "UnetLoaderGGUF").load_unet(
            bundle["model"]["filename"]
        )[0]
        text_encoder = _loader(comfy_nodes, "CLIPLoaderGGUF").load_clip(
            bundle["text_encoder"]["filename"],
            type="lumina2",
        )[0]
        vae = _load_vae(comfy_nodes, bundle["vae"]["filename"])
        return model, text_encoder, vae


class QwenImageEdit2511BundleLoader:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        return {
            "required": {
                "quantization": (
                    qwen_edit_quantization_names(),
                    {"default": "Q4_K_M"},
                ),
                "force_redownload": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "text_encoder", "vae")
    FUNCTION = "load_bundle"
    CATEGORY = "ComfyColab/loaders"
    DESCRIPTION = (
        "Downloads a verified Qwen Image Edit 2511 GGUF bundle and returns the "
        "model, Qwen 2.5 VL text encoder, and Qwen Image VAE."
    )

    @classmethod
    def IS_CHANGED(cls, quantization: str, force_redownload: bool = False):
        return float("nan") if force_redownload else quantization

    def load_bundle(self, quantization: str, force_redownload: bool = False):
        folder_paths = importlib.import_module("folder_paths")
        comfy_nodes = importlib.import_module("nodes")
        bundle = qwen_edit_bundle_for(quantization)

        _download_bundle(
            folder_paths,
            bundle,
            {
                "model": "unet_gguf",
                "text_encoder": "text_encoders",
                "vae": "vae",
            },
            force_redownload,
        )

        model = _loader(comfy_nodes, "UnetLoaderGGUF").load_unet(
            bundle["model"]["filename"]
        )[0]
        text_encoder = _loader(comfy_nodes, "CLIPLoader").load_clip(
            bundle["text_encoder"]["filename"],
            type="qwen_image",
        )[0]
        vae = _load_vae(comfy_nodes, bundle["vae"]["filename"])
        return model, text_encoder, vae


class Krea2BundleLoader:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        return {
            "required": {
                "variant": (krea_2_variants(), {"default": "Turbo FP8"}),
                "force_redownload": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "text_encoder", "vae")
    FUNCTION = "load_bundle"
    CATEGORY = "ComfyColab/loaders"
    DESCRIPTION = (
        "Downloads a verified official Krea 2 FP8 bundle and returns the model, "
        "Qwen3-VL text encoder, and Qwen Image VAE."
    )

    @classmethod
    def IS_CHANGED(cls, variant: str, force_redownload: bool = False):
        return float("nan") if force_redownload else variant

    def load_bundle(self, variant: str, force_redownload: bool = False):
        folder_paths = importlib.import_module("folder_paths")
        comfy_nodes = importlib.import_module("nodes")
        bundle = krea_2_bundle_for(variant)

        _download_bundle(
            folder_paths,
            bundle,
            {
                "model": "diffusion_models",
                "text_encoder": "text_encoders",
                "vae": "vae",
            },
            force_redownload,
        )

        model = _loader(comfy_nodes, "UNETLoader").load_unet(
            bundle["model"]["filename"],
            weight_dtype="default",
        )[0]
        text_encoder = _loader(comfy_nodes, "CLIPLoader").load_clip(
            bundle["text_encoder"]["filename"],
            type="krea2",
        )[0]
        vae = _load_vae(comfy_nodes, bundle["vae"]["filename"])
        return model, text_encoder, vae


def _load_flux_2_bundle(
    bundle: dict[str, dict[str, Any]],
    force_redownload: bool,
):
    folder_paths = importlib.import_module("folder_paths")
    comfy_nodes = importlib.import_module("nodes")
    _download_bundle(
        folder_paths,
        bundle,
        {
            "model": "unet_gguf",
            "text_encoder": "text_encoders",
            "vae": "vae",
        },
        force_redownload,
    )
    model = _loader(comfy_nodes, "UnetLoaderGGUF").load_unet(
        bundle["model"]["filename"]
    )[0]
    text_encoder = _loader(comfy_nodes, "CLIPLoader").load_clip(
        bundle["text_encoder"]["filename"],
        type="flux2",
    )[0]
    vae = _load_vae(comfy_nodes, bundle["vae"]["filename"])
    return model, text_encoder, vae


class _Flux2GGUFBundleLoader:
    QUANTIZATIONS: Any
    BUNDLE_FOR: Any

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
        return {
            "required": {
                "quantization": (cls.QUANTIZATIONS(), {"default": "Q4_K_M"}),
                "force_redownload": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "text_encoder", "vae")
    FUNCTION = "load_bundle"
    CATEGORY = "ComfyColab/loaders"

    @classmethod
    def IS_CHANGED(cls, quantization: str, force_redownload: bool = False):
        return float("nan") if force_redownload else quantization

    def load_bundle(self, quantization: str, force_redownload: bool = False):
        return _load_flux_2_bundle(
            self.BUNDLE_FOR(quantization),
            force_redownload,
        )


class Flux2Klein4BBundleLoader(_Flux2GGUFBundleLoader):
    QUANTIZATIONS = staticmethod(flux_2_klein_4b_variants)
    BUNDLE_FOR = staticmethod(flux_2_klein_4b_bundle_for)
    DESCRIPTION = (
        "Downloads a verified FLUX.2 Klein 4B GGUF bundle and returns the model, "
        "Qwen3-4B text encoder, and FLUX.2 VAE. Distilled model: 4 steps, CFG 1."
    )


class Flux2Klein9BBundleLoader(_Flux2GGUFBundleLoader):
    QUANTIZATIONS = staticmethod(flux_2_klein_9b_variants)
    BUNDLE_FOR = staticmethod(flux_2_klein_9b_bundle_for)
    DESCRIPTION = (
        "Downloads a verified FLUX.2 Klein 9B GGUF bundle and returns the model, "
        "Qwen3-8B text encoder, and FLUX.2 VAE. Non-commercial license."
    )


class Flux2DevBundleLoader(_Flux2GGUFBundleLoader):
    QUANTIZATIONS = staticmethod(flux_2_dev_variants)
    BUNDLE_FOR = staticmethod(flux_2_dev_bundle_for)
    DESCRIPTION = (
        "Downloads a verified FLUX.2 Dev GGUF bundle and returns the model, "
        "Mistral Small text encoder, and FLUX.2 VAE. Non-commercial license."
    )
