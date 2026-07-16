from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .download import download_file


HF_REPO_ID = "VAST-AI/TripoSplat"
HF_REVISION = "de3b99ab2627d565a8d5fc40f2db52557b82b974"
HF_BASE_URL = f"https://huggingface.co/{HF_REPO_ID}/resolve/{HF_REVISION}"


@dataclass(frozen=True)
class ModelAsset:
    filename: str
    folder_key: str
    sha256: str
    size_bytes: int

    @property
    def url(self) -> str:
        return f"{HF_BASE_URL}/{self.folder_key}/{self.filename}"


MODEL_ASSETS: tuple[ModelAsset, ...] = (
    ModelAsset(
        filename="birefnet.safetensors",
        folder_key="background_removal",
        sha256="9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154",
        size_bytes=444_473_596,
    ),
    ModelAsset(
        filename="dino_v3_vit_h.safetensors",
        folder_key="clip_vision",
        sha256="a29ef35101a16966972a0d50732a6f3a608ff7cfffb2afa9bbe9007cb842cc53",
        size_bytes=1_681_247_696,
    ),
    ModelAsset(
        filename="triposplat_fp16.safetensors",
        folder_key="diffusion_models",
        sha256="c870b97ac1d6bc9177608a5ec625e19ef9f3c5019aa68f64b0fb7803abcd6d20",
        size_bytes=741_106_994,
    ),
    ModelAsset(
        filename="flux2-vae.safetensors",
        folder_key="vae",
        sha256="d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5",
        size_bytes=336_213_556,
    ),
    ModelAsset(
        filename="triposplat_vae_decoder_fp16.safetensors",
        folder_key="vae",
        sha256="ed0d0c3d43b599e326845d0ec70f3cf77be9a55e2d97627ac3b34d2830763cc8",
        size_bytes=576_148_442,
    ),
)

MODEL_FILENAMES = {
    "background_removal": "birefnet.safetensors",
    "clip_vision": "dino_v3_vit_h.safetensors",
    "diffusion_model": "triposplat_fp16.safetensors",
    "flux_vae": "flux2-vae.safetensors",
    "decoder_vae": "triposplat_vae_decoder_fp16.safetensors",
}


def _first_model_path(folder_paths: Any, key: str) -> Path:
    paths = folder_paths.get_folder_paths(key)
    if not paths:
        raise RuntimeError(f"ComfyUI has no configured model folder for '{key}'.")
    destination = Path(paths[0])
    destination.mkdir(parents=True, exist_ok=True)
    return destination


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


def ensure_model_assets(*, force_redownload: bool = False) -> dict[str, str]:
    folder_paths = importlib.import_module("folder_paths")
    resolved: dict[str, str] = {}
    progress = _ComfyProgress()
    for asset in MODEL_ASSETS:
        destination = _first_model_path(folder_paths, asset.folder_key) / asset.filename
        download_file(
            url=asset.url,
            destination=destination,
            expected_sha256=asset.sha256,
            expected_size=asset.size_bytes,
            force=force_redownload,
            progress=progress,
        )
        resolved[asset.filename] = str(destination)
    return resolved


def filenames_by_role() -> dict[str, str]:
    return dict(MODEL_FILENAMES)
