from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .catalog import selected_assets
from .download import download_file


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


def ensure_model_assets(
    gguf_model: str,
    spatial_upscaler: str,
    fps: int | str,
    force_redownload: bool = False,
) -> dict[str, str]:
    folder_paths = importlib.import_module("folder_paths")
    assets = selected_assets(gguf_model, spatial_upscaler, fps)
    progress = _ComfyProgress()
    filenames: dict[str, str] = {}
    for role, specification in assets.items():
        destination = (
            _first_model_path(folder_paths, specification["folder_key"])
            / specification["filename"]
        )
        download_file(
            url=specification["url"],
            destination=destination,
            expected_sha256=specification["sha256"],
            expected_size=specification["size_bytes"],
            force=force_redownload,
            progress=progress,
        )
        filenames[role] = specification["filename"]
    return filenames


def filenames_by_role(
    gguf_model: str,
    spatial_upscaler: str,
    fps: int | str,
) -> dict[str, str]:
    return {
        role: specification["filename"]
        for role, specification in selected_assets(
            gguf_model, spatial_upscaler, fps
        ).items()
    }
