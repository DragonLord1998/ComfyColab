from __future__ import annotations

import importlib
from typing import Any

from .catalog import backbone_names
from .graph import REQUIRED_NODES, build_pid_graph
from .models import ensure_pid_assets


MAX_SEED = (2**63) - 1
SCALE_OPTIONS = ["4x", "Experimental 16x (tiled)"]
MAX_OUTPUT_4X = 4096
MAX_OUTPUT_16X = 8192


def _io():
    return importlib.import_module("comfy_api.latest").io


def _require_upstream_nodes() -> None:
    try:
        registry = importlib.import_module("nodes").NODE_CLASS_MAPPINGS
    except (ModuleNotFoundError, AttributeError):
        return
    missing = sorted(REQUIRED_NODES - set(registry))
    if missing:
        raise RuntimeError(
            "ComfyColab PiD requires a pinned ComfyUI build with PixelDiT/PiD "
            f"support. Missing node IDs: {', '.join(missing)}. Restart with "
            "`comfycolab start --refresh`."
        )


def _image_dimensions(image: Any) -> tuple[int, int]:
    shape = getattr(image, "shape", None)
    if shape is None and image is not None:
        shape = getattr(getattr(image, "movedim", None), "shape", None)
    if shape is None or len(shape) < 3:
        raise ValueError("PiD upscale requires an IMAGE tensor with shape [B, H, W, C].")
    return int(shape[2]), int(shape[1])


def _validate_dimensions(width: int, height: int, scale: str) -> None:
    factor = 16 if scale == "Experimental 16x (tiled)" else 4
    cap = MAX_OUTPUT_16X if factor == 16 else MAX_OUTPUT_4X
    if width * factor > cap or height * factor > cap:
        raise ValueError(
            f"PiD {scale} output is capped at {cap}px per side in this facade; "
            f"received {width}x{height}, which would output "
            f"{width * factor}x{height * factor}."
        )


class ComfyColabPiDUpscale:
    @classmethod
    def define_schema(cls):
        io = _io()
        return io.Schema(
            node_id="ComfyColabPiDUpscale",
            display_name="ComfyColab PiD — Image Upscaler",
            category="ComfyColab/Image",
            description=(
                "4x image upscale through NVIDIA PiD / PixelDiT with a matching "
                "VAE. Experimental 16x runs two 4x passes and tiles the second pass."
            ),
            enable_expand=True,
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input(
                    "vae_family",
                    options=backbone_names(),
                    default="FLUX.1",
                    tooltip="Selects the PiD checkpoint and matching VAE family.",
                ),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="high fidelity detailed image upscale",
                ),
                io.Combo.Input("scale", options=SCALE_OPTIONS, default="4x"),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Float.Input(
                    "degrade_sigma",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    advanced=True,
                ),
                io.Int.Input(
                    "tile_size",
                    default=1536,
                    min=512,
                    max=4096,
                    step=64,
                    advanced=True,
                ),
                io.Int.Input(
                    "tile_overlap",
                    default=384,
                    min=64,
                    max=1024,
                    step=32,
                    advanced=True,
                ),
                io.Boolean.Input(
                    "accept_nvidia_noncommercial_license",
                    default=False,
                    tooltip=(
                        "Required before downloads. Comfy-Org PixelDiT is "
                        "published under the NVIDIA Source Code License V1."
                    ),
                ),
                io.Boolean.Input(
                    "force_redownload",
                    default=False,
                    advanced=True,
                    tooltip="Discard resumable cached files and download selected assets again.",
                ),
            ],
            outputs=[io.Image.Output("image")],
        )

    @classmethod
    def execute(
        cls,
        image,
        vae_family="FLUX.1",
        prompt="high fidelity detailed image upscale",
        scale="4x",
        seed=0,
        degrade_sigma=0.0,
        tile_size=1536,
        tile_overlap=384,
        accept_nvidia_noncommercial_license=False,
        force_redownload=False,
    ):
        if not accept_nvidia_noncommercial_license:
            raise PermissionError(
                "PiD downloads are blocked until "
                "accept_nvidia_noncommercial_license is true. "
                "Comfy-Org PixelDiT/PiD is published under NVIDIA Source Code License V1."
            )
        if vae_family not in backbone_names():
            raise ValueError(f"Unknown PiD VAE family: {vae_family}.")
        if scale not in SCALE_OPTIONS:
            raise ValueError("PiD scale must be 4x or Experimental 16x (tiled).")
        seed = int(seed)
        degrade_sigma = float(degrade_sigma)
        tile_size = int(tile_size)
        tile_overlap = int(tile_overlap)
        if seed < 0 or seed > MAX_SEED:
            raise ValueError(f"seed must be between 0 and {MAX_SEED}.")
        if not 0.0 <= degrade_sigma <= 1.0:
            raise ValueError("degrade_sigma must be between 0 and 1.")
        if tile_overlap >= tile_size:
            raise ValueError("tile_overlap must be smaller than tile_size.")
        width, height = _image_dimensions(image)
        _validate_dimensions(width, height, scale)
        _require_upstream_nodes()
        model_names = ensure_pid_assets(
            vae_family,
            force_redownload=bool(force_redownload),
        )
        return build_pid_graph(
            image=image,
            prompt=str(prompt),
            scale=scale,
            width=width,
            height=height,
            seed=seed,
            degrade_sigma=degrade_sigma,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            vae_family=vae_family,
            model_names=model_names,
        )


PUBLIC_NODE_CLASS_MAPPINGS = {
    "ComfyColabPiDUpscale": ComfyColabPiDUpscale,
}

NODE_CLASS_MAPPINGS = dict(PUBLIC_NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyColabPiDUpscale": "ComfyColab PiD — Image Upscaler",
}
