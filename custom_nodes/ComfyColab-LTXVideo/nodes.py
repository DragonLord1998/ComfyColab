from __future__ import annotations

import importlib
from typing import Any

from .catalog import gguf_names, spatial_upscaler_names
from .graph import build_ltx23_graph, required_nodes
from .models import ensure_model_assets


MAX_SEED = (2**63) - 1
FPS_OPTIONS = ["24", "48"]


def _io():
    return importlib.import_module("comfy_api.latest").io


def _require_upstream_nodes(node_ids: set[str]) -> None:
    try:
        registry = importlib.import_module("nodes").NODE_CLASS_MAPPINGS
    except (ModuleNotFoundError, AttributeError):
        return
    missing = sorted(node_ids - set(registry))
    if missing:
        raise RuntimeError(
            "ComfyColab LTX-2.3 requires the pinned ComfyUI, ComfyUI-GGUF, and "
            "Lighttricks ComfyUI-LTXVideo node packs. Missing node IDs: "
            f"{', '.join(missing)}. Restart with `comfycolab start --refresh`."
        )


def _video_output(io: Any):
    return io.Video.Output("video")


class ComfyColabLTX23Video:
    @classmethod
    def define_schema(cls):
        io = _io()
        return io.Schema(
            node_id="ComfyColabLTX23Video",
            display_name="ComfyColab LTX-2.3 — Text/Image to Video",
            category="ComfyColab/Video",
            description=(
                "Generates synchronized video and audio with the latest LTX-2.3 "
                "Distilled 1.1 GGUF. Width and height are the base generation size "
                "before the optional spatial upscaler. 48 FPS uses the official "
                "temporal x2 latent upscaler after 24 FPS generation."
            ),
            enable_expand=True,
            inputs=[
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Describe the shot, motion, dialogue, and ambient audio.",
                ),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Optional first-frame reference. Leave disconnected for text-to-video.",
                ),
                io.Combo.Input(
                    "gguf_model",
                    options=gguf_names(),
                    default="Q3_K_S",
                    tooltip="Q3_K_S is the safest low-memory default; Q4 variants improve fidelity.",
                ),
                io.Combo.Input(
                    "fps",
                    options=FPS_OPTIONS,
                    default="24",
                    tooltip="48 applies real temporal x2 latent upscaling.",
                ),
                io.Combo.Input(
                    "spatial_upscaler",
                    options=spatial_upscaler_names(),
                    default="2x",
                    tooltip=(
                        "2x uses the latest v1.1 hotfix; 1.5x uses the latest "
                        "available v1.0 and requires 64-divisible base dimensions."
                    ),
                ),
                io.Int.Input("width", default=960, min=256, max=1536, step=32),
                io.Int.Input("height", default=544, min=256, max=1536, step=32),
                io.Int.Input(
                    "frame_count",
                    default=121,
                    min=9,
                    max=257,
                    step=8,
                    tooltip="Must satisfy (frame_count - 1) % 8 == 0.",
                ),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Float.Input(
                    "image_strength",
                    default=0.7,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    advanced=True,
                ),
                io.Boolean.Input(
                    "force_redownload",
                    default=False,
                    advanced=True,
                    tooltip="Discard resumable cached files and download selected assets again.",
                ),
            ],
            outputs=[
                _video_output(io),
                io.Image.Output("frames"),
                io.Audio.Output("audio"),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompt,
        image=None,
        gguf_model="Q3_K_S",
        fps="24",
        spatial_upscaler="2x",
        width=960,
        height=544,
        frame_count=121,
        seed=0,
        image_strength=0.7,
        force_redownload=False,
    ):
        fps_value = int(fps)
        width = int(width)
        height = int(height)
        frame_count = int(frame_count)
        seed = int(seed)
        image_strength = float(image_strength)
        if not str(prompt).strip():
            raise ValueError("LTX-2.3 requires a non-empty prompt.")
        if fps_value not in {24, 48}:
            raise ValueError("LTX-2.3 FPS must be 24 or 48.")
        if width % 32 or height % 32:
            raise ValueError("LTX-2.3 width and height must be divisible by 32.")
        if spatial_upscaler == "1.5x" and (width % 64 or height % 64):
            raise ValueError(
                "LTX-2.3 1.5x spatial upscaling requires width and height "
                "to be divisible by 64."
            )
        if (frame_count - 1) % 8:
            raise ValueError("LTX-2.3 frame_count must satisfy (frame_count - 1) % 8 == 0.")
        if seed < 0 or seed > MAX_SEED:
            raise ValueError(f"seed must be between 0 and {MAX_SEED}.")
        if not 0.0 <= image_strength <= 1.0:
            raise ValueError("image_strength must be between 0 and 1.")
        _require_upstream_nodes(
            required_nodes(
                image_to_video=image is not None,
                spatial=spatial_upscaler != "None",
                temporal=fps_value == 48,
            )
        )
        model_names = ensure_model_assets(
            gguf_model,
            spatial_upscaler,
            fps_value,
            force_redownload=bool(force_redownload),
        )
        return build_ltx23_graph(
            prompt=str(prompt),
            image=image,
            fps=fps_value,
            spatial_upscaler=spatial_upscaler,
            width=width,
            height=height,
            frame_count=frame_count,
            seed=seed,
            image_strength=image_strength,
            model_names=model_names,
        )


PUBLIC_NODE_CLASS_MAPPINGS = {
    "ComfyColabLTX23Video": ComfyColabLTX23Video,
}

NODE_CLASS_MAPPINGS = dict(PUBLIC_NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyColabLTX23Video": "ComfyColab LTX-2.3 — Text/Image to Video",
}
