from __future__ import annotations

import importlib
from typing import Any

from .graph import QUALITY_PRESETS, build_triposplat_graph, require_native_nodes, resolve_num_gaussians
from .models import ensure_model_assets, filenames_by_role


OUTPUT_FORMATS = ["ply", "spz", "ksplat"]
MAX_SEED = (2**63) - 1


def _io():
    return importlib.import_module("comfy_api.latest").io


def _splat_output(io: Any):
    return io.Splat.Output("splat") if hasattr(io, "Splat") else io.AnyType.Output("splat")


def _file3d_splat_output(io: Any):
    if hasattr(io, "File3DSplatAny"):
        return io.File3DSplatAny.Output("model_3d")
    if hasattr(io, "File3D"):
        return io.File3D.Output("model_3d")
    return io.AnyType.Output("model_3d")


class ComfyColabTripoSplatImageToGaussianSplat:
    @classmethod
    def define_schema(cls):
        io = _io()
        return io.Schema(
            node_id="ComfyColabTripoSplatImageToGaussianSplat",
            display_name="ComfyColab TripoSplat — Image to Gaussian Splat",
            category="ComfyColab/3D",
            description=(
                "Generates a native Gaussian SPLAT and splat File3D from one image "
                "using ComfyUI's pinned TripoSplat core nodes."
            ),
            enable_expand=True,
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("quality", options=list(QUALITY_PRESETS), default="Quality — 262K"),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Boolean.Input("remove_background", default=True, advanced=True),
                io.Int.Input("sampling_steps", default=20, min=1, max=50, advanced=True),
                io.Float.Input("guidance_scale", default=3.0, min=1.0, max=10.0, step=0.1, advanced=True),
                io.Boolean.Input("enable_sampling_preview", default=True, advanced=True),
                io.Combo.Input("output_format", options=OUTPUT_FORMATS, default="ply", advanced=True),
            ],
            outputs=[_splat_output(io), _file3d_splat_output(io)],
        )

    @classmethod
    def execute(
        cls,
        image,
        quality="Quality — 262K",
        seed=0,
        remove_background=True,
        sampling_steps=20,
        guidance_scale=3.0,
        enable_sampling_preview=True,
        output_format="ply",
    ):
        seed = int(seed)
        if seed < 0 or seed > MAX_SEED:
            raise ValueError(f"seed must be between 0 and {MAX_SEED}")
        if output_format not in OUTPUT_FORMATS:
            raise ValueError(f"Unsupported TripoSplat output format: {output_format}")
        require_native_nodes()
        ensure_model_assets()
        names = filenames_by_role()
        return build_triposplat_graph(
            image,
            num_gaussians=resolve_num_gaussians(quality),
            seed=seed,
            remove_background=bool(remove_background),
            sampling_steps=int(sampling_steps),
            guidance_scale=float(guidance_scale),
            enable_sampling_preview=bool(enable_sampling_preview),
            output_format=output_format,
            model_names={
                "background_removal": names["background_removal"],
                "clip_vision": names["clip_vision"],
                "diffusion_model": names["diffusion_model"],
                "flux_vae": names["flux_vae"],
                "decoder_vae": names["decoder_vae"],
            },
        )


class ComfyColabTripoSplatOpaqueMask:
    @classmethod
    def define_schema(cls):
        io = _io()
        return io.Schema(
            node_id="ComfyColabTripoSplatOpaqueMask",
            display_name=None,
            category="ComfyColab/3D/Adapters",
            description="Private helper used by the ComfyColab TripoSplat facade.",
            is_dev_only=True,
            inputs=[io.Image.Input("image")],
            outputs=[io.Mask.Output("mask")],
        )

    @classmethod
    def execute(cls, image):
        torch = importlib.import_module("torch")
        mask = torch.ones(image.shape[0:3], dtype=image.dtype, device=image.device)
        return _io().NodeOutput(mask)


PUBLIC_NODE_CLASS_MAPPINGS = {
    "ComfyColabTripoSplatImageToGaussianSplat": ComfyColabTripoSplatImageToGaussianSplat,
}

NODE_CLASS_MAPPINGS = {
    **PUBLIC_NODE_CLASS_MAPPINGS,
    "ComfyColabTripoSplatOpaqueMask": ComfyColabTripoSplatOpaqueMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyColabTripoSplatImageToGaussianSplat": "ComfyColab TripoSplat — Image to Gaussian Splat",
}
