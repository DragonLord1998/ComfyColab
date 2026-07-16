from __future__ import annotations

import importlib
from typing import Any


COMFYUI_REF = "8b099de36acd81acd1afa3b5442951dc847e0a52"
REQUIRED_NATIVE_NODES = frozenset(
    {
        "TripoSplatPreprocessImage",
        "TripoSplatConditioning",
        "TripoSplatSamplingPreview",
        "VAEDecodeTripoSplat",
        "SplatToFile3D",
        "LoadBackgroundRemovalModel",
        "RemoveBackground",
        "CLIPVisionLoader",
        "VAELoader",
        "UNETLoader",
        "KSampler",
    }
)


QUALITY_PRESETS = {
    "Fast — 65K": 65_536,
    "Balanced — 131K": 131_072,
    "Quality — 262K": 262_144,
}


def _builder():
    return importlib.import_module("comfy_execution.graph_utils").GraphBuilder()


def _finish(graph, splat, model_3d):
    io = importlib.import_module("comfy_api.latest").io
    return io.NodeOutput(splat, model_3d, expand=graph.finalize())


def require_native_nodes() -> None:
    try:
        registry = importlib.import_module("nodes").NODE_CLASS_MAPPINGS
    except (ModuleNotFoundError, AttributeError):
        return
    missing = sorted(REQUIRED_NATIVE_NODES - set(registry))
    if missing:
        raise RuntimeError(
            "ComfyColab TripoSplat requires ComfyUI core nodes from pinned "
            f"commit {COMFYUI_REF}. Missing node IDs: {', '.join(missing)}. "
            "Restart with the latest ComfyColab bootstrap or refresh ComfyUI."
        )


def resolve_num_gaussians(quality: str) -> int:
    try:
        return QUALITY_PRESETS[quality]
    except KeyError as error:
        raise ValueError(f"Unknown TripoSplat quality preset: {quality}") from error


def build_triposplat_graph(
    image: Any,
    *,
    num_gaussians: int,
    seed: int,
    remove_background: bool,
    sampling_steps: int,
    guidance_scale: float,
    enable_sampling_preview: bool,
    output_format: str,
    model_names: dict[str, str],
):
    graph = _builder()
    if remove_background:
        bg_model = graph.node(
            "LoadBackgroundRemovalModel",
            bg_removal_name=model_names["background_removal"],
        )
        mask = graph.node(
            "RemoveBackground",
            bg_removal_model=bg_model.out(0),
            image=image,
        )
        prepared_mask = mask.out(0)
    else:
        mask = graph.node("ComfyColabTripoSplatOpaqueMask", image=image)
        prepared_mask = mask.out(0)

    prepared = graph.node(
        "TripoSplatPreprocessImage",
        image=image,
        mask=prepared_mask,
        erode_radius=1 if remove_background else 0,
        size=1024,
    )
    clip_vision = graph.node("CLIPVisionLoader", clip_name=model_names["clip_vision"])
    flux_vae = graph.node("VAELoader", vae_name=model_names["flux_vae"])
    conditioning = graph.node(
        "TripoSplatConditioning",
        clip_vision=clip_vision.out(0),
        vae=flux_vae.out(0),
        image=prepared.out(0),
    )
    model = graph.node(
        "UNETLoader",
        unet_name=model_names["diffusion_model"],
        weight_dtype="default",
    )
    decoder_vae = graph.node("VAELoader", vae_name=model_names["decoder_vae"])
    sampler_model = model
    if enable_sampling_preview:
        sampler_model = graph.node(
            "TripoSplatSamplingPreview",
            model=model.out(0),
            vae=decoder_vae.out(0),
            octree_level=5,
            num_gaussians=min(num_gaussians, 16_384),
            yaw=90.0,
            pitch=15.0,
            point_size=2,
        )
    sampled = graph.node(
        "KSampler",
        model=sampler_model.out(0),
        seed=seed,
        steps=sampling_steps,
        cfg=guidance_scale,
        sampler_name="dpmpp_2m",
        scheduler="simple",
        positive=conditioning.out(0),
        negative=conditioning.out(1),
        latent_image=conditioning.out(2),
        denoise=1.0,
    )
    splat = graph.node(
        "VAEDecodeTripoSplat",
        samples=sampled.out(0),
        vae=decoder_vae.out(0),
        num_gaussians=num_gaussians,
        seed=seed,
    )
    file_3d = graph.node("SplatToFile3D", splat=splat.out(0), format=output_format)
    return _finish(graph, splat.out(0), file_3d.out(0))
