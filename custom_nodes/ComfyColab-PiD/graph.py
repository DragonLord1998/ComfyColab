from __future__ import annotations

import importlib
from typing import Any

from .catalog import MAGE_VAE_EXPERIMENTAL


SIGMAS = "0.999,0.866,0.634,0.342,0"
ZIMAGE_DENOISE = 0.33
ZIMAGE_STEPS = 5
ZIMAGE_QUANTIZATION = "Q4_K_M"
LATENT_FORMATS = {
    "FLUX.1": "flux",
    "FLUX.2": "flux",
    "Qwen Image": "qwenimage",
    MAGE_VAE_EXPERIMENTAL: "flux",
}
SCALE_FACTORS = {"2x": 2, "4x": 4, "8x": 8, "16x": 16}

REQUIRED_NODES = frozenset(
    {
        "UNETLoader",
        "CLIPLoader",
        "CLIPTextEncode",
        "VAELoader",
        "VAEEncode",
        "VAEEncodeTiled",
        "PiDConditioning",
        "EmptyChromaRadianceLatentImage",
        "KSamplerSelect",
        "ManualSigmas",
        "SamplerCustom",
        "VAEDecode",
        "VAEDecodeTiled",
        "ImageScale",
        "SplitImageToTileList",
        "ImageMergeTileList",
        "ComfyColabZImageTurboBundleLoader",
        "ModelSamplingAuraFlow",
        "KSampler",
    }
)
MAGE_REQUIRED_NODES = frozenset({"ComfyColabMageVAEEncode"})


def _builder():
    return importlib.import_module("comfy_execution.graph_utils").GraphBuilder()


def _finish(graph, image):
    io = importlib.import_module("comfy_api.latest").io
    return io.NodeOutput(image, expand=graph.finalize())


def _aligned(value: int, multiple: int = 16) -> int:
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


def build_pid_graph(
    *,
    image: Any,
    prompt: str,
    scale: str,
    width: int,
    height: int,
    seed: int,
    degrade_sigma: float,
    tile_size: int,
    tile_overlap: int,
    vae_family: str,
    model_names: dict[str, str],
    force_redownload: bool = False,
):
    graph = _builder()
    model = graph.node(
        "UNETLoader",
        unet_name=model_names["model"],
        weight_dtype="default",
    )
    clip = graph.node(
        "CLIPLoader",
        clip_name=model_names["text_encoder"],
        type="pixeldit",
        device="default",
    )
    input_vae = (
        None
        if vae_family == MAGE_VAE_EXPERIMENTAL
        else graph.node("VAELoader", vae_name=model_names["vae"]).out(0)
    )
    pixel_vae = graph.node("VAELoader", vae_name="pixel_space")
    positive = graph.node("CLIPTextEncode", clip=clip.out(0), text=prompt)
    negative = graph.node(
        "CLIPTextEncode",
        clip=clip.out(0),
        text="low quality, worst quality, blurry, deformed, watermark",
    )
    sampler = graph.node("KSamplerSelect", sampler_name="lcm")
    sigmas = graph.node("ManualSigmas", sigmas=SIGMAS)

    requested_factor = SCALE_FACTORS[scale]
    first_stage_factor = 2 if requested_factor == 2 else 4
    generated = _tiled_pid_stage(
        graph,
        image=image,
        width=width,
        height=height,
        model=model.out(0),
        input_vae=input_vae,
        vae_name=model_names["vae"],
        vae_family=vae_family,
        pixel_vae=pixel_vae.out(0),
        positive=positive.out(0),
        negative=negative.out(0),
        sampler=sampler.out(0),
        sigmas=sigmas.out(0),
        seed=seed,
        degrade_sigma=degrade_sigma,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        stage_index=0,
        output_factor=first_stage_factor,
    )
    generated_width = width * first_stage_factor
    generated_height = height * first_stage_factor

    if requested_factor >= 8:
        second_stage_factor = 2 if requested_factor == 8 else 4
        generated = _tiled_pid_stage(
            graph,
            image=generated,
            width=generated_width,
            height=generated_height,
            model=model.out(0),
            input_vae=input_vae,
            vae_name=model_names["vae"],
            vae_family=vae_family,
            pixel_vae=pixel_vae.out(0),
            positive=positive.out(0),
            negative=negative.out(0),
            sampler=sampler.out(0),
            sigmas=sigmas.out(0),
            seed=seed,
            degrade_sigma=degrade_sigma,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            stage_index=1,
            output_factor=second_stage_factor,
        )
        generated_width *= second_stage_factor
        generated_height *= second_stage_factor

    exact_width = width * requested_factor
    exact_height = height * requested_factor
    if (generated_width, generated_height) != (exact_width, exact_height):
        raise RuntimeError("Internal PiD scale mapping produced the wrong dimensions.")

    cleaned = _tiled_zimage_cleanup(
        graph,
        image=generated,
        prompt=prompt,
        width=exact_width,
        height=exact_height,
        seed=seed,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        force_redownload=force_redownload,
    )
    return _finish(graph, cleaned)


def _tiled_pid_stage(
    graph: Any,
    *,
    image: Any,
    width: int,
    height: int,
    model: Any,
    input_vae: Any,
    vae_name: str,
    vae_family: str,
    pixel_vae: Any,
    positive: Any,
    negative: Any,
    sampler: Any,
    sigmas: Any,
    seed: int,
    degrade_sigma: float,
    tile_size: int,
    tile_overlap: int,
    stage_index: int,
    output_factor: int,
):
    source_tile_size = max(16, (tile_size // 4 // 16) * 16)
    source_overlap = min(
        source_tile_size - 16,
        max(0, (tile_overlap // 4 // 16) * 16),
    )
    split = graph.node(
        "SplitImageToTileList",
        image=image,
        tile_width=source_tile_size,
        tile_height=source_tile_size,
        overlap=source_overlap,
    )
    tile_width = min(width, source_tile_size)
    tile_height = min(height, source_tile_size)
    output_width = _aligned(tile_width * 4)
    output_height = _aligned(tile_height * 4)
    tiles = _pid_pass(
        graph,
        image=split.out(0),
        model=model,
        input_vae=input_vae,
        vae_name=vae_name,
        vae_family=vae_family,
        pixel_vae=pixel_vae,
        positive=positive,
        negative=negative,
        sampler=sampler,
        sigmas=sigmas,
        latent_format=LATENT_FORMATS[vae_family],
        output_width=output_width,
        output_height=output_height,
        seed=(seed + stage_index * 100_000) % (2**63),
        degrade_sigma=degrade_sigma,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )
    exact_tile_width = tile_width * output_factor
    exact_tile_height = tile_height * output_factor
    if (output_width, output_height) != (exact_tile_width, exact_tile_height):
        tiles = graph.node(
            "ImageScale",
            image=tiles,
            upscale_method="lanczos",
            width=exact_tile_width,
            height=exact_tile_height,
            crop="disabled",
        ).out(0)
    return graph.node(
        "ImageMergeTileList",
        image_list=tiles,
        final_width=width * output_factor,
        final_height=height * output_factor,
        overlap=source_overlap * output_factor,
    ).out(0)


def _tiled_zimage_cleanup(
    graph: Any,
    *,
    image: Any,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    tile_size: int,
    tile_overlap: int,
    force_redownload: bool,
):
    bundle = graph.node(
        "ComfyColabZImageTurboBundleLoader",
        quantization=ZIMAGE_QUANTIZATION,
        force_redownload=force_redownload,
    )
    model = graph.node("ModelSamplingAuraFlow", model=bundle.out(0), shift=3.0)
    positive = graph.node(
        "CLIPTextEncode",
        clip=bundle.out(1),
        text=(
            f"{prompt}. Preserve the source composition and identity; remove only "
            "upscale artifacts, ringing, seams, and compression noise."
        ),
    )
    negative = graph.node("CLIPTextEncode", clip=bundle.out(1), text="")
    split = graph.node(
        "SplitImageToTileList",
        image=image,
        tile_width=tile_size,
        tile_height=tile_size,
        overlap=tile_overlap,
    )
    source_tile_width = min(width, tile_size)
    source_tile_height = min(height, tile_size)
    work_tile_width = _aligned(source_tile_width, 8)
    work_tile_height = _aligned(source_tile_height, 8)
    cleanup_tiles = split.out(0)
    if (work_tile_width, work_tile_height) != (
        source_tile_width,
        source_tile_height,
    ):
        cleanup_tiles = graph.node(
            "ImageScale",
            image=cleanup_tiles,
            upscale_method="lanczos",
            width=work_tile_width,
            height=work_tile_height,
            crop="disabled",
        ).out(0)
    encoded = graph.node(
        "VAEEncodeTiled",
        pixels=cleanup_tiles,
        vae=bundle.out(2),
        tile_size=tile_size,
        overlap=tile_overlap,
    )
    sampled = graph.node(
        "KSampler",
        model=model.out(0),
        seed=(seed + 1_000_000) % (2**63),
        steps=ZIMAGE_STEPS,
        cfg=1.0,
        sampler_name="dpmpp_2m_sde",
        scheduler="beta",
        positive=positive.out(0),
        negative=negative.out(0),
        latent_image=encoded.out(0),
        denoise=ZIMAGE_DENOISE,
    )
    decoded = graph.node(
        "VAEDecodeTiled",
        samples=sampled.out(0),
        vae=bundle.out(2),
        tile_size=tile_size,
        overlap=tile_overlap,
    ).out(0)
    if (work_tile_width, work_tile_height) != (
        source_tile_width,
        source_tile_height,
    ):
        decoded = graph.node(
            "ImageScale",
            image=decoded,
            upscale_method="lanczos",
            width=source_tile_width,
            height=source_tile_height,
            crop="disabled",
        ).out(0)
    return graph.node(
        "ImageMergeTileList",
        image_list=decoded,
        final_width=width,
        final_height=height,
        overlap=tile_overlap,
    ).out(0)


def _pid_pass(
    graph: Any,
    *,
    image: Any,
    model: Any,
    input_vae: Any,
    vae_name: str,
    vae_family: str,
    pixel_vae: Any,
    positive: Any,
    negative: Any,
    sampler: Any,
    sigmas: Any,
    latent_format: str,
    output_width: int,
    output_height: int,
    seed: int,
    degrade_sigma: float,
    tile_size: int,
    tile_overlap: int,
):
    if vae_family == MAGE_VAE_EXPERIMENTAL:
        encoded = graph.node(
            "ComfyColabMageVAEEncode",
            image=image,
            vae_name=vae_name,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            keep_worker_loaded=True,
        )
    else:
        encoded = graph.node("VAEEncode", pixels=image, vae=input_vae)
    conditioning = graph.node(
        "PiDConditioning",
        positive=positive,
        latent=encoded.out(0),
        latent_format=latent_format,
        degrade_sigma=degrade_sigma,
    )
    latent = graph.node(
        "EmptyChromaRadianceLatentImage",
        width=output_width,
        height=output_height,
        batch_size=1,
    )
    sampled = graph.node(
        "SamplerCustom",
        model=model,
        add_noise=True,
        noise_seed=seed,
        cfg=1.0,
        positive=conditioning.out(0),
        negative=negative,
        sampler=sampler,
        sigmas=sigmas,
        latent_image=latent.out(0),
    )
    return graph.node(
        "VAEDecode",
        samples=sampled.out(0),
        vae=pixel_vae,
    ).out(0)
