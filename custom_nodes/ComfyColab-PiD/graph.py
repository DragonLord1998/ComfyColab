from __future__ import annotations

import importlib
from typing import Any

from .catalog import MAGE_VAE_EXPERIMENTAL


SIGMAS = "0.999,0.866,0.634,0.342,0"
LATENT_FORMATS = {
    "FLUX.1": "flux",
    "FLUX.2": "flux",
    "Qwen Image": "qwenimage",
    MAGE_VAE_EXPERIMENTAL: "flux",
}

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
        "ContextWindowsManual",
        "ImageScale",
    }
)
MAGE_REQUIRED_NODES = frozenset({"ComfyColabMageVAEEncode"})


def _builder():
    return importlib.import_module("comfy_execution.graph_utils").GraphBuilder()


def _finish(graph, image):
    io = importlib.import_module("comfy_api.latest").io
    return io.NodeOutput(image, expand=graph.finalize())


def _aligned(value: int) -> int:
    return max(16, ((value + 15) // 16) * 16)


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

    if vae_family == MAGE_VAE_EXPERIMENTAL:
        first_width = _aligned(width) * 4
        first_height = _aligned(height) * 4
    else:
        first_width = _aligned(width * 4)
        first_height = _aligned(height * 4)
    first = _pid_pass(
        graph,
        image=image,
        model=model.out(0),
        input_vae=input_vae,
        vae_name=model_names["vae"],
        vae_family=vae_family,
        pixel_vae=pixel_vae.out(0),
        positive=positive.out(0),
        negative=negative.out(0),
        sampler=sampler.out(0),
        sigmas=sigmas.out(0),
        latent_format=LATENT_FORMATS[vae_family],
        output_width=first_width,
        output_height=first_height,
        seed=seed,
        degrade_sigma=degrade_sigma,
        tiled_encode=False,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )

    final_image = first
    generated_width = first_width
    generated_height = first_height
    factor = 4
    if scale == "Experimental 16x (tiled)":
        tiled_model = graph.node(
            "ContextWindowsManual",
            model=model.out(0),
            context_length=tile_size,
            context_overlap=tile_overlap,
            context_schedule="standard_static",
            context_stride=1,
            closed_loop=False,
            fuse_method="pyramid",
            dim=2,
            freenoise=False,
            cond_retain_index_list="",
            split_conds_to_windows=False,
            latent_retain_index_list="",
            causal_window_fix=True,
        )
        generated_width = first_width * 4
        generated_height = first_height * 4
        final_image = _pid_pass(
            graph,
            image=first,
            model=tiled_model.out(0),
            input_vae=input_vae,
            vae_name=model_names["vae"],
            vae_family=vae_family,
            pixel_vae=pixel_vae.out(0),
            positive=positive.out(0),
            negative=negative.out(0),
            sampler=sampler.out(0),
            sigmas=sigmas.out(0),
            latent_format=LATENT_FORMATS[vae_family],
            output_width=generated_width,
            output_height=generated_height,
            seed=(seed + 1) % (2**63),
            degrade_sigma=degrade_sigma,
            tiled_encode=True,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        factor = 16

    exact_width = width * factor
    exact_height = height * factor
    if (generated_width, generated_height) != (exact_width, exact_height):
        final_image = graph.node(
            "ImageScale",
            image=final_image,
            upscale_method="lanczos",
            width=exact_width,
            height=exact_height,
            crop="disabled",
        ).out(0)

    return _finish(graph, final_image)


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
    tiled_encode: bool,
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
        encode_class = "VAEEncodeTiled" if tiled_encode else "VAEEncode"
        encode_inputs = {"pixels": image, "vae": input_vae}
        if tiled_encode:
            encode_inputs.update({"tile_size": tile_size, "overlap": tile_overlap})
        encoded = graph.node(encode_class, **encode_inputs)
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
