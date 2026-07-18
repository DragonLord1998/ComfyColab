from __future__ import annotations

import importlib
from typing import Any


BASE_FPS = 24
STAGE_ONE_SIGMAS = (
    "1.0, 0.99375, 0.9875, 0.98125, 0.975, "
    "0.909375, 0.725, 0.421875, 0.0"
)
STAGE_TWO_SIGMAS = "0.909375, 0.725, 0.421875, 0.0"

REQUIRED_NODES = frozenset(
    {
        "UnetLoaderGGUF",
        "DualCLIPLoaderGGUF",
        "VAELoader",
        "CLIPTextEncode",
        "LTXVConditioning",
        "EmptyLTXVLatentVideo",
        "LTXVEmptyLatentAudio",
        "LTXVConcatAVLatent",
        "RandomNoise",
        "BasicGuider",
        "KSamplerSelect",
        "ManualSigmas",
        "SamplerCustomAdvanced",
        "LTXVSeparateAVLatent",
        "LTXVTiledVAEDecode",
        "LTXVAudioVAEDecode",
        "CreateVideo",
    }
)


def _builder():
    return importlib.import_module("comfy_execution.graph_utils").GraphBuilder()


def _finish(graph, video, images, audio):
    io = importlib.import_module("comfy_api.latest").io
    return io.NodeOutput(video, images, audio, expand=graph.finalize())


def required_nodes(*, image_to_video: bool, spatial: bool, temporal: bool) -> set[str]:
    required = set(REQUIRED_NODES)
    if image_to_video:
        required.update({"LTXVPreprocess", "LTXVImgToVideoConditionOnly"})
    if spatial or temporal:
        required.update({"LatentUpscaleModelLoader", "LTXVLatentUpsampler"})
    return required


def build_ltx23_graph(
    *,
    prompt: str,
    image: Any | None,
    fps: int,
    spatial_upscaler: str,
    width: int,
    height: int,
    frame_count: int,
    seed: int,
    image_strength: float,
    model_names: dict[str, str],
):
    graph = _builder()
    model = graph.node("UnetLoaderGGUF", unet_name=model_names["model"])
    clip = graph.node(
        "DualCLIPLoaderGGUF",
        clip_name1=model_names["text_encoder"],
        clip_name2=model_names["connector"],
        type="ltxv",
    )
    video_vae = graph.node("VAELoader", vae_name=model_names["video_vae"])
    audio_vae = graph.node("VAELoader", vae_name=model_names["audio_vae"])
    positive = graph.node(
        "CLIPTextEncode",
        clip=clip.out(0),
        text=prompt,
    )
    conditioning = graph.node(
        "LTXVConditioning",
        positive=positive.out(0),
        negative=positive.out(0),
        frame_rate=float(BASE_FPS),
    )

    empty_video = graph.node(
        "EmptyLTXVLatentVideo",
        width=width,
        height=height,
        length=frame_count,
        batch_size=1,
    )
    video_latent = empty_video.out(0)
    prepared_image = None
    if image is not None:
        prepared_image = graph.node(
            "LTXVPreprocess",
            image=image,
            img_compression=18,
        )
        image_conditioned = graph.node(
            "LTXVImgToVideoConditionOnly",
            vae=video_vae.out(0),
            image=prepared_image.out(0),
            latent=video_latent,
            strength=image_strength,
            bypass=False,
        )
        video_latent = image_conditioned.out(0)

    empty_audio = graph.node(
        "LTXVEmptyLatentAudio",
        frames_number=frame_count,
        frame_rate=BASE_FPS,
        batch_size=1,
        audio_vae=audio_vae.out(0),
    )
    initial_av = graph.node(
        "LTXVConcatAVLatent",
        video_latent=video_latent,
        audio_latent=empty_audio.out(0),
    )
    stage_one_noise = graph.node("RandomNoise", noise_seed=seed)
    stage_one_guider = graph.node(
        "BasicGuider",
        model=model.out(0),
        conditioning=conditioning.out(0),
    )
    stage_one_sampler = graph.node(
        "KSamplerSelect",
        sampler_name="euler",
    )
    stage_one_sigmas = graph.node("ManualSigmas", sigmas=STAGE_ONE_SIGMAS)
    stage_one = graph.node(
        "SamplerCustomAdvanced",
        noise=stage_one_noise.out(0),
        guider=stage_one_guider.out(0),
        sampler=stage_one_sampler.out(0),
        sigmas=stage_one_sigmas.out(0),
        latent_image=initial_av.out(0),
    )
    stage_one_split = graph.node(
        "LTXVSeparateAVLatent",
        av_latent=stage_one.out(0),
    )
    final_video_latent = stage_one_split.out(0)
    final_audio_latent = stage_one_split.out(1)

    if spatial_upscaler != "None":
        spatial_model = graph.node(
            "LatentUpscaleModelLoader",
            model_name=model_names["spatial_upscaler"],
        )
        spatial_video = graph.node(
            "LTXVLatentUpsampler",
            samples=final_video_latent,
            upscale_model=spatial_model.out(0),
            vae=video_vae.out(0),
        )
        refined_video_latent = spatial_video.out(0)
        if prepared_image is not None:
            refined_image = graph.node(
                "LTXVImgToVideoConditionOnly",
                vae=video_vae.out(0),
                image=prepared_image.out(0),
                latent=refined_video_latent,
                strength=1.0,
                bypass=False,
            )
            refined_video_latent = refined_image.out(0)
        refined_av = graph.node(
            "LTXVConcatAVLatent",
            video_latent=refined_video_latent,
            audio_latent=final_audio_latent,
        )
        stage_two_noise = graph.node(
            "RandomNoise",
            noise_seed=(seed + 1) % (2**63),
        )
        stage_two_guider = graph.node(
            "BasicGuider",
            model=model.out(0),
            conditioning=conditioning.out(0),
        )
        stage_two_sampler = graph.node(
            "KSamplerSelect",
            sampler_name="euler",
        )
        stage_two_sigmas = graph.node("ManualSigmas", sigmas=STAGE_TWO_SIGMAS)
        stage_two = graph.node(
            "SamplerCustomAdvanced",
            noise=stage_two_noise.out(0),
            guider=stage_two_guider.out(0),
            sampler=stage_two_sampler.out(0),
            sigmas=stage_two_sigmas.out(0),
            latent_image=refined_av.out(0),
        )
        stage_two_split = graph.node(
            "LTXVSeparateAVLatent",
            av_latent=stage_two.out(0),
        )
        final_video_latent = stage_two_split.out(0)
        final_audio_latent = stage_two_split.out(1)

    if fps == 48:
        temporal_model = graph.node(
            "LatentUpscaleModelLoader",
            model_name=model_names["temporal_2x"],
        )
        temporal_video = graph.node(
            "LTXVLatentUpsampler",
            samples=final_video_latent,
            upscale_model=temporal_model.out(0),
            vae=video_vae.out(0),
        )
        final_video_latent = temporal_video.out(0)

    decoded_images = graph.node(
        "LTXVTiledVAEDecode",
        vae=video_vae.out(0),
        latents=final_video_latent,
        horizontal_tiles=2,
        vertical_tiles=2,
        overlap=6,
        last_frame_fix=False,
        working_device="auto",
        working_dtype="auto",
    )
    decoded_audio = graph.node(
        "LTXVAudioVAEDecode",
        samples=final_audio_latent,
        audio_vae=audio_vae.out(0),
    )
    video = graph.node(
        "CreateVideo",
        images=decoded_images.out(0),
        audio=decoded_audio.out(0),
        fps=float(fps),
    )
    return _finish(
        graph,
        video.out(0),
        decoded_images.out(0),
        decoded_audio.out(0),
    )
