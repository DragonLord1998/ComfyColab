from __future__ import annotations

import importlib
from typing import Any


BASE_FPS = 24
BASE_MAX_AREA = 768 * 1344
MAX_SEED = (2**63) - 1

FL2VA_REQUIRED_NODES = frozenset(
    {
        "MiniMaxH3ImageToVideo",
        "RandomNoise",
        "BasicGuider",
        "KSamplerSelect",
        "BasicScheduler",
        "SamplerCustomAdvanced",
        "VAEDecode",
        "VAEDecodeAudio",
        "CreateVideo",
    }
)
REF2VA_REQUIRED_NODES = frozenset(
    {
        "MiniMaxH3ReferenceToVideo",
        *FL2VA_REQUIRED_NODES,
    }
)


def _builder():
    return importlib.import_module("comfy_execution.graph_utils").GraphBuilder()


def _finish(graph, video, images, audio):
    io = importlib.import_module("comfy_api.latest").io
    return io.NodeOutput(video, images, audio, expand=graph.finalize())


def required_h3_nodes(reference: bool = False) -> set[str]:
    return set(REF2VA_REQUIRED_NODES if reference else FL2VA_REQUIRED_NODES)


def snap_h3_frames(duration_seconds: float) -> int:
    requested = max(5, round(float(duration_seconds) * BASE_FPS))
    return requested + (5 - (requested % 17)) % 17


def validate_h3_common(
    *,
    prompt: str,
    duration_seconds: float,
    width: int,
    height: int,
    seed: int,
) -> int:
    if not str(prompt).strip():
        raise ValueError("MiniMax H3 requires a non-empty prompt.")
    if not 4.0 <= float(duration_seconds) <= 15.0:
        raise ValueError("MiniMax H3 duration_seconds must be between 4 and 15.")
    if width % 32 or height % 32:
        raise ValueError("MiniMax H3 width and height must be divisible by 32.")
    if width * height > BASE_MAX_AREA:
        raise ValueError("MiniMax H3 Base canvas must not exceed 768 x 1344 pixels.")
    if seed < 0 or seed > MAX_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_SEED}.")
    return snap_h3_frames(duration_seconds)


def require_h3_bundle(bundle: dict[str, Any], expected_variant: str) -> None:
    if not isinstance(bundle, dict) or bundle.get("family") != "minimax_h3":
        raise ValueError("MiniMax H3 requires a bundle from MiniMax H3 Bundle Loader.")
    variant = bundle.get("variant")
    if variant != expected_variant:
        target = "Reference Video node" if variant == "Ref2VA" else "Text/Image Video node"
        raise ValueError(f"MiniMax H3 {variant or 'unknown'} bundle is incompatible; use the {target}.")


def validate_reference_inputs(
    *,
    ref_images: dict[str, Any],
    ref_videos: dict[str, Any],
    ref_video_audios: dict[str, Any],
    ref_audios: dict[str, Any],
) -> None:
    if not ref_images and not ref_videos:
        raise ValueError("MiniMax H3 Ref2VA requires at least one reference image or video.")
    if len(ref_images) > 9:
        raise ValueError("MiniMax H3 Ref2VA accepts at most 9 reference images.")
    if len(ref_videos) > 3:
        raise ValueError("MiniMax H3 Ref2VA accepts at most 3 reference videos.")
    for key in ref_video_audios:
        index = _reference_index(key)
        video_key = _matching_reference_key(ref_videos, index)
        if video_key is None:
            raise ValueError("Paired reference audio requires the matching reference video.")
    if len(ref_audios) > 3:
        raise ValueError("MiniMax H3 Ref2VA accepts at most 3 standalone audio clips.")
    total_files = len(ref_images) + len(ref_videos) + len(ref_video_audios) + len(ref_audios)
    if total_files > 12:
        raise ValueError("MiniMax H3 Ref2VA accepts at most 12 reference files total.")
    video_durations = [_video_duration_seconds(video) for video in ref_videos.values()]
    audio_durations = [
        _audio_duration_seconds(audio)
        for audio in [*ref_video_audios.values(), *ref_audios.values()]
    ]
    for duration in video_durations:
        if not 2.0 <= duration <= 15.0:
            raise ValueError("MiniMax H3 Ref2VA reference videos must be 2-15 seconds at 24 FPS.")
    for duration in audio_durations:
        if not 2.0 <= duration <= 15.0:
            raise ValueError("MiniMax H3 Ref2VA reference audio clips must be 2-15 seconds.")
    if sum(video_durations) > 15.0:
        raise ValueError("MiniMax H3 Ref2VA total reference-video duration must be at most 15 seconds.")
    if sum(audio_durations) > 15.0:
        raise ValueError("MiniMax H3 Ref2VA total reference-audio duration must be at most 15 seconds.")


def _reference_index(key: str) -> str:
    digits = "".join(character for character in str(key) if character.isdigit())
    return digits or str(key)


def _matching_reference_key(values: dict[str, Any], index: str) -> str | None:
    for key in values:
        if _reference_index(key) == index:
            return key
    return None


def _video_duration_seconds(video: Any) -> float:
    if isinstance(video, dict):
        if "frames" in video:
            return _frame_count(video["frames"]) / BASE_FPS
        raise ValueError("MiniMax H3 Ref2VA reference video duration is unavailable.")
    return _frame_count(video) / BASE_FPS


def _frame_count(frames: Any) -> int:
    if hasattr(frames, "shape") and frames.shape:
        return int(frames.shape[0])
    try:
        return len(frames)
    except TypeError as error:
        raise ValueError("MiniMax H3 Ref2VA reference video duration is unavailable.") from error


def _audio_duration_seconds(audio: Any) -> float:
    if not isinstance(audio, dict):
        raise ValueError("MiniMax H3 Ref2VA reference audio must include waveform and sample_rate.")
    sample_rate = int(audio.get("sample_rate") or 0)
    waveform = audio.get("waveform")
    if sample_rate <= 0 or waveform is None:
        raise ValueError("MiniMax H3 Ref2VA reference audio must include waveform and sample_rate.")
    samples = _sample_count(waveform)
    return samples / sample_rate


def _sample_count(waveform: Any) -> int:
    if hasattr(waveform, "shape") and waveform.shape:
        return int(waveform.shape[-1])
    try:
        return len(waveform)
    except TypeError as error:
        raise ValueError("MiniMax H3 Ref2VA reference audio duration is unavailable.") from error


def _sampling_tail(
    graph,
    *,
    model,
    conditioning,
    latent,
    video_vae,
    audio_vae,
    seed: int,
    scheduler: str,
):
    noise = graph.node("RandomNoise", noise_seed=seed)
    guider = graph.node("BasicGuider", model=model, conditioning=conditioning)
    sampler = graph.node("KSamplerSelect", sampler_name="res_multistep")
    sigmas = graph.node(
        "BasicScheduler",
        model=model,
        scheduler=scheduler,
        steps=20,
        denoise=1.0,
    )
    sampled = graph.node(
        "SamplerCustomAdvanced",
        noise=noise.out(0),
        guider=guider.out(0),
        sampler=sampler.out(0),
        sigmas=sigmas.out(0),
        latent_image=latent,
    )
    decoded_images = graph.node("VAEDecode", samples=sampled.out(0), vae=video_vae)
    decoded_audio = graph.node(
        "VAEDecodeAudio",
        samples=sampled.out(0),
        vae=audio_vae,
    )
    video = graph.node(
        "CreateVideo",
        images=decoded_images.out(0),
        audio=decoded_audio.out(0),
        fps=float(BASE_FPS),
    )
    return video.out(0), decoded_images.out(0), decoded_audio.out(0)


def build_h3_fl2va_graph(
    *,
    bundle: dict[str, Any],
    prompt: str,
    first_frame: Any | None,
    last_frame: Any | None,
    duration_seconds: float,
    width: int,
    height: int,
    seed: int,
):
    require_h3_bundle(bundle, "FL2VA")
    frame_count = validate_h3_common(
        prompt=prompt,
        duration_seconds=duration_seconds,
        width=width,
        height=height,
        seed=seed,
    )
    graph = _builder()
    conditioning = graph.node(
        "MiniMaxH3ImageToVideo",
        clip=bundle["text_encoder"],
        vae=bundle["video_vae"],
        prompt=prompt,
        width=width,
        height=height,
        length=frame_count,
        first_frame=first_frame,
        last_frame=last_frame,
    )
    video, images, audio = _sampling_tail(
        graph,
        model=bundle["model"],
        conditioning=conditioning.out(0),
        latent=conditioning.out(1),
        video_vae=bundle["video_vae"],
        audio_vae=bundle["audio_vae"],
        seed=seed,
        scheduler="simple",
    )
    return _finish(graph, video, images, audio)


def build_h3_ref2va_graph(
    *,
    bundle: dict[str, Any],
    prompt: str,
    ref_images: dict[str, Any],
    ref_videos: dict[str, Any],
    ref_video_audios: dict[str, Any],
    ref_audios: dict[str, Any],
    duration_seconds: float,
    width: int,
    height: int,
    seed: int,
    ref_image_size: str,
    scheduler: str,
):
    require_h3_bundle(bundle, "Ref2VA")
    frame_count = validate_h3_common(
        prompt=prompt,
        duration_seconds=duration_seconds,
        width=width,
        height=height,
        seed=seed,
    )
    validate_reference_inputs(
        ref_images=ref_images,
        ref_videos=ref_videos,
        ref_video_audios=ref_video_audios,
        ref_audios=ref_audios,
    )
    graph = _builder()
    conditioning = graph.node(
        "MiniMaxH3ReferenceToVideo",
        clip=bundle["text_encoder"],
        vae=bundle["video_vae"],
        audio_vae=bundle["audio_vae"],
        prompt=prompt,
        width=width,
        height=height,
        length=frame_count,
        ref_image_size=ref_image_size,
        ref_images=ref_images,
        ref_videos=ref_videos,
        ref_video_audios=ref_video_audios,
        ref_audios=ref_audios,
    )
    video, images, audio = _sampling_tail(
        graph,
        model=bundle["model"],
        conditioning=conditioning.out(0),
        latent=conditioning.out(1),
        video_vae=bundle["video_vae"],
        audio_vae=bundle["audio_vae"],
        seed=seed,
        scheduler=scheduler,
    )
    return _finish(graph, video, images, audio)
