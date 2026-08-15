from __future__ import annotations

import importlib
from typing import Any

from .catalog import gguf_names, normalize_h3_variant, spatial_upscaler_names
from .graph import build_ltx23_graph, required_nodes
from .graph_h3 import (
    build_h3_fl2va_graph,
    build_h3_ref2va_graph,
    required_h3_nodes,
    validate_reference_inputs,
)
from .h3_prompt_policy import PROMPT_MODE_LABELS
from .h3_prompt_worker import enhance_h3_prompt
from .models import ensure_h3_model_assets, ensure_model_assets


MAX_SEED = (2**63) - 1
FPS_OPTIONS = ["24", "48"]
H3_VARIANT_LABELS = [
    "FL2VA — Text / First / Last Frame",
    "Ref2VA — Reference Images / Video / Audio",
]
H3_REF_IMAGE_SIZE_OPTIONS = ["match", "max"]
H3_REF_SCHEDULER_OPTIONS = ["beta", "normal", "simple"]
H3_LOADER_NODES = {"UNETLoader", "CLIPLoader", "VAELoader"}
H3_ATTENTION_BACKEND = "sage"
H3_PROMPT_MAX_SEED = (2**31) - 1


def _io():
    return importlib.import_module("comfy_api.latest").io


def _missing_upstream_nodes(node_ids: set[str]) -> list[str]:
    try:
        registry = importlib.import_module("nodes").NODE_CLASS_MAPPINGS
    except (ModuleNotFoundError, AttributeError):
        return []
    return sorted(node_ids - set(registry))


def _require_upstream_nodes(node_ids: set[str]) -> None:
    missing = _missing_upstream_nodes(node_ids)
    if missing:
        raise RuntimeError(
            "ComfyColab LTX-2.3 requires the pinned ComfyUI, ComfyUI-GGUF, and "
            "Lighttricks ComfyUI-LTXVideo node packs. Missing node IDs: "
            f"{', '.join(missing)}. Restart with `comfycolab start --refresh`."
        )


def _require_h3_upstream_nodes(node_ids: set[str]) -> None:
    missing = _missing_upstream_nodes(node_ids)
    if missing:
        raise RuntimeError(
            "MiniMax H3 requires the pinned H3-capable ComfyUI engine. "
            "Missing node IDs: "
            f"{', '.join(missing)}. Restart with `comfycolab start --refresh`."
        )


def _loader(comfy_nodes: Any, name: str) -> Any:
    loader_class = comfy_nodes.NODE_CLASS_MAPPINGS.get(name)
    if loader_class is None:
        raise RuntimeError(
            f"Required loader '{name}' is unavailable. Restart with "
            "`comfycolab start --refresh`."
        )
    return loader_class()


def _h3_sage_attention() -> Any:
    try:
        attention = importlib.import_module("comfy.ldm.modules.attention")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "MiniMax H3 requires the G4/SM120 SageAttention runtime. Select a "
            "G4 runtime and restart with `comfycolab start --refresh`."
        ) from error
    getter = getattr(attention, "get_attention_function", None)
    sage = getter(H3_ATTENTION_BACKEND, None) if callable(getter) else None
    if sage is None:
        raise RuntimeError(
            "MiniMax H3 requires the G4/SM120 SageAttention 2.2.0 runtime. "
            "Select a G4 runtime and restart with `comfycolab start --refresh`."
        )
    return sage


def _patch_h3_sage_attention(model: Any, sage_attention: Any) -> Any:
    if not hasattr(model, "clone") or not hasattr(model, "model_options"):
        raise RuntimeError("MiniMax H3 received an incompatible ComfyUI model patcher.")
    patched = model.clone()
    transformer_options = patched.model_options.setdefault("transformer_options", {})
    sage_impl = getattr(sage_attention, "__wrapped__", sage_attention)

    def attention_override(_current_attention, *args, **kwargs):
        return sage_impl(*args, **kwargs)

    transformer_options["optimized_attention_override"] = attention_override
    return patched


def _video_output(io: Any):
    return io.Video.Output("video")


def _h3_bundle_output(io: Any):
    return io.Custom("MINIMAX_H3_BUNDLE").Output("bundle")


def _release_comfy_gpu_models() -> None:
    try:
        model_management = importlib.import_module("comfy.model_management")
    except ModuleNotFoundError:
        return
    unload = getattr(model_management, "unload_all_models", None)
    if callable(unload):
        unload()
    empty_cache = getattr(model_management, "soft_empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def _collect_autogrow(values: Any) -> dict[str, Any]:
    if values is None:
        return {}
    if isinstance(values, dict):
        return {str(key): value for key, value in values.items() if value is not None}
    if isinstance(values, (list, tuple)):
        return {
            str(index): value
            for index, value in enumerate(values, start=1)
            if value is not None
        }
    raise ValueError("MiniMax H3 reference inputs must be expandable input groups.")


def _h3_bundle(
    *,
    variant: str,
    model: Any,
    text_encoder: Any,
    video_vae: Any,
    audio_vae: Any,
    filenames: dict[str, str],
) -> dict[str, Any]:
    return {
        "family": "minimax_h3",
        "variant": variant,
        "model": model,
        "text_encoder": text_encoder,
        "video_vae": video_vae,
        "audio_vae": audio_vae,
        "attention_backend": H3_ATTENTION_BACKEND,
        "filenames": dict(filenames),
    }


class ComfyColabMiniMaxH3BundleLoader:
    @classmethod
    def define_schema(cls):
        io = _io()
        return io.Schema(
            node_id="ComfyColabMiniMaxH3BundleLoader",
            display_name="MiniMax H3 Bundle Loader",
            category="ComfyColab/loaders",
            description=(
                "Downloads and loads one official optimized MiniMax H3 Base "
                "variant plus the shared Qwen3-VL text encoder, video VAE, and "
                "audio VAE. The H3 diffusion model uses model-scoped SageAttention "
                "for faster video sampling. Requires explicit H3 license acknowledgement."
            ),
            inputs=[
                io.Combo.Input(
                    "model_variant",
                    options=H3_VARIANT_LABELS,
                    default=H3_VARIANT_LABELS[0],
                    tooltip="FL2VA is for text, first frame, last frame, or both. Ref2VA is for reference media.",
                ),
                io.Boolean.Input(
                    "accept_h3_license",
                    default=False,
                    tooltip="Confirm that you reviewed the MiniMax H3 Community License.",
                ),
                io.Boolean.Input(
                    "force_redownload",
                    default=False,
                    advanced=True,
                    tooltip="Discard resumable cached files and download selected H3 assets again.",
                ),
            ],
            outputs=[
                _h3_bundle_output(io),
                io.Model.Output("model"),
                io.Clip.Output("text_encoder"),
                io.Vae.Output("video_vae"),
                io.Vae.Output("audio_vae"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_variant=H3_VARIANT_LABELS[0],
        accept_h3_license=False,
        force_redownload=False,
    ):
        if not accept_h3_license:
            raise PermissionError(
                "MiniMax H3 download requires accept_h3_license=true after reviewing "
                "the MiniMax H3 Community License."
            )
        variant = normalize_h3_variant(str(model_variant).split(" ", 1)[0])
        _require_h3_upstream_nodes(
            H3_LOADER_NODES
            | required_h3_nodes(reference=False)
            | required_h3_nodes(reference=True)
        )
        sage_attention = _h3_sage_attention()
        filenames = ensure_h3_model_assets(
            variant,
            force_redownload=bool(force_redownload),
        )
        comfy_nodes = importlib.import_module("nodes")
        model = _loader(comfy_nodes, "UNETLoader").load_unet(
            filenames["model"],
            weight_dtype="default",
        )[0]
        model = _patch_h3_sage_attention(model, sage_attention)
        text_encoder = _loader(comfy_nodes, "CLIPLoader").load_clip(
            filenames["text_encoder"],
            type="minimax",
        )[0]
        video_vae = _loader(comfy_nodes, "VAELoader").load_vae(
            filenames["video_vae"]
        )[0]
        audio_vae = _loader(comfy_nodes, "VAELoader").load_vae(
            filenames["audio_vae"]
        )[0]
        bundle = _h3_bundle(
            variant=variant,
            model=model,
            text_encoder=text_encoder,
            video_vae=video_vae,
            audio_vae=audio_vae,
            filenames=filenames,
        )
        return bundle, model, text_encoder, video_vae, audio_vae


class ComfyColabMiniMaxH3PromptEnhancer:
    @classmethod
    def define_schema(cls):
        io = _io()
        return io.Schema(
            node_id="ComfyColabMiniMaxH3PromptEnhancer",
            display_name="ComfyColab MiniMax H3 — Prompt Enhancer",
            category="ComfyColab/prompt",
            description=(
                "Rewrites a user prompt with thinking-enabled Qwen3.8-27B Q4_K_M "
                "into the exact official MiniMax H3 Base or Ref2VA prompt structure. "
                "The isolated llama.cpp process exits before H3 sampling."
            ),
            inputs=[
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Your plain-language video idea. This output connects to an H3 prompt input.",
                ),
                io.Combo.Input(
                    "prompt_mode",
                    options=PROMPT_MODE_LABELS,
                    default=PROMPT_MODE_LABELS[0],
                    tooltip="Match this to the H3 input path and connected reference media.",
                ),
                io.Float.Input(
                    "duration_seconds",
                    default=5.0,
                    min=4.0,
                    max=15.0,
                    step=0.25,
                    tooltip="Must match the duration configured on the downstream H3 node.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=H3_PROMPT_MAX_SEED,
                    advanced=True,
                ),
                io.Int.Input(
                    "max_tokens",
                    default=8192,
                    min=4096,
                    max=16384,
                    step=256,
                    advanced=True,
                    tooltip=(
                        "Includes Qwen's bounded private reasoning plus the final "
                        "structured prompt."
                    ),
                ),
                io.Float.Input(
                    "temperature",
                    default=1.0,
                    min=0.0,
                    max=1.5,
                    step=0.05,
                    advanced=True,
                    tooltip="Qwen3.8 thinking-mode sampling temperature.",
                ),
                io.Boolean.Input(
                    "force_redownload",
                    default=False,
                    advanced=True,
                    tooltip="Download and verify the pinned 17.1 GB Q4_K_M GGUF again.",
                ),
            ],
            outputs=[io.String.Output("enhanced_prompt")],
        )

    @classmethod
    def execute(
        cls,
        prompt,
        prompt_mode=PROMPT_MODE_LABELS[0],
        duration_seconds=5.0,
        seed=0,
        max_tokens=8192,
        temperature=1.0,
        force_redownload=False,
    ):
        _release_comfy_gpu_models()
        enhanced = enhance_h3_prompt(
            str(prompt),
            str(prompt_mode),
            float(duration_seconds),
            seed=int(seed),
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            force_redownload=bool(force_redownload),
        )
        return (enhanced,)


class ComfyColabMiniMaxH3Video:
    @classmethod
    def define_schema(cls):
        io = _io()
        return io.Schema(
            node_id="ComfyColabMiniMaxH3Video",
            display_name="ComfyColab MiniMax H3 — Text/Image to Video",
            category="ComfyColab/Video",
            description=(
                "Generates 24 FPS audio-video with MiniMax H3 FL2VA from text, "
                "an optional first frame, an optional last frame, or both."
            ),
            enable_expand=True,
            inputs=[
                io.Custom("MINIMAX_H3_BUNDLE").Input("bundle"),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Describe the shot, motion, dialogue, and sounds to generate.",
                ),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Float.Input(
                    "duration_seconds",
                    default=5.0,
                    min=4.0,
                    max=15.0,
                    step=0.25,
                    tooltip="H3 runs at 24 FPS and snaps upward to the valid 17k+5 frame grid.",
                ),
                io.Int.Input("width", default=864, min=256, max=1344, step=32),
                io.Int.Input("height", default=480, min=256, max=1344, step=32),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
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
        bundle,
        prompt,
        first_frame=None,
        last_frame=None,
        duration_seconds=5.0,
        width=864,
        height=480,
        seed=0,
    ):
        _require_h3_upstream_nodes(required_h3_nodes(reference=False))
        return build_h3_fl2va_graph(
            bundle=bundle,
            prompt=str(prompt),
            first_frame=first_frame,
            last_frame=last_frame,
            duration_seconds=float(duration_seconds),
            width=int(width),
            height=int(height),
            seed=int(seed),
        )


class ComfyColabMiniMaxH3ReferenceVideo:
    @classmethod
    def define_schema(cls):
        io = _io()
        ref_images = io.Autogrow.Input(
            "ref_images",
            template=io.Autogrow.TemplatePrefix(
                input=io.Image.Input("ref_image"),
                prefix="ref_image_",
                min=0,
                max=9,
            ),
        )
        ref_videos = io.Autogrow.Input(
            "ref_videos",
            template=io.Autogrow.TemplatePrefix(
                input=io.Image.Input("ref_video"),
                prefix="ref_video_",
                min=0,
                max=3,
            ),
        )
        ref_video_audios = io.Autogrow.Input(
            "ref_video_audios",
            template=io.Autogrow.TemplatePrefix(
                input=io.Audio.Input("ref_video_audio"),
                prefix="ref_video_audio_",
                min=0,
                max=3,
            ),
        )
        ref_audios = io.Autogrow.Input(
            "ref_audios",
            template=io.Autogrow.TemplatePrefix(
                input=io.Audio.Input("ref_audio"),
                prefix="ref_audio_",
                min=0,
                max=3,
            ),
        )
        return io.Schema(
            node_id="ComfyColabMiniMaxH3ReferenceVideo",
            display_name="ComfyColab MiniMax H3 — Reference to Video",
            category="ComfyColab/Video",
            description=(
                "Generates 24 FPS audio-video with MiniMax H3 Ref2VA from ordered "
                "reference pictures, videos, paired video audio, and standalone audio. "
                "Prompt tags use <Picture 1>, <Video 1>, and <Audio 1> order."
            ),
            enable_expand=True,
            inputs=[
                io.Custom("MINIMAX_H3_BUNDLE").Input("bundle"),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Use one-based tags such as <Picture 1>, <Video 1>, and <Audio 1>.",
                ),
                io.Float.Input(
                    "duration_seconds",
                    default=5.0,
                    min=4.0,
                    max=15.0,
                    step=0.25,
                ),
                io.Int.Input("width", default=864, min=256, max=1344, step=32),
                io.Int.Input("height", default=480, min=256, max=1344, step=32),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Combo.Input(
                    "ref_image_size",
                    options=H3_REF_IMAGE_SIZE_OPTIONS,
                    default="match",
                    advanced=True,
                ),
                io.Combo.Input(
                    "scheduler",
                    options=H3_REF_SCHEDULER_OPTIONS,
                    default="beta",
                    advanced=True,
                ),
                ref_images,
                ref_videos,
                ref_video_audios,
                ref_audios,
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
        bundle,
        prompt,
        duration_seconds=5.0,
        width=864,
        height=480,
        seed=0,
        ref_image_size="match",
        scheduler="beta",
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
    ):
        ref_images = _collect_autogrow(ref_images)
        ref_videos = _collect_autogrow(ref_videos)
        ref_video_audios = _collect_autogrow(ref_video_audios)
        ref_audios = _collect_autogrow(ref_audios)
        validate_reference_inputs(
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_video_audios=ref_video_audios,
            ref_audios=ref_audios,
        )
        if ref_image_size not in H3_REF_IMAGE_SIZE_OPTIONS:
            raise ValueError("MiniMax H3 ref_image_size must be match or max.")
        if scheduler not in H3_REF_SCHEDULER_OPTIONS:
            raise ValueError("MiniMax H3 scheduler must be beta, normal, or simple.")
        _require_h3_upstream_nodes(required_h3_nodes(reference=True))
        return build_h3_ref2va_graph(
            bundle=bundle,
            prompt=str(prompt),
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_video_audios=ref_video_audios,
            ref_audios=ref_audios,
            duration_seconds=float(duration_seconds),
            width=int(width),
            height=int(height),
            seed=int(seed),
            ref_image_size=str(ref_image_size),
            scheduler=str(scheduler),
        )


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
    "ComfyColabMiniMaxH3BundleLoader": ComfyColabMiniMaxH3BundleLoader,
    "ComfyColabMiniMaxH3PromptEnhancer": ComfyColabMiniMaxH3PromptEnhancer,
    "ComfyColabMiniMaxH3Video": ComfyColabMiniMaxH3Video,
    "ComfyColabMiniMaxH3ReferenceVideo": ComfyColabMiniMaxH3ReferenceVideo,
}

NODE_CLASS_MAPPINGS = dict(PUBLIC_NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyColabLTX23Video": "ComfyColab LTX-2.3 — Text/Image to Video",
    "ComfyColabMiniMaxH3BundleLoader": "MiniMax H3 Bundle Loader",
    "ComfyColabMiniMaxH3PromptEnhancer": "ComfyColab MiniMax H3 — Prompt Enhancer",
    "ComfyColabMiniMaxH3Video": "ComfyColab MiniMax H3 — Text/Image to Video",
    "ComfyColabMiniMaxH3ReferenceVideo": "ComfyColab MiniMax H3 — Reference to Video",
}
