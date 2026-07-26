from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .cache import (
    cache_path,
    canonical_glb_geometry_digest,
    canonical_trimesh_digest,
    deterministic_cache_key,
    cubepart_cache_key,
    meshflow_cache_key,
    pixal3d_cache_key,
    pixal3d_multiview_cache_key,
    skintokens_cache_key,
    texture_cache_key,
    trellis_cache_key,
    trellis_multiview_cache_key,
    ultrashape_geometry_cache_key,
)
from .file3d import copy_file3d_to, export_trimesh_atomic, load_glb_trimesh, materialize_file3d, publish_glb
from .geometry_quality import validate_volumetric_glb, validate_volumetric_mesh
from .graph import (
    COMFYUI_REF,
    BIREFNET_MODEL_REF,
    TRELLIS_PATCH_ID,
    TRELLIS_MULTIVIEW_PATCH_ID,
    TRELLIS_WRAPPER_REF,
    build_trellis_graph,
    build_trellis_multiview_graph,
    build_pixal3d_graph,
    build_pixal3d_multiview_graph,
    build_ultrashape_cached_geometry_graph,
    build_ultrashape_graph,
)
from .presets import (
    CACHE_MODES,
    PIXAL3D_EXPERIMENTAL_PRESETS,
    PIXAL3D_PRESETS,
    RESOLUTION_OVERRIDES,
    TRELLIS_PRESETS,
    ULTRASHAPE_EXPERIMENTAL_PRESETS,
    ULTRASHAPE_PRESETS,
    resolve_trellis_settings,
    resolve_pixal3d_settings,
    resolve_ultrashape_settings,
)
from .transforms import Normalization, normalization_for
from .worker import (
    UltraShapeCommand,
    atomic_replace_cache_directory,
    run_ultrashape_worker,
    validate_geometry_cache_record,
    write_geometry_cache_record,
)
from .pixal3d_worker import RESULT_PREFIX, Pixal3DWorkerCommand, global_pixal3d_worker_pool
from .skintokens_worker import (
    SkinTokensWorkerCommand,
    global_skintokens_worker_pool,
    validate_skintokens_output,
)
from .cubepart_worker import (
    CubePartWorkerCommand,
    global_cubepart_worker_pool,
    normalize_part_names,
    validate_cubepart_output,
)
from .meshflow_worker import MeshFlowWorkerCommand, run_meshflow_worker

ULTRASHAPE_SOURCE_REF = "5e8dcef05df101ab00ab6cd5fdd0ed0c74fbca66"
DEFAULT_ULTRASHAPE_SOURCE = "/content/UltraShape-1.0"
DEFAULT_ULTRASHAPE_PYTHON = str(Path.home() / ".ce/.pixi/envs/trellis2-nodes/bin/python")
DEFAULT_PIXAL3D_SOURCE = "/content/Pixal3D"
DEFAULT_PIXAL3D_PYTHON = str(Path.home() / ".ce/.pixi/envs/pixal3d-worker/bin/python")
DEFAULT_MESHFLOW_SOURCE = "/content/meshflow"
TRANSFORM_SCHEMA = "comfycolab-3d-transform-v1"

CANONICAL_VIEW_PROMPTS = {
    "front": (
        "Preserve the exact same object identity, proportions, materials, colors, and "
        "details. Render one isolated orthographic FRONT view, camera centered at zero "
        "elevation, full object visible, neutral lighting, plain background. Do not add, "
        "remove, mirror, crop, stylize, or redesign anything."
    ),
    "back": (
        "Preserve the exact same object identity, proportions, materials, colors, and "
        "details. Rotate the same object exactly 180 degrees and render one isolated "
        "orthographic BACK view, camera centered at zero elevation, full object visible, "
        "neutral lighting, plain background. Do not add, remove, mirror, crop, stylize, "
        "or redesign anything."
    ),
    "left": (
        "Preserve the exact same object identity, proportions, materials, colors, and "
        "details. Rotate the same object exactly 90 degrees to show its LEFT side and "
        "render one isolated orthographic view, camera centered at zero elevation, full "
        "object visible, neutral lighting, plain background. Do not add, remove, mirror, "
        "crop, stylize, or redesign anything."
    ),
    "right": (
        "Preserve the exact same object identity, proportions, materials, colors, and "
        "details. Rotate the same object exactly 90 degrees to show its RIGHT side and "
        "render one isolated orthographic view, camera centered at zero elevation, full "
        "object visible, neutral lighting, plain background. Do not add, remove, mirror, "
        "crop, stylize, or redesign anything."
    ),
}


def _io():
    return importlib.import_module("comfy_api.latest").io


def _cache_root() -> Path:
    root = os.environ.get("COMFYCOLAB_3D_CACHE")
    return Path(root) if root else Path("/content/.comfycolab/cache/3d")


def _hidden_value(node_class: type, name: str):
    return getattr(getattr(node_class, "hidden", None), name, None)


def _send_progress_text(node_id: str | None, text: str) -> None:
    if not node_id:
        return
    try:
        prompt_server = importlib.import_module("server").PromptServer
        prompt_server.instance.send_progress_text(text, node_id)
    except (AttributeError, ModuleNotFoundError):
        pass


def _connected_preview_target(
    prompt: dict[str, Any] | None,
    source_node_id: str | None,
    source_slot: int = 0,
) -> dict[str, Any] | None:
    if not isinstance(prompt, dict) or source_node_id is None:
        return None
    expected_link = [str(source_node_id), source_slot]
    for node_id, node in prompt.items():
        class_type = node.get("class_type")
        if class_type not in {"Preview3D", "Preview3DAdvanced"}:
            continue
        inputs = node.get("inputs", {})
        model_input = "model_3d" if class_type == "Preview3DAdvanced" else "model_file"
        value = inputs.get(model_input)
        if isinstance(value, (list, tuple)) and list(value) == expected_link:
            return {
                "node_id": str(node_id),
                "class_type": class_type,
                "inputs": dict(inputs),
            }
    return None


def _make_temp_directory(prefix: str) -> Path:
    try:
        comfy_temp = Path(importlib.import_module("folder_paths").get_temp_directory())
    except (ModuleNotFoundError, AttributeError):
        return Path(tempfile.mkdtemp(prefix=prefix))
    comfy_temp.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=comfy_temp))


def _remove_owned_ultrashape_temp(path: str | Path) -> None:
    parent = Path(path).resolve().parent
    roots = {Path(tempfile.gettempdir()).resolve()}
    try:
        roots.add(Path(importlib.import_module("folder_paths").get_temp_directory()).resolve())
    except (ModuleNotFoundError, AttributeError):
        pass
    if (
        Path(path).name in {"refined.glb", "geometry.glb"}
        and parent.name.startswith("comfycolab-ultrashape-")
        and parent.parent in roots
    ):
        shutil.rmtree(parent, ignore_errors=True)


def _remove_owned_pixal3d_temp(path: str | Path) -> None:
    resolved = Path(path).resolve()
    parent = resolved.parent
    roots = {Path(tempfile.gettempdir()).resolve()}
    try:
        roots.add(Path(importlib.import_module("folder_paths").get_temp_directory()).resolve())
    except (ModuleNotFoundError, AttributeError):
        pass
    if (
        resolved.name == "model.glb"
        and parent.name.startswith("comfycolab-pixal3d-")
        and parent.parent in roots
    ):
        shutil.rmtree(parent, ignore_errors=True)


def _remove_owned_pixal3d_surface_temp(path: str | Path) -> None:
    resolved = Path(path).resolve()
    parent = resolved.parent
    roots = {Path(tempfile.gettempdir()).resolve()}
    try:
        roots.add(Path(importlib.import_module("folder_paths").get_temp_directory()).resolve())
    except (ModuleNotFoundError, AttributeError):
        pass
    if (
        resolved.name == "surface_point_cloud.ply"
        and parent.name.startswith("comfycolab-pixal3d-surface-")
        and parent.parent in roots
    ):
        shutil.rmtree(parent, ignore_errors=True)


def _remove_owned_meshflow_temp(path: str | Path) -> None:
    resolved = Path(path).resolve()
    parent = resolved.parent
    roots = {Path(tempfile.gettempdir()).resolve()}
    try:
        roots.add(Path(importlib.import_module("folder_paths").get_temp_directory()).resolve())
    except (ModuleNotFoundError, AttributeError):
        pass
    if (
        resolved.name == "meshflow.glb"
        and parent.name.startswith("comfycolab-meshflow-")
        and parent.parent in roots
    ):
        shutil.rmtree(parent, ignore_errors=True)


def _require_upstream_nodes(node_ids: set[str]) -> None:
    try:
        registry = importlib.import_module("nodes").NODE_CLASS_MAPPINGS
    except (ModuleNotFoundError, AttributeError):
        return
    missing = sorted(node_ids - set(registry))
    if missing:
        raise RuntimeError(
            "ComfyColab 3D requires the pinned ComfyUI core and node packs. "
            f"Missing node IDs: {', '.join(missing)}. Restart with `comfycolab start --refresh`."
        )


def _resolve_lora_path(name: str) -> Path:
    folder_paths = importlib.import_module("folder_paths")
    resolver = getattr(folder_paths, "get_full_path_or_raise", None)
    if callable(resolver):
        return Path(resolver("loras", name))
    path = folder_paths.get_full_path("loras", name)
    if not path:
        raise FileNotFoundError(
            f"FLUX.2 canonical-view LoRA is missing from ComfyUI/models/loras: {name}"
        )
    return Path(path)


def _load_worker_artifact_provisioner(repo_root: Path, lane: str):
    module_name = f"comfycolab_{lane}_artifacts"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = repo_root / "worker" / lane / "artifacts.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {lane} artifact provisioner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _worker_callbacks():
    try:
        model_management = importlib.import_module("comfy.model_management")
        progress_bar = importlib.import_module("comfy.utils").ProgressBar(100)
    except (AttributeError, ModuleNotFoundError):
        model_management = None
        progress_bar = None

    def progress(event: dict) -> None:
        if model_management is not None:
            model_management.throw_exception_if_processing_interrupted()
        if progress_bar is not None:
            current = int(event.get("current", event.get("downloaded_bytes", 0)) or 0)
            total = max(1, int(event.get("total", event.get("total_bytes", 100)) or 100))
            progress_bar.update_absolute(current, total)

    def cancelled() -> bool:
        if model_management is not None:
            model_management.throw_exception_if_processing_interrupted()
        return False

    return progress, cancelled


def _copy_worker_glb_to_cache(source: Path, destination: Path, validator) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    try:
        shutil.copyfile(source, partial)
        validator(partial)
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def _publish_worker_directory(source: Path, key: str, suffix: str) -> Path:
    cache_path(_cache_root(), "published-worker-directory", key)
    output_root = Path(os.environ.get("COMFYCOLAB_3D_OUTPUT", "/content/ComfyUI/output/3d"))
    destination = output_root / f"{key}-{suffix}"
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{key}-{suffix}-", dir=output_root))
    shutil.rmtree(staging)
    try:
        shutil.copytree(source, staging)
        atomic_replace_cache_directory(staging, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return destination


class ComfyColabTrellisImageTo3D:
    @classmethod
    def define_schema(cls):
        io = _io()
        return io.Schema(
            node_id="ComfyColabTrellisImageTo3D",
            display_name="ComfyColab TRELLIS.2 — Image to 3D",
            category="ComfyColab/3D",
            description=(
                "Generates a textured GLB with visible stage progress. Connect Preview 3D "
                "to receive an early untextured geometry preview before texture baking finishes."
            ),
            enable_expand=True,
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("quality", options=list(TRELLIS_PRESETS), default="1024 — Quality"),
                io.Int.Input("seed", default=0, min=0, max=(2**31) - 1),
                io.Combo.Input("exact_resolution", options=list(RESOLUTION_OVERRIDES), default="Auto", advanced=True),
                io.Int.Input("sampling_steps", default=0, min=0, max=50, advanced=True),
                io.Int.Input("target_face_count", default=0, min=0, max=2_000_000, advanced=True,
                             tooltip="0 uses the preset; manual values must be at least 1000"),
                io.Int.Input("texture_size", default=0, min=0, max=8192, advanced=True,
                             tooltip="0 uses the preset; manual values must be at least 512"),
                io.Int.Input("max_tokens", default=49_152, min=16_384, max=262_144, advanced=True),
                io.Combo.Input("remove_background", options=["Auto", "On", "Off"], default="Auto", advanced=True),
                io.Combo.Input("cache_mode", options=list(CACHE_MODES), default="Use cache", advanced=True),
            ],
            outputs=[io.File3DGLB.Output("model_3d")],
            hidden=[io.Hidden.unique_id, io.Hidden.prompt],
        )

    @classmethod
    def execute(
        cls, image, quality="1024 — Quality", seed=0, exact_resolution="Auto", sampling_steps=0,
        target_face_count=0, texture_size=0, max_tokens=49_152, remove_background="Auto", cache_mode="Use cache",
    ):
        if int(seed) < 0 or int(seed) > (2**31) - 1:
            raise ValueError("TRELLIS seed must be between 0 and 2147483647")
        settings = resolve_trellis_settings(
            quality,
            resolution=exact_resolution,
            sampling_steps=sampling_steps,
            target_face_count=target_face_count,
            texture_size=texture_size,
            max_tokens=max_tokens,
        )
        progress_node_id = _hidden_value(cls, "unique_id")
        preview_target = _connected_preview_target(
            _hidden_value(cls, "prompt"), progress_node_id
        )
        key = trellis_cache_key(
            image,
            settings=settings,
            seed=seed,
            remove_background=remove_background,
            comfyui_ref=COMFYUI_REF,
            trellis_ref=TRELLIS_WRAPPER_REF,
            trellis_patch_id=TRELLIS_PATCH_ID,
            birefnet_ref=BIREFNET_MODEL_REF,
        )
        destination = cache_path(_cache_root(), "trellis", key)
        if cache_mode == "Use cache" and _valid_cached_glb(destination, require_textured=True):
            _send_progress_text(progress_node_id, "Complete - Loaded cached 3D model")
            return _io().NodeOutput(materialize_file3d(publish_glb(destination, key)))
        required = {
            "LoadTrellis2Models", "Trellis2GetConditioning", "Trellis2ImageToShape",
            "Trellis2ShapeToTexturedMesh", "Trellis2ProcessMesh", "Trellis2RasterizePBR",
        }
        if remove_background != "Off":
            required.add("Trellis2RemoveBackground")
        _require_upstream_nodes(required)
        _send_progress_text(progress_node_id, "Stage 1/5 - Preparing models and input...")
        return build_trellis_graph(
            image, settings, seed=seed, remove_background=remove_background, cache_mode=cache_mode,
            cache_key=key,
            progress_node_id=progress_node_id,
            preview_target=preview_target,
        )


class ComfyColabTrellis2MV:
    @classmethod
    def define_schema(cls):
        io = _io()
        return io.Schema(
            node_id="ComfyColabTrellis2MV",
            display_name="ComfyColab TRELLIS2MV — Multi-View to 3D",
            category="ComfyColab/3D",
            description=(
                "Generates one textured GLB from four labeled horizontal views or six full views. "
                "Geometry uses the pinned community multiview sampler; texture uses the front view."
            ),
            enable_expand=True,
            inputs=[
                io.Image.Input("front_image", tooltip="Object viewed from the front (+Z by default)."),
                io.Image.Input("back_image", tooltip="Object viewed from directly behind."),
                io.Image.Input("left_image", tooltip="Object viewed from its left side."),
                io.Image.Input("right_image", tooltip="Object viewed from its right side."),
                io.Image.Input("top_image", optional=True, tooltip="Optional top-down view; connect bottom_image too."),
                io.Image.Input("bottom_image", optional=True, tooltip="Optional bottom-up view; connect top_image too."),
                io.Combo.Input("quality", options=list(TRELLIS_PRESETS), default="1024 — Quality"),
                io.Int.Input("seed", default=0, min=0, max=(2**31) - 1),
                io.Combo.Input("exact_resolution", options=list(RESOLUTION_OVERRIDES), default="Auto", advanced=True),
                io.Int.Input("sampling_steps", default=0, min=0, max=50, advanced=True),
                io.Int.Input("target_face_count", default=0, min=0, max=2_000_000, advanced=True),
                io.Int.Input("texture_size", default=0, min=0, max=8192, advanced=True),
                io.Int.Input("max_tokens", default=49_152, min=16_384, max=262_144, advanced=True),
                io.Combo.Input("front_axis", options=["Z forward", "X forward"], default="Z forward", advanced=True),
                io.Float.Input(
                    "blend_temperature", default=2.0, min=0.1, max=10.0, step=0.1, advanced=True,
                    tooltip="Higher values favor the nearest labeled camera more strongly.",
                ),
                io.Combo.Input("remove_background", options=["Auto", "On", "Off"], default="Auto", advanced=True),
                io.Combo.Input("cache_mode", options=list(CACHE_MODES), default="Use cache", advanced=True),
            ],
            outputs=[io.File3DGLB.Output("model_3d")],
            hidden=[io.Hidden.unique_id, io.Hidden.prompt],
        )

    @classmethod
    def execute(
        cls,
        front_image,
        back_image,
        left_image,
        right_image,
        top_image=None,
        bottom_image=None,
        quality="1024 — Quality",
        seed=0,
        exact_resolution="Auto",
        sampling_steps=0,
        target_face_count=0,
        texture_size=0,
        max_tokens=49_152,
        front_axis="Z forward",
        blend_temperature=2.0,
        remove_background="Auto",
        cache_mode="Use cache",
    ):
        if (top_image is None) != (bottom_image is None):
            raise ValueError("TRELLIS2MV requires both top_image and bottom_image for six-view mode")
        if int(seed) < 0 or int(seed) > (2**31) - 1:
            raise ValueError("TRELLIS2MV seed must be between 0 and 2147483647")
        if front_axis not in {"Z forward", "X forward"}:
            raise ValueError("front_axis must be Z forward or X forward")
        if not 0.1 <= float(blend_temperature) <= 10.0:
            raise ValueError("blend_temperature must be between 0.1 and 10.0")
        settings = resolve_trellis_settings(
            quality,
            resolution=exact_resolution,
            sampling_steps=int(sampling_steps),
            target_face_count=int(target_face_count),
            texture_size=int(texture_size),
            max_tokens=int(max_tokens),
        )
        views = {
            "front": front_image,
            "back": back_image,
            "left": left_image,
            "right": right_image,
        }
        if top_image is not None:
            views.update(top=top_image, bottom=bottom_image)
        axis = "z" if front_axis == "Z forward" else "x"
        key = trellis_multiview_cache_key(
            views,
            settings=settings,
            seed=int(seed),
            remove_background=remove_background,
            front_axis=axis,
            blend_temperature=float(blend_temperature),
            comfyui_ref=COMFYUI_REF,
            trellis_ref=TRELLIS_WRAPPER_REF,
            trellis_patch_id=TRELLIS_PATCH_ID,
            trellis_multiview_patch_id=TRELLIS_MULTIVIEW_PATCH_ID,
            birefnet_ref=BIREFNET_MODEL_REF,
        )
        progress_node_id = _hidden_value(cls, "unique_id")
        destination = cache_path(_cache_root(), "trellis-multiview", key)
        if cache_mode == "Use cache" and _valid_cached_glb(destination, require_textured=True):
            _send_progress_text(progress_node_id, "Complete - Loaded cached TRELLIS2MV model")
            return _io().NodeOutput(materialize_file3d(publish_glb(destination, key)))
        required = {
            "LoadTrellis2Models", "Trellis2GetConditioning", "Trellis2MultiViewImageToShape",
            "Trellis2ShapeToTexturedMesh", "Trellis2ProcessMesh", "Trellis2RasterizePBR",
        }
        if remove_background != "Off":
            required.add("Trellis2RemoveBackground")
        _require_upstream_nodes(required)
        _send_progress_text(progress_node_id, f"Stage 1/5 - Preparing {len(views)} labeled views...")
        return build_trellis_multiview_graph(
            front_image,
            settings,
            back_image=back_image,
            left_image=left_image,
            right_image=right_image,
            top_image=top_image,
            bottom_image=bottom_image,
            seed=int(seed),
            remove_background=remove_background,
            front_axis=axis,
            blend_temperature=float(blend_temperature),
            cache_mode=cache_mode,
            cache_key=key,
            progress_node_id=progress_node_id,
            preview_target=_connected_preview_target(
                _hidden_value(cls, "prompt"), progress_node_id
            ),
        )


class ComfyColabUltraShapeRefine:
    @classmethod
    def define_schema(cls):
        io = _io()
        return io.Schema(
            node_id="ComfyColabUltraShapeRefine",
            display_name="ComfyColab UltraShape — Refine Geometry",
            category="ComfyColab/3D",
            enable_expand=True,
            description=(
                "Refines a validated volumetric GLB. Conservative is the public 512 decode tier; "
                "Detailed and Ultra preserve the existing 1024 behavior as experimental choices."
            ),
            inputs=[
                io.File3DGLB.Input("model_3d"),
                io.Image.Input("reference_image"),
                io.Combo.Input("detail", options=list(ULTRASHAPE_PRESETS), default="Conservative"),
                io.Int.Input("seed", default=0, min=0, max=(2**31) - 1),
                io.Boolean.Input("retexture", default=True, advanced=True),
                io.Int.Input("steps", default=0, min=0, max=100, advanced=True),
                io.Int.Input("num_latents", default=0, min=0, max=131_072, advanced=True),
                io.Int.Input("octree_resolution", default=0, min=0, max=4096, advanced=True),
                io.Int.Input("decode_chunk_size", default=0, min=0, max=8192, advanced=True),
                io.Int.Input("target_face_count", default=0, min=0, max=2_000_000, advanced=True,
                             tooltip="0 uses the preset; manual values must be at least 1000"),
                io.Int.Input("texture_size", default=0, min=0, max=8192, advanced=True,
                             tooltip="0 uses the preset; manual values must be at least 512"),
                io.Combo.Input("low_vram", options=["Auto", "On", "Off"], default="Auto", advanced=True),
                io.Combo.Input("cache_mode", options=list(CACHE_MODES), default="Use cache", advanced=True),
            ],
            outputs=[io.File3DGLB.Output("refined_model_3d")],
        )

    @classmethod
    def execute(
        cls, model_3d, reference_image, detail="Conservative", seed=0, retexture=True, steps=0,
        num_latents=0, octree_resolution=0, decode_chunk_size=0, target_face_count=0,
        texture_size=0, low_vram="Auto", cache_mode="Use cache",
    ):
        if int(seed) < 0 or int(seed) > (2**31) - 1:
            raise ValueError("UltraShape seed must be between 0 and 2147483647")
        if 0 < int(target_face_count) < 1000:
            raise ValueError("target_face_count must be 0 for the preset or at least 1000")
        if 0 < int(texture_size) < 512:
            raise ValueError("texture_size must be 0 for the preset or at least 512")
        resolved = resolve_ultrashape_settings(
            detail, steps=steps, num_latents=num_latents,
            octree_resolution=octree_resolution, decode_chunk_size=decode_chunk_size,
        )
        print(
            "COMFYCOLAB_ULTRASHAPE_SETTINGS="
            + json.dumps(
                {
                    "detail": detail,
                    "steps": resolved.steps,
                    "num_latents": resolved.num_latents,
                    "octree_resolution": resolved.octree_resolution,
                    "decode_chunk_size": resolved.decode_chunk_size,
                    "seed": seed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if detail in ULTRASHAPE_EXPERIMENTAL_PRESETS or resolved.octree_resolution >= 1024:
            print(
                "[ComfyColab 3D] Warning: UltraShape 1024 decoding is experimental and "
                "has not passed its two-run live release gate.",
                flush=True,
            )
        low_vram_value = low_vram.lower()
        face_count = target_face_count or 500_000
        resolved_texture_size = texture_size or 2048
        with tempfile.TemporaryDirectory(prefix="comfycolab-3d-facade-") as directory:
            source = copy_file3d_to(model_3d, Path(directory) / "input.glb")
            validate_volumetric_glb(source, stage="UltraShape input GLB")
            source_digest = canonical_glb_geometry_digest(source)
        artifact_module = _load_artifact_provisioner(Path(__file__).resolve().parents[2])
        geometry_key = ultrashape_geometry_cache_key(
            source_digest,
            reference_image,
            detail=detail,
            seed=seed,
            steps=resolved.steps,
            num_latents=resolved.num_latents,
            octree_resolution=resolved.octree_resolution,
            decode_chunk_size=resolved.decode_chunk_size,
            low_vram=low_vram_value,
            worker_ref=os.environ.get("COMFYCOLAB_ULTRASHAPE_REF", ULTRASHAPE_SOURCE_REF),
            checkpoint_ref=artifact_module.ULTRASHAPE_REVISION,
            dinov2_ref=artifact_module.DINOV2_REVISION,
            transform_schema=TRANSFORM_SCHEMA,
        )
        geometry_path = cache_path(_cache_root(), "ultrashape", geometry_key, "geometry.glb")
        if cache_mode == "Use cache" and validate_geometry_cache_record(
            geometry_path.parent, geometry_key
        ):
            if not retexture:
                print(
                    f"[ComfyColab 3D] UltraShape geometry cache hit: {geometry_key}",
                    flush=True,
                )
                _require_upstream_nodes({"Trellis2ProcessMesh"})
                return build_ultrashape_cached_geometry_graph(
                    str(geometry_path), target_face_count=face_count
                )
            if retexture:
                refined_digest = canonical_glb_geometry_digest(geometry_path)
                final_key = texture_cache_key(
                    refined_digest,
                    reference_image,
                    seed=seed,
                    target_face_count=face_count,
                    texture_size=resolved_texture_size,
                    texture_sampling_steps=12,
                    trellis_ref=TRELLIS_WRAPPER_REF,
                )
                final_path = cache_path(_cache_root(), "texture", final_key)
                if _valid_cached_glb(final_path, require_textured=True):
                    return _io().NodeOutput(materialize_file3d(publish_glb(final_path, final_key)))
        required = {"Trellis2RemoveBackground", "Trellis2ProcessMesh"}
        if retexture:
            required.update({
                "LoadTrellis2Models", "Trellis2GetConditioning",
                "Trellis2EncodeMesh", "Trellis2TextureMesh", "Trellis2ProcessMesh",
                "Trellis2RasterizePBR",
            })
        _require_upstream_nodes(required)
        return build_ultrashape_graph(
            model_3d, reference_image, detail=detail, seed=seed, retexture=retexture,
            steps=resolved.steps, num_latents=resolved.num_latents,
            octree_resolution=resolved.octree_resolution, decode_chunk_size=resolved.decode_chunk_size,
            target_face_count=face_count, texture_size=resolved_texture_size,
            low_vram=low_vram_value, cache_mode=cache_mode, geometry_cache_key=geometry_key,
        )


class ComfyColabPixal3DImageTo3D:
    @classmethod
    def define_schema(cls):
        io = _io()
        return io.Schema(
            node_id="ComfyColabPixal3DImageTo3D",
            display_name="ComfyColab Pixal3D — Image to 3D",
            category="ComfyColab/3D",
            description=(
                "Generates a textured PBR GLB from one image through the isolated, persistent "
                "official Pixal3D worker. The first run downloads roughly 24 GB of pinned models."
            ),
            enable_expand=True,
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input(
                    "quality", options=list(PIXAL3D_PRESETS), default="1024 — Stable"
                ),
                io.Int.Input("seed", default=0, min=0, max=(2**31) - 1),
                io.Combo.Input(
                    "remove_background",
                    options=["Auto", "On", "Off"],
                    default="Auto",
                    advanced=True,
                ),
                io.Float.Input(
                    "camera_fov_degrees",
                    default=0.0,
                    min=0.0,
                    max=178.0,
                    step=0.1,
                    advanced=True,
                    tooltip="0 uses automatic MoGe camera estimation; a positive value is manual horizontal FOV.",
                ),
                io.Int.Input("sampling_steps", default=0, min=0, max=100, advanced=True),
                io.Int.Input(
                    "target_face_count",
                    default=0,
                    min=0,
                    max=2_000_000,
                    advanced=True,
                    tooltip="0 uses the selected quality preset; manual values must be at least 1000.",
                ),
                io.Int.Input(
                    "texture_size",
                    default=0,
                    min=0,
                    max=8192,
                    advanced=True,
                    tooltip="0 uses the selected quality preset; manual values must be at least 512.",
                ),
                io.Int.Input(
                    "max_tokens",
                    default=49_152,
                    min=16_384,
                    max=262_144,
                    advanced=True,
                ),
                io.Boolean.Input("keep_worker_loaded", default=True, advanced=True),
                io.Combo.Input(
                    "cache_mode", options=list(CACHE_MODES), default="Use cache", advanced=True
                ),
            ],
            outputs=[io.File3DGLB.Output("model_3d")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        image,
        quality="1024 — Stable",
        seed=0,
        remove_background="Auto",
        camera_fov_degrees=0.0,
        sampling_steps=0,
        target_face_count=0,
        texture_size=0,
        max_tokens=49_152,
        keep_worker_loaded=True,
        cache_mode="Use cache",
    ):
        seed = int(seed)
        fov = float(camera_fov_degrees)
        if seed < 0 or seed > (2**31) - 1:
            raise ValueError("Pixal3D seed must be between 0 and 2147483647")
        if not 0.0 <= fov < 179.0:
            raise ValueError(
                "camera_fov_degrees must be 0 for automatic estimation or between 0 and 179"
            )
        if remove_background not in {"Auto", "On", "Off"}:
            raise ValueError("remove_background must be Auto, On, or Off")
        if cache_mode not in CACHE_MODES:
            raise ValueError(f"Unknown Pixal3D cache mode: {cache_mode}")
        settings = resolve_pixal3d_settings(
            quality,
            sampling_steps=int(sampling_steps),
            target_face_count=int(target_face_count),
            texture_size=int(texture_size),
            max_tokens=int(max_tokens),
        )
        if quality in PIXAL3D_EXPERIMENTAL_PRESETS:
            print(
                "[ComfyColab 3D] Warning: Pixal3D 1536 is experimental and never silently "
                "downgrades; insufficient tokens or memory will return a clear failure.",
                flush=True,
            )
        repo_root = Path(__file__).resolve().parents[2]
        artifacts = _load_pixal3d_artifact_provisioner(repo_root)
        key = pixal3d_cache_key(
            image,
            settings=settings,
            seed=seed,
            remove_background=remove_background,
            camera_fov_degrees=fov,
            source_ref=artifacts.PIXAL3D_SOURCE_REF,
            model_ref=artifacts.PIXAL3D_MODEL_REF,
            dinov3_ref=artifacts.DINOV3_MODEL_REF,
            moge_ref=artifacts.MOGE_MODEL_REF,
            naf_ref=artifacts.NAF_SOURCE_REF,
            environment_ref=artifacts.PIXAL3D_ENVIRONMENT_REF,
        )
        destination = cache_path(_cache_root(), "pixal3d", key)
        if cache_mode == "Use cache" and _valid_cached_glb(destination, require_textured=True):
            _send_progress_text(_hidden_value(cls, "unique_id"), "Complete - Loaded cached Pixal3D model")
            return _io().NodeOutput(materialize_file3d(publish_glb(destination, key)))
        progress_node_id = _hidden_value(cls, "unique_id")
        _send_progress_text(progress_node_id, "Stage 1/3 - Preparing Pixal3D input and worker...")
        return build_pixal3d_graph(
            image,
            settings,
            seed=seed,
            remove_background=remove_background,
            camera_fov_degrees=fov,
            keep_worker_loaded=bool(keep_worker_loaded),
            cache_mode=cache_mode,
            cache_key=key,
            progress_node_id=progress_node_id,
        )


class ComfyColabPixal3DMV:
    @classmethod
    def define_schema(cls):
        io = _io()
        return io.Schema(
            node_id="ComfyColabPixal3DMV",
            display_name="ComfyColab Pixal3DMV (Experimental) — Multi-View to 3D",
            category="ComfyColab/3D",
            description=(
                "Experimental zero-shot multiview adapter for Pixal3D. It fuses labeled, "
                "view-aligned projected features using a ReconViaGen-inspired strategy; "
                "this is not official Pixal3D multiview support."
            ),
            enable_expand=True,
            inputs=[
                io.Image.Input("front_image", tooltip="Object viewed from the front."),
                io.Image.Input("back_image", tooltip="Object viewed from directly behind."),
                io.Image.Input("left_image", tooltip="Object viewed from its left side."),
                io.Image.Input("right_image", tooltip="Object viewed from its right side."),
                io.Image.Input("top_image", optional=True, tooltip="Optional top-down view; connect bottom_image too."),
                io.Image.Input("bottom_image", optional=True, tooltip="Optional bottom-up view; connect top_image too."),
                io.Combo.Input("quality", options=list(PIXAL3D_PRESETS), default="1024 — Stable"),
                io.Int.Input("seed", default=0, min=0, max=(2**31) - 1),
                io.Combo.Input(
                    "fusion_strategy",
                    options=["Directional projection", "Average projection"],
                    default="Directional projection",
                    advanced=True,
                ),
                io.Float.Input("fusion_temperature", default=2.0, min=0.1, max=10.0, step=0.1, advanced=True),
                io.Combo.Input(
                    "remove_background",
                    options=["Auto", "On", "Off"],
                    default="Auto",
                    tooltip=(
                        "Auto and On run pinned BEN2 foreground extraction inside the "
                        "isolated Pixal3D worker. Off preserves the complete input image."
                    ),
                    advanced=True,
                ),
                io.Float.Input("camera_fov_degrees", default=0.0, min=0.0, max=178.0, step=0.1, advanced=True),
                io.Int.Input("sampling_steps", default=0, min=0, max=100, advanced=True),
                io.Int.Input("target_face_count", default=0, min=0, max=2_000_000, advanced=True),
                io.Int.Input("texture_size", default=0, min=0, max=8192, advanced=True),
                io.Int.Input("max_tokens", default=49_152, min=16_384, max=262_144, advanced=True),
                io.Boolean.Input("keep_worker_loaded", default=True, advanced=True),
                io.Combo.Input("cache_mode", options=list(CACHE_MODES), default="Use cache", advanced=True),
            ],
            outputs=[io.File3DGLB.Output("model_3d")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        front_image,
        back_image,
        left_image,
        right_image,
        top_image=None,
        bottom_image=None,
        quality="1024 — Stable",
        seed=0,
        fusion_strategy="Directional projection",
        fusion_temperature=2.0,
        remove_background="Auto",
        camera_fov_degrees=0.0,
        sampling_steps=0,
        target_face_count=0,
        texture_size=0,
        max_tokens=49_152,
        keep_worker_loaded=True,
        cache_mode="Use cache",
    ):
        if (top_image is None) != (bottom_image is None):
            raise ValueError("Pixal3DMV requires both top_image and bottom_image for six-view mode")
        seed = int(seed)
        fov = float(camera_fov_degrees)
        if seed < 0 or seed > (2**31) - 1:
            raise ValueError("Pixal3DMV seed must be between 0 and 2147483647")
        if not 0.0 <= fov < 179.0:
            raise ValueError("camera_fov_degrees must be 0 or between 0 and 179")
        strategy_map = {
            "Directional projection": "directional_softmax",
            "Average projection": "average",
        }
        if fusion_strategy not in strategy_map:
            raise ValueError(f"Unknown Pixal3DMV fusion strategy: {fusion_strategy}")
        if not 0.1 <= float(fusion_temperature) <= 10.0:
            raise ValueError("fusion_temperature must be between 0.1 and 10.0")
        settings = resolve_pixal3d_settings(
            quality,
            sampling_steps=int(sampling_steps),
            target_face_count=int(target_face_count),
            texture_size=int(texture_size),
            max_tokens=int(max_tokens),
        )
        repo_root = Path(__file__).resolve().parents[2]
        artifacts = _load_pixal3d_artifact_provisioner(repo_root)
        views = {
            "front": front_image,
            "back": back_image,
            "left": left_image,
            "right": right_image,
        }
        if top_image is not None:
            views.update(top=top_image, bottom=bottom_image)
        resolved_strategy = strategy_map[fusion_strategy]
        key = pixal3d_multiview_cache_key(
            views,
            settings=settings,
            seed=seed,
            remove_background=remove_background,
            camera_fov_degrees=fov,
            fusion_strategy=resolved_strategy,
            fusion_temperature=float(fusion_temperature),
            source_ref=artifacts.PIXAL3D_SOURCE_REF,
            model_ref=artifacts.PIXAL3D_MODEL_REF,
            dinov3_ref=artifacts.DINOV3_MODEL_REF,
            moge_ref=artifacts.MOGE_MODEL_REF,
            naf_ref=artifacts.NAF_SOURCE_REF,
            environment_ref=artifacts.PIXAL3D_ENVIRONMENT_REF,
        )
        destination = cache_path(_cache_root(), "pixal3d", key)
        progress_node_id = _hidden_value(cls, "unique_id")
        if cache_mode == "Use cache" and _valid_cached_glb(destination, require_textured=True):
            _send_progress_text(progress_node_id, "Complete - Loaded cached Pixal3DMV model")
            return _io().NodeOutput(materialize_file3d(publish_glb(destination, key)))
        _send_progress_text(
            progress_node_id,
            f"Stage 1/3 - Preparing {len(views)} views for experimental Pixal3D fusion...",
        )
        return build_pixal3d_multiview_graph(
            front_image,
            settings,
            back_image=back_image,
            left_image=left_image,
            right_image=right_image,
            top_image=top_image,
            bottom_image=bottom_image,
            seed=seed,
            remove_background=remove_background,
            camera_fov_degrees=fov,
            fusion_strategy=resolved_strategy,
            fusion_temperature=float(fusion_temperature),
            keep_worker_loaded=bool(keep_worker_loaded),
            cache_mode=cache_mode,
            cache_key=key,
            progress_node_id=progress_node_id,
        )


class ComfyColabPixal3DMVAdvanced:
    @classmethod
    def define_schema(cls):
        io = _io()
        return io.Schema(
            node_id="ComfyColabPixal3DMVAdvanced",
            display_name="ComfyColab Pixal3DMV Advanced — VGGT-Ω Guided Multi-View to 3D",
            category="ComfyColab/3D",
            description=(
                "Advanced staged reconstruction. Either supply four canonical images or "
                "provide one source image and a FLUX.2 Klein 9B canonical-view LoRA; the "
                "node generates four separate full-resolution views, runs frozen VGGT-Ω "
                "guided Pixal3D fusion, and can pass the Pixal mesh through MeshFlow. "
                "This remains an experimental adapter, not official/native Pixal3D "
                "multiview support. Both MeshFlow and VGGT-Ω remain noncommercial "
                "research components."
            ),
            enable_expand=True,
            inputs=[
                io.Image.Input(
                    "front_image",
                    tooltip=(
                        "Explicit front view, or the single source image when generated-view "
                        "mode is selected."
                    ),
                ),
                io.Image.Input("back_image", optional=True, tooltip="Explicit back view."),
                io.Image.Input("left_image", optional=True, tooltip="Explicit left-side view."),
                io.Image.Input("right_image", optional=True, tooltip="Explicit right-side view."),
                io.Image.Input("top_image", optional=True, tooltip="Optional top-down view; connect bottom_image too."),
                io.Image.Input("bottom_image", optional=True, tooltip="Optional bottom-up view; connect top_image too."),
                io.Combo.Input(
                    "view_mode",
                    options=[
                        "Supplied canonical views",
                        "Generate separate views — FLUX.2 Klein 9B LoRA",
                    ],
                    default="Supplied canonical views",
                ),
                io.String.Input(
                    "multiview_lora_name",
                    default="comfycolab_flux2_klein_9b_canonical_views.safetensors",
                    tooltip=(
                        "Filename under ComfyUI/models/loras. The LoRA must generate one "
                        "requested canonical angle per inference, not a contact sheet."
                    ),
                    advanced=True,
                ),
                io.Combo.Input(
                    "flux_quantization",
                    options=["Q4_K_M", "Q5_K_M", "Q8_0", "Q3_K_M"],
                    default="Q4_K_M",
                    advanced=True,
                ),
                io.Combo.Input("quality", options=list(PIXAL3D_PRESETS), default="1024 — Stable"),
                io.Int.Input("seed", default=0, min=0, max=(2**31) - 1),
                io.Float.Input("front_quality", default=1.0, min=0.0, max=10.0, step=0.05, advanced=True),
                io.Float.Input("back_quality", default=1.0, min=0.0, max=10.0, step=0.05, advanced=True),
                io.Float.Input("left_quality", default=1.0, min=0.0, max=10.0, step=0.05, advanced=True),
                io.Float.Input("right_quality", default=1.0, min=0.0, max=10.0, step=0.05, advanced=True),
                io.Float.Input("top_quality", default=1.0, min=0.0, max=10.0, step=0.05, advanced=True, optional=True),
                io.Float.Input("bottom_quality", default=1.0, min=0.0, max=10.0, step=0.05, advanced=True, optional=True),
                io.Combo.Input(
                    "fusion_strategy",
                    options=["Directional projection", "Average projection"],
                    default="Directional projection",
                    advanced=True,
                ),
                io.Float.Input("fusion_temperature", default=2.0, min=0.1, max=10.0, step=0.1, advanced=True),
                io.Combo.Input(
                    "geometry_fallback",
                    options=["Strict — require VGGT-Ω", "Weighted MV fallback"],
                    default="Strict — require VGGT-Ω",
                    tooltip=(
                        "Strict fails unless the exact verified VGGT-Ω checkpoint runs. "
                        "Fallback explicitly records that ordinary weighted MV was used."
                    ),
                ),
                io.Float.Input("geometry_strength", default=0.75, min=0.0, max=1.0, step=0.05, advanced=True),
                io.Float.Input("confidence_exponent", default=1.0, min=0.0, max=4.0, step=0.05, advanced=True),
                io.Float.Input("depth_tolerance", default=0.12, min=0.0001, max=1.0, step=0.01, advanced=True),
                io.Float.Input("occlusion_margin", default=0.04, min=0.0, max=0.5, step=0.01, advanced=True),
                io.Float.Input("occlusion_tau", default=0.03, min=0.0001, max=0.5, step=0.01, advanced=True),
                io.Float.Input("geometry_floor", default=0.05, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Float.Input(
                    "max_normalized_alignment_error",
                    default=0.35,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Combo.Input(
                    "remove_background",
                    options=["Auto", "On", "Off"],
                    default="Auto",
                    tooltip=(
                        "Auto and On remove each view background with pinned BEN2 before "
                        "VGGT-Omega and Pixal3D. Off preserves every input pixel."
                    ),
                    advanced=True,
                ),
                io.Float.Input("camera_fov_degrees", default=0.0, min=0.0, max=178.0, step=0.1, advanced=True),
                io.Int.Input("sampling_steps", default=0, min=0, max=100, advanced=True),
                io.Int.Input("target_face_count", default=0, min=0, max=2_000_000, advanced=True),
                io.Int.Input("texture_size", default=0, min=0, max=8192, advanced=True),
                io.Int.Input("max_tokens", default=49_152, min=16_384, max=262_144, advanced=True),
                io.Boolean.Input("keep_worker_loaded", default=True, advanced=True),
                io.Boolean.Input(
                    "run_meshflow",
                    default=False,
                    tooltip="Generate a second artist-topology mesh conditioned by the Pixal mesh.",
                ),
                io.Int.Input("meshflow_steps", default=28, min=1, max=100, advanced=True),
                io.Int.Input(
                    "meshflow_num_verts",
                    default=4096,
                    min=1024,
                    max=16384,
                    step=1024,
                    advanced=True,
                ),
                io.Float.Input(
                    "meshflow_guidance_scale",
                    default=2.5,
                    min=0.0,
                    max=30.0,
                    step=0.1,
                    advanced=True,
                ),
                io.Combo.Input(
                    "meshflow_dtype",
                    options=["fp16", "bf16", "fp32"],
                    default="fp16",
                    advanced=True,
                ),
                io.Boolean.Input("meshflow_compile", default=False, advanced=True),
                io.Boolean.Input(
                    "accept_meshflow_research_license",
                    default=False,
                    tooltip="Required to download and execute the gated MeshFlow checkpoint.",
                ),
                io.Combo.Input("cache_mode", options=list(CACHE_MODES), default="Use cache", advanced=True),
            ],
            outputs=[
                io.File3DGLB.Output("model_3d"),
                io.File3DGLB.Output("pixal_stage_model_3d"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        front_image,
        back_image=None,
        left_image=None,
        right_image=None,
        top_image=None,
        bottom_image=None,
        view_mode="Supplied canonical views",
        multiview_lora_name="comfycolab_flux2_klein_9b_canonical_views.safetensors",
        flux_quantization="Q4_K_M",
        quality="1024 — Stable",
        seed=0,
        front_quality=1.0,
        back_quality=1.0,
        left_quality=1.0,
        right_quality=1.0,
        top_quality=1.0,
        bottom_quality=1.0,
        fusion_strategy="Directional projection",
        fusion_temperature=2.0,
        geometry_fallback="Strict — require VGGT-Ω",
        geometry_strength=0.75,
        confidence_exponent=1.0,
        depth_tolerance=0.12,
        occlusion_margin=0.04,
        occlusion_tau=0.03,
        geometry_floor=0.05,
        max_normalized_alignment_error=0.35,
        remove_background="Auto",
        camera_fov_degrees=0.0,
        sampling_steps=0,
        target_face_count=0,
        texture_size=0,
        max_tokens=49_152,
        keep_worker_loaded=True,
        run_meshflow=False,
        meshflow_steps=28,
        meshflow_num_verts=4096,
        meshflow_guidance_scale=2.5,
        meshflow_dtype="fp16",
        meshflow_compile=False,
        accept_meshflow_research_license=False,
        cache_mode="Use cache",
    ):
        generated_view_mode = (
            view_mode == "Generate separate views — FLUX.2 Klein 9B LoRA"
        )
        if view_mode not in {
            "Supplied canonical views",
            "Generate separate views — FLUX.2 Klein 9B LoRA",
        }:
            raise ValueError(f"Unknown Advanced Pixal3DMV view_mode: {view_mode}")
        if not generated_view_mode and any(
            image is None for image in (back_image, left_image, right_image)
        ):
            raise ValueError(
                "Supplied-view mode requires front, back, left, and right images"
            )
        if generated_view_mode and any(
            image is not None for image in (back_image, left_image, right_image)
        ):
            raise ValueError(
                "Generated-view mode accepts one source image only; disconnect explicit "
                "back, left, and right images"
            )
        if generated_view_mode and (top_image is not None or bottom_image is not None):
            raise ValueError("Generated-view mode currently produces exactly four views")
        if generated_view_mode and not str(multiview_lora_name).strip():
            raise ValueError("Generated-view mode requires multiview_lora_name")
        if flux_quantization not in {"Q3_K_M", "Q4_K_M", "Q5_K_M", "Q8_0"}:
            raise ValueError(f"Unsupported FLUX Klein quantization: {flux_quantization}")
        if (top_image is None) != (bottom_image is None):
            raise ValueError("Pixal3DMV requires both top_image and bottom_image for six-view mode")
        if run_meshflow and not accept_meshflow_research_license:
            raise ValueError(
                "MeshFlow requires explicit acceptance of its noncommercial research license"
            )
        if not 1 <= int(meshflow_steps) <= 100:
            raise ValueError("meshflow_steps must be between 1 and 100")
        if not 1024 <= int(meshflow_num_verts) <= 16384:
            raise ValueError("meshflow_num_verts must be between 1024 and 16384")
        if not 0.0 <= float(meshflow_guidance_scale) <= 30.0:
            raise ValueError("meshflow_guidance_scale must be between 0 and 30")
        if meshflow_dtype not in {"fp16", "bf16", "fp32"}:
            raise ValueError("meshflow_dtype must be fp16, bf16, or fp32")
        seed = int(seed)
        fov = float(camera_fov_degrees)
        if seed < 0 or seed > (2**31) - 1:
            raise ValueError("Pixal3DMV seed must be between 0 and 2147483647")
        if not 0.0 <= fov < 179.0:
            raise ValueError("camera_fov_degrees must be 0 or between 0 and 179")
        strategy_map = {
            "Directional projection": "directional_softmax",
            "Average projection": "average",
        }
        if fusion_strategy not in strategy_map:
            raise ValueError(f"Unknown Pixal3DMV fusion strategy: {fusion_strategy}")
        if not 0.1 <= float(fusion_temperature) <= 10.0:
            raise ValueError("fusion_temperature must be between 0.1 and 10.0")
        fallback_map = {
            "Strict — require VGGT-Ω": "strict",
            "Weighted MV fallback": "weighted_mv",
        }
        if geometry_fallback not in fallback_map:
            raise ValueError(
                f"Unknown Advanced Pixal3DMV geometry fallback: {geometry_fallback}"
            )
        geometry_controls = {
            "geometry_strength": (float(geometry_strength), 0.0, 1.0),
            "confidence_exponent": (float(confidence_exponent), 0.0, 4.0),
            "depth_tolerance": (float(depth_tolerance), 0.0001, 1.0),
            "occlusion_margin": (float(occlusion_margin), 0.0, 0.5),
            "occlusion_tau": (float(occlusion_tau), 0.0001, 0.5),
            "geometry_floor": (float(geometry_floor), 0.0, 1.0),
            "max_normalized_alignment_error": (
                float(max_normalized_alignment_error),
                0.0,
                1.0,
            ),
        }
        for name, (value, minimum, maximum) in geometry_controls.items():
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}")
        settings = resolve_pixal3d_settings(
            quality,
            sampling_steps=int(sampling_steps),
            target_face_count=int(target_face_count),
            texture_size=int(texture_size),
            max_tokens=int(max_tokens),
        )
        repo_root = Path(__file__).resolve().parents[2]
        artifacts = _load_pixal3d_artifact_provisioner(repo_root)
        views = (
            {"source": front_image}
            if generated_view_mode
            else {
                "front": front_image,
                "back": back_image,
                "left": left_image,
                "right": right_image,
            }
        )
        if top_image is not None:
            views.update(top=top_image, bottom=bottom_image)
        resolved_strategy = strategy_map[fusion_strategy]
        quality_map = {
            "front": float(front_quality),
            "back": float(back_quality),
            "left": float(left_quality),
            "right": float(right_quality),
        }
        if top_image is not None:
            quality_map.update(top=float(top_quality), bottom=float(bottom_quality))
        resolved_fallback = fallback_map[geometry_fallback]
        view_generation = (
            {
                "backend": "flux2-klein-9b",
                "quantization": flux_quantization,
                "lora_name": str(multiview_lora_name),
                "lora_path": str(_resolve_lora_path(str(multiview_lora_name))),
                "separate_outputs": True,
                "resolution": 1024,
                "prompts": CANONICAL_VIEW_PROMPTS,
            }
            if generated_view_mode
            else None
        )
        key = pixal3d_multiview_cache_key(
            views,
            settings=settings,
            seed=seed,
            remove_background=remove_background,
            camera_fov_degrees=fov,
            fusion_strategy=resolved_strategy,
            fusion_temperature=float(fusion_temperature),
            view_quality=quality_map,
            geometry_guidance="vggt_omega_depth_conf",
            geometry_fallback=resolved_fallback,
            vggt_omega_source_ref=artifacts.VGGT_OMEGA_SOURCE_REF,
            vggt_omega_checkpoint_ref=artifacts.VGGT_OMEGA_MODEL_REF,
            vggt_omega_image_resolution=512,
            geometry_strength=geometry_controls["geometry_strength"][0],
            confidence_exponent=geometry_controls["confidence_exponent"][0],
            depth_tolerance=geometry_controls["depth_tolerance"][0],
            occlusion_margin=geometry_controls["occlusion_margin"][0],
            occlusion_tau=geometry_controls["occlusion_tau"][0],
            geometry_floor=geometry_controls["geometry_floor"][0],
            max_normalized_alignment_error=geometry_controls[
                "max_normalized_alignment_error"
            ][0],
            source_ref=artifacts.PIXAL3D_SOURCE_REF,
            model_ref=artifacts.PIXAL3D_MODEL_REF,
            dinov3_ref=artifacts.DINOV3_MODEL_REF,
            moge_ref=artifacts.MOGE_MODEL_REF,
            naf_ref=artifacts.NAF_SOURCE_REF,
            environment_ref=artifacts.PIXAL3D_ENVIRONMENT_REF,
            view_generation=view_generation,
        )
        destination = cache_path(_cache_root(), "pixal3d", key)
        meshflow_config = None
        meshflow_destination = None
        if run_meshflow:
            meshflow_artifacts = _load_worker_artifact_provisioner(
                repo_root, "meshflow"
            )
            meshflow_key = meshflow_cache_key(
                key,
                reference_images=views,
                steps=int(meshflow_steps),
                num_verts=int(meshflow_num_verts),
                guidance_scale=float(meshflow_guidance_scale),
                seed=seed,
                dtype=meshflow_dtype,
                compile_models=bool(meshflow_compile),
                source_ref=meshflow_artifacts.MESHFLOW_SOURCE_REF,
                model_ref=meshflow_artifacts.MESHFLOW_MODEL_REF,
            )
            meshflow_destination = cache_path(
                _cache_root(), "meshflow", meshflow_key
            )
            meshflow_config = {
                "steps": int(meshflow_steps),
                "num_verts": int(meshflow_num_verts),
                "guidance_scale": float(meshflow_guidance_scale),
                "seed": seed,
                "dtype": meshflow_dtype,
                "compile_models": bool(meshflow_compile),
                "accept_research_license": bool(
                    accept_meshflow_research_license
                ),
                "cache_key": meshflow_key,
            }
        progress_node_id = _hidden_value(cls, "unique_id")
        pixal_cache_valid = (
            cache_mode == "Use cache"
            and _valid_cached_glb(destination, require_textured=True)
        )
        meshflow_cache_valid = (
            not run_meshflow
            or (
                meshflow_destination is not None
                and _valid_cached_glb(meshflow_destination)
            )
        )
        if pixal_cache_valid and meshflow_cache_valid:
            _send_progress_text(progress_node_id, "Complete - Loaded cached advanced Pixal3DMV model")
            pixal_file = materialize_file3d(publish_glb(destination, key))
            final_file = (
                materialize_file3d(
                    publish_glb(
                        meshflow_destination,
                        str(meshflow_config["cache_key"]),
                    )
                )
                if run_meshflow
                else pixal_file
            )
            return _io().NodeOutput(final_file, pixal_file)
        if generated_view_mode:
            _require_upstream_nodes(
                {
                    "ComfyColabFlux2Klein9BBundleLoader",
                    "LoraLoaderModelOnly",
                    "ImageScaleToTotalPixels",
                    "VAEEncode",
                    "CLIPTextEncode",
                    "ReferenceLatent",
                    "EmptyFlux2LatentImage",
                    "RandomNoise",
                    "CFGGuider",
                    "KSamplerSelect",
                    "Flux2Scheduler",
                    "SamplerCustomAdvanced",
                    "VAEDecode",
                    "SaveImage",
                }
            )
        _require_upstream_nodes({"SaveGLB"})
        _send_progress_text(
            progress_node_id,
            (
                "Stage 1/3 - Generating four separate FLUX canonical views..."
                if generated_view_mode
                else f"Stage 1/3 - Preparing {len(views)} supplied views..."
            ),
        )
        return build_pixal3d_multiview_graph(
            front_image,
            settings,
            back_image=back_image,
            left_image=left_image,
            right_image=right_image,
            top_image=top_image,
            bottom_image=bottom_image,
            view_quality=quality_map,
            geometry_guidance="vggt_omega_depth_conf",
            geometry_fallback=resolved_fallback,
            vggt_omega_image_resolution=512,
            geometry_strength=geometry_controls["geometry_strength"][0],
            confidence_exponent=geometry_controls["confidence_exponent"][0],
            depth_tolerance=geometry_controls["depth_tolerance"][0],
            occlusion_margin=geometry_controls["occlusion_margin"][0],
            occlusion_tau=geometry_controls["occlusion_tau"][0],
            geometry_floor=geometry_controls["geometry_floor"][0],
            max_normalized_alignment_error=geometry_controls[
                "max_normalized_alignment_error"
            ][0],
            seed=seed,
            remove_background=remove_background,
            camera_fov_degrees=fov,
            fusion_strategy=resolved_strategy,
            fusion_temperature=float(fusion_temperature),
            keep_worker_loaded=(
                bool(keep_worker_loaded)
                and not generated_view_mode
                and not run_meshflow
            ),
            cache_mode=cache_mode,
            cache_key=key,
            progress_node_id=progress_node_id,
            generated_view_source=front_image if generated_view_mode else None,
            generated_view_lora=(
                str(multiview_lora_name) if generated_view_mode else ""
            ),
            generated_view_prompts=(
                CANONICAL_VIEW_PROMPTS if generated_view_mode else None
            ),
            flux_quantization=flux_quantization,
            meshflow_config=meshflow_config,
            return_stage_outputs=True,
        )


class ComfyColabSkinTokensAutoRig:
    @classmethod
    def define_schema(cls):
        io = _io()
        return io.Schema(
            node_id="ComfyColabSkinTokensAutoRig",
            display_name="ComfyColab SkinTokens — Auto Rig 3D",
            category="ComfyColab/3D",
            description=(
                "Automatically creates a skeleton and dense skin weights for a GLB using "
                "the pinned SkinTokens/TokenRig release. Texture transfer can be preserved."
            ),
            inputs=[
                io.File3DGLB.Input("model_3d"),
                io.Boolean.Input(
                    "preserve_texture",
                    default=True,
                    tooltip="Uses TokenRig transfer mode to retain the input texture and scale.",
                ),
                io.Boolean.Input("use_postprocess", default=False, advanced=True),
                io.Boolean.Input("keep_worker_loaded", default=True, advanced=True),
                io.Combo.Input("cache_mode", options=list(CACHE_MODES), default="Use cache", advanced=True),
            ],
            outputs=[io.File3DGLB.Output("rigged_model_3d")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        model_3d,
        preserve_texture=True,
        use_postprocess=False,
        keep_worker_loaded=True,
        cache_mode="Use cache",
    ):
        repo_root = Path(__file__).resolve().parents[2]
        artifact_module = _load_worker_artifact_provisioner(repo_root, "skintokens")
        staging = _make_temp_directory("comfycolab-skintokens-")
        input_glb = staging / "input.glb"
        output_glb = staging / "rigged.glb"
        metadata_output = staging / "rigged.json"
        progress_node_id = _hidden_value(cls, "unique_id")
        try:
            copy_file3d_to(model_3d, input_glb)
            key = skintokens_cache_key(
                input_glb,
                preserve_texture=bool(preserve_texture),
                use_postprocess=bool(use_postprocess),
                source_ref=artifact_module.SKINTOKENS_SOURCE_REF,
                model_ref=artifact_module.SKINTOKENS_MODEL_REF,
                qwen_ref=artifact_module.SKINTOKENS_QWEN_REF,
                environment_ref=artifact_module.SKINTOKENS_ENVIRONMENT_REF,
            )
            destination = cache_path(_cache_root(), "skintokens", key)
            if cache_mode == "Use cache" and destination.is_file():
                try:
                    validate_skintokens_output(
                        destination, preserve_texture=bool(preserve_texture)
                    )
                except (OSError, ValueError):
                    shutil.rmtree(destination.parent, ignore_errors=True)
                else:
                    _send_progress_text(progress_node_id, "Complete - Loaded cached rigged model")
                    return _io().NodeOutput(materialize_file3d(publish_glb(destination, key)))

            progress, cancelled = _worker_callbacks()
            _send_progress_text(progress_node_id, "Stage 1/2 - Preparing SkinTokens worker and models...")
            artifacts = artifact_module.ensure_skintokens_artifacts(
                Path(
                    os.environ.get(
                        "COMFYCOLAB_SKINTOKENS_MODEL_ROOT",
                        "/content/.comfycolab/models/3d/skintokens",
                    )
                ),
                source_dir=os.environ.get("COMFYCOLAB_SKINTOKENS_SOURCE") or None,
                env_dir=Path(
                    os.environ.get(
                        "COMFYCOLAB_SKINTOKENS_ENV_ROOT",
                        "/content/.comfycolab/envs/skintokens",
                    )
                ),
                progress=progress,
            )
            command = SkinTokensWorkerCommand(
                python=os.environ.get("COMFYCOLAB_SKINTOKENS_PYTHON", str(artifacts.python)),
                worker_script=str(artifacts.worker_script),
                source_dir=str(artifacts.source_dir),
                model_dir=str(artifacts.model_dir),
                qwen_dir=str(artifacts.qwen_dir),
                checkpoint=str(artifacts.tokenrig_checkpoint),
                input_glb=str(input_glb),
                output_glb=str(output_glb),
                metadata_output=str(metadata_output),
                request_id=f"{os.getpid()}-{time.time_ns()}",
                source_ref=artifact_module.SKINTOKENS_SOURCE_REF,
                model_ref=artifact_module.SKINTOKENS_MODEL_REF,
                qwen_ref=artifact_module.SKINTOKENS_QWEN_REF,
                environment_ref=artifact_module.SKINTOKENS_ENVIRONMENT_REF,
                preserve_texture=bool(preserve_texture),
                use_transfer=bool(preserve_texture),
                use_postprocess=bool(use_postprocess),
                keep_worker_loaded=bool(keep_worker_loaded),
            )
            _send_progress_text(progress_node_id, "Stage 2/2 - Generating skeleton and skin weights...")
            worker_result = global_skintokens_worker_pool().run(
                command,
                is_cancelled=cancelled,
                on_progress=progress,
            )
            worker_evidence = {
                key: worker_result.get(key)
                for key in (
                    "schema",
                    "request_id",
                    "revisions",
                    "environment",
                    "environment_versions",
                    "generation",
                    "rig_contract",
                    "runtime_seconds",
                    "model_load_count",
                    "bytes",
                    "sha256",
                )
            }
            print(
                "COMFYCOLAB_SKINTOKENS_RESULT="
                + json.dumps(worker_evidence, sort_keys=True),
                flush=True,
            )
            final_glb = output_glb
            if cache_mode != "Disable cache":
                _copy_worker_glb_to_cache(
                    output_glb,
                    destination,
                    lambda path: validate_skintokens_output(
                        path, preserve_texture=bool(preserve_texture)
                    ),
                )
                if metadata_output.is_file():
                    metadata_destination = destination.parent / "metadata.json"
                    metadata_partial = destination.parent / f".metadata.json.{os.getpid()}.partial"
                    try:
                        shutil.copyfile(metadata_output, metadata_partial)
                        os.replace(metadata_partial, metadata_destination)
                    finally:
                        metadata_partial.unlink(missing_ok=True)
                final_glb = destination
            _send_progress_text(progress_node_id, "Complete - Rigged GLB ready")
            return _io().NodeOutput(materialize_file3d(publish_glb(final_glb, key)))
        finally:
            shutil.rmtree(staging, ignore_errors=True)


class ComfyColabCubePartSegment:
    @classmethod
    def define_schema(cls):
        io = _io()
        return io.Schema(
            node_id="ComfyColabCubePartSegment",
            display_name="ComfyColab CubePart — Segment 3D Parts",
            category="ComfyColab/3D",
            description=(
                "Schema-conditioned CubePart decomposition. Enter the ordered semantic part "
                "names you want; this is not unlabeled segment-anything behavior. CubePart's "
                "research-only license must be accepted explicitly."
            ),
            inputs=[
                io.File3DGLB.Input("model_3d"),
                io.String.Input(
                    "part_names",
                    default="body, head, left arm, right arm, left leg, right leg",
                    multiline=True,
                    tooltip="Comma- or newline-separated ordered semantic part schema (maximum 8).",
                ),
                io.Boolean.Input("accept_research_license", default=False),
                io.Int.Input("seed", default=0, min=0, max=(2**31) - 1, advanced=True),
                io.Float.Input("guidance_scale", default=7.5, min=0.0, max=30.0, step=0.1, advanced=True),
                io.Int.Input("num_inference_steps", default=50, min=1, max=200, advanced=True),
                io.Boolean.Input("keep_worker_loaded", default=True, advanced=True),
                io.Combo.Input("cache_mode", options=list(CACHE_MODES), default="Use cache", advanced=True),
            ],
            outputs=[
                io.File3DGLB.Output("segmented_model_3d"),
                io.String.Output("parts_directory"),
                io.String.Output("manifest_json"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        model_3d,
        part_names="body, head, left arm, right arm, left leg, right leg",
        accept_research_license=False,
        seed=0,
        guidance_scale=7.5,
        num_inference_steps=50,
        keep_worker_loaded=True,
        cache_mode="Use cache",
    ):
        if not accept_research_license:
            raise PermissionError(
                "CubePart uses research-only RAIL-MS artifacts. Enable "
                "accept_research_license only after accepting those terms."
            )
        parts = normalize_part_names(part_names)
        repo_root = Path(__file__).resolve().parents[2]
        artifact_module = _load_worker_artifact_provisioner(repo_root, "cubepart")
        staging = _make_temp_directory("comfycolab-cubepart-")
        input_glb = staging / "input.glb"
        output_dir = staging / "parts"
        progress_node_id = _hidden_value(cls, "unique_id")
        try:
            copy_file3d_to(model_3d, input_glb)
            key = cubepart_cache_key(
                canonical_glb_geometry_digest(input_glb),
                part_names=list(parts),
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(num_inference_steps),
                seed=int(seed),
                source_ref=artifact_module.CUBEPART_SOURCE_REF,
                model_ref=artifact_module.CUBEPART_MODEL_REF,
                environment_ref=artifact_module.CUBEPART_ENVIRONMENT_REF,
            )
            cache_dir = cache_path(
                _cache_root(), "cubepart", key, filename="parts.glb"
            ).parent
            if cache_mode == "Use cache" and cache_dir.is_dir():
                try:
                    manifest = validate_cubepart_output(cache_dir, parts)
                except (OSError, ValueError, RuntimeError):
                    shutil.rmtree(cache_dir, ignore_errors=True)
                else:
                    published_dir = _publish_worker_directory(cache_dir, key, "cubepart")
                    _send_progress_text(progress_node_id, "Complete - Loaded cached CubePart decomposition")
                    return _io().NodeOutput(
                        materialize_file3d(published_dir / "parts.glb"),
                        str(published_dir),
                        json.dumps(manifest, sort_keys=True),
                    )

            progress, cancelled = _worker_callbacks()
            _send_progress_text(progress_node_id, "Stage 1/2 - Preparing CubePart worker and weights...")
            artifacts = artifact_module.ensure_cubepart_artifacts(
                accept_research_license=True,
                source_dir=os.environ.get(
                    "COMFYCOLAB_CUBEPART_SOURCE", "/content/cube/cubepart"
                ),
                environment_dir=os.environ.get(
                    "COMFYCOLAB_CUBEPART_ENV_ROOT", "/content/.comfycolab/envs/cubepart"
                ),
                weights_root=os.environ.get(
                    "COMFYCOLAB_CUBEPART_MODEL_ROOT",
                    "/content/.comfycolab/models/3d/cubepart",
                ),
                progress=progress,
            )
            command = CubePartWorkerCommand(
                python=os.environ.get("COMFYCOLAB_CUBEPART_PYTHON", str(artifacts.python)),
                worker_script=str(repo_root / "worker/cubepart/worker_main.py"),
                source_dir=str(artifacts.source_dir),
                weights_dir=str(artifacts.weights_dir),
                input_mesh=str(input_glb),
                output_dir=str(output_dir),
                request_id=f"{os.getpid()}-{time.time_ns()}",
                part_names=parts,
                accept_research_license=True,
                seed=int(seed),
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(num_inference_steps),
                source_ref=artifact_module.CUBEPART_SOURCE_REF,
                model_ref=artifact_module.CUBEPART_MODEL_REF,
                environment_ref=artifact_module.CUBEPART_ENVIRONMENT_REF,
                keep_worker_loaded=bool(keep_worker_loaded),
            )
            _send_progress_text(progress_node_id, "Stage 2/2 - Decomposing mesh into requested parts...")
            global_cubepart_worker_pool().run(
                command,
                is_cancelled=cancelled,
                on_progress=progress,
            )
            result_dir = output_dir
            if cache_mode != "Disable cache":
                atomic_replace_cache_directory(output_dir, cache_dir)
                result_dir = cache_dir
            manifest = validate_cubepart_output(result_dir, parts)
            published_dir = _publish_worker_directory(result_dir, key, "cubepart")
            _send_progress_text(progress_node_id, "Complete - CubePart decomposition ready")
            return _io().NodeOutput(
                materialize_file3d(published_dir / "parts.glb"),
                str(published_dir),
                json.dumps(manifest, sort_keys=True),
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def _valid_cached_glb(
    path: Path,
    *,
    require_material: bool = False,
    require_textured: bool = False,
) -> bool:
    if not path.exists():
        return False
    try:
        validate_volumetric_glb(
            path,
            stage="cached 3D result",
            require_material=require_material or require_textured,
            require_texture=require_textured,
            require_uv=require_textured,
        )
    except (OSError, ValueError):
        path.unlink(missing_ok=True)
        return False
    return True


class _DevNode:
    @classmethod
    def _schema(cls, inputs, outputs):
        return _io().Schema(
            node_id=cls.__name__, display_name=cls.__name__, category="ComfyColab/3D/Internal",
            is_dev_only=True, inputs=inputs, outputs=outputs,
        )


class ComfyColab3DProgressCheckpoint(_DevNode):
    @classmethod
    def define_schema(cls):
        io = _io()
        return cls._schema(
            [
                io.AnyType.Input("value"),
                io.String.Input("progress_node_id"),
                io.Int.Input("completed"),
                io.Int.Input("total"),
                io.String.Input("status"),
                io.AnyType.Input("wait_for", optional=True),
            ],
            [io.AnyType.Output()],
        )

    @classmethod
    async def execute(
        cls,
        value,
        progress_node_id,
        completed,
        total,
        status,
        wait_for=None,
    ):
        del wait_for
        _send_progress_text(progress_node_id, status)
        api = importlib.import_module("comfy_api.latest").ComfyAPI()
        await api.execution.set_progress(
            value=float(completed),
            max_value=float(total),
            node_id=progress_node_id,
        )
        return _io().NodeOutput(value)


class ComfyColab3DImageOpaqueMask(_DevNode):
    @classmethod
    def define_schema(cls):
        io = _io()
        return cls._schema([io.Image.Input("image")], [io.Mask.Output()])

    @classmethod
    def execute(cls, image):
        torch = importlib.import_module("torch")
        return _io().NodeOutput(torch.ones(image.shape[0:3], dtype=image.dtype, device=image.device))


class ComfyColab3DPathToFile3D(_DevNode):
    @classmethod
    def define_schema(cls):
        io = _io()
        return cls._schema(
            [io.String.Input("glb_path"), io.Boolean.Input("delete_source", default=False)],
            [io.File3DGLB.Output()],
        )

    @classmethod
    def execute(cls, glb_path, delete_source=False):
        validate_volumetric_glb(glb_path, stage="UltraShape worker output GLB")
        key = deterministic_cache_key("published-worker-mesh", geometry=canonical_glb_geometry_digest(glb_path))
        try:
            result = materialize_file3d(publish_glb(glb_path, key))
        finally:
            if delete_source:
                _remove_owned_ultrashape_temp(glb_path)
        return _io().NodeOutput(result)


def _export_z_up_mesh(
    trimesh,
    destination: Path,
    *,
    require_material: bool = False,
    require_textured: bool = False,
    stage: str = "3D export mesh",
) -> None:
    numpy = importlib.import_module("numpy")
    mesh = trimesh.copy()
    validate_volumetric_mesh(mesh, stage=f"{stage} before coordinate conversion")
    mesh.apply_transform(numpy.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]))
    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    if uv is not None:
        converted_uv = uv.copy()
        converted_uv[:, 1] = 1.0 - converted_uv[:, 1]
        visual.uv = converted_uv
    validate_volumetric_mesh(mesh, stage=f"{stage} after coordinate conversion")
    export_trimesh_atomic(
        mesh,
        destination,
        require_material=require_material or require_textured,
        require_texture=require_textured,
        require_uv=require_textured,
    )
    validate_volumetric_glb(
        destination,
        stage=f"{stage} exported GLB",
        require_material=require_material or require_textured,
        require_texture=require_textured,
        require_uv=require_textured,
    )


class ComfyColab3DValidateMesh(_DevNode):
    @classmethod
    def define_schema(cls):
        io = _io()
        return cls._schema(
            [
                io.Custom("TRIMESH").Input("trimesh"),
                io.String.Input("stage"),
                io.Combo.Input(
                    "analysis_mode",
                    options=["exact", "raw"],
                    default="exact",
                    advanced=True,
                ),
            ],
            [io.Custom("TRIMESH").Output()],
        )

    @classmethod
    def execute(cls, trimesh, stage, analysis_mode="exact"):
        metrics = validate_volumetric_mesh(
            trimesh,
            stage=stage,
            analysis_mode=analysis_mode,
        )
        payload = metrics.to_dict() if hasattr(metrics, "to_dict") else metrics
        print(
            "COMFYCOLAB_GEOMETRY_QUALITY=" + json.dumps(payload, sort_keys=True),
            flush=True,
        )
        if getattr(metrics, "is_very_thin", False):
            print(
                f"[ComfyColab 3D] Warning: {stage} is very thin but remains rank 3; "
                "the mesh is accepted and should be checked from a side view.",
                flush=True,
            )
        return _io().NodeOutput(trimesh)


class ComfyColab3DTrimeshToFile3D(_DevNode):
    @classmethod
    def define_schema(cls):
        io = _io()
        return cls._schema(
            [
                io.Custom("TRIMESH").Input("trimesh"),
                io.String.Input("cache_stage"),
                io.String.Input("cache_key"),
                io.Combo.Input("cache_mode", options=list(CACHE_MODES)),
            ],
            [io.File3DGLB.Output()],
        )

    @classmethod
    def execute(cls, trimesh, cache_stage, cache_key, cache_mode="Use cache"):
        destination = cache_path(_cache_root(), cache_stage, cache_key)
        if cache_mode == "Use cache" and destination.exists():
            try:
                validate_volumetric_glb(
                    destination,
                    stage=f"cached {cache_stage} GLB",
                    require_material=cache_stage == "trellis",
                    require_texture=cache_stage == "trellis",
                    require_uv=cache_stage == "trellis",
                )
            except (OSError, ValueError):
                destination.unlink(missing_ok=True)
            else:
                return _io().NodeOutput(materialize_file3d(publish_glb(destination, cache_key)))
        temporary_root = None
        if cache_mode == "Disable cache":
            temporary_root = _make_temp_directory("comfycolab-3d-")
            destination = temporary_root / "model.glb"
        try:
            _export_z_up_mesh(
                trimesh,
                destination,
                require_textured=cache_stage == "trellis",
                stage=f"{cache_stage} final mesh",
            )
            result = materialize_file3d(publish_glb(destination, cache_key))
        finally:
            if temporary_root is not None:
                shutil.rmtree(temporary_root, ignore_errors=True)
        return _io().NodeOutput(result)


class ComfyColab3DNeutralMeshToFile3D(_DevNode):
    @classmethod
    def define_schema(cls):
        io = _io()
        return cls._schema(
            [io.Custom("TRIMESH").Input("trimesh")],
            [io.File3DGLB.Output()],
        )

    @classmethod
    def execute(cls, trimesh):
        numpy = importlib.import_module("numpy")
        trimesh_module = importlib.import_module("trimesh")
        mesh = trimesh.copy()
        validate_volumetric_mesh(mesh, stage="neutral preview/output mesh")
        mesh.visual = trimesh_module.visual.TextureVisuals(
            uv=numpy.zeros((len(mesh.vertices), 2), dtype=numpy.float32),
            material=trimesh_module.visual.material.PBRMaterial(
                baseColorFactor=[0.72, 0.72, 0.72, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.8,
            ),
        )
        key = deterministic_cache_key(
            "ultrashape-neutral-output", geometry=canonical_trimesh_digest(mesh)
        )
        staging = _make_temp_directory("comfycolab-ultrashape-neutral-")
        destination = staging / "geometry.glb"
        try:
            _export_z_up_mesh(
                mesh,
                destination,
                require_material=True,
                stage="neutral preview/output mesh",
            )
            result = materialize_file3d(publish_glb(destination, key))
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return _io().NodeOutput(result)


class ComfyColab3DTextureToFile3D(_DevNode):
    @classmethod
    def define_schema(cls):
        io = _io()
        return cls._schema(
            [
                io.Custom("TRIMESH").Input("trimesh"),
                io.Image.Input("reference_image"),
                io.String.Input("refined_geometry_digest"),
                io.Int.Input("seed"),
                io.Int.Input("target_face_count"),
                io.Int.Input("texture_size"),
                io.Int.Input("texture_sampling_steps"),
                io.Combo.Input("cache_mode", options=list(CACHE_MODES)),
            ],
            [io.File3DGLB.Output()],
        )

    @classmethod
    def execute(
        cls, trimesh, reference_image, refined_geometry_digest, seed, target_face_count,
        texture_size, texture_sampling_steps, cache_mode="Use cache",
    ):
        key = texture_cache_key(
            refined_geometry_digest,
            reference_image,
            seed=seed,
            target_face_count=target_face_count,
            texture_size=texture_size,
            texture_sampling_steps=texture_sampling_steps,
            trellis_ref=TRELLIS_WRAPPER_REF,
        )
        destination = cache_path(_cache_root(), "texture", key)
        if cache_mode == "Use cache" and _valid_cached_glb(destination, require_textured=True):
            return _io().NodeOutput(materialize_file3d(publish_glb(destination, key)))
        temporary_root = None
        if cache_mode == "Disable cache":
            temporary_root = _make_temp_directory("comfycolab-3d-")
            destination = temporary_root / "model.glb"
        try:
            _export_z_up_mesh(
                trimesh,
                destination,
                require_textured=True,
                stage="textured refined mesh",
            )
            result = materialize_file3d(publish_glb(destination, key))
        finally:
            if temporary_root is not None:
                shutil.rmtree(temporary_root, ignore_errors=True)
        return _io().NodeOutput(result)


class ComfyColab3DGLBToTrellisMesh(_DevNode):
    @classmethod
    def define_schema(cls):
        io = _io()
        return cls._schema(
            [io.String.Input("glb_path"), io.Boolean.Input("delete_source", default=False)],
            [
                io.Custom("TRIMESH").Output(),
                io.Custom("COMFYCOLAB_MESH_TRANSFORM").Output(),
                io.String.Output(),
            ],
        )

    @classmethod
    def execute(cls, glb_path, delete_source=False):
        try:
            validate_volumetric_glb(glb_path, stage="refined UltraShape GLB")
            geometry_digest = canonical_glb_geometry_digest(glb_path)
            mesh = load_glb_trimesh(glb_path)
        finally:
            if delete_source:
                _remove_owned_ultrashape_temp(glb_path)
        transform = normalization_for(mesh.vertices)
        return _io().NodeOutput(
            mesh,
            {"center": transform.center, "scale": transform.scale},
            geometry_digest,
        )


class ComfyColab3DEncodedMeshToTrimesh(_DevNode):
    @classmethod
    def define_schema(cls):
        io = _io()
        return cls._schema([io.Custom("TRELLIS2_SHAPE_LATENT").Input("shape_latent")], [io.Custom("TRIMESH").Output()])

    @classmethod
    def execute(cls, shape_latent):
        if isinstance(shape_latent, dict):
            if "preprocessed_vertices" in shape_latent and "preprocessed_faces" in shape_latent:
                trimesh = importlib.import_module("trimesh")
                vertices = shape_latent["preprocessed_vertices"]
                faces = shape_latent["preprocessed_faces"]
                if hasattr(vertices, "detach"):
                    vertices = vertices.detach().cpu().numpy()
                if hasattr(faces, "detach"):
                    faces = faces.detach().cpu().numpy()
                return _io().NodeOutput(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))
            for key in ("trimesh", "mesh", "preprocessed_mesh"):
                if key in shape_latent:
                    return _io().NodeOutput(shape_latent[key])
        for attribute in ("trimesh", "mesh", "preprocessed_mesh"):
            if hasattr(shape_latent, attribute):
                return _io().NodeOutput(getattr(shape_latent, attribute))
        raise ValueError("Trellis2EncodeMesh did not expose its normalized mesh")


class ComfyColab3DRestoreMeshTransform(_DevNode):
    @classmethod
    def define_schema(cls):
        io = _io()
        return cls._schema(
            [io.Custom("TRIMESH").Input("trimesh"), io.Custom("COMFYCOLAB_MESH_TRANSFORM").Input("transform")],
            [io.Custom("TRIMESH").Output()],
        )

    @classmethod
    def execute(cls, trimesh, transform):
        mesh = trimesh.copy()
        center, scale = transform["center"], float(transform["scale"])
        mesh.vertices = mesh.vertices / scale + center
        validate_volumetric_mesh(mesh, stage="restored UltraShape mesh")
        return _io().NodeOutput(mesh)


def _save_reference_image(image, mask, path: Path) -> None:
    numpy = importlib.import_module("numpy")
    pil_image = importlib.import_module("PIL.Image")
    value = image[0].detach().cpu().numpy() if hasattr(image, "detach") else image[0]
    mask_value = mask[0].detach().cpu().numpy() if hasattr(mask, "detach") else mask[0]
    rgb = numpy.clip(value[..., :3] * 255.0, 0, 255).astype(numpy.uint8)
    alpha = numpy.clip(mask_value * 255.0, 0, 255).astype(numpy.uint8)[..., None]
    array = numpy.concatenate((rgb, alpha), axis=-1)
    pil_image.fromarray(array).save(path)


def _save_rgb_reference_image(image, path: Path) -> None:
    numpy = importlib.import_module("numpy")
    pil_image = importlib.import_module("PIL.Image")
    value = image[0].detach().cpu().numpy() if hasattr(image, "detach") else image[0]
    rgb = numpy.clip(value[..., :3] * 255.0, 0, 255).astype(numpy.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.fromarray(rgb).save(path)


def _load_pixal3d_artifact_provisioner(repo_root: Path):
    module_name = "comfycolab_pixal3d_artifacts"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = repo_root / "worker/pixal3d/artifacts.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Pixal3D artifact provisioner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


class ComfyColab3DPixal3DWorker(_DevNode):
    @classmethod
    def define_schema(cls):
        io = _io()
        return cls._schema(
            [
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.String.Input("pipeline_type"),
                io.Int.Input("seed"),
                io.Int.Input("sampling_steps"),
                io.Int.Input("target_face_count"),
                io.Int.Input("texture_size"),
                io.Int.Input("max_tokens"),
                io.Float.Input("camera_fov_degrees"),
                io.Boolean.Input("keep_worker_loaded"),
                io.Combo.Input("cache_mode", options=list(CACHE_MODES)),
                io.String.Input("cache_key"),
                io.String.Input("background_removal", default="none", optional=True),
            ],
            [io.String.Output("glb_path")],
        )

    @classmethod
    def execute(
        cls,
        image,
        mask,
        pipeline_type,
        seed,
        sampling_steps,
        target_face_count,
        texture_size,
        max_tokens,
        camera_fov_degrees,
        keep_worker_loaded=True,
        cache_mode="Use cache",
        cache_key="",
        background_removal="none",
    ):
        del cache_mode, cache_key
        if background_removal not in {"none", "ben2"}:
            raise ValueError("Pixal3D background_removal must be none or ben2")
        repo_root = Path(__file__).resolve().parents[2]
        input_directory = Path(tempfile.mkdtemp(prefix="comfycolab-pixal3d-input-"))
        output_directory = _make_temp_directory("comfycolab-pixal3d-")
        image_path = input_directory / "prepared.png"
        output = output_directory / "model.glb"
        metadata = output_directory / "model.json"
        try:
            _save_reference_image(image, mask, image_path)
            artifact_module = _load_pixal3d_artifact_provisioner(repo_root)
            model_management = importlib.import_module("comfy.model_management")
            progress_bar = importlib.import_module("comfy.utils").ProgressBar(100)

            def progress(event: dict) -> None:
                model_management.throw_exception_if_processing_interrupted()
                current = int(event.get("current", event.get("downloaded_bytes", 0)) or 0)
                total = max(1, int(event.get("total", event.get("total_bytes", 100)) or 100))
                progress_bar.update_absolute(current, total)

            artifacts = artifact_module.ensure_pixal3d_artifacts(
                Path(
                    os.environ.get(
                        "COMFYCOLAB_PIXAL3D_MODEL_ROOT",
                        "/content/.comfycolab/models/3d/pixal3d",
                    )
                ),
                progress=progress,
            )
            request_id = f"{os.getpid()}-{time.time_ns()}"
            command = Pixal3DWorkerCommand(
                python=os.environ.get("COMFYCOLAB_PIXAL3D_PYTHON", DEFAULT_PIXAL3D_PYTHON),
                worker_script=str(repo_root / "worker/pixal3d/worker_main.py"),
                source_dir=os.environ.get("COMFYCOLAB_PIXAL3D_SOURCE", DEFAULT_PIXAL3D_SOURCE),
                checkpoint_dir=str(artifacts.model_dir),
                dinov3_dir=str(artifacts.dinov3_dir),
                moge_dir=str(artifacts.moge_dir),
                naf_source_dir=str(artifacts.naf_source_dir),
                naf_checkpoint=str(artifacts.naf_checkpoint),
                image_path=str(image_path),
                output_mesh=str(output),
                metadata_output=str(metadata),
                request_id=request_id,
                seed=int(seed),
                camera_fov_degrees=float(camera_fov_degrees),
                texture_size=int(texture_size),
                target_face_count=int(target_face_count),
                pipeline_type=str(pipeline_type),
                max_tokens=int(max_tokens),
                inference_steps=int(sampling_steps),
                source_ref=artifact_module.PIXAL3D_SOURCE_REF,
                model_ref=artifact_module.PIXAL3D_MODEL_REF,
                dinov3_ref=artifact_module.DINOV3_MODEL_REF,
                moge_ref=artifact_module.MOGE_MODEL_REF,
                naf_ref=artifact_module.NAF_SOURCE_REF,
                naf_checkpoint_ref=artifact_module.NAF_CHECKPOINT_SHA256,
                environment_ref=artifact_module.PIXAL3D_ENVIRONMENT_REF,
                keep_worker_loaded=bool(keep_worker_loaded),
                background_removal=str(background_removal),
            )

            def cancelled() -> bool:
                model_management.throw_exception_if_processing_interrupted()
                return False

            result = global_pixal3d_worker_pool().run(
                command,
                is_cancelled=cancelled,
                on_progress=progress,
            )
            print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
            return _io().NodeOutput(str(output))
        except BaseException:
            shutil.rmtree(output_directory, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(input_directory, ignore_errors=True)


class ComfyColab3DPixal3DMultiViewWorker(_DevNode):
    @classmethod
    def define_schema(cls):
        io = _io()
        return cls._schema(
            [
                io.Image.Input("front_image"),
                io.Mask.Input("front_mask"),
                io.Float.Input("front_quality", default=1.0, min=0.0, max=10.0, step=0.05, advanced=True),
                io.Image.Input("back_image"),
                io.Mask.Input("back_mask"),
                io.Float.Input("back_quality", default=1.0, min=0.0, max=10.0, step=0.05, advanced=True),
                io.Image.Input("left_image"),
                io.Mask.Input("left_mask"),
                io.Float.Input("left_quality", default=1.0, min=0.0, max=10.0, step=0.05, advanced=True),
                io.Image.Input("right_image"),
                io.Mask.Input("right_mask"),
                io.Float.Input("right_quality", default=1.0, min=0.0, max=10.0, step=0.05, advanced=True),
                io.String.Input("pipeline_type"),
                io.Int.Input("seed"),
                io.Int.Input("sampling_steps"),
                io.Int.Input("target_face_count"),
                io.Int.Input("texture_size"),
                io.Int.Input("max_tokens"),
                io.Float.Input("camera_fov_degrees"),
                io.String.Input("fusion_strategy"),
                io.Float.Input("fusion_temperature"),
                io.Boolean.Input("keep_worker_loaded"),
                io.Combo.Input("cache_mode", options=list(CACHE_MODES)),
                io.String.Input("cache_key"),
                io.String.Input("background_removal", default="none", optional=True),
                io.Image.Input("top_image", optional=True),
                io.Mask.Input("top_mask", optional=True),
                io.Float.Input("top_quality", default=1.0, min=0.0, max=10.0, step=0.05, advanced=True, optional=True),
                io.Image.Input("bottom_image", optional=True),
                io.Mask.Input("bottom_mask", optional=True),
                io.Float.Input("bottom_quality", default=1.0, min=0.0, max=10.0, step=0.05, advanced=True, optional=True),
                io.String.Input("geometry_guidance", default="none", optional=True),
                io.String.Input("geometry_fallback", default="strict", optional=True),
                io.Int.Input(
                    "vggt_omega_image_resolution",
                    default=512,
                    min=512,
                    max=512,
                    optional=True,
                ),
                io.Float.Input("geometry_strength", default=0.75, min=0.0, max=1.0, optional=True),
                io.Float.Input("confidence_exponent", default=1.0, min=0.0, max=4.0, optional=True),
                io.Float.Input("depth_tolerance", default=0.12, min=0.0001, max=1.0, optional=True),
                io.Float.Input("occlusion_margin", default=0.04, min=0.0, max=0.5, optional=True),
                io.Float.Input("occlusion_tau", default=0.03, min=0.0001, max=0.5, optional=True),
                io.Float.Input("geometry_floor", default=0.05, min=0.0, max=1.0, optional=True),
                io.Float.Input(
                    "max_normalized_alignment_error",
                    default=0.35,
                    min=0.0,
                    max=1.0,
                    optional=True,
                ),
            ],
            [
                io.String.Output("glb_path"),
                io.String.Output("surface_point_cloud_path"),
            ],
        )

    @classmethod
    def execute(
        cls,
        front_image,
        front_mask,
        front_quality,
        back_image,
        back_mask,
        back_quality,
        left_image,
        left_mask,
        left_quality,
        right_image,
        right_mask,
        right_quality,
        pipeline_type,
        seed,
        sampling_steps,
        target_face_count,
        texture_size,
        max_tokens,
        camera_fov_degrees,
        fusion_strategy,
        fusion_temperature,
        keep_worker_loaded=True,
        cache_mode="Use cache",
        cache_key="",
        background_removal="none",
        top_image=None,
        top_mask=None,
        top_quality=1.0,
        bottom_image=None,
        bottom_mask=None,
        bottom_quality=1.0,
        geometry_guidance="none",
        geometry_fallback="strict",
        vggt_omega_image_resolution=512,
        geometry_strength=0.75,
        confidence_exponent=1.0,
        depth_tolerance=0.12,
        occlusion_margin=0.04,
        occlusion_tau=0.03,
        geometry_floor=0.05,
        max_normalized_alignment_error=0.35,
    ):
        del cache_mode, cache_key
        if any(value is None for value in (top_image, top_mask, bottom_image, bottom_mask)) and any(
            value is not None for value in (top_image, top_mask, bottom_image, bottom_mask)
        ):
            raise ValueError("Pixal3DMV worker requires complete top and bottom image/mask pairs")
        if fusion_strategy not in {"directional_softmax", "average"}:
            raise ValueError("Pixal3DMV fusion_strategy must be directional_softmax or average")
        if background_removal not in {"none", "ben2"}:
            raise ValueError("Pixal3DMV background_removal must be none or ben2")
        if geometry_guidance not in {"none", "vggt_omega_depth_conf"}:
            raise ValueError(
                "Pixal3DMV geometry_guidance must be none or vggt_omega_depth_conf"
            )
        if geometry_fallback not in {"strict", "weighted_mv"}:
            raise ValueError(
                "Pixal3DMV geometry_fallback must be strict or weighted_mv"
            )
        repo_root = Path(__file__).resolve().parents[2]
        input_directory = Path(tempfile.mkdtemp(prefix="comfycolab-pixal3d-mv-input-"))
        output_directory = _make_temp_directory("comfycolab-pixal3d-")
        output = output_directory / "model.glb"
        metadata = output_directory / "model.json"
        surface_point_cloud = output_directory / "surface_point_cloud.ply"
        view_values = [
            ("front", front_image, front_mask, float(front_quality)),
            ("back", back_image, back_mask, float(back_quality)),
            ("left", left_image, left_mask, float(left_quality)),
            ("right", right_image, right_mask, float(right_quality)),
        ]
        if top_image is not None:
            view_values.extend(
                (
                    ("top", top_image, top_mask, float(top_quality)),
                    ("bottom", bottom_image, bottom_mask, float(bottom_quality)),
                )
            )
        try:
            serialized_views = []
            for name, image, mask, quality in view_values:
                image_path = input_directory / f"{name}.png"
                _save_reference_image(image, mask, image_path)
                serialized_views.append(
                    {"name": name, "image_path": str(image_path), "quality": quality}
                )

            artifact_module = _load_pixal3d_artifact_provisioner(repo_root)
            progress, cancelled = _worker_callbacks()
            model_root = Path(
                os.environ.get(
                    "COMFYCOLAB_PIXAL3D_MODEL_ROOT",
                    "/content/.comfycolab/models/3d/pixal3d",
                )
            )
            effective_geometry_guidance = str(geometry_guidance)
            provisioning_fallback_reason = ""
            if geometry_guidance == "vggt_omega_depth_conf":
                try:
                    artifacts = artifact_module.ensure_pixal3d_advanced_artifacts(
                        model_root,
                        progress=progress,
                    )
                except RuntimeError as error:
                    if geometry_fallback != "weighted_mv":
                        raise
                    provisioning_fallback_reason = str(error)
                    effective_geometry_guidance = "none"
                    artifacts = artifact_module.ensure_pixal3d_artifacts(
                        model_root,
                        progress=progress,
                    )
                    print(
                        "[ComfyColab 3D] VGGT-Ω artifact provisioning failed; "
                        "using the explicitly selected weighted-MV fallback.",
                        flush=True,
                    )
            else:
                artifacts = artifact_module.ensure_pixal3d_artifacts(
                    model_root,
                    progress=progress,
                )
            command = Pixal3DWorkerCommand(
                python=os.environ.get("COMFYCOLAB_PIXAL3D_PYTHON", DEFAULT_PIXAL3D_PYTHON),
                worker_script=str(repo_root / "worker/pixal3d/worker_main.py"),
                source_dir=os.environ.get("COMFYCOLAB_PIXAL3D_SOURCE", DEFAULT_PIXAL3D_SOURCE),
                checkpoint_dir=str(artifacts.model_dir),
                dinov3_dir=str(artifacts.dinov3_dir),
                moge_dir=str(artifacts.moge_dir),
                naf_source_dir=str(artifacts.naf_source_dir),
                naf_checkpoint=str(artifacts.naf_checkpoint),
                image_path=serialized_views[0]["image_path"],
                output_mesh=str(output),
                metadata_output=str(metadata),
                request_id=f"{os.getpid()}-{time.time_ns()}",
                seed=int(seed),
                camera_fov_degrees=float(camera_fov_degrees),
                texture_size=int(texture_size),
                target_face_count=int(target_face_count),
                pipeline_type=str(pipeline_type),
                max_tokens=int(max_tokens),
                inference_steps=int(sampling_steps),
                source_ref=artifact_module.PIXAL3D_SOURCE_REF,
                model_ref=artifact_module.PIXAL3D_MODEL_REF,
                dinov3_ref=artifact_module.DINOV3_MODEL_REF,
                moge_ref=artifact_module.MOGE_MODEL_REF,
                naf_ref=artifact_module.NAF_SOURCE_REF,
                naf_checkpoint_ref=artifact_module.NAF_CHECKPOINT_SHA256,
                environment_ref=artifact_module.PIXAL3D_ENVIRONMENT_REF,
                keep_worker_loaded=bool(keep_worker_loaded),
                background_removal=str(background_removal),
                views=tuple(serialized_views),
                fusion_temperature=float(fusion_temperature),
                fusion_strategy=str(fusion_strategy),
                geometry_guidance=effective_geometry_guidance,
                geometry_fallback=str(geometry_fallback),
                geometry_requested=(
                    "vggt_omega_depth_conf"
                    if provisioning_fallback_reason
                    else ""
                ),
                geometry_fallback_stage=(
                    "artifact_provisioning"
                    if provisioning_fallback_reason
                    else ""
                ),
                geometry_fallback_reason=provisioning_fallback_reason,
                vggt_omega_source_dir=str(
                    getattr(artifacts, "vggt_omega_source_dir", "")
                ),
                vggt_omega_checkpoint=str(
                    getattr(artifacts, "vggt_omega_checkpoint", "")
                ),
                vggt_omega_source_ref=(
                    artifact_module.VGGT_OMEGA_SOURCE_REF
                    if effective_geometry_guidance == "vggt_omega_depth_conf"
                    else ""
                ),
                vggt_omega_checkpoint_ref=(
                    getattr(
                        artifacts,
                        "vggt_omega_checkpoint_ref",
                        artifact_module.VGGT_OMEGA_MODEL_REF,
                    )
                    if effective_geometry_guidance == "vggt_omega_depth_conf"
                    else ""
                ),
                vggt_omega_image_resolution=int(vggt_omega_image_resolution),
                geometry_strength=float(geometry_strength),
                confidence_exponent=float(confidence_exponent),
                depth_tolerance=float(depth_tolerance),
                occlusion_margin=float(occlusion_margin),
                occlusion_tau=float(occlusion_tau),
                geometry_floor=float(geometry_floor),
                max_normalized_alignment_error=float(
                    max_normalized_alignment_error
                ),
                surface_point_cloud=str(surface_point_cloud),
                surface_point_count=65_536,
            )
            result = global_pixal3d_worker_pool().run(
                command,
                is_cancelled=cancelled,
                on_progress=progress,
            )
            print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
            return _io().NodeOutput(str(output), str(surface_point_cloud))
        except BaseException:
            shutil.rmtree(output_directory, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(input_directory, ignore_errors=True)


class ComfyColab3DPixal3DPathToFile3D(_DevNode):
    @classmethod
    def define_schema(cls):
        io = _io()
        return cls._schema(
            [
                io.String.Input("glb_path"),
                io.String.Input("cache_key"),
                io.Combo.Input("cache_mode", options=list(CACHE_MODES)),
                io.String.Input("surface_point_cloud_path", optional=True),
            ],
            [
                io.File3DGLB.Output("model_3d"),
                io.String.Output("surface_point_cloud_path"),
            ],
        )

    @classmethod
    def execute(
        cls,
        glb_path,
        cache_key,
        cache_mode="Use cache",
        surface_point_cloud_path="",
    ):
        source = Path(glb_path)
        source_point_cloud = None
        stable_point_cloud = None
        try:
            # Validate the facade-owned deterministic key even when result caching is disabled.
            cache_path(_cache_root(), "pixal3d", cache_key)
            if str(surface_point_cloud_path).strip():
                source_point_cloud = Path(str(surface_point_cloud_path)).resolve()
                if (
                    not source_point_cloud.is_file()
                    or source_point_cloud.suffix.lower() != ".ply"
                    or source_point_cloud.stat().st_size <= 0
                ):
                    raise FileNotFoundError(
                        "Pixal3D surface point cloud is missing, empty, or is not PLY: "
                        f"{source_point_cloud}"
                    )
            validate_volumetric_glb(
                source,
                stage="Pixal3D final GLB",
                require_material=True,
                require_texture=True,
                require_uv=True,
            )
            if cache_mode == "Disable cache":
                published = publish_glb(source, cache_key)
                if source_point_cloud is not None:
                    stable_directory = _make_temp_directory(
                        "comfycolab-pixal3d-surface-"
                    )
                    stable_point_cloud = stable_directory / "surface_point_cloud.ply"
                    shutil.copy2(source_point_cloud, stable_point_cloud)
                return _io().NodeOutput(
                    materialize_file3d(published),
                    str(stable_point_cloud) if stable_point_cloud is not None else "",
                )
            destination = cache_path(_cache_root(), "pixal3d", cache_key)
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
            partial_metadata = destination.parent / f".metadata.json.{os.getpid()}.partial"
            point_cloud_destination = destination.parent / "surface_point_cloud.ply"
            partial_point_cloud = destination.parent / (
                f".surface_point_cloud.ply.{os.getpid()}.partial"
            )
            try:
                shutil.copyfile(source, partial)
                validate_volumetric_glb(
                    partial,
                    stage="Pixal3D cache candidate",
                    require_material=True,
                    require_texture=True,
                    require_uv=True,
                )
                source_metadata = source.with_suffix(".json")
                if source_metadata.is_file():
                    shutil.copyfile(source_metadata, partial_metadata)
                if source_point_cloud is not None:
                    shutil.copy2(source_point_cloud, partial_point_cloud)
                    if partial_point_cloud.stat().st_size <= 0:
                        raise RuntimeError(
                            "Pixal3D cached surface point cloud is empty"
                        )
                    os.replace(partial_point_cloud, point_cloud_destination)
                    stable_point_cloud = point_cloud_destination
                os.replace(partial, destination)
                if partial_metadata.is_file():
                    os.replace(partial_metadata, destination.parent / "metadata.json")
            finally:
                partial.unlink(missing_ok=True)
                partial_metadata.unlink(missing_ok=True)
                partial_point_cloud.unlink(missing_ok=True)
            published = publish_glb(destination, cache_key)
            return _io().NodeOutput(
                materialize_file3d(published),
                str(stable_point_cloud) if stable_point_cloud is not None else "",
            )
        except BaseException:
            if stable_point_cloud is not None:
                _remove_owned_pixal3d_surface_temp(stable_point_cloud)
            raise
        finally:
            _remove_owned_pixal3d_temp(source)


class ComfyColab3DMeshFlowWorker(_DevNode):
    @classmethod
    def define_schema(cls):
        io = _io()
        return cls._schema(
            [
                io.File3DGLB.Input("model_3d"),
                io.String.Input("surface_point_cloud_path", optional=True),
                io.Image.Input("front_reference_image", optional=True),
                io.Image.Input("back_reference_image", optional=True),
                io.Image.Input("left_reference_image", optional=True),
                io.Image.Input("right_reference_image", optional=True),
                io.Image.Input("top_reference_image", optional=True),
                io.Image.Input("bottom_reference_image", optional=True),
                io.Int.Input("steps", default=28, min=1, max=100),
                io.Int.Input("num_verts", default=4096, min=1024, max=16384),
                io.Float.Input("guidance_scale", default=2.5, min=0.0, max=30.0, step=0.1),
                io.Int.Input("seed", default=0, min=0, max=(2**31) - 1),
                io.Combo.Input("dtype", options=["fp16", "bf16", "fp32"]),
                io.Boolean.Input("compile_models", default=False),
                io.Boolean.Input("accept_research_license", default=False),
            ],
            [io.String.Output("glb_path")],
        )

    @classmethod
    def execute(
        cls,
        model_3d,
        surface_point_cloud_path="",
        front_reference_image=None,
        back_reference_image=None,
        left_reference_image=None,
        right_reference_image=None,
        top_reference_image=None,
        bottom_reference_image=None,
        steps=28,
        num_verts=4096,
        guidance_scale=2.5,
        seed=0,
        dtype="fp16",
        compile_models=False,
        accept_research_license=False,
    ):
        if not accept_research_license:
            raise ValueError("MeshFlow research license acceptance is required")
        repo_root = Path(__file__).resolve().parents[2]
        staging = _make_temp_directory("comfycolab-meshflow-input-")
        output_directory = _make_temp_directory("comfycolab-meshflow-")
        appearance_mesh = staging / "pixal.glb"
        input_geometry = appearance_mesh
        output_mesh = output_directory / "meshflow.glb"
        metadata = output_directory / "meshflow.json"
        reference_inputs = (
            ("front", front_reference_image),
            ("back", back_reference_image),
            ("left", left_reference_image),
            ("right", right_reference_image),
            ("top", top_reference_image),
            ("bottom", bottom_reference_image),
        )
        reference_paths: list[Path] = []
        source_point_cloud = None
        try:
            copy_file3d_to(model_3d, appearance_mesh)
            if str(surface_point_cloud_path).strip():
                source_point_cloud = Path(str(surface_point_cloud_path)).resolve()
                if (
                    not source_point_cloud.is_file()
                    or source_point_cloud.suffix.lower() != ".ply"
                ):
                    raise FileNotFoundError(
                        "MeshFlow Pixal surface point cloud is missing or is not PLY: "
                        f"{source_point_cloud}"
                    )
                input_geometry = staging / "pixal_surface.ply"
                shutil.copy2(source_point_cloud, input_geometry)
            for name, reference in reference_inputs:
                if reference is None:
                    continue
                reference_path = staging / f"reference-{name}.png"
                _save_rgb_reference_image(reference, reference_path)
                reference_paths.append(reference_path)
            validate_volumetric_glb(
                appearance_mesh,
                stage="MeshFlow Pixal appearance input",
                require_material=True,
                require_texture=True,
                require_uv=True,
            )
            artifact_module = _load_worker_artifact_provisioner(
                repo_root, "meshflow"
            )
            progress, cancelled = _worker_callbacks()
            artifacts = artifact_module.ensure_meshflow_artifacts(
                os.environ.get(
                    "COMFYCOLAB_MESHFLOW_MODEL_ROOT",
                    "/content/.comfycolab/models/3d/meshflow",
                ),
                include_dinov3=bool(reference_paths),
                progress=progress,
            )
            command = MeshFlowWorkerCommand(
                python=os.environ.get(
                    "COMFYCOLAB_PIXAL3D_PYTHON", DEFAULT_PIXAL3D_PYTHON
                ),
                worker_script=str(repo_root / "worker/meshflow/worker_main.py"),
                source_dir=os.environ.get(
                    "COMFYCOLAB_MESHFLOW_SOURCE", DEFAULT_MESHFLOW_SOURCE
                ),
                checkpoint_dir=str(artifacts.checkpoint_dir),
                dinov3_model_dir=(
                    str(artifacts.dinov3_dir)
                    if artifacts.dinov3_dir is not None
                    else ""
                ),
                input_mesh=str(input_geometry),
                appearance_mesh=str(appearance_mesh),
                reference_images=tuple(str(path) for path in reference_paths),
                output_mesh=str(output_mesh),
                metadata_output=str(metadata),
                steps=int(steps),
                num_verts=int(num_verts),
                guidance_scale=float(guidance_scale),
                seed=int(seed),
                dtype=str(dtype),
                compile_models=bool(compile_models),
                source_ref=artifact_module.MESHFLOW_SOURCE_REF,
                model_ref=artifact_module.MESHFLOW_MODEL_REF,
            )
            run_meshflow_worker(
                command,
                is_cancelled=cancelled,
                on_progress=progress,
            )
            return _io().NodeOutput(str(output_mesh))
        except BaseException:
            shutil.rmtree(output_directory, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            if source_point_cloud is not None:
                _remove_owned_pixal3d_surface_temp(source_point_cloud)


class ComfyColab3DMeshFlowPathToFile3D(_DevNode):
    @classmethod
    def define_schema(cls):
        io = _io()
        return cls._schema(
            [
                io.String.Input("glb_path"),
                io.String.Input("cache_key"),
                io.Combo.Input("cache_mode", options=list(CACHE_MODES)),
            ],
            [io.File3DGLB.Output("model_3d")],
        )

    @classmethod
    def execute(cls, glb_path, cache_key, cache_mode="Use cache"):
        source = Path(glb_path)
        try:
            validate_volumetric_glb(
                source,
                stage="MeshFlow final GLB",
                require_material=False,
            )
            if cache_mode == "Disable cache":
                return _io().NodeOutput(
                    materialize_file3d(publish_glb(source, cache_key))
                )
            destination = cache_path(_cache_root(), "meshflow", cache_key)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _copy_worker_glb_to_cache(
                source,
                destination,
                lambda path: validate_volumetric_glb(
                    path,
                    stage="MeshFlow cache candidate",
                    require_material=False,
                ),
            )
            source_metadata = source.with_suffix(".json")
            if source_metadata.is_file():
                shutil.copyfile(
                    source_metadata, destination.parent / "metadata.json"
                )
            return _io().NodeOutput(
                materialize_file3d(publish_glb(destination, cache_key))
            )
        finally:
            _remove_owned_meshflow_temp(source)


def _load_artifact_provisioner(repo_root: Path):
    module_name = "comfycolab_ultrashape_artifacts"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = repo_root / "worker/ultrashape/artifacts.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load UltraShape artifact provisioner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


class ComfyColab3DUltraShapeWorker(_DevNode):
    @classmethod
    def define_schema(cls):
        io = _io()
        return cls._schema(
            [
                io.File3DGLB.Input("model_3d"), io.Image.Input("reference_image"),
                io.Mask.Input("reference_mask"), io.String.Input("detail"),
                io.Int.Input("seed"), io.Int.Input("steps"), io.Int.Input("num_latents"),
                io.Int.Input("octree_resolution"), io.Int.Input("decode_chunk_size"), io.String.Input("low_vram"),
                io.Combo.Input("cache_mode", options=list(CACHE_MODES)),
                io.String.Input("geometry_cache_key", default=""),
            ],
            [io.String.Output()],
        )

    @classmethod
    def execute(
        cls, model_3d, reference_image, reference_mask, detail, seed, steps, num_latents, octree_resolution,
        decode_chunk_size, low_vram, cache_mode="Use cache", geometry_cache_key="",
    ):
        repo_root = Path(__file__).resolve().parents[2]
        staging = Path(tempfile.mkdtemp(prefix="comfycolab-ultrashape-input-"))
        try:
            input_mesh, image_path = staging / "input.glb", staging / "reference.png"
            copy_file3d_to(model_3d, input_mesh)
            validate_volumetric_glb(input_mesh, stage="UltraShape worker input GLB")
            artifact_module = _load_artifact_provisioner(repo_root)
            _save_reference_image(reference_image, reference_mask, image_path)
            key = geometry_cache_key or ultrashape_geometry_cache_key(
                canonical_glb_geometry_digest(input_mesh),
                reference_image,
                detail=detail,
                seed=seed,
                steps=steps,
                num_latents=num_latents,
                octree_resolution=octree_resolution,
                decode_chunk_size=decode_chunk_size,
                low_vram=low_vram,
                worker_ref=os.environ.get("COMFYCOLAB_ULTRASHAPE_REF", ULTRASHAPE_SOURCE_REF),
                checkpoint_ref=artifact_module.ULTRASHAPE_REVISION,
                dinov2_ref=artifact_module.DINOV2_REVISION,
                transform_schema=TRANSFORM_SCHEMA,
            )
            cache_directory = cache_path(
                _cache_root(), "ultrashape", key, "geometry.glb"
            ).parent
            if cache_mode == "Use cache" and validate_geometry_cache_record(
                cache_directory, key
            ):
                return _io().NodeOutput(str(cache_directory / "geometry.glb"))
            if cache_mode == "Disable cache":
                staging_output = _make_temp_directory("comfycolab-ultrashape-")
            else:
                cache_directory.parent.mkdir(parents=True, exist_ok=True)
                staging_output = Path(
                    tempfile.mkdtemp(
                        prefix=f".{key}.", suffix=".partial", dir=cache_directory.parent
                    )
                )
            output = staging_output / "geometry.glb"
            metadata_output = staging_output / "transform.json"
            worker_script = repo_root / "worker/ultrashape/worker_main.py"
            model_management = importlib.import_module("comfy.model_management")
            progress_bar = importlib.import_module("comfy.utils").ProgressBar(100)

            def progress(event: dict) -> None:
                model_management.throw_exception_if_processing_interrupted()
                current = int(event.get("current", event.get("downloaded_bytes", 0)) or 0)
                total = max(1, int(event.get("total", event.get("total_bytes", 100)) or 100))
                progress_bar.update_absolute(current, total)

            artifacts = artifact_module.ensure_ultrashape_artifacts(
                Path(os.environ.get("COMFYCOLAB_3D_MODEL_ROOT", "/content/.comfycolab/models/3d")),
                progress=progress,
            )
            command = UltraShapeCommand(
                python=os.environ.get("COMFYCOLAB_ULTRASHAPE_PYTHON", DEFAULT_ULTRASHAPE_PYTHON),
                worker_script=str(worker_script),
                source_dir=os.environ.get("COMFYCOLAB_ULTRASHAPE_SOURCE", DEFAULT_ULTRASHAPE_SOURCE),
                checkpoint=str(artifacts.checkpoint),
                dinov2_dir=str(artifacts.dinov2_dir),
                input_mesh=str(input_mesh), reference_image=str(image_path), output_mesh=str(output),
                metadata_output=str(metadata_output), steps=steps, num_latents=num_latents,
                octree_resolution=octree_resolution, decode_chunk_size=decode_chunk_size, seed=seed,
                low_vram=low_vram, checkpoint_sha256="",
            )

            def cancelled() -> bool:
                model_management.throw_exception_if_processing_interrupted()
                return False

            try:
                worker_result = run_ultrashape_worker(
                    command,
                    is_cancelled=cancelled,
                    on_progress=progress,
                )
                worker_settings = worker_result.get("settings")
                if not isinstance(worker_settings, dict):
                    raise RuntimeError("UltraShape worker result omitted resolved settings")
                print(
                    "COMFYCOLAB_ULTRASHAPE_WORKER_SETTINGS="
                    + json.dumps(worker_settings, sort_keys=True),
                    flush=True,
                )
                validate_volumetric_glb(
                    output,
                    stage="UltraShape worker refined GLB",
                    require_material=True,
                )
                write_geometry_cache_record(staging_output, key)
                if cache_mode != "Disable cache":
                    atomic_replace_cache_directory(staging_output, cache_directory)
                    output = cache_directory / "geometry.glb"
                return _io().NodeOutput(str(output))
            except BaseException:
                shutil.rmtree(staging_output, ignore_errors=True)
                raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)


NODE_CLASS_MAPPINGS = {
    "ComfyColabTrellisImageTo3D": ComfyColabTrellisImageTo3D,
    "ComfyColabTrellis2MV": ComfyColabTrellis2MV,
    "ComfyColabUltraShapeRefine": ComfyColabUltraShapeRefine,
    "ComfyColabPixal3DImageTo3D": ComfyColabPixal3DImageTo3D,
    "ComfyColabPixal3DMV": ComfyColabPixal3DMV,
    "ComfyColabPixal3DMVAdvanced": ComfyColabPixal3DMVAdvanced,
    "ComfyColabSkinTokensAutoRig": ComfyColabSkinTokensAutoRig,
    "ComfyColabCubePartSegment": ComfyColabCubePartSegment,
    "ComfyColab3DProgressCheckpoint": ComfyColab3DProgressCheckpoint,
    "ComfyColab3DImageOpaqueMask": ComfyColab3DImageOpaqueMask,
    "ComfyColab3DPathToFile3D": ComfyColab3DPathToFile3D,
    "ComfyColab3DValidateMesh": ComfyColab3DValidateMesh,
    "ComfyColab3DTrimeshToFile3D": ComfyColab3DTrimeshToFile3D,
    "ComfyColab3DNeutralMeshToFile3D": ComfyColab3DNeutralMeshToFile3D,
    "ComfyColab3DTextureToFile3D": ComfyColab3DTextureToFile3D,
    "ComfyColab3DGLBToTrellisMesh": ComfyColab3DGLBToTrellisMesh,
    "ComfyColab3DEncodedMeshToTrimesh": ComfyColab3DEncodedMeshToTrimesh,
    "ComfyColab3DRestoreMeshTransform": ComfyColab3DRestoreMeshTransform,
    "ComfyColab3DUltraShapeWorker": ComfyColab3DUltraShapeWorker,
    "ComfyColab3DPixal3DWorker": ComfyColab3DPixal3DWorker,
    "ComfyColab3DPixal3DMultiViewWorker": ComfyColab3DPixal3DMultiViewWorker,
    "ComfyColab3DPixal3DPathToFile3D": ComfyColab3DPixal3DPathToFile3D,
    "ComfyColab3DMeshFlowWorker": ComfyColab3DMeshFlowWorker,
    "ComfyColab3DMeshFlowPathToFile3D": ComfyColab3DMeshFlowPathToFile3D,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyColabTrellisImageTo3D": "ComfyColab TRELLIS.2 — Image to 3D",
    "ComfyColabTrellis2MV": "ComfyColab TRELLIS2MV — Multi-View to 3D",
    "ComfyColabUltraShapeRefine": "ComfyColab UltraShape — Refine Geometry",
    "ComfyColabPixal3DImageTo3D": "ComfyColab Pixal3D — Image to 3D",
    "ComfyColabPixal3DMV": "ComfyColab Pixal3DMV (Experimental) — Multi-View to 3D",
    "ComfyColabPixal3DMVAdvanced": "ComfyColab Pixal3DMV Advanced — Single Image / Multi-View / MeshFlow",
    "ComfyColabSkinTokensAutoRig": "ComfyColab SkinTokens — Auto Rig 3D",
    "ComfyColabCubePartSegment": "ComfyColab CubePart — Segment 3D Parts",
}
