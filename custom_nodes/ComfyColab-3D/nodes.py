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
    pixal3d_cache_key,
    texture_cache_key,
    trellis_cache_key,
    ultrashape_geometry_cache_key,
)
from .file3d import copy_file3d_to, export_trimesh_atomic, load_glb_trimesh, materialize_file3d, publish_glb
from .geometry_quality import validate_volumetric_glb, validate_volumetric_mesh
from .graph import (
    COMFYUI_REF,
    BIREFNET_MODEL_REF,
    TRELLIS_PATCH_ID,
    TRELLIS_WRAPPER_REF,
    build_trellis_graph,
    build_pixal3d_graph,
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

ULTRASHAPE_SOURCE_REF = "5e8dcef05df101ab00ab6cd5fdd0ed0c74fbca66"
DEFAULT_ULTRASHAPE_SOURCE = "/content/UltraShape-1.0"
DEFAULT_ULTRASHAPE_PYTHON = str(Path.home() / ".ce/.pixi/envs/trellis2-nodes/bin/python")
DEFAULT_PIXAL3D_SOURCE = "/content/Pixal3D"
DEFAULT_PIXAL3D_PYTHON = str(Path.home() / ".ce/.pixi/envs/pixal3d-worker/bin/python")
TRANSFORM_SCHEMA = "comfycolab-3d-transform-v1"


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


def _require_upstream_nodes(node_ids: set[str]) -> None:
    try:
        registry = importlib.import_module("nodes").NODE_CLASS_MAPPINGS
    except (ModuleNotFoundError, AttributeError):
        return
    missing = sorted(node_ids - set(registry))
    if missing:
        raise RuntimeError(
            "ComfyColab 3D requires the pinned ComfyUI-TRELLIS2 node pack. "
            f"Missing node IDs: {', '.join(missing)}. Restart with `comfycolab start --refresh`."
        )


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
        if remove_background != "Off":
            _require_upstream_nodes({"Trellis2RemoveBackground"})
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
    ):
        del cache_mode, cache_key
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


class ComfyColab3DPixal3DPathToFile3D(_DevNode):
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
            # Validate the facade-owned deterministic key even when result caching is disabled.
            cache_path(_cache_root(), "pixal3d", cache_key)
            validate_volumetric_glb(
                source,
                stage="Pixal3D final GLB",
                require_material=True,
                require_texture=True,
                require_uv=True,
            )
            if cache_mode == "Disable cache":
                published = publish_glb(source, cache_key)
                return _io().NodeOutput(materialize_file3d(published))
            destination = cache_path(_cache_root(), "pixal3d", cache_key)
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
            partial_metadata = destination.parent / f".metadata.json.{os.getpid()}.partial"
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
                os.replace(partial, destination)
                if partial_metadata.is_file():
                    os.replace(partial_metadata, destination.parent / "metadata.json")
            finally:
                partial.unlink(missing_ok=True)
                partial_metadata.unlink(missing_ok=True)
            published = publish_glb(destination, cache_key)
            return _io().NodeOutput(materialize_file3d(published))
        finally:
            _remove_owned_pixal3d_temp(source)


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
    "ComfyColabUltraShapeRefine": ComfyColabUltraShapeRefine,
    "ComfyColabPixal3DImageTo3D": ComfyColabPixal3DImageTo3D,
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
    "ComfyColab3DPixal3DPathToFile3D": ComfyColab3DPixal3DPathToFile3D,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyColabTrellisImageTo3D": "ComfyColab TRELLIS.2 — Image to 3D",
    "ComfyColabUltraShapeRefine": "ComfyColab UltraShape — Refine Geometry",
    "ComfyColabPixal3DImageTo3D": "ComfyColab Pixal3D — Image to 3D",
}
