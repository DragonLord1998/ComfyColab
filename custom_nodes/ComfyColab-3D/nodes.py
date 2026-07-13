from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from .cache import (
    cache_path,
    canonical_glb_geometry_digest,
    canonical_trimesh_digest,
    deterministic_cache_key,
    texture_cache_key,
    trellis_cache_key,
    ultrashape_geometry_cache_key,
)
from .file3d import copy_file3d_to, export_trimesh_atomic, load_glb_trimesh, materialize_file3d, publish_glb, validate_glb
from .graph import (
    COMFYUI_REF,
    BIREFNET_MODEL_REF,
    TRELLIS_PATCH_ID,
    TRELLIS_WRAPPER_REF,
    build_trellis_graph,
    build_ultrashape_cached_geometry_graph,
    build_ultrashape_graph,
)
from .presets import (
    CACHE_MODES,
    RESOLUTION_OVERRIDES,
    TRELLIS_PRESETS,
    ULTRASHAPE_PRESETS,
    resolve_trellis_settings,
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

ULTRASHAPE_SOURCE_REF = "5e8dcef05df101ab00ab6cd5fdd0ed0c74fbca66"
DEFAULT_ULTRASHAPE_SOURCE = "/content/UltraShape-1.0"
DEFAULT_ULTRASHAPE_PYTHON = str(Path.home() / ".ce/.pixi/envs/trellis2-nodes/bin/python")
TRANSFORM_SCHEMA = "comfycolab-3d-transform-v1"


def _io():
    return importlib.import_module("comfy_api.latest").io


def _cache_root() -> Path:
    root = os.environ.get("COMFYCOLAB_3D_CACHE")
    return Path(root) if root else Path("/content/.comfycolab/cache/3d")


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
            return _io().NodeOutput(materialize_file3d(publish_glb(destination, key)))
        required = {
            "LoadTrellis2Models", "Trellis2GetConditioning", "Trellis2ImageToShape",
            "Trellis2ShapeToTexturedMesh", "Trellis2ProcessMesh", "Trellis2RasterizePBR",
        }
        if remove_background != "Off":
            required.add("Trellis2RemoveBackground")
        _require_upstream_nodes(required)
        return build_trellis_graph(
            image, settings, seed=seed, remove_background=remove_background, cache_mode=cache_mode,
            cache_key=key,
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
            inputs=[
                io.File3DGLB.Input("model_3d"),
                io.Image.Input("reference_image"),
                io.Combo.Input("detail", options=list(ULTRASHAPE_PRESETS), default="Detailed"),
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
        cls, model_3d, reference_image, detail="Detailed", seed=0, retexture=True, steps=0,
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
        low_vram_value = low_vram.lower()
        face_count = target_face_count or 500_000
        resolved_texture_size = texture_size or 2048
        with tempfile.TemporaryDirectory(prefix="comfycolab-3d-facade-") as directory:
            source = copy_file3d_to(model_3d, Path(directory) / "input.glb")
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


def _valid_cached_glb(
    path: Path,
    *,
    require_material: bool = False,
    require_textured: bool = False,
) -> bool:
    if not path.exists():
        return False
    try:
        validate_glb(
            path,
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
) -> None:
    numpy = importlib.import_module("numpy")
    mesh = trimesh.copy()
    mesh.apply_transform(numpy.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]))
    export_trimesh_atomic(
        mesh,
        destination,
        require_material=require_material or require_textured,
        require_texture=require_textured,
        require_uv=require_textured,
    )


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
                validate_glb(
                    destination,
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
            _export_z_up_mesh(mesh, destination, require_material=True)
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
            _export_z_up_mesh(trimesh, destination, require_textured=True)
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
        artifact_module = _load_artifact_provisioner(repo_root)
        staging = Path(tempfile.mkdtemp(prefix="comfycolab-ultrashape-input-"))
        try:
            input_mesh, image_path = staging / "input.glb", staging / "reference.png"
            copy_file3d_to(model_3d, input_mesh)
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
                run_ultrashape_worker(command, is_cancelled=cancelled, on_progress=progress)
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
    "ComfyColab3DImageOpaqueMask": ComfyColab3DImageOpaqueMask,
    "ComfyColab3DPathToFile3D": ComfyColab3DPathToFile3D,
    "ComfyColab3DTrimeshToFile3D": ComfyColab3DTrimeshToFile3D,
    "ComfyColab3DNeutralMeshToFile3D": ComfyColab3DNeutralMeshToFile3D,
    "ComfyColab3DTextureToFile3D": ComfyColab3DTextureToFile3D,
    "ComfyColab3DGLBToTrellisMesh": ComfyColab3DGLBToTrellisMesh,
    "ComfyColab3DEncodedMeshToTrimesh": ComfyColab3DEncodedMeshToTrimesh,
    "ComfyColab3DRestoreMeshTransform": ComfyColab3DRestoreMeshTransform,
    "ComfyColab3DUltraShapeWorker": ComfyColab3DUltraShapeWorker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyColabTrellisImageTo3D": "ComfyColab TRELLIS.2 — Image to 3D",
    "ComfyColabUltraShapeRefine": "ComfyColab UltraShape — Refine Geometry",
}
