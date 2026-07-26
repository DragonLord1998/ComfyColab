#!/usr/bin/env python3
"""Persistent process-isolated runner for the pinned official Pixal3D pipeline."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any


READY_PREFIX = "COMFYCOLAB_PIXAL3D_READY="
PROGRESS_PREFIX = "COMFYCOLAB_PIXAL3D_PROGRESS="
RESULT_PREFIX = "COMFYCOLAB_PIXAL3D_RESULT="
PROTOCOL_VERSION = 1
MAX_PIXAL3D_CONNECTED_COMPONENTS = 64
BEN2_SOURCE_REF = "2c99a5da477b5523585bfa5c893888a6e818a8f6"
BEN2_MODEL_REPO = "PramaLLC/BEN2"
BEN2_MODEL_REF = "e48a20765fb421d19dcdb0bf3cc61e802ca5ec8f"
BEN2_BATCH_SIZE = 3
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
NODE_PACK = REPO_ROOT / "custom_nodes" / "ComfyColab-3D"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from multiview import (  # noqa: E402
    ADAPTER_NAME,
    GEOMETRY_CONDITIONING_POLICY,
    SPARSE_STRUCTURE_POLICY,
    camera_transform_for_view,
    run_multiview_projection_fusion,
    validate_multiview_request,
)
from vggt_omega_adapter import (  # noqa: E402
    VGGT_OMEGA_ADAPTER_NAME,
    build_geometry_fusion_context,
    run_vggt_omega_depth_prepass,
)


def _emit(prefix: str, payload: dict[str, Any]) -> None:
    print(prefix + json.dumps(payload, sort_keys=True), flush=True)


def emit_progress(request_id: str, stage: str, current: int, total: int, **details: Any) -> None:
    _emit(
        PROGRESS_PREFIX,
        {"request_id": request_id, "stage": stage, "current": current, "total": total, **details},
    )


def emit_result(**details: Any) -> None:
    _emit(RESULT_PREFIX, details)


def _load_comfycolab_contract():
    package_name = "comfycolab_pixal3d_contract"
    if package_name not in sys.modules:
        specification = importlib.util.spec_from_file_location(
            package_name,
            NODE_PACK / "__init__.py",
            submodule_search_locations=[str(NODE_PACK)],
        )
        if specification is None or specification.loader is None:
            raise RuntimeError("Unable to load the ComfyColab GLB contract")
        package = importlib.util.module_from_spec(specification)
        sys.modules[package_name] = package
        specification.loader.exec_module(package)
    return importlib.import_module(f"{package_name}.file3d")


def _load_official_inference(source_dir: Path) -> ModuleType:
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    path = source_dir / "inference.py"
    if not path.is_file():
        raise FileNotFoundError(f"Pinned Pixal3D inference entrypoint is missing: {path}")
    name = "comfycolab_official_pixal3d_inference"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Unable to load official Pixal3D inference from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _init_pipeline_without_rmbg(
    official: ModuleType,
    checkpoint_dir: Path,
):
    """Load Pixal3D without constructing its unused gated background model."""

    pipeline_config = checkpoint_dir / "pipeline.json"
    try:
        payload = json.loads(pipeline_config.read_text(encoding="utf-8"))
        rembg_config = payload["args"]["rembg_model"]
        factory_name = str(rembg_config["name"])
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Pinned Pixal3D pipeline config has no valid rembg_model: {pipeline_config}"
        ) from error

    pipeline_class = getattr(official, "Pixal3DImageTo3DPipeline", None)
    pipeline_module_name = getattr(pipeline_class, "__module__", "")
    if not pipeline_module_name:
        raise RuntimeError("Pinned Pixal3D inference omitted its pipeline class")
    pipeline_module = sys.modules.get(pipeline_module_name)
    if pipeline_module is None:
        pipeline_module = importlib.import_module(pipeline_module_name)
    rembg_module = getattr(pipeline_module, "rembg", None)
    original_factory = getattr(rembg_module, factory_name, None)
    if not callable(original_factory):
        raise RuntimeError(
            f"Pinned Pixal3D rembg factory is unavailable: {factory_name}"
        )

    setattr(rembg_module, factory_name, lambda **_kwargs: None)
    try:
        pipeline = official.init_pipeline(
            str(checkpoint_dir), device="cuda", low_vram=True
        )
    finally:
        setattr(rembg_module, factory_name, original_factory)
    if getattr(pipeline, "rembg_model", None) is not None:
        raise RuntimeError("Pixal3D unexpectedly loaded a background-removal model")
    return pipeline


def _install_native_aliases() -> None:
    """Expose Comfy-env's ABI-pinned package names under upstream import names."""

    aliases = {
        "flex_gemm": ("flex_gemm_ap",),
        "cumesh": ("cumesh_vb",),
        "o_voxel": ("o_voxel_vb_ap",),
    }
    for canonical, alternatives in aliases.items():
        try:
            importlib.import_module(canonical)
            continue
        except ImportError:
            pass
        for alternative in alternatives:
            try:
                module = importlib.import_module(alternative)
            except ImportError:
                continue
            sys.modules[canonical] = module
            break
        else:
            raise ImportError(
                f"Pixal3D native module {canonical} is missing (tried {', '.join(alternatives)})"
            )
    o_voxel = importlib.import_module("o_voxel")
    _install_ovoxel_postprocess(o_voxel)


def _install_ovoxel_postprocess(o_voxel: ModuleType) -> None:
    """Restore the pure-Python exporter omitted by the ABI-pinned native wheel."""

    existing = getattr(o_voxel, "postprocess", None)
    if callable(getattr(existing, "to_glb", None)):
        return
    postprocess = importlib.import_module("ovoxel_postprocess")
    if not callable(getattr(postprocess, "to_glb", None)):
        raise ImportError("Vendored O-Voxel postprocess module omitted to_glb")
    o_voxel.postprocess = postprocess
    sys.modules["o_voxel.postprocess"] = postprocess


def _prepare_image_without_rmbg(image_path: Path):
    """Apply Pixal3D's crop/composite policy without loading its gated RMBG model."""

    import numpy as np
    from PIL import Image

    image = Image.open(image_path).convert("RGBA")
    max_size = max(image.size)
    scale = min(1.0, 1024.0 / max_size)
    if scale < 1.0:
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    rgba = np.asarray(image)
    foreground = np.argwhere(rgba[:, :, 3] > int(0.8 * 255))
    if foreground.size == 0:
        raise ValueError("Prepared Pixal3D input contains no visible foreground")
    left, top = int(foreground[:, 1].min()), int(foreground[:, 0].min())
    right, bottom = int(foreground[:, 1].max()), int(foreground[:, 0].max())
    center_x, center_y = (left + right) / 2.0, (top + bottom) / 2.0
    size = max(1, int(max(right - left, bottom - top) * 1.1))
    crop_left = math.floor(center_x - size / 2.0)
    crop_top = math.floor(center_y - size / 2.0)
    crop = (crop_left, crop_top, crop_left + size, crop_top + size)
    cropped = np.asarray(image.crop(crop)).astype(np.float32) / 255.0
    rgb, alpha = cropped[:, :, :3], cropped[:, :, 3:4]
    composited = np.clip(rgb * alpha, 0.0, 1.0)
    return Image.fromarray((composited * 255.0).astype(np.uint8), mode="RGB")


def _run_ben2_background_removal(
    image_paths: list[Path],
    output_paths: list[Path],
    *,
    request_id: str,
) -> None:
    """Extract transparent foregrounds with the pinned open-source BEN2 Base model."""

    if not image_paths or len(image_paths) != len(output_paths):
        raise ValueError("BEN2 requires matching non-empty input and output path lists")
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("BEN2 background removal requires the Pixal3D CUDA runtime")
    ben2 = importlib.import_module("ben2")
    huggingface_hub = importlib.import_module("huggingface_hub")
    pil_image = importlib.import_module("PIL.Image")
    snapshot = huggingface_hub.snapshot_download(
        repo_id=BEN2_MODEL_REPO,
        revision=BEN2_MODEL_REF,
    )
    model = ben2.BEN_Base.from_pretrained(snapshot)
    model.to("cuda").eval()
    completed = 0
    emit_progress(
        request_id,
        "background_removal",
        completed,
        len(image_paths),
        backend="ben2",
    )
    try:
        for offset in range(0, len(image_paths), BEN2_BATCH_SIZE):
            chunk_paths = image_paths[offset : offset + BEN2_BATCH_SIZE]
            images = [pil_image.open(path).convert("RGB") for path in chunk_paths]
            try:
                with torch.inference_mode():
                    foregrounds = model.inference(
                        images,
                        refine_foreground=True,
                    )
                if not isinstance(foregrounds, list):
                    foregrounds = [foregrounds]
                if len(foregrounds) != len(chunk_paths):
                    raise RuntimeError(
                        "BEN2 returned a different number of foregrounds than inputs"
                    )
                for foreground, output_path in zip(
                    foregrounds,
                    output_paths[offset : offset + len(chunk_paths)],
                    strict=True,
                ):
                    foreground.convert("RGBA").save(output_path)
                    completed += 1
                    emit_progress(
                        request_id,
                        "background_removal",
                        completed,
                        len(image_paths),
                        backend="ben2",
                    )
            finally:
                for image in images:
                    image.close()
    finally:
        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _validate_request(request: dict[str, Any]) -> None:
    if int(request.get("protocol", -1)) != PROTOCOL_VERSION:
        raise ValueError("Unsupported Pixal3D worker protocol")
    if request.get("pipeline_type") not in {"1024_cascade", "1536_cascade"}:
        raise ValueError("pipeline_type must be 1024_cascade or 1536_cascade")
    seed = int(request.get("seed", -1))
    if seed < 0 or seed > (2**31) - 1:
        raise ValueError("Pixal3D seed must be between 0 and 2147483647")
    if not 1 <= int(request.get("sampling_steps", 0)) <= 100:
        raise ValueError("Pixal3D sampling_steps must be between 1 and 100")
    if int(request.get("target_face_count", 0)) < 1000:
        raise ValueError("Pixal3D target_face_count must be at least 1000")
    if int(request.get("texture_size", 0)) < 512:
        raise ValueError("Pixal3D texture_size must be at least 512")
    if int(request.get("max_tokens", 0)) < 16_384:
        raise ValueError("Pixal3D max_tokens must be at least 16384")
    if str(request.get("background_removal", "none")) not in {"none", "ben2"}:
        raise ValueError("background_removal must be none or ben2")
    if request.get("surface_point_cloud"):
        if not str(request["surface_point_cloud"]).lower().endswith(".ply"):
            raise ValueError("surface_point_cloud must use a .ply path")
        if int(request.get("surface_point_count", 0)) < 4096:
            raise ValueError("surface_point_count must be at least 4096")
    for name in ("image_path", "output_mesh", "metadata_output", "request_id"):
        if not str(request.get(name, "")):
            raise ValueError(f"Pixal3D request omitted {name}")
    if not Path(str(request["image_path"])).is_file():
        raise FileNotFoundError(f"Prepared Pixal3D input does not exist: {request['image_path']}")
    fov = request.get("camera_fov_radians")
    if fov is not None and (not math.isfinite(float(fov)) or not 0.0 < float(fov) < math.pi):
        raise ValueError("Manual Pixal3D FOV must be between 0 and pi radians")
    views = validate_multiview_request(request)
    guidance = str(request.get("geometry_guidance", "none"))
    if guidance not in {"none", "vggt_omega_depth_conf"}:
        raise ValueError("geometry_guidance must be none or vggt_omega_depth_conf")
    if guidance == "none":
        requested = str(request.get("geometry_requested", ""))
        if not requested:
            return
        if requested != "vggt_omega_depth_conf":
            raise ValueError(
                "geometry_requested must identify vggt_omega_depth_conf"
            )
        if not views:
            raise ValueError("A VGGT-Omega fallback requires multiview inputs")
        if str(request.get("geometry_fallback", "")) != "weighted_mv":
            raise ValueError(
                "A recorded VGGT-Omega fallback requires weighted_mv policy"
            )
        if not str(request.get("geometry_fallback_stage", "")):
            raise ValueError("A recorded VGGT-Omega fallback omitted its stage")
        if not str(request.get("geometry_fallback_reason", "")):
            raise ValueError("A recorded VGGT-Omega fallback omitted its reason")
        return
    if not views:
        raise ValueError("VGGT-Omega geometry guidance requires Pixal3D multiview inputs")
    if len(views) < 3:
        raise ValueError("VGGT-Omega geometry guidance requires at least three views")
    fallback = str(request.get("geometry_fallback", "strict"))
    if fallback not in {"strict", "weighted_mv"}:
        raise ValueError("geometry_fallback must be strict or weighted_mv")
    if int(request.get("vggt_omega_image_resolution", 512)) != 512:
        raise ValueError("The pinned VGGT-Omega checkpoint requires image resolution 512")
    for name, minimum, maximum in (
        ("geometry_strength", 0.0, 1.0),
        ("confidence_exponent", 0.0, 4.0),
        ("depth_tolerance", 1e-4, 1.0),
        ("occlusion_margin", 0.0, 0.5),
        ("occlusion_tau", 1e-4, 0.5),
        ("geometry_floor", 0.0, 1.0),
        ("max_normalized_alignment_error", 0.0, 1.0),
    ):
        value = float(request.get(name, minimum))
        if not math.isfinite(value) or value < minimum or value > maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export_surface_point_cloud(
    vertices,
    faces,
    output_path: Path,
    *,
    point_count: int,
    seed: int,
) -> dict[str, Any]:
    """Export deterministic area-weighted samples from Pixal's decoded surface."""

    numpy = importlib.import_module("numpy")
    trimesh = importlib.import_module("trimesh")
    vertex_array = vertices.detach().float().cpu().numpy()
    face_array = faces.detach().long().cpu().numpy()
    keep_faces, removed_planar_components = importlib.import_module(
        "mesh_cleanup"
    ).large_planar_component_face_mask(vertex_array, face_array)
    face_array = face_array[keep_faces]
    if face_array.shape[0] == 0:
        raise RuntimeError(
            "Pixal3D surface point-cloud export removed every decoded face"
        )

    triangles = vertex_array[face_array]
    cross = numpy.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = numpy.linalg.norm(cross, axis=1) * 0.5
    valid = numpy.isfinite(areas) & (areas > 1e-12)
    triangles = triangles[valid]
    areas = areas[valid]
    if triangles.shape[0] == 0:
        raise RuntimeError(
            "Pixal3D surface point-cloud export found no finite nondegenerate faces"
        )

    count = int(point_count)
    if count < 4096:
        raise ValueError("Pixal3D surface point cloud requires at least 4096 points")
    rng = numpy.random.default_rng(int(seed))
    sampled_faces = rng.choice(
        triangles.shape[0],
        size=count,
        replace=True,
        p=areas / areas.sum(),
    )
    selected = triangles[sampled_faces]
    first = rng.random(count, dtype=numpy.float32)
    second = rng.random(count, dtype=numpy.float32)
    root = numpy.sqrt(first)
    barycentric = numpy.stack(
        (1.0 - root, root * (1.0 - second), root * second),
        axis=1,
    )
    points = (selected * barycentric[:, :, None]).sum(axis=1).astype(
        numpy.float32,
        copy=False,
    )

    # Match the final Pixal GLB orientation while preserving the pre-remesh
    # decoded surface. O-Voxel's internal axis swap followed by the worker's
    # GLB transform simplifies to a 180-degree rotation around Y.
    points[:, 0] *= -1.0
    points[:, 2] *= -1.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(
        f".{output_path.stem}.partial{output_path.suffix}"
    )
    partial.unlink(missing_ok=True)
    trimesh.PointCloud(points).export(str(partial))
    if not partial.is_file() or partial.stat().st_size <= 0:
        raise RuntimeError("Pixal3D failed to export its surface point cloud")
    os.replace(partial, output_path)
    return {
        "path": str(output_path),
        "sha256": _sha256(output_path),
        "points": int(points.shape[0]),
        "source_vertices": int(vertex_array.shape[0]),
        "source_faces": int(faces.shape[0]),
        "sampled_faces": int(triangles.shape[0]),
        "removed_planar_components": removed_planar_components,
        "sampling": "deterministic_area_weighted_surface_v1",
        "coordinate_frame": "pixal_final_glb",
        "pre_glb_remesh": True,
    }


def _git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Pinned source checkout is invalid: {path}") from error


def _snapshot_revision(path: Path) -> str:
    try:
        marker = json.loads(
            (path / ".comfycolab-artifact.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Pinned model snapshot marker is invalid: {path}") from error
    revision = str(marker.get("revision", ""))
    if not revision:
        raise RuntimeError(f"Pinned model snapshot marker omitted its revision: {path}")
    return revision


def _geometry_guidance_enabled(request: dict[str, Any]) -> bool:
    return str(request.get("geometry_guidance", "none")) == "vggt_omega_depth_conf"


def _reject_excessive_fragmentation(validation) -> None:
    if isinstance(validation, dict):
        component_count = int(validation.get("connected_component_count", -1))
        exact = bool(validation.get("connected_components_exact", False))
    else:
        component_count = int(getattr(validation, "connected_component_count", -1))
        exact = bool(getattr(validation, "connected_components_exact", False))
    component_limit = int(
        os.environ.get(
            "COMFYCOLAB_PIXAL3D_MAX_CONNECTED_COMPONENTS",
            str(MAX_PIXAL3D_CONNECTED_COMPONENTS),
        )
    )
    if component_limit < 1:
        raise ValueError("Pixal3D connected-component limit must be positive")
    if exact and component_count > component_limit:
        raise RuntimeError(
            "Pixal3D generated an excessively fragmented mesh "
            f"({component_count} connected components; maximum accepted "
            f"{component_limit}). The artifact was rejected "
            "before publication."
        )


def _geometry_fallback_metadata(
    request: dict[str, Any],
    error: BaseException,
    stage: str,
) -> dict[str, Any]:
    return {
        "adapter": VGGT_OMEGA_ADAPTER_NAME,
        "frozen": True,
        "official_pixal3d_support": False,
        "canonical_camera_policy": "exact_labeled_pixal_cameras",
        "predicted_camera_policy": "not_used_fallback",
        "register_injection": False,
        "status": "fallback_weighted_mv",
        "fallback": str(request.get("geometry_fallback", "strict")),
        "failed_stage": stage,
        "warning": (
            "VGGT-Omega geometry guidance failed; generated with weighted "
            "Pixal3D projection fusion only."
        ),
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _recorded_geometry_fallback_metadata(
    request: dict[str, Any],
) -> dict[str, Any] | None:
    if str(request.get("geometry_requested", "")) != "vggt_omega_depth_conf":
        return None
    return {
        "adapter": VGGT_OMEGA_ADAPTER_NAME,
        "frozen": True,
        "official_pixal3d_support": False,
        "canonical_camera_policy": "exact_labeled_pixal_cameras",
        "predicted_camera_policy": "not_used_fallback",
        "register_injection": False,
        "status": "fallback_weighted_mv",
        "fallback": "weighted_mv",
        "failed_stage": str(request["geometry_fallback_stage"]),
        "warning": (
            "VGGT-Omega geometry guidance was requested but its artifacts were "
            "unavailable; generated with weighted Pixal3D projection fusion only."
        ),
        "error_type": "ArtifactProvisioningError",
        "error": str(request["geometry_fallback_reason"]),
    }


class Pixal3DRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.official: ModuleType | None = None
        self.pipeline = None
        self.torch = None
        self.pipeline_load_count = 0

    def _pinned_torch_hub_loader(self, original_load):
        checkpoint = self.args.naf_checkpoint
        source = self.args.naf_source_dir

        def load(repo_or_dir, model, *args, **kwargs):
            if repo_or_dir != "valeoai/NAF":
                return original_load(repo_or_dir, model, *args, **kwargs)
            if not source.is_dir() or not checkpoint.is_file():
                raise FileNotFoundError("Pinned NAF source/checkpoint is missing")
            device = kwargs.get("device", "cpu")
            pinned_kwargs = dict(kwargs)
            pinned_kwargs.pop("trust_repo", None)
            pinned_kwargs["pretrained"] = False
            pinned_kwargs["device"] = device
            pinned_kwargs["source"] = "local"
            model_value = original_load(str(source), model, *args, **pinned_kwargs)
            try:
                state = self.torch.load(checkpoint, map_location=device, weights_only=True)
            except TypeError:
                state = self.torch.load(checkpoint, map_location=device)
            model_value.load_state_dict(state, strict=True)
            return model_value

        return load

    def ensure_pipeline(self, request_id: str):
        if self.pipeline is not None:
            return self.pipeline
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault(
            "ATTN_BACKEND", os.environ.get("COMFYCOLAB_PIXAL3D_ATTN_BACKEND", "sdpa")
        )
        emit_progress(request_id, "load_pipeline", 0, 1)
        _install_native_aliases()
        self.official = _load_official_inference(self.args.source_dir)
        self.torch = importlib.import_module("torch")
        if not self.torch.cuda.is_available():
            raise RuntimeError("Pixal3D requires a CUDA runtime")
        if not self.args.checkpoint_dir.joinpath("pipeline.json").is_file():
            raise FileNotFoundError("Pinned Pixal3D model snapshot is incomplete")
        if not self.args.dinov3_dir.joinpath("config.json").is_file():
            raise FileNotFoundError("Pinned DINOv3 snapshot is incomplete")
        if not self.args.moge_dir.joinpath("model.pt").is_file():
            raise FileNotFoundError("Pinned MoGe snapshot is incomplete")
        for config in self.official.IMAGE_COND_CONFIGS.values():
            config["model_name"] = str(self.args.dinov3_dir)
        original_load = self.torch.hub.load
        self.torch.hub.load = self._pinned_torch_hub_loader(original_load)
        try:
            self.pipeline = _init_pipeline_without_rmbg(
                self.official,
                self.args.checkpoint_dir,
            )
        finally:
            self.torch.hub.load = original_load
        self.pipeline_load_count += 1
        emit_progress(
            request_id,
            "load_pipeline",
            1,
            1,
            pipeline_load_count=self.pipeline_load_count,
        )
        return self.pipeline

    def resolved_revisions(self, request: dict[str, Any]) -> dict[str, str]:
        actual = {
            "source": _git_revision(self.args.source_dir),
            "model": _snapshot_revision(self.args.checkpoint_dir),
            "dinov3": _snapshot_revision(self.args.dinov3_dir),
            "moge": _snapshot_revision(self.args.moge_dir),
            "naf": _git_revision(self.args.naf_source_dir),
            "environment": os.environ.get("COMFYCOLAB_PIXAL3D_ENVIRONMENT_REF", ""),
            "naf_checkpoint": _sha256(self.args.naf_checkpoint),
        }
        if _geometry_guidance_enabled(request):
            source_dir = self.args.vggt_omega_source_dir
            checkpoint = self.args.vggt_omega_checkpoint
            if source_dir is None or not Path(source_dir).is_dir():
                raise RuntimeError("VGGT-Omega source checkout is required for geometry guidance")
            if checkpoint is None or not Path(checkpoint).is_file():
                raise RuntimeError("VGGT-Omega checkpoint is required for geometry guidance")
            actual.update(
                {
                    "vggt_omega_source": _git_revision(Path(source_dir)),
                    "vggt_omega_checkpoint": _snapshot_revision(Path(checkpoint).parent),
                }
            )
        requested = request.get("revisions")
        if not isinstance(requested, dict):
            raise RuntimeError("Pixal3D request omitted pinned revision claims")
        required = [
            "source",
            "model",
            "dinov3",
            "moge",
            "naf",
            "naf_checkpoint",
            "environment",
        ]
        if _geometry_guidance_enabled(request):
            required.extend(("vggt_omega_source", "vggt_omega_checkpoint"))
        for name in required:
            if not actual[name] or actual[name] != str(requested.get(name, "")):
                raise RuntimeError(
                    f"Pixal3D {name} revision mismatch: requested "
                    f"{requested.get(name)!r}, resolved {actual[name]!r}"
                )
        return actual

    def _camera_params(
        self, request: dict[str, Any], prepared_image_path: Path
    ) -> dict[str, float]:
        assert self.official is not None and self.torch is not None
        fov = request.get("camera_fov_radians")
        if fov is not None:
            camera_angle_x = float(fov)
            distance = self.official.distance_from_fov(
                camera_angle_x,
                self.torch.tensor([-1.0, 0.0, 0.0]),
                self.torch.tensor([0.0, 511.0]),
                1.0,
                512,
            )["distance_from_x"]
            return {
                "camera_angle_x": camera_angle_x,
                "distance": float(distance),
                "mesh_scale": 1.0,
            }
        moge = self.official.load_moge_model(
            device="cuda", model_name=str(self.args.moge_dir / "model.pt")
        )
        try:
            return self.official.get_camera_params_wild_moge(
                str(prepared_image_path),
                moge,
                device="cuda",
                mesh_scale=1.0,
                extend_pixel=0,
                image_resolution=512,
            )
        finally:
            moge.cpu()
            del moge
            gc.collect()
            self.torch.cuda.empty_cache()

    @staticmethod
    def _token_count(shape_slat) -> int:
        coords = getattr(shape_slat, "coords", None)
        if coords is None:
            return 0
        shape = getattr(coords, "shape", ())
        return int(shape[0]) if shape else 0

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        _validate_request(request)
        request_id = str(request["request_id"])
        started = time.monotonic()
        output = Path(str(request["output_mesh"])).resolve()
        metadata_output = Path(str(request["metadata_output"])).resolve()
        surface_point_cloud = (
            Path(str(request["surface_point_cloud"])).resolve()
            if request.get("surface_point_cloud")
            else None
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        if surface_point_cloud is not None:
            surface_point_cloud.parent.mkdir(parents=True, exist_ok=True)
        partial_output = output.with_name(f".{output.stem}.{request_id}.partial.glb")
        partial_metadata = metadata_output.with_name(
            f".{metadata_output.stem}.{request_id}.partial.json"
        )
        prepared_camera_image = output.with_name(f".{output.stem}.{request_id}.camera.png")
        prepared_view_paths: list[Path] = []
        background_removed_paths: list[Path] = []
        for path in (
            output,
            metadata_output,
            partial_output,
            partial_metadata,
            prepared_camera_image,
            *([surface_point_cloud] if surface_point_cloud is not None else []),
        ):
            path.unlink(missing_ok=True)
        try:
            resolved_revisions = self.resolved_revisions(request)
            views = validate_multiview_request(request)
            labels = [str(view["name"]) for view in views]
            raw_input_paths = (
                [Path(str(view["image_path"])) for view in views]
                if views
                else [Path(str(request["image_path"]))]
            )
            prepared_input_paths = raw_input_paths
            background_removal = str(request.get("background_removal", "none"))
            if background_removal == "ben2":
                background_removed_paths = [
                    output.with_name(
                        f".{output.stem}.{request_id}.ben2-{index}.png"
                    )
                    for index in range(len(raw_input_paths))
                ]
                for path in background_removed_paths:
                    path.unlink(missing_ok=True)
                _run_ben2_background_removal(
                    raw_input_paths,
                    background_removed_paths,
                    request_id=request_id,
                )
                prepared_input_paths = background_removed_paths
            image = _prepare_image_without_rmbg(prepared_input_paths[0])
            image.save(prepared_camera_image)
            view_images = []
            for view, prepared_input_path in zip(
                views,
                prepared_input_paths if views else [],
                strict=True,
            ):
                label = str(view["name"])
                prepared_path = output.with_name(
                    f".{output.stem}.{request_id}.{label}.png"
                )
                prepared_path.unlink(missing_ok=True)
                prepared_image = _prepare_image_without_rmbg(
                    prepared_input_path
                )
                prepared_image.save(prepared_path)
                prepared_view_paths.append(prepared_path)
                view_images.append(prepared_image)

            omega_predictions = None
            advanced_geometry = _recorded_geometry_fallback_metadata(request)
            geometry_context = None
            if _geometry_guidance_enabled(request):
                try:
                    emit_progress(request_id, "vggt_omega", 0, 2, stage_detail="depth_prepass")
                    omega_predictions = run_vggt_omega_depth_prepass(
                        prepared_view_paths,
                        labels,
                        source_dir=self.args.vggt_omega_source_dir,
                        checkpoint_path=self.args.vggt_omega_checkpoint,
                        image_resolution=int(request.get("vggt_omega_image_resolution", 512)),
                        device="cuda",
                    )
                    emit_progress(request_id, "vggt_omega", 1, 2, stage_detail="depth_prepass")
                except BaseException as error:
                    if str(request.get("geometry_fallback", "strict")) != "weighted_mv":
                        raise RuntimeError(
                            "VGGT-Omega geometry guidance failed in strict mode: "
                            f"{type(error).__name__}: {error}"
                        ) from error
                    advanced_geometry = _geometry_fallback_metadata(
                        request,
                        error,
                        "depth_prepass",
                    )
            pipeline = self.ensure_pipeline(request_id)
            assert self.official is not None and self.torch is not None
            self.torch.cuda.reset_peak_memory_stats()
            emit_progress(request_id, "camera", 0, 1)
            camera_params = self._camera_params(request, prepared_camera_image)
            emit_progress(
                request_id,
                "camera",
                1,
                1,
                camera_angle_x=camera_params["camera_angle_x"],
                distance=camera_params["distance"],
            )
            steps = int(request["sampling_steps"])
            guidance = float(request["guidance_scale"])
            sampler_ss = {
                "steps": steps,
                "guidance_strength": guidance,
                "guidance_rescale": 0.7,
                "rescale_t": 5.0,
            }
            sampler_shape = {
                "steps": steps,
                "guidance_strength": guidance,
                "guidance_rescale": 0.5,
                "rescale_t": 3.0,
            }
            sampler_texture = {
                "steps": steps,
                "guidance_strength": 1.0,
                "guidance_rescale": 0.0,
                "rescale_t": 3.0,
            }
            if omega_predictions is not None:
                try:
                    canonical_transforms = self.torch.stack(
                        [
                            camera_transform_for_view(
                                label,
                                float(camera_params["distance"]),
                            )
                            for label in labels
                        ]
                    ).to(pipeline.device)
                    geometry_context = build_geometry_fusion_context(
                        omega_predictions,
                        canonical_transforms=canonical_transforms,
                        camera_angle_x=float(camera_params["camera_angle_x"]),
                        camera_distance=float(camera_params["distance"]),
                        projection_grid=pipeline.image_cond_model_ss.proj_grid,
                        geometry_strength=float(request.get("geometry_strength", 0.75)),
                        confidence_exponent=float(
                            request.get("confidence_exponent", 1.0)
                        ),
                        depth_tolerance=float(request.get("depth_tolerance", 0.12)),
                        occlusion_margin=float(request.get("occlusion_margin", 0.04)),
                        occlusion_tau=float(request.get("occlusion_tau", 0.03)),
                        geometry_floor=float(request.get("geometry_floor", 0.05)),
                        max_normalized_alignment_error=float(
                            request.get("max_normalized_alignment_error", 0.35)
                        ),
                    )
                    advanced_geometry = geometry_context.metadata()
                    emit_progress(
                        request_id,
                        "vggt_omega",
                        2,
                        2,
                        stage_detail="geometry_context",
                    )
                except BaseException as error:
                    if str(request.get("geometry_fallback", "strict")) != "weighted_mv":
                        raise RuntimeError(
                            "VGGT-Omega geometry context failed in strict mode: "
                            f"{type(error).__name__}: {error}"
                        ) from error
                    geometry_context = None
                    advanced_geometry = _geometry_fallback_metadata(
                        request,
                        error,
                        "geometry_context",
                    )
            emit_progress(request_id, "generate", 0, 5)
            if views:
                mesh_list, (shape_slat, _tex_slat, actual_resolution) = (
                    run_multiview_projection_fusion(
                        pipeline,
                        view_images,
                        camera_params,
                        labels=labels,
                        view_qualities={
                            str(view["name"]): float(view.get("quality", 1.0))
                            for view in views
                        },
                        seed=int(request["seed"]),
                        sparse_structure_sampler_params=sampler_ss,
                        shape_slat_sampler_params=sampler_shape,
                        tex_slat_sampler_params=sampler_texture,
                        pipeline_type=str(request["pipeline_type"]),
                        max_num_tokens=int(request["max_tokens"]),
                        fusion_strategy=str(
                            request.get("fusion_strategy", "directional_softmax")
                        ),
                        fusion_temperature=float(
                            request.get("fusion_temperature", 2.0)
                        ),
                        geometry_context=geometry_context,
                        return_latent=True,
                    )
                )
            else:
                mesh_list, (shape_slat, _tex_slat, actual_resolution) = pipeline.run(
                    image,
                    camera_params=camera_params,
                    seed=int(request["seed"]),
                    sparse_structure_sampler_params=sampler_ss,
                    shape_slat_sampler_params=sampler_shape,
                    tex_slat_sampler_params=sampler_texture,
                    preprocess_image=False,
                    return_latent=True,
                    pipeline_type=str(request["pipeline_type"]),
                    max_num_tokens=int(request["max_tokens"]),
                )
            expected_resolution = 1536 if request["pipeline_type"] == "1536_cascade" else 1024
            token_count = self._token_count(shape_slat)
            if int(actual_resolution) != expected_resolution:
                raise RuntimeError(
                    f"Pixal3D {expected_resolution} request required more than "
                    f"max_tokens={request['max_tokens']} and internally resolved to "
                    f"{actual_resolution}. No downgraded artifact was returned; increase max_tokens "
                    "or use 1024 — Stable."
                )
            if not mesh_list:
                raise RuntimeError("Pixal3D returned no mesh")
            emit_progress(request_id, "generate", 5, 5, actual_resolution=actual_resolution)
            mesh = mesh_list[0]
            o_voxel = importlib.import_module("o_voxel")
            numpy = importlib.import_module("numpy")
            surface_point_cloud_metadata = None
            if surface_point_cloud is not None:
                emit_progress(request_id, "surface_point_cloud", 0, 1)
                surface_point_cloud_metadata = _export_surface_point_cloud(
                    mesh.vertices,
                    mesh.faces,
                    surface_point_cloud,
                    point_count=int(request.get("surface_point_count", 65_536)),
                    seed=int(request["seed"]),
                )
                emit_progress(
                    request_id,
                    "surface_point_cloud",
                    1,
                    1,
                    points=surface_point_cloud_metadata["points"],
                )
            emit_progress(request_id, "export", 0, 1)
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=pipeline.pbr_attr_layout,
                grid_size=int(actual_resolution),
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=int(request["target_face_count"]),
                texture_size=int(request["texture_size"]),
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                use_tqdm=True,
            )
            glb.apply_transform(
                numpy.array(
                    [
                        [-1, 0, 0, 0],
                        [0, 0, -1, 0],
                        [0, -1, 0, 0],
                        [0, 0, 0, 1],
                    ],
                    dtype=numpy.float64,
                )
            )
            # Use core glTF image sources for broad viewer compatibility.
            # EXT_texture_webp-only textures render as glossy black in viewers
            # that do not implement that optional extension.
            glb.export(str(partial_output), extension_webp=False)
            validation = _load_comfycolab_contract().validate_volumetric_glb(
                partial_output,
                stage="Pixal3D generated GLB",
                require_material=True,
                require_texture=True,
                require_uv=True,
            )
            _reject_excessive_fragmentation(validation)
            validation_payload = (
                validation.to_dict() if hasattr(validation, "to_dict") else validation
            )
            metadata = {
                "schema": "comfycolab-pixal3d-worker-result-v1",
                "request_id": request_id,
                "settings": {
                    "pipeline_type": request["pipeline_type"],
                    "seed": int(request["seed"]),
                    "sampling_steps": steps,
                    "target_face_count": int(request["target_face_count"]),
                    "texture_size": int(request["texture_size"]),
                    "max_tokens": int(request["max_tokens"]),
                },
                "background_removal": {
                    "backend": background_removal,
                    "source_ref": (
                        BEN2_SOURCE_REF if background_removal == "ben2" else ""
                    ),
                    "model_repo": (
                        BEN2_MODEL_REPO if background_removal == "ben2" else ""
                    ),
                    "model_ref": (
                        BEN2_MODEL_REF if background_removal == "ben2" else ""
                    ),
                    "refine_foreground": background_removal == "ben2",
                },
                "camera": camera_params,
                "actual_resolution": int(actual_resolution),
                "token_count": token_count,
                "peak_vram_bytes": int(self.torch.cuda.max_memory_allocated()),
                "runtime_seconds": time.monotonic() - started,
                "worker_pid": os.getpid(),
                "pipeline_load_count": self.pipeline_load_count,
                "revisions": resolved_revisions,
                "validation": validation_payload,
                "surface_point_cloud_metadata": surface_point_cloud_metadata,
                "bytes": partial_output.stat().st_size,
            }
            if views:
                metadata["experimental_multiview"] = {
                    "adapter": ADAPTER_NAME,
                    "geometry_conditioning_policy": GEOMETRY_CONDITIONING_POLICY,
                    "sparse_structure_policy": SPARSE_STRUCTURE_POLICY,
                    "official_pixal3d_support": False,
                    "views": [
                        {
                            "name": str(view["name"]),
                            "image_path": str(Path(str(view["image_path"])).resolve()),
                            "quality": float(view.get("quality", 1.0)),
                        }
                        for view in views
                    ],
                    "fusion_strategy": str(
                        request.get("fusion_strategy", "directional_softmax")
                    ),
                    "fusion_temperature": float(
                        request.get("fusion_temperature", 2.0)
                    ),
                }
                if advanced_geometry is not None:
                    metadata["experimental_multiview"][
                        "advanced_geometry"
                    ] = advanced_geometry
            partial_metadata.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(partial_output, output)
            os.replace(partial_metadata, metadata_output)
            emit_progress(request_id, "export", 1, 1, bytes=output.stat().st_size)
            return metadata
        except BaseException:
            for path in (
                output,
                metadata_output,
                partial_output,
                partial_metadata,
                prepared_camera_image,
                *prepared_view_paths,
                *background_removed_paths,
                *([surface_point_cloud] if surface_point_cloud is not None else []),
            ):
                path.unlink(missing_ok=True)
            raise
        finally:
            prepared_camera_image.unlink(missing_ok=True)
            for path in prepared_view_paths:
                path.unlink(missing_ok=True)
            for path in background_removed_paths:
                path.unlink(missing_ok=True)


def _request_from_args(args: argparse.Namespace) -> dict[str, Any]:
    fov = math.radians(args.camera_fov_degrees) if args.camera_fov_degrees > 0 else None
    request = {
        "protocol": PROTOCOL_VERSION,
        "request_id": args.request_id,
        "image_path": str(args.image_path),
        "output_mesh": str(args.output_mesh),
        "metadata_output": str(args.metadata_output),
        "seed": args.seed,
        "pipeline_type": args.pipeline_type,
        "sampling_steps": args.sampling_steps,
        "guidance_scale": 7.5,
        "camera_fov_radians": fov,
        "target_face_count": args.target_face_count,
        "texture_size": args.texture_size,
        "max_tokens": args.max_tokens,
        "revisions": {},
    }
    if args.surface_point_cloud:
        request["surface_point_cloud"] = str(args.surface_point_cloud)
        request["surface_point_count"] = int(args.surface_point_count)
    return request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--one-shot", action="store_true")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dinov3-dir", type=Path, required=True)
    parser.add_argument("--moge-dir", type=Path, required=True)
    parser.add_argument("--naf-source-dir", type=Path, required=True)
    parser.add_argument("--naf-checkpoint", type=Path, required=True)
    parser.add_argument("--vggt-omega-source-dir", type=Path)
    parser.add_argument("--vggt-omega-checkpoint", type=Path)
    parser.add_argument("--request-id", default="")
    parser.add_argument("--image-path", type=Path)
    parser.add_argument("--output-mesh", type=Path)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--surface-point-cloud", type=Path)
    parser.add_argument("--surface-point-count", type=int, default=65_536)
    parser.add_argument("--pipeline-type", default="1024_cascade")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-fov-degrees", type=float, default=0.0)
    parser.add_argument("--sampling-steps", type=int, default=12)
    parser.add_argument("--target-face-count", type=int, default=200_000)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=49_152)
    return parser


def _handle(runtime: Pixal3DRuntime, request: dict[str, Any]) -> bool:
    if request.get("command") == "shutdown":
        return False
    request_id = str(request.get("request_id", "unknown"))
    try:
        metadata = runtime.run(request)
        emit_result(
            **{
                **metadata,
                "request_id": request_id,
                "status": "ok",
                "output_mesh": str(Path(request["output_mesh"]).resolve()),
                "metadata_output": str(Path(request["metadata_output"]).resolve()),
                "surface_point_cloud": (
                    str(Path(request["surface_point_cloud"]).resolve())
                    if request.get("surface_point_cloud")
                    else ""
                ),
            }
        )
    except BaseException as error:
        emit_result(
            request_id=request_id,
            status="error",
            error=str(error),
            error_type=type(error).__name__,
            traceback="".join(traceback.format_exception(error))[-8000:],
        )
    return True


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = Pixal3DRuntime(args)
    if args.one_shot:
        return 0 if _handle(runtime, _request_from_args(args)) else 0
    if not args.server:
        raise SystemExit("Use --server or --one-shot")
    _emit(READY_PREFIX, {"protocol": PROTOCOL_VERSION, "pid": os.getpid()})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Pixal3D protocol requests must be JSON objects")
        except BaseException as error:
            emit_result(
                request_id="unknown",
                status="error",
                error=str(error),
                error_type=type(error).__name__,
            )
            continue
        if not _handle(runtime, request):
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
