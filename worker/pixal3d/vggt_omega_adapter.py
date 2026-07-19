"""Frozen VGGT-Omega geometry guidance for experimental Pixal3D multiview fusion.

VGGT-Omega predicts depth, confidence, and cameras.  Pixal3D's exact canonical
label cameras remain authoritative: predicted cameras are used only to fit one
diagnostic Sim(3), align the predicted depth point clouds, and build per-view
z-buffers for soft projection-feature weighting.

This module deliberately avoids the upstream ``vggt_omega`` package name so
the worker can import ``vggt_omega.models`` from the pinned source checkout.
"""

from __future__ import annotations

import gc
import importlib
import math
import sys
from types import ModuleType
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VGGT_OMEGA_ADAPTER_NAME = "vggt-omega-depth-confidence-fusion-v1"


def _module_resolves_within(module: ModuleType, source_dir: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    try:
        Path(module_file).resolve().relative_to(source_dir)
    except (OSError, ValueError):
        return False
    return True


def _import_pinned_vggt_omega(source_dir: Path):
    """Import upstream modules only from the verified pinned checkout."""

    source_dir = source_dir.resolve()
    source_text = str(source_dir)
    sys.path[:] = [entry for entry in sys.path if entry != source_text]
    sys.path.insert(0, source_text)
    for name in tuple(sys.modules):
        if name == "vggt_omega" or name.startswith("vggt_omega."):
            del sys.modules[name]
    importlib.invalidate_caches()

    models = importlib.import_module("vggt_omega.models")
    load_fn = importlib.import_module("vggt_omega.utils.load_fn")
    pose_enc = importlib.import_module("vggt_omega.utils.pose_enc")
    resolved = {
        "vggt_omega.models": models,
        "vggt_omega.utils.load_fn": load_fn,
        "vggt_omega.utils.pose_enc": pose_enc,
    }
    escaped = [
        name
        for name, module in resolved.items()
        if not _module_resolves_within(module, source_dir)
    ]
    if escaped:
        raise RuntimeError(
            "Pinned VGGT-Omega imports escaped the verified source checkout: "
            + ", ".join(escaped)
        )
    return (
        models.VGGTOmega,
        load_fn.load_and_preprocess_images,
        pose_enc.encoding_to_camera,
    )


@dataclass(frozen=True)
class Sim3Alignment:
    valid: bool
    scale: float
    rotation: tuple[tuple[float, float, float], ...]
    translation: tuple[float, float, float]
    rms_error: float
    normalized_rms_error: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "scale": self.scale,
            "rotation": [list(row) for row in self.rotation],
            "translation": list(self.translation),
            "rms_error": self.rms_error,
            "normalized_rms_error": self.normalized_rms_error,
            "reason": self.reason,
        }


@dataclass
class VGGTOmegaPredictions:
    labels: tuple[str, ...]
    image_size_hw: tuple[int, int]
    depth: Any
    depth_conf: Any
    extrinsics_cv: Any
    intrinsics: Any
    diagnostics: dict[str, Any]


@dataclass
class GeometryFusionContext:
    labels: tuple[str, ...]
    z_buffers: Any
    confidence_buffers: Any
    alignment: Sim3Alignment
    geometry_strength: float
    confidence_exponent: float
    depth_tolerance: float
    occlusion_margin: float
    occlusion_tau: float
    geometry_floor: float
    diagnostics: dict[str, Any]

    def weights_for_projection(
        self,
        proj_grid,
        camera: dict[str, Any],
        *,
        coords=None,
        resolution: int | None = None,
    ):
        """Return multiplicative geometry weights with shape [query_count, views]."""

        import torch
        import torch.nn.functional as torch_functional

        view_count = len(self.labels)
        grid_points = proj_grid.grid_points
        if coords is not None:
            if resolution is None or int(resolution) <= 1:
                raise ValueError("Sparse VGGT-Omega geometry weights require resolution")
            grid = grid_points.reshape(int(resolution), int(resolution), int(resolution), 3)
            grid_points = grid[
                coords[:, 1].long(),
                coords[:, 2].long(),
                coords[:, 3].long(),
            ]
        grid_points = grid_points.unsqueeze(0).expand(view_count, -1, -1)
        grid_points = (
            grid_points
            / camera["mesh_scale"].unsqueeze(-1).unsqueeze(-1)
            / 2.0
        )

        module_globals = proj_grid.__class__.forward.__globals__
        project_points = module_globals["project_points_to_image_batch"]
        image_points, query_depth, valid = project_points(
            grid_points,
            camera["transform_matrix"],
            camera["camera_angle_x"],
            int(proj_grid.image_resolution),
        )
        sample_grid = (
            (image_points + 0.5) / int(proj_grid.image_resolution) * 2.0 - 1.0
        )
        sample_grid = sample_grid.unsqueeze(2)
        z_buffers = self.z_buffers.to(
            device=query_depth.device,
            dtype=query_depth.dtype,
        ).unsqueeze(1)
        confidence_buffers = self.confidence_buffers.to(
            device=query_depth.device,
            dtype=query_depth.dtype,
        ).unsqueeze(1)
        observed_depth = torch_functional.grid_sample(
            z_buffers,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[:, 0, :, 0]
        observed_confidence = torch_functional.grid_sample(
            confidence_buffers,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[:, 0, :, 0]

        observed_valid = (
            valid
            & torch.isfinite(observed_depth)
            & (observed_depth > 0)
            & torch.isfinite(observed_confidence)
            & (observed_confidence > 0)
        )
        depth_error = (query_depth - observed_depth).abs()
        agreement = torch.exp(
            -depth_error / max(float(self.depth_tolerance), 1e-6)
        )
        visibility = torch.sigmoid(
            (
                observed_depth
                + float(self.occlusion_margin)
                - query_depth
            )
            / max(float(self.occlusion_tau), 1e-6)
        )
        confidence = observed_confidence.clamp(0.0, 1.0).pow(
            float(self.confidence_exponent)
        )
        guided = confidence * agreement * visibility
        guided = float(self.geometry_floor) + (
            1.0 - float(self.geometry_floor)
        ) * guided
        neutral = torch.ones_like(guided)
        guided = torch.where(observed_valid, guided, neutral)
        blended = (
            (1.0 - float(self.geometry_strength)) * neutral
            + float(self.geometry_strength) * guided
        )
        return blended.transpose(0, 1).contiguous()

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": VGGT_OMEGA_ADAPTER_NAME,
            "frozen": True,
            "official_pixal3d_support": False,
            "canonical_camera_policy": "exact_labeled_pixal_cameras",
            "predicted_camera_policy": "diagnostic_sim3_only",
            "register_injection": False,
            "alignment": self.alignment.to_dict(),
            "settings": {
                "geometry_strength": self.geometry_strength,
                "confidence_exponent": self.confidence_exponent,
                "depth_tolerance": self.depth_tolerance,
                "occlusion_margin": self.occlusion_margin,
                "occlusion_tau": self.occlusion_tau,
                "geometry_floor": self.geometry_floor,
            },
            **self.diagnostics,
        }


def fit_sim3_alignment(
    source_points,
    target_points,
    *,
    normalization_distance: float = 1.0,
) -> Sim3Alignment:
    """Fit one Umeyama Sim(3) mapping source points to target points."""

    import numpy

    identity = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    zero = (0.0, 0.0, 0.0)
    try:
        source = numpy.asarray(source_points, dtype=numpy.float64)
        target = numpy.asarray(target_points, dtype=numpy.float64)
    except (TypeError, ValueError) as error:
        return Sim3Alignment(False, 1.0, identity, zero, math.inf, math.inf, str(error))
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        return Sim3Alignment(
            False,
            1.0,
            identity,
            zero,
            math.inf,
            math.inf,
            f"Expected matching [N,3] camera centers, got {source.shape} and {target.shape}",
        )
    finite = numpy.isfinite(source).all(axis=1) & numpy.isfinite(target).all(axis=1)
    source = source[finite]
    target = target[finite]
    if source.shape[0] < 3:
        return Sim3Alignment(
            False,
            1.0,
            identity,
            zero,
            math.inf,
            math.inf,
            "At least three finite camera centers are required",
        )

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    if numpy.linalg.matrix_rank(source_centered) < 2:
        return Sim3Alignment(
            False,
            1.0,
            identity,
            zero,
            math.inf,
            math.inf,
            "VGGT-Omega camera centers are degenerate",
        )
    variance = float(numpy.square(source_centered).sum() / source.shape[0])
    if not math.isfinite(variance) or variance <= 1e-12:
        return Sim3Alignment(
            False,
            1.0,
            identity,
            zero,
            math.inf,
            math.inf,
            "VGGT-Omega camera-center variance is zero",
        )

    covariance = target_centered.T @ source_centered / source.shape[0]
    try:
        left, singular_values, right_t = numpy.linalg.svd(covariance)
    except numpy.linalg.LinAlgError as error:
        return Sim3Alignment(False, 1.0, identity, zero, math.inf, math.inf, str(error))
    correction = numpy.eye(3, dtype=numpy.float64)
    if numpy.linalg.det(left @ right_t) < 0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_t
    scale = float((singular_values * numpy.diag(correction)).sum() / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    mapped = scale * (source @ rotation.T) + translation
    rms = float(numpy.sqrt(numpy.square(mapped - target).sum(axis=1).mean()))
    denominator = max(float(normalization_distance), 1e-8)
    normalized_rms = rms / denominator
    if (
        not math.isfinite(scale)
        or scale <= 0
        or not numpy.isfinite(rotation).all()
        or not numpy.isfinite(translation).all()
        or not math.isfinite(rms)
    ):
        return Sim3Alignment(
            False,
            1.0,
            identity,
            zero,
            math.inf,
            math.inf,
            "Sim(3) fit produced non-finite values",
        )
    return Sim3Alignment(
        True,
        scale,
        tuple(tuple(float(value) for value in row) for row in rotation),
        tuple(float(value) for value in translation),
        rms,
        normalized_rms,
    )


def run_vggt_omega_depth_prepass(
    image_paths: list[str | Path],
    labels: list[str],
    *,
    source_dir: str | Path,
    checkpoint_path: str | Path,
    image_resolution: int = 512,
    device: str = "cuda",
) -> VGGTOmegaPredictions:
    """Run the official frozen VGGT-Omega camera/depth inference path."""

    import torch

    source_dir = Path(source_dir)
    checkpoint_path = Path(checkpoint_path)
    if not source_dir.joinpath("vggt_omega/models/vggt_omega.py").is_file():
        raise FileNotFoundError(
            f"Pinned VGGT-Omega source checkout is incomplete: {source_dir}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Pinned VGGT-Omega checkpoint is missing: {checkpoint_path}"
        )
    if len(image_paths) != len(labels):
        raise ValueError("VGGT-Omega image paths and labels must have equal length")
    if len(image_paths) < 3:
        raise ValueError("VGGT-Omega Sim(3) guidance requires at least three views")
    if int(image_resolution) != 512:
        raise ValueError("The pinned VGGT-Omega depth checkpoint requires resolution 512")
    (
        VGGTOmega,
        load_and_preprocess_images,
        encoding_to_camera,
    ) = _import_pinned_vggt_omega(source_dir)

    images = load_and_preprocess_images(
        [str(path) for path in image_paths],
        mode="max_size",
        image_resolution=int(image_resolution),
    )
    model = VGGTOmega().eval()
    try:
        try:
            state_dict = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        model = model.to(device)
        images = images.to(device)
        with torch.inference_mode():
            predictions = model(images)
        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
        )
        depth = predictions["depth"][0].detach().float().cpu().contiguous()
        confidence = (
            predictions["depth_conf"][0].detach().float().cpu().contiguous()
        )
        extrinsics = extrinsics[0].detach().float().cpu().contiguous()
        intrinsics = intrinsics[0].detach().float().cpu().contiguous()
        tokens = predictions["camera_and_register_tokens"][0]
        register_count = max(0, int(tokens.shape[1]) - 1)
        register_norm = float(tokens[:, 1:].float().norm(dim=-1).mean().item())
        finite_depth = depth[torch.isfinite(depth) & (depth > 0)]
        finite_conf = confidence[torch.isfinite(confidence)]
        diagnostics = {
            "checkpoint_kind": "VGGT-Omega-1B-512",
            "image_resolution": int(image_resolution),
            "input_tensor_shape": list(predictions["images"].shape),
            "register_token_count": register_count,
            "register_norm_mean": register_norm,
            "depth_min": float(finite_depth.min().item()) if finite_depth.numel() else None,
            "depth_max": float(finite_depth.max().item()) if finite_depth.numel() else None,
            "confidence_median": (
                float(finite_conf.median().item()) if finite_conf.numel() else None
            ),
        }
        return VGGTOmegaPredictions(
            labels=tuple(labels),
            image_size_hw=(int(depth.shape[1]), int(depth.shape[2])),
            depth=depth,
            depth_conf=confidence,
            extrinsics_cv=extrinsics,
            intrinsics=intrinsics,
            diagnostics=diagnostics,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_geometry_fusion_context(
    predictions: VGGTOmegaPredictions,
    *,
    canonical_transforms,
    camera_angle_x: float,
    camera_distance: float,
    projection_grid,
    geometry_strength: float,
    confidence_exponent: float,
    depth_tolerance: float,
    occlusion_margin: float,
    occlusion_tau: float,
    geometry_floor: float,
    max_normalized_alignment_error: float,
) -> GeometryFusionContext:
    """Align VGGT-Omega depth once and rasterize exact-camera z-buffers."""

    import numpy
    import torch

    labels = predictions.labels
    view_count = len(labels)
    if canonical_transforms.shape != (view_count, 4, 4):
        raise ValueError("Canonical VGGT-Omega transforms must have shape [views,4,4]")
    extrinsics = predictions.extrinsics_cv
    rotations = extrinsics[:, :3, :3]
    translations = extrinsics[:, :3, 3]
    source_centers = -torch.matmul(
        rotations.transpose(-1, -2),
        translations.unsqueeze(-1),
    ).squeeze(-1)
    target_centers = canonical_transforms[:, :3, 3].detach().float().cpu()
    alignment = fit_sim3_alignment(
        source_centers.numpy(),
        target_centers.numpy(),
        normalization_distance=float(camera_distance),
    )
    if not alignment.valid:
        raise RuntimeError(f"VGGT-Omega Sim(3) alignment failed: {alignment.reason}")
    if alignment.normalized_rms_error > float(max_normalized_alignment_error):
        raise RuntimeError(
            "VGGT-Omega Sim(3) alignment exceeded the configured normalized RMS: "
            f"{alignment.normalized_rms_error:.6f} > "
            f"{float(max_normalized_alignment_error):.6f}"
        )

    device = projection_grid.grid_points.device
    dtype = projection_grid.grid_points.dtype
    source_centers = source_centers.to(device=device, dtype=dtype)
    canonical_transforms = canonical_transforms.to(device=device, dtype=dtype)
    sim_rotation = torch.tensor(
        alignment.rotation,
        device=device,
        dtype=dtype,
    )
    sim_translation = torch.tensor(
        alignment.translation,
        device=device,
        dtype=dtype,
    )
    depth = predictions.depth.to(device=device, dtype=dtype)
    confidence = predictions.depth_conf.to(device=device, dtype=dtype)
    extrinsics = predictions.extrinsics_cv.to(device=device, dtype=dtype)
    intrinsics = predictions.intrinsics.to(device=device, dtype=dtype)
    height, width = predictions.image_size_hw
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    x_coords = x_coords.reshape(-1)
    y_coords = y_coords.reshape(-1)

    module_globals = projection_grid.__class__.forward.__globals__
    project_points = module_globals["project_points_to_image_batch"]
    output_resolution = int(projection_grid.image_resolution)
    z_buffers = []
    confidence_buffers = []
    coverage = {}
    for index, label in enumerate(labels):
        view_depth = depth[index, :, :, 0].reshape(-1)
        view_confidence = confidence[index].reshape(-1)
        fx = intrinsics[index, 0, 0]
        fy = intrinsics[index, 1, 1]
        cx = intrinsics[index, 0, 2]
        cy = intrinsics[index, 1, 2]
        camera_points = torch.stack(
            (
                (x_coords - cx) / fx * view_depth,
                (y_coords - cy) / fy * view_depth,
                view_depth,
            ),
            dim=-1,
        )
        camera_rotation = extrinsics[index, :3, :3]
        camera_translation = extrinsics[index, :3, 3]
        world_points = torch.matmul(
            camera_points - camera_translation,
            camera_rotation,
        )
        canonical_points = (
            float(alignment.scale)
            * torch.matmul(world_points, sim_rotation.transpose(0, 1))
            + sim_translation
        )
        image_points, projected_depth, valid = project_points(
            canonical_points,
            canonical_transforms[index : index + 1],
            torch.tensor([camera_angle_x], device=device, dtype=dtype),
            output_resolution,
        )
        image_points = image_points[0]
        projected_depth = projected_depth[0]
        valid = (
            valid[0]
            & torch.isfinite(canonical_points).all(dim=1)
            & torch.isfinite(projected_depth)
            & torch.isfinite(view_confidence)
            & (view_depth > 0)
            & (view_confidence > 0)
        )
        pixel_x = image_points[:, 0].round().long().clamp(0, output_resolution - 1)
        pixel_y = image_points[:, 1].round().long().clamp(0, output_resolution - 1)
        flat_index = pixel_y * output_resolution + pixel_x
        flat_index = flat_index[valid]
        valid_depth = projected_depth[valid]
        valid_confidence = view_confidence[valid]

        flat_z = torch.full(
            (output_resolution * output_resolution,),
            float("inf"),
            device=device,
            dtype=dtype,
        )
        flat_z.scatter_reduce_(
            0,
            flat_index,
            valid_depth,
            reduce="amin",
            include_self=True,
        )
        nearest = (
            (valid_depth - flat_z[flat_index]).abs()
            <= max(float(depth_tolerance) * 0.25, 1e-4)
        )
        finite_confidence = valid_confidence[nearest]
        if finite_confidence.numel():
            lower = torch.quantile(finite_confidence, 0.2)
            upper = torch.quantile(finite_confidence, 0.8)
            normalized_confidence = (
                (valid_confidence - lower)
                / (upper - lower).clamp(min=1e-6)
            ).clamp(0.0, 1.0)
        else:
            normalized_confidence = torch.zeros_like(valid_confidence)
        flat_confidence = torch.zeros(
            (output_resolution * output_resolution,),
            device=device,
            dtype=dtype,
        )
        flat_confidence.scatter_reduce_(
            0,
            flat_index[nearest],
            normalized_confidence[nearest],
            reduce="amax",
            include_self=True,
        )
        z_buffer = flat_z.reshape(output_resolution, output_resolution)
        confidence_buffer = flat_confidence.reshape(
            output_resolution,
            output_resolution,
        )
        valid_pixels = torch.isfinite(z_buffer)
        coverage[label] = {
            "valid_pixels": int(valid_pixels.sum().item()),
            "coverage_fraction": float(valid_pixels.float().mean().item()),
            "confidence_mean": (
                float(confidence_buffer[valid_pixels].mean().item())
                if valid_pixels.any()
                else 0.0
            ),
        }
        z_buffer = torch.where(
            valid_pixels,
            z_buffer,
            torch.zeros_like(z_buffer),
        )
        z_buffers.append(z_buffer)
        confidence_buffers.append(confidence_buffer)

    return GeometryFusionContext(
        labels=labels,
        z_buffers=torch.stack(z_buffers, dim=0),
        confidence_buffers=torch.stack(confidence_buffers, dim=0),
        alignment=alignment,
        geometry_strength=float(geometry_strength),
        confidence_exponent=float(confidence_exponent),
        depth_tolerance=float(depth_tolerance),
        occlusion_margin=float(occlusion_margin),
        occlusion_tau=float(occlusion_tau),
        geometry_floor=float(geometry_floor),
        diagnostics={
            "vggt_omega": predictions.diagnostics,
            "z_buffer_coverage": coverage,
        },
    )


__all__ = [
    "GeometryFusionContext",
    "Sim3Alignment",
    "VGGTOmegaPredictions",
    "VGGT_OMEGA_ADAPTER_NAME",
    "build_geometry_fusion_context",
    "fit_sim3_alignment",
    "run_vggt_omega_depth_prepass",
]
