"""Experimental Pixal3D multiview projection-fusion adapter.

This is not native Pixal3D multiview support.  It keeps the upstream Pixal3D
flow samplers intact, encodes each labeled view with Pixal3D's own projection
conditioners, then fuses those per-view conditioning tensors back to the
single-view shapes expected by the existing flow models.  The adapter also
accepts optional per-view quality weights so stronger reference views can carry
more influence during fusion.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import MethodType
from typing import Any


ADAPTER_NAME = "reconviagen-inspired-projection-fusion-v1"
VIEW_ORDER = ("front", "back", "left", "right", "top", "bottom")
CAMERA_DIRS_BLENDER = {
    "front": (0.0, -1.0, 0.0),
    "back": (0.0, 1.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "top": (0.0, 0.0, 1.0),
    "bottom": (0.0, 0.0, -1.0),
}
FUSION_DIRS_MODEL = {
    "front": (0.0, 0.0, 1.0),
    "back": (0.0, 0.0, -1.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "top": (0.0, 1.0, 0.0),
    "bottom": (0.0, -1.0, 0.0),
}
FUSION_STRATEGIES = ("directional_softmax", "average")


def validate_multiview_request(request: dict[str, Any]) -> list[dict[str, Any]]:
    raw_views = request.get("views")
    if raw_views is None:
        return []
    if not isinstance(raw_views, list):
        raise ValueError("Pixal3D multiview views must be an ordered list")
    if not 2 <= len(raw_views) <= len(VIEW_ORDER):
        raise ValueError("Pixal3D multiview requires 2 to 6 ordered views")

    seen: set[str] = set()
    views: list[dict[str, Any]] = []
    expected_prefix = VIEW_ORDER[: len(raw_views)]
    for index, item in enumerate(raw_views):
        if not isinstance(item, dict):
            raise ValueError("Pixal3D multiview entries must be objects")
        name = str(item.get("name", ""))
        if name not in VIEW_ORDER:
            raise ValueError(f"Unsupported Pixal3D multiview label: {name}")
        if name != expected_prefix[index]:
            raise ValueError(
                "Pixal3D multiview views must be ordered front, back, left, right, top, bottom"
            )
        if name in seen:
            raise ValueError(f"Duplicate Pixal3D multiview label: {name}")
        path = Path(str(item.get("image_path", "")))
        if not path.is_file():
            raise FileNotFoundError(f"Pixal3D multiview input does not exist: {path}")
        quality = float(item.get("quality", 1.0))
        if not math.isfinite(quality) or quality < 0.0 or quality > 10.0:
            raise ValueError("Pixal3D multiview view quality must be in [0, 10]")
        seen.add(name)
        views.append({"name": name, "image_path": str(path), "quality": quality})
    if views[0]["name"] != "front":
        raise ValueError("Pixal3D multiview requires front as the first view")

    strategy = str(request.get("fusion_strategy", "directional_softmax"))
    if strategy not in FUSION_STRATEGIES:
        raise ValueError(f"Pixal3D multiview fusion_strategy must be one of {FUSION_STRATEGIES}")
    temperature = float(request.get("fusion_temperature", 2.0))
    if not math.isfinite(temperature) or temperature <= 0.0 or temperature > 20.0:
        raise ValueError("Pixal3D multiview fusion_temperature must be in (0, 20]")
    return views


def camera_transform_for_view(label: str, distance: float):
    """Return a Blender-style camera-to-world transform for a canonical view label."""

    import torch

    return torch.tensor(camera_transform_matrix(label, distance), dtype=torch.float32)


def camera_transform_matrix(label: str, distance: float) -> list[list[float]]:
    """Pure-Python canonical camera-to-world matrix for tests and metadata."""

    if label not in CAMERA_DIRS_BLENDER:
        raise ValueError(f"Unsupported Pixal3D multiview label: {label}")
    if not math.isfinite(float(distance)) or float(distance) <= 0.0:
        raise ValueError("Pixal3D multiview camera distance must be positive")

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def normalize(v):
        norm = math.sqrt(dot(v, v))
        return tuple(component / norm for component in v)

    direction = CAMERA_DIRS_BLENDER[label]
    location = tuple(component * float(distance) for component in direction)
    forward = normalize(tuple(-component for component in location))
    up_hint = (0.0, 0.0, 1.0)
    if abs(dot(forward, up_hint)) > 0.98:
        up_hint = (0.0, 1.0, 0.0)
    right = normalize(cross(forward, up_hint))
    up = cross(right, forward)
    back = tuple(-component for component in forward)
    return [
        [right[0], up[0], back[0], location[0]],
        [right[1], up[1], back[1], location[1]],
        [right[2], up[2], back[2], location[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def directional_softmax_weight_rows(
    labels: list[str] | tuple[str, ...],
    points: list[tuple[float, float, float]],
    *,
    temperature: float = 2.0,
) -> list[list[float]]:
    """Pure-Python directional softmax rows for deterministic CPU tests."""

    if not labels:
        raise ValueError("At least one view label is required")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("fusion temperature must be positive")
    rows: list[list[float]] = []
    for point in points:
        scores = [
            sum(point[axis] * FUSION_DIRS_MODEL[label][axis] for axis in range(3))
            * float(temperature)
            for label in labels
        ]
        peak = max(scores)
        exps = [math.exp(score - peak) for score in scores]
        total = sum(exps)
        rows.append([value / total for value in exps])
    return rows


def directional_softmax_weights(
    labels: list[str] | tuple[str, ...],
    *,
    temperature: float = 2.0,
    coords=None,
    resolution: int | None = None,
    grid_resolution: int | None = None,
):
    """Compute deterministic directional softmax weights over labeled views."""

    import torch

    if not labels:
        raise ValueError("At least one view label is required")
    for label in labels:
        if label not in FUSION_DIRS_MODEL:
            raise ValueError(f"Unsupported Pixal3D multiview label: {label}")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("fusion temperature must be positive")

    if coords is not None:
        if resolution is None or int(resolution) <= 1:
            raise ValueError("resolution is required for sparse directional weights")
        device = coords.device
        points = torch.stack(
            (
                (coords[:, 1].float() / (int(resolution) - 1)) * 2.0 - 1.0,
                (coords[:, 2].float() / (int(resolution) - 1)) * 2.0 - 1.0,
                (coords[:, 3].float() / (int(resolution) - 1)) * 2.0 - 1.0,
            ),
            dim=1,
        )
    else:
        if grid_resolution is None or int(grid_resolution) <= 1:
            raise ValueError("grid_resolution is required for dense directional weights")
        device = "cpu"
        axis = torch.linspace(-1.0, 1.0, int(grid_resolution), device=device)
        x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
        points = torch.stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)), dim=1)

    dirs = torch.tensor(
        [FUSION_DIRS_MODEL[label] for label in labels], device=points.device
    )
    scores = points @ dirs.T
    return torch.softmax(scores * float(temperature), dim=1)


def _patch_projection_grid(proj_grid) -> None:
    if getattr(proj_grid, "_comfycolab_multiview_transform_patch", False):
        return

    def forward(
        self,
        features_map,
        camera_angle_x,
        distance,
        mesh_scale,
        transform_matrix=None,
        BHWC=True,
    ):
        module_globals = self.__class__.forward.__globals__
        project_points_to_image_batch = module_globals["project_points_to_image_batch"]
        sample_features = module_globals["sample_features"]
        torch = module_globals["torch"]

        if BHWC:
            batch_size = features_map.shape[0]
        else:
            batch_size = features_map.shape[0]
        grid_points = self.grid_points.expand(batch_size, -1, -1)
        grid_points = grid_points / mesh_scale.unsqueeze(-1).unsqueeze(-1) / 2
        if transform_matrix is None:
            transform_matrix = self.front_view_transform_matrix
            transform_matrix = transform_matrix.expand(batch_size, -1, -1).clone()
            transform_matrix[:, 1, 3] = -distance
        image_points, _depth, _valid_mask = project_points_to_image_batch(
            grid_points, transform_matrix, camera_angle_x, self.image_resolution
        )
        image_points_norm = (image_points + 0.5) / self.image_resolution * 2 - 1
        if BHWC:
            features_map = features_map.permute(0, 3, 1, 2)
        sampled = sample_features(features_map, image_points_norm)
        return sampled.permute(0, 2, 1)

    proj_grid.forward = MethodType(forward, proj_grid)
    proj_grid._comfycolab_multiview_transform_patch = True


def patch_pipeline_projection_grids(pipeline) -> None:
    for attr in (
        "image_cond_model_ss",
        "image_cond_model_shape_512",
        "image_cond_model_shape_1024",
        "image_cond_model_tex_1024",
    ):
        model = getattr(pipeline, attr, None)
        grid = getattr(model, "proj_grid", None)
        if grid is not None:
            _patch_projection_grid(grid)


def _view_tensors(pipeline, labels: list[str], camera_params: dict[str, float]):
    import torch

    device = pipeline.device
    camera_angle_x = float(camera_params["camera_angle_x"])
    distance = float(camera_params["distance"])
    mesh_scale = float(camera_params.get("mesh_scale", 1.0))
    return {
        "camera_angle_x": torch.full((len(labels),), camera_angle_x, device=device),
        "distance": torch.full((len(labels),), distance, device=device),
        "mesh_scale": torch.full((len(labels),), mesh_scale, device=device),
        "transform_matrix": torch.stack(
            [camera_transform_for_view(label, distance) for label in labels]
        ).to(device),
    }


def _fuse_dense_projection(
    values,
    labels: list[str],
    strategy: str,
    temperature: float,
    *,
    qualities: list[float] | None = None,
):
    import torch

    if values.shape[0] == 1:
        return values
    if qualities is None:
        qualities = [1.0] * len(labels)
    quality_weights = torch.tensor(qualities, device=values.device, dtype=values.dtype)
    if float(quality_weights.sum()) <= 0.0:
        quality_weights = torch.ones_like(quality_weights)
    quality_weights = quality_weights / quality_weights.sum()
    if strategy == "average":
        return (values * quality_weights[:, None, None, None]).sum(dim=0, keepdim=True)
    weights = directional_softmax_weights(
        labels,
        temperature=temperature,
        grid_resolution=round(values.shape[1] ** (1.0 / 3.0)),
    ).to(device=values.device, dtype=values.dtype)
    weighted = weights * quality_weights[None, :]
    weighted = weighted / weighted.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return (values * weighted.T[:, :, None]).sum(dim=0, keepdim=True)


def _fuse_sparse_projection(
    values,
    labels: list[str],
    strategy: str,
    temperature: float,
    coords,
    resolution: int,
    *,
    qualities: list[float] | None = None,
):
    import torch

    per_view = values.reshape(len(labels), coords.shape[0], -1)
    if qualities is None:
        qualities = [1.0] * len(labels)
    quality_weights = torch.tensor(qualities, device=values.device, dtype=values.dtype)
    if float(quality_weights.sum()) <= 0.0:
        quality_weights = torch.ones_like(quality_weights)
    quality_weights = quality_weights / quality_weights.sum()
    if strategy == "average":
        return (per_view * quality_weights[:, None, None]).sum(dim=0)
    weights = directional_softmax_weights(
        labels, temperature=temperature, coords=coords, resolution=resolution
    ).to(device=values.device, dtype=values.dtype)
    weighted = weights * quality_weights[None, :]
    weighted = weighted / weighted.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return (per_view * weighted.T[:, :, None]).sum(dim=0)


def _fuse_cond(
    cond: dict[str, Any],
    labels: list[str],
    strategy: str,
    temperature: float,
    *,
    qualities: list[float] | None = None,
    coords=None,
    resolution: int | None = None,
):
    import torch

    fused: dict[str, Any] = {}
    for key, value in cond.items():
        if key == "global" and torch.is_tensor(value):
            if qualities is None:
                fused[key] = value.mean(dim=0, keepdim=True)
            else:
                quality_weights = torch.tensor(qualities, device=value.device, dtype=value.dtype)
                if float(quality_weights.sum()) <= 0.0:
                    quality_weights = torch.ones_like(quality_weights)
                quality_weights = quality_weights / quality_weights.sum()
                fused[key] = (value * quality_weights[:, None, None]).sum(dim=0, keepdim=True)
        elif key.startswith("proj") and torch.is_tensor(value):
            if coords is None:
                fused[key] = _fuse_dense_projection(
                    value,
                    labels,
                    strategy,
                    temperature,
                    qualities=qualities,
                )
            else:
                if resolution is None:
                    raise ValueError("resolution is required for sparse projection fusion")
                fused[key] = _fuse_sparse_projection(
                    value,
                    labels,
                    strategy,
                    temperature,
                    coords,
                    resolution,
                    qualities=qualities,
                )
        else:
            fused[key] = value
    return fused


def _zero_like_cond(cond: dict[str, Any], coords=None, sparse_cls=None) -> dict[str, Any]:
    import torch

    zeros: dict[str, Any] = {}
    for key, value in cond.items():
        if torch.is_tensor(value):
            zeros[key] = torch.zeros_like(value)
        elif hasattr(value, "feats") and coords is not None and sparse_cls is not None:
            zeros[key] = sparse_cls(feats=torch.zeros_like(value.feats), coords=coords)
    return zeros


def _encode_fused_cond(
    pipeline,
    image_cond_model,
    images,
    labels: list[str],
    camera_params: dict[str, float],
    strategy: str,
    temperature: float,
    *,
    qualities: list[float] | None = None,
    coords=None,
    grid_resolution_override: int | None = None,
    sparse_resolution: int | None = None,
) -> dict[str, Any]:
    import torch
    from pixal3d.modules.sparse import SparseTensor

    patch_pipeline_projection_grids(pipeline)
    device = pipeline.device
    if pipeline.low_vram:
        image_cond_model.to(device)
    orig_grid_res = image_cond_model.grid_resolution
    if grid_resolution_override is not None and grid_resolution_override != orig_grid_res:
        image_cond_model.grid_resolution = grid_resolution_override
        image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
            grid_resolution=grid_resolution_override,
            image_resolution=image_cond_model.proj_grid.image_resolution,
        ).to(device)
        _patch_projection_grid(image_cond_model.proj_grid)
    try:
        camera = _view_tensors(pipeline, labels, camera_params)
        outputs = image_cond_model(images, **camera)
        if len(outputs) == 3:
            cond = {
                "global": outputs[0],
                "proj_semantic": outputs[1],
                "proj_color": outputs[2],
            }
        else:
            cond = {"global": outputs[0], "proj": outputs[1]}
        if coords is not None:
            grid_res = image_cond_model.grid_resolution
            x_coords = coords[:, 1].long()
            y_coords = coords[:, 2].long()
            z_coords = coords[:, 3].long()
            for key, value in list(cond.items()):
                if key.startswith("proj") and torch.is_tensor(value):
                    grid = value.reshape(len(labels), grid_res, grid_res, grid_res, -1)
                    gathered = []
                    for view_index in range(len(labels)):
                        gathered.append(
                            grid[view_index, x_coords, y_coords, z_coords]
                        )
                    cond[key] = torch.stack(gathered, dim=0)
        fused = _fuse_cond(
            cond,
            labels,
            strategy,
            temperature,
            qualities=qualities,
            coords=coords,
            resolution=sparse_resolution,
        )
        if coords is not None:
            for key in list(fused):
                if key.startswith("proj") and torch.is_tensor(fused[key]):
                    fused[key] = SparseTensor(feats=fused[key], coords=coords)
        return {"cond": fused, "neg_cond": _zero_like_cond(fused, coords=coords, sparse_cls=SparseTensor)}
    finally:
        if grid_resolution_override is not None and grid_resolution_override != orig_grid_res:
            image_cond_model.grid_resolution = orig_grid_res
            image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
                grid_resolution=orig_grid_res,
                image_resolution=image_cond_model.proj_grid.image_resolution,
            ).to(device)
            _patch_projection_grid(image_cond_model.proj_grid)
        if pipeline.low_vram:
            image_cond_model.cpu()


def run_multiview_projection_fusion(
    pipeline,
    images,
    camera_params: dict[str, float],
    *,
    labels: list[str],
    seed: int,
    sparse_structure_sampler_params: dict[str, Any],
    shape_slat_sampler_params: dict[str, Any],
    tex_slat_sampler_params: dict[str, Any],
    pipeline_type: str,
    max_num_tokens: int,
    fusion_strategy: str = "directional_softmax",
    fusion_temperature: float = 2.0,
    view_qualities: dict[str, float] | None = None,
    return_latent: bool = False,
):
    import torch
    from pixal3d.modules.sparse import SparseTensor

    if pipeline_type == "1024_cascade":
        hr_resolution = 1024
    elif pipeline_type == "1536_cascade":
        hr_resolution = 1536
    else:
        raise ValueError(f"Invalid Pixal3D pipeline type: {pipeline_type}")
    validate_multiview_request(
        {
            "views": [{"name": name, "image_path": __file__} for name in labels],
            "fusion_strategy": fusion_strategy,
            "fusion_temperature": fusion_temperature,
        }
    )
    qualities = [
        float(view_qualities.get(label, 1.0)) if view_qualities is not None else 1.0
        for label in labels
    ]

    torch.manual_seed(seed)
    patch_pipeline_projection_grids(pipeline)

    cond_ss = _encode_fused_cond(
        pipeline,
        pipeline.image_cond_model_ss,
        images,
        labels,
        camera_params,
        fusion_strategy,
        fusion_temperature,
        qualities=qualities,
    )
    ss_res = 32
    coords = pipeline.sample_sparse_structure(
        cond_ss, ss_res, 1, sparse_structure_sampler_params
    )
    del cond_ss
    torch.cuda.empty_cache()

    cond_shape_lr = _encode_fused_cond(
        pipeline,
        pipeline.image_cond_model_shape_512,
        images,
        labels,
        camera_params,
        fusion_strategy,
        fusion_temperature,
        qualities=qualities,
        coords=coords,
        sparse_resolution=32,
    )
    lr_slat = pipeline.sample_shape_slat(
        cond_shape_lr,
        pipeline.models["shape_slat_flow_model_512"],
        coords,
        shape_slat_sampler_params,
    )
    del cond_shape_lr
    torch.cuda.empty_cache()

    if pipeline.low_vram:
        pipeline.models["shape_slat_decoder"].to(pipeline.device)
        pipeline.models["shape_slat_decoder"].low_vram = True
    hr_coords = pipeline.models["shape_slat_decoder"].upsample(lr_slat, upsample_times=4)
    if pipeline.low_vram:
        pipeline.models["shape_slat_decoder"].cpu()
        pipeline.models["shape_slat_decoder"].low_vram = False

    lr_resolution = 512
    actual_hr_resolution = hr_resolution
    while True:
        grid_res = actual_hr_resolution // 16
        quant_coords = torch.cat(
            [
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (grid_res - 1)).round().int(),
            ],
            dim=1,
        )
        hr_coords_unique = quant_coords.unique(dim=0)
        if hr_coords_unique.shape[0] < max_num_tokens or actual_hr_resolution == 1024:
            break
        actual_hr_resolution -= 128

    actual_grid_res = actual_hr_resolution // 16
    del lr_slat, hr_coords, quant_coords
    torch.cuda.empty_cache()

    cond_shape_hr = _encode_fused_cond(
        pipeline,
        pipeline.image_cond_model_shape_1024,
        images,
        labels,
        camera_params,
        fusion_strategy,
        fusion_temperature,
        qualities=qualities,
        coords=hr_coords_unique,
        grid_resolution_override=actual_grid_res,
        sparse_resolution=actual_grid_res,
    )
    noise_hr = SparseTensor(
        feats=torch.randn(
            hr_coords_unique.shape[0],
            pipeline.models["shape_slat_flow_model_1024"].in_channels,
        ).to(pipeline.device),
        coords=hr_coords_unique,
    )
    sampler_params_hr = {**pipeline.shape_slat_sampler_params, **shape_slat_sampler_params}
    flow_model_hr = pipeline.models["shape_slat_flow_model_1024"]
    if pipeline.low_vram:
        flow_model_hr.to(pipeline.device)
    hr_slat = pipeline.shape_slat_sampler.sample(
        flow_model_hr,
        noise_hr,
        **cond_shape_hr,
        **sampler_params_hr,
        verbose=True,
        tqdm_desc=f"Sampling HR shape SLat (experimental projection fusion, {actual_hr_resolution})",
    ).samples
    if pipeline.low_vram:
        flow_model_hr.cpu()
    std = torch.tensor(pipeline.shape_slat_normalization["std"])[None].to(hr_slat.device)
    mean = torch.tensor(pipeline.shape_slat_normalization["mean"])[None].to(hr_slat.device)
    shape_slat = hr_slat * std + mean
    del cond_shape_hr, noise_hr, hr_slat, hr_coords_unique
    torch.cuda.empty_cache()

    tex_grid_res = actual_hr_resolution // 16
    cond_tex = _encode_fused_cond(
        pipeline,
        pipeline.image_cond_model_tex_1024,
        images,
        labels,
        camera_params,
        fusion_strategy,
        fusion_temperature,
        qualities=qualities,
        coords=shape_slat.coords,
        grid_resolution_override=tex_grid_res,
        sparse_resolution=tex_grid_res,
    )
    tex_slat = pipeline.sample_tex_slat(
        cond_tex,
        pipeline.models["tex_slat_flow_model_1024"],
        shape_slat,
        tex_slat_sampler_params,
    )
    del cond_tex
    torch.cuda.empty_cache()

    out_mesh = pipeline.decode_latent(shape_slat, tex_slat, actual_hr_resolution)
    if return_latent:
        return out_mesh, (shape_slat, tex_slat, actual_hr_resolution)
    return out_mesh
