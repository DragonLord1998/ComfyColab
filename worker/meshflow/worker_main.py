from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import time
from pathlib import Path


PROGRESS_PREFIX = "COMFYCOLAB_PROGRESS="
RESULT_PREFIX = "COMFYCOLAB_RESULT="
RESULT_SCHEMA = "comfycolab-meshflow-result-v2"


def _dinov3_prenorm_tokens(outputs):
    hidden_states = getattr(outputs, "hidden_states", None)
    if not hidden_states:
        raise RuntimeError(
            "Transformers DINOv3 did not return pre-final-normalization hidden states"
        )
    return hidden_states[-1]


def _meshflow_num_verts_supported(config_path: Path) -> bool:
    text = config_path.read_text(encoding="utf-8", errors="ignore")
    return bool(
        re.search(
            r"(?im)^\s*use_proj_cond_on_temb\s*:\s*true\s*(?:#.*)?$",
            text,
        )
    )


def _install_transformers_dinov3_adapter(
    source_dir: Path,
    model_dir: Path,
) -> None:
    """Use the pinned Transformers DINOv3 snapshot for MeshFlow conditioning."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoModel

    sys.path.insert(0, str(source_dir))
    import meshflow.pipelines.meshflow_pipeline as pipeline_module

    class TransformersDINOv3Encoder(nn.Module):
        def __init__(self, image_size: int):
            super().__init__()
            self.dino_model = AutoModel.from_pretrained(
                str(model_dir),
                local_files_only=True,
            )
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        image_size,
                        transforms.InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    transforms.CenterCrop(image_size),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            self.dino_model.eval().requires_grad_(False)

        @property
        def dtype(self):
            return next(self.parameters()).dtype

        def encode_image(self, image):
            if isinstance(image, torch.Tensor):
                values = image.permute(0, 3, 1, 2)
            else:
                if not isinstance(image, Image.Image):
                    raise TypeError(
                        f"Unsupported DINOv3 image type: {type(image).__name__}"
                    )
                values = transforms.ToTensor()(image).unsqueeze(0)
            values = self.transform(values)
            reference = next(self.dino_model.parameters())
            values = values.to(device=reference.device, dtype=reference.dtype)
            with torch.autocast(
                device_type=reference.device.type,
                enabled=False,
            ):
                outputs = self.dino_model(
                    pixel_values=values,
                    output_hidden_states=True,
                )
                output = _dinov3_prenorm_tokens(outputs)
            return functional.layer_norm(output, output.shape[-1:])

    def build_transformers_encoder(_encoder_type: str, config: dict):
        return TransformersDINOv3Encoder(int(config["image_size"]))

    pipeline_module.build_dinov3_encoder = build_transformers_encoder


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_glb_vertex_color_material(path: Path) -> None:
    payload = path.read_bytes()
    if len(payload) < 20:
        raise ValueError(f"MeshFlow GLB is truncated: {path}")
    magic, version, declared_length = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(payload):
        raise ValueError(f"MeshFlow GLB header is invalid: {path}")

    chunks: list[tuple[bytes, bytes]] = []
    document = None
    offset = 12
    while offset < len(payload):
        chunk_length, chunk_type = struct.unpack_from("<I4s", payload, offset)
        offset += 8
        chunk = payload[offset : offset + chunk_length]
        offset += chunk_length
        if len(chunk) != chunk_length:
            raise ValueError(f"MeshFlow GLB chunk is truncated: {path}")
        if chunk_type == b"JSON":
            document = json.loads(
                chunk.decode("utf-8").rstrip(" \t\r\n\x00")
            )
        else:
            chunks.append((chunk_type, chunk))
    if not isinstance(document, dict):
        raise ValueError(f"MeshFlow GLB has no JSON document: {path}")

    materials = document.setdefault("materials", [])
    if not materials:
        materials.append(
            {
                "name": "MeshFlow Vertex Colors",
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.85,
                },
            }
        )
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if "COLOR_0" in primitive.get("attributes", {}):
                primitive.setdefault("material", 0)

    json_chunk = json.dumps(
        document,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    json_chunk += b" " * ((4 - len(json_chunk) % 4) % 4)
    body = struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk
    for chunk_type, chunk in chunks:
        padded = chunk + b"\x00" * ((4 - len(chunk) % 4) % 4)
        body += struct.pack("<I4s", len(padded), chunk_type) + padded
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)


def _export_host_glb(host_mesh, path: Path) -> None:
    _ = host_mesh.vertex_normals
    host_mesh.export(str(path), include_normals=True)
    _ensure_glb_vertex_color_material(path)


def _emit_progress(stage: str, current: int, total: int) -> None:
    print(
        PROGRESS_PREFIX
        + json.dumps({"stage": stage, "current": current, "total": total}),
        flush=True,
    )


def _as_mesh(loaded):
    if hasattr(loaded, "geometry"):
        meshes = [
            geometry
            for geometry in loaded.geometry.values()
            if hasattr(geometry, "vertices") and len(geometry.vertices)
        ]
        if not meshes:
            return None
        import trimesh

        return trimesh.util.concatenate(meshes)
    return loaded


def _normalized_vertices(vertices):
    import numpy

    values = numpy.asarray(vertices, dtype=numpy.float32)
    if values.size == 0:
        return values
    center = (values.min(axis=0) + values.max(axis=0)) * 0.5
    shifted = values - center
    scale = float(numpy.linalg.norm(shifted, axis=1).max())
    if scale <= 0:
        return shifted
    return shifted / scale


def _nearest_vertex_colors(source_vertices, source_colors, target_vertices):
    import numpy

    source = _normalized_vertices(source_vertices)
    target = _normalized_vertices(target_vertices)
    colors = numpy.asarray(source_colors, dtype=numpy.uint8)
    if colors.shape[1] == 3:
        colors = numpy.concatenate(
            (colors, numpy.full((colors.shape[0], 1), 255, dtype=numpy.uint8)),
            axis=1,
        )
    transferred = numpy.empty((target.shape[0], 4), dtype=numpy.uint8)
    chunk = 128
    for start in range(0, target.shape[0], chunk):
        stop = min(start + chunk, target.shape[0])
        delta = target[start:stop, None, :] - source[None, :, :]
        indices = numpy.argmin(numpy.sum(delta * delta, axis=2), axis=1)
        transferred[start:stop] = colors[indices, :4]
    return transferred


def _source_vertex_colors(path: Path):
    import numpy
    import trimesh

    loaded = trimesh.load(str(path), force="scene", process=False)
    source = _as_mesh(loaded)
    if source is None or len(source.vertices) == 0:
        return None
    visual = getattr(source, "visual", None)
    if visual is None:
        return None
    colors = None
    try:
        colors = visual.to_color().vertex_colors
    except BaseException:
        colors = getattr(visual, "vertex_colors", None)
    if colors is None or len(colors) != len(source.vertices):
        return None
    colors = numpy.asarray(colors, dtype=numpy.uint8)
    if colors.size == 0:
        return None
    return numpy.asarray(source.vertices, dtype=numpy.float32), colors


def _meshflow_to_trimesh(mesh, appearance_mesh: Path | None):
    import numpy
    import trimesh

    raw = mesh.to_trimesh()
    vertices = numpy.asarray(raw.vertices, dtype=numpy.float32)
    faces = numpy.asarray(raw.faces, dtype=numpy.int64)
    normals = getattr(mesh, "v_nrm", None)
    if normals is not None:
        if hasattr(normals, "detach"):
            normals = normals.detach().cpu().numpy()
        else:
            normals = numpy.asarray(normals)
        if normals.shape != vertices.shape:
            normals = None
    transferred = None
    source_colors = (
        _source_vertex_colors(appearance_mesh)
        if appearance_mesh is not None
        else None
    )
    appearance_method = "none"
    if source_colors is not None:
        transferred = _nearest_vertex_colors(source_colors[0], source_colors[1], vertices)
        appearance_method = "nearest_source_vertex_color"
    host = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        process=False,
    )
    if transferred is not None:
        host.visual = trimesh.visual.ColorVisuals(host, vertex_colors=transferred)
    else:
        host.visual = trimesh.visual.ColorVisuals(
            host,
            vertex_colors=numpy.tile(
                numpy.array([[190, 190, 190, 255]], dtype=numpy.uint8),
                (len(vertices), 1),
            ),
        )
        appearance_method = "fallback_neutral_vertex_color"
    # Force smooth normals to be computed before export when the upstream mesh
    # did not provide them. Trimesh may choose the final GLB accessor layout,
    # but this prevents flat position-only host meshes.
    _ = host.vertex_normals
    return host, appearance_method


def _load_geometry_points(path: Path, *, sample_count: int, seed: int):
    import numpy
    import trimesh

    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.PointCloud):
        points = numpy.asarray(loaded.vertices, dtype=numpy.float32)
    else:
        mesh = _as_mesh(loaded)
        if mesh is None or len(mesh.vertices) == 0:
            raise ValueError(f"MeshFlow input geometry is empty: {path}")
        if len(mesh.faces) == 0:
            points = numpy.asarray(mesh.vertices, dtype=numpy.float32)
        else:
            points, _ = trimesh.sample.sample_surface(
                mesh,
                int(sample_count),
                seed=int(seed),
            )
            points = numpy.asarray(points, dtype=numpy.float32)
    if len(points) < int(sample_count):
        raise ValueError(
            f"MeshFlow point cloud must contain at least {sample_count} points; "
            f"got {len(points)}"
        )
    if len(points) > int(sample_count):
        indices = numpy.random.default_rng(int(seed)).choice(
            len(points),
            int(sample_count),
            replace=False,
        )
        points = points[indices]
    return _normalized_vertices(points)


def _sanitize_candidate_mesh(host_mesh):
    """Remove tiny disconnected debris without hiding a globally broken mesh."""
    import trimesh

    try:
        components = sorted(
            host_mesh.split(only_watertight=False),
            key=lambda item: float(item.area),
            reverse=True,
        )
    except BaseException:
        components = []
    if len(components) <= 1:
        return host_mesh, {
            "cleanup_applied": False,
            "raw_component_count": max(1, len(components)),
            "dominant_area_fraction": 1.0,
            "dominant_face_fraction": 1.0,
        }

    total_area = sum(max(0.0, float(item.area)) for item in components)
    total_faces = sum(int(len(item.faces)) for item in components)
    dominant = components[0]
    area_fraction = float(
        max(0.0, float(dominant.area)) / max(total_area, 1.0e-12)
    )
    face_fraction = float(
        int(len(dominant.faces)) / max(1, total_faces)
    )
    diagnostics = {
        "cleanup_applied": False,
        "raw_component_count": len(components),
        "dominant_area_fraction": area_fraction,
        "dominant_face_fraction": face_fraction,
    }
    if area_fraction < 0.90 or face_fraction < 0.85:
        return host_mesh, diagnostics

    cleaned = dominant.copy()
    cleaned.process(validate=True)
    cleaned.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(cleaned, multibody=True)
    trimesh.repair.fill_holes(cleaned)
    diagnostics["cleanup_applied"] = True
    diagnostics["cleaned_vertices"] = int(len(cleaned.vertices))
    diagnostics["cleaned_faces"] = int(len(cleaned.faces))
    return cleaned, diagnostics


def _nearest_mean_squared(first, second) -> float:
    import numpy

    first = numpy.asarray(first, dtype=numpy.float32)
    second = numpy.asarray(second, dtype=numpy.float32)
    total = 0.0
    count = 0
    for start in range(0, len(first), 128):
        block = first[start : start + 128]
        delta = block[:, None, :] - second[None, :, :]
        minimum = numpy.min(numpy.sum(delta * delta, axis=2), axis=1)
        total += float(minimum.sum())
        count += int(len(minimum))
    return total / max(1, count)


def _candidate_metrics(host_mesh, source_points, *, seed: int) -> dict:
    import numpy
    import trimesh

    vertices = numpy.asarray(host_mesh.vertices, dtype=numpy.float32)
    faces = numpy.asarray(host_mesh.faces, dtype=numpy.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("MeshFlow candidate has empty geometry")

    edges = numpy.sort(
        numpy.concatenate(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
            axis=0,
        ),
        axis=1,
    )
    _unique_edges, edge_counts = numpy.unique(edges, axis=0, return_counts=True)
    boundary_edges = int(numpy.count_nonzero(edge_counts == 1))
    nonmanifold_edges = int(numpy.count_nonzero(edge_counts > 2))
    unique_edge_count = int(len(edge_counts))

    sorted_faces = numpy.sort(faces, axis=1)
    duplicate_faces = int(
        len(sorted_faces) - len(numpy.unique(sorted_faces, axis=0))
    )
    triangles = vertices[faces]
    double_area = numpy.linalg.norm(
        numpy.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    degenerate_faces = int(numpy.count_nonzero(double_area <= 1e-10))
    components = int(len(host_mesh.split(only_watertight=False)))

    sample_count = min(4096, max(1024, len(vertices)))
    candidate_points, _ = trimesh.sample.sample_surface(
        host_mesh,
        sample_count,
        seed=int(seed),
    )
    candidate_points = _normalized_vertices(candidate_points)
    source_subset = source_points
    if len(source_subset) > sample_count:
        indices = numpy.random.default_rng(int(seed)).choice(
            len(source_subset),
            sample_count,
            replace=False,
        )
        source_subset = source_subset[indices]
    chamfer = 0.5 * (
        _nearest_mean_squared(candidate_points, source_subset)
        + _nearest_mean_squared(source_subset, candidate_points)
    )
    face_vertex_ratio = float(len(faces) / max(1, len(vertices)))
    nonmanifold_ratio = float(nonmanifold_edges / max(1, unique_edge_count))
    boundary_ratio = float(boundary_edges / max(1, unique_edge_count))
    accepted = bool(
        len(vertices) >= 512
        and face_vertex_ratio <= 6.0
        and components <= 8
        and nonmanifold_ratio <= 0.005
        and boundary_ratio <= 0.10
        and duplicate_faces == 0
        and degenerate_faces == 0
    )
    score = float(
        chamfer
        + min(1.0, nonmanifold_ratio) * 100.0
        + min(1.0, boundary_ratio) * 10.0
        + max(0, components - 1) * 0.25
        + max(0.0, face_vertex_ratio - 3.0)
    )
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "face_vertex_ratio": face_vertex_ratio,
        "connected_components": components,
        "unique_edges": unique_edge_count,
        "boundary_edges": boundary_edges,
        "boundary_edge_ratio": boundary_ratio,
        "nonmanifold_edges": nonmanifold_edges,
        "nonmanifold_edge_ratio": nonmanifold_ratio,
        "duplicate_faces": duplicate_faces,
        "degenerate_faces": degenerate_faces,
        "symmetric_chamfer_mse": float(chamfer),
        "accepted": accepted,
        "selection_score": score,
    }


def run(args: argparse.Namespace) -> dict:
    source_dir = Path(args.source_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    input_mesh = Path(args.input_mesh)
    appearance_mesh = (
        Path(args.appearance_mesh)
        if args.appearance_mesh
        else (
            input_mesh
            if input_mesh.suffix.lower() in {".glb", ".gltf", ".obj", ".stl"}
            else None
        )
    )
    reference_images = [Path(value) for value in args.reference_image]
    dinov3_model_dir = (
        Path(args.dinov3_model_dir) if args.dinov3_model_dir else None
    )
    output_mesh = Path(args.output_mesh)
    metadata_output = Path(args.metadata_output)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"MeshFlow source is missing: {source_dir}")
    if not (checkpoint_dir / "config.yaml").is_file():
        raise FileNotFoundError("MeshFlow config.yaml is missing")
    if not (checkpoint_dir / "model.pth").is_file():
        raise FileNotFoundError("MeshFlow model.pth is missing")
    if not input_mesh.is_file():
        raise FileNotFoundError(f"MeshFlow input mesh is missing: {input_mesh}")
    if appearance_mesh is not None and not appearance_mesh.is_file():
        raise FileNotFoundError(
            f"MeshFlow appearance mesh is missing: {appearance_mesh}"
        )
    for reference_image in reference_images:
        if not reference_image.is_file():
            raise FileNotFoundError(
                f"MeshFlow reference image is missing: {reference_image}"
            )
    if reference_images:
        if dinov3_model_dir is None or not dinov3_model_dir.is_dir():
            raise FileNotFoundError(
                "MeshFlow reference conditioning requires a DINOv3 model directory"
            )
        for filename in ("config.json", "model.safetensors"):
            if not (dinov3_model_dir / filename).is_file():
                raise FileNotFoundError(
                    f"MeshFlow DINOv3 bundle is missing {filename}"
                )
    if not 1 <= args.steps <= 100:
        raise ValueError("MeshFlow steps must be between 1 and 100")
    if not 1024 <= args.num_verts <= 16384:
        raise ValueError("MeshFlow num_verts must be between 1024 and 16384")
    if not 0.0 <= float(args.guidance_scale) <= 30.0:
        raise ValueError("MeshFlow guidance_scale must be between 0 and 30")
    num_verts_supported = _meshflow_num_verts_supported(checkpoint_dir / "config.yaml")

    if reference_images and dinov3_model_dir is not None:
        _install_transformers_dinov3_adapter(source_dir, dinov3_model_dir)
    else:
        sys.path.insert(0, str(source_dir))
    _emit_progress("load_pipeline", 0, 1)
    from meshflow.pipelines import MeshFlowPipeline

    started = time.monotonic()
    pipeline = MeshFlowPipeline.from_pretrained(
        str(checkpoint_dir),
        device="cuda",
        dtype=args.dtype,
        compile_models=args.compile,
        num_verts=args.num_verts,
    )
    _emit_progress("load_pipeline", 1, 1)
    candidate_inputs = []
    if reference_images:
        from PIL import Image

        for reference_image in reference_images:
            with Image.open(reference_image) as loaded_image:
                candidate_inputs.append(
                    (
                        reference_image.stem.removeprefix("reference-"),
                        reference_image,
                        loaded_image.convert("RGB").copy(),
                    )
                )
    else:
        candidate_inputs.append(("geometry-only", None, None))

    source_points = _load_geometry_points(
        input_mesh,
        sample_count=4096,
        seed=int(args.seed),
    )
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    _emit_progress(
        "generate_mesh",
        0,
        args.steps * len(candidate_inputs),
    )
    candidates = []
    completed_steps = 0
    for index, (view_name, reference_path, image) in enumerate(candidate_inputs):
        candidate_seed = int(args.seed) + index
        generated = pipeline.run(
            mesh=str(input_mesh),
            image=image,
            steps=args.steps,
            guidance_scale=float(args.guidance_scale),
            seed=candidate_seed,
            disable_prog=True,
            num_verts=args.num_verts,
        )
        host_mesh, appearance_method = _meshflow_to_trimesh(
            generated,
            appearance_mesh,
        )
        host_mesh, cleanup = _sanitize_candidate_mesh(host_mesh)
        metrics = _candidate_metrics(
            host_mesh,
            source_points,
            seed=candidate_seed,
        )
        candidate_path = output_mesh.with_name(
            f"{output_mesh.stem}.candidate-{view_name}{output_mesh.suffix}"
        )
        _export_host_glb(host_mesh, candidate_path)
        candidates.append(
            {
                "view": view_name,
                "reference_image": (
                    str(reference_path) if reference_path is not None else ""
                ),
                "candidate_filename": candidate_path.name,
                "candidate_artifact_persisted": False,
                "appearance_method": appearance_method,
                "candidate_seed": candidate_seed,
                "cleanup": cleanup,
                **metrics,
                "_host_mesh": host_mesh,
            }
        )
        completed_steps += int(args.steps)
        _emit_progress(
            "generate_mesh",
            completed_steps,
            args.steps * len(candidate_inputs),
        )

    accepted_candidates = [
        candidate for candidate in candidates if candidate["accepted"]
    ]
    if not accepted_candidates:
        diagnostics = [
            {
                key: value
                for key, value in candidate.items()
                if key != "_host_mesh"
            }
            for candidate in candidates
        ]
        raise RuntimeError(
            "MeshFlow rejected every reference-conditioned candidate on topology "
            f"and geometry gates: {json.dumps(diagnostics, sort_keys=True)}"
        )
    selected = min(
        accepted_candidates,
        key=lambda candidate: float(candidate["selection_score"]),
    )
    selected_host_mesh = selected["_host_mesh"]
    _export_host_glb(selected_host_mesh, output_mesh)
    if not output_mesh.is_file() or output_mesh.stat().st_size <= 0:
        raise RuntimeError("MeshFlow did not produce a GLB")
    vertices = int(len(selected_host_mesh.vertices))
    faces = int(len(selected_host_mesh.faces))
    if vertices <= 0 or faces <= 0:
        raise RuntimeError("MeshFlow output has no volumetric mesh geometry")
    serializable_candidates = [
        {key: value for key, value in candidate.items() if key != "_host_mesh"}
        for candidate in candidates
    ]

    metadata = {
        "schema": RESULT_SCHEMA,
        "status": "ok",
        "input_mesh": str(input_mesh),
        "input_geometry_type": (
            "surface_point_cloud"
            if input_mesh.suffix.lower() in {".ply", ".pcd", ".xyz", ".pts", ".npy", ".npz"}
            else "mesh_surface_sample"
        ),
        "appearance_mesh": str(appearance_mesh) if appearance_mesh is not None else "",
        "output_mesh": str(output_mesh),
        "output_sha256": _sha256(output_mesh),
        "vertices": vertices,
        "faces": faces,
        "steps": args.steps,
        "requested_num_verts": args.num_verts,
        "actual_vertices": vertices,
        "actual_faces": faces,
        "num_verts_control_supported": num_verts_supported,
        "num_verts_warning": (
            ""
            if num_verts_supported or vertices == int(args.num_verts)
            else "This MeshFlow checkpoint does not enable projected timestep vertex-count conditioning; requested num_verts may be ignored."
        ),
        "reference_images": [str(path) for path in reference_images],
        "dinov3_model_dir": (
            str(dinov3_model_dir) if dinov3_model_dir is not None else ""
        ),
        "dinov3_backend": (
            "transformers_local" if reference_images else "disabled"
        ),
        "image_conditioned": bool(reference_images),
        "conditioning_view_count": len(reference_images),
        "conditioning_fusion": (
            "separate_candidates_best_valid"
            if len(reference_images) > 1
            else ("single_view" if reference_images else "disabled")
        ),
        "selected_view": selected["view"],
        "selected_reference_image": selected["reference_image"],
        "candidate_selection": {
            "policy": "topology_gates_then_symmetric_chamfer_v1",
            "candidate_count": len(serializable_candidates),
            "accepted_count": len(accepted_candidates),
            "candidates": serializable_candidates,
        },
        "guidance_scale": float(args.guidance_scale),
        "appearance_method": selected["appearance_method"],
        "seed": args.seed,
        "dtype": args.dtype,
        "source_ref": args.source_ref,
        "model_ref": args.model_ref,
        "runtime_seconds": time.monotonic() - started,
        "geometry_only": not bool(reference_images),
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata["metadata_output"] = str(metadata_output)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ComfyColab MeshFlow worker")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dinov3-model-dir", default="")
    parser.add_argument("--input-mesh", required=True)
    parser.add_argument("--appearance-mesh", default="")
    parser.add_argument("--reference-image", action="append", default=[])
    parser.add_argument("--output-mesh", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--num-verts", type=int, default=4096)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--model-ref", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run(args)
    except BaseException as error:
        result = {
            "schema": RESULT_SCHEMA,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
        return 1
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
