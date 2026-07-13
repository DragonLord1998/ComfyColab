#!/usr/bin/env python3
"""Run pinned UltraShape inference in a process-isolated file boundary."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from transform_contract import TRANSFORM_SCHEMA, normalization_from_bounds  # noqa: E402
from seed_contract import make_numpy_rng  # noqa: E402


PROGRESS_PREFIX = "COMFYCOLAB_PROGRESS="
RESULT_PREFIX = "COMFYCOLAB_RESULT="
ULTRASHAPE_CHECKPOINT_SHA256 = (
    "c96ae010c4169597fd0006dcb08056bf6104a1fca249b10fed7ddded324c3f0f"
)


def emit_progress(stage: str, current: int, total: int, **details: Any) -> None:
    payload = {"stage": stage, "current": current, "total": total, **details}
    print(PROGRESS_PREFIX + json.dumps(payload, sort_keys=True), flush=True)


def emit_result(**details: Any) -> None:
    print(RESULT_PREFIX + json.dumps(details, sort_keys=True), flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs(args: argparse.Namespace) -> None:
    required_files = {
        "UltraShape source": args.source_dir,
        "UltraShape checkpoint": args.checkpoint,
        "DINOv2 model": args.dinov2_dir,
        "input mesh": args.input_mesh,
        "reference image": args.reference_image,
    }
    for label, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not (args.source_dir / "configs" / "infer_dit_refine.yaml").is_file():
        raise FileNotFoundError("Pinned UltraShape inference config is missing.")
    if args.checkpoint_sha256 and sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise RuntimeError(
            "UltraShape checkpoint checksum mismatch. Delete the partial model and retry the download."
        )


def _load_flattened_mesh(path: Path):
    import trimesh

    loaded = trimesh.load(str(path), force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
        scene_metadata: list[dict[str, object]] = []
    else:
        scene_metadata = []
        meshes = []
        for node_name in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph.get(node_name)
            geometry = loaded.geometry[geometry_name].copy()
            geometry.apply_transform(transform)
            meshes.append(geometry)
            scene_metadata.append(
                {
                    "node": str(node_name),
                    "geometry": str(geometry_name),
                    "matrix": transform.tolist(),
                }
            )
        if not meshes:
            raise ValueError("Input GLB contains no mesh geometry.")
        mesh = trimesh.util.concatenate(meshes)

    vertices = mesh.vertices
    faces = mesh.faces
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Input GLB must contain vertices and triangle faces.")
    if not all(math.isfinite(float(value)) for row in vertices for value in row):
        raise ValueError("Input GLB contains non-finite vertices.")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError("Input GLB contains invalid face indices.")
    return mesh, scene_metadata


def _validate_output_mesh(path: Path) -> dict[str, object]:
    mesh, _ = _load_flattened_mesh(path)
    if path.stat().st_size < 20:
        raise ValueError("UltraShape output GLB is unexpectedly small.")
    return {
        "bytes": path.stat().st_size,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds": mesh.bounds.tolist(),
    }


def _configure_ultrashape(args: argparse.Namespace):
    source = str(args.source_dir)
    if source not in sys.path:
        sys.path.insert(0, source)

    import torch
    from omegaconf import OmegaConf
    from ultrashape.pipelines import UltraShapePipeline
    from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
    from ultrashape.utils import voxelize_from_point
    from ultrashape.utils.misc import instantiate_from_config

    if not torch.cuda.is_available():
        raise RuntimeError("UltraShape requires a CUDA runtime.")
    torch.cuda.reset_peak_memory_stats()

    config = OmegaConf.load(args.source_dir / "configs" / "infer_dit_refine.yaml")
    encoder = config.model.params.conditioner_config.params.main_image_encoder
    encoder.kwargs.version = str(args.dinov2_dir)

    emit_progress("load_models", 1, 6, component="vae")
    vae = instantiate_from_config(config.model.params.vae_config)
    emit_progress("load_models", 2, 6, component="dit")
    dit = instantiate_from_config(config.model.params.dit_cfg)
    emit_progress("load_models", 3, 6, component="conditioner")
    conditioner = instantiate_from_config(config.model.params.conditioner_config)
    emit_progress("load_models", 4, 6, component="scheduler")
    scheduler = instantiate_from_config(config.model.params.scheduler_cfg)
    image_processor = instantiate_from_config(config.model.params.image_processor_cfg)
    emit_progress("load_models", 5, 6, component="checkpoint")
    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vae.load_state_dict(weights["vae"], strict=True)
    dit.load_state_dict(weights["dit"], strict=True)
    conditioner.load_state_dict(weights["conditioner"], strict=True)

    device = torch.device("cuda")
    vae.eval().to(device)
    dit.eval().to(device)
    conditioner.eval().to(device)
    if hasattr(vae, "enable_flashvdm_decoder"):
        vae.enable_flashvdm_decoder()
    pipeline = UltraShapePipeline(
        vae=vae,
        model=dit,
        scheduler=scheduler,
        conditioner=conditioner,
        image_processor=image_processor,
    )
    low_vram = args.low_vram == "on"
    if args.low_vram == "auto":
        free_bytes, _ = torch.cuda.mem_get_info()
        low_vram = free_bytes < 48 * 1024**3
    if low_vram:
        pipeline.enable_model_cpu_offload()
    emit_progress("load_models", 6, 6, low_vram=low_vram)
    loader = SharpEdgeSurfaceLoader(
        num_sharp_points=204800,
        num_uniform_points=204800,
    )
    return torch, config, pipeline, loader, voxelize_from_point, low_vram


def run(args: argparse.Namespace) -> dict[str, object]:
    started_at = time.monotonic()
    validate_inputs(args)
    emit_progress("prepare", 0, 1)

    input_mesh, scene_transforms = _load_flattened_mesh(args.input_mesh)
    transform = normalization_from_bounds(
        input_mesh.bounds[0], input_mesh.bounds[1], normalize_scale=args.normalize_scale
    )
    output_parent = args.output_mesh.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="comfycolab-ultrashape-") as directory:
        temp_root = Path(directory)
        canonical_input = temp_root / "canonical-input.glb"
        input_mesh.export(str(canonical_input), file_type="glb")
        emit_progress("prepare", 1, 1, vertices=len(input_mesh.vertices), faces=len(input_mesh.faces))

        torch, config, pipeline, loader, voxelize_from_point, low_vram = _configure_ultrashape(args)
        from PIL import Image

        emit_progress("preprocess", 0, 2)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        image = Image.open(args.reference_image).convert("RGBA")
        sampling_rng = make_numpy_rng(args.seed)
        surface = loader(
            str(canonical_input), normalize_scale=args.normalize_scale, rng=sampling_rng
        ).to("cuda", dtype=torch.float16)
        emit_progress("preprocess", 1, 2)
        voxel_resolution = config.model.params.vae_config.params.voxel_query_res
        _, voxel_indices = voxelize_from_point(
            surface[:, :, :3], args.num_latents, resolution=voxel_resolution
        )
        emit_progress("preprocess", 2, 2, num_latents=args.num_latents)

        generator_device = "cpu" if low_vram else "cuda"
        generator = torch.Generator(generator_device).manual_seed(args.seed)
        emit_progress("diffusion", 0, args.steps)

        def diffusion_progress(step_index, _timestep, _outputs):
            emit_progress("diffusion", min(args.steps, int(step_index) + 1), args.steps)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            meshes, _ = pipeline(
                image=image,
                voxel_cond=voxel_indices,
                generator=generator,
                box_v=1.0,
                mc_level=0.0,
                octree_resolution=args.octree_resolution,
                num_inference_steps=args.steps,
                num_chunks=args.decode_chunk_size,
                callback=diffusion_progress,
                callback_steps=1,
            )
        emit_progress("diffusion", args.steps, args.steps)
        if not meshes or meshes[0] is None:
            raise RuntimeError("UltraShape returned no refined mesh.")

        refined = meshes[0]
        refined.apply_transform(transform["inverse"])
        import numpy as np
        import trimesh

        refined.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(refined.vertices), 2), dtype=np.float32),
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[0.72, 0.72, 0.72, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.8,
            )
        )
        partial_output = args.output_mesh.with_name(
            f"{args.output_mesh.stem}.partial{args.output_mesh.suffix}"
        )
        partial_metadata = args.metadata_output.with_suffix(args.metadata_output.suffix + ".partial")
        partial_output.unlink(missing_ok=True)
        partial_metadata.unlink(missing_ok=True)
        refined.export(str(partial_output), file_type="glb")
        validation = _validate_output_mesh(partial_output)
        peak_vram = int(torch.cuda.max_memory_allocated())
        metadata = {
            "schema": TRANSFORM_SCHEMA,
            "source_mesh": str(args.input_mesh),
            "source_scene_transforms": scene_transforms,
            "ultrashape_normalization": transform,
            "output_space": "gltf-y-up-restored-world",
            "geometry_only": True,
            "settings": {
                "steps": args.steps,
                "num_latents": args.num_latents,
                "octree_resolution": args.octree_resolution,
                "decode_chunk_size": args.decode_chunk_size,
                "seed": args.seed,
                "low_vram": low_vram,
            },
            "validation": validation,
            "peak_vram_bytes": peak_vram,
            "runtime_seconds": time.monotonic() - started_at,
        }
        partial_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(partial_output, args.output_mesh)
        os.replace(partial_metadata, args.metadata_output)
        emit_progress("complete", 1, 1, **validation)
        return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", default=ULTRASHAPE_CHECKPOINT_SHA256)
    parser.add_argument("--dinov2-dir", type=Path, required=True)
    parser.add_argument("--input-mesh", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--output-mesh", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--num-latents", type=int, required=True)
    parser.add_argument("--octree-resolution", type=int, required=True)
    parser.add_argument("--decode-chunk-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--low-vram", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--normalize-scale", type=float, default=0.99)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for output in (args.output_mesh, args.metadata_output):
        output.unlink(missing_ok=True)
        if output.suffix.lower() == ".glb":
            output.with_name(f"{output.stem}.partial{output.suffix}").unlink(missing_ok=True)
        else:
            output.with_suffix(output.suffix + ".partial").unlink(missing_ok=True)
    try:
        result = run(args)
    except BaseException as error:
        for output in (args.output_mesh, args.metadata_output):
            output.unlink(missing_ok=True)
            if output.suffix.lower() == ".glb":
                output.with_name(f"{output.stem}.partial{output.suffix}").unlink(missing_ok=True)
            else:
                output.with_suffix(output.suffix + ".partial").unlink(missing_ok=True)
        emit_result(status="error", error=str(error), error_type=type(error).__name__)
        traceback.print_exc()
        return 1
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    emit_result(
        status="ok",
        output_mesh=str(args.output_mesh),
        metadata_output=str(args.metadata_output),
        runtime_seconds=result["runtime_seconds"],
        peak_vram_bytes=result["peak_vram_bytes"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
