#!/usr/bin/env python3
"""Persistent process-isolated runner for official CubePart decomposition."""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


READY_PREFIX = "COMFYCOLAB_CUBEPART_READY="
PROGRESS_PREFIX = "COMFYCOLAB_CUBEPART_PROGRESS="
RESULT_PREFIX = "COMFYCOLAB_CUBEPART_RESULT="
PROTOCOL_VERSION = 1
MAX_PARTS = 8
CUBEPART_SOURCE_REF = "3c6d06ddbef3160a1e1950cb13ab63dd12a61e50"
CUBEPART_MODEL_REPO = "Roblox/cubepart"
CUBEPART_MODEL_REF = "28431d124e77040fcaf34c0a71623ff61d35a6c0"
CUBEPART_CODE_LICENSE = "Cube3D Research-Only RAIL-MS"
CUBEPART_WEIGHTS_LICENSE = "OpenRAIL / Cube3D Research-Only RAIL-MS"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
NODE_PACK = REPO_ROOT / "custom_nodes" / "ComfyColab-3D"


def _emit(prefix: str, payload: dict[str, Any]) -> None:
    print(prefix + json.dumps(payload, sort_keys=True), flush=True)


def emit_progress(request_id: str, stage: str, current: int, total: int, **details: Any) -> None:
    _emit(PROGRESS_PREFIX, {"request_id": request_id, "stage": stage, "current": current, "total": total, **details})


def emit_result(**details: Any) -> None:
    _emit(RESULT_PREFIX, details)


def _license_metadata() -> dict[str, str]:
    return {
        "source_ref": CUBEPART_SOURCE_REF,
        "source_license": CUBEPART_CODE_LICENSE,
        "weights_repo": CUBEPART_MODEL_REPO,
        "weights_ref": CUBEPART_MODEL_REF,
        "weights_license": CUBEPART_WEIGHTS_LICENSE,
        "required_acceptance": "accept_research_license",
    }


def _normalize_part_names(part_names: str | Iterable[str]) -> list[str]:
    if isinstance(part_names, str):
        values = [part.strip() for part in re.split(r"[\n,]+", part_names) if part.strip()]
    else:
        values = [str(part).strip() for part in part_names if str(part).strip()]
    if not values:
        raise ValueError("CubePart requires a non-empty ordered part_names schema")
    if len(values) > MAX_PARTS:
        raise ValueError(f"CubePart supports at most {MAX_PARTS} ordered part names")
    return values


def _validate_request(request: dict[str, Any]) -> list[str]:
    if int(request.get("protocol", -1)) != PROTOCOL_VERSION:
        raise ValueError("Unsupported CubePart worker protocol")
    if request.get("accept_research_license") is not True:
        raise PermissionError("CubePart requires accept_research_license=True before running")
    if request.get("license") != _license_metadata():
        raise PermissionError("CubePart request omitted exact code/weights license metadata")
    for name in ("request_id", "input_mesh", "output_dir"):
        if not str(request.get(name, "")):
            raise ValueError(f"CubePart request omitted {name}")
    input_mesh = Path(str(request["input_mesh"]))
    if not input_mesh.is_file():
        raise FileNotFoundError(f"CubePart input GLB does not exist: {input_mesh}")
    parts = _normalize_part_names(request.get("part_names", []))
    if int(request.get("seed", -1)) < 0 or int(request.get("seed", -1)) > (2**31) - 1:
        raise ValueError("CubePart seed must be between 0 and 2147483647")
    if int(request.get("num_inference_steps", 0)) <= 0:
        raise ValueError("CubePart num_inference_steps must be positive")
    if int(request.get("num_samples", 0)) <= 0:
        raise ValueError("CubePart num_samples must be positive")
    if request.get("scheduler") not in {"euler", "heun", "dpm_solver"}:
        raise ValueError("CubePart scheduler must be euler, heun, or dpm_solver")
    return parts


def _load_comfycolab_contract():
    package_name = "comfycolab_cubepart_contract"
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


def _load_official_pipeline(source_dir: Path) -> ModuleType:
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    return importlib.import_module("cube_part.pipelines")


def _load_official_mesh_utils(source_dir: Path) -> ModuleType:
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    return importlib.import_module("cube_part.utils.mesh")


def _git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Pinned CubePart source checkout is invalid: {path}") from error


def _snapshot_revision(path: Path) -> str:
    try:
        marker = json.loads((path / ".comfycolab-artifact.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return CUBEPART_MODEL_REF
    return str(marker.get("revision") or CUBEPART_MODEL_REF)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._")
    return value or "part"


def _palette(n: int):
    numpy = importlib.import_module("numpy")
    colors = []
    for index in range(max(n, 1)):
        rgb = colorsys.hsv_to_rgb((index / max(n, 1)) % 1.0, 0.55, 0.95)
        colors.append(tuple(int(channel * 255) for channel in rgb))
    return numpy.array(colors, dtype=numpy.uint8)


class CubePartRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.pipeline = None
        self.pipeline_module: ModuleType | None = None
        self.mesh_utils: ModuleType | None = None
        self.torch = None
        self.trimesh = None
        self.pipeline_load_count = 0

    def ensure_pipeline(self, request_id: str):
        if self.pipeline is not None:
            return self.pipeline
        emit_progress(request_id, "load_pipeline", 0, 1)
        self.pipeline_module = _load_official_pipeline(self.args.source_dir)
        self.mesh_utils = _load_official_mesh_utils(self.args.source_dir)
        self.torch = importlib.import_module("torch")
        self.trimesh = importlib.import_module("trimesh")
        if not self.torch.cuda.is_available():
            raise RuntimeError("CubePart requires a CUDA runtime")
        checkpoint = self.args.weights_dir / "multi_part_dit.safetensors"
        vae = self.args.weights_dir / "vae.safetensors"
        if not checkpoint.is_file() or not vae.is_file():
            raise FileNotFoundError("Pinned CubePart weights are incomplete")
        config = self.args.source_dir / "configs" / "shape_denoiser_multimesh.yaml"
        if not config.is_file():
            raise FileNotFoundError(f"Pinned CubePart config is missing: {config}")
        self.pipeline = self.pipeline_module.PartShapeDenoiserPipeline(
            config_path=str(config),
            checkpoint_path=str(checkpoint),
            vae_checkpoint_path=str(vae),
            device="cuda",
            extract_geometry_fn_name="extract_geometry_coarse_to_fine",
        )
        self.pipeline_load_count += 1
        emit_progress(request_id, "load_pipeline", 1, 1, pipeline_load_count=self.pipeline_load_count)
        return self.pipeline

    def resolved_revisions(self, request: dict[str, Any]) -> dict[str, str]:
        actual = {
            "source": _git_revision(self.args.source_dir),
            "model": _snapshot_revision(self.args.weights_dir),
            "environment": os.environ.get("COMFYCOLAB_CUBEPART_ENVIRONMENT_REF", ""),
        }
        expected = request.get("revisions")
        if not isinstance(expected, dict):
            raise RuntimeError("CubePart request omitted pinned revision claims")
        for name, value in actual.items():
            requested = str(expected.get(name, ""))
            if requested and requested != value:
                raise RuntimeError(f"CubePart {name} revision mismatch: requested {requested!r}, resolved {value!r}")
        return actual

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        parts = _validate_request(request)
        request_id = str(request["request_id"])
        started = time.monotonic()
        final_dir = Path(str(request["output_dir"])).resolve()
        partial_dir = final_dir.parent / f".{final_dir.name}.{request_id}.partial"
        shutil.rmtree(partial_dir, ignore_errors=True)
        shutil.rmtree(final_dir, ignore_errors=True)
        partial_dir.mkdir(parents=True, exist_ok=True)
        try:
            revisions = self.resolved_revisions(request)
            pipe = self.ensure_pipeline(request_id)
            assert self.mesh_utils is not None and self.torch is not None and self.trimesh is not None
            emit_progress(request_id, "encode", 0, 1)
            mesh, _, _ = self.mesh_utils.load_mesh(str(request["input_mesh"]))
            surface = self.mesh_utils.sample_surface(mesh, num_samples=int(request["num_samples"]))
            surface = self.torch.from_numpy(surface).to(pipe.device).unsqueeze(0).float()
            latents, _ = pipe.encode_shape(surface)
            emit_progress(request_id, "encode", 1, 1)
            emit_progress(request_id, "generate", 0, 1, part_count=len(parts))
            shape_input = self.pipeline_module.ShapeInput(prompt=[parts], latents=latents)
            part_meshes = pipe.input_to_part_shape(
                shape_input,
                guidance_scale=float(request["guidance_scale"]),
                resolution_base=float(request["resolution_base"]),
                scheduler_type=str(request["scheduler"]),
                timeshift=float(request["timeshift"]),
                num_inference_steps=int(request["num_inference_steps"]),
                seed=int(request["seed"]),
                output_mesh=True,
            )
            emit_progress(request_id, "generate", 1, 1)
            emit_progress(request_id, "export", 0, len(parts))
            palette = _palette(len(parts))
            scene = self.trimesh.Scene()
            manifest_parts = []
            for index, name in enumerate(parts):
                vertices, faces = part_meshes[index] if index < len(part_meshes) else (None, None)
                if vertices is None or faces is None:
                    continue
                mesh = self.trimesh.Trimesh(vertices, faces)
                filename = f"part_{index:02d}_{_safe_name(name)}.glb"
                path = partial_dir / filename
                mesh.export(path)
                colored = mesh.copy()
                colored.visual.face_colors = palette[index % len(palette)]
                scene.add_geometry(colored, geom_name=f"part_{index:02d}_{_safe_name(name)}")
                manifest_parts.append(
                    {
                        "index": index,
                        "name": name,
                        "file": filename,
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
                emit_progress(request_id, "export", index + 1, len(parts), part=name)
            if not manifest_parts:
                raise RuntimeError("CubePart produced no part geometry")
            combined = partial_dir / "parts.glb"
            scene.export(combined)
            validation = _load_comfycolab_contract().validate_volumetric_glb(
                combined,
                stage="CubePart combined parts GLB",
            )
            validation_payload = validation.to_dict() if hasattr(validation, "to_dict") else validation
            manifest = {
                "schema": "comfycolab-cubepart-result-v1",
                "request_id": request_id,
                "part_names": parts,
                "parts": manifest_parts,
                "combined": {
                    "file": "parts.glb",
                    "bytes": combined.stat().st_size,
                    "sha256": _sha256(combined),
                },
                "settings": {
                    "seed": int(request["seed"]),
                    "guidance_scale": float(request["guidance_scale"]),
                    "num_inference_steps": int(request["num_inference_steps"]),
                    "resolution_base": float(request["resolution_base"]),
                    "scheduler": request["scheduler"],
                    "timeshift": float(request["timeshift"]),
                    "num_samples": int(request["num_samples"]),
                },
                "license": request["license"],
                "revisions": revisions,
                "validation": validation_payload,
                "runtime_seconds": time.monotonic() - started,
                "worker_pid": os.getpid(),
                "pipeline_load_count": self.pipeline_load_count,
            }
            (partial_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(partial_dir, final_dir)
            return manifest
        except BaseException:
            shutil.rmtree(partial_dir, ignore_errors=True)
            shutil.rmtree(final_dir, ignore_errors=True)
            raise


def _request_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VERSION,
        "request_id": args.request_id,
        "input_mesh": str(args.input_mesh),
        "output_dir": str(args.output_dir),
        "part_names": json.loads(args.parts_json),
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "resolution_base": args.resolution_base,
        "scheduler": args.scheduler,
        "timeshift": args.timeshift,
        "num_samples": args.num_samples,
        "accept_research_license": args.accept_research_license.lower() == "true",
        "license": _license_metadata(),
        "revisions": {
            "source": CUBEPART_SOURCE_REF,
            "model": CUBEPART_MODEL_REF,
            "environment": os.environ.get("COMFYCOLAB_CUBEPART_ENVIRONMENT_REF", ""),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--one-shot", action="store_true")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--request-id", default="")
    parser.add_argument("--input-mesh", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parts-json", default="[]")
    parser.add_argument("--accept-research-license", default="false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--resolution-base", type=float, default=8.5)
    parser.add_argument("--scheduler", default="dpm_solver", choices=["euler", "heun", "dpm_solver"])
    parser.add_argument("--timeshift", type=float, default=4.0)
    parser.add_argument("--num-samples", type=int, default=128_000)
    return parser


def _handle(runtime: CubePartRuntime, request: dict[str, Any]) -> bool:
    if request.get("command") == "shutdown":
        return False
    request_id = str(request.get("request_id", "unknown"))
    try:
        manifest = runtime.run(request)
        emit_result(
            request_id=request_id,
            status="ok",
            output_dir=str(Path(request["output_dir"]).resolve()),
            manifest_path=str(Path(request["output_dir"]).resolve() / "manifest.json"),
            combined_mesh=str(Path(request["output_dir"]).resolve() / "parts.glb"),
            part_count=len(manifest["parts"]),
            worker_pid=os.getpid(),
            pipeline_load_count=runtime.pipeline_load_count,
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
    runtime = CubePartRuntime(args)
    if args.one_shot:
        return 0 if _handle(runtime, _request_from_args(args)) else 0
    if not args.server:
        raise SystemExit("Use --server or --one-shot")
    _emit(READY_PREFIX, {"protocol": PROTOCOL_VERSION, "pid": os.getpid()})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("CubePart protocol requests must be JSON objects")
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
