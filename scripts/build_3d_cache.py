#!/usr/bin/env python3
"""Validate and package the combined G4 TRELLIS.2 + UltraShape environment.

Run this inside the pinned G4 Colab runtime after bootstrap and live inference
smokes pass. The script creates release-ready archive parts and updates the
combined-cache manifest; it deliberately does not publish anything to GitHub.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_MANIFEST = ROOT / "cache" / "3d-g4-v2.json"
DEFAULT_PART_BYTES = 1_900_000_000
LIVE_VALIDATION_SCHEMA = "comfycolab-3d-live-validation-v1"
SKINTOKENS_PATCH_ID = "skintokens-py31115-bpy4222-webp-retry-validation-v3"
REQUIRED_LIVE_GATES = (
    "trellis_512_textured_glb",
    "trellis_1024_cascade_textured_glb",
    "trellis_1536_cascade_genuine",
    "trellis_1536_default_cap_no_downgrade",
    "trellis_multiview_4view_textured_glb",
    "trellis_multiview_6view_textured_glb",
    "combined_environment_cuda_probes",
    "ultrashape_384_refinement",
    "ultrashape_512_refinement",
    "ultrashape_1024_run_1",
    "ultrashape_1024_run_2",
    "pixal3d_cold_1024_textured_glb",
    "pixal3d_object_auto_1024",
    "pixal3d_transparent_1024",
    "pixal3d_worker_reuse_1024",
    "pixal3d_cache_hit_no_inference",
    "pixal3d_cancellation_cleanup",
    "pixal3d_preview_save_glb_reader",
    "pixal3d_1536_experimental",
    "pixal3d_multiview_4view_experimental_glb",
    "pixal3d_multiview_advanced_vggt_omega_glb",
    "skintokens_auto_rig_glb",
    "cubepart_schema_decomposition_glb",
    "full_workflow_hard_surface",
    "full_workflow_organic",
    "full_workflow_thin",
    "full_workflow_holed",
    "full_workflow_transparent_background",
    "cache_hit_no_inference",
    "cancellation_cleanup",
    "advanced_trellis_workflow",
    "preview_and_save_native_file3d",
)
REQUIRED_LIVE_BENCHMARKS = {
    "trellis_512": 512,
    "trellis_1024_cascade": 1024,
    "trellis_1536_cascade": 1536,
    "ultrashape_384": 384,
    "ultrashape_512": 512,
    "ultrashape_1024_run_1": 1024,
    "ultrashape_1024_run_2": 1024,
    "pixal3d_cold_1024": 1024,
    "pixal3d_object_auto_1024": 1024,
    "pixal3d_transparent_1024": 1024,
    "pixal3d_worker_reuse_1024": 1024,
    "pixal3d_preview_save_glb_reader": 1024,
    "pixal3d_1536_experimental": 1536,
}


def skintokens_environment_ref() -> str:
    path = ROOT / "worker" / "skintokens" / "artifacts.py"
    specification = importlib.util.spec_from_file_location(
        "comfycolab_build_skintokens_artifacts", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Unable to load SkinTokens artifact constants: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    try:
        specification.loader.exec_module(module)
    finally:
        sys.modules.pop(specification.name, None)
    return str(module.SKINTOKENS_ENVIRONMENT_REF)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_validation_sources(remote_bootstrap) -> dict[str, str]:
    sources = {
        "comfy": remote_bootstrap.COMFY_REF,
        "trellis": remote_bootstrap.TRELLIS_REF,
        "geometry": remote_bootstrap.GEOMETRY_REF,
        "ultrashape": remote_bootstrap.ULTRASHAPE_REF,
        "cubvh": remote_bootstrap.ULTRASHAPE_CUBVH_REF,
        "birefnet": remote_bootstrap.BIREFNET_MODEL_REF,
        "comfyEnv": remote_bootstrap.COMFY_ENV_VERSION,
    }
    pixal3d_sources = getattr(remote_bootstrap, "expected_pixal3d_sources", None)
    if callable(pixal3d_sources):
        pinned = pixal3d_sources()
        sources.update({
            "pixal3d": pinned["pixal3d"],
            "pixal3dModel": (
                f"{remote_bootstrap.PIXAL3D_MODEL_REPO}@{pinned['pixal3dModel']}"
            ),
            "pixal3dDinov3": (
                f"{remote_bootstrap.PIXAL3D_DINOV3_MODEL_REPO}@{pinned['dinov3']}"
            ),
            "pixal3dMoge": (
                f"{remote_bootstrap.PIXAL3D_MOGE_MODEL_REPO}@{pinned['mogeModel']}"
            ),
            "pixal3dMogeSource": (
                f"microsoft/MoGe@{pinned['mogeSource']}"
            ),
            "pixal3dNaf": (
                f"{remote_bootstrap.PIXAL3D_NAF_REPO}@{pinned['naf']}"
            ),
            "pixal3dNafCheckpoint": pinned["nafCheckpoint"],
            "pixal3dUtils3d": pinned["utils3d"],
            "pixal3dNatten": pinned["natten"],
            "pixal3dNvdiffrast": (
                f"NVlabs/nvdiffrast@{pinned['nvdiffrast']}"
            ),
            "pixal3dEnvironment": pinned["environment"],
        })
    elif hasattr(remote_bootstrap, "PIXAL3D_REF"):
        sources["pixal3d"] = remote_bootstrap.PIXAL3D_REF
    if hasattr(remote_bootstrap, "SKINTOKENS_REF"):
        sources.update({
            "skinTokens": remote_bootstrap.SKINTOKENS_REF,
            "skinTokensModel": (
                f"{remote_bootstrap.SKINTOKENS_MODEL_REPO}@"
                f"{remote_bootstrap.SKINTOKENS_MODEL_REF}"
            ),
            "skinTokensQwen": (
                f"{remote_bootstrap.SKINTOKENS_QWEN_REPO}@"
                f"{remote_bootstrap.SKINTOKENS_QWEN_REF}"
            ),
            "skinTokensEnvironment": skintokens_environment_ref(),
        })
    if hasattr(remote_bootstrap, "CUBEPART_REF"):
        sources.update({
            "cubePart": remote_bootstrap.CUBEPART_REF,
            "cubePartModel": (
                f"{remote_bootstrap.CUBEPART_MODEL_REPO}@"
                f"{remote_bootstrap.CUBEPART_MODEL_REF}"
            ),
        })
    return sources


def expected_validation_patches(remote_bootstrap) -> dict[str, str]:
    patches = {
        "trellis": remote_bootstrap.TRELLIS_PATCH_ID,
        "trellisCategory": remote_bootstrap.TRELLIS_CATEGORY_PATCH_ID,
        "ultrashape": remote_bootstrap.ULTRASHAPE_PATCH_ID,
    }
    pixal3d_patch = getattr(remote_bootstrap, "PIXAL3D_PATCH_ID", None)
    if pixal3d_patch is not None:
        patches["pixal3d"] = pixal3d_patch
    trellis_multiview_patch = getattr(
        remote_bootstrap, "TRELLIS_MULTIVIEW_PATCH_ID", None
    )
    if trellis_multiview_patch is not None:
        patches["trellisMultiview"] = trellis_multiview_patch
    if hasattr(remote_bootstrap, "SKINTOKENS_REF"):
        patches["skinTokens"] = SKINTOKENS_PATCH_ID
    return patches


def _positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def load_live_validation_record(
    path: Path,
    *,
    expected_profile: str,
    remote_bootstrap,
) -> dict[str, object]:
    """Load and strictly validate the live G4 release gate."""

    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Live validation record is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Live validation record is not valid JSON: {path}") from error
    if not isinstance(record, dict):
        raise RuntimeError("Live validation record must be a JSON object.")
    if record.get("schema") != LIVE_VALIDATION_SCHEMA:
        raise RuntimeError(
            f"Live validation record must use schema {LIVE_VALIDATION_SCHEMA}."
        )
    if record.get("status") != "passed":
        raise RuntimeError(
            "Live validation is not passed; the combined cache cannot be marked ready."
        )
    if record.get("profile") != expected_profile:
        raise RuntimeError("Live validation profile does not match the cache profile.")
    for field in ("runId", "completedAt"):
        if not isinstance(record.get(field), str) or not str(record[field]).strip():
            raise RuntimeError(f"Live validation record requires a non-empty {field}.")
    if record.get("sources") != expected_validation_sources(remote_bootstrap):
        raise RuntimeError("Live validation sources do not match the pinned cache sources.")
    if record.get("patches") != expected_validation_patches(remote_bootstrap):
        raise RuntimeError("Live validation patches do not match the pinned cache patches.")

    gates = record.get("gates")
    if not isinstance(gates, dict):
        raise RuntimeError("Live validation record has no gate evidence.")
    failed_gates: list[str] = []
    for name in REQUIRED_LIVE_GATES:
        gate = gates.get(name)
        if (
            not isinstance(gate, dict)
            or gate.get("status") != "passed"
            or not isinstance(gate.get("evidence"), str)
            or not gate["evidence"].strip()
        ):
            failed_gates.append(name)
    if failed_gates:
        raise RuntimeError(
            "Live validation gates are incomplete: " + ", ".join(failed_gates)
        )

    benchmarks = record.get("benchmarks")
    if not isinstance(benchmarks, dict):
        raise RuntimeError("Live validation record has no benchmark evidence.")
    invalid_benchmarks: list[str] = []
    for name, expected_resolution in REQUIRED_LIVE_BENCHMARKS.items():
        benchmark = benchmarks.get(name)
        common_metrics = (
            "runtimeSeconds",
            "peakVramBytes",
            "glbBytes",
            "faces",
        )
        valid = (
            isinstance(benchmark, dict)
            and benchmark.get("status") == "passed"
            and benchmark.get("actualResolution") == expected_resolution
            and benchmark.get("glbValidated") is True
            and all(_positive_number(benchmark.get(metric)) for metric in common_metrics)
        )
        if valid and name.startswith(("trellis_", "pixal3d_")):
            valid = _positive_number(benchmark.get("tokens")) and _positive_number(
                benchmark.get("textureSize")
            )
        if valid and name.startswith("pixal3d_"):
            valid = all(
                _positive_number(benchmark.get(metric))
                for metric in ("workerPeakVramBytes", "pipelineLoadCount", "workerPid")
            )
        if not valid:
            invalid_benchmarks.append(name)
    if invalid_benchmarks:
        raise RuntimeError(
            "Live validation benchmarks are incomplete: "
            + ", ".join(invalid_benchmarks)
        )
    return record


def live_validation_provenance(path: Path, record: dict[str, object]) -> dict[str, object]:
    payload = path.read_bytes()
    try:
        current_record = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError("Live validation record changed after validation.") from error
    if current_record != record:
        raise RuntimeError("Live validation record changed after validation.")
    return {
        "schema": LIVE_VALIDATION_SCHEMA,
        "recordFile": path.name,
        "recordBytes": len(payload),
        "recordSha256": hashlib.sha256(payload).hexdigest(),
        "runId": record["runId"],
        "completedAt": record["completedAt"],
        "profile": record["profile"],
        "passedGates": list(REQUIRED_LIVE_GATES),
    }


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(f"[comfycolab-cache] $ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def ensure_g4_runtime(remote_bootstrap) -> None:
    if not remote_bootstrap.trellis_cache_compatible():
        raise RuntimeError(
            "The combined 3D cache must be built on the pinned Linux G4 runtime "
            "(Python 3.12.13, torch 2.11.0+cu128, CUDA 12.8, SM120, glibc 2.35)."
        )


def directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def split_archive(archive: Path, *, part_bytes: int) -> list[dict[str, object]]:
    parts: list[dict[str, object]] = []
    copied = 0
    total = archive.stat().st_size
    with archive.open("rb") as source:
        index = 0
        while copied < total:
            destination = archive.with_name(f"{archive.name}.part-{index:03d}")
            digest = hashlib.sha256()
            written = 0
            with destination.open("wb") as output:
                while written < part_bytes:
                    chunk = source.read(min(8 * 1024 * 1024, part_bytes - written))
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    copied += len(chunk)
                    print(
                        f"[comfycolab-cache] split {copied / total * 100:5.1f}% "
                        f"({copied / 1_000_000_000:.2f}/{total / 1_000_000_000:.2f} GB)",
                        end="\r",
                        flush=True,
                    )
            parts.append(
                {"name": destination.name, "bytes": written, "sha256": digest.hexdigest()}
            )
            index += 1
    print(flush=True)
    return parts


def runtime_metadata(python: Path) -> dict[str, object]:
    probe = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import json, platform, torch; "
                "print(json.dumps({'python': platform.python_version(), "
                "'torch': torch.__version__, 'torchCuda': torch.version.cuda, "
                "'gpu': torch.cuda.get_device_name(0), "
                "'computeCapability': list(torch.cuda.get_device_capability(0))}))"
            ),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    data = json.loads(probe.stdout)
    data.update(
        {
            "platform": "linux-64",
            "glibc": list(platform.libc_ver()),
            "environment": "trellis2-nodes",
        }
    )
    return data


def build_manifest(
    *,
    template: dict[str, object],
    workspace: Path,
    archive: Path,
    parts: list[dict[str, object]],
    unpacked_bytes: int,
    live_validation: dict[str, object],
    remote_bootstrap,
) -> dict[str, object]:
    env_python = workspace / ".pixi" / "envs" / "trellis2-nodes" / "bin" / "python"
    profile = str(template["profile"])
    return {
        "schema": 1,
        "status": "ready",
        "profile": profile,
        "releaseTag": str(template["releaseTag"]),
        "fallbackProfile": str(template["fallbackProfile"]),
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "archive": {
            "name": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
            "unpackedBytes": unpacked_bytes,
            "parts": parts,
        },
        "runtime": runtime_metadata(env_python),
        "sources": expected_validation_sources(remote_bootstrap),
        "patches": expected_validation_patches(remote_bootstrap),
        "inputs": {
            "pixiTomlSha256": sha256_file(workspace / "pixi.toml"),
            "pixiLockSha256": sha256_file(workspace / "pixi.lock"),
            "installHash": (workspace / "install.hash").read_text(encoding="utf-8").strip(),
        },
        "validation": {
            "trellisImports": [
                "cumesh_vb",
                "drtk",
                "flash_attn",
                "flex_gemm_ap",
                "o_voxel_vb_ap",
                "sageattention",
            ],
            "geometryImports": ["cumesh"],
            "ultrashapeImports": [
                "cubvh",
                "ultrashape.pipelines.UltraShapePipeline",
                "ultrashape.surface_loaders.SharpEdgeSurfaceLoader",
            ],
            "cudaTensorProbe": True,
        },
        "liveValidation": live_validation,
    }


def build(args: argparse.Namespace) -> None:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from comfycolab import remote_bootstrap

    template = json.loads(args.manifest.read_text(encoding="utf-8"))
    if template.get("status") not in {"awaiting-build", "ready"}:
        raise RuntimeError("Combined cache manifest has an unsupported status.")
    ensure_g4_runtime(remote_bootstrap)
    validation_record = load_live_validation_record(
        args.validation_record.resolve(),
        expected_profile=str(template["profile"]),
        remote_bootstrap=remote_bootstrap,
    )
    validation_provenance = live_validation_provenance(
        args.validation_record.resolve(), validation_record
    )
    if shutil.which("zstd") is None:
        run(["apt-get", "update", "-qq"])
        run(["apt-get", "install", "-y", "-qq", "zstd"])

    workspace = args.workspace.expanduser().resolve()
    if args.install_overlay:
        remote_bootstrap.install_ultrashape_overlay()
    remote_bootstrap.validate_trellis_cache(workspace, validate_ultrashape=True)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{template['profile']}.tar.zst"
    existing = [archive, *output_dir.glob(f"{archive.name}.part-*")]
    if any(path.exists() for path in existing) and not args.force:
        raise FileExistsError(
            f"Cache output already exists in {output_dir}; pass --force to rebuild it."
        )
    for path in existing:
        path.unlink(missing_ok=True)

    unpacked_bytes = directory_size(workspace)
    run(
        [
            "tar",
            "--zstd",
            "--numeric-owner",
            "-cf",
            str(archive),
            "-C",
            str(workspace.parent),
            workspace.name,
        ]
    )
    remote_bootstrap.validate_trellis_archive(archive)
    parts = split_archive(archive, part_bytes=args.part_bytes)
    manifest = build_manifest(
        template=template,
        workspace=workspace,
        archive=archive,
        parts=parts,
        unpacked_bytes=unpacked_bytes,
        live_validation=validation_provenance,
        remote_bootstrap=remote_bootstrap,
    )
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"[comfycolab-cache] Ready manifest: {args.manifest}", flush=True)
    print(f"[comfycolab-cache] Release parts: {output_dir}", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    result.add_argument(
        "--validation-record",
        type=Path,
        required=True,
        help="Passed machine-readable G4 validation record to bind into the ready manifest.",
    )
    result.add_argument("--workspace", type=Path, default=Path.home() / ".ce")
    result.add_argument("--output-dir", type=Path, default=Path("/content/.comfycolab/cache-build"))
    result.add_argument("--part-bytes", type=int, default=DEFAULT_PART_BYTES)
    result.add_argument("--install-overlay", action="store_true")
    result.add_argument("--force", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.part_bytes < 64 * 1024 * 1024:
        raise ValueError("--part-bytes must be at least 64 MiB.")
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
