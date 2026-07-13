#!/usr/bin/env python3
"""Validate and package the combined G4 TRELLIS.2 + UltraShape environment.

Run this inside the pinned G4 Colab runtime after bootstrap and live inference
smokes pass. The script creates release-ready archive parts and updates the
combined-cache manifest; it deliberately does not publish anything to GitHub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "sources": {
            "comfy": remote_bootstrap.COMFY_REF,
            "trellis": remote_bootstrap.TRELLIS_REF,
            "geometry": remote_bootstrap.GEOMETRY_REF,
            "ultrashape": remote_bootstrap.ULTRASHAPE_REF,
            "cubvh": remote_bootstrap.ULTRASHAPE_CUBVH_REF,
            "birefnet": remote_bootstrap.BIREFNET_MODEL_REF,
            "comfyEnv": remote_bootstrap.COMFY_ENV_VERSION,
        },
        "patches": {
            "trellis": remote_bootstrap.TRELLIS_PATCH_ID,
            "ultrashape": remote_bootstrap.ULTRASHAPE_PATCH_ID,
        },
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
    }


def build(args: argparse.Namespace) -> None:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from comfycolab import remote_bootstrap

    ensure_g4_runtime(remote_bootstrap)
    if shutil.which("zstd") is None:
        run(["apt-get", "update", "-qq"])
        run(["apt-get", "install", "-y", "-qq", "zstd"])

    template = json.loads(args.manifest.read_text(encoding="utf-8"))
    if template.get("status") not in {"awaiting-build", "ready"}:
        raise RuntimeError("Combined cache manifest has an unsupported status.")
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
