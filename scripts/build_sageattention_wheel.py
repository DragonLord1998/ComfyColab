#!/usr/bin/env python3
"""Build the pinned SM120 SageAttention wheel used by the Colab runtime."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys


SAGE_ATTENTION_SOURCE_REF = "eb615cf6cf4d221338033340ee2de1c37fbdba4a"
SAGE_ATTENTION_SETUP_SHA256 = (
    "50213b02a99365e8907d98c2c8607d368cf1197098a3612ea695cfcef95403b9"
)
CUDA_HOME = Path("/usr/local/cuda-13.0")
_DEBUG_FLAGS = (
    'CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", '
    '"-DENABLE_BF16"]'
)
_RELEASE_FLAGS = (
    'CXX_FLAGS = ["-O3", "-fopenmp", "-lgomp", "-std=c++17", '
    '"-DENABLE_BF16"]'
)
_SM80_CONDITION = "if HAS_SM80 or HAS_SM86 or HAS_SM89 or HAS_SM90 or HAS_SM120:"
_TARGET_CONDITION = "if HAS_SM80 or HAS_SM86:"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_sm120_build(setup_path: Path) -> None:
    actual_digest = sha256_file(setup_path)
    if actual_digest != SAGE_ATTENTION_SETUP_SHA256:
        raise RuntimeError(
            "SageAttention setup.py digest mismatch: "
            f"expected {SAGE_ATTENTION_SETUP_SHA256}, got {actual_digest}"
        )
    source = setup_path.read_text(encoding="utf-8")
    if source.count(_DEBUG_FLAGS) != 1 or source.count(_SM80_CONDITION) != 1:
        raise RuntimeError("SageAttention SM120 build patch did not match exactly once")
    source = source.replace(_DEBUG_FLAGS, _RELEASE_FLAGS, 1)
    source = source.replace(_SM80_CONDITION, _TARGET_CONDITION, 1)
    setup_path.write_text(source, encoding="utf-8")


def build_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_HOME": str(CUDA_HOME),
            "PATH": f"{CUDA_HOME / 'bin'}:{environment.get('PATH', '')}",
            "LD_LIBRARY_PATH": (
                f"{CUDA_HOME / 'lib64'}:{environment.get('LD_LIBRARY_PATH', '')}"
            ),
            "TORCH_CUDA_ARCH_LIST": "12.0",
            "EXT_PARALLEL": "1",
            "MAX_JOBS": "2",
            "NVCC_APPEND_FLAGS": "--threads 2",
        }
    )
    return environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != SAGE_ATTENTION_SOURCE_REF:
        raise RuntimeError(
            f"Expected SageAttention {SAGE_ATTENTION_SOURCE_REF}, got {commit}"
        )
    patch_sm120_build(source / "setup.py")
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--verbose",
            "--no-deps",
            "--no-build-isolation",
            ".",
            "--wheel-dir",
            str(output),
        ],
        cwd=source,
        env=build_environment(),
        check=True,
    )
    artifacts = sorted(output.glob("sageattention-2.2.0-*.whl"))
    if len(artifacts) != 1:
        raise RuntimeError(f"Expected one SageAttention wheel, found {artifacts}")
    artifact = artifacts[0]
    print(f"SAGE_WHEEL={artifact}")
    print(f"SAGE_WHEEL_BYTES={artifact.stat().st_size}")
    print(f"SAGE_WHEEL_SHA256={sha256_file(artifact)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
