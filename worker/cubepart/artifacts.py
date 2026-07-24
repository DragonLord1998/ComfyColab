from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


CUBEPART_SOURCE_REF = "3c6d06ddbef3160a1e1950cb13ab63dd12a61e50"
CUBEPART_SOURCE_DEFAULT = "/content/cube/cubepart"
CUBEPART_ENV_DEFAULT = "/content/.comfycolab/envs/cubepart"
CUBEPART_MODEL_ROOT_DEFAULT = "/content/.comfycolab/models/3d/cubepart"
CUBEPART_MODEL_REPO = "Roblox/cubepart"
CUBEPART_MODEL_REF = "28431d124e77040fcaf34c0a71623ff61d35a6c0"
CUBEPART_CHECKPOINT = "multi_part_dit.safetensors"
CUBEPART_VAE_CHECKPOINT = "vae.safetensors"
CUBEPART_ENVIRONMENT_REF = "g4-linux64-py31213-cubepart-v2"
CUBEPART_RUNTIME_REQUIREMENTS = (
    "diffusers==0.37.1",
    "transformers==4.57.3",
    "accelerate==1.13.0",
    "huggingface_hub[hf_xet]>=0.36.0,<2",
)
CUBEPART_CODE_LICENSE = "Cube3D Research-Only RAIL-MS"
CUBEPART_WEIGHTS_LICENSE = "OpenRAIL / Cube3D Research-Only RAIL-MS"
ARTIFACT_SCHEMA = "comfycolab-cubepart-artifacts-v1"
MIN_FREE_BYTES = 14 * 1024**3


@dataclass(frozen=True)
class CubePartArtifacts:
    source_dir: Path
    environment_dir: Path
    python: Path
    weights_dir: Path
    checkpoint: Path
    vae_checkpoint: Path
    license_metadata: dict[str, str]


def cubepart_license_metadata() -> dict[str, str]:
    return {
        "source_ref": CUBEPART_SOURCE_REF,
        "source_license": CUBEPART_CODE_LICENSE,
        "weights_repo": CUBEPART_MODEL_REPO,
        "weights_ref": CUBEPART_MODEL_REF,
        "weights_license": CUBEPART_WEIGHTS_LICENSE,
        "required_acceptance": "accept_research_license",
    }


def require_research_license_acceptance(accept_research_license: bool) -> None:
    if not accept_research_license:
        raise PermissionError(
            "CubePart uses Cube3D Research-Only RAIL-MS artifacts. Set "
            "accept_research_license=True only after accepting the research-only license terms."
        )


def _git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"CubePart source checkout is invalid: {path}") from error


def _ensure_environment(source: Path, environment: Path) -> Path:
    if _git_revision(source.parent) != CUBEPART_SOURCE_REF:
        raise RuntimeError(
            f"CubePart source checkout at {source.parent} is not pinned to {CUBEPART_SOURCE_REF}"
        )
    if not (source / "pyproject.toml").is_file():
        raise RuntimeError(f"CubePart package metadata is missing: {source / 'pyproject.toml'}")

    python = environment / "bin" / "python"
    marker = environment / ".comfycolab-cubepart-env.json"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    if (
        payload.get("environment_ref") == CUBEPART_ENVIRONMENT_REF
        and payload.get("source_ref") == CUBEPART_SOURCE_REF
        and payload.get("runtime_requirements") == list(CUBEPART_RUNTIME_REQUIREMENTS)
        and python.is_file()
    ):
        return python

    if os.environ.get("COMFYCOLAB_CUBEPART_SKIP_ENV_INSTALL") == "1":
        environment.mkdir(parents=True, exist_ok=True)
        return Path(sys.executable)

    if environment.exists():
        shutil.rmtree(environment)
    environment.parent.mkdir(parents=True, exist_ok=True)
    base_python = Path(
        os.environ.get(
            "COMFYCOLAB_CUBEPART_BASE_PYTHON",
            str(Path.home() / ".ce/.pixi/envs/trellis2-nodes/bin/python"),
        )
    )
    if not base_python.is_file():
        raise RuntimeError(
            "CubePart requires the verified ComfyColab TRELLIS CUDA environment as its base"
        )
    subprocess.check_call(
        [str(base_python), "-m", "venv", "--system-site-packages", str(environment)]
    )
    subprocess.check_call(
        [str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
    )
    subprocess.check_call(
        [str(python), "-m", "pip", "install", *CUBEPART_RUNTIME_REQUIREMENTS]
    )
    subprocess.check_call([str(python), "-m", "pip", "install", "-e", str(source)])
    payload = {
        "schema": ARTIFACT_SCHEMA,
        "environment_ref": CUBEPART_ENVIRONMENT_REF,
        "runtime_requirements": list(CUBEPART_RUNTIME_REQUIREMENTS),
        "source_dir": str(source),
        "source_ref": CUBEPART_SOURCE_REF,
        "license": cubepart_license_metadata(),
    }
    partial = marker.with_name(f".{marker.name}.{os.getpid()}.partial")
    try:
        partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(partial, marker)
    finally:
        partial.unlink(missing_ok=True)
    return python


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _emit(progress: Callable[[dict], None], stage: str, **details) -> None:
    progress({"stage": stage, **details})


def _snapshot_inventory(directory: Path) -> dict[str, dict[str, object]]:
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != ".comfycolab-artifact.json":
            files[path.relative_to(directory).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    return files


def _marker_valid(directory: Path) -> bool:
    marker = directory / ".comfycolab-artifact.json"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        payload.get("schema") != ARTIFACT_SCHEMA
        or payload.get("repo") != CUBEPART_MODEL_REPO
        or payload.get("revision") != CUBEPART_MODEL_REF
        or payload.get("license") != cubepart_license_metadata()
    ):
        return False
    expected = payload.get("files")
    if not isinstance(expected, dict):
        return False
    for relative, metadata in expected.items():
        path = directory / str(relative)
        try:
            if (
                not isinstance(metadata, dict)
                or not path.is_file()
                or path.stat().st_size != int(metadata.get("bytes", -1))
                or _sha256(path) != metadata.get("sha256")
            ):
                return False
        except (OSError, TypeError, ValueError):
            return False
    return True


def ensure_cubepart_artifacts(
    *,
    accept_research_license: bool,
    source_dir: str | Path = CUBEPART_SOURCE_DEFAULT,
    environment_dir: str | Path = CUBEPART_ENV_DEFAULT,
    weights_root: str | Path = CUBEPART_MODEL_ROOT_DEFAULT,
    progress: Callable[[dict], None] = lambda _event: None,
) -> CubePartArtifacts:
    """Provision CubePart paths after an explicit research-license gate.

    The ComfyUI process should call this before starting the isolated worker. The gate
    runs before any environment directory, cache directory, or model snapshot is created.
    """

    require_research_license_acceptance(accept_research_license)
    source = Path(source_dir)
    env = Path(environment_dir)
    root = Path(weights_root)
    if _git_revision(source.parent) != CUBEPART_SOURCE_REF:
        raise RuntimeError(
            f"CubePart source checkout at {source.parent} is not pinned to {CUBEPART_SOURCE_REF}"
        )
    if not (source / "pyproject.toml").is_file():
        raise RuntimeError(f"CubePart package metadata is missing: {source / 'pyproject.toml'}")
    weights = root / f"cubepart-{CUBEPART_MODEL_REF[:12]}"
    checkpoint = weights / CUBEPART_CHECKPOINT
    vae = weights / CUBEPART_VAE_CHECKPOINT

    if not (checkpoint.is_file() and vae.is_file() and _marker_valid(weights)):
        root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root).free < MIN_FREE_BYTES:
            free_gib = shutil.disk_usage(root).free / 1024**3
            raise RuntimeError(
                "CubePart first install needs at least 14 GiB free for its roughly "
                f"10 GiB model set; only {free_gib:.1f} GiB is available."
            )
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError(
                "huggingface_hub is required to provision pinned CubePart model artifacts"
            ) from error
        token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )
        candidates: tuple[str | bool | None, ...] = (
            (token, False) if token else (False,)
        )
        _emit(progress, "snapshot", repo=CUBEPART_MODEL_REPO, revision=CUBEPART_MODEL_REF)
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                snapshot_download(
                    repo_id=CUBEPART_MODEL_REPO,
                    revision=CUBEPART_MODEL_REF,
                    local_dir=str(weights),
                    allow_patterns=[CUBEPART_CHECKPOINT, CUBEPART_VAE_CHECKPOINT],
                    token=candidate,
                )
                last_error = None
                break
            except Exception as error:
                last_error = error
        if last_error is not None:
            raise RuntimeError(
                f"Unable to download pinned CubePart artifact "
                f"{CUBEPART_MODEL_REPO}@{CUBEPART_MODEL_REF}. If the repository "
                "requires access, accept its terms and provide HF_TOKEN."
            ) from last_error
        missing = [name for name in (CUBEPART_CHECKPOINT, CUBEPART_VAE_CHECKPOINT) if not (weights / name).is_file()]
        if missing:
            raise RuntimeError(
                f"Pinned CubePart artifact {CUBEPART_MODEL_REPO}@{CUBEPART_MODEL_REF} "
                f"is incomplete: missing {', '.join(missing)}"
            )
        marker = weights / ".comfycolab-artifact.json"
        partial = marker.with_name(f".{marker.name}.{os.getpid()}.partial")
        payload = {
            "schema": ARTIFACT_SCHEMA,
            "repo": CUBEPART_MODEL_REPO,
            "revision": CUBEPART_MODEL_REF,
            "license": cubepart_license_metadata(),
            "files": _snapshot_inventory(weights),
        }
        try:
            partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(partial, marker)
        finally:
            partial.unlink(missing_ok=True)

    python = _ensure_environment(source, env)

    return CubePartArtifacts(
        source_dir=source,
        environment_dir=env,
        python=python,
        weights_dir=weights,
        checkpoint=checkpoint,
        vae_checkpoint=vae,
        license_metadata=cubepart_license_metadata(),
    )


__all__ = [
    "ARTIFACT_SCHEMA",
    "CUBEPART_CHECKPOINT",
    "CUBEPART_CODE_LICENSE",
    "CUBEPART_ENVIRONMENT_REF",
    "CUBEPART_MODEL_REF",
    "CUBEPART_MODEL_REPO",
    "CUBEPART_RUNTIME_REQUIREMENTS",
    "CUBEPART_SOURCE_REF",
    "CUBEPART_VAE_CHECKPOINT",
    "CUBEPART_WEIGHTS_LICENSE",
    "CubePartArtifacts",
    "cubepart_license_metadata",
    "ensure_cubepart_artifacts",
    "require_research_license_acceptance",
]
