from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


SKINTOKENS_SOURCE_REPO = "https://github.com/VAST-AI-Research/SkinTokens.git"
SKINTOKENS_SOURCE_REF = "273b691d35989d71cd17ff2895fdc735097b92d1"
SKINTOKENS_MODEL_REPO = "VAST-AI/SkinTokens"
SKINTOKENS_MODEL_REF = "79736cad0fd84de384d5eede659b4ebd24effe33"
SKINTOKENS_QWEN_REPO = "Qwen/Qwen3-0.6B"
SKINTOKENS_QWEN_REF = "c1899de289a04d12100db370d81485cdf75e47ca"
SKINTOKENS_ENVIRONMENT_REF = "g4-linux64-py311-torch270-cu128-skintokens-v1"
SKINTOKENS_LICENSE = {
    "name": "MIT",
    "copyright": "Copyright (c) 2025 VAST-AI-Research",
    "source": SKINTOKENS_SOURCE_REPO,
}
ARTIFACT_SCHEMA = "comfycolab-skintokens-artifacts-v1"
DEFAULT_SOURCE_DIR = Path("/content/SkinTokens")
DEFAULT_MODEL_ROOT = Path("/content/.comfycolab/models/3d/skintokens")
DEFAULT_ENV_ROOT = Path("/content/.comfycolab/envs/skintokens")
MIN_FREE_BYTES = 18 * 1024**3

MODEL_FILES = (
    "experiments/skin_vae_2_10_32768/last.ckpt",
    "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
)
TOKENRIG_CHECKPOINT = MODEL_FILES[1]


@dataclass(frozen=True)
class SkinTokensArtifacts:
    source_dir: Path
    model_dir: Path
    qwen_dir: Path
    env_dir: Path
    python: Path
    worker_script: Path
    tokenrig_checkpoint: Path
    skin_vae_checkpoint: Path


def _emit(progress: Callable[[dict], None], stage: str, **details) -> None:
    progress({"stage": stage, **details})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _inventory(root: Path) -> dict[str, dict[str, object]]:
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != ".comfycolab-artifact.json"
        and ".cache" not in path.relative_to(root).parts
    }


def _inventory_valid(root: Path, payload: dict) -> bool:
    expected = payload.get("files")
    if not isinstance(expected, dict) or not expected:
        return False
    try:
        for relative, metadata in expected.items():
            relative_path = Path(str(relative))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                return False
            path = root / relative_path
            if (
                not isinstance(metadata, dict)
                or not path.is_file()
                or path.stat().st_size != int(metadata.get("bytes", -1))
                or _sha256(path) != metadata.get("sha256")
            ):
                return False
    except (OSError, TypeError, ValueError):
        return False
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != ".comfycolab-artifact.json"
        and ".cache" not in path.relative_to(root).parts
    }
    return actual == set(expected)


def _ensure_source(source_dir: Path) -> Path:
    try:
        current = _git("rev-parse", "HEAD", cwd=source_dir)
    except (FileNotFoundError, subprocess.CalledProcessError):
        current = ""
    if current == SKINTOKENS_SOURCE_REF and (source_dir / "demo.py").is_file():
        return source_dir
    if source_dir.exists() and any(source_dir.iterdir()):
        raise RuntimeError(
            f"SkinTokens source checkout at {source_dir} is not the pinned "
            f"{SKINTOKENS_SOURCE_REF}; set COMFYCOLAB_SKINTOKENS_SOURCE to a valid checkout."
        )
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".skintokens-source-", dir=source_dir.parent))
    try:
        _git("clone", "--filter=blob:none", "--no-checkout", SKINTOKENS_SOURCE_REPO, str(staging))
        _git("fetch", "--depth", "1", "origin", SKINTOKENS_SOURCE_REF, cwd=staging)
        _git("checkout", "--detach", SKINTOKENS_SOURCE_REF, cwd=staging)
        if _git("rev-parse", "HEAD", cwd=staging) != SKINTOKENS_SOURCE_REF:
            raise RuntimeError("SkinTokens source checkout did not resolve to the pinned commit")
        os.replace(staging, source_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return source_dir


def _ensure_snapshot(
    *,
    repo_id: str,
    revision: str,
    destination: Path,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
    sentinel: str,
    progress: Callable[[dict], None],
) -> Path:
    marker = destination / ".comfycolab-artifact.json"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        payload = {}
    if (
        payload.get("schema") == ARTIFACT_SCHEMA
        and payload.get("repo") == repo_id
        and payload.get("revision") == revision
        and (destination / sentinel).is_file()
        and _inventory_valid(destination, payload)
    ):
        return destination

    destination.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required to provision SkinTokens artifacts") from error

    _emit(progress, "snapshot", repo=repo_id, revision=revision)
    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(destination),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            resume_download=True,
        )
    except BaseException as error:
        raise RuntimeError(
            f"Unable to download pinned artifact {repo_id}@{revision}. "
            "If the repository requires access, accept its terms and provide HF_TOKEN."
        ) from error
    if not (destination / sentinel).is_file():
        raise RuntimeError(f"Pinned artifact {repo_id}@{revision} is incomplete: missing {sentinel}")
    inventory = _inventory(destination)
    if not inventory:
        raise RuntimeError(f"Pinned artifact {repo_id}@{revision} contains no files")
    _atomic_write_json(
        marker,
        {
            "schema": ARTIFACT_SCHEMA,
            "repo": repo_id,
            "revision": revision,
            "files": inventory,
        },
    )
    return destination


def _env_python(env_dir: Path) -> Path:
    return env_dir / "bin" / "python"


def _ensure_environment(source_dir: Path, env_dir: Path, progress: Callable[[dict], None]) -> Path:
    python = _env_python(env_dir)
    marker = env_dir / ".comfycolab-environment.json"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        payload = {}
    if payload.get("environment_ref") == SKINTOKENS_ENVIRONMENT_REF and python.is_file():
        return python

    env_dir.parent.mkdir(parents=True, exist_ok=True)
    _emit(progress, "environment", path=str(env_dir), environment_ref=SKINTOKENS_ENVIRONMENT_REF)
    if os.environ.get("COMFYCOLAB_SKINTOKENS_SKIP_ENV_INSTALL") == "1":
        env_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            marker,
            {
                "schema": ARTIFACT_SCHEMA,
                "environment_ref": SKINTOKENS_ENVIRONMENT_REF,
                "skipped": True,
            },
        )
        return Path(sys.executable)

    base_python = Path(
        os.environ.get(
            "COMFYCOLAB_SKINTOKENS_BASE_PYTHON",
            str(Path.home() / ".ce/.pixi/envs/trellis2-nodes/bin/python"),
        )
    )
    if not base_python.is_file():
        raise RuntimeError(
            "SkinTokens requires the verified ComfyColab TRELLIS Python runtime as its base"
        )
    subprocess.check_call([str(base_python), "-m", "venv", str(env_dir)])
    subprocess.check_call(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
            "packaging",
            "ninja",
        ]
    )
    install = [
        str(python),
        "-m",
        "pip",
        "install",
        "torch==2.7.0",
        "torchvision==0.22.0",
        "torchaudio==2.7.0",
        "--index-url",
        "https://download.pytorch.org/whl/cu128",
    ]
    subprocess.check_call(install)
    subprocess.check_call([str(python), "-m", "pip", "install", "-r", str(source_dir / "requirements.txt")])
    subprocess.check_call([str(python), "-m", "pip", "install", "flash-attn", "--no-build-isolation"])
    _atomic_write_json(
        marker,
        {
            "schema": ARTIFACT_SCHEMA,
            "environment_ref": SKINTOKENS_ENVIRONMENT_REF,
            "python": str(python),
            "source_ref": SKINTOKENS_SOURCE_REF,
        },
    )
    return python


def ensure_skintokens_artifacts(
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    *,
    source_dir: str | Path | None = None,
    env_dir: str | Path = DEFAULT_ENV_ROOT,
    progress: Callable[[dict], None] = lambda _event: None,
) -> SkinTokensArtifacts:
    source = Path(
        source_dir
        or os.environ.get("COMFYCOLAB_SKINTOKENS_SOURCE", "")
        or DEFAULT_SOURCE_DIR
    )
    model_root = Path(model_root)
    env_dir = Path(env_dir)
    model_root.mkdir(parents=True, exist_ok=True)
    if not (model_root / f"skintokens-{SKINTOKENS_MODEL_REF[:12]}" / TOKENRIG_CHECKPOINT).is_file():
        free_bytes = shutil.disk_usage(model_root).free
        if free_bytes < MIN_FREE_BYTES:
            raise RuntimeError(
                "SkinTokens first install needs at least 18 GiB free for model artifacts; "
                f"only {free_bytes / 1024**3:.1f} GiB is available."
            )
    source = _ensure_source(source)
    model_dir = _ensure_snapshot(
        repo_id=SKINTOKENS_MODEL_REPO,
        revision=SKINTOKENS_MODEL_REF,
        destination=model_root / f"skintokens-{SKINTOKENS_MODEL_REF[:12]}",
        allow_patterns=list(MODEL_FILES),
        ignore_patterns=None,
        sentinel=TOKENRIG_CHECKPOINT,
        progress=progress,
    )
    qwen_dir = _ensure_snapshot(
        repo_id=SKINTOKENS_QWEN_REPO,
        revision=SKINTOKENS_QWEN_REF,
        destination=model_root / f"qwen3-0.6b-{SKINTOKENS_QWEN_REF[:12]}",
        allow_patterns=None,
        ignore_patterns=["*.bin", "*.safetensors"],
        sentinel="config.json",
        progress=progress,
    )
    python = _ensure_environment(source, env_dir, progress)
    return SkinTokensArtifacts(
        source_dir=source,
        model_dir=model_dir,
        qwen_dir=qwen_dir,
        env_dir=env_dir,
        python=python,
        worker_script=Path(__file__).with_name("worker_main.py"),
        tokenrig_checkpoint=model_dir / TOKENRIG_CHECKPOINT,
        skin_vae_checkpoint=model_dir / MODEL_FILES[0],
    )


__all__ = [
    "ARTIFACT_SCHEMA",
    "DEFAULT_ENV_ROOT",
    "DEFAULT_MODEL_ROOT",
    "DEFAULT_SOURCE_DIR",
    "SKINTOKENS_ENVIRONMENT_REF",
    "SKINTOKENS_LICENSE",
    "SKINTOKENS_MODEL_REF",
    "SKINTOKENS_MODEL_REPO",
    "SKINTOKENS_QWEN_REF",
    "SKINTOKENS_QWEN_REPO",
    "SKINTOKENS_SOURCE_REF",
    "SKINTOKENS_SOURCE_REPO",
    "SkinTokensArtifacts",
    "ensure_skintokens_artifacts",
]
