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
SKINTOKENS_ENVIRONMENT_REF = (
    "g4-linux64-py31115-torch270-cu128-bpy4222-skintokens-v2"
)
SKINTOKENS_PYTHON_VERSION = "3.11.15"
SKINTOKENS_TORCH_PACKAGES = (
    "torch==2.7.0",
    "torchvision==0.22.0",
    "torchaudio==2.7.0",
)
SKINTOKENS_RUNTIME_PINS = (
    "numpy==1.26.4",
    "bpy==4.2.22",
    "transformers==4.57.3",
    "diffusers==0.37.1",
    "huggingface_hub[hf_xet]>=0.36.0,<2",
)
SKINTOKENS_FLASH_ATTN_PACKAGE = "flash-attn==2.8.3.post1"
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
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required to provision SkinTokens artifacts") from error

    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    candidates: tuple[str | bool | None, ...] = (
        (token, False) if token else (False,)
    )
    _emit(progress, "snapshot", repo=repo_id, revision=revision)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=str(destination),
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                token=candidate,
            )
            last_error = None
            break
        except Exception as error:
            last_error = error
    if last_error is not None:
        raise RuntimeError(
            f"Unable to download pinned artifact {repo_id}@{revision}. "
            "If the repository requires access, accept its terms and provide HF_TOKEN."
        ) from last_error
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


def _probe_environment(python: Path) -> dict[str, str]:
    completed = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import json,sys,bpy,diffusers,flash_attn,numpy,torch,transformers;"
                "print(json.dumps({"
                "'python':'.'.join(map(str,sys.version_info[:3])),"
                "'bpy':'.'.join(map(str,bpy.app.version)),"
                "'diffusers':diffusers.__version__,"
                "'flash_attn':flash_attn.__version__,"
                "'numpy':numpy.__version__,"
                "'torch':torch.__version__,"
                "'transformers':transformers.__version__"
                "},sort_keys=True))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    try:
        versions = json.loads(completed.stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError("SkinTokens environment probe returned invalid output") from error
    expected = {
        "python": SKINTOKENS_PYTHON_VERSION,
        "bpy": "4.2.22",
        "diffusers": "0.37.1",
        "flash_attn": "2.8.3.post1",
        "numpy": "1.26.4",
        "torch": "2.7.0+cu128",
        "transformers": "4.57.3",
    }
    if versions != expected:
        raise RuntimeError(
            f"SkinTokens environment versions do not match the pinned recipe: {versions}"
        )
    return versions


def _promote_environment(staging: Path, env_dir: Path) -> None:
    backup = env_dir.with_name(f".{env_dir.name}.{os.getpid()}.backup")
    shutil.rmtree(backup, ignore_errors=True)
    had_existing = env_dir.exists()
    if had_existing:
        os.replace(env_dir, backup)
    try:
        os.replace(staging, env_dir)
    except BaseException:
        if had_existing and backup.exists() and not env_dir.exists():
            os.replace(backup, env_dir)
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)


def _ensure_environment(source_dir: Path, env_dir: Path, progress: Callable[[dict], None]) -> Path:
    python = _env_python(env_dir)
    marker = env_dir / ".comfycolab-environment.json"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        payload = {}
    if (
        payload.get("schema") == ARTIFACT_SCHEMA
        and payload.get("environment_ref") == SKINTOKENS_ENVIRONMENT_REF
        and payload.get("python_version") == SKINTOKENS_PYTHON_VERSION
        and not payload.get("skipped")
        and isinstance(payload.get("versions"), dict)
        and python.is_file()
    ):
        try:
            measured_versions = _probe_environment(python)
        except Exception:
            measured_versions = None
        if measured_versions == payload["versions"]:
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

    uv = os.environ.get("COMFYCOLAB_SKINTOKENS_UV") or shutil.which("uv")
    if not uv:
        raise RuntimeError(
            "SkinTokens requires uv to create its pinned Python 3.11 runtime"
        )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{env_dir.name}-", dir=env_dir.parent)
    )
    try:
        subprocess.check_call(
            [
                uv,
                "venv",
                "--python",
                SKINTOKENS_PYTHON_VERSION,
                "--managed-python",
                "--seed",
                str(staging),
            ]
        )
        staging_python = _env_python(staging)
        subprocess.check_call(
            [
                str(staging_python),
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
        subprocess.check_call(
            [
                str(staging_python),
                "-m",
                "pip",
                "install",
                *SKINTOKENS_TORCH_PACKAGES,
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ]
        )
        subprocess.check_call(
            [
                str(staging_python),
                "-m",
                "pip",
                "install",
                *SKINTOKENS_RUNTIME_PINS,
            ]
        )
        subprocess.check_call(
            [
                str(staging_python),
                "-m",
                "pip",
                "install",
                "-r",
                str(source_dir / "requirements.txt"),
            ]
        )
        subprocess.check_call(
            [
                str(staging_python),
                "-m",
                "pip",
                "install",
                SKINTOKENS_FLASH_ATTN_PACKAGE,
                "--no-build-isolation",
            ],
            env={**os.environ, "MAX_JOBS": "8"},
        )
        versions = _probe_environment(staging_python)
        _atomic_write_json(
            staging / ".comfycolab-environment.json",
            {
                "schema": ARTIFACT_SCHEMA,
                "environment_ref": SKINTOKENS_ENVIRONMENT_REF,
                "python": str(_env_python(env_dir)),
                "python_version": SKINTOKENS_PYTHON_VERSION,
                "source_ref": SKINTOKENS_SOURCE_REF,
                "versions": versions,
            },
        )
        _promote_environment(staging, env_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
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
    "SKINTOKENS_FLASH_ATTN_PACKAGE",
    "SKINTOKENS_LICENSE",
    "SKINTOKENS_MODEL_REF",
    "SKINTOKENS_MODEL_REPO",
    "SKINTOKENS_PYTHON_VERSION",
    "SKINTOKENS_QWEN_REF",
    "SKINTOKENS_QWEN_REPO",
    "SKINTOKENS_RUNTIME_PINS",
    "SKINTOKENS_SOURCE_REF",
    "SKINTOKENS_SOURCE_REPO",
    "SKINTOKENS_TORCH_PACKAGES",
    "SkinTokensArtifacts",
    "ensure_skintokens_artifacts",
]
