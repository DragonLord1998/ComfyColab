from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PIXAL3D_SOURCE_REF = "cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af"
PIXAL3D_MODEL_REPO = "TencentARC/Pixal3D"
PIXAL3D_MODEL_REF = "0b31f9160aa400719af409098bff7936a932f726"
DINOV3_MODEL_REPO = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_MODEL_REF = "3c276edd87d6f6e569ff0c4400e086807d0f3881"
MOGE_MODEL_REPO = "Ruicheng/moge-2-vitl"
MOGE_MODEL_REF = "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"
MOGE_SOURCE_REF = "07444410f1e33f402353b99d6ccd26bd31e469e8"
NAF_SOURCE_REPO = "https://github.com/valeoai/NAF.git"
NAF_SOURCE_REF = "37f2dfc180f2de53d98bd601109c0da0dd6b0f43"
NAF_CHECKPOINT_URL = "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth"
NAF_CHECKPOINT_SHA256 = "c096c1ab2217a5c3ac136365f721685e2201379cb69d509cfb0261183847c98f"
PIXAL3D_ENVIRONMENT_REF = "g4-linux64-py31213-torch2110-cu128-sm120-pixal3d-v1"
ARTIFACT_SCHEMA = "comfycolab-pixal3d-artifacts-v1"
MIN_FREE_BYTES = 35 * 1024**3
_VALIDATED_SNAPSHOT_STATS: dict[tuple[str, str, str], tuple[tuple[str, int, int], ...]] = {}


@dataclass(frozen=True)
class Pixal3DArtifacts:
    model_dir: Path
    dinov3_dir: Path
    moge_dir: Path
    naf_source_dir: Path
    naf_checkpoint: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _emit(progress: Callable[[dict], None], stage: str, **details) -> None:
    progress({"stage": stage, **details})


def _download_verified(
    url: str,
    destination: Path,
    sha256: str,
    progress: Callable[[dict], None],
) -> Path:
    if destination.is_file() and _sha256(destination) == sha256:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    partial.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0") or 0)
            downloaded = 0
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                _emit(progress, "download", downloaded_bytes=downloaded, total_bytes=total)
            output.flush()
            os.fsync(output.fileno())
        if _sha256(partial) != sha256:
            raise RuntimeError(f"Checksum mismatch while downloading {destination.name}")
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def _snapshot_dir(root: Path, label: str, revision: str) -> Path:
    return root / f"{label}-{revision[:12]}"


def _snapshot_files(destination: Path) -> list[Path]:
    marker = destination / ".comfycolab-artifact.json"
    return sorted(
        path
        for path in destination.rglob("*")
        if path.is_file()
        and path != marker
        and ".cache" not in path.relative_to(destination).parts
    )


def _snapshot_inventory(destination: Path) -> dict[str, dict[str, object]]:
    return {
        path.relative_to(destination).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in _snapshot_files(destination)
    }


def _snapshot_stat_fingerprint(
    destination: Path, expected: dict
) -> tuple[tuple[str, int, int], ...] | None:
    rows: list[tuple[str, int, int]] = []
    try:
        for relative in sorted(expected):
            relative_path = Path(str(relative))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                return None
            stat = (destination / relative_path).stat()
            rows.append((relative_path.as_posix(), stat.st_size, stat.st_mtime_ns))
    except OSError:
        return None
    return tuple(rows)


def _snapshot_inventory_valid(destination: Path, payload: dict) -> tuple[bool, list[Path]]:
    expected = payload.get("files")
    if not isinstance(expected, dict) or not expected:
        return False, []
    cache_key = (
        str(destination.resolve()),
        str(payload.get("repo", "")),
        str(payload.get("revision", "")),
    )
    fingerprint = _snapshot_stat_fingerprint(destination, expected)
    if fingerprint is not None and _VALIDATED_SNAPSHOT_STATS.get(cache_key) == fingerprint:
        return True, []
    invalid: list[Path] = []
    for relative, metadata in expected.items():
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            invalid.append(destination / ".invalid-artifact-marker")
            continue
        path = destination / relative_path
        try:
            valid = (
                isinstance(metadata, dict)
                and path.is_file()
                and path.stat().st_size == int(metadata.get("bytes", -1))
                and _sha256(path) == metadata.get("sha256")
            )
        except (OSError, TypeError, ValueError):
            valid = False
        if not valid:
            invalid.append(path)
    actual = {path.relative_to(destination).as_posix() for path in _snapshot_files(destination)}
    if actual != set(expected):
        invalid.extend(destination / relative for relative in actual - set(expected))
    if not invalid and fingerprint is not None:
        _VALIDATED_SNAPSHOT_STATS[cache_key] = fingerprint
    return not invalid, invalid


def _ensure_snapshot(
    *,
    repo_id: str,
    revision: str,
    destination: Path,
    sentinel: str,
    progress: Callable[[dict], None],
) -> Path:
    marker = destination / ".comfycolab-artifact.json"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        payload = {}
    inventory_valid, invalid_files = _snapshot_inventory_valid(destination, payload)
    if (
        payload.get("schema") == ARTIFACT_SCHEMA
        and payload.get("repo") == repo_id
        and payload.get("revision") == revision
        and (destination / sentinel).is_file()
        and inventory_valid
    ):
        return destination
    for invalid in invalid_files:
        if invalid.is_file() and destination in invalid.parents:
            invalid.unlink(missing_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required to provision the pinned Pixal3D model artifacts"
        ) from error
    _emit(progress, "snapshot", repo=repo_id, revision=revision)
    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(destination),
            resume_download=True,
        )
    except BaseException as error:
        raise RuntimeError(
            f"Unable to download pinned artifact {repo_id}@{revision}. "
            "If the repository requires access, accept its terms and provide HF_TOKEN."
        ) from error
    if not (destination / sentinel).is_file():
        raise RuntimeError(f"Pinned artifact {repo_id}@{revision} is incomplete: missing {sentinel}")
    inventory = _snapshot_inventory(destination)
    if not inventory:
        raise RuntimeError(f"Pinned artifact {repo_id}@{revision} contains no files")
    marker_payload = {
        "schema": ARTIFACT_SCHEMA,
        "repo": repo_id,
        "revision": revision,
        "files": inventory,
    }
    partial_marker = marker.with_name(f".{marker.name}.{os.getpid()}.partial")
    try:
        partial_marker.write_text(
            json.dumps(marker_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(partial_marker, marker)
    finally:
        partial_marker.unlink(missing_ok=True)
    fingerprint = _snapshot_stat_fingerprint(destination, inventory)
    if fingerprint is not None:
        _VALIDATED_SNAPSHOT_STATS[(str(destination.resolve()), repo_id, revision)] = fingerprint
    return destination


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(cwd) if cwd else None, text=True, stderr=subprocess.STDOUT
    ).strip()


def _ensure_naf_source(destination: Path) -> Path:
    try:
        current = _git("rev-parse", "HEAD", cwd=destination)
    except (FileNotFoundError, subprocess.CalledProcessError):
        current = ""
    if current == NAF_SOURCE_REF and (destination / "hubconf.py").is_file():
        return destination
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".naf-", dir=destination.parent))
    try:
        _git("clone", "--filter=blob:none", "--no-checkout", NAF_SOURCE_REPO, str(staging))
        _git("fetch", "--depth", "1", "origin", NAF_SOURCE_REF, cwd=staging)
        _git("checkout", "--detach", NAF_SOURCE_REF, cwd=staging)
        if _git("rev-parse", "HEAD", cwd=staging) != NAF_SOURCE_REF:
            raise RuntimeError("NAF source checkout did not resolve to the pinned commit")
        os.replace(staging, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return destination


def ensure_pixal3d_artifacts(
    root: str | Path,
    *,
    progress: Callable[[dict], None] = lambda _event: None,
) -> Pixal3DArtifacts:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    model_dir = _snapshot_dir(root, "pixal3d", PIXAL3D_MODEL_REF)
    if not (model_dir / "pipeline.json").is_file():
        free_bytes = shutil.disk_usage(root).free
        if free_bytes < MIN_FREE_BYTES:
            raise RuntimeError(
                "Pixal3D first install needs at least 35 GiB free for its roughly 24 GiB model set; "
                f"only {free_bytes / 1024**3:.1f} GiB is available."
            )
    model_dir = _ensure_snapshot(
        repo_id=PIXAL3D_MODEL_REPO,
        revision=PIXAL3D_MODEL_REF,
        destination=model_dir,
        sentinel="pipeline.json",
        progress=progress,
    )
    dinov3_dir = _ensure_snapshot(
        repo_id=DINOV3_MODEL_REPO,
        revision=DINOV3_MODEL_REF,
        destination=_snapshot_dir(root, "dinov3", DINOV3_MODEL_REF),
        sentinel="config.json",
        progress=progress,
    )
    moge_dir = _ensure_snapshot(
        repo_id=MOGE_MODEL_REPO,
        revision=MOGE_MODEL_REF,
        destination=_snapshot_dir(root, "moge", MOGE_MODEL_REF),
        sentinel="model.pt",
        progress=progress,
    )
    naf_source_dir = _ensure_naf_source(root / f"naf-{NAF_SOURCE_REF[:12]}")
    naf_checkpoint = _download_verified(
        NAF_CHECKPOINT_URL,
        root / "naf_release.pth",
        NAF_CHECKPOINT_SHA256,
        progress,
    )
    return Pixal3DArtifacts(
        model_dir=model_dir,
        dinov3_dir=dinov3_dir,
        moge_dir=moge_dir,
        naf_source_dir=naf_source_dir,
        naf_checkpoint=naf_checkpoint,
    )


__all__ = [
    "ARTIFACT_SCHEMA",
    "DINOV3_MODEL_REF",
    "MOGE_MODEL_REF",
    "NAF_CHECKPOINT_SHA256",
    "NAF_SOURCE_REF",
    "PIXAL3D_ENVIRONMENT_REF",
    "PIXAL3D_MODEL_REF",
    "PIXAL3D_SOURCE_REF",
    "Pixal3DArtifacts",
    "ensure_pixal3d_artifacts",
]
