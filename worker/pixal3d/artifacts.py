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
VGGT_OMEGA_SOURCE_REPO = "https://github.com/facebookresearch/vggt-omega.git"
VGGT_OMEGA_SOURCE_REF = "39a0cb8af88554f15ddcb5354cd52bde588fa014"
VGGT_OMEGA_MODEL_REPO = "facebook/VGGT-Omega"
VGGT_OMEGA_MODEL_REF = "05654241adc2f218dfb089c373a011f8a7040576"
VGGT_OMEGA_FALLBACK_MODEL_REPO = "1kaiser/vggt-omega-jax"
VGGT_OMEGA_FALLBACK_MODEL_REF = "a8c3a718e0cf78e9e4c6847229efea793d37f060"
VGGT_OMEGA_CHECKPOINT = "vggt_omega_1b_512.pt"
VGGT_OMEGA_FALLBACK_CHECKPOINT_URL = (
    "https://huggingface.co/1kaiser/vggt-omega-jax/resolve/"
    "a8c3a718e0cf78e9e4c6847229efea793d37f060/"
    "vggt_omega_1b_512.pt?download=true"
)
VGGT_OMEGA_CHECKPOINT_BYTES = 4_576_706_117
VGGT_OMEGA_CHECKPOINT_SHA256 = (
    "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934"
)
PIXAL3D_ENVIRONMENT_REF = "g4-linux64-py31213-torch2110-cu130-sm120-pixal3d-v5"
ARTIFACT_SCHEMA = "comfycolab-pixal3d-artifacts-v1"
MIN_FREE_BYTES = 35 * 1024**3
_VALIDATED_SNAPSHOT_STATS: dict[tuple[str, str, str], tuple[tuple[str, int, int], ...]] = {}


def _configure_hf_transfer() -> str | None:
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


@dataclass(frozen=True)
class Pixal3DArtifacts:
    model_dir: Path
    dinov3_dir: Path
    moge_dir: Path
    naf_source_dir: Path
    naf_checkpoint: Path


@dataclass(frozen=True)
class Pixal3DAdvancedArtifacts(Pixal3DArtifacts):
    vggt_omega_source_dir: Path
    vggt_omega_checkpoint: Path
    vggt_omega_checkpoint_repo: str
    vggt_omega_checkpoint_ref: str
    vggt_omega_checkpoint_fallback: bool


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
    allow_patterns: list[str] | None = None,
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
        and payload.get("allow_patterns") == (allow_patterns or None)
        and (destination / sentinel).is_file()
        and inventory_valid
    ):
        return destination
    for invalid in invalid_files:
        if invalid.is_file() and destination in invalid.parents:
            invalid.unlink(missing_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    token = _configure_hf_transfer()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required to provision the pinned Pixal3D model artifacts"
        ) from error
    _emit(progress, "snapshot", repo=repo_id, revision=revision)
    candidates: tuple[str | bool | None, ...] = (token, False) if token else (False,)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=str(destination),
                allow_patterns=allow_patterns,
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
    inventory = _snapshot_inventory(destination)
    if not inventory:
        raise RuntimeError(f"Pinned artifact {repo_id}@{revision} contains no files")
    marker_payload = {
        "schema": ARTIFACT_SCHEMA,
        "repo": repo_id,
        "revision": revision,
        "allow_patterns": allow_patterns or None,
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


def _ensure_git_source(
    *,
    repository: str,
    revision: str,
    destination: Path,
    sentinel: str,
) -> Path:
    try:
        current = _git("rev-parse", "HEAD", cwd=destination)
    except (FileNotFoundError, subprocess.CalledProcessError):
        current = ""
    if current == revision and (destination / sentinel).is_file():
        return destination
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".source-", dir=destination.parent))
    try:
        try:
            _git("clone", "--filter=blob:none", "--no-checkout", repository, str(staging))
            _git("fetch", "--depth", "1", "origin", revision, cwd=staging)
            _git("checkout", "--detach", revision, cwd=staging)
            resolved_revision = _git("rev-parse", "HEAD", cwd=staging)
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                f"Unable to provision pinned source {repository}@{revision}"
            ) from error
        if resolved_revision != revision:
            raise RuntimeError(f"Source checkout did not resolve to {revision}")
        if not (staging / sentinel).is_file():
            raise RuntimeError(
                f"Pinned source checkout is incomplete: missing {sentinel}"
            )
        os.replace(staging, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return destination


def _validate_vggt_omega_checkpoint(checkpoint: Path) -> None:
    try:
        size = checkpoint.stat().st_size
    except OSError as error:
        raise RuntimeError(
            f"Pinned VGGT-Omega checkpoint is unavailable: {checkpoint}"
        ) from error
    if size != VGGT_OMEGA_CHECKPOINT_BYTES:
        raise RuntimeError(
            "Pinned VGGT-Omega checkpoint size mismatch: "
            f"expected {VGGT_OMEGA_CHECKPOINT_BYTES}, got {size}"
        )
    digest = _sha256(checkpoint)
    if digest != VGGT_OMEGA_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Pinned VGGT-Omega checkpoint SHA-256 mismatch: "
            f"expected {VGGT_OMEGA_CHECKPOINT_SHA256}, got {digest}"
        )


def _ensure_direct_hf_snapshot(
    *,
    url: str,
    repo_id: str,
    revision: str,
    destination: Path,
    filename: str,
    expected_bytes: int,
    expected_sha256: str,
    progress: Callable[[dict], None],
) -> Path:
    """Download one immutable Hub file without the optional Xet client."""

    marker = destination / ".comfycolab-artifact.json"
    checkpoint = destination / filename
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        payload = {}
    inventory_valid, invalid_files = _snapshot_inventory_valid(destination, payload)
    if (
        payload.get("schema") == ARTIFACT_SCHEMA
        and payload.get("repo") == repo_id
        and payload.get("revision") == revision
        and payload.get("allow_patterns") == [filename]
        and checkpoint.is_file()
        and inventory_valid
    ):
        return destination
    for invalid in invalid_files:
        if invalid.is_file() and destination in invalid.parents:
            invalid.unlink(missing_ok=True)

    destination.mkdir(parents=True, exist_ok=True)
    if checkpoint.is_file():
        try:
            checkpoint_valid = (
                checkpoint.stat().st_size == expected_bytes
                and _sha256(checkpoint) == expected_sha256
            )
        except OSError:
            checkpoint_valid = False
        if not checkpoint_valid:
            checkpoint.unlink(missing_ok=True)

    if not checkpoint.is_file():
        partial = checkpoint.with_name(f".{checkpoint.name}.download.partial")
        try:
            resume_from = partial.stat().st_size
        except OSError:
            resume_from = 0
        if resume_from < 0 or resume_from > expected_bytes:
            partial.unlink(missing_ok=True)
            resume_from = 0
        headers = {"User-Agent": "ComfyColab/1.0"}
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"
        _emit(
            progress,
            "snapshot_direct",
            repo=repo_id,
            revision=revision,
            url=url,
            resume_from_bytes=resume_from,
        )
        request = urllib.request.Request(url, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=120)
        except BaseException as error:
            raise RuntimeError(
                f"Unable to download pinned artifact {repo_id}@{revision} "
                "through its immutable direct URL."
            ) from error
        with response:
            content_range = response.headers.get("Content-Range", "")
            append = bool(
                resume_from
                and int(getattr(response, "status", 0)) == 206
                and content_range.startswith(f"bytes {resume_from}-")
            )
            if not append:
                resume_from = 0
            downloaded = resume_from
            with partial.open("ab" if append else "wb") as output:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    _emit(
                        progress,
                        "download",
                        artifact="VGGT-Omega-1B-512",
                        downloaded_bytes=downloaded,
                        total_bytes=expected_bytes,
                    )
                output.flush()
                os.fsync(output.fileno())
        try:
            actual_bytes = partial.stat().st_size
        except OSError as error:
            raise RuntimeError(
                f"Direct download for {repo_id}@{revision} produced no file."
            ) from error
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"Direct download for {repo_id}@{revision} is incomplete: "
                f"expected {expected_bytes} bytes, got {actual_bytes}."
            )
        digest = _sha256(partial)
        if digest != expected_sha256:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"Direct download for {repo_id}@{revision} failed SHA-256 "
                f"validation: expected {expected_sha256}, got {digest}."
            )
        os.replace(partial, checkpoint)

    inventory = {
        filename: {
            "bytes": expected_bytes,
            "sha256": expected_sha256,
        }
    }
    marker_payload = {
        "schema": ARTIFACT_SCHEMA,
        "repo": repo_id,
        "revision": revision,
        "allow_patterns": [filename],
        "files": inventory,
        "transport": "immutable-direct-resolve",
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
        _VALIDATED_SNAPSHOT_STATS[(str(destination.resolve()), repo_id, revision)] = (
            fingerprint
        )
    return destination


def _ensure_vggt_omega_checkpoint(
    root: Path,
    *,
    progress: Callable[[dict], None],
) -> tuple[Path, str, str, bool]:
    attempts = (
        (
            VGGT_OMEGA_MODEL_REPO,
            VGGT_OMEGA_MODEL_REF,
            "vggt-omega",
            False,
        ),
        (
            VGGT_OMEGA_FALLBACK_MODEL_REPO,
            VGGT_OMEGA_FALLBACK_MODEL_REF,
            "vggt-omega-fallback",
            True,
        ),
    )
    failures: list[str] = []
    for repo_id, revision, label, is_fallback in attempts:
        destination = _snapshot_dir(root, label, revision)
        try:
            try:
                destination = _ensure_snapshot(
                    repo_id=repo_id,
                    revision=revision,
                    destination=destination,
                    sentinel=VGGT_OMEGA_CHECKPOINT,
                    progress=progress,
                    allow_patterns=[VGGT_OMEGA_CHECKPOINT],
                )
            except RuntimeError as snapshot_error:
                if not is_fallback:
                    raise
                _emit(
                    progress,
                    "snapshot_transport_fallback",
                    artifact="VGGT-Omega-1B-512",
                    repo=repo_id,
                    revision=revision,
                    failed_transport="huggingface_hub",
                    fallback_transport="immutable_direct_resolve",
                    reason=str(snapshot_error),
                )
                destination = _ensure_direct_hf_snapshot(
                    url=VGGT_OMEGA_FALLBACK_CHECKPOINT_URL,
                    repo_id=repo_id,
                    revision=revision,
                    destination=destination,
                    filename=VGGT_OMEGA_CHECKPOINT,
                    expected_bytes=VGGT_OMEGA_CHECKPOINT_BYTES,
                    expected_sha256=VGGT_OMEGA_CHECKPOINT_SHA256,
                    progress=progress,
                )
            checkpoint = destination / VGGT_OMEGA_CHECKPOINT
            _validate_vggt_omega_checkpoint(checkpoint)
        except RuntimeError as error:
            failures.append(f"{repo_id}@{revision}: {error}")
            if not is_fallback:
                _emit(
                    progress,
                    "snapshot_fallback",
                    artifact="VGGT-Omega-1B-512",
                    failed_repo=repo_id,
                    failed_revision=revision,
                    fallback_repo=VGGT_OMEGA_FALLBACK_MODEL_REPO,
                    fallback_revision=VGGT_OMEGA_FALLBACK_MODEL_REF,
                    checkpoint_sha256=VGGT_OMEGA_CHECKPOINT_SHA256,
                )
                continue
            raise RuntimeError(
                "Unable to provision the pinned VGGT-Omega-1B-512 checkpoint "
                "from either the official gated repository or the configured "
                "public mirror. " + " | ".join(failures)
            ) from error
        _emit(
            progress,
            "snapshot_ready",
            artifact="VGGT-Omega-1B-512",
            repo=repo_id,
            revision=revision,
            checkpoint_sha256=VGGT_OMEGA_CHECKPOINT_SHA256,
            fallback=is_fallback,
        )
        return checkpoint, repo_id, revision, is_fallback
    raise AssertionError("VGGT-Omega provisioning exhausted no attempts")


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


def ensure_pixal3d_advanced_artifacts(
    root: str | Path,
    *,
    progress: Callable[[dict], None] = lambda _event: None,
) -> Pixal3DAdvancedArtifacts:
    root = Path(root)
    base = ensure_pixal3d_artifacts(root, progress=progress)
    source_dir = _ensure_git_source(
        repository=VGGT_OMEGA_SOURCE_REPO,
        revision=VGGT_OMEGA_SOURCE_REF,
        destination=root / f"vggt-omega-{VGGT_OMEGA_SOURCE_REF[:12]}",
        sentinel="vggt_omega/models/vggt_omega.py",
    )
    official_checkpoint = (
        _snapshot_dir(root, "vggt-omega", VGGT_OMEGA_MODEL_REF)
        / VGGT_OMEGA_CHECKPOINT
    )
    fallback_checkpoint = (
        _snapshot_dir(
            root,
            "vggt-omega-fallback",
            VGGT_OMEGA_FALLBACK_MODEL_REF,
        )
        / VGGT_OMEGA_CHECKPOINT
    )
    if not official_checkpoint.is_file() and not fallback_checkpoint.is_file():
        free_bytes = shutil.disk_usage(root).free
        minimum = VGGT_OMEGA_CHECKPOINT_BYTES + 2 * 1024**3
        if free_bytes < minimum:
            raise RuntimeError(
                "Advanced Pixal3DMV needs about 4.6 GB for the VGGT-Omega "
                f"checkpoint; only {free_bytes / 1024**3:.1f} GiB is available."
            )
    checkpoint, checkpoint_repo, checkpoint_ref, used_fallback = (
        _ensure_vggt_omega_checkpoint(root, progress=progress)
    )
    return Pixal3DAdvancedArtifacts(
        model_dir=base.model_dir,
        dinov3_dir=base.dinov3_dir,
        moge_dir=base.moge_dir,
        naf_source_dir=base.naf_source_dir,
        naf_checkpoint=base.naf_checkpoint,
        vggt_omega_source_dir=source_dir,
        vggt_omega_checkpoint=checkpoint,
        vggt_omega_checkpoint_repo=checkpoint_repo,
        vggt_omega_checkpoint_ref=checkpoint_ref,
        vggt_omega_checkpoint_fallback=used_fallback,
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
    "VGGT_OMEGA_CHECKPOINT",
    "VGGT_OMEGA_CHECKPOINT_BYTES",
    "VGGT_OMEGA_CHECKPOINT_SHA256",
    "VGGT_OMEGA_FALLBACK_CHECKPOINT_URL",
    "VGGT_OMEGA_FALLBACK_MODEL_REF",
    "VGGT_OMEGA_FALLBACK_MODEL_REPO",
    "VGGT_OMEGA_MODEL_REF",
    "VGGT_OMEGA_MODEL_REPO",
    "VGGT_OMEGA_SOURCE_REF",
    "VGGT_OMEGA_SOURCE_REPO",
    "Pixal3DAdvancedArtifacts",
    "Pixal3DArtifacts",
    "ensure_pixal3d_advanced_artifacts",
    "ensure_pixal3d_artifacts",
]
