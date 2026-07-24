"""Provision pinned UltraShape inference artifacts without importing CUDA code.

The downloads intentionally use immutable Hugging Face commit URLs instead of
repository aliases. Large files are streamed into ``.partial`` files, resumed
with HTTP Range requests, verified, and only then atomically promoted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping


DOWNLOAD_PROGRESS_PREFIX = "COMFYCOLAB_PROGRESS="
DEFAULT_ARTIFACT_ROOT = Path("/content/.comfycolab/models/3d")
ULTRASHAPE_REPOSITORY = "infinith/UltraShape"
ULTRASHAPE_REVISION = "5aeb21a7185d39f042d02b2695802f125a6f5159"
ULTRASHAPE_CHECKPOINT_FILENAME = "ultrashape_v1.pt"
ULTRASHAPE_CHECKPOINT_SHA256 = (
    "c96ae010c4169597fd0006dcb08056bf6104a1fca249b10fed7ddded324c3f0f"
)
DINOV2_REPOSITORY = "facebook/dinov2-large"
DINOV2_REVISION = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
DINOV2_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
)
DINOV2_FILE_METADATA = {
    "config.json": (
        549,
        "12df51c069a2dc1305e34ba71ef58bc2407ea553b75f4722a1715c1bce3bbed0",
    ),
    "model.safetensors": (
        1_217_522_888,
        "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23",
    ),
    "preprocessor_config.json": (
        436,
        "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
    ),
}
CHUNK_SIZE = 8 * 1024 * 1024


ProgressCallback = Callable[[dict[str, object]], None]


class ArtifactDownloadError(RuntimeError):
    """Raised when a pinned inference artifact cannot be safely provisioned."""


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    repository: str
    revision: str
    filename: str
    expected_sha256: str | None = None
    expected_size: int | None = None

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/{self.repository}/resolve/"
            f"{self.revision}/{self.filename}?download=true"
        )


@dataclass(frozen=True)
class UltraShapeArtifacts:
    checkpoint: Path
    dinov2_dir: Path


ULTRASHAPE_CHECKPOINT = ArtifactSpec(
    name="ultrashape_checkpoint",
    repository=ULTRASHAPE_REPOSITORY,
    revision=ULTRASHAPE_REVISION,
    filename=ULTRASHAPE_CHECKPOINT_FILENAME,
    expected_sha256=ULTRASHAPE_CHECKPOINT_SHA256,
    expected_size=7_366_231_254,
)
DINOV2_ARTIFACTS = tuple(
    ArtifactSpec(
        name=f"dinov2_{filename.replace('.', '_')}",
        repository=DINOV2_REPOSITORY,
        revision=DINOV2_REVISION,
        filename=filename,
        expected_size=DINOV2_FILE_METADATA[filename][0],
        expected_sha256=DINOV2_FILE_METADATA[filename][1],
    )
    for filename in DINOV2_REQUIRED_FILES
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def print_progress(event: dict[str, object]) -> None:
    """Print a machine-readable progress event for notebook/ComfyUI parsers."""

    print(DOWNLOAD_PROGRESS_PREFIX + json.dumps(event, sort_keys=True), flush=True)


def _sidecar_path(destination: Path) -> Path:
    return destination.with_suffix(destination.suffix + ".sha256")


def _partial_path(destination: Path) -> Path:
    return destination.with_suffix(destination.suffix + ".partial")


def _write_text_atomic(destination: Path, value: str) -> None:
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        partial.write_text(value, encoding="ascii")
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def _read_sidecar(destination: Path) -> tuple[str, int] | None:
    try:
        digest, size = _sidecar_path(destination).read_text(encoding="ascii").split()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            return None
        return digest, int(size)
    except (OSError, ValueError):
        return None


def _verified(
    destination: Path,
    expected_sha256: str | None,
    expected_size: int | None,
) -> bool:
    if not destination.is_file():
        return False
    if expected_size is not None and destination.stat().st_size != expected_size:
        return False
    sidecar = _read_sidecar(destination)
    if sidecar is None:
        return False
    recorded_sha256, recorded_size = sidecar
    if recorded_size != destination.stat().st_size:
        return False
    if expected_sha256 is not None and recorded_sha256 != expected_sha256:
        return False
    return sha256_file(destination) == recorded_sha256


def _request(spec: ArtifactSpec, offset: int) -> urllib.request.Request:
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "ComfyColab/0.1",
    }
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.Request(spec.url, headers=headers)


def _configure_hf_transfer() -> str | None:
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _download_with_hub(spec: ArtifactSpec, destination: Path) -> None:
    """Use authenticated hf-xet first, leaving verification to this pack."""

    token = _configure_hf_transfer()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ArtifactDownloadError(
            "huggingface_hub with hf-xet is unavailable"
        ) from error

    attempts: tuple[str | bool | None, ...] = (token, False) if token else (False,)
    last_error: Exception | None = None
    for candidate in attempts:
        try:
            downloaded = Path(
                hf_hub_download(
                    repo_id=spec.repository,
                    revision=spec.revision,
                    filename=spec.filename,
                    local_dir=str(destination.parent),
                    token=candidate,
                )
            )
            if downloaded.resolve() != destination.resolve():
                raise ArtifactDownloadError(
                    f"hf_hub_download returned an unexpected path for {spec.name}: "
                    f"{downloaded}"
                )
            actual_sha256 = sha256_file(destination)
            if (
                spec.expected_size is not None
                and destination.stat().st_size != spec.expected_size
            ):
                raise ArtifactDownloadError(
                    f"Size mismatch for {destination.name}: expected "
                    f"{spec.expected_size} bytes, received {destination.stat().st_size}."
                )
            if (
                spec.expected_sha256 is not None
                and actual_sha256 != spec.expected_sha256
            ):
                raise ArtifactDownloadError(
                    f"Checksum mismatch for {destination.name}: expected "
                    f"{spec.expected_sha256}, received {actual_sha256}."
                )
            _write_text_atomic(
                _sidecar_path(destination),
                f"{actual_sha256} {destination.stat().st_size}\n",
            )
            return
        except Exception as error:
            last_error = error
            destination.unlink(missing_ok=True)
            _sidecar_path(destination).unlink(missing_ok=True)
    raise ArtifactDownloadError(
        f"hf-xet download failed for {spec.name}: {last_error}"
    )


def _parse_content_range(
    headers: Mapping[str, str],
) -> tuple[int, int, int] | None:
    content_range = headers.get("Content-Range")
    if content_range:
        match = re.fullmatch(
            r"bytes\s+(\d+)-(\d+)/(\d+)",
            content_range.strip(),
        )
        if match:
            return tuple(int(value) for value in match.groups())
    return None


def _response_total(headers: Mapping[str, str], offset: int) -> int | None:
    content_range = _parse_content_range(headers)
    if content_range is not None:
        return content_range[2]
    content_length = headers.get("Content-Length")
    if content_length:
        try:
            return offset + int(content_length)
        except ValueError:
            return None
    return None


class _ProgressReporter:
    def __init__(
        self,
        spec: ArtifactSpec,
        callback: ProgressCallback,
        *,
        interval: float,
    ) -> None:
        self.spec = spec
        self.callback = callback
        self.interval = interval
        self.started_at = time.monotonic()
        self.last_report = 0.0
        self.attempt_started_at = self.started_at
        self.attempt_offset = 0

    def begin_attempt(self, offset: int) -> None:
        self.attempt_started_at = time.monotonic()
        self.attempt_offset = offset

    def report(
        self,
        downloaded: int,
        total: int | None,
        attempt: int,
        status: str,
        *,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        if not force and now - self.last_report < self.interval:
            return
        if downloaded < self.attempt_offset:
            self.begin_attempt(downloaded)
        elapsed = max(now - self.attempt_started_at, 1e-9)
        speed = max(downloaded - self.attempt_offset, 0) / elapsed
        remaining = max(total - downloaded, 0) if total is not None else None
        event: dict[str, object] = {
            "stage": "artifact_download",
            "artifact": self.spec.name,
            "filename": self.spec.filename,
            "revision": self.spec.revision,
            "status": status,
            "attempt": attempt,
            "downloaded_bytes": downloaded,
            "total_bytes": total,
            "percent": round(downloaded / total * 100, 1) if total else None,
            "bytes_per_second": round(speed, 1),
            "eta_seconds": round(remaining / speed, 1) if remaining is not None and speed else None,
        }
        self.callback(event)
        self.last_report = now


def _promote_verified(
    partial: Path,
    destination: Path,
    expected_sha256: str | None,
    expected_size: int | None,
) -> str:
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise ArtifactDownloadError(
            f"Size mismatch for {destination.name}: expected {expected_size} bytes, "
            f"received {partial.stat().st_size}."
        )
    actual_sha256 = sha256_file(partial)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        partial.unlink(missing_ok=True)
        raise ArtifactDownloadError(
            f"Checksum mismatch for {destination.name}: expected {expected_sha256}, "
            f"received {actual_sha256}. The partial file was removed for a clean retry."
        )
    os.replace(partial, destination)
    _write_text_atomic(
        _sidecar_path(destination),
        f"{actual_sha256} {destination.stat().st_size}\n",
    )
    return actual_sha256


def download_artifact(
    spec: ArtifactSpec,
    destination: Path,
    *,
    progress: ProgressCallback = print_progress,
    attempts: int = 5,
    timeout: float = 60.0,
    progress_interval: float = 1.0,
    retry_delay: float = 1.0,
) -> Path:
    """Download one artifact with resume, retries, verification, and atomic publish."""

    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_path(destination)
    sidecar = _sidecar_path(destination)
    reporter = _ProgressReporter(spec, progress, interval=progress_interval)

    if _verified(destination, spec.expected_sha256, spec.expected_size):
        reporter.report(
            destination.stat().st_size,
            destination.stat().st_size,
            0,
            "verified",
            force=True,
        )
        return destination
    if destination.exists() or sidecar.exists():
        destination.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)

    if (
        partial.is_file()
        and spec.expected_size is not None
        and partial.stat().st_size > spec.expected_size
    ):
        partial.unlink()
    if partial.is_file() and spec.expected_sha256 is not None:
        if sha256_file(partial) == spec.expected_sha256:
            _promote_verified(
                partial,
                destination,
                spec.expected_sha256,
                spec.expected_size,
            )
            reporter.report(
                destination.stat().st_size,
                destination.stat().st_size,
                0,
                "complete",
                force=True,
            )
            return destination

    last_error: BaseException | None = None
    try:
        reporter.report(0, spec.expected_size, 1, "xet", force=True)
        _download_with_hub(spec, destination)
        partial.unlink(missing_ok=True)
        reporter.report(
            destination.stat().st_size,
            destination.stat().st_size,
            1,
            "complete",
            force=True,
        )
        return destination
    except ArtifactDownloadError as error:
        last_error = error
        progress(
            {
                "stage": "artifact_download",
                "artifact": spec.name,
                "filename": spec.filename,
                "revision": spec.revision,
                "status": "transport_fallback",
                "failed_transport": "huggingface_hub_hf_xet",
                "fallback_transport": "resumable_urllib",
                "reason": str(error),
            }
        )

    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.is_file() else 0
        reporter.begin_attempt(offset)
        reporter.report(offset, None, attempt, "connecting", force=True)
        try:
            with urllib.request.urlopen(_request(spec, offset), timeout=timeout) as response:
                status = getattr(response, "status", 200) or 200
                resumed = offset > 0 and status == 206
                if resumed:
                    content_range = _parse_content_range(response.headers)
                    if content_range is None or content_range[0] != offset:
                        partial.unlink(missing_ok=True)
                        raise ArtifactDownloadError(
                            f"Server returned an invalid resume range for "
                            f"{destination.name}; retrying from byte zero."
                        )
                if offset and not resumed:
                    partial.unlink(missing_ok=True)
                    offset = 0
                total = _response_total(response.headers, offset)
                if spec.expected_size is not None:
                    if total is not None and total != spec.expected_size:
                        raise ArtifactDownloadError(
                            f"Pinned size mismatch for {destination.name}: expected "
                            f"{spec.expected_size} bytes, server reported {total}."
                        )
                    total = spec.expected_size
                remaining = total - offset if total is not None else None
                if remaining is not None and shutil.disk_usage(destination.parent).free < remaining:
                    raise ArtifactDownloadError(
                        f"Not enough temporary disk space for {destination.name}: "
                        f"need {remaining} more bytes."
                    )

                completed = offset
                mode = "ab" if resumed else "wb"
                reporter.report(completed, total, attempt, "downloading", force=True)
                with partial.open(mode) as output:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        output.write(chunk)
                        completed += len(chunk)
                        reporter.report(completed, total, attempt, "downloading")
                if total is not None and completed != total:
                    raise ArtifactDownloadError(
                        f"Stalled download for {destination.name}: received {completed} of "
                        f"{total} bytes. Retrying from the partial file."
                    )

            _promote_verified(
                partial,
                destination,
                spec.expected_sha256,
                spec.expected_size,
            )
            reporter.report(
                destination.stat().st_size,
                destination.stat().st_size,
                attempt,
                "complete",
                force=True,
            )
            return destination
        except urllib.error.HTTPError as error:
            if (
                error.code == 416
                and partial.is_file()
                and spec.expected_sha256 is not None
            ):
                try:
                    _promote_verified(
                        partial,
                        destination,
                        spec.expected_sha256,
                        spec.expected_size,
                    )
                except ArtifactDownloadError:
                    partial.unlink(missing_ok=True)
                else:
                    reporter.report(
                        destination.stat().st_size,
                        destination.stat().st_size,
                        attempt,
                        "complete",
                        force=True,
                    )
                    return destination
            elif error.code == 416:
                partial.unlink(missing_ok=True)
            last_error = error
        except (
            ArtifactDownloadError,
            OSError,
            TimeoutError,
            socket.timeout,
            urllib.error.URLError,
        ) as error:
            last_error = error

        reporter.report(
            partial.stat().st_size if partial.is_file() else 0,
            None,
            attempt,
            "retrying" if attempt < attempts else "failed",
            force=True,
        )
        if attempt < attempts:
            time.sleep(retry_delay * (2 ** (attempt - 1)))

    raise ArtifactDownloadError(
        f"Unable to download pinned artifact {spec.name} after {attempts} attempts: "
        f"{last_error}"
    )


def ensure_ultrashape_artifacts(
    root: Path = DEFAULT_ARTIFACT_ROOT,
    *,
    progress: ProgressCallback = print_progress,
) -> UltraShapeArtifacts:
    """Ensure the pinned UltraShape checkpoint and minimal DINOv2 snapshot exist."""

    root = Path(root)
    checkpoint_dir = root / "ultrashape" / ULTRASHAPE_REVISION
    dinov2_dir = root / "dinov2-large" / DINOV2_REVISION
    checkpoint = download_artifact(
        ULTRASHAPE_CHECKPOINT,
        checkpoint_dir / ULTRASHAPE_CHECKPOINT.filename,
        progress=progress,
    )
    for artifact in DINOV2_ARTIFACTS:
        download_artifact(artifact, dinov2_dir / artifact.filename, progress=progress)
    return UltraShapeArtifacts(checkpoint=checkpoint, dinov2_dir=dinov2_dir)
