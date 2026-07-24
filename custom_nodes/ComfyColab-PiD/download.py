from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runtime import hf_download as _hf_download


ProgressCallback = _hf_download.ProgressCallback
CHUNK_SIZE = _hf_download.CHUNK_SIZE
DEFAULT_ATTEMPTS = _hf_download.DEFAULT_ATTEMPTS
RETRYABLE_HTTP_CODES = _hf_download.RETRYABLE_HTTP_CODES
DownloadError = _hf_download.DownloadError
urllib = _hf_download.urllib
time = _hf_download.time
sha256_file = _hf_download.sha256_file
_VERIFIED_FILES: set[tuple[str, int, int, str]] = set()


def _verification_key(path: Path, expected_sha256: str) -> tuple[str, int, int, str]:
    stat = path.stat()
    return (str(path.resolve()), stat.st_size, stat.st_mtime_ns, expected_sha256)


def _forget_verified(path: Path) -> None:
    resolved = str(path.resolve())
    _VERIFIED_FILES.difference_update(
        key for key in _VERIFIED_FILES if key[0] == resolved
    )


def _record_verified(
    path: Path,
    expected_sha256: str,
    expected_size: int | None,
) -> None:
    actual_size = path.stat().st_size if expected_size is None else expected_size
    marker = path.with_suffix(path.suffix + ".sha256")
    marker.write_text(f"{expected_sha256} {actual_size}\n", encoding="ascii")
    _VERIFIED_FILES.add(_verification_key(path, expected_sha256))


def _verified(
    path: Path,
    expected_sha256: str,
    expected_size: int | None,
) -> bool:
    if expected_size is None:
        return path.is_file() and sha256_file(path) == expected_sha256
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    key = _verification_key(path, expected_sha256)
    if key in _VERIFIED_FILES:
        return True
    if sha256_file(path) != expected_sha256:
        return False
    _record_verified(path, expected_sha256, expected_size)
    return True


def download_file(
    *,
    url: str,
    destination: Path,
    expected_sha256: str,
    expected_size: int,
    force: bool = False,
    progress: ProgressCallback | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
) -> Path:
    return _hf_download.download_file(
        url=url,
        destination=destination,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        force=force,
        progress=progress,
        attempts=attempts,
        user_agent="ComfyColab-PiD/0.1",
        is_verified=_verified,
        record_verified=_record_verified,
        forget_verified=_forget_verified,
    )
