from __future__ import annotations

import sys
from pathlib import Path

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


def _record_verified(
    path: Path,
    expected_sha256: str,
    _expected_size: int | None = None,
) -> None:
    marker = path.with_suffix(path.suffix + ".sha256")
    marker.write_text(f"{expected_sha256} {path.stat().st_size}\n", encoding="ascii")


def _verified(
    path: Path,
    expected_sha256: str,
    _expected_size: int | None = None,
) -> bool:
    if not path.is_file():
        return False
    marker = path.with_suffix(path.suffix + ".sha256")
    try:
        marker_sha256, marker_size = marker.read_text(encoding="ascii").split()
        if marker_sha256 == expected_sha256 and int(marker_size) == path.stat().st_size:
            return True
    except (OSError, ValueError):
        pass
    if sha256_file(path) != expected_sha256:
        return False
    _record_verified(path, expected_sha256)
    return True


def download_file(
    *,
    url: str,
    destination: Path,
    expected_sha256: str,
    force: bool = False,
    progress: ProgressCallback | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
) -> Path:
    return _hf_download.download_file(
        url=url,
        destination=destination,
        expected_sha256=expected_sha256,
        force=force,
        progress=progress,
        attempts=attempts,
        user_agent="ComfyColab/0.1",
        is_verified=_verified,
        record_verified=_record_verified,
    )
