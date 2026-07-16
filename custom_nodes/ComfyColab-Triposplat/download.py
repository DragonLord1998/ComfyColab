from __future__ import annotations

import hashlib
import http.client
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional


ProgressCallback = Callable[[int, Optional[int]], None]
CHUNK_SIZE = 4 * 1024 * 1024
DEFAULT_ATTEMPTS = 5
RETRYABLE_HTTP_CODES = frozenset({401, 403, 408, 416, 425, 429, 500, 502, 503, 504})
_VERIFIED_FILES: set[tuple[str, int, int, str]] = set()


class DownloadError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _verification_key(path: Path, expected_sha256: str) -> tuple[str, int, int, str]:
    stat = path.stat()
    return (str(path.resolve()), stat.st_size, stat.st_mtime_ns, expected_sha256)


def _forget_verified(path: Path) -> None:
    resolved = str(path.resolve())
    _VERIFIED_FILES.difference_update(key for key in _VERIFIED_FILES if key[0] == resolved)


def _record_verified(path: Path, expected_sha256: str, expected_size: int) -> None:
    marker = path.with_suffix(path.suffix + ".sha256")
    marker.write_text(f"{expected_sha256} {expected_size}\n", encoding="ascii")
    _VERIFIED_FILES.add(_verification_key(path, expected_sha256))


def _verified(path: Path, expected_sha256: str, expected_size: int) -> bool:
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    key = _verification_key(path, expected_sha256)
    if key in _VERIFIED_FILES:
        return True
    if sha256_file(path) != expected_sha256:
        return False
    _record_verified(path, expected_sha256, expected_size)
    return True


def _request(url: str, offset: int, *, include_auth: bool) -> urllib.request.Request:
    headers = {
        "Accept-Encoding": "identity",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "User-Agent": "ComfyColab-Triposplat/0.1",
    }
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and include_auth:
        headers["Authorization"] = f"Bearer {token}"
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.Request(url, headers=headers)


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
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    destination.parent.mkdir(parents=True, exist_ok=True)
    marker = destination.with_suffix(destination.suffix + ".sha256")
    partial = destination.with_suffix(destination.suffix + ".part")

    if force:
        _forget_verified(destination)
        destination.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    elif _verified(destination, expected_sha256, expected_size):
        return destination
    elif destination.exists():
        _forget_verified(destination)
        destination.unlink()
        marker.unlink(missing_ok=True)

    include_auth = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_size:
            partial.unlink(missing_ok=True)
            offset = 0
        try:
            with urllib.request.urlopen(
                _request(url, offset, include_auth=include_auth),
                timeout=120,
            ) as response:
                status = getattr(response, "status", 200) or 200
                resumed = offset > 0 and status == 206
                if offset and not resumed:
                    offset = 0
                content_length = response.headers.get("Content-Length")
                remaining = int(content_length) if content_length else None
                total = offset + remaining if remaining is not None else expected_size
                if remaining is not None and shutil.disk_usage(destination.parent).free < remaining:
                    raise DownloadError(
                        f"Not enough temporary disk space for {destination.name}: "
                        f"need {remaining} more bytes."
                    )

                completed = offset
                with partial.open("ab" if resumed else "wb") as output:
                    try:
                        while chunk := response.read(CHUNK_SIZE):
                            output.write(chunk)
                            completed += len(chunk)
                            if progress:
                                progress(completed, total)
                    except http.client.IncompleteRead as error:
                        if error.partial:
                            output.write(error.partial)
                            completed += len(error.partial)
                            if progress:
                                progress(completed, total)
                        raise DownloadError(
                            f"Connection closed early for {destination.name} "
                            f"at {completed} of {total or 'unknown'} bytes."
                        ) from error
                if completed != expected_size:
                    raise DownloadError(
                        f"Wrong byte count for {destination.name}: expected "
                        f"{expected_size}, received {completed}."
                    )

            actual_sha256 = sha256_file(partial)
            if actual_sha256 != expected_sha256:
                partial.unlink(missing_ok=True)
                raise DownloadError(
                    f"Checksum mismatch for {destination.name}: expected "
                    f"{expected_sha256}, received {actual_sha256}."
                )
            partial.replace(destination)
            _record_verified(destination, expected_sha256, expected_size)
            return destination
        except (OSError, http.client.IncompleteRead, urllib.error.URLError, DownloadError) as error:
            if (
                isinstance(error, urllib.error.HTTPError)
                and error.code == 416
                and partial.is_file()
                and partial.stat().st_size == expected_size
                and sha256_file(partial) == expected_sha256
            ):
                partial.replace(destination)
                _record_verified(destination, expected_sha256, expected_size)
                return destination
            if isinstance(error, urllib.error.HTTPError):
                if error.code == 416:
                    partial.unlink(missing_ok=True)
                if error.code not in RETRYABLE_HTTP_CODES:
                    raise DownloadError(
                        f"Unable to download {destination.name}: HTTP {error.code} "
                        f"is not retryable ({error.reason})."
                    ) from error
                if error.code in {401, 403} and include_auth:
                    include_auth = False
            last_error = error
            if attempt < attempts:
                time.sleep(min(2**attempt, 30))

    partial_note = f" Partial data was kept at {partial}." if partial.exists() else ""
    raise DownloadError(
        f"Unable to download {destination.name} after {attempts} attempts: "
        f"{last_error}.{partial_note}"
    )
