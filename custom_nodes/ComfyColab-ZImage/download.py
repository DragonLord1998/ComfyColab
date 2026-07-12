from __future__ import annotations

import hashlib
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable


ProgressCallback = Callable[[int, int | None], None]
CHUNK_SIZE = 4 * 1024 * 1024


class DownloadError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path, expected_sha256: str) -> bool:
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
    marker.write_text(f"{expected_sha256} {path.stat().st_size}\n", encoding="ascii")
    return True


def _request(url: str, offset: int) -> urllib.request.Request:
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "ComfyColab/0.1",
    }
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.Request(url, headers=headers)


def download_file(
    *,
    url: str,
    destination: Path,
    expected_sha256: str,
    force: bool = False,
    progress: ProgressCallback | None = None,
    attempts: int = 3,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    marker = destination.with_suffix(destination.suffix + ".sha256")
    partial = destination.with_suffix(destination.suffix + ".part")

    if force:
        destination.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    elif _verified(destination, expected_sha256):
        return destination
    elif destination.exists():
        destination.unlink()
        marker.unlink(missing_ok=True)

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            with urllib.request.urlopen(_request(url, offset), timeout=120) as response:
                status = getattr(response, "status", 200) or 200
                resumed = offset > 0 and status == 206
                if offset and not resumed:
                    offset = 0
                content_length = response.headers.get("Content-Length")
                remaining = int(content_length) if content_length else None
                total = offset + remaining if remaining is not None else None
                if remaining is not None and shutil.disk_usage(destination.parent).free < remaining:
                    raise DownloadError(
                        f"Not enough temporary disk space for {destination.name}: "
                        f"need {remaining} more bytes."
                    )

                mode = "ab" if resumed else "wb"
                completed = offset
                with partial.open(mode) as output:
                    while chunk := response.read(CHUNK_SIZE):
                        output.write(chunk)
                        completed += len(chunk)
                        if progress:
                            progress(completed, total)

            actual_sha256 = sha256_file(partial)
            if actual_sha256 != expected_sha256:
                partial.unlink(missing_ok=True)
                raise DownloadError(
                    f"Checksum mismatch for {destination.name}: expected "
                    f"{expected_sha256}, received {actual_sha256}."
                )
            partial.replace(destination)
            marker.write_text(
                f"{expected_sha256} {destination.stat().st_size}\n",
                encoding="ascii",
            )
            return destination
        except (OSError, urllib.error.URLError, DownloadError) as error:
            if (
                isinstance(error, urllib.error.HTTPError)
                and error.code == 416
                and partial.is_file()
                and sha256_file(partial) == expected_sha256
            ):
                partial.replace(destination)
                marker.write_text(
                    f"{expected_sha256} {destination.stat().st_size}\n",
                    encoding="ascii",
                )
                return destination
            last_error = error
            if attempt < attempts:
                time.sleep(2 ** (attempt - 1))

    raise DownloadError(f"Unable to download {destination.name}: {last_error}")
