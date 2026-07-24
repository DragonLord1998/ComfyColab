from __future__ import annotations

import hashlib
import http.client
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


ProgressCallback = Callable[[int, Optional[int]], None]
CHUNK_SIZE = 4 * 1024 * 1024
DEFAULT_ATTEMPTS = 5
RETRYABLE_HTTP_CODES = frozenset({401, 403, 408, 416, 425, 429, 500, 502, 503, 504})


class DownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class HuggingFaceFile:
    repo_id: str
    revision: str
    filename: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def parse_huggingface_url(url: str) -> HuggingFaceFile | None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
        return None
    parts = [part for part in parsed.path.split("/") if part]
    try:
        resolve_index = parts.index("resolve")
    except ValueError:
        return None
    if resolve_index < 2 or len(parts) <= resolve_index + 2:
        return None
    repo_id = "/".join(parts[:resolve_index])
    revision = parts[resolve_index + 1]
    filename = "/".join(parts[resolve_index + 2 :])
    if not repo_id or not revision or not filename:
        return None
    return HuggingFaceFile(
        repo_id=urllib.parse.unquote(repo_id),
        revision=urllib.parse.unquote(revision),
        filename=urllib.parse.unquote(filename),
    )


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _request(url: str, offset: int, *, include_auth: bool, user_agent: str) -> urllib.request.Request:
    headers = {
        "Accept-Encoding": "identity",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "User-Agent": user_agent,
    }
    token = _token()
    if token and include_auth:
        headers["Authorization"] = f"Bearer {token}"
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.Request(url, headers=headers)


def _is_stale_token_error(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in {401, 403}
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) in {401, 403}


def _download_with_hub(
    *,
    hf_file: HuggingFaceFile,
    destination: Path,
    expected_sha256: str,
    expected_size: int | None,
    force: bool,
    include_auth: bool,
    progress: ProgressCallback | None,
) -> Path:
    # Set before importing huggingface_hub so hf-xet sees the intended mode.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    from huggingface_hub import hf_hub_download

    token: str | bool | None = _token() if include_auth else False
    staging = destination.parent / f".{destination.name}.hf-xet"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        cached = Path(
            hf_hub_download(
                repo_id=hf_file.repo_id,
                filename=hf_file.filename,
                revision=hf_file.revision,
                token=token,
                force_download=force,
                local_dir=str(staging),
            )
        )
        actual_size = cached.stat().st_size
        if expected_size is not None and actual_size != expected_size:
            raise DownloadError(
                f"Wrong byte count for {destination.name}: expected "
                f"{expected_size}, received {actual_size}."
            )
        actual_sha256 = sha256_file(cached)
        if actual_sha256 != expected_sha256:
            raise DownloadError(
                f"Checksum mismatch for {destination.name}: expected "
                f"{expected_sha256}, received {actual_sha256}."
            )
        if progress:
            progress(actual_size, actual_size)
        os.replace(cached, destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        if destination.is_file():
            shutil.rmtree(staging, ignore_errors=True)


def _download_with_urllib(
    *,
    url: str,
    destination: Path,
    expected_sha256: str,
    expected_size: int | None,
    progress: ProgressCallback | None,
    attempts: int,
    user_agent: str,
) -> Path:
    partial = destination.with_suffix(destination.suffix + ".part")
    include_auth = bool(_token())
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if expected_size is not None and offset > expected_size:
            partial.unlink(missing_ok=True)
            offset = 0
        try:
            with urllib.request.urlopen(
                _request(url, offset, include_auth=include_auth, user_agent=user_agent),
                timeout=120,
            ) as response:
                status = getattr(response, "status", 200) or 200
                resumed = offset > 0 and status == 206
                if offset and not resumed:
                    offset = 0
                content_length = response.headers.get("Content-Length")
                remaining = int(content_length) if content_length else None
                total = offset + remaining if remaining is not None else expected_size
                if (
                    remaining is not None
                    and shutil.disk_usage(destination.parent).free < remaining
                ):
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
                if expected_size is not None and completed != expected_size:
                    raise DownloadError(
                        f"Wrong byte count for {destination.name}: expected "
                        f"{expected_size}, received {completed}."
                    )
                if expected_size is None and total is not None and completed < total:
                    raise DownloadError(
                        f"Connection closed early for {destination.name} "
                        f"at {completed} of {total} bytes."
                    )

            actual_sha256 = sha256_file(partial)
            if actual_sha256 != expected_sha256:
                partial.unlink(missing_ok=True)
                raise DownloadError(
                    f"Checksum mismatch for {destination.name}: expected "
                    f"{expected_sha256}, received {actual_sha256}."
                )
            partial.replace(destination)
            return destination
        except (
            OSError,
            http.client.IncompleteRead,
            urllib.error.URLError,
            DownloadError,
        ) as error:
            if (
                isinstance(error, urllib.error.HTTPError)
                and error.code == 416
                and partial.is_file()
                and (expected_size is None or partial.stat().st_size == expected_size)
                and sha256_file(partial) == expected_sha256
            ):
                partial.replace(destination)
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


def download_file(
    *,
    url: str,
    destination: Path,
    expected_sha256: str,
    expected_size: int | None = None,
    force: bool = False,
    progress: ProgressCallback | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
    user_agent: str = "ComfyColab/0.1",
    is_verified: Callable[[Path, str, int | None], bool],
    record_verified: Callable[[Path, str, int | None], None],
    forget_verified: Callable[[Path], None] | None = None,
) -> Path:
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    destination.parent.mkdir(parents=True, exist_ok=True)
    marker = destination.with_suffix(destination.suffix + ".sha256")
    partial = destination.with_suffix(destination.suffix + ".part")

    if force:
        if forget_verified:
            forget_verified(destination)
        destination.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    elif is_verified(destination, expected_sha256, expected_size):
        return destination
    elif destination.exists():
        if forget_verified:
            forget_verified(destination)
        destination.unlink()
        marker.unlink(missing_ok=True)

    hf_file = parse_huggingface_url(url)
    if hf_file is not None:
        include_auth = bool(_token())
        try:
            result = _download_with_hub(
                hf_file=hf_file,
                destination=destination,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
                force=force,
                include_auth=include_auth,
                progress=progress,
            )
            partial.unlink(missing_ok=True)
            record_verified(result, expected_sha256, expected_size)
            return result
        except Exception as error:
            if include_auth and _is_stale_token_error(error):
                try:
                    result = _download_with_hub(
                        hf_file=hf_file,
                        destination=destination,
                        expected_sha256=expected_sha256,
                        expected_size=expected_size,
                        force=force,
                        include_auth=False,
                        progress=progress,
                    )
                    partial.unlink(missing_ok=True)
                    record_verified(result, expected_sha256, expected_size)
                    return result
                except Exception:
                    pass
            # Keep urllib as a compatibility fallback for hub import/version/network
            # failures and for environments where a stale token only breaks hub.

    result = _download_with_urllib(
        url=url,
        destination=destination,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        progress=progress,
        attempts=attempts,
        user_agent=user_agent,
    )
    record_verified(result, expected_sha256, expected_size)
    return result
