"""Executed by google-colab-cli inside a temporary Colab runtime."""

from __future__ import annotations

import base64
import errno
import http.client
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import posixpath
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from pathlib import Path, PurePosixPath


CONFIG_B64 = "__COMFYCOLAB_CONFIG_B64__"
DEFAULT_CONFIG = {
    "repository_url": "https://github.com/DragonLord1998/ComfyColab.git",
    "repository_ref": "main",
    "port": 8188,
    "refresh": False,
    "colab_proxy": False,
}
CONFIG = (
    DEFAULT_CONFIG
    if CONFIG_B64.startswith("__COMFYCOLAB_CONFIG_")
    else json.loads(base64.b64decode(CONFIG_B64).decode("utf-8"))
)

CONTENT = Path("/content")
COMFY_DIR = CONTENT / "ComfyUI"
REPO_DIR = CONTENT / "ComfyColab"
ULTRASHAPE_DIR = CONTENT / "UltraShape-1.0"
PIXAL3D_DIR = CONTENT / "Pixal3D"
STATE_DIR = CONTENT / ".comfycolab"
STATE_FILE = STATE_DIR / "runtime.json"
COMFY_LOG = STATE_DIR / "comfyui.log"
TUNNEL_LOG = STATE_DIR / "cloudflared.log"
GGUF_DIR = COMFY_DIR / "custom_nodes" / "ComfyUI-GGUF"
TRELLIS_DIR = COMFY_DIR / "custom_nodes" / "ComfyUI-TRELLIS2"
GEOMETRY_DIR = COMFY_DIR / "custom_nodes" / "ComfyUI-GeometryPack"
NODE_TARGET = COMFY_DIR / "custom_nodes" / "ComfyColab-ZImage"
NODE_3D_TARGET = COMFY_DIR / "custom_nodes" / "ComfyColab-3D"
NODE_TRIPOSPLAT_TARGET = COMFY_DIR / "custom_nodes" / "ComfyColab-Triposplat"
READY_PREFIX = "COMFYCOLAB_READY="
COMFY_REF = "8b099de36acd81acd1afa3b5442951dc847e0a52"
GGUF_REF = "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
TRELLIS_REF = "9b878516f2dc2fd873f4f6cceadba403dd12d83e"
GEOMETRY_REF = "c67199de05705642258e727fa118f412877b4ebf"
ULTRASHAPE_REF = "5e8dcef05df101ab00ab6cd5fdd0ed0c74fbca66"
TRELLIS_PATCH_ID = "trellis2-strict-1536-birefnet-pin-metrics-v4"
TRELLIS_CATEGORY_PATCH_ID = "trellis2-advanced-categories-v1"
ULTRASHAPE_PATCH_ID = "ultrashape-inference-compat-v3"
COMBINED_CACHE_MANIFEST = "3d-g4-v2.json"
PIXAL3D_CACHE_MANIFEST = "pixal3d-g4-v1.json"
ULTRASHAPE_CUBVH_REF = "757b913bfbf19ed65e3a379d159391a8e29efa0f"
BIREFNET_MODEL_REF = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
PIXAL3D_REF = "cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af"
PIXAL3D_MODEL_REPO = "TencentARC/Pixal3D"
PIXAL3D_MODEL_REF = "0b31f9160aa400719af409098bff7936a932f726"
PIXAL3D_DINOV3_MODEL_REPO = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
PIXAL3D_DINOV3_MODEL_REF = "3c276edd87d6f6e569ff0c4400e086807d0f3881"
PIXAL3D_MOGE_MODEL_REPO = "Ruicheng/moge-2-vitl"
PIXAL3D_MOGE_MODEL_REF = "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"
PIXAL3D_MOGE_SOURCE_REF = "07444410f1e33f402353b99d6ccd26bd31e469e8"
PIXAL3D_NAF_REF = "37f2dfc180f2de53d98bd601109c0da0dd6b0f43"
PIXAL3D_NAF_REPO = "valeoai/NAF"
PIXAL3D_NAF_CHECKPOINT_SHA256 = (
    "c096c1ab2217a5c3ac136365f721685e2201379cb69d509cfb0261183847c98f"
)
PIXAL3D_UTILS3D_WHEEL = (
    "https://github.com/LDYang694/Storages/releases/download/"
    "20260430/utils3d-0.0.2-py3-none-any.whl"
)
PIXAL3D_WORKER_ENVIRONMENT = "pixal3d-worker"
PIXAL3D_WORKER_PROFILE = "g4-linux64-py31213-torch2110-cu128-sm120-pixal3d-v1"
PIXAL3D_ENVIRONMENT_REF = PIXAL3D_WORKER_PROFILE
PIXAL3D_PATCH_ID = "pixal3d-persistent-worker-v1"
PIXAL3D_NATTEN_PACKAGE = "natten==0.21.6+torch2110cu128"
PIXAL3D_NATTEN_WHEEL_INDEX = "https://whl.natten.org"
PIXAL3D_INFERENCE_REQUIREMENTS = (
    f"git+https://github.com/microsoft/MoGe.git@{PIXAL3D_MOGE_SOURCE_REF}",
    "pillow==12.0.0",
    "imageio==2.37.2",
    "imageio-ffmpeg==0.6.0",
    "tqdm==4.67.1",
    "easydict==1.13",
    "opencv-python-headless==4.12.0.88",
    "trimesh==4.10.1",
    "transformers==4.57.3",
    "zstandard==0.25.0",
    "kornia==0.8.2",
    "timm==1.0.22",
    "diffusers==0.37.1",
    "accelerate==1.13.0",
    "plyfile==1.1.3",
    "huggingface_hub>=0.36.0",
)
COMFY_ENV_VERSION = "0.3.89"
COMFY_ENV_CALL_TIMEOUT_SECONDS = 7200
COMFY_ENV_TIMEOUT_PATCH_ID = "comfy-env-call-timeout-v1"
TRIPOSPLAT_CORE_REQUIREMENTS = {
    "comfy_extras/nodes_triposplat.py": (
        "TripoSplatPreprocessImage",
        "TripoSplatConditioning",
        "TripoSplatSamplingPreview",
        "VAEDecodeTripoSplat",
    ),
    "comfy_extras/nodes_gaussian_splat.py": (
        "SplatToFile3D",
        "RenderSplat",
    ),
}
ULTRASHAPE_INFERENCE_REQUIREMENTS = (
    "accelerate==1.1.1",
    "diffusers==0.30.0",
    "omegaconf==2.3.0",
    "scikit-image==0.24.0",
)
TRELLIS_CACHE = {
    "profile": "g4-linux64-py31213-torch2110-cu128-sm120-glibc235-v1",
    "release_base": (
        "https://github.com/DragonLord1998/ComfyColab/releases/download/"
        "trellis2-cache-v1"
    ),
    "archive_sha256": "ce618e97c9326910490124eae19b8ce6958317726476757a71a368834be886d6",
    "pixi_toml_sha256": "4977680375788c2a3fda6f8b0db9ee6037c73099e2af1b87b7b59bf4514c3432",
    "pixi_lock_sha256": "d7012b83f004007abc5fd75891304daa1a6a4fdd242376f705700a221554cf27",
    "install_hash": "ee16059316dd3f784413fd5d5682d8723918f70de4c053501d7426cf1c25917b",
    "parts": [
        {
            "name": (
                "trellis2-cache-g4-linux64-py31213-torch2110-cu128-"
                "sm120-glibc235-v1.tar.zst.part-000"
            ),
            "bytes": 1992294400,
            "sha256": "f47f75c2bb480daa1a36f928a7f574f31e4c8948b221f06d06446a0005e03e14",
        },
        {
            "name": (
                "trellis2-cache-g4-linux64-py31213-torch2110-cu128-"
                "sm120-glibc235-v1.tar.zst.part-001"
            ),
            "bytes": 1992294400,
            "sha256": "9a43b3cba2dbe1bd8c412145c5a4773aab6f353c7bd123b25a13ba168d07bd1e",
        },
        {
            "name": (
                "trellis2-cache-g4-linux64-py31213-torch2110-cu128-"
                "sm120-glibc235-v1.tar.zst.part-002"
            ),
            "bytes": 1036858644,
            "sha256": "aaf7979b3f85e0e30c00ee12dd6db10978e8bd13e764077e130df21fd6bbd96b",
        },
    ],
}


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    print(f"[comfycolab] $ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def clone_or_update(url: str, destination: Path, ref: str = "main") -> None:
    if not (destination / ".git").is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--no-checkout",
                url,
                str(destination),
            ]
        )

    run(["git", "fetch", "origin", ref, "--depth", "1"], cwd=destination)
    run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=destination)


def git_commit(destination: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=destination,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_pinned_patch(repository: Path, specification_path: Path) -> str:
    """Apply a content-addressed patch only to its exact upstream revision."""
    specification = json.loads(specification_path.read_text(encoding="utf-8"))
    if specification.get("schema") != 1:
        raise RuntimeError(f"Unsupported patch schema: {specification_path}")
    patch_id = str(specification.get("patch_id", ""))
    expected_revision = str(specification.get("revision", ""))
    if not patch_id or not expected_revision:
        raise RuntimeError(f"Incomplete patch metadata: {specification_path}")
    actual_revision = git_commit(repository)
    if actual_revision != expected_revision:
        raise RuntimeError(
            f"Patch {patch_id} requires revision {expected_revision}, got {actual_revision}."
        )

    repository_root = repository.resolve()
    prepared: list[tuple[Path, str, int]] = []
    states: set[str] = set()
    files = specification.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"Patch {patch_id} has no file specifications.")
    for file_specification in files:
        if not isinstance(file_specification, dict):
            raise RuntimeError(f"Patch {patch_id} contains malformed file metadata.")
        relative_path = PurePosixPath(str(file_specification.get("path", "")))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"Patch {patch_id} contains an unsafe path: {relative_path}")
        path = (repository / Path(*relative_path.parts)).resolve()
        if path != repository_root and repository_root not in path.parents:
            raise RuntimeError(f"Patch {patch_id} escapes its repository: {relative_path}")
        before_sha256 = str(file_specification.get("before_sha256", ""))
        after_sha256 = str(file_specification.get("after_sha256", ""))
        actual_sha256 = sha256_file(path)
        if actual_sha256 == after_sha256:
            states.add("after")
            continue
        if actual_sha256 != before_sha256:
            raise RuntimeError(
                f"Patch {patch_id} refused unexpected content in {relative_path}: "
                f"expected {before_sha256}, got {actual_sha256}."
            )
        states.add("before")
        content = path.read_text(encoding="utf-8")
        replacements = file_specification.get("replacements")
        if not isinstance(replacements, list) or not replacements:
            raise RuntimeError(f"Patch {patch_id} has no replacements for {relative_path}.")
        for replacement in replacements:
            before_lines = replacement.get("before_lines")
            after_lines = replacement.get("after_lines")
            if not isinstance(before_lines, list) or not isinstance(after_lines, list):
                raise RuntimeError(f"Patch {patch_id} contains malformed replacement lines.")
            occurrences = replacement.get("occurrences", 1)
            if not isinstance(occurrences, int) or isinstance(occurrences, bool) or occurrences < 1:
                raise RuntimeError(f"Patch {patch_id} contains an invalid occurrence count.")
            before = "\n".join(str(line) for line in before_lines) + "\n"
            after = "\n".join(str(line) for line in after_lines)
            if after_lines:
                after += "\n"
            actual_occurrences = content.count(before) if before else 0
            if not before or actual_occurrences != occurrences:
                raise RuntimeError(
                    f"Patch {patch_id} expected {occurrences} match(es) in {relative_path}, "
                    f"found {actual_occurrences}."
                )
            content = content.replace(before, after, occurrences)
        if hashlib.sha256(content.encode("utf-8")).hexdigest() != after_sha256:
            raise RuntimeError(f"Patch {patch_id} produced an unexpected {relative_path} hash.")
        prepared.append((path, content, path.stat().st_mode))

    if states == {"after"}:
        return patch_id
    if states != {"before"}:
        raise RuntimeError(f"Patch {patch_id} found a partially patched repository.")
    for path, content, mode in prepared:
        temporary = path.with_suffix(path.suffix + ".comfycolab-patch")
        temporary.write_text(content, encoding="utf-8")
        temporary.chmod(mode)
        temporary.replace(path)
    return patch_id


def trellis_cache_compatible() -> bool:
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        return False
    if sys.version_info[:3] != (3, 12, 13) or platform.libc_ver() != ("glibc", "2.35"):
        return False
    try:
        import torch
    except ImportError:
        return False
    torch_version = torch.__version__
    cuda_version = torch.version.cuda or ""
    if torch_version != "2.11.0+cu128" or cuda_version != "12.8":
        return False
    if not torch.cuda.is_available():
        return False
    return (
        "RTX PRO 6000" in torch.cuda.get_device_name(0).upper()
        and torch.cuda.get_device_capability(0) == (12, 0)
    )


class CacheDownloadRetryableError(RuntimeError):
    """A cache transfer failure that is safe to retry."""


class CacheDownloadProgress:
    def __init__(self, parts: list[dict[str, object]], *, report_interval: float = 5.0):
        self.totals = {str(part["name"]): int(part["bytes"]) for part in parts}
        self.downloaded = dict.fromkeys(self.totals, 0)
        self.report_interval = report_interval
        self.started_at = time.monotonic()
        self.last_report_at = self.started_at - report_interval
        self.network_bytes = 0
        self.samples: deque[tuple[float, int]] = deque([(self.started_at, 0)])
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.reporter: threading.Thread | None = None

    def start(self) -> None:
        if self.reporter is not None:
            return
        self.report(force=True)
        self.reporter = threading.Thread(
            target=self._report_loop,
            name="trellis-cache-progress",
            daemon=True,
        )
        self.reporter.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.reporter is not None:
            self.reporter.join(timeout=self.report_interval + 1)
        self.report(force=True)

    def report(self, *, force: bool = False) -> None:
        with self.lock:
            now = time.monotonic()
            self._record_sample_locked(now)
            self._report_locked(now=now, force=force)

    def _report_loop(self) -> None:
        while not self.stop_event.wait(self.report_interval):
            self.report(force=True)

    def set_downloaded(self, name: str, byte_count: int, *, force: bool = False) -> None:
        with self.lock:
            self.downloaded[name] = max(0, min(byte_count, self.totals[name]))
            self._report_locked(force=force)

    def advance(self, name: str, byte_count: int) -> None:
        with self.lock:
            self.downloaded[name] = min(
                self.downloaded[name] + byte_count,
                self.totals[name],
            )
            self.network_bytes += byte_count
            now = time.monotonic()
            self._record_sample_locked(now)
            self._report_locked(now=now)

    def _record_sample_locked(self, now: float) -> None:
        self.samples.append((now, self.network_bytes))
        while len(self.samples) > 1 and now - self.samples[0][0] > 15:
            self.samples.popleft()

    def _report_locked(self, *, now: float | None = None, force: bool = False) -> None:
        now = time.monotonic() if now is None else now
        if not force and now - self.last_report_at < self.report_interval:
            return
        total = sum(self.totals.values())
        downloaded = sum(self.downloaded.values())
        oldest_time, oldest_bytes = self.samples[0]
        elapsed = now - oldest_time
        speed = (self.network_bytes - oldest_bytes) / elapsed if elapsed > 0 else 0.0
        remaining = max(0, total - downloaded)
        eta = remaining / speed if speed > 0 else None
        print(
            "[comfycolab] TRELLIS cache download: "
            f"{_format_bytes(downloaded)}/{_format_bytes(total)} "
            f"({downloaded / total * 100:.1f}%) | {_format_speed(speed)} | "
            f"ETA {_format_duration(eta)}",
            flush=True,
        )
        self.last_report_at = now


def _format_bytes(byte_count: int) -> str:
    return f"{byte_count / 1_000_000_000:.2f} GB"


def _format_speed(bytes_per_second: float) -> str:
    return f"{bytes_per_second / 1_000_000:.1f} MB/s"


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    rounded = max(0, int(seconds + 0.5))
    minutes, seconds = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def _response_header(response: object, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    return headers.get(name) if headers is not None else None


def _response_status(response: object) -> int | None:
    status = getattr(response, "status", None)
    if status is not None:
        return int(status)
    getcode = getattr(response, "getcode", None)
    value = getcode() if callable(getcode) else None
    return int(value) if value is not None else None


def _retryable_http_error(error: urllib.error.HTTPError) -> bool:
    return error.code in {408, 409, 425, 429} or 500 <= error.code <= 599


def _retryable_os_error(error: OSError) -> bool:
    return error.errno in {
        errno.ECONNABORTED,
        errno.ECONNRESET,
        errno.EHOSTUNREACH,
        errno.ENETDOWN,
        errno.ENETRESET,
        errno.ENETUNREACH,
        errno.EPIPE,
        errno.ETIMEDOUT,
    }


def download_cache_part(
    part: dict[str, object],
    destination: Path,
    progress: CacheDownloadProgress | None = None,
    *,
    max_attempts: int = 5,
    stall_timeout: float = 30,
) -> None:
    expected = part["sha256"]
    expected_size = int(part.get("bytes", 0))
    name = str(part.get("name", destination.name))
    if destination.is_file() and sha256_file(destination) == expected:
        if progress is not None:
            progress.set_downloaded(name, expected_size, force=True)
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    if expected_size and partial.is_file() and partial.stat().st_size > expected_size:
        partial.unlink()
    if progress is not None:
        progress.set_downloaded(name, partial.stat().st_size if partial.is_file() else 0)

    for attempt in range(1, max_attempts + 1):
        offset = partial.stat().st_size if partial.is_file() else 0
        if expected_size and offset > expected_size:
            partial.unlink()
            offset = 0
            if progress is not None:
                progress.set_downloaded(name, 0, force=True)
        if expected_size and offset == expected_size:
            if sha256_file(partial) == expected:
                partial.replace(destination)
                if progress is not None:
                    progress.set_downloaded(name, expected_size, force=True)
                return
            partial.unlink()
            offset = 0
            if progress is not None:
                progress.set_downloaded(name, 0, force=True)
        headers = {
            "Accept-Encoding": "identity",
            "User-Agent": "ComfyColab-TRELLIS-cache/1",
        }
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(str(part["url"]), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=stall_timeout) as response:
                status = _response_status(response)
                mode = "ab"
                if offset:
                    content_range = _response_header(response, "Content-Range")
                    if status == 206 and content_range and content_range.startswith(
                        f"bytes {offset}-"
                    ):
                        pass
                    elif status in {None, 200}:
                        offset = 0
                        mode = "wb"
                        if progress is not None:
                            progress.set_downloaded(name, 0, force=True)
                    else:
                        partial.unlink(missing_ok=True)
                        if progress is not None:
                            progress.set_downloaded(name, 0, force=True)
                        raise CacheDownloadRetryableError(
                            f"server rejected resume with HTTP {status}"
                        )
                with partial.open(mode) as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        if progress is not None:
                            progress.advance(name, len(chunk))

            actual_size = partial.stat().st_size
            if expected_size and actual_size != expected_size:
                raise CacheDownloadRetryableError(
                    f"size mismatch: expected {expected_size}, got {actual_size}"
                )
            actual = sha256_file(partial)
            if actual != expected:
                partial.unlink(missing_ok=True)
                if progress is not None:
                    progress.set_downloaded(name, 0, force=True)
                raise CacheDownloadRetryableError(
                    f"checksum mismatch: expected {expected}, got {actual}"
                )
            partial.replace(destination)
            if progress is not None:
                progress.set_downloaded(name, expected_size or actual_size, force=True)
            return
        except urllib.error.HTTPError as error:
            if not _retryable_http_error(error):
                raise RuntimeError(
                    f"TRELLIS cache download failed for {name}: HTTP {error.code}"
                ) from error
            failure: Exception = error
        except (
            CacheDownloadRetryableError,
            urllib.error.URLError,
            http.client.IncompleteRead,
            socket.timeout,
            TimeoutError,
            ConnectionError,
        ) as error:
            failure = error
        except OSError as error:
            if not _retryable_os_error(error):
                raise
            failure = error

        if attempt == max_attempts:
            raise RuntimeError(
                f"TRELLIS cache download failed for {name} after {max_attempts} "
                f"attempts: {failure}"
            ) from failure
        delay = min(2 ** attempt, 16)
        resume_at = partial.stat().st_size if partial.is_file() else 0
        print(
            f"[comfycolab] Retrying {name} after {failure} "
            f"(attempt {attempt + 1}/{max_attempts}, in {delay}s, "
            f"resume at {_format_bytes(resume_at)})...",
            flush=True,
        )
        time.sleep(delay)


def trellis_workspace_metadata_valid(
    workspace: Path,
    cache: dict[str, object] | None = None,
) -> bool:
    cache = TRELLIS_CACHE if cache is None else cache
    expected_files = {
        workspace / "pixi.toml": str(cache["pixi_toml_sha256"]),
        workspace / "pixi.lock": str(cache["pixi_lock_sha256"]),
    }
    for path, expected in expected_files.items():
        if not path.is_file() or sha256_file(path) != expected:
            return False
    install_hash = workspace / "install.hash"
    if (
        not install_hash.is_file()
        or install_hash.read_text(encoding="utf-8").strip()
        != cache["install_hash"]
    ):
        return False
    envs = workspace / ".pixi" / "envs"
    return all(
        (envs / name / "bin" / "python").is_file()
        for name in ("trellis2-nodes", "geometrypack-nodes")
    )


def validate_trellis_archive(archive: Path) -> None:
    result = subprocess.run(
        ["tar", "--zstd", "-tvf", str(archive)],
        check=True,
        text=True,
        capture_output=True,
        timeout=180,
    )
    entries = result.stdout.splitlines()
    if not entries:
        raise RuntimeError("The TRELLIS cache archive is empty.")
    for entry in entries:
        fields = entry.split(maxsplit=5)
        if len(fields) != 6:
            raise RuntimeError(f"Malformed TRELLIS cache archive entry: {entry}")
        kind = entry[0]
        details = fields[5]
        if kind == "l":
            if " -> " not in details:
                raise RuntimeError(f"Malformed TRELLIS cache symlink: {entry}")
            member, target = details.rsplit(" -> ", 1)
        elif kind == "h":
            if " link to " not in details:
                raise RuntimeError(f"Malformed TRELLIS cache hard link: {entry}")
            member, target = details.rsplit(" link to ", 1)
        elif kind in {"-", "d"}:
            member, target = details, None
        else:
            raise RuntimeError(f"Unsupported TRELLIS cache archive entry: {entry}")

        if not safe_cache_member(member):
            raise RuntimeError(f"Unsafe TRELLIS cache archive member: {member}")
        if target is None:
            continue
        target_path = PurePosixPath(target)
        if target_path.is_absolute():
            final_root = PurePosixPath("/root/.ce")
            if target_path != final_root and final_root not in target_path.parents:
                raise RuntimeError(f"Unsafe TRELLIS cache link target: {target}")
        elif kind == "h":
            if not safe_cache_member(target):
                raise RuntimeError(f"Unsafe TRELLIS cache hard-link target: {target}")
        else:
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(member), target))
            if not safe_cache_member(resolved):
                raise RuntimeError(f"Unsafe TRELLIS cache symlink target: {target}")


def safe_cache_member(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(
        path.parts
        and not path.is_absolute()
        and ".." not in path.parts
        and path.parts[0] == ".ce"
    )


def validate_restored_links(workspace: Path) -> None:
    final_workspace = Path.home() / ".ce"
    for root, directories, files in os.walk(workspace, followlinks=False):
        for name in [*directories, *files]:
            path = Path(root) / name
            if not path.is_symlink():
                continue
            target = Path(os.readlink(path))
            if target.is_absolute():
                if target != final_workspace and final_workspace not in target.parents:
                    raise RuntimeError(f"Unsafe absolute symlink in TRELLIS cache: {path}")
                continue
            resolved = (path.parent / target).resolve(strict=False)
            if resolved != workspace and workspace not in resolved.parents:
                raise RuntimeError(f"Unsafe relative symlink in TRELLIS cache: {path}")


def combined_cache_specification() -> dict[str, object] | None:
    manifest_path = REPO_DIR / "cache" / COMBINED_CACHE_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if manifest.get("schema") != 1 or manifest.get("status") != "ready":
        return None
    sources = manifest.get("sources", {})
    patches = manifest.get("patches", {})
    if not isinstance(sources, dict) or not isinstance(patches, dict):
        raise RuntimeError("The combined 3D cache manifest has malformed source metadata.")
    expected_sources = {
        "comfy": COMFY_REF,
        "trellis": TRELLIS_REF,
        "geometry": GEOMETRY_REF,
        "ultrashape": ULTRASHAPE_REF,
        "cubvh": ULTRASHAPE_CUBVH_REF,
        "birefnet": BIREFNET_MODEL_REF,
        "comfyEnv": COMFY_ENV_VERSION,
    }
    expected_patches = {
        "trellis": TRELLIS_PATCH_ID,
        "ultrashape": ULTRASHAPE_PATCH_ID,
    }
    if sources != {**sources, **expected_sources} or patches != {**patches, **expected_patches}:
        raise RuntimeError("The combined 3D cache manifest does not match pinned sources.")
    archive = manifest.get("archive")
    inputs = manifest.get("inputs")
    if not isinstance(archive, dict) or not isinstance(inputs, dict):
        raise RuntimeError("The ready combined 3D cache manifest is incomplete.")
    parts = archive.get("parts")
    if not isinstance(parts, list) or not parts:
        raise RuntimeError("The ready combined 3D cache manifest has no archive parts.")
    release_tag = str(manifest.get("releaseTag", ""))
    if not release_tag:
        raise RuntimeError("The ready combined 3D cache manifest has no release tag.")
    return {
        "profile": str(manifest["profile"]),
        "release_base": (
            "https://github.com/DragonLord1998/ComfyColab/releases/download/"
            f"{release_tag}"
        ),
        "archive_sha256": str(archive["sha256"]),
        "pixi_toml_sha256": str(inputs["pixiTomlSha256"]),
        "pixi_lock_sha256": str(inputs["pixiLockSha256"]),
        "install_hash": str(inputs["installHash"]),
        "parts": parts,
    }


def expected_pixal3d_sources() -> dict[str, str]:
    return {
        "pixal3d": PIXAL3D_REF,
        "pixal3dModel": PIXAL3D_MODEL_REF,
        "dinov3": PIXAL3D_DINOV3_MODEL_REF,
        "mogeModel": PIXAL3D_MOGE_MODEL_REF,
        "mogeSource": PIXAL3D_MOGE_SOURCE_REF,
        "naf": PIXAL3D_NAF_REF,
        "nafCheckpoint": PIXAL3D_NAF_CHECKPOINT_SHA256,
        "utils3d": PIXAL3D_UTILS3D_WHEEL,
        "natten": PIXAL3D_NATTEN_PACKAGE,
        "environment": PIXAL3D_ENVIRONMENT_REF,
        "comfyEnv": COMFY_ENV_VERSION,
    }


def pixal3d_cache_specification() -> dict[str, object] | None:
    manifest_path = REPO_DIR / "cache" / PIXAL3D_CACHE_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if manifest.get("schema") != 1:
        raise RuntimeError("The Pixal3D cache manifest has an unsupported schema.")
    if manifest.get("profile") != PIXAL3D_WORKER_PROFILE:
        raise RuntimeError("The Pixal3D cache manifest does not match the worker profile.")
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("The Pixal3D cache manifest has malformed source metadata.")
    expected_sources = expected_pixal3d_sources()
    if sources != {**sources, **expected_sources}:
        raise RuntimeError("The Pixal3D cache manifest does not match pinned sources.")
    if manifest.get("status") != "ready":
        print(
            "[comfycolab] Pixal3D cache is not ready; using source install "
            f"({manifest.get('status', 'unknown')}).",
            flush=True,
        )
        return None
    archive = manifest.get("archive")
    inputs = manifest.get("inputs")
    if not isinstance(archive, dict) or not isinstance(inputs, dict):
        raise RuntimeError("The ready Pixal3D cache manifest is incomplete.")
    parts = archive.get("parts")
    if not isinstance(parts, list) or not parts:
        raise RuntimeError("The ready Pixal3D cache manifest has no archive parts.")
    release_tag = str(manifest.get("releaseTag", ""))
    if not release_tag:
        raise RuntimeError("The ready Pixal3D cache manifest has no release tag.")
    return {
        "profile": str(manifest["profile"]),
        "release_base": (
            "https://github.com/DragonLord1998/ComfyColab/releases/download/"
            f"{release_tag}"
        ),
        "archive_sha256": str(archive["sha256"]),
        "environment_toml_sha256": str(inputs["environmentTomlSha256"]),
        "install_hash": str(inputs["installHash"]),
        "parts": parts,
    }


def pixal3d_workspace() -> Path:
    return STATE_DIR / PIXAL3D_WORKER_ENVIRONMENT


def pixal3d_python(workspace: Path | None = None) -> Path:
    workspace = pixal3d_workspace() if workspace is None else workspace
    return workspace / "venv" / "bin" / "python"


def pixal3d_environment_toml() -> Path:
    return REPO_DIR / "worker" / "pixal3d" / "environment.toml"


def pixal3d_install_hash(environment_toml_sha256: str) -> str:
    payload = {
        "schema": 1,
        "profile": PIXAL3D_WORKER_PROFILE,
        "sources": expected_pixal3d_sources(),
        "environmentTomlSha256": environment_toml_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def pixal3d_workspace_metadata_valid(
    workspace: Path,
    cache: dict[str, object] | None = None,
) -> bool:
    environment_toml = pixal3d_environment_toml()
    if not environment_toml.is_file():
        return False
    environment_sha256 = sha256_file(environment_toml)
    expected_install_hash = (
        str(cache["install_hash"]) if cache is not None
        else pixal3d_install_hash(environment_sha256)
    )
    expected_environment_sha256 = (
        str(cache["environment_toml_sha256"]) if cache is not None
        else environment_sha256
    )
    metadata = {
        workspace / "environment.toml": expected_environment_sha256,
    }
    for path, expected in metadata.items():
        if not path.is_file() or sha256_file(path) != expected:
            return False
    install_hash = workspace / "install.hash"
    if (
        not install_hash.is_file()
        or install_hash.read_text(encoding="utf-8").strip() != expected_install_hash
    ):
        return False
    return pixal3d_python(workspace).is_file()


def safe_pixal3d_cache_member(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(
        path.parts
        and not path.is_absolute()
        and ".." not in path.parts
        and path.parts[0] == PIXAL3D_WORKER_ENVIRONMENT
    )


def validate_pixal3d_archive(archive: Path) -> None:
    result = subprocess.run(
        ["tar", "--zstd", "-tvf", str(archive)],
        check=True,
        text=True,
        capture_output=True,
        timeout=180,
    )
    entries = result.stdout.splitlines()
    if not entries:
        raise RuntimeError("The Pixal3D cache archive is empty.")
    for entry in entries:
        fields = entry.split(maxsplit=5)
        if len(fields) != 6:
            raise RuntimeError(f"Malformed Pixal3D cache archive entry: {entry}")
        kind = entry[0]
        details = fields[5]
        if kind == "l":
            if " -> " not in details:
                raise RuntimeError(f"Malformed Pixal3D cache symlink: {entry}")
            member, target = details.rsplit(" -> ", 1)
        elif kind == "h":
            if " link to " not in details:
                raise RuntimeError(f"Malformed Pixal3D cache hard link: {entry}")
            member, target = details.rsplit(" link to ", 1)
        elif kind in {"-", "d"}:
            member, target = details, None
        else:
            raise RuntimeError(f"Unsupported Pixal3D cache archive entry: {entry}")

        if not safe_pixal3d_cache_member(member):
            raise RuntimeError(f"Unsafe Pixal3D cache archive member: {member}")
        if target is None:
            continue
        target_path = PurePosixPath(target)
        if target_path.is_absolute():
            final_root = PurePosixPath(str(pixal3d_workspace()))
            if target_path != final_root and final_root not in target_path.parents:
                raise RuntimeError(f"Unsafe Pixal3D cache link target: {target}")
        elif kind == "h":
            if not safe_pixal3d_cache_member(target):
                raise RuntimeError(f"Unsafe Pixal3D cache hard-link target: {target}")
        else:
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(member), target))
            if not safe_pixal3d_cache_member(resolved):
                raise RuntimeError(f"Unsafe Pixal3D cache symlink target: {target}")


def validate_pixal3d_runtime(python: Path | None = None) -> None:
    python = pixal3d_python() if python is None else python
    probe = (
        "import importlib, importlib.util, json, sys, torch; "
        "assert torch.cuda.is_available(); "
        "assert torch.__version__ == '2.11.0+cu128'; "
        "assert torch.version.cuda == '12.8'; "
        "assert torch.cuda.get_device_capability() == (12, 0); "
        "x = torch.ones(4, device='cuda'); torch.cuda.synchronize(); "
        "assert x.sum().item() == 4.0; "
        "aliases = {'flex_gemm': ('flex_gemm_ap',), 'cumesh': ('cumesh_vb',), "
        "'o_voxel': ('o_voxel_vb_ap',)}; "
        "[(sys.modules.setdefault(name, importlib.import_module(next(candidate for candidate in candidates "
        "if importlib.util.find_spec(candidate) is not None))) if importlib.util.find_spec(name) is None "
        "else None) for name, candidates in aliases.items()]; "
        "import pixal3d, utils3d, moge, o_voxel, cumesh, flex_gemm, drtk, trimesh; "
        "from pixal3d.pipelines import Pixal3DImageTo3DPipeline; "
        "natten = importlib.import_module('natten'); "
        "assert getattr(natten, 'HAS_LIBNATTEN', False); "
        "export_glb = getattr(trimesh.Trimesh, 'export', None); "
        "assert callable(export_glb); "
        "print(json.dumps({'status': 'ok'}))"
    )
    subprocess.run(
        [str(python), "-c", probe],
        cwd=PIXAL3D_DIR,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )


def restore_pixal3d_cache(cache: dict[str, object] | None = None) -> bool:
    cache = pixal3d_cache_specification() if cache is None else cache
    if cache is None:
        return False
    parts = cache.get("parts", [])
    archive_sha256 = cache.get("archive_sha256", "")
    if not parts or not archive_sha256 or not trellis_cache_compatible():
        return False

    workspace = pixal3d_workspace()
    if pixal3d_workspace_metadata_valid(workspace, cache):
        try:
            validate_pixal3d_runtime(pixal3d_python(workspace))
        except Exception:
            pass
        else:
            print("[comfycolab] Reusing the verified Pixal3D worker environment.", flush=True)
            return True

    cache_dir = STATE_DIR / "environment-cache" / str(cache["profile"])
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[comfycolab] Restoring prebuilt Pixal3D cache "
            f"({cache['profile']}, {len(parts)} part(s))...",
            flush=True,
        )
        release_base = str(cache["release_base"])
        download_jobs: list[tuple[dict[str, str], Path]] = []
        for index, configured_part in enumerate(parts):
            part = dict(configured_part)
            part["url"] = f"{release_base}/{part['name']}"
            download_jobs.append((part, cache_dir / f"part-{index:03d}"))
        progress = CacheDownloadProgress([part for part, _ in download_jobs])
        progress.start()
        try:
            with ThreadPoolExecutor(max_workers=len(download_jobs)) as executor:
                futures = [
                    executor.submit(download_cache_part, part, destination, progress)
                    for part, destination in download_jobs
                ]
                for future in futures:
                    future.result()
        finally:
            progress.stop()
        archive = cache_dir / "pixal3d-worker-cache.tar.zst"
        with archive.open("wb") as output:
            for _, part_path in download_jobs:
                with part_path.open("rb") as source:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        actual_archive_sha256 = sha256_file(archive)
        if actual_archive_sha256 != archive_sha256:
            raise RuntimeError(
                "Pixal3D cache checksum mismatch: "
                f"expected {archive_sha256}, got {actual_archive_sha256}"
            )
        if shutil.which("zstd") is None:
            run(["apt-get", "update", "-qq"])
            run(["apt-get", "install", "-y", "-qq", "zstd"])
        validate_pixal3d_archive(archive)
        staging = cache_dir / "restore"
        staging.mkdir()
        run(
            [
                "tar",
                "--zstd",
                "--no-same-owner",
                "--no-same-permissions",
                "-xf",
                str(archive),
                "-C",
                str(staging),
            ]
        )
        restored_workspace = staging / PIXAL3D_WORKER_ENVIRONMENT
        if not pixal3d_workspace_metadata_valid(restored_workspace, cache):
            raise RuntimeError("The restored Pixal3D cache metadata is incomplete.")
        backup = cache_dir / "previous-workspace"
        if workspace.exists():
            workspace.replace(backup)
        try:
            restored_workspace.replace(workspace)
            validate_pixal3d_runtime(pixal3d_python(workspace))
        except Exception:
            shutil.rmtree(workspace, ignore_errors=True)
            if backup.exists():
                backup.replace(workspace)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        shutil.rmtree(cache_dir, ignore_errors=True)
        raise
    else:
        shutil.rmtree(cache_dir, ignore_errors=True)
    print("[comfycolab] Prebuilt Pixal3D cache restored.", flush=True)
    return True


def install_pixal3d_source() -> str:
    workspace = pixal3d_workspace()
    environment_toml = pixal3d_environment_toml()
    if not environment_toml.is_file():
        if not PIXAL3D_DIR.exists():
            print(
                "[comfycolab] Pixal3D source is unavailable; skipping Pixal3D worker install.",
                flush=True,
            )
            return "unavailable"
        raise RuntimeError(f"Pixal3D environment manifest is missing: {environment_toml}")
    environment_sha256 = sha256_file(environment_toml)
    install_hash = pixal3d_install_hash(environment_sha256)
    if pixal3d_workspace_metadata_valid(workspace):
        try:
            validate_pixal3d_runtime(pixal3d_python(workspace))
        except Exception:
            pass
        else:
            print("[comfycolab] Reusing the source-installed Pixal3D worker environment.", flush=True)
            return "source-install"

    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    native_base_python = (
        Path.home() / ".ce" / ".pixi" / "envs" / "trellis2-nodes" / "bin" / "python"
    )
    if not native_base_python.is_file():
        raise RuntimeError(
            "The verified trellis2-nodes CUDA base is required before building the isolated "
            "Pixal3D worker environment."
        )
    run(
        [
            str(native_base_python),
            "-m",
            "venv",
            "--system-site-packages",
            str(workspace / "venv"),
        ]
    )
    python = pixal3d_python(workspace)
    run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([str(python), "-m", "pip", "install", *PIXAL3D_INFERENCE_REQUIREMENTS])
    run([str(python), "-m", "pip", "install", PIXAL3D_UTILS3D_WHEEL])
    natten_env = os.environ.copy()
    natten_env.setdefault("NATTEN_CUDA_ARCH", "120")
    natten_env.setdefault("NATTEN_N_WORKERS", str(max(1, os.cpu_count() or 1)))
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "-f",
            PIXAL3D_NATTEN_WHEEL_INDEX,
            PIXAL3D_NATTEN_PACKAGE,
        ],
        env=natten_env,
    )
    shutil.copy2(environment_toml, workspace / "environment.toml")
    (workspace / "install.hash").write_text(install_hash + "\n", encoding="utf-8")
    validate_pixal3d_runtime(python)
    return "source-install"


def install_pixal3d() -> str:
    try:
        if restore_pixal3d_cache():
            return PIXAL3D_WORKER_PROFILE
    except Exception as error:
        print(
            f"[comfycolab] Pixal3D cache restore failed ({error}); using source install.",
            flush=True,
        )
    return install_pixal3d_source()


def restore_trellis_cache(
    cache: dict[str, object] | None = None,
    *,
    label: str = "TRELLIS.2",
    validate_ultrashape: bool = False,
) -> bool:
    cache = TRELLIS_CACHE if cache is None else cache
    parts = cache.get("parts", [])
    archive_sha256 = cache.get("archive_sha256", "")
    if not parts or not archive_sha256 or not trellis_cache_compatible():
        return False

    workspace = Path.home() / ".ce"
    if trellis_workspace_metadata_valid(workspace, cache):
        try:
            validate_trellis_cache(workspace, validate_ultrashape=validate_ultrashape)
        except Exception:
            pass
        else:
            print(f"[comfycolab] Reusing the verified {label} environment.", flush=True)
            return True

    cache_dir = STATE_DIR / "environment-cache" / str(cache["profile"])
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[comfycolab] Restoring prebuilt {label} cache "
            f"({cache['profile']}, {len(parts)} part(s))...",
            flush=True,
        )
        release_base = str(cache["release_base"])
        download_jobs: list[tuple[dict[str, str], Path]] = []
        for index, configured_part in enumerate(parts):
            part = dict(configured_part)
            part["url"] = f"{release_base}/{part['name']}"
            download_jobs.append((part, cache_dir / f"part-{index:03d}"))
        progress = CacheDownloadProgress([part for part, _ in download_jobs])
        progress.start()
        try:
            with ThreadPoolExecutor(max_workers=len(download_jobs)) as executor:
                futures = [
                    executor.submit(download_cache_part, part, destination, progress)
                    for part, destination in download_jobs
                ]
                for future in futures:
                    future.result()
        finally:
            progress.stop()
        part_paths = [destination for _, destination in download_jobs]

        archive = cache_dir / "trellis2-cache.tar.zst"
        with archive.open("wb") as output:
            for part_path in part_paths:
                with part_path.open("rb") as source:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        actual_archive_sha256 = sha256_file(archive)
        if actual_archive_sha256 != archive_sha256:
            raise RuntimeError(
                "Combined TRELLIS cache checksum mismatch: "
                f"expected {archive_sha256}, got {actual_archive_sha256}"
            )

        if shutil.which("zstd") is None:
            run(["apt-get", "update", "-qq"])
            run(["apt-get", "install", "-y", "-qq", "zstd"])
        validate_trellis_archive(archive)

        staging = cache_dir / "restore"
        staging.mkdir()
        run(
            [
                "tar",
                "--zstd",
                "--no-same-owner",
                "--no-same-permissions",
                "-xf",
                str(archive),
                "-C",
                str(staging),
            ]
        )
        restored_workspace = staging / ".ce"
        if not trellis_workspace_metadata_valid(restored_workspace, cache):
            raise RuntimeError("The restored TRELLIS cache metadata is incomplete.")
        validate_restored_links(restored_workspace)

        backup = cache_dir / "previous-workspace"
        if workspace.exists():
            workspace.replace(backup)
        try:
            restored_workspace.replace(workspace)
            validate_trellis_cache(workspace, validate_ultrashape=validate_ultrashape)
        except Exception:
            shutil.rmtree(workspace, ignore_errors=True)
            if backup.exists():
                backup.replace(workspace)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        shutil.rmtree(cache_dir, ignore_errors=True)
        raise
    else:
        shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"[comfycolab] Prebuilt {label} cache restored.", flush=True)
    return True


def validate_trellis_cache(workspace: Path, *, validate_ultrashape: bool = False) -> None:
    envs = workspace / ".pixi" / "envs"
    probes = {
        "trellis2-nodes": (
            "import torch, cumesh_vb, drtk, flash_attn, flex_gemm_ap, "
            "o_voxel_vb_ap, sageattention; "
            "assert torch.__version__ == '2.11.0+cu128'; "
            "assert torch.version.cuda == '12.8'; "
            "assert torch.cuda.get_device_capability() == (12, 0); "
            "x = torch.ones(4, device='cuda'); torch.cuda.synchronize(); "
            "assert x.sum().item() == 4.0"
        ),
        "geometrypack-nodes": (
            "import torch, cumesh; "
            "assert torch.__version__ == '2.11.0+cu128'; "
            "assert torch.version.cuda == '12.8'; "
            "assert torch.cuda.get_device_capability() == (12, 0); "
            "x = torch.ones(4, device='cuda'); torch.cuda.synchronize(); "
            "assert x.sum().item() == 4.0"
        ),
    }
    for env_name, source in probes.items():
        python = envs / env_name / "bin" / "python"
        if not python.is_file():
            raise RuntimeError(f"The restored TRELLIS cache is missing {env_name}.")
        subprocess.run(
            [str(python), "-c", source],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
    if validate_ultrashape:
        validate_ultrashape_imports(
            envs / "trellis2-nodes" / "bin" / "python",
            require_sm120=True,
        )


def validate_ultrashape_imports(python: Path, *, require_sm120: bool = False) -> None:
    kernel_probe = ""
    if require_sm120:
        kernel_probe = (
            "; assert torch.cuda.get_device_capability() == (12, 0)"
            "; vertices = torch.tensor([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],"
            "[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], dtype=torch.float32)"
            "; faces = torch.tensor([[0,2,1],[0,3,2],[4,5,6],[4,6,7],"
            "[0,1,5],[0,5,4],[2,3,7],[2,7,6],[0,4,7],[0,7,3],"
            "[1,2,6],[1,6,5]], dtype=torch.int32)"
            "; bvh = cubvh.cuBVH(vertices, faces)"
            "; distance = bvh.unsigned_distance(torch.tensor([[0.,0.,0.]], device='cuda'))[0]"
            "; torch.cuda.synchronize()"
            "; assert distance.shape == (1,) and torch.isfinite(distance).all()"
        )
    subprocess.run(
        [
            str(python),
            "-c",
            (
                "import torch, cubvh; "
                "from ultrashape.pipelines import UltraShapePipeline; "
                "from ultrashape.surface_loaders import SharpEdgeSurfaceLoader"
                f"{kernel_probe}"
            ),
        ],
        cwd=ULTRASHAPE_DIR,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )


def restore_3d_environment_cache() -> str | None:
    try:
        combined = combined_cache_specification()
    except Exception as error:
        print(
            f"[comfycolab] Combined 3D cache manifest rejected ({error}); "
            "falling back to the TRELLIS.2 cache.",
            flush=True,
        )
        combined = None
    if combined is not None:
        try:
            if restore_trellis_cache(
                combined,
                label="TRELLIS.2 + UltraShape",
                validate_ultrashape=True,
            ):
                return str(combined["profile"])
        except Exception as error:
            print(
                f"[comfycolab] Combined 3D cache restore failed ({error}); "
                "falling back to the TRELLIS.2 cache.",
                flush=True,
            )
    if restore_trellis_cache():
        return str(TRELLIS_CACHE["profile"])
    return None


def install_ultrashape_overlay() -> None:
    python = Path.home() / ".ce" / ".pixi" / "envs" / "trellis2-nodes" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError(f"The shared trellis2-nodes Python is missing: {python}")
    run([str(python), "-m", "pip", "install", *ULTRASHAPE_INFERENCE_REQUIREMENTS])
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            f"git+https://github.com/ashawkey/cubvh.git@{ULTRASHAPE_CUBVH_REF}",
        ]
    )
    validate_trellis_cache(Path.home() / ".ce", validate_ultrashape=True)


def install_dependencies() -> str:
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=COMFY_DIR)
    gguf_requirements = GGUF_DIR / "requirements.txt"
    if gguf_requirements.exists():
        run([sys.executable, "-m", "pip", "install", "-r", str(gguf_requirements)])
    trellis_requirements = TRELLIS_DIR / "requirements.txt"
    if not trellis_requirements.is_file():
        raise RuntimeError(f"TRELLIS.2 requirements are missing: {trellis_requirements}")
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(trellis_requirements),
            "--upgrade",
        ]
    )
    try:
        cache_profile = restore_3d_environment_cache()
    except Exception as error:
        (Path.home() / ".ce" / "install.hash").unlink(missing_ok=True)
        print(
            f"[comfycolab] TRELLIS.2 cache restore failed ({error}); "
            "using the normal installer.",
            flush=True,
        )
        cache_profile = None
    run([sys.executable, "install.py"], cwd=TRELLIS_DIR)
    combined = combined_cache_specification()
    if combined is None or cache_profile != str(combined["profile"]):
        print(
            "[comfycolab] Installing the pinned UltraShape inference overlay into "
            "the shared trellis2-nodes environment...",
            flush=True,
        )
        install_ultrashape_overlay()
    patch_comfyenv_call_timeout()
    return cache_profile or "source-install"


def patch_comfyenv_call_timeout(
    metadata_path: Path | None = None,
    *,
    installed_version: str | None = None,
) -> str:
    """Make the pinned isolation timeout configurable without discarding caches."""

    version = installed_version or importlib.metadata.version("comfy-env")
    if version != COMFY_ENV_VERSION:
        raise RuntimeError(
            "Refusing to patch an unexpected comfy-env version: "
            f"expected {COMFY_ENV_VERSION}, got {version}."
        )
    if metadata_path is None:
        specification = importlib.util.find_spec("comfy_env.isolation.metadata")
        if specification is None or not specification.origin:
            raise RuntimeError("Unable to locate comfy-env isolation metadata.py.")
        metadata_path = Path(specification.origin)

    original_timeout = "                    timeout=600.0,\n"
    patched_timeout = (
        "                    timeout=float(os.environ.get("
        "\"COMFY_ENV_CALL_TIMEOUT\", \"600.0\")),\n"
    )
    original_cleanup = "            except (RuntimeError, ConnectionError):\n"
    patched_cleanup = (
        "            except (RuntimeError, ConnectionError, TimeoutError):\n"
    )
    content = metadata_path.read_text(encoding="utf-8")
    states = {
        "timeout": (
            content.count(original_timeout),
            content.count(patched_timeout),
        ),
        "cleanup": (
            content.count(original_cleanup),
            content.count(patched_cleanup),
        ),
    }
    if states == {"timeout": (0, 1), "cleanup": (0, 1)}:
        return COMFY_ENV_TIMEOUT_PATCH_ID
    if states != {"timeout": (1, 0), "cleanup": (1, 0)}:
        raise RuntimeError(
            "Refusing to patch drifted or partially patched comfy-env isolation metadata: "
            f"{states}."
        )

    patched = content.replace(original_timeout, patched_timeout, 1).replace(
        original_cleanup,
        patched_cleanup,
        1,
    )
    temporary = metadata_path.with_suffix(".py.comfycolab-patch")
    temporary.write_text(patched, encoding="utf-8")
    temporary.chmod(metadata_path.stat().st_mode)
    temporary.replace(metadata_path)
    print(
        "[comfycolab] Patched comfy-env isolated calls to honor "
        "COMFY_ENV_CALL_TIMEOUT while preserving resumable model caches.",
        flush=True,
    )
    return COMFY_ENV_TIMEOUT_PATCH_ID


def invalidate_comfyenv_metadata_cache(workspace: Path | None = None) -> list[Path]:
    """Force isolated node metadata to match the patched upstream sources."""

    workspace = workspace or (Path.home() / ".ce")
    removed: list[Path] = []
    for environment_name in ("trellis2-nodes", "geometrypack-nodes"):
        metadata = workspace / ".pixi" / "envs" / environment_name / ".metadata_cache.pkl"
        if metadata.is_file():
            metadata.unlink()
            removed.append(metadata)
    if removed:
        print(
            "[comfycolab] Invalidated isolated node metadata after applying pinned patches.",
            flush=True,
        )
    return removed


def install_node_pack() -> None:
    node_packs = (
        (REPO_DIR / "custom_nodes" / "ComfyColab-ZImage", NODE_TARGET),
        (REPO_DIR / "custom_nodes" / "ComfyColab-3D", NODE_3D_TARGET),
        (
            REPO_DIR / "custom_nodes" / "ComfyColab-Triposplat",
            NODE_TRIPOSPLAT_TARGET,
        ),
    )
    for source, target in node_packs:
        if not source.is_dir():
            raise RuntimeError(f"Node pack is missing from repository: {source}")
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.exists():
            shutil.rmtree(target)
        target.symlink_to(source, target_is_directory=True)


def validate_triposplat_core_support() -> None:
    """Fail before startup if the pinned ComfyUI checkout lacks native TripoSplat."""
    missing: list[str] = []
    for relative_path, required_symbols in TRIPOSPLAT_CORE_REQUIREMENTS.items():
        path = COMFY_DIR / relative_path
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            missing.append(relative_path)
            continue
        for symbol in required_symbols:
            if symbol not in source:
                missing.append(f"{relative_path}:{symbol}")
    if missing:
        raise RuntimeError(
            "The pinned ComfyUI checkout does not provide the native TripoSplat "
            "runtime required by ComfyColab-Triposplat. Missing: "
            + ", ".join(missing)
        )


def cloudflared_path() -> Path:
    existing = shutil.which("cloudflared")
    if existing:
        return Path(existing)

    machine = platform.machine().lower()
    architecture = "arm64" if machine in {"aarch64", "arm64"} else "amd64"
    destination = STATE_DIR / "cloudflared"
    url = (
        "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        f"cloudflared-linux-{architecture}"
    )
    print(f"[comfycolab] Downloading cloudflared ({architecture})...", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    destination.chmod(0o755)
    return destination


def pid_alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        waited, _ = os.waitpid(pid, os.WNOHANG)
        if waited == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def stop_managed_process(pid: object) -> None:
    if not pid_alive(pid):
        return
    process_id = int(pid)
    try:
        process_group = os.getpgid(process_id)
        os.killpg(process_group, signal.SIGTERM)
    except OSError:
        try:
            os.kill(process_id, signal.SIGTERM)
        except OSError:
            return
    for _ in range(20):
        if not pid_alive(pid):
            return
        time.sleep(0.25)
    try:
        os.killpg(os.getpgid(process_id), signal.SIGKILL)
    except OSError:
        try:
            os.kill(process_id, signal.SIGKILL)
        except OSError:
            pass
    for _ in range(20):
        if not pid_alive(pid):
            return
        time.sleep(0.1)
    raise RuntimeError(f"Managed process {process_id} did not terminate.")


def stop_started_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        process.wait()
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except OSError:
        process.terminate()
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except OSError:
        process.kill()
    process.wait(timeout=2)


def http_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/object_info", timeout=2) as response:
            return 200 <= response.status < 500
    except Exception:
        return False


def wait_for_comfy(port: int, process: subprocess.Popen[bytes], timeout: int = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = COMFY_LOG.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"ComfyUI exited during startup.\n{tail}")
        if http_ready(port):
            return
        time.sleep(1)
    raise TimeoutError(f"ComfyUI did not become ready on port {port} within {timeout}s.")


def wait_for_tunnel(process: subprocess.Popen[bytes], timeout: int = 60) -> str:
    pattern = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if TUNNEL_LOG.exists():
            content = TUNNEL_LOG.read_text(encoding="utf-8", errors="replace")
            if match := pattern.search(content):
                return match.group(0)
        if process.poll() is not None:
            tail = TUNNEL_LOG.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"cloudflared exited during startup.\n{tail}")
        time.sleep(0.5)
    raise TimeoutError("cloudflared did not publish a trycloudflare.com URL.")


def eval_colab_js(expression: str, timeout: int) -> object:
    from google.colab.output import eval_js

    return eval_js(expression, timeout_sec=timeout)


def probe_colab_proxy_url(url: str, timeout: int = 15) -> bool:
    base_url = json.dumps(url)
    expression = f"""
(async () => {{
  const baseUrl = {base_url};
  await fetch(new URL("system_stats", baseUrl), {{
    mode: "no-cors",
    credentials: "include",
    cache: "no-store",
  }});
  const socketUrl = new URL("ws", baseUrl);
  socketUrl.protocol = socketUrl.protocol === "https:" ? "wss:" : "ws:";
  socketUrl.searchParams.set("clientId", crypto.randomUUID());
  return await new Promise((resolve) => {{
    const socket = new WebSocket(socketUrl);
    const timer = setTimeout(() => {{
      socket.close();
      resolve(false);
    }}, 10000);
    socket.onopen = () => {{
      clearTimeout(timer);
      socket.close();
      resolve(true);
    }};
    socket.onerror = () => {{
      clearTimeout(timer);
      resolve(false);
    }};
  }});
}})()
""".strip()
    return eval_colab_js(expression, timeout) is True


def request_colab_proxy_url(
    port: int,
    timeout: int = 15,
    attempts: int = 3,
) -> str | None:
    if not bool(CONFIG.get("colab_proxy", False)):
        return None

    print(
        "[comfycolab] Requesting a private Google Colab proxy URL...",
        flush=True,
    )
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            expression = f"""
(async () => {{
  if (!google.colab.kernel.accessAllowed) {{
    throw new Error("Colab kernel proxy access is not allowed");
  }}
  const proxy = await google.colab.kernel.proxyPort({port});
  return new URL("/", proxy).toString();
}})()
""".strip()
            value = eval_colab_js(expression, timeout)
            if value is None:
                raise TimeoutError("The attached Colab page did not return a proxy URL.")
            if not isinstance(value, str):
                raise ValueError("Colab returned a non-string proxy value.")
            parsed = urlparse(value)
            hostname = (parsed.hostname or "").lower()
            if parsed.scheme != "https" or not (
                hostname == "googleusercontent.com"
                or hostname.endswith(".googleusercontent.com")
            ):
                raise ValueError("Colab returned an untrusted proxy URL.")
            if not probe_colab_proxy_url(value):
                raise RuntimeError("ComfyUI did not pass the proxy HTTP/WebSocket probe.")
            return value
        except ValueError as error:
            last_error = error
            break
        except Exception as error:
            last_error = error
            if attempt < attempts:
                print(
                    f"[comfycolab] Proxy handshake attempt {attempt} failed; retrying...",
                    flush=True,
                )
                time.sleep(2)

    print(
        f"[comfycolab] Colab proxy unavailable ({last_error}); using Cloudflare fallback.",
        flush=True,
    )
    return None


def load_state() -> dict[str, object]:
    try:
        value = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def save_state(payload: dict[str, object]) -> None:
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE_FILE)


def emit_ready(payload: dict[str, object]) -> None:
    print(READY_PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)


def main() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    port = int(CONFIG["port"])
    refresh = bool(CONFIG.get("refresh", False))
    colab_proxy = bool(CONFIG.get("colab_proxy", False))
    previous = load_state()

    reusable_comfy = (
        not refresh
        and http_ready(port)
        and pid_alive(previous.get("comfyPid"))
        and bool(previous.get("comfyUrl"))
    )
    reusable_tunnel = reusable_comfy and pid_alive(previous.get("tunnelPid"))
    if reusable_comfy:
        proxy_url = request_colab_proxy_url(port) if colab_proxy else None
        if reusable_tunnel or proxy_url:
            cloudflare_url = None
            if reusable_tunnel:
                cloudflare_url = str(
                    previous.get("cloudflareUrl") or previous["comfyUrl"]
                )
            previous.update(
                {
                    "comfyUrl": proxy_url or cloudflare_url,
                    "cloudflareUrl": cloudflare_url,
                    "colabProxyUrl": proxy_url,
                    "tunnelPid": previous.get("tunnelPid") if reusable_tunnel else None,
                }
            )
            save_state(previous)
            emit_ready(previous)
            return

    stop_managed_process(previous.get("tunnelPid"))
    stop_managed_process(previous.get("comfyPid"))
    STATE_FILE.unlink(missing_ok=True)
    if http_ready(port):
        raise RuntimeError(
            f"Port {port} is already occupied by a process not managed by ComfyColab."
        )

    clone_or_update("https://github.com/Comfy-Org/ComfyUI.git", COMFY_DIR, COMFY_REF)
    clone_or_update("https://github.com/city96/ComfyUI-GGUF.git", GGUF_DIR, GGUF_REF)
    clone_or_update(
        "https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2.git",
        TRELLIS_DIR,
        TRELLIS_REF,
    )
    clone_or_update(
        "https://github.com/PozzettiAndrea/ComfyUI-GeometryPack.git",
        GEOMETRY_DIR,
        GEOMETRY_REF,
    )
    clone_or_update(
        "https://github.com/PKU-YuanGroup/UltraShape-1.0.git",
        ULTRASHAPE_DIR,
        ULTRASHAPE_REF,
    )
    clone_or_update(
        "https://github.com/TencentARC/Pixal3D.git",
        PIXAL3D_DIR,
        PIXAL3D_REF,
    )
    clone_or_update(
        str(CONFIG["repository_url"]),
        REPO_DIR,
        str(CONFIG["repository_ref"]),
    )
    trellis_patch = apply_pinned_patch(
        TRELLIS_DIR,
        REPO_DIR / "patches" / "trellis2-no-1536-downgrade.json",
    )
    trellis_category_patch = apply_pinned_patch(
        TRELLIS_DIR,
        REPO_DIR / "patches" / "trellis2-advanced-categories.json",
    )
    ultrashape_patch = apply_pinned_patch(
        ULTRASHAPE_DIR,
        REPO_DIR / "patches" / "ultrashape-inference-only-imports.json",
    )
    validate_triposplat_core_support()
    install_node_pack()
    environment_cache_profile = install_dependencies()
    pixal3d_cache_profile = install_pixal3d()
    if pixal3d_cache_profile != "unavailable":
        validate_pixal3d_runtime()
    invalidate_comfyenv_metadata_cache()

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["COMFY_ENV_CALL_TIMEOUT"] = os.environ.get(
        "COMFY_ENV_CALL_TIMEOUT", str(COMFY_ENV_CALL_TIMEOUT_SECONDS)
    )
    environment["COMFYCOLAB_PIXAL3D_PYTHON"] = str(pixal3d_python())
    environment["COMFYCOLAB_PIXAL3D_SOURCE"] = str(PIXAL3D_DIR)
    environment["COMFYCOLAB_PIXAL3D_MODEL_REPO"] = PIXAL3D_MODEL_REPO
    environment["COMFYCOLAB_PIXAL3D_MODEL_REF"] = PIXAL3D_MODEL_REF
    environment["COMFYCOLAB_PIXAL3D_DINOV3_MODEL_REPO"] = PIXAL3D_DINOV3_MODEL_REPO
    environment["COMFYCOLAB_PIXAL3D_DINOV3_MODEL_REF"] = PIXAL3D_DINOV3_MODEL_REF
    environment["COMFYCOLAB_PIXAL3D_MOGE_MODEL_REPO"] = PIXAL3D_MOGE_MODEL_REPO
    environment["COMFYCOLAB_PIXAL3D_MOGE_MODEL_REF"] = PIXAL3D_MOGE_MODEL_REF
    environment["COMFYCOLAB_PIXAL3D_NAF_REF"] = PIXAL3D_NAF_REF
    environment["COMFYCOLAB_PIXAL3D_ENVIRONMENT_REF"] = PIXAL3D_ENVIRONMENT_REF
    comfy: subprocess.Popen[bytes] | None = None
    tunnel: subprocess.Popen[bytes] | None = None
    ready = False
    try:
        with COMFY_LOG.open("wb") as comfy_log:
            comfy = subprocess.Popen(
                [
                    sys.executable,
                    "main.py",
                    "--listen",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=COMFY_DIR,
                stdout=comfy_log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
        save_state({"status": "starting_comfy", "comfyPid": comfy.pid, "port": port})
        wait_for_comfy(port, comfy)

        proxy_url = request_colab_proxy_url(port) if colab_proxy else None
        cloudflare_url: str | None = None
        try:
            cloudflared = cloudflared_path()
            with TUNNEL_LOG.open("wb") as tunnel_log:
                tunnel = subprocess.Popen(
                    [
                        str(cloudflared),
                        "tunnel",
                        "--url",
                        f"http://127.0.0.1:{port}",
                        "--no-autoupdate",
                    ],
                    stdout=tunnel_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            save_state(
                {
                    "status": "starting_tunnel",
                    "comfyPid": comfy.pid,
                    "tunnelPid": tunnel.pid,
                    "port": port,
                }
            )
            cloudflare_url = wait_for_tunnel(tunnel)
        except Exception as error:
            if tunnel is not None:
                stop_started_process(tunnel)
                tunnel = None
            if not proxy_url:
                raise
            print(
                f"[comfycolab] Cloudflare fallback unavailable ({error}); "
                "continuing with the Colab proxy.",
                flush=True,
            )

        payload: dict[str, object] = {
            "status": "ready",
            "comfyUrl": proxy_url or cloudflare_url,
            "cloudflareUrl": cloudflare_url,
            "colabProxyUrl": proxy_url,
            "comfyPid": comfy.pid,
            "tunnelPid": tunnel.pid if tunnel is not None else None,
            "port": port,
            "storage": "temporary",
            "repositoryUrl": CONFIG["repository_url"],
            "repositoryRef": CONFIG["repository_ref"],
            "repositoryCommit": git_commit(REPO_DIR),
            "comfyCommit": git_commit(COMFY_DIR),
            "ggufCommit": git_commit(GGUF_DIR),
            "trellisCommit": git_commit(TRELLIS_DIR),
            "geometryCommit": git_commit(GEOMETRY_DIR),
            "ultrashapeCommit": git_commit(ULTRASHAPE_DIR),
            "pixal3dCommit": git_commit(PIXAL3D_DIR),
            "birefnetModelRef": BIREFNET_MODEL_REF,
            "triposplatCoreRef": COMFY_REF,
            "triposplatCoreReady": True,
            "pixal3dModelRef": PIXAL3D_MODEL_REF,
            "pixal3dDinov3ModelRef": PIXAL3D_DINOV3_MODEL_REF,
            "pixal3dMogeModelRef": PIXAL3D_MOGE_MODEL_REF,
            "pixal3dNafRef": PIXAL3D_NAF_REF,
            "pixal3dEnvironmentRef": PIXAL3D_ENVIRONMENT_REF,
            "trellisPatch": trellis_patch,
            "trellisCategoryPatch": trellis_category_patch,
            "ultrashapePatch": ultrashape_patch,
            "comfyEnvTimeoutPatch": COMFY_ENV_TIMEOUT_PATCH_ID,
            "isolatedCallTimeoutSeconds": int(environment["COMFY_ENV_CALL_TIMEOUT"]),
            "environmentCacheProfile": environment_cache_profile,
            "pixal3dCacheProfile": pixal3d_cache_profile,
        }
        save_state(payload)
        ready = True
        emit_ready(payload)
    finally:
        if not ready:
            if tunnel is not None:
                stop_started_process(tunnel)
            if comfy is not None:
                stop_started_process(comfy)
            STATE_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
