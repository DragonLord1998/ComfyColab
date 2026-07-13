"""Executed by google-colab-cli inside a temporary Colab runtime."""

from __future__ import annotations

import base64
import errno
import http.client
import hashlib
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
STATE_DIR = CONTENT / ".comfycolab"
STATE_FILE = STATE_DIR / "runtime.json"
COMFY_LOG = STATE_DIR / "comfyui.log"
TUNNEL_LOG = STATE_DIR / "cloudflared.log"
GGUF_DIR = COMFY_DIR / "custom_nodes" / "ComfyUI-GGUF"
TRELLIS_DIR = COMFY_DIR / "custom_nodes" / "ComfyUI-TRELLIS2"
GEOMETRY_DIR = COMFY_DIR / "custom_nodes" / "ComfyUI-GeometryPack"
NODE_TARGET = COMFY_DIR / "custom_nodes" / "ComfyColab-ZImage"
READY_PREFIX = "COMFYCOLAB_READY="
COMFY_REF = "8b099de36acd81acd1afa3b5442951dc847e0a52"
GGUF_REF = "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
TRELLIS_REF = "9b878516f2dc2fd873f4f6cceadba403dd12d83e"
GEOMETRY_REF = "c67199de05705642258e727fa118f412877b4ebf"
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


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print(f"[comfycolab] $ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


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


def trellis_workspace_metadata_valid(workspace: Path) -> bool:
    expected_files = {
        workspace / "pixi.toml": str(TRELLIS_CACHE["pixi_toml_sha256"]),
        workspace / "pixi.lock": str(TRELLIS_CACHE["pixi_lock_sha256"]),
    }
    for path, expected in expected_files.items():
        if not path.is_file() or sha256_file(path) != expected:
            return False
    install_hash = workspace / "install.hash"
    if (
        not install_hash.is_file()
        or install_hash.read_text(encoding="utf-8").strip()
        != TRELLIS_CACHE["install_hash"]
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


def restore_trellis_cache() -> bool:
    parts = TRELLIS_CACHE.get("parts", [])
    archive_sha256 = TRELLIS_CACHE.get("archive_sha256", "")
    if not parts or not archive_sha256 or not trellis_cache_compatible():
        return False

    workspace = Path.home() / ".ce"
    if trellis_workspace_metadata_valid(workspace):
        try:
            validate_trellis_cache(workspace)
        except Exception:
            pass
        else:
            print("[comfycolab] Reusing the verified TRELLIS.2 environment.", flush=True)
            return True

    cache_dir = STATE_DIR / "trellis-cache" / str(TRELLIS_CACHE["profile"])
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[comfycolab] Restoring prebuilt TRELLIS.2 cache "
            f"({TRELLIS_CACHE['profile']}, {len(parts)} part(s))...",
            flush=True,
        )
        release_base = str(TRELLIS_CACHE["release_base"])
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
        if not trellis_workspace_metadata_valid(restored_workspace):
            raise RuntimeError("The restored TRELLIS cache metadata is incomplete.")
        validate_restored_links(restored_workspace)

        backup = cache_dir / "previous-workspace"
        if workspace.exists():
            workspace.replace(backup)
        try:
            restored_workspace.replace(workspace)
            validate_trellis_cache(workspace)
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
    print("[comfycolab] Prebuilt TRELLIS.2 cache restored.", flush=True)
    return True


def validate_trellis_cache(workspace: Path) -> None:
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


def install_dependencies() -> None:
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
        restore_trellis_cache()
    except Exception as error:
        (Path.home() / ".ce" / "install.hash").unlink(missing_ok=True)
        print(
            f"[comfycolab] TRELLIS.2 cache restore failed ({error}); "
            "using the normal installer.",
            flush=True,
        )
    run([sys.executable, "install.py"], cwd=TRELLIS_DIR)


def install_node_pack() -> None:
    source = REPO_DIR / "custom_nodes" / "ComfyColab-ZImage"
    if not source.is_dir():
        raise RuntimeError(f"Node pack is missing from repository: {source}")

    if NODE_TARGET.is_symlink() or NODE_TARGET.is_file():
        NODE_TARGET.unlink()
    elif NODE_TARGET.exists():
        shutil.rmtree(NODE_TARGET)
    NODE_TARGET.symlink_to(source, target_is_directory=True)


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
        str(CONFIG["repository_url"]),
        REPO_DIR,
        str(CONFIG["repository_ref"]),
    )
    install_node_pack()
    install_dependencies()

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
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
