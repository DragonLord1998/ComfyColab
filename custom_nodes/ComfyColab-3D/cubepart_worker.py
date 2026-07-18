from __future__ import annotations

import atexit
import json
import os
import queue
import re
import signal
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TextIO

READY_PREFIX = "COMFYCOLAB_CUBEPART_READY="
PROGRESS_PREFIX = "COMFYCOLAB_CUBEPART_PROGRESS="
RESULT_PREFIX = "COMFYCOLAB_CUBEPART_RESULT="
PROTOCOL_VERSION = 1
MAX_PARTS = 8
CUBEPART_SOURCE_REF = "3c6d06ddbef3160a1e1950cb13ab63dd12a61e50"
CUBEPART_MODEL_REPO = "Roblox/cubepart"
CUBEPART_MODEL_REF = "28431d124e77040fcaf34c0a71623ff61d35a6c0"
CUBEPART_ENVIRONMENT_REF = "g4-linux64-py31213-cubepart-v2"
CUBEPART_CODE_LICENSE = "Cube3D Research-Only RAIL-MS"
CUBEPART_WEIGHTS_LICENSE = "OpenRAIL / Cube3D Research-Only RAIL-MS"


def validate_volumetric_glb(*args, **kwargs):
    from .geometry_quality import validate_volumetric_glb as _validate_volumetric_glb

    return _validate_volumetric_glb(*args, **kwargs)


def normalize_part_names(part_names: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(part_names, str):
        values = [part.strip() for part in re.split(r"[\n,]+", part_names) if part.strip()]
    else:
        values = [str(part).strip() for part in part_names if str(part).strip()]
    if not values:
        raise ValueError("CubePart requires a non-empty ordered part_names schema")
    if len(values) > MAX_PARTS:
        raise ValueError(f"CubePart supports at most {MAX_PARTS} ordered part names")
    return tuple(values)


def _license_metadata() -> dict[str, str]:
    return {
        "source_ref": CUBEPART_SOURCE_REF,
        "source_license": CUBEPART_CODE_LICENSE,
        "weights_repo": CUBEPART_MODEL_REPO,
        "weights_ref": CUBEPART_MODEL_REF,
        "weights_license": CUBEPART_WEIGHTS_LICENSE,
        "required_acceptance": "accept_research_license",
    }


@dataclass(frozen=True)
class CubePartWorkerCommand:
    python: str
    worker_script: str
    source_dir: str
    weights_dir: str
    input_mesh: str
    output_dir: str
    request_id: str
    part_names: str | tuple[str, ...] | list[str]
    accept_research_license: bool
    seed: int = 0
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    resolution_base: float = 8.5
    scheduler: str = "dpm_solver"
    timeshift: float = 4.0
    num_samples: int = 128_000
    source_ref: str = CUBEPART_SOURCE_REF
    model_ref: str = CUBEPART_MODEL_REF
    environment_ref: str = CUBEPART_ENVIRONMENT_REF
    keep_worker_loaded: bool = True

    def server_argv(self) -> list[str]:
        return [
            self.python,
            self.worker_script,
            "--server",
            "--source-dir",
            self.source_dir,
            "--weights-dir",
            self.weights_dir,
        ]

    def argv(self) -> list[str]:
        request = build_cubepart_request(self)
        values = self.server_argv() + ["--one-shot"]
        for name in (
            "request_id",
            "input_mesh",
            "output_dir",
            "seed",
            "guidance_scale",
            "num_inference_steps",
            "resolution_base",
            "scheduler",
            "timeshift",
            "num_samples",
        ):
            values.extend((f"--{name.replace('_', '-')}", str(request[name])))
        values.extend(("--parts-json", json.dumps(request["part_names"])))
        values.extend(("--accept-research-license", "true"))
        return values


def build_cubepart_request(command: CubePartWorkerCommand) -> dict:
    if not command.accept_research_license:
        raise PermissionError(
            "CubePart uses research-only RAIL-MS code and weights; pass "
            "accept_research_license=True only after accepting those terms."
        )
    parts = normalize_part_names(command.part_names)
    if int(command.seed) < 0 or int(command.seed) > (2**31) - 1:
        raise ValueError("CubePart seed must be between 0 and 2147483647")
    if int(command.num_inference_steps) <= 0:
        raise ValueError("CubePart num_inference_steps must be positive")
    if int(command.num_samples) <= 0:
        raise ValueError("CubePart num_samples must be positive")
    return {
        "protocol": PROTOCOL_VERSION,
        "request_id": command.request_id,
        "input_mesh": command.input_mesh,
        "output_dir": command.output_dir,
        "part_names": list(parts),
        "seed": int(command.seed),
        "guidance_scale": float(command.guidance_scale),
        "num_inference_steps": int(command.num_inference_steps),
        "resolution_base": float(command.resolution_base),
        "scheduler": command.scheduler,
        "timeshift": float(command.timeshift),
        "num_samples": int(command.num_samples),
        "revisions": {
            "source": command.source_ref,
            "model": command.model_ref,
            "environment": command.environment_ref,
        },
        "license": _license_metadata(),
        "accept_research_license": True,
    }


def _reader(stream: TextIO, output: queue.Queue[str | None]) -> None:
    try:
        for line in iter(stream.readline, ""):
            output.put(line.rstrip("\n"))
    finally:
        output.put(None)


def _terminate_process(process, timeout: float = 5.0) -> None:
    try:
        if process.poll() is not None:
            return
    except (AttributeError, OSError):
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            terminate()
    wait = getattr(process, "wait", None)
    if not callable(wait):
        return
    try:
        wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError, PermissionError, OSError):
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()


def _cleanup(command: CubePartWorkerCommand) -> None:
    output = Path(command.output_dir)
    shutil.rmtree(output, ignore_errors=True)
    parent = output.parent
    if parent.exists():
        for path in parent.glob(f".{output.name}.*.partial"):
            shutil.rmtree(path, ignore_errors=True)


def _validate_manifest(output_dir: Path, expected_parts: tuple[str, ...]) -> dict:
    manifest_path = output_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("CubePart worker manifest is missing or invalid") from error
    if manifest.get("schema") != "comfycolab-cubepart-result-v1":
        raise RuntimeError("CubePart worker manifest has an unsupported schema")
    if tuple(manifest.get("part_names", ())) != expected_parts:
        raise RuntimeError("CubePart worker manifest part_names do not match the request")
    combined = output_dir / "parts.glb"
    if not combined.is_file():
        raise RuntimeError("CubePart worker omitted combined parts.glb")
    validate_volumetric_glb(combined, stage="CubePart combined output")
    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts:
        raise RuntimeError("CubePart worker manifest omitted part GLBs")
    if len(parts) != len(expected_parts):
        raise RuntimeError(
            "CubePart worker did not return one GLB for every requested part name"
        )
    for index, part in enumerate(parts):
        if not isinstance(part, dict) or part.get("name") != expected_parts[index]:
            raise RuntimeError("CubePart worker manifest part order is invalid")
        path = output_dir / str(part.get("file", ""))
        if path.resolve().parent != output_dir.resolve() or not path.is_file():
            raise RuntimeError("CubePart worker manifest references a missing part GLB")
        validate_volumetric_glb(path, stage=f"CubePart part {index}")
    return manifest


def validate_cubepart_output(
    output_dir: str | Path,
    part_names: str | Iterable[str],
) -> dict:
    return _validate_manifest(Path(output_dir), normalize_part_names(part_names))


class CubePartWorkerPool:
    """Serialize CubePart decomposition through one isolated worker process."""

    def __init__(
        self,
        *,
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        poll_interval: float = 0.1,
        startup_timeout: float = 120.0,
    ) -> None:
        self._popen_factory = popen_factory
        self._poll_interval = poll_interval
        self._startup_timeout = startup_timeout
        self._process = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._signature: tuple[str, ...] | None = None
        self._lock = threading.RLock()

    def _process_running(self) -> bool:
        if self._process is None:
            return False
        try:
            return self._process.poll() is None
        except (AttributeError, OSError):
            return True

    def _launch(self, command: CubePartWorkerCommand) -> None:
        self.close()
        self._lines = queue.Queue()
        argv = command.server_argv()
        env = os.environ.copy()
        env["COMFYCOLAB_CUBEPART_ENVIRONMENT_REF"] = command.environment_ref
        self._process = self._popen_factory(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=env,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self.close()
            raise RuntimeError("CubePart worker pipes are unavailable")
        threading.Thread(target=_reader, args=(self._process.stdout, self._lines), daemon=True).start()
        self._signature = tuple(argv)
        deadline = time.monotonic() + self._startup_timeout
        tail: list[str] = []
        while time.monotonic() < deadline:
            try:
                line = self._lines.get(timeout=self._poll_interval)
            except queue.Empty:
                if not self._process_running():
                    break
                continue
            if line is None:
                break
            tail = (tail + [line])[-40:]
            if line.startswith(READY_PREFIX):
                payload = json.loads(line.split("=", 1)[1])
                if int(payload.get("protocol", -1)) != PROTOCOL_VERSION:
                    self.close()
                    raise RuntimeError("CubePart worker protocol version mismatch")
                return
        self.close()
        raise RuntimeError(
            "CubePart worker failed to become ready" + (f": {' | '.join(tail)}" if tail else "")
        )

    def _ensure_process(self, command: CubePartWorkerCommand) -> None:
        signature = tuple(command.server_argv())
        if not self._process_running() or self._signature != signature:
            self._launch(command)

    def run(
        self,
        command: CubePartWorkerCommand,
        *,
        is_cancelled: Callable[[], bool] = lambda: False,
        on_progress: Callable[[dict], None] = lambda _event: None,
    ) -> dict:
        with self._lock:
            _cleanup(command)
            try:
                request = build_cubepart_request(command)
                expected_parts = tuple(request["part_names"])
                self._ensure_process(command)
                process = self._process
                if process is None or process.stdin is None:
                    raise RuntimeError("CubePart worker is unavailable")
                process.stdin.write(json.dumps(request, sort_keys=True) + "\n")
                process.stdin.flush()
                tail: list[str] = []
                while True:
                    if is_cancelled():
                        self.close()
                        raise InterruptedError("CubePart decomposition was cancelled")
                    try:
                        line = self._lines.get(timeout=self._poll_interval)
                    except queue.Empty:
                        if not self._process_running():
                            raise RuntimeError(
                                "CubePart worker exited before returning a matching result"
                                + (f": {' | '.join(tail)}" if tail else "")
                            )
                        continue
                    if line is None:
                        if not self._process_running():
                            raise RuntimeError("CubePart worker output closed unexpectedly")
                        continue
                    tail = (tail + [line])[-40:]
                    if line.startswith(PROGRESS_PREFIX):
                        event = json.loads(line.split("=", 1)[1])
                        if event.get("request_id") == command.request_id:
                            on_progress(event)
                        continue
                    if not line.startswith(RESULT_PREFIX):
                        continue
                    result = json.loads(line.split("=", 1)[1])
                    if result.get("request_id") != command.request_id:
                        continue
                    if result.get("status") != "ok":
                        error_type = str(result.get("error_type") or "RuntimeError")
                        message = str(result.get("error") or "unknown worker failure")
                        raise RuntimeError(f"CubePart worker failed: {error_type}: {message}")
                    output_dir = Path(str(result.get("output_dir", "")))
                    if output_dir.resolve() != Path(command.output_dir).resolve():
                        raise RuntimeError("CubePart worker reported an unexpected output directory")
                    result["manifest"] = _validate_manifest(output_dir, expected_parts)
                    if not command.keep_worker_loaded:
                        self.close()
                    return result
            except BaseException:
                _cleanup(command)
                self.close()
                raise

    def close(self) -> None:
        process, self._process = self._process, None
        self._signature = None
        if process is not None:
            _terminate_process(process)


_GLOBAL_POOL: CubePartWorkerPool | None = None
_GLOBAL_POOL_LOCK = threading.Lock()


def global_cubepart_worker_pool() -> CubePartWorkerPool:
    global _GLOBAL_POOL
    with _GLOBAL_POOL_LOCK:
        if _GLOBAL_POOL is None:
            _GLOBAL_POOL = CubePartWorkerPool()
        return _GLOBAL_POOL


def _close_global_pool() -> None:
    if _GLOBAL_POOL is not None:
        _GLOBAL_POOL.close()


atexit.register(_close_global_pool)


__all__ = [
    "CubePartWorkerCommand",
    "CubePartWorkerPool",
    "build_cubepart_request",
    "global_cubepart_worker_pool",
    "normalize_part_names",
    "validate_cubepart_output",
]
