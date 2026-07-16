from __future__ import annotations

import atexit
import json
import math
import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

from .geometry_quality import validate_volumetric_glb


READY_PREFIX = "COMFYCOLAB_PIXAL3D_READY="
PROGRESS_PREFIX = "COMFYCOLAB_PIXAL3D_PROGRESS="
RESULT_PREFIX = "COMFYCOLAB_PIXAL3D_RESULT="
PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class Pixal3DWorkerCommand:
    python: str
    worker_script: str
    source_dir: str
    checkpoint_dir: str
    image_path: str
    output_mesh: str
    metadata_output: str
    request_id: str
    seed: int
    camera_fov_degrees: float
    texture_size: int
    pipeline_type: str
    max_tokens: int
    target_face_count: int = 1_000_000
    inference_steps: int = 12
    guidance_scale: float = 7.5
    dinov3_dir: str = ""
    moge_dir: str = ""
    naf_source_dir: str = ""
    naf_checkpoint: str = ""
    source_ref: str = ""
    model_ref: str = ""
    dinov3_ref: str = ""
    moge_ref: str = ""
    naf_ref: str = ""
    naf_checkpoint_ref: str = ""
    environment_ref: str = ""
    keep_worker_loaded: bool = True

    def argv(self) -> list[str]:
        """Return a complete one-request invocation for diagnostics and tests."""

        values = self.server_argv() + ["--one-shot"]
        request = build_pixal3d_request(self)
        for name, value in (
            ("--request-id", request["request_id"]),
            ("--image-path", request["image_path"]),
            ("--output-mesh", request["output_mesh"]),
            ("--metadata-output", request["metadata_output"]),
            ("--pipeline-type", request["pipeline_type"]),
            ("--seed", request["seed"]),
            ("--camera-fov-degrees", self.camera_fov_degrees),
            ("--sampling-steps", request["sampling_steps"]),
            ("--target-face-count", request["target_face_count"]),
            ("--texture-size", request["texture_size"]),
            ("--max-tokens", request["max_tokens"]),
        ):
            values.extend((name, str(value)))
        return values

    def server_argv(self) -> list[str]:
        values = [
            self.python,
            self.worker_script,
            "--server",
            "--source-dir",
            self.source_dir,
            "--checkpoint-dir",
            self.checkpoint_dir,
        ]
        for name, value in (
            ("--dinov3-dir", self.dinov3_dir),
            ("--moge-dir", self.moge_dir),
            ("--naf-source-dir", self.naf_source_dir),
            ("--naf-checkpoint", self.naf_checkpoint),
        ):
            if value:
                values.extend((name, value))
        return values


def build_pixal3d_request(command: Pixal3DWorkerCommand) -> dict:
    fov_degrees = float(command.camera_fov_degrees)
    if not math.isfinite(fov_degrees) or fov_degrees < 0.0 or fov_degrees >= 179.0:
        raise ValueError("camera_fov_degrees must be 0 for automatic estimation or between 0 and 179")
    camera_angle_x = math.radians(fov_degrees) if fov_degrees > 0.0 else None
    return {
        "protocol": PROTOCOL_VERSION,
        "request_id": command.request_id,
        "image_path": command.image_path,
        "output_mesh": command.output_mesh,
        "metadata_output": command.metadata_output,
        "seed": int(command.seed),
        "pipeline_type": command.pipeline_type,
        "sampling_steps": int(command.inference_steps),
        "guidance_scale": float(command.guidance_scale),
        "camera_mode": "manual" if camera_angle_x is not None else "moge",
        "camera_fov_radians": camera_angle_x,
        "camera_params": None if camera_angle_x is None else {"camera_angle_x": camera_angle_x},
        "target_face_count": int(command.target_face_count),
        "texture_size": int(command.texture_size),
        "max_tokens": int(command.max_tokens),
        "revisions": {
            "source": command.source_ref,
            "model": command.model_ref,
            "dinov3": command.dinov3_ref,
            "moge": command.moge_ref,
            "naf": command.naf_ref,
            "naf_checkpoint": command.naf_checkpoint_ref,
            "environment": command.environment_ref,
        },
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
            try:
                terminate()
            except OSError:
                pass
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
                try:
                    kill()
                except OSError:
                    pass
        try:
            wait(timeout=timeout)
        except (subprocess.TimeoutExpired, OSError):
            pass


def _owned_artifacts(command: Pixal3DWorkerCommand) -> tuple[Path, ...]:
    output = Path(command.output_mesh)
    metadata = Path(command.metadata_output)
    return (
        output,
        metadata,
        output.with_suffix(output.suffix + ".partial"),
        output.with_name(output.stem + ".partial" + output.suffix),
        output.with_name("." + output.stem + ".partial" + output.suffix),
        metadata.with_suffix(metadata.suffix + ".partial"),
        metadata.with_name(metadata.stem + ".partial" + metadata.suffix),
        metadata.with_name("." + metadata.stem + ".partial" + metadata.suffix),
    )


def _cleanup(command: Pixal3DWorkerCommand, *, include_final: bool = True) -> None:
    artifacts = _owned_artifacts(command)
    if not include_final:
        artifacts = artifacts[2:]
    for artifact in artifacts:
        artifact.unlink(missing_ok=True)


class Pixal3DWorkerPool:
    """Serialize requests through one long-lived isolated Pixal3D process."""

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
        self._reader_thread: threading.Thread | None = None
        self._signature: tuple[str, ...] | None = None
        self._lock = threading.RLock()

    def _process_running(self) -> bool:
        if self._process is None:
            return False
        try:
            return self._process.poll() is None
        except (AttributeError, OSError):
            return True

    def _launch(self, command: Pixal3DWorkerCommand) -> None:
        self.close()
        self._lines = queue.Queue()
        argv = command.server_argv()
        self._process = self._popen_factory(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self.close()
            raise RuntimeError("Pixal3D worker pipes are unavailable")
        self._reader_thread = threading.Thread(
            target=_reader, args=(self._process.stdout, self._lines), daemon=True
        )
        self._reader_thread.start()
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
                    raise RuntimeError("Pixal3D worker protocol version mismatch")
                return
        self.close()
        raise RuntimeError(
            "Pixal3D worker failed to become ready"
            + (f": {' | '.join(tail)}" if tail else "")
        )

    def _ensure_process(self, command: Pixal3DWorkerCommand) -> None:
        signature = tuple(command.server_argv())
        if not self._process_running() or self._signature != signature:
            self._launch(command)

    def run(
        self,
        command: Pixal3DWorkerCommand,
        *,
        is_cancelled: Callable[[], bool] = lambda: False,
        on_progress: Callable[[dict], None] = lambda _event: None,
    ) -> dict:
        with self._lock:
            _cleanup(command, include_final=False)
            try:
                self._ensure_process(command)
                process = self._process
                if process is None or process.stdin is None:
                    raise RuntimeError("Pixal3D worker is unavailable")
                request = build_pixal3d_request(command)
                process.stdin.write(json.dumps(request, sort_keys=True) + "\n")
                process.stdin.flush()
                tail: list[str] = []
                while True:
                    if is_cancelled():
                        self.close()
                        raise InterruptedError("Pixal3D generation was cancelled")
                    try:
                        line = self._lines.get(timeout=self._poll_interval)
                    except queue.Empty:
                        if not self._process_running():
                            raise RuntimeError(
                                "Pixal3D worker exited before returning a matching result"
                                + (f": {' | '.join(tail)}" if tail else "")
                            )
                        continue
                    if line is None:
                        if not self._process_running():
                            raise RuntimeError("Pixal3D worker output closed unexpectedly")
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
                        raise RuntimeError(f"Pixal3D worker failed: {error_type}: {message}")
                    output = Path(str(result.get("output_mesh", "")))
                    metadata = Path(str(result.get("metadata_output", "")))
                    if output.resolve() != Path(command.output_mesh).resolve():
                        raise RuntimeError("Pixal3D worker reported an unexpected output path")
                    if metadata.resolve() != Path(command.metadata_output).resolve():
                        raise RuntimeError("Pixal3D worker reported an unexpected metadata path")
                    validate_volumetric_glb(
                        output,
                        stage="Pixal3D worker output",
                        require_material=True,
                        require_texture=True,
                        require_uv=True,
                    )
                    if not metadata.is_file():
                        raise RuntimeError("Pixal3D worker metadata is missing")
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


_GLOBAL_POOL: Pixal3DWorkerPool | None = None
_GLOBAL_POOL_LOCK = threading.Lock()


def global_pixal3d_worker_pool() -> Pixal3DWorkerPool:
    global _GLOBAL_POOL
    with _GLOBAL_POOL_LOCK:
        if _GLOBAL_POOL is None:
            _GLOBAL_POOL = Pixal3DWorkerPool()
        return _GLOBAL_POOL


def _close_global_pool() -> None:
    if _GLOBAL_POOL is not None:
        _GLOBAL_POOL.close()


atexit.register(_close_global_pool)


__all__ = [
    "Pixal3DWorkerCommand",
    "Pixal3DWorkerPool",
    "build_pixal3d_request",
    "global_pixal3d_worker_pool",
]
