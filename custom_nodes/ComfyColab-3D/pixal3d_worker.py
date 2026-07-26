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
PIXAL3D_VIEW_ORDER = ("front", "back", "left", "right", "top", "bottom")
PIXAL3D_FUSION_STRATEGIES = ("directional_softmax", "average")
PIXAL3D_GEOMETRY_GUIDANCE = ("none", "vggt_omega_depth_conf")
PIXAL3D_GEOMETRY_FALLBACKS = ("strict", "weighted_mv")


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
    views: tuple[dict[str, str | float], ...] | None = None
    fusion_temperature: float = 2.0
    fusion_strategy: str = "directional_softmax"
    geometry_guidance: str = "none"
    geometry_fallback: str = "strict"
    geometry_requested: str = ""
    geometry_fallback_stage: str = ""
    geometry_fallback_reason: str = ""
    vggt_omega_source_dir: str = ""
    vggt_omega_checkpoint: str = ""
    vggt_omega_source_ref: str = ""
    vggt_omega_checkpoint_ref: str = ""
    vggt_omega_image_resolution: int = 512
    geometry_strength: float = 0.75
    confidence_exponent: float = 1.0
    depth_tolerance: float = 0.12
    occlusion_margin: float = 0.04
    occlusion_tau: float = 0.03
    geometry_floor: float = 0.05
    max_normalized_alignment_error: float = 0.35
    background_removal: str = "none"
    surface_point_cloud: str = ""
    surface_point_count: int = 65_536

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
        if self.surface_point_cloud:
            values.extend(
                (
                    "--surface-point-cloud",
                    self.surface_point_cloud,
                    "--surface-point-count",
                    str(int(self.surface_point_count)),
                )
            )
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
            ("--vggt-omega-source-dir", self.vggt_omega_source_dir),
            ("--vggt-omega-checkpoint", self.vggt_omega_checkpoint),
        ):
            if value:
                values.extend((name, value))
        return values


def build_pixal3d_request(command: Pixal3DWorkerCommand) -> dict:
    fov_degrees = float(command.camera_fov_degrees)
    if not math.isfinite(fov_degrees) or fov_degrees < 0.0 or fov_degrees >= 179.0:
        raise ValueError("camera_fov_degrees must be 0 for automatic estimation or between 0 and 179")
    camera_angle_x = math.radians(fov_degrees) if fov_degrees > 0.0 else None
    request = {
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
        "background_removal": str(command.background_removal),
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
    if command.surface_point_cloud:
        if int(command.surface_point_count) < 4096:
            raise ValueError("surface_point_count must be at least 4096")
        request["surface_point_cloud"] = str(command.surface_point_cloud)
        request["surface_point_count"] = int(command.surface_point_count)
    if command.views is not None:
        request["views"] = _validate_pixal3d_views(command.views)
        request["fusion_temperature"] = _validate_fusion_temperature(
            command.fusion_temperature
        )
        if command.fusion_strategy not in PIXAL3D_FUSION_STRATEGIES:
            raise ValueError(
                "fusion_strategy must be directional_softmax or average"
            )
        request["fusion_strategy"] = command.fusion_strategy
        if command.geometry_guidance not in PIXAL3D_GEOMETRY_GUIDANCE:
            raise ValueError(
                f"geometry_guidance must be one of {PIXAL3D_GEOMETRY_GUIDANCE}"
            )
        request["geometry_guidance"] = command.geometry_guidance
        if command.geometry_guidance != "none":
            if command.geometry_fallback not in PIXAL3D_GEOMETRY_FALLBACKS:
                raise ValueError(
                    f"geometry_fallback must be one of {PIXAL3D_GEOMETRY_FALLBACKS}"
                )
            for name, value in (
                ("vggt_omega_source_dir", command.vggt_omega_source_dir),
                ("vggt_omega_checkpoint", command.vggt_omega_checkpoint),
                ("vggt_omega_source_ref", command.vggt_omega_source_ref),
                ("vggt_omega_checkpoint_ref", command.vggt_omega_checkpoint_ref),
            ):
                if not value:
                    raise ValueError(
                        f"{name} is required for VGGT-Omega geometry guidance"
                    )
            controls = {
                "geometry_strength": _bounded_float(
                    command.geometry_strength, "geometry_strength", 0.0, 1.0
                ),
                "confidence_exponent": _bounded_float(
                    command.confidence_exponent,
                    "confidence_exponent",
                    0.0,
                    4.0,
                ),
                "depth_tolerance": _bounded_float(
                    command.depth_tolerance,
                    "depth_tolerance",
                    1e-4,
                    1.0,
                ),
                "occlusion_margin": _bounded_float(
                    command.occlusion_margin,
                    "occlusion_margin",
                    0.0,
                    0.5,
                ),
                "occlusion_tau": _bounded_float(
                    command.occlusion_tau,
                    "occlusion_tau",
                    1e-4,
                    0.5,
                ),
                "geometry_floor": _bounded_float(
                    command.geometry_floor,
                    "geometry_floor",
                    0.0,
                    1.0,
                ),
                "max_normalized_alignment_error": _bounded_float(
                    command.max_normalized_alignment_error,
                    "max_normalized_alignment_error",
                    0.0,
                    1.0,
                ),
            }
            if int(command.vggt_omega_image_resolution) != 512:
                raise ValueError(
                    "The pinned VGGT-Omega checkpoint requires image resolution 512"
                )
            request.update(
                {
                    "geometry_guidance": command.geometry_guidance,
                    "geometry_fallback": command.geometry_fallback,
                    "vggt_omega_image_resolution": 512,
                    **controls,
                }
            )
            request["revisions"].update(
                {
                    "vggt_omega_source": command.vggt_omega_source_ref,
                    "vggt_omega_checkpoint": command.vggt_omega_checkpoint_ref,
                }
            )
        if command.geometry_fallback_reason:
            if command.geometry_guidance != "none":
                raise ValueError(
                    "A provisioning fallback cannot also enable geometry guidance"
                )
            if command.geometry_requested != "vggt_omega_depth_conf":
                raise ValueError(
                    "geometry_requested must identify vggt_omega_depth_conf"
                )
            if command.geometry_fallback != "weighted_mv":
                raise ValueError(
                    "A provisioning fallback requires geometry_fallback=weighted_mv"
                )
            if not command.geometry_fallback_stage:
                raise ValueError("A provisioning fallback requires its failed stage")
            request.update(
                {
                    "geometry_requested": command.geometry_requested,
                    "geometry_fallback": command.geometry_fallback,
                    "geometry_fallback_stage": command.geometry_fallback_stage,
                    "geometry_fallback_reason": command.geometry_fallback_reason,
                }
            )
    return request


def _validate_fusion_temperature(value: float) -> float:
    temperature = float(value)
    if not math.isfinite(temperature) or temperature <= 0.0 or temperature > 20.0:
        raise ValueError("fusion_temperature must be in (0, 20]")
    return temperature


def _bounded_float(
    value: float,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    result = float(value)
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _validate_pixal3d_views(
    views: tuple[dict[str, str | float], ...] | list[dict[str, str | float]]
) -> list[dict[str, str | float]]:
    if not 2 <= len(views) <= len(PIXAL3D_VIEW_ORDER):
        raise ValueError("Pixal3D multiview requires 2 to 6 ordered views")
    expected = PIXAL3D_VIEW_ORDER[: len(views)]
    serialized: list[dict[str, str | float]] = []
    seen: set[str] = set()
    for index, view in enumerate(views):
        name = str(view.get("name", ""))
        if name != expected[index]:
            raise ValueError(
                "Pixal3D multiview views must be ordered front, back, left, right, top, bottom"
            )
        if name in seen:
            raise ValueError(f"Duplicate Pixal3D multiview label: {name}")
        image_path = str(view.get("image_path", ""))
        if not image_path:
            raise ValueError(f"Pixal3D multiview view {name} omitted image_path")
        quality = float(view.get("quality", 1.0))
        if not math.isfinite(quality) or quality < 0.0 or quality > 10.0:
            raise ValueError("Pixal3D multiview view quality must be in [0, 10]")
        seen.add(name)
        serialized.append({"name": name, "image_path": image_path, "quality": quality})
    if serialized[0]["name"] != "front":
        raise ValueError("Pixal3D multiview requires front as the first view")
    return serialized


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
    artifacts = [
        output,
        metadata,
        output.with_suffix(output.suffix + ".partial"),
        output.with_name(output.stem + ".partial" + output.suffix),
        output.with_name("." + output.stem + ".partial" + output.suffix),
        metadata.with_suffix(metadata.suffix + ".partial"),
        metadata.with_name(metadata.stem + ".partial" + metadata.suffix),
        metadata.with_name("." + metadata.stem + ".partial" + metadata.suffix),
    ]
    if command.surface_point_cloud:
        point_cloud = Path(command.surface_point_cloud)
        artifacts.extend(
            [
                point_cloud,
                point_cloud.with_suffix(point_cloud.suffix + ".partial"),
                point_cloud.with_name(
                    "." + point_cloud.stem + ".partial" + point_cloud.suffix
                ),
            ]
        )
    return tuple(artifacts)


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
        env = os.environ.copy()
        env["COMFYCOLAB_PIXAL3D_ENVIRONMENT_REF"] = command.environment_ref
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
                        worker_traceback = str(result.get("traceback") or "").strip()
                        detail = (
                            f"\nWorker traceback:\n{worker_traceback}"
                            if worker_traceback
                            else ""
                        )
                        raise RuntimeError(
                            f"Pixal3D worker failed: {error_type}: {message}{detail}"
                        )
                    output = Path(str(result.get("output_mesh", "")))
                    metadata = Path(str(result.get("metadata_output", "")))
                    if output.resolve() != Path(command.output_mesh).resolve():
                        raise RuntimeError("Pixal3D worker reported an unexpected output path")
                    if metadata.resolve() != Path(command.metadata_output).resolve():
                        raise RuntimeError("Pixal3D worker reported an unexpected metadata path")
                    if command.surface_point_cloud:
                        point_cloud = Path(
                            str(result.get("surface_point_cloud", ""))
                        )
                        if point_cloud.resolve() != Path(
                            command.surface_point_cloud
                        ).resolve():
                            raise RuntimeError(
                                "Pixal3D worker reported an unexpected surface point-cloud path"
                            )
                        if not point_cloud.is_file() or point_cloud.stat().st_size <= 0:
                            raise RuntimeError(
                                "Pixal3D worker surface point cloud is missing"
                            )
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
