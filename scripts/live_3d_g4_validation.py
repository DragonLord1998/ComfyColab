#!/usr/bin/env python3
"""Stage and record live ComfyColab 3D validation runs on a Colab G4.

Cases normally use ComfyUI's HTTP API and can continue after a Colab client
disconnects. Fresh TRELLIS facade cases additionally use a short-lived
WebSocket connection to prove the five user-visible stages and both previews.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlparse


STATE_SCHEMA = "comfycolab-3d-live-run-state-v1"
CASE_SCHEMA = "comfycolab-3d-live-case-v1"
EVENT_PREFIX = "COMFYCOLAB_LIVE3D="
SHAPE_METRICS = re.compile(
    r"ComfyColab shape metrics:\s*(?P<tokens>\d+)\s+tokens\s+at\s+resolution\s+(?P<resolution>\d+)"
)
GEOMETRY_QUALITY_EVENT = re.compile(r"COMFYCOLAB_GEOMETRY_QUALITY=(?P<payload>\{[^\n]+\})")
ULTRASHAPE_SETTINGS_EVENT = re.compile(
    r"COMFYCOLAB_ULTRASHAPE_SETTINGS=(?P<payload>\{[^\n]+\})"
)
ULTRASHAPE_WORKER_SETTINGS_EVENT = re.compile(
    r"COMFYCOLAB_ULTRASHAPE_WORKER_SETTINGS=(?P<payload>\{[^\n]+\})"
)
TRIPOSPLAT_MODEL_REVISION = (
    "VAST-AI/TripoSplat@de3b99ab2627d565a8d5fc40f2db52557b82b974"
)
PIXAL3D_WORKER_RESULT_EVENT = re.compile(
    r"COMFYCOLAB_PIXAL3D_RESULT=(?P<payload>\{[^\n]+\})"
)
FIVE_STAGE_TEXTS = (
    "Stage 1/5 - Preparing models and input...",
    "Stage 2/5 - Generating 3D shape...",
    "Stage 3/5 - Building geometry preview...",
    "Stage 4/5 - Geometry preview ready; generating texture...",
    "Stage 5/5 - Baking PBR material and final GLB...",
    "Complete - 3D model ready",
)
BINARY_TEXT_EVENT = 3
DEFAULT_STATE_DIR = Path("/content/.comfycolab/live-3d-validation")
DEFAULT_COMFY_ROOT = Path("/content/ComfyUI")
DEFAULT_LOG = Path("/content/.comfycolab/comfyui.log")
DEFAULT_BASE_URL = "http://127.0.0.1:8188"
GEOMETRY_METRICS_SCHEMA = "comfycolab-geometry-metrics-v1"
GEOMETRY_QUALITY_PATH = (
    Path(__file__).resolve().parents[1]
    / "custom_nodes"
    / "ComfyColab-3D"
    / "geometry_quality.py"
)


def _geometry_quality_module():
    module_name = "comfycolab_live_geometry_quality"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    specification = importlib.util.spec_from_file_location(module_name, GEOMETRY_QUALITY_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Unable to load geometry quality contract: {GEOMETRY_QUALITY_PATH}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


_GEOMETRY_QUALITY = _geometry_quality_module()
INTRINSIC_RANK_RELATIVE_THRESHOLD = (
    getattr(_GEOMETRY_QUALITY, "SINGULAR_COLLAPSE_RATIO", None)
    or getattr(_GEOMETRY_QUALITY, "_SINGULAR_COLLAPSE_RATIO")
)
MIN_NONDEGENERATE_FACE_RATIO = 0.0


@dataclass(frozen=True)
class CaseSpec:
    name: str
    kind: str
    gate: str | None = None
    benchmark: str | None = None
    resolution: str | None = None
    actual_resolution: int | None = None
    quality: str = "512 — Fast"
    texture_size: int = 1024
    detail: str = "Fast"
    octree_resolution: int = 0
    retexture: bool = False
    require_textured: bool = False


CASES: dict[str, CaseSpec] = {
    "trellis_512": CaseSpec(
        "trellis_512", "trellis", "trellis_512_textured_glb", "trellis_512",
        "512", 512, "512 — Fast", 1024, require_textured=True,
    ),
    "trellis_1024_cascade": CaseSpec(
        "trellis_1024_cascade", "trellis", "trellis_1024_cascade_textured_glb",
        "trellis_1024_cascade", "1024_cascade", 1024, "1024 — Quality", 2048,
        require_textured=True,
    ),
    "trellis_1536_cascade": CaseSpec(
        "trellis_1536_cascade", "trellis", "trellis_1536_cascade_genuine",
        "trellis_1536_cascade", "1536_cascade", 1536, "1536 — Maximum", 4096,
        require_textured=True,
    ),
    "trellis_1536_default_cap": CaseSpec(
        "trellis_1536_default_cap", "strict1536",
        "trellis_1536_default_cap_no_downgrade", resolution="1536_cascade",
        actual_resolution=1536, quality="1536 — Maximum", texture_size=4096,
        require_textured=True,
    ),
    "ultrashape_384": CaseSpec(
        "ultrashape_384", "ultrashape", "ultrashape_384_refinement",
        "ultrashape_384", actual_resolution=384, detail="Fast", octree_resolution=384,
    ),
    "ultrashape_512": CaseSpec(
        "ultrashape_512", "ultrashape", "ultrashape_512_refinement",
        "ultrashape_512", actual_resolution=512, detail="Conservative", octree_resolution=0,
    ),
    "ultrashape_1024_run_1": CaseSpec(
        "ultrashape_1024_run_1", "ultrashape", "ultrashape_1024_run_1",
        "ultrashape_1024_run_1", actual_resolution=1024, detail="Ultra", octree_resolution=0,
    ),
    "ultrashape_1024_run_2": CaseSpec(
        "ultrashape_1024_run_2", "ultrashape", "ultrashape_1024_run_2",
        "ultrashape_1024_run_2", actual_resolution=1024, detail="Ultra", octree_resolution=0,
    ),
    "triposplat_fast_65k": CaseSpec(
        "triposplat_fast_65k", "triposplat", "triposplat_fast_65k_file3d_ply",
        "triposplat_fast_65k", actual_resolution=65536, quality="Fast — 65K",
    ),
    "pixal3d_cold_1024": CaseSpec(
        "pixal3d_cold_1024", "pixal3d", "pixal3d_cold_1024_textured_glb",
        "pixal3d_cold_1024", "1024_cascade", 1024, "1024 — Stable", 2048,
        require_textured=True,
    ),
    "pixal3d_object_auto_1024": CaseSpec(
        "pixal3d_object_auto_1024", "pixal3d", "pixal3d_object_auto_1024",
        "pixal3d_object_auto_1024", "1024_cascade", 1024, "1024 — Stable", 2048,
        require_textured=True,
    ),
    "pixal3d_transparent_1024": CaseSpec(
        "pixal3d_transparent_1024", "pixal3d", "pixal3d_transparent_1024",
        "pixal3d_transparent_1024", "1024_cascade", 1024, "1024 — Stable", 2048,
        require_textured=True,
    ),
    "pixal3d_worker_reuse_1024": CaseSpec(
        "pixal3d_worker_reuse_1024", "pixal3d_reuse", "pixal3d_worker_reuse_1024",
        "pixal3d_worker_reuse_1024", "1024_cascade", 1024, "1024 — Stable", 2048,
        require_textured=True,
    ),
    "pixal3d_cache_hit_no_inference": CaseSpec(
        "pixal3d_cache_hit_no_inference", "pixal3d_cache", "pixal3d_cache_hit_no_inference",
        resolution="1024_cascade", actual_resolution=1024, quality="1024 — Stable",
        texture_size=2048, require_textured=True,
    ),
    "pixal3d_cancellation_cleanup": CaseSpec(
        "pixal3d_cancellation_cleanup", "pixal3d_cancel", "pixal3d_cancellation_cleanup",
        resolution="1024_cascade", actual_resolution=1024, quality="1024 — Stable",
        texture_size=2048,
    ),
    "pixal3d_preview_save_glb_reader": CaseSpec(
        "pixal3d_preview_save_glb_reader", "pixal3d", "pixal3d_preview_save_glb_reader",
        "pixal3d_preview_save_glb_reader", "1024_cascade", 1024, "1024 — Stable", 2048,
        require_textured=True,
    ),
    "pixal3d_1536_experimental": CaseSpec(
        "pixal3d_1536_experimental", "pixal3d", "pixal3d_1536_experimental",
        "pixal3d_1536_experimental", "1536_cascade", 1536, "1536 — Experimental", 4096,
        require_textured=True,
    ),
    "full_workflow_hard_surface": CaseSpec(
        "full_workflow_hard_surface", "full", "full_workflow_hard_surface",
        resolution="512", actual_resolution=512, detail="Conservative", octree_resolution=0,
        retexture=True, require_textured=True,
    ),
    "full_workflow_organic": CaseSpec(
        "full_workflow_organic", "full", "full_workflow_organic",
        resolution="512", actual_resolution=512, detail="Conservative", octree_resolution=0,
        retexture=True, require_textured=True,
    ),
    "full_workflow_thin": CaseSpec(
        "full_workflow_thin", "full", "full_workflow_thin",
        resolution="512", actual_resolution=512, detail="Conservative", octree_resolution=0,
        retexture=True, require_textured=True,
    ),
    "full_workflow_holed": CaseSpec(
        "full_workflow_holed", "full", "full_workflow_holed",
        resolution="512", actual_resolution=512, detail="Conservative", octree_resolution=0,
        retexture=True, require_textured=True,
    ),
    "full_workflow_transparent_background": CaseSpec(
        "full_workflow_transparent_background", "full",
        "full_workflow_transparent_background", resolution="512", actual_resolution=512,
        detail="Conservative", octree_resolution=0, retexture=True, require_textured=True,
    ),
    "cache_hit_no_inference": CaseSpec(
        "cache_hit_no_inference", "cache", "cache_hit_no_inference",
        resolution="512", actual_resolution=512, require_textured=True,
    ),
    "cancellation_cleanup": CaseSpec(
        "cancellation_cleanup", "cancel", "cancellation_cleanup",
        actual_resolution=512, detail="Conservative", octree_resolution=0,
    ),
    "advanced_trellis_workflow": CaseSpec(
        "advanced_trellis_workflow", "advanced", "advanced_trellis_workflow",
        resolution="512", actual_resolution=512, require_textured=True,
    ),
    "combined_environment_cuda_probes": CaseSpec(
        "combined_environment_cuda_probes", "probe", "combined_environment_cuda_probes",
    ),
}


def _collect_glb_names(value: Any) -> list[str]:
    names: list[str] = []
    if isinstance(value, str) and value.lower().endswith(".glb"):
        names.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            names.extend(_collect_glb_names(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            names.extend(_collect_glb_names(item))
    return names


class FiveStageVerifier:
    """Collect and validate native ComfyUI stage/progress/preview events."""

    def __init__(self, facade_node: str, preview_node: str = "90"):
        self.facade_node = str(facade_node)
        self.preview_node = str(preview_node)
        self.prompt_id: str | None = None
        self.sequence = 0
        self.stage_events: list[dict[str, Any]] = []
        self.progress_events: list[dict[str, Any]] = []
        self.preview_events: list[dict[str, Any]] = []

    def _next(self) -> int:
        self.sequence += 1
        return self.sequence

    def record_binary(self, payload: bytes) -> None:
        sequence = self._next()
        if len(payload) < 8 or struct.unpack(">I", payload[:4])[0] != BINARY_TEXT_EVENT:
            return
        node_length = struct.unpack(">I", payload[4:8])[0]
        text_start = 8 + node_length
        if text_start > len(payload):
            return
        node_id = payload[8:text_start].decode("utf-8", "replace")
        if node_id != self.facade_node:
            return
        self.stage_events.append(
            {
                "sequence": sequence,
                "node": node_id,
                "text": payload[text_start:].decode("utf-8", "replace"),
            }
        )

    def record_json(self, payload: dict[str, Any]) -> str | None:
        sequence = self._next()
        event_type = payload.get("type")
        data = payload.get("data") or {}
        prompt_id = data.get("prompt_id")
        if self.prompt_id and prompt_id not in (None, self.prompt_id):
            return None
        if event_type == "progress" and str(data.get("node")) == self.facade_node:
            self._record_progress(sequence, data.get("value"), data.get("max"), "progress")
        elif event_type == "progress_state":
            state = (data.get("nodes") or {}).get(self.facade_node)
            if isinstance(state, dict):
                self._record_progress(
                    sequence,
                    state.get("value"),
                    state.get("max"),
                    "progress_state",
                )
        elif event_type == "executed":
            display_node = str(data.get("display_node") or data.get("node"))
            if display_node == self.preview_node:
                self.preview_events.append(
                    {
                        "sequence": sequence,
                        "node": str(data.get("node")),
                        "displayNode": display_node,
                        "glbs": _collect_glb_names(data.get("output")),
                    }
                )
        elif event_type in {"execution_error", "execution_interrupted"}:
            return "failed"
        elif event_type == "executing" and data.get("node") is None:
            return "completed"
        return None

    def _record_progress(self, sequence: int, value: Any, maximum: Any, source: str) -> None:
        try:
            numeric_value = int(float(value))
            numeric_max = int(float(maximum))
        except (TypeError, ValueError):
            return
        self.progress_events.append(
            {
                "sequence": sequence,
                "node": self.facade_node,
                "value": numeric_value,
                "max": numeric_max,
                "source": source,
            }
        )

    @staticmethod
    def _ordered_positions(events: list[dict[str, Any]], expected: list[Any], key: str) -> list[int]:
        positions: list[int] = []
        after = -1
        for value in expected:
            match = next(
                (
                    int(event["sequence"])
                    for event in events
                    if event.get(key) == value and int(event["sequence"]) > after
                ),
                None,
            )
            if match is None:
                return []
            positions.append(match)
            after = match
        return positions

    def verify(self) -> dict[str, Any]:
        stage_positions = self._ordered_positions(
            self.stage_events,
            list(FIVE_STAGE_TEXTS),
            "text",
        )
        progress = [event for event in self.progress_events if event.get("max") == 5]
        progress_positions = self._ordered_positions(progress, [1, 2, 3, 4, 5], "value")
        early = next(
            (event for event in self.preview_events if event["node"] != self.preview_node),
            None,
        )
        final = next(
            (
                event
                for event in self.preview_events
                if event["node"] == self.preview_node
                and (early is None or event["sequence"] > early["sequence"])
            ),
            None,
        )
        stage4 = stage_positions[3] if stage_positions else None
        complete = stage_positions[-1] if stage_positions else None
        progress_matches_stages = bool(stage_positions and progress_positions) and all(
            stage_positions[index + 1] < progress_positions[index]
            and (
                index == 4
                or progress_positions[index] < stage_positions[index + 2]
            )
            for index in range(5)
        )
        checks = {
            "allStageTextsInOrder": bool(stage_positions),
            "progressOneThroughFiveInOrder": bool(progress_positions),
            "progressMatchesStageTransitions": progress_matches_stages,
            "earlyPreviewObserved": early is not None,
            "earlyPreviewArtifactNamed": bool(early and early["glbs"]),
            "finalPreviewObserved": final is not None,
            "finalPreviewArtifactNamed": bool(final and final["glbs"]),
            "earlyPreviewBeforeTexture": bool(
                early is not None
                and len(progress_positions) == 5
                and stage4 is not None
                and progress_positions[1] < early["sequence"] < stage4
            ),
            "finalPreviewAfterComplete": bool(
                final is not None
                and complete is not None
                and len(progress_positions) == 5
                and final["sequence"] > max(complete, progress_positions[-1])
            ),
        }
        proof = {
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "stageEvents": self.stage_events,
            "progressEvents": progress,
            "previewEvents": self.preview_events,
        }
        if proof["status"] != "passed":
            failed = ", ".join(name for name, passed in checks.items() if not passed)
            raise RuntimeError(f"Five-stage progress/preview verification failed: {failed}")
        return proof


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Recorder:
    def __init__(self, state_dir: Path, case: str):
        self.state_dir = state_dir
        self.case = case
        self.case_dir = state_dir / "cases" / case
        self.case_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.case_dir / "events.jsonl"
        self.current_path = self.case_dir / "current.json"

    def event(self, stage: str, **details: Any) -> dict[str, Any]:
        event = {"at": utc_now(), "case": self.case, "stage": stage, **details}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(EVENT_PREFIX + json.dumps(event, sort_keys=True), flush=True)
        return event

    def status(self, status: str, **details: Any) -> None:
        atomic_json(
            self.current_path,
            {
                "schema": STATE_SCHEMA,
                "case": self.case,
                "status": status,
                "pid": os.getpid(),
                "updatedAt": utc_now(),
                **details,
            },
        )


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def json(self, method: str, path: str, payload: Any | None = None) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urlrequest.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        try:
            with urlrequest.urlopen(request, timeout=self.timeout) as response:
                data = response.read()
        except urlerror.HTTPError as exc:
            message = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"ComfyUI API {method} {path} returned {exc.code}: {message}") from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f"ComfyUI API is unavailable at {self.base_url}: {exc.reason}") from exc
        return json.loads(data) if data else {}

    def get(self, path: str) -> Any:
        return self.json("GET", path)

    def post(self, path: str, payload: Any | None = None) -> Any:
        return self.json("POST", path, {} if payload is None else payload)


class VramSampler:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def sample() -> int:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            return sum(int(float(line.strip())) for line in output.splitlines() if line.strip()) * 1024**2
        except (FileNotFoundError, subprocess.SubprocessError, ValueError):
            return 0

    def __enter__(self):
        def poll() -> None:
            while not self._stop.wait(self.interval):
                self.peak_bytes = max(self.peak_bytes, self.sample())

        self.peak_bytes = self.sample()
        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval * 3 + 1)
        self.peak_bytes = max(self.peak_bytes, self.sample())


def ensure_run(state_dir: Path) -> dict[str, Any]:
    run_path = state_dir / "run.json"
    run = read_json(run_path)
    if isinstance(run, dict) and run.get("schema") == STATE_SCHEMA:
        return run
    run = {"schema": STATE_SCHEMA, "runId": f"g4-{uuid.uuid4().hex[:16]}", "createdAt": utc_now()}
    atomic_json(run_path, run)
    return run


def copy_input_image(source: Path, comfy_root: Path, case: str) -> str:
    if not source.is_file():
        raise FileNotFoundError(f"Reference image is missing: {source}")
    digest = sha256_file(source)[:16]
    suffix = source.suffix.lower() if source.suffix else ".png"
    name = f"comfycolab-live3d/{case}-{digest}{suffix}"
    destination = comfy_root / "input" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or sha256_file(destination) != sha256_file(source):
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    return name


def trellis_inputs(spec: CaseSpec, image_node: str, args: argparse.Namespace, cache_mode: str) -> dict[str, Any]:
    return {
        "image": [image_node, 0],
        "quality": spec.quality,
        "seed": args.seed,
        "exact_resolution": spec.resolution or "512",
        "sampling_steps": args.sampling_steps,
        "target_face_count": args.target_face_count,
        "texture_size": args.texture_size or spec.texture_size,
        "max_tokens": args.max_tokens,
        "remove_background": args.remove_background,
        "cache_mode": cache_mode,
    }


def ultra_inputs(spec: CaseSpec, model_node: str, image_node: str, args: argparse.Namespace, cache_mode: str) -> dict[str, Any]:
    return {
        "model_3d": [model_node, 0],
        "reference_image": [image_node, 0],
        "detail": spec.detail,
        "seed": args.seed,
        "retexture": spec.retexture,
        "steps": args.steps,
        "num_latents": args.num_latents,
        "octree_resolution": args.octree_resolution or spec.octree_resolution,
        "decode_chunk_size": args.decode_chunk_size,
        "target_face_count": args.target_face_count,
        "texture_size": args.texture_size,
        "low_vram": args.low_vram,
        "cache_mode": cache_mode,
    }


def triposplat_inputs(spec: CaseSpec, image_node: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "image": [image_node, 0],
        "quality": spec.quality,
        "seed": args.seed,
        "remove_background": True,
        "sampling_steps": args.sampling_steps or 20,
        "guidance_scale": 3.0,
        "enable_sampling_preview": True,
        "output_format": "ply",
    }


def pixal3d_inputs(spec: CaseSpec, image_node: str, args: argparse.Namespace, cache_mode: str) -> dict[str, Any]:
    return {
        "image": [image_node, 0],
        "quality": spec.quality,
        "seed": args.seed,
        "remove_background": args.remove_background,
        "camera_fov_degrees": getattr(args, "camera_fov_degrees", 0.0),
        "sampling_steps": args.sampling_steps,
        "target_face_count": args.target_face_count,
        "texture_size": args.texture_size or spec.texture_size,
        "max_tokens": args.max_tokens,
        "keep_worker_loaded": getattr(args, "keep_worker_loaded", True),
        "cache_mode": cache_mode,
    }


def add_preview_and_save(prompt: dict[str, Any], source: str, prefix: str) -> None:
    prompt["90"] = {"class_type": "Preview3D", "inputs": {"model_file": [source, 0]}}
    prompt["91"] = {
        "class_type": "SaveGLB",
        "inputs": {"mesh": [source, 0], "filename_prefix": prefix},
    }


def add_preview_and_file3d_save(
    prompt: dict[str, Any],
    source: str,
    output_index: int,
    prefix: str,
) -> None:
    prompt["90"] = {
        "class_type": "Preview3D",
        "inputs": {"model_file": [source, output_index]},
    }
    prompt["91"] = {
        "class_type": "SaveGLB",
        "inputs": {"mesh": [source, output_index], "filename_prefix": prefix},
    }


def build_prompt(
    spec: CaseSpec,
    args: argparse.Namespace,
    image_name: str,
    run_id: str,
    *,
    cache_mode: str | None = None,
) -> dict[str, Any]:
    cache_mode = cache_mode or args.cache_mode
    prompt: dict[str, Any] = {"1": {"class_type": "LoadImage", "inputs": {"image": image_name}}}
    output_node: str
    if spec.kind in {"trellis", "cache", "strict1536"}:
        prompt["2"] = {
            "class_type": "ComfyColabTrellisImageTo3D",
            "inputs": trellis_inputs(spec, "1", args, cache_mode),
        }
        output_node = "2"
    elif spec.kind in {"ultrashape", "cancel"}:
        if not args.model:
            raise ValueError(f"Case {spec.name} requires --model PATH")
        model = Path(args.model).resolve()
        if not model.is_file():
            raise FileNotFoundError(f"Input GLB is missing: {model}")
        prompt["2"] = {
            "class_type": "ComfyColab3DPathToFile3D",
            "inputs": {"glb_path": str(model), "delete_source": False},
        }
        prompt["3"] = {
            "class_type": "ComfyColabUltraShapeRefine",
            "inputs": ultra_inputs(spec, "2", "1", args, cache_mode),
        }
        output_node = "3"
    elif spec.kind == "full":
        prompt["2"] = {
            "class_type": "ComfyColabTrellisImageTo3D",
            "inputs": trellis_inputs(spec, "1", args, cache_mode),
        }
        prompt["3"] = {
            "class_type": "ComfyColabUltraShapeRefine",
            "inputs": ultra_inputs(spec, "2", "1", args, cache_mode),
        }
        output_node = "3"
    elif spec.kind == "triposplat":
        prompt["2"] = {
            "class_type": "ComfyColabTripoSplatImageToGaussianSplat",
            "inputs": triposplat_inputs(spec, "1", args),
        }
        output_node = "2"
    elif spec.kind in {"pixal3d", "pixal3d_cache", "pixal3d_cancel", "pixal3d_reuse"}:
        prompt["2"] = {
            "class_type": "ComfyColabPixal3DImageTo3D",
            "inputs": pixal3d_inputs(spec, "1", args, cache_mode),
        }
        output_node = "2"
    elif spec.kind == "advanced":
        prompt.update(build_advanced_nodes(args))
        output_node = "9"
    else:
        raise ValueError(f"Case {spec.name} does not use a ComfyUI prompt")
    prefix = f"3d/validation/{run_id}-{spec.name}"
    if spec.kind == "triposplat":
        add_preview_and_file3d_save(prompt, output_node, 1, prefix)
    else:
        add_preview_and_save(prompt, output_node, prefix)
    return prompt


def build_advanced_nodes(args: argparse.Namespace) -> dict[str, Any]:
    """Build the pinned manual TRELLIS path without the public facade."""
    steps = args.sampling_steps or 12
    return {
        "2": {"class_type": "LoadTrellis2Models", "inputs": {"resolution": "512"}},
        "3": {"class_type": "Trellis2RemoveBackground", "inputs": {"image": ["1", 0], "low_vram": True}},
        "4": {"class_type": "Trellis2GetConditioning", "inputs": {
            "model_config": ["2", 0], "image": ["3", 0], "mask": ["3", 1], "background_color": "black",
        }},
        "5": {"class_type": "Trellis2ImageToShape", "inputs": {
            "model_config": ["2", 0], "conditioning": ["4", 0], "seed": args.seed,
            "ss_sampling_steps": steps, "shape_sampling_steps": steps, "max_tokens": args.max_tokens,
        }},
        "6": {"class_type": "Trellis2ShapeToTexturedMesh", "inputs": {
            "model_config": ["2", 0], "conditioning": ["4", 0], "shape_slat": ["5", 1],
            "subs": ["5", 2], "seed": args.seed, "tex_sampling_steps": steps,
        }},
        "7": {"class_type": "Trellis2ProcessMesh", "inputs": {
            "trimesh": ["5", 0], "target_face_count": args.target_face_count or 100000,
            "floater_threshold": 0.001, "weld_vertices": True, "remesh": "off",
            "remesh.fill_holes": True, "remesh.fill_holes_perimeter": 0.03,
        }},
        "8": {"class_type": "Trellis2RasterizePBR", "inputs": {
            "trimesh": ["7", 0], "voxelgrid": ["6", 0], "texture_size": args.texture_size or 1024,
            "original_mesh": ["5", 0],
        }},
        "9": {"class_type": "ComfyColab3DTrimeshToFile3D", "inputs": {
            "trimesh": ["8", 0], "cache_stage": "trellis",
            "cache_key": uuid.uuid4().hex, "cache_mode": "Disable cache",
        }},
    }


def required_image(spec: CaseSpec) -> bool:
    return spec.kind not in {"probe"}


def output_index_for(spec: CaseSpec) -> int:
    return 1 if spec.kind == "triposplat" else 0


def expected_file3d_type_for(spec: CaseSpec) -> str:
    return "FILE_3D_SPLAT_ANY" if spec.kind == "triposplat" else "FILE_3D_GLB"


def save_node_contract_for(spec: CaseSpec) -> tuple[str, str]:
    return ("SaveGLB", "mesh")


def _accepted_type(info: dict[str, Any], node_type: str, input_name: str) -> Any:
    value = (
        info.get(node_type, {})
        .get("input", {})
        .get("required", {})
        .get(input_name)
    )
    return value[0] if isinstance(value, list) and value else value


def check_object_info(
    api: ApiClient,
    prompt: dict[str, Any],
    source_node: str,
    spec: CaseSpec | None = None,
) -> dict[str, Any]:
    spec = spec or CASES["trellis_512"]
    info = api.get("/object_info")
    required = {value["class_type"] for value in prompt.values()}
    missing = sorted(node for node in required if node not in info)
    if missing:
        raise RuntimeError(f"ComfyUI is missing required nodes: {', '.join(missing)}")
    facade_type = prompt[source_node]["class_type"]
    facade_outputs = info.get(facade_type, {}).get("output", [])
    expected_type = expected_file3d_type_for(spec)
    output_index = output_index_for(spec)
    save_node, save_input = save_node_contract_for(spec)
    preview_type = _accepted_type(info, "Preview3D", "model_file")
    save_type = _accepted_type(info, save_node, save_input)
    output_type = facade_outputs[output_index] if len(facade_outputs) > output_index else None
    if output_type != expected_type:
        raise RuntimeError(f"{facade_type} did not expose {expected_type} (got {output_type!r})")
    for label, accepted in (("Preview3D", preview_type), (save_node, save_type)):
        if expected_type not in str(accepted):
            raise RuntimeError(f"{label} does not accept {expected_type} (got {accepted!r})")
    if prompt["90"]["inputs"]["model_file"] != [source_node, output_index]:
        raise RuntimeError("Preview3D is not connected to the facade output")
    if prompt["91"]["class_type"] != save_node:
        raise RuntimeError(f"{save_node} is not present as the explicit save node")
    if prompt["91"]["inputs"][save_input] != [source_node, output_index]:
        raise RuntimeError(f"{save_node} is not connected to the facade output")
    return {
        "facade": facade_type,
        "outputType": output_type,
        "previewAcceptedType": preview_type,
        "saveAcceptedType": save_type,
        "previewNode": "90",
        "saveNode": "91",
        "saveNodeType": save_node,
        "saveInput": save_input,
    }


def queue_prompt(api: ApiClient, prompt: dict[str, Any], client_id: str) -> str:
    response = api.post("/prompt", {"prompt": prompt, "client_id": client_id})
    prompt_id = response.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise RuntimeError(f"ComfyUI did not return a prompt ID: {response}")
    return prompt_id


async def _queue_and_capture_five_stage_events(
    api: ApiClient,
    prompt: dict[str, Any],
    client_id: str,
    timeout: float,
    recorder: Recorder,
    facade_node: str,
) -> tuple[str, FiveStageVerifier]:
    try:
        import aiohttp
    except ImportError as error:
        raise RuntimeError(
            "Five-stage live verification requires aiohttp from the pinned ComfyUI environment"
        ) from error

    parsed = urlparse(api.base_url)
    websocket_scheme = "wss" if parsed.scheme == "https" else "ws"
    websocket_url = (
        f"{websocket_scheme}://{parsed.netloc}{parsed.path.rstrip('/')}/ws"
        f"?clientId={client_id}"
    )
    verifier = FiveStageVerifier(facade_node)
    deadline = time.monotonic() + timeout
    client_timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=None)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.ws_connect(websocket_url, heartbeat=20) as websocket:
            async with session.post(
                api.base_url + "/prompt",
                json={"prompt": prompt, "client_id": client_id},
            ) as response:
                queued = await response.json()
                if response.status != 200:
                    raise RuntimeError(
                        f"ComfyUI prompt submission returned {response.status}: {queued}"
                    )
            prompt_id = queued.get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise RuntimeError(f"ComfyUI did not return a prompt ID: {queued}")
            verifier.prompt_id = prompt_id
            recorder.event("queued", promptId=prompt_id, fiveStageVerification=True)
            last_waiting = 0.0
            while time.monotonic() < deadline:
                remaining = max(0.1, deadline - time.monotonic())
                try:
                    message = await asyncio.wait_for(
                        websocket.receive(),
                        timeout=min(10.0, remaining),
                    )
                except asyncio.TimeoutError:
                    now = time.monotonic()
                    if now - last_waiting >= 10:
                        recorder.event("waiting", promptId=prompt_id)
                        last_waiting = now
                    continue
                outcome = None
                if message.type == aiohttp.WSMsgType.BINARY:
                    verifier.record_binary(bytes(message.data))
                elif message.type == aiohttp.WSMsgType.TEXT:
                    outcome = verifier.record_json(json.loads(message.data))
                elif message.type in {
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.ERROR,
                }:
                    raise RuntimeError("ComfyUI WebSocket closed during five-stage verification")
                if outcome in {"completed", "failed"}:
                    return prompt_id, verifier
    raise TimeoutError(
        f"ComfyUI prompt did not finish five-stage event capture within {timeout:.0f}s"
    )


def queue_and_capture_five_stage_events(
    api: ApiClient,
    prompt: dict[str, Any],
    client_id: str,
    timeout: float,
    recorder: Recorder,
    facade_node: str,
) -> tuple[str, FiveStageVerifier]:
    return asyncio.run(
        _queue_and_capture_five_stage_events(
            api,
            prompt,
            client_id,
            timeout,
            recorder,
            facade_node,
        )
    )


def history_entry(payload: Any, prompt_id: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    entry = payload.get(prompt_id, payload if "status" in payload else None)
    return entry if isinstance(entry, dict) else None


def history_failure(entry: dict[str, Any]) -> str | None:
    messages = (entry.get("status") or {}).get("messages") or []
    for message in messages:
        if isinstance(message, (list, tuple)) and message:
            kind = str(message[0])
            if kind in {"execution_error", "execution_interrupted"}:
                return json.dumps(message, sort_keys=True, default=str)
    return None


def wait_prompt(api: ApiClient, prompt_id: str, timeout: float, recorder: Recorder) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_event = 0.0
    while time.monotonic() < deadline:
        entry = history_entry(api.get(f"/history/{prompt_id}"), prompt_id)
        if entry is not None:
            failure = history_failure(entry)
            completed = (entry.get("status") or {}).get("completed") is True
            if failure:
                raise RuntimeError(f"ComfyUI prompt {prompt_id} failed: {failure}")
            if completed:
                return entry
        if time.monotonic() - last_event >= 10:
            recorder.event("waiting", promptId=prompt_id)
            last_event = time.monotonic()
        time.sleep(1)
    raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish within {timeout:.0f}s")


def output_snapshot(output_root: Path) -> dict[Path, tuple[int, int]]:
    result: dict[Path, tuple[int, int]] = {}
    if not output_root.exists():
        return result
    for suffix in ("*.glb", "*.ply"):
        for path in output_root.rglob(suffix):
            try:
                stat = path.stat()
                result[path.resolve()] = (stat.st_mtime_ns, stat.st_size)
            except OSError:
                pass
    return result


def changed_artifacts(
    before: dict[Path, tuple[int, int]],
    output_root: Path,
    *,
    suffixes: tuple[str, ...],
) -> list[Path]:
    after = output_snapshot(output_root)
    wanted = tuple(item.lower() for item in suffixes)
    return sorted(
        (
            path
            for path, value in after.items()
            if before.get(path) != value and path.suffix.lower() in wanted
        ),
        key=lambda item: after[item][0],
    )


def changed_glbs(before: dict[Path, tuple[int, int]], output_root: Path) -> list[Path]:
    return changed_artifacts(before, output_root, suffixes=(".glb",))


def history_output_paths(
    history: dict[str, Any],
    node_id: str,
    output_root: Path,
    *,
    suffixes: tuple[str, ...] = (".glb",),
) -> list[Path]:
    node_output = (history.get("outputs") or {}).get(str(node_id), {})
    paths: list[Path] = []
    wanted = tuple(item.lower() for item in suffixes)
    for entries in node_output.values() if isinstance(node_output, dict) else []:
        if not isinstance(entries, list):
            continue
        for entry in entries:
            filename = str(entry.get("filename", ""))
            if not isinstance(entry, dict) or not filename.lower().endswith(wanted):
                continue
            if entry.get("type") != "output":
                continue
            path = output_root / str(entry.get("subfolder") or "") / filename
            paths.append(path.resolve())
    return paths


def preview_event_paths(event: dict[str, Any], output_root: Path) -> list[Path]:
    root = output_root.resolve()
    paths: list[Path] = []
    for name in event.get("glbs", []):
        relative = Path(str(name))
        if relative.is_absolute():
            raise RuntimeError(f"Preview reported an absolute output path: {name}")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise RuntimeError(f"Preview output escaped the ComfyUI output directory: {name}") from error
        paths.append(candidate)
    return paths


def classify_glb(path: Path, *, require_noncollapsed: bool = False) -> dict[str, Any]:
    try:
        return {
            **inspect_glb(
                path,
                require_textured=True,
                require_noncollapsed=require_noncollapsed,
            ),
            "artifactKind": "textured",
        }
    except ValueError:
        return {
            **inspect_glb(
                path,
                require_textured=False,
                require_noncollapsed=require_noncollapsed,
            ),
            "artifactKind": "geometry",
        }


def _parse_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    if len(payload) < 20:
        raise ValueError("GLB is truncated")
    magic, version, length = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or length != len(payload):
        raise ValueError("GLB header is invalid")
    offset = 12
    document: dict[str, Any] | None = None
    binary = b""
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise ValueError("GLB chunk header is truncated")
        chunk_length, chunk_type = struct.unpack_from("<I4s", payload, offset)
        offset += 8
        end = offset + chunk_length
        if end > len(payload):
            raise ValueError("GLB chunk is truncated")
        chunk = payload[offset:end]
        offset = end
        if chunk_type == b"JSON":
            document = json.loads(chunk.decode("utf-8").rstrip(" \t\r\n\x00"))
        elif chunk_type == b"BIN\x00":
            binary = chunk
    if not isinstance(document, dict):
        raise ValueError("GLB has no JSON document")
    return document, binary


_COMPONENTS = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}
_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def iter_accessor(
    document: dict[str, Any], binary: bytes, accessor_index: int
) -> tuple[int, Iterable[tuple[Any, ...]]]:
    accessors = document.get("accessors") or []
    views = document.get("bufferViews") or []
    if not 0 <= accessor_index < len(accessors):
        raise ValueError(f"GLB accessor {accessor_index} is invalid")
    accessor = accessors[accessor_index]
    view_index = accessor.get("bufferView")
    if not isinstance(view_index, int) or not 0 <= view_index < len(views):
        raise ValueError(f"GLB accessor {accessor_index} has no embedded buffer view")
    view = views[view_index]
    if int(view.get("buffer", 0)) != 0:
        raise ValueError("GLB references an external mesh buffer")
    component = _COMPONENTS.get(int(accessor.get("componentType", 0)))
    width = _WIDTHS.get(str(accessor.get("type", "")))
    count = int(accessor.get("count", 0))
    if component is None or width is None or count <= 0:
        raise ValueError(f"GLB accessor {accessor_index} has an unsupported representation")
    format_code, component_bytes = component
    packed_bytes = component_bytes * width
    stride = int(view.get("byteStride", packed_bytes))
    if stride < packed_bytes:
        raise ValueError(f"GLB accessor {accessor_index} has an invalid stride")
    view_start = int(view.get("byteOffset", 0))
    view_end = view_start + int(view.get("byteLength", 0))
    offset = view_start + int(accessor.get("byteOffset", 0))
    final_end = offset + (count - 1) * stride + packed_bytes
    if offset < 0 or final_end > min(view_end, len(binary)):
        raise ValueError(f"GLB accessor {accessor_index} exceeds its embedded buffer")
    unpack = struct.Struct("<" + format_code * width).unpack_from
    return count, (unpack(binary, offset + item * stride) for item in range(count))


def compute_geometry_metrics(
    vertices: Iterable[tuple[float, float, float]],
    faces: Iterable[tuple[int, int, int]],
) -> dict[str, Any]:
    """Adapt the shared runtime geometry contract to the live-record schema."""

    quality = _GEOMETRY_QUALITY.analyze_geometry(
        vertices,
        faces,
        stage="live GLB validation",
    )
    payload = quality.to_dict()
    minimum = payload["bounds_min"]
    maximum = payload["bounds_max"]
    extents = payload["extents"]
    diagonal = math.sqrt(math.fsum(extent * extent for extent in extents))
    singular_values = payload["singular_values"]
    largest_singular = singular_values[0]
    singular_ratios = (
        [value / largest_singular for value in singular_values]
        if largest_singular > 0.0
        else [0.0, 0.0, 0.0]
    )
    intrinsic_rank = payload["numerical_rank"]
    surface_area = payload["surface_area"]
    nondegenerate_faces = payload["nondegenerate_face_count"]
    nondegenerate_ratio = payload["nondegenerate_face_ratio"]
    checks = {
        "hasSpatialExtent": largest_singular > 0.0,
        "intrinsicRankThree": intrinsic_rank == 3,
        "hasNondegenerateFaces": nondegenerate_faces > 0,
        "positiveSurfaceArea": surface_area > 0.0,
    }
    return {
        "schema": GEOMETRY_METRICS_SCHEMA,
        "contractSchema": payload["schema"],
        "bounds": {
            "minimum": minimum,
            "maximum": maximum,
            "extents": extents,
            "diagonal": diagonal,
        },
        "centeredSvd": {
            "singularValues": singular_values,
            "singularValueRatios": singular_ratios,
            "relativeRankThreshold": INTRINSIC_RANK_RELATIVE_THRESHOLD,
            "intrinsicRank": intrinsic_rank,
        },
        "surfaceArea": surface_area,
        "nondegenerateFaces": nondegenerate_faces,
        "nondegenerateFaceRatio": nondegenerate_ratio,
        "minimumNondegenerateFaceRatio": MIN_NONDEGENERATE_FACE_RATIO,
        "connectedComponents": payload["connected_component_count"],
        "warnings": payload["warnings"],
        "checks": checks,
        "nonCollapsed": payload["passes_volumetric_validation"] and all(checks.values()),
    }


def require_noncollapsed_geometry(metrics: dict[str, Any], *, stage: str) -> None:
    if metrics.get("schema") == GEOMETRY_METRICS_SCHEMA and metrics.get("nonCollapsed") is True:
        return
    failed = ", ".join(
        name for name, passed in (metrics.get("checks") or {}).items() if not passed
    ) or "geometry metrics missing"
    svd = metrics.get("centeredSvd") or {}
    raise ValueError(
        f"{stage} is geometrically collapsed ({failed}); "
        f"intrinsicRank={svd.get('intrinsicRank')}, "
        f"singularValueRatios={svd.get('singularValueRatios')}, "
        f"nondegenerateFaceRatio={metrics.get('nondegenerateFaceRatio')}, "
        f"surfaceArea={metrics.get('surfaceArea')}"
    )


def inspect_glb(
    path: Path,
    *,
    require_textured: bool,
    require_noncollapsed: bool = False,
) -> dict[str, Any]:
    document, binary = _parse_glb(path)
    accessors = document.get("accessors") or []
    materials = document.get("materials") or []
    textures = document.get("textures") or []
    images = document.get("images") or []
    faces = vertices = primitives = 0
    metric_vertices: list[tuple[float, float, float]] = []
    metric_faces: list[tuple[int, int, int]] = []
    for mesh in document.get("meshes") or []:
        for primitive in mesh.get("primitives") or []:
            primitives += 1
            attributes = primitive.get("attributes") or {}
            position = attributes.get("POSITION")
            indices = primitive.get("indices")
            if not isinstance(position, int) or not 0 <= position < len(accessors):
                raise ValueError("GLB primitive has no valid POSITION accessor")
            if not isinstance(indices, int) or not 0 <= indices < len(accessors):
                raise ValueError("GLB primitive has no valid index accessor")
            position_accessor = accessors[position]
            index_accessor = accessors[indices]
            vertex_count = int(position_accessor.get("count", 0))
            index_count = int(index_accessor.get("count", 0))
            if vertex_count <= 0 or index_count <= 0 or index_count % 3:
                raise ValueError("GLB primitive has invalid vertex/index counts")
            actual_vertex_count, position_rows = iter_accessor(document, binary, position)
            position_values = [tuple(float(value) for value in row) for row in position_rows]
            if actual_vertex_count != vertex_count or not all(
                len(row) == 3 and
                isinstance(value, (int, float)) and float("-inf") < float(value) < float("inf")
                for row in position_values for value in row
            ):
                raise ValueError("GLB primitive has non-finite vertices")
            actual_index_count, index_rows = iter_accessor(document, binary, indices)
            index_values = [int(row[0]) for row in index_rows]
            if (
                actual_index_count != index_count
                or not index_values
                or min(index_values) < 0
                or max(index_values) >= vertex_count
            ):
                raise ValueError("GLB primitive has out-of-range triangle indices")
            vertices += vertex_count
            faces += index_count // 3
            vertex_offset = len(metric_vertices)
            metric_vertices.extend(position_values)
            metric_faces.extend(
                (
                    vertex_offset + index_values[index],
                    vertex_offset + index_values[index + 1],
                    vertex_offset + index_values[index + 2],
                )
                for index in range(0, len(index_values), 3)
            )
            if require_textured:
                uv = attributes.get("TEXCOORD_0")
                material = primitive.get("material")
                if not isinstance(uv, int) or not 0 <= uv < len(accessors):
                    raise ValueError("Textured GLB primitive has no UV accessor")
                if int(accessors[uv].get("count", 0)) != vertex_count:
                    raise ValueError("Textured GLB UV count does not match vertices")
                uv_count, uv_rows = iter_accessor(document, binary, uv)
                if uv_count != vertex_count or not all(
                    isinstance(value, (int, float)) and float("-inf") < float(value) < float("inf")
                    for row in uv_rows for value in row
                ):
                    raise ValueError("Textured GLB has invalid UV coordinates")
                if not isinstance(material, int) or not 0 <= material < len(materials):
                    raise ValueError("Textured GLB primitive has no material")
                texture = (materials[material].get("pbrMetallicRoughness") or {}).get("baseColorTexture")
                texture_index = texture.get("index") if isinstance(texture, dict) else None
                if not isinstance(texture_index, int) or not 0 <= texture_index < len(textures):
                    raise ValueError("Textured GLB has no base-color texture")
                image_index = textures[texture_index].get("source")
                if not isinstance(image_index, int) or not 0 <= image_index < len(images):
                    raise ValueError("Textured GLB texture has no image")
                image = images[image_index]
                image_view = image.get("bufferView")
                if image_view is not None:
                    views = document.get("bufferViews") or []
                    if not isinstance(image_view, int) or not 0 <= image_view < len(views):
                        raise ValueError("Textured GLB image has an invalid buffer view")
                    view = views[image_view]
                    start = int(view.get("byteOffset", 0))
                    length = int(view.get("byteLength", 0))
                    if int(view.get("buffer", 0)) != 0 or start < 0 or length <= 0 or start + length > len(binary):
                        raise ValueError("Textured GLB embedded image exceeds its buffer")
                elif not str(image.get("uri", "")).startswith("data:"):
                    raise ValueError("Textured GLB image is not embedded")
    if not primitives or not binary:
        raise ValueError("GLB has no embedded mesh data")
    geometry_metrics = compute_geometry_metrics(metric_vertices, metric_faces)
    if require_noncollapsed:
        require_noncollapsed_geometry(geometry_metrics, stage=f"GLB artifact {path.name}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "vertices": vertices,
        "faces": faces,
        "primitives": primitives,
        "materialCount": len(materials),
        "textureCount": len(textures),
        "embeddedTextureValidated": require_textured,
        "geometryMetrics": geometry_metrics,
        "nonCollapsedGeometryValidated": geometry_metrics["nonCollapsed"],
    }


_PLY_SCALAR_TYPES = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}
_PLY_REQUIRED_GROUPS = {
    "position": ("x", "y", "z"),
    "dc": ("f_dc_0", "f_dc_1", "f_dc_2"),
    "opacity": ("opacity",),
    "scale": ("scale_0", "scale_1", "scale_2"),
    "rotation": ("rot_0", "rot_1", "rot_2", "rot_3"),
}


def inspect_3dgs_ply(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if not payload:
        raise ValueError("PLY artifact is empty")
    marker = b"end_header\n"
    header_end = payload.find(marker)
    if header_end < 0:
        raise ValueError("PLY header is missing end_header")
    header_end += len(marker)
    header = payload[:header_end].decode("ascii", "strict").splitlines()
    if len(header) < 3 or header[0] != "ply":
        raise ValueError("PLY header is invalid")
    if header[1] != "format binary_little_endian 1.0":
        raise ValueError("PLY must be binary_little_endian 1.0")
    vertex_count: int | None = None
    vertex_properties: list[tuple[str, str]] = []
    current_element: str | None = None
    for line in header[2:]:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            current_element = parts[1] if len(parts) > 1 else None
            if current_element == "vertex":
                if len(parts) != 3:
                    raise ValueError("PLY vertex element is malformed")
                vertex_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) != 3 or parts[1] == "list":
                raise ValueError("PLY vertex properties must be scalar")
            if parts[1] not in _PLY_SCALAR_TYPES:
                raise ValueError(f"PLY vertex property {parts[2]} has unsupported type {parts[1]}")
            vertex_properties.append((parts[1], parts[2]))
    if vertex_count is None or vertex_count <= 0:
        raise ValueError("PLY vertex count must be positive")
    names = [name for _kind, name in vertex_properties]
    missing = {
        group: [name for name in required if name not in names]
        for group, required in _PLY_REQUIRED_GROUPS.items()
    }
    missing = {group: values for group, values in missing.items() if values}
    if missing:
        raise ValueError(f"PLY lacks required 3DGS properties: {missing}")
    row_bytes = sum(_PLY_SCALAR_TYPES[kind][1] for kind, _name in vertex_properties)
    expected_bytes = header_end + vertex_count * row_bytes
    if len(payload) < expected_bytes:
        raise ValueError("PLY vertex payload is truncated")
    first_vertex: dict[str, float] = {}
    offset = header_end
    for kind, name in vertex_properties:
        code, width = _PLY_SCALAR_TYPES[kind]
        value = struct.unpack_from("<" + code, payload, offset)[0]
        offset += width
        first_vertex[name] = float(value)
    for name in ("x", "y", "z", "opacity"):
        value = first_vertex[name]
        if not math.isfinite(value):
            raise ValueError(f"PLY property {name} has a non-finite value")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "format": "binary_little_endian",
        "artifactKind": "3dgs-ply",
        "gaussianCount": vertex_count,
        "vertices": vertex_count,
        "properties": names,
        "requiredPropertyGroups": {
            group: list(required) for group, required in _PLY_REQUIRED_GROUPS.items()
        },
        "plyValidated": True,
        "file3dValidated": True,
        "modelRevision": TRIPOSPLAT_MODEL_REVISION,
    }


def read_log_since(path: Path, offset: int) -> tuple[str, int]:
    try:
        size = path.stat().st_size
        if size < offset:
            offset = 0
        with path.open("rb") as stream:
            stream.seek(offset)
            payload = stream.read()
            return payload.decode("utf-8", "replace"), stream.tell()
    except OSError:
        return "", offset


def read_settled_log_since(
    path: Path,
    offset: int,
    *,
    require_shape_marker: bool = False,
    timeout: float = 10.0,
    settle_seconds: float = 1.5,
) -> tuple[str, int]:
    """Wait briefly for asynchronously forwarded isolated-node output."""

    deadline = time.monotonic() + timeout
    last_end = offset
    last_change = time.monotonic()
    text = ""
    while True:
        text, end = read_log_since(path, offset)
        now = time.monotonic()
        if end != last_end:
            last_end = end
            last_change = now
        required_marker_ready = not require_shape_marker or SHAPE_METRICS.search(text) is not None
        if required_marker_ready and now - last_change >= settle_seconds:
            return text, end
        if now >= deadline:
            return text, end
        time.sleep(0.25)


def compact_log_evidence(text: str, *, tail_bytes: int = 12_000) -> str:
    """Retain release markers even when verbose mesh logs exceed the evidence tail."""

    tail = text[-tail_bytes:]
    markers = [match.group(0) for match in SHAPE_METRICS.finditer(text)]
    markers.extend(match.group(0) for match in GEOMETRY_QUALITY_EVENT.finditer(text))
    markers.extend(match.group(0) for match in ULTRASHAPE_SETTINGS_EVENT.finditer(text))
    markers.extend(
        match.group(0) for match in ULTRASHAPE_WORKER_SETTINGS_EVENT.finditer(text)
    )
    markers.extend(match.group(0) for match in PIXAL3D_WORKER_RESULT_EVENT.finditer(text))
    missing = [marker for marker in markers if marker not in tail]
    return "\n".join([*missing, tail]) if missing else tail


def geometry_quality_events(text: str) -> list[dict[str, Any]]:
    events = []
    for match in GEOMETRY_QUALITY_EVENT.finditer(text):
        try:
            payload = json.loads(match.group("payload"))
        except json.JSONDecodeError as exc:
            raise ValueError("ComfyUI emitted malformed geometry-quality evidence") from exc
        if payload.get("schema") != _GEOMETRY_QUALITY.GEOMETRY_QUALITY_SCHEMA:
            raise ValueError("ComfyUI emitted geometry-quality evidence with an unknown schema")
        events.append(payload)
    return events


def _settings_events(
    text: str,
    pattern: re.Pattern[str],
    *,
    label: str,
) -> list[dict[str, Any]]:
    events = []
    for match in pattern.finditer(text):
        try:
            payload = json.loads(match.group("payload"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"ComfyUI emitted malformed {label} evidence") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"ComfyUI emitted invalid {label} evidence")
        events.append(payload)
    return events


def ultrashape_resolved_settings_events(text: str) -> list[dict[str, Any]]:
    return _settings_events(
        text,
        ULTRASHAPE_SETTINGS_EVENT,
        label="UltraShape resolved-settings",
    )


def ultrashape_worker_settings_events(text: str) -> list[dict[str, Any]]:
    return _settings_events(
        text,
        ULTRASHAPE_WORKER_SETTINGS_EVENT,
        label="UltraShape worker-settings",
    )


def pixal3d_worker_result_events(text: str) -> list[dict[str, Any]]:
    return _settings_events(
        text,
        PIXAL3D_WORKER_RESULT_EVENT,
        label="Pixal3D worker-result",
    )


def source_node_for(spec: CaseSpec) -> str:
    if spec.kind in {"trellis", "cache", "strict1536"}:
        return "2"
    if spec.kind in {"triposplat", "pixal3d", "pixal3d_cache", "pixal3d_cancel", "pixal3d_reuse"}:
        return "2"
    if spec.kind in {"ultrashape", "full", "cancel"}:
        return "3"
    if spec.kind == "advanced":
        return "9"
    raise ValueError(spec.kind)


def evidence_id(record: dict[str, Any]) -> str:
    compact = dict(record)
    compact.pop("resultFiles", None)
    digest = hashlib.sha256(json.dumps(compact, sort_keys=True).encode("utf-8")).hexdigest()
    return f"live-g4:{record['runId']}:{record['case']}:{digest}"


def benchmark_from(
    spec: CaseSpec,
    runtime: float,
    peak_vram: int,
    glb: dict[str, Any],
    log_text: str,
    *,
    observed_resolution: int | None = None,
    texture_size: int | None = None,
    pixal3d_worker_result: dict[str, Any] | None = None,
    require_geometry_metrics: bool = False,
) -> dict[str, Any] | None:
    if not spec.benchmark:
        return None
    if spec.kind == "triposplat":
        if (
            runtime <= 0
            or peak_vram <= 0
            or int(glb.get("bytes", 0)) <= 0
            or int(glb.get("gaussianCount", 0)) <= 0
            or glb.get("plyValidated") is not True
            or glb.get("file3dValidated") is not True
        ):
            raise RuntimeError(
                "Benchmark metrics are incomplete; runtime, peak VRAM, PLY bytes, "
                "Gaussian count, and FILE_3D validation must be positive"
            )
        return {
            "status": "passed",
            "quality": spec.quality,
            "gaussianCount": int(glb["gaussianCount"]),
            "runtimeSeconds": round(runtime, 3),
            "peakVramBytes": peak_vram,
            "plyBytes": glb["bytes"],
            "plySha256": glb["sha256"],
            "plyValidated": True,
            "file3dValidated": True,
            "outputFormat": "ply",
            "modelRevision": glb.get("modelRevision", TRIPOSPLAT_MODEL_REVISION),
        }
    if runtime <= 0 or peak_vram <= 0 or int(glb.get("bytes", 0)) <= 0 or int(glb.get("faces", 0)) <= 0:
        raise RuntimeError(
            "Benchmark metrics are incomplete; runtime, peak VRAM, GLB bytes, and faces must be positive"
        )
    geometry_metrics = glb.get("geometryMetrics")
    geometry_validated = bool(
        isinstance(geometry_metrics, dict)
        and geometry_metrics.get("schema") == GEOMETRY_METRICS_SCHEMA
        and geometry_metrics.get("nonCollapsed") is True
        and glb.get("nonCollapsedGeometryValidated") is True
    )
    if require_geometry_metrics and not geometry_validated:
        raise RuntimeError(
            "Benchmark artifact lacks passing non-collapsed geometry metrics"
        )
    matches = list(SHAPE_METRICS.finditer(log_text))
    if spec.kind == "trellis":
        if not matches:
            raise RuntimeError("TRELLIS completed without a `ComfyColab shape metrics` marker")
        marker = matches[-1]
        actual_resolution = int(marker.group("resolution"))
        tokens = int(marker.group("tokens"))
        if actual_resolution != spec.actual_resolution:
            raise RuntimeError(
                f"TRELLIS requested {spec.actual_resolution} but actually ran {actual_resolution}; silent downgrade rejected"
            )
    elif spec.kind in {"pixal3d", "pixal3d_cache", "pixal3d_reuse"}:
        if not isinstance(pixal3d_worker_result, dict):
            raise RuntimeError("Pixal3D completed without a machine-readable worker result")
        actual_resolution = int(pixal3d_worker_result.get("actual_resolution", 0))
        tokens = int(pixal3d_worker_result.get("token_count", 0))
        if actual_resolution != spec.actual_resolution:
            raise RuntimeError(
                f"Pixal3D requested {spec.actual_resolution} but actually ran "
                f"{actual_resolution}; silent downgrade rejected"
            )
        if tokens <= 0:
            raise RuntimeError("Pixal3D worker result omitted a positive token count")
    else:
        if not observed_resolution:
            raise RuntimeError(
                f"{spec.name} lacks machine-observed UltraShape decode resolution"
            )
        actual_resolution = int(observed_resolution)
        tokens = None
        if actual_resolution != spec.actual_resolution:
            raise RuntimeError(
                f"{spec.name} requires octree resolution {spec.actual_resolution}, got {actual_resolution}"
            )
    benchmark = {
        "status": "passed",
        "actualResolution": actual_resolution,
        "runtimeSeconds": round(runtime, 3),
        "peakVramBytes": peak_vram,
        "glbBytes": glb["bytes"],
        "faces": glb["faces"],
        "glbValidated": True,
        "nonCollapsedGeometryValidated": geometry_validated,
        "geometryMetrics": geometry_metrics if geometry_validated else None,
    }
    if spec.kind in {"trellis", "pixal3d", "pixal3d_cache", "pixal3d_reuse"}:
        benchmark.update(tokens=tokens, textureSize=texture_size or spec.texture_size)
    if spec.kind in {"pixal3d", "pixal3d_cache", "pixal3d_reuse"}:
        benchmark.update(
            workerPeakVramBytes=int(pixal3d_worker_result.get("peak_vram_bytes", 0)),
            pipelineLoadCount=int(pixal3d_worker_result.get("pipeline_load_count", 0)),
            workerPid=int(pixal3d_worker_result.get("worker_pid", 0)),
        )
    return benchmark


def run_prompt_once(
    spec: CaseSpec,
    args: argparse.Namespace,
    run_id: str,
    image_name: str,
    recorder: Recorder,
    *,
    cache_mode: str | None = None,
) -> dict[str, Any]:
    api = ApiClient(args.base_url)
    # CLI namespaces set this true. Omitting it is the explicit compatibility
    # path for older structural-only unit fixtures that contain planar GLBs.
    require_geometry_evidence = bool(
        getattr(args, "require_geometry_evidence", False)
    )
    effective_cache_mode = cache_mode or args.cache_mode
    prompt = build_prompt(
        spec,
        args,
        image_name,
        run_id,
        cache_mode=effective_cache_mode,
    )
    source_node = source_node_for(spec)
    proof = check_object_info(api, prompt, source_node, spec)
    verify_five_stages = (
        spec.kind in {"trellis", "cache"}
        and effective_cache_mode != "Use cache"
    )
    output_root = Path(args.comfy_root) / "output"
    before = output_snapshot(output_root)
    log_offset = Path(args.comfy_log).stat().st_size if Path(args.comfy_log).exists() else 0
    started = time.monotonic()
    stage_verifier: FiveStageVerifier | None = None
    with VramSampler(args.vram_interval) as sampler:
        client_id = f"comfycolab-live3d-{run_id}-{spec.name}"
        if verify_five_stages:
            prompt_id, stage_verifier = queue_and_capture_five_stage_events(
                api,
                prompt,
                client_id,
                args.timeout,
                recorder,
                facade_node=source_node,
            )
            history = wait_prompt(api, prompt_id, min(args.timeout, 60.0), recorder)
        else:
            prompt_id = queue_prompt(api, prompt, client_id)
            recorder.event("queued", promptId=prompt_id)
            history = wait_prompt(api, prompt_id, args.timeout, recorder)
    runtime = time.monotonic() - started
    log_text, _ = read_settled_log_since(
        Path(args.comfy_log),
        log_offset,
        require_shape_marker=spec.kind == "trellis",
    )
    stage_geometry_metrics = geometry_quality_events(log_text)
    resolved_ultrashape_events = ultrashape_resolved_settings_events(log_text)
    worker_ultrashape_events = ultrashape_worker_settings_events(log_text)
    worker_pixal3d_events = pixal3d_worker_result_events(log_text)
    resolved_ultrashape_settings = (
        resolved_ultrashape_events[-1] if resolved_ultrashape_events else None
    )
    worker_ultrashape_settings = (
        worker_ultrashape_events[-1] if worker_ultrashape_events else None
    )
    worker_pixal3d_result = worker_pixal3d_events[-1] if worker_pixal3d_events else None
    if (
        spec.kind in {"pixal3d", "pixal3d_cache", "pixal3d_reuse"}
        and effective_cache_mode != "Use cache"
        and not isinstance(worker_pixal3d_result, dict)
    ):
        raise RuntimeError("Pixal3D live run lacked machine-readable worker result evidence")
    if (
        require_geometry_evidence
        and spec.kind in {"trellis", "cache"}
        and effective_cache_mode != "Use cache"
    ):
        required_stages = {
            "TRELLIS raw shape",
            "TRELLIS processed mesh",
            "TRELLIS rasterized mesh",
        }
        observed_stages = {
            str(metrics.get("stage")) for metrics in stage_geometry_metrics
        }
        missing_stages = sorted(required_stages - observed_stages)
        if missing_stages:
            raise RuntimeError(
                "TRELLIS live run lacked stage geometry evidence: "
                + ", ".join(missing_stages)
            )
    if require_geometry_evidence and spec.kind in {"ultrashape", "full"}:
        if not isinstance(resolved_ultrashape_settings, dict):
            raise RuntimeError("UltraShape live run lacked resolved preset evidence")
        if not isinstance(worker_ultrashape_settings, dict):
            raise RuntimeError("UltraShape live run lacked worker settings evidence")
        expected_resolution = int(spec.actual_resolution or 0)
        for label, settings in (
            ("resolved preset", resolved_ultrashape_settings),
            ("worker", worker_ultrashape_settings),
        ):
            observed_resolution = int(settings.get("octree_resolution", 0))
            if observed_resolution != expected_resolution:
                raise RuntimeError(
                    f"UltraShape {label} used octree resolution {observed_resolution}; "
                    f"expected {expected_resolution}"
                )
        for field in ("steps", "num_latents", "octree_resolution", "decode_chunk_size", "seed"):
            if resolved_ultrashape_settings.get(field) != worker_ultrashape_settings.get(field):
                raise RuntimeError(
                    f"UltraShape resolved and worker settings disagree for {field}"
                )
    files = (
        changed_artifacts(before, output_root, suffixes=(".ply",))
        if spec.kind == "triposplat"
        else changed_glbs(before, output_root)
    )
    if not files:
        expected = "PLY" if spec.kind == "triposplat" else "GLB"
        raise RuntimeError(f"ComfyUI completed but produced no new or changed {expected}")
    if verify_five_stages:
        assert stage_verifier is not None
        stage_proof = stage_verifier.verify()
        changed = {path.resolve() for path in files}
        early_event = next(
            event
            for event in stage_proof["previewEvents"]
            if event["node"] != stage_verifier.preview_node
        )
        final_event = next(
            event
            for event in stage_proof["previewEvents"]
            if event["node"] == stage_verifier.preview_node
        )
        early_paths = preview_event_paths(early_event, output_root)
        final_preview_paths = preview_event_paths(final_event, output_root)
        if any(path not in changed for path in [*early_paths, *final_preview_paths]):
            raise RuntimeError("A recorded preview did not point to a GLB from this prompt")
        early_artifacts = [
            classify_glb(
                path,
                require_noncollapsed=require_geometry_evidence,
            )
            for path in early_paths
        ]
        if not early_artifacts or any(
            item["artifactKind"] != "geometry" for item in early_artifacts
        ):
            raise RuntimeError("The recorded early preview was not a geometry-only GLB")
        final_previews = [
            inspect_glb(
                path,
                require_textured=True,
                require_noncollapsed=require_geometry_evidence,
            )
            for path in final_preview_paths
        ]
        if not final_previews:
            raise RuntimeError("The final preview did not report a textured GLB")
        saved_paths = history_output_paths(history, "91", output_root)
        if not saved_paths:
            raise RuntimeError("SaveGLB node 91 did not report an output artifact")
        if any(path not in changed for path in saved_paths):
            raise RuntimeError("SaveGLB node 91 reported an artifact outside this prompt")
        saved = [
            inspect_glb(
                path,
                require_textured=True,
                require_noncollapsed=require_geometry_evidence,
            )
            for path in saved_paths
        ]
        current_paths = dict.fromkeys(
            [*early_paths, *final_preview_paths, *saved_paths]
        )
        validated = [
            classify_glb(
                path,
                require_noncollapsed=require_geometry_evidence,
            )
            for path in current_paths
        ]
        stage_proof["checks"].update(
            earlyGeometryArtifactValidated=True,
            finalTexturedArtifactValidated=True,
            explicitSaveGLBArtifactValidated=True,
        )
        stage_proof["artifacts"] = {
            "geometryPreviewCount": len(early_artifacts),
            "finalPreviewCount": len(final_previews),
            "saveGLBOutputCount": len(saved),
        }
        proof["fiveStageProof"] = stage_proof
        primary = max(saved, key=lambda item: item["bytes"])
    else:
        if spec.kind == "triposplat":
            saved_paths = history_output_paths(history, "91", output_root, suffixes=(".ply",))
            if not saved_paths:
                raise RuntimeError("SaveGLB node 91 did not report a PLY output artifact")
            changed = {path.resolve() for path in files}
            if any(path not in changed for path in saved_paths):
                raise RuntimeError("SaveGLB node 91 reported a PLY artifact outside this prompt")
            validated = [inspect_3dgs_ply(path) for path in saved_paths]
        else:
            validated = [
                inspect_glb(
                    path,
                    require_textured=spec.require_textured,
                    require_noncollapsed=require_geometry_evidence,
                )
                for path in files
            ]
        primary = max(validated, key=lambda item: item["bytes"])
    proof.update(historyCompleted=True, saveArtifactValidated=True)
    return {
        "promptId": prompt_id,
        "runtimeSeconds": round(runtime, 3),
        "peakVramBytes": sampler.peak_bytes,
        "historyStatus": history.get("status"),
        "previewSaveProof": proof,
        "glb": primary,
        "artifact": primary,
        "resultFiles": validated,
        "stageGeometryMetrics": stage_geometry_metrics,
        "resolvedUltraShapeSettings": resolved_ultrashape_settings,
        "workerUltraShapeSettings": worker_ultrashape_settings,
        "workerPixal3DResult": worker_pixal3d_result,
        "logExcerpt": compact_log_evidence(log_text),
    }


def run_cache_case(
    spec: CaseSpec, args: argparse.Namespace, run_id: str, image_name: str, recorder: Recorder
) -> dict[str, Any]:
    first = run_prompt_once(spec, args, run_id, image_name, recorder, cache_mode="Refresh this node")
    recorder.event("cache_seed_complete", promptId=first["promptId"])
    second = run_prompt_once(spec, args, run_id, image_name, recorder, cache_mode="Use cache")
    second_markers = list(SHAPE_METRICS.finditer(second.get("logExcerpt", "")))
    if second_markers:
        raise RuntimeError("Unchanged cache rerun emitted shape inference metrics; inference was not skipped")
    second_pixal3d_results = pixal3d_worker_result_events(second.get("logExcerpt", ""))
    if second_pixal3d_results:
        raise RuntimeError("Unchanged Pixal3D cache rerun emitted worker results; inference was not skipped")
    return {
        **second,
        "stageGeometryMetrics": first.get("stageGeometryMetrics", []),
        "workerPixal3DResult": first.get("workerPixal3DResult"),
        "cacheProof": {
            "firstPromptId": first["promptId"],
            "secondPromptId": second["promptId"],
            "secondRunShapeMetricCount": 0,
            "secondRunPixal3DWorkerResultCount": 0,
            "firstRuntimeSeconds": first["runtimeSeconds"],
            "secondRuntimeSeconds": second["runtimeSeconds"],
            "noModelInference": True,
            "freshFiveStageProof": first["previewSaveProof"].get("fiveStageProof"),
            "freshStageGeometryMetrics": first.get("stageGeometryMetrics", []),
        },
    }


def run_pixal3d_reuse_case(
    spec: CaseSpec,
    args: argparse.Namespace,
    run_id: str,
    image_name: str,
    recorder: Recorder,
) -> dict[str, Any]:
    first_args = argparse.Namespace(**{**vars(args), "keep_worker_loaded": True})
    second_seed = (int(args.seed) + 1) % (2**31)
    second_args = argparse.Namespace(
        **{**vars(args), "seed": second_seed, "keep_worker_loaded": True}
    )
    first = run_prompt_once(
        spec, first_args, run_id, image_name, recorder, cache_mode="Disable cache"
    )
    second = run_prompt_once(
        spec, second_args, run_id, image_name, recorder, cache_mode="Disable cache"
    )
    first_result = first.get("workerPixal3DResult")
    second_result = second.get("workerPixal3DResult")
    if not isinstance(first_result, dict) or not isinstance(second_result, dict):
        raise RuntimeError("Pixal3D reuse case lacked worker result evidence")
    first_pid = int(first_result.get("worker_pid", 0))
    second_pid = int(second_result.get("worker_pid", 0))
    if first_pid <= 0 or first_pid != second_pid:
        raise RuntimeError("Pixal3D reuse case relaunched the isolated worker")
    if int(first_result.get("pipeline_load_count", 0)) != 1 or int(
        second_result.get("pipeline_load_count", 0)
    ) != 1:
        raise RuntimeError("Pixal3D reuse case reloaded the pipeline")
    return {
        **second,
        "reuseProof": {
            "firstPromptId": first["promptId"],
            "secondPromptId": second["promptId"],
            "firstSeed": int(args.seed),
            "secondSeed": second_seed,
            "workerPid": first_pid,
            "pipelineLoadCount": 1,
            "workerReused": True,
            "pipelineReused": True,
        },
    }


def run_strict_1536_default_case(
    spec: CaseSpec, args: argparse.Namespace, run_id: str, image_name: str, recorder: Recorder
) -> dict[str, Any]:
    """Prove the public default cap either runs at 1536 or fails without downgrade."""

    strict_args = argparse.Namespace(**{**vars(args), "max_tokens": 49152})
    api = ApiClient(args.base_url)
    prompt = build_prompt(spec, strict_args, image_name, run_id, cache_mode="Disable cache")
    proof = check_object_info(api, prompt, source_node_for(spec))
    output_root = Path(args.comfy_root) / "output"
    before = output_snapshot(output_root)
    log_offset = Path(args.comfy_log).stat().st_size if Path(args.comfy_log).exists() else 0
    started = time.monotonic()
    with VramSampler(args.vram_interval) as sampler:
        prompt_id = queue_prompt(api, prompt, f"comfycolab-live3d-{run_id}-strict-1536")
        recorder.event("queued", promptId=prompt_id, maxTokens=49152)
        deadline = time.monotonic() + args.timeout
        entry = None
        failure = None
        while time.monotonic() < deadline:
            entry = history_entry(api.get(f"/history/{prompt_id}"), prompt_id)
            if entry is not None:
                failure = history_failure(entry)
                if failure or (entry.get("status") or {}).get("completed") is True:
                    break
            time.sleep(1)
        else:
            raise TimeoutError(f"Strict 1536 prompt {prompt_id} did not finish within {args.timeout:.0f}s")
    log_text, _ = read_settled_log_since(Path(args.comfy_log), log_offset)
    markers = [
        {"tokens": int(match.group("tokens")), "resolution": int(match.group("resolution"))}
        for match in SHAPE_METRICS.finditer(log_text)
    ]
    downgraded = [marker for marker in markers if marker["resolution"] < 1536]
    if downgraded:
        raise RuntimeError(f"Default-cap 1536 silently downgraded: {downgraded}")
    runtime = time.monotonic() - started
    if failure:
        combined_error = failure + "\n" + log_text
        if "Increase max_tokens" not in combined_error or "manually select 1024" not in combined_error:
            raise RuntimeError(
                "Default-cap 1536 failed without the required actionable max_tokens/1024 guidance: "
                + failure
            )
        if changed_glbs(before, output_root):
            raise RuntimeError("Failed strict 1536 run left a completed GLB behind")
        return {
            "promptId": prompt_id,
            "runtimeSeconds": round(runtime, 3),
            "peakVramBytes": sampler.peak_bytes,
            "strictDefaultCapProof": {
                "maxTokens": 49152,
                "outcome": "actionable-error",
                "silentDowngrade": False,
                "observedShapeMetrics": markers,
                "error": failure,
            },
        }
    files = changed_glbs(before, output_root)
    if not files:
        raise RuntimeError("Successful strict 1536 run produced no GLB")
    validated = [
        inspect_glb(path, require_textured=True, require_noncollapsed=True)
        for path in files
    ]
    marker = markers[-1] if markers else None
    if not marker or marker["resolution"] != 1536:
        raise RuntimeError("Successful strict 1536 run did not emit a genuine 1536 shape marker")
    proof.update(historyCompleted=True, saveArtifactValidated=True)
    return {
        "promptId": prompt_id,
        "runtimeSeconds": round(runtime, 3),
        "peakVramBytes": sampler.peak_bytes,
        "previewSaveProof": proof,
        "glb": max(validated, key=lambda item: item["bytes"]),
        "resultFiles": validated,
        "strictDefaultCapProof": {
            "maxTokens": 49152,
            "outcome": "genuine-1536",
            "silentDowngrade": False,
            "observedShapeMetrics": markers,
        },
    }


def run_probe_case(args: argparse.Namespace) -> dict[str, Any]:
    python = Path(args.trellis_python)
    if not python.is_file():
        raise FileNotFoundError(f"TRELLIS interpreter is missing: {python}")
    environment = dict(os.environ)
    repo_src = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = repo_src + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    regression = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from comfycolab.remote_bootstrap import validate_trellis_cache; "
                "validate_trellis_cache(Path.home() / '.ce', validate_ultrashape=True)"
            ),
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=args.timeout,
    )
    if regression.returncode:
        raise RuntimeError(
            "Pinned TRELLIS/GeometryPack/UltraShape regression probes failed: "
            f"{regression.stdout}\n{regression.stderr}"
        )
    code = """
import json, torch
import cubvh
from ultrashape.pipelines import UltraShapePipeline
from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
vertices = torch.tensor([
  [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
  [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1],
], dtype=torch.float32)
faces = torch.tensor([
  [0,2,1], [0,3,2], [4,5,6], [4,6,7], [0,1,5], [0,5,4],
  [2,3,7], [2,7,6], [0,4,7], [0,7,3], [1,2,6], [1,6,5],
], dtype=torch.int32)
bvh = cubvh.cuBVH(vertices, faces)
distance = bvh.unsigned_distance(torch.tensor([[0.,0.,0.]], device='cuda'))[0]
torch.cuda.synchronize()
if distance.shape != (1,) or not torch.isfinite(distance).all():
    raise RuntimeError('cubvh SM120 distance probe returned an invalid value')
payload = {
  'cuda': torch.cuda.is_available(),
  'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  'capability': list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
  'torch': torch.__version__,
  'cubvh': getattr(cubvh, '__file__', None),
  'ultrashapePipeline': UltraShapePipeline.__name__,
  'surfaceLoader': SharpEdgeSurfaceLoader.__name__,
  'cubvhDistanceKernel': True,
}
if not payload['cuda'] or payload['capability'] != [12, 0]:
    raise RuntimeError(payload)
print(json.dumps(payload, sort_keys=True))
"""
    source = str(Path(args.ultrashape_source))
    environment["PYTHONPATH"] = source + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    result = subprocess.run(
        [str(python), "-c", code], capture_output=True, text=True, env=environment, timeout=args.timeout
    )
    if result.returncode:
        raise RuntimeError(f"Combined environment probe failed: {result.stdout}\n{result.stderr}")
    lines = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"Combined environment probe returned no JSON: {result.stdout}")
    return {
        "probe": json.loads(lines[-1]),
        "bootstrapRegressionProbes": True,
        "runtimeSeconds": 0,
        "peakVramBytes": VramSampler.sample(),
    }


def partial_artifacts(roots: Iterable[Path]) -> list[str]:
    patterns = ("*.partial", "*.partial.glb", ".*.partial", ".*.partial.glb")
    found: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            found.update(str(path) for path in root.rglob(pattern) if path.is_file())
    return sorted(found)


def worker_pids(pattern: str = "worker/ultrashape/worker_main.py") -> list[int]:
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5
        )
        return [int(line) for line in result.stdout.splitlines() if line.strip().isdigit() and int(line) != os.getpid()]
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return []


def worker_pattern_for(spec: CaseSpec) -> str:
    return (
        "worker/pixal3d/worker_main.py"
        if spec.kind == "pixal3d_cancel"
        else "worker/ultrashape/worker_main.py"
    )


def run_cancellation_case(
    spec: CaseSpec, args: argparse.Namespace, run_id: str, image_name: str, recorder: Recorder
) -> dict[str, Any]:
    api = ApiClient(args.base_url)
    prompt = build_prompt(spec, args, image_name, run_id, cache_mode="Disable cache")
    check_object_info(api, prompt, source_node_for(spec))
    roots = [Path(args.comfy_root) / "temp", Path(args.comfy_root) / "output", Path(args.state_dir)]
    before_partial = set(partial_artifacts(roots))
    baseline_vram = VramSampler.sample()
    prompt_id = queue_prompt(api, prompt, f"comfycolab-live3d-{run_id}-cancel")
    recorder.event("queued_for_cancellation", promptId=prompt_id)
    worker_pattern = worker_pattern_for(spec)
    worker_label = "Pixal3D" if spec.kind == "pixal3d_cancel" else "UltraShape"
    deadline = time.monotonic() + args.cancel_start_timeout
    started_worker = False
    while time.monotonic() < deadline:
        if worker_pids(worker_pattern):
            started_worker = True
            break
        time.sleep(1)
    if not started_worker:
        raise RuntimeError(f"{worker_label} worker did not start before the cancellation deadline")
    time.sleep(args.cancel_after)
    api.post("/interrupt")
    recorder.event("interrupt_sent", promptId=prompt_id)
    deadline = time.monotonic() + 120
    entry = None
    while time.monotonic() < deadline:
        entry = history_entry(api.get(f"/history/{prompt_id}"), prompt_id)
        if entry is not None and history_failure(entry):
            break
        time.sleep(1)
    with contextlib.suppress(Exception):
        api.post("/free", {"unload_models": True, "free_memory": True})
    time.sleep(5)
    retained_workers = worker_pids(worker_pattern)
    after_partial = set(partial_artifacts(roots))
    new_partial = sorted(after_partial - before_partial)
    final_vram = VramSampler.sample()
    interrupted = bool(entry and history_failure(entry))
    gpu_released = final_vram <= baseline_vram + 512 * 1024**2
    if not interrupted or retained_workers or new_partial or not gpu_released:
        raise RuntimeError(
            "Cancellation cleanup failed: "
            f"interrupted={interrupted}, workers={retained_workers}, "
            f"partials={new_partial}, baseline_vram={baseline_vram}, final_vram={final_vram}"
        )
    return {
        "promptId": prompt_id,
        "interrupted": interrupted,
        "cleanupProof": {
            "workerStarted": True,
            "retainedWorkerPids": retained_workers,
            "newPartialArtifacts": new_partial,
            "baselineVramBytes": baseline_vram,
            "finalVramBytes": final_vram,
            "gpuAllocationReleased": gpu_released,
        },
    }


GEOMETRY_OUTPUT_KINDS = {
    "trellis",
    "cache",
    "strict1536",
    "ultrashape",
    "full",
    "advanced",
    "pixal3d",
    "pixal3d_cache",
    "pixal3d_reuse",
}


def record_has_noncollapsed_geometry(record: dict[str, Any]) -> bool:
    """Return whether a successful geometry-producing case carries semantic proof."""

    if record.get("kind") == "triposplat":
        artifact = record.get("artifact") or record.get("glb")
        return bool(
            isinstance(artifact, dict)
            and artifact.get("plyValidated") is True
            and artifact.get("file3dValidated") is True
            and int(artifact.get("gaussianCount", 0)) > 0
        )
    if record.get("kind") not in GEOMETRY_OUTPUT_KINDS:
        return True
    strict_outcome = (record.get("strictDefaultCapProof") or {}).get("outcome")
    if record.get("kind") == "strict1536" and strict_outcome == "actionable-error":
        return True
    glb = record.get("glb")
    metrics = glb.get("geometryMetrics") if isinstance(glb, dict) else None
    return bool(
        isinstance(glb, dict)
        and glb.get("nonCollapsedGeometryValidated") is True
        and isinstance(metrics, dict)
        and metrics.get("schema") == GEOMETRY_METRICS_SCHEMA
        and metrics.get("nonCollapsed") is True
    )


def execute_case(args: argparse.Namespace) -> int:
    spec = CASES[args.case]
    state_dir = Path(args.state_dir).resolve()
    run = ensure_run(state_dir)
    recorder = Recorder(state_dir, spec.name)
    recorder.status("running", runId=run["runId"], startedAt=utc_now())
    recorder.event("started", runId=run["runId"], pid=os.getpid())
    started = time.monotonic()
    try:
        if required_image(spec):
            if not args.image:
                raise ValueError(f"Case {spec.name} requires --image PATH")
            image_name = copy_input_image(Path(args.image).resolve(), Path(args.comfy_root), spec.name)
        else:
            image_name = ""
        if spec.kind == "probe":
            result = run_probe_case(args)
        elif spec.kind in {"cache", "pixal3d_cache"}:
            result = run_cache_case(spec, args, run["runId"], image_name, recorder)
        elif spec.kind == "pixal3d_reuse":
            result = run_pixal3d_reuse_case(
                spec, args, run["runId"], image_name, recorder
            )
        elif spec.kind == "strict1536":
            result = run_strict_1536_default_case(spec, args, run["runId"], image_name, recorder)
        elif spec.kind in {"cancel", "pixal3d_cancel"}:
            result = run_cancellation_case(spec, args, run["runId"], image_name, recorder)
        else:
            result = run_prompt_once(spec, args, run["runId"], image_name, recorder)
        candidate_record = {"kind": spec.kind, **result}
        require_geometry_evidence = bool(
            getattr(args, "require_geometry_evidence", False)
        )
        if require_geometry_evidence and not record_has_noncollapsed_geometry(candidate_record):
            raise RuntimeError(
                f"{spec.name} completed without passing non-collapsed geometry evidence"
            )
        benchmark = None
        if spec.benchmark:
            benchmark = benchmark_from(
                spec,
                float(result["runtimeSeconds"]),
                int(result["peakVramBytes"]),
                result["glb"],
                result.get("logExcerpt", ""),
                observed_resolution=(
                    (result.get("workerUltraShapeSettings") or {}).get(
                        "octree_resolution"
                    )
                ),
                texture_size=args.texture_size or spec.texture_size,
                pixal3d_worker_result=result.get("workerPixal3DResult"),
                require_geometry_metrics=require_geometry_evidence,
            )
        record = {
            "schema": CASE_SCHEMA,
            "status": "passed",
            "case": spec.name,
            "kind": spec.kind,
            "gate": spec.gate,
            "benchmarkName": spec.benchmark,
            "runId": run["runId"],
            "startedAt": recorder.current_path.exists() and read_json(recorder.current_path, {}).get("startedAt"),
            "completedAt": utc_now(),
            "wallSeconds": round(time.monotonic() - started, 3),
            "benchmark": benchmark,
            **result,
        }
        record["evidence"] = evidence_id(record)
        atomic_json(recorder.case_dir / "record.json", record)
        recorder.status("passed", runId=run["runId"], record=str(recorder.case_dir / "record.json"))
        recorder.event("passed", evidence=record["evidence"])
        return 0
    except BaseException as exc:
        failure = {
            "schema": CASE_SCHEMA,
            "status": "failed",
            "case": spec.name,
            "kind": spec.kind,
            "gate": spec.gate,
            "benchmarkName": spec.benchmark,
            "runId": run["runId"],
            "completedAt": utc_now(),
            "wallSeconds": round(time.monotonic() - started, 3),
            "errorType": type(exc).__name__,
            "error": str(exc),
        }
        atomic_json(recorder.case_dir / "record.json", failure)
        recorder.status("failed", runId=run["runId"], error=str(exc))
        recorder.event("failed", errorType=type(exc).__name__, error=str(exc))
        return 1


def pid_is_running(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def launch_case(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir).resolve()
    ensure_run(state_dir)
    recorder = Recorder(state_dir, args.case)
    current = read_json(recorder.current_path, {})
    if current.get("status") in {"launching", "running"} and pid_is_running(current.get("pid")):
        raise RuntimeError(f"Case {args.case} is already running as PID {current['pid']}")
    argv = [sys.executable, str(Path(__file__).resolve()), "run"]
    excluded = {"command", "func", "require_geometry_evidence"}
    for name, value in vars(args).items():
        if name in excluded or value is None or value is False:
            continue
        option = "--" + name.replace("_", "-")
        if value is True:
            argv.append(option)
        else:
            argv.extend([option, str(value)])
    log_path = recorder.case_dir / "runner.log"
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
    atomic_json(
        recorder.current_path,
        {
            "schema": STATE_SCHEMA,
            "case": args.case,
            "status": "launching",
            "pid": process.pid,
            "updatedAt": utc_now(),
            "log": str(log_path),
        },
    )
    print(json.dumps({"status": "launched", "case": args.case, "pid": process.pid, "log": str(log_path)}))
    return 0


def status_command(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir).resolve()
    cases = [args.case] if args.case else sorted(CASES)
    payload: dict[str, Any] = {"schema": STATE_SCHEMA, "run": read_json(state_dir / "run.json"), "cases": {}}
    for name in cases:
        case_dir = state_dir / "cases" / name
        current = read_json(case_dir / "current.json", {"case": name, "status": "not-started"})
        if current.get("status") in {"launching", "running"}:
            current["processAlive"] = pid_is_running(current.get("pid"))
        record = read_json(case_dir / "record.json")
        payload["cases"][name] = {"current": current, "record": record}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cancel_command(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir).resolve()
    recorder = Recorder(state_dir, args.case)
    current = read_json(recorder.current_path, {})
    pid = current.get("pid")
    if pid_is_running(pid):
        try:
            os.killpg(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    with contextlib.suppress(Exception):
        ApiClient(args.base_url).post("/interrupt")
    recorder.status("cancelled", previousPid=pid)
    recorder.event("cancelled", previousPid=pid)
    print(json.dumps({"status": "cancelled", "case": args.case, "pid": pid}))
    return 0


def merge_command(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir).resolve()
    template_path = Path(args.template).resolve()
    output_path = Path(args.output).resolve()
    template = read_json(template_path)
    if not isinstance(template, dict) or template.get("schema") != "comfycolab-3d-live-validation-v1":
        raise RuntimeError(f"Validation template has the wrong schema: {template_path}")
    run = read_json(state_dir / "run.json", {})
    passed: dict[str, dict[str, Any]] = {}
    # Parser-created CLI namespaces always enable semantic evidence. Direct
    # callers that omit this field retain the legacy structural-fixture path.
    require_geometry_evidence = bool(getattr(args, "require_geometry_evidence", False))
    for name in CASES:
        record = read_json(state_dir / "cases" / name / "record.json")
        if isinstance(record, dict) and record.get("status") == "passed" and record.get("runId") == run.get("runId"):
            if require_geometry_evidence and not record_has_noncollapsed_geometry(record):
                continue
            passed[name] = record
            gate = record.get("gate")
            if gate in template.get("gates", {}):
                template["gates"][gate] = {"status": "passed", "evidence": record["evidence"]}
            benchmark_name = record.get("benchmarkName")
            if benchmark_name in template.get("benchmarks", {}) and isinstance(record.get("benchmark"), dict):
                template["benchmarks"][benchmark_name] = record["benchmark"]
    trellis_proof = any(
        record.get("previewSaveProof", {}).get("saveArtifactValidated")
        for record in passed.values() if record.get("kind") in {"trellis", "advanced"}
    )
    ultra_proof = any(
        record.get("previewSaveProof", {}).get("saveArtifactValidated")
        for record in passed.values() if record.get("kind") in {"ultrashape", "full"}
    )
    if trellis_proof and ultra_proof:
        proof_records = sorted(
            record["evidence"] for record in passed.values()
            if record.get("kind") in {"trellis", "advanced", "ultrashape", "full", "pixal3d"}
            and record.get("previewSaveProof", {}).get("saveArtifactValidated")
        )
        digest = hashlib.sha256("\n".join(proof_records).encode("utf-8")).hexdigest()
        template["gates"]["preview_and_save_native_file3d"] = {
            "status": "passed", "evidence": f"live-g4:{run.get('runId')}:preview-save:{digest}"
        }
    gates_passed = all(
        isinstance(gate, dict) and gate.get("status") == "passed" and isinstance(gate.get("evidence"), str)
        for gate in template.get("gates", {}).values()
    )
    def benchmark_passed(value: Any) -> bool:
        if not isinstance(value, dict) or value.get("status") != "passed":
            return False
        if value.get("plyValidated") is True:
            return (
                value.get("file3dValidated") is True
                and int(value.get("gaussianCount", 0)) > 0
                and int(value.get("plyBytes", 0)) > 0
                and isinstance(value.get("plySha256"), str)
            )
        return (
            value.get("glbValidated") is True
            and value.get("nonCollapsedGeometryValidated") is True
            and isinstance(value.get("geometryMetrics"), dict)
            and value["geometryMetrics"].get("nonCollapsed") is True
        )

    benchmarks_passed = all(benchmark_passed(value) for value in template.get("benchmarks", {}).values())
    template["runId"] = run.get("runId")
    template["status"] = "passed" if gates_passed and benchmarks_passed else "pending"
    template["completedAt"] = utc_now() if template["status"] == "passed" else None
    atomic_json(output_path, template)
    print(json.dumps({"status": template["status"], "output": str(output_path), "passedCases": sorted(passed)}))
    return 0


def add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(require_geometry_evidence=True)
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--comfy-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling-steps", type=int, default=0)
    parser.add_argument("--target-face-count", type=int, default=0)
    parser.add_argument("--texture-size", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=49152)
    parser.add_argument("--remove-background", choices=("Auto", "On", "Off"), default="Auto")
    parser.add_argument("--camera-fov-degrees", type=float, default=0.0)
    parser.add_argument("--keep-worker-loaded", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-mode", choices=("Use cache", "Refresh this node", "Disable cache"), default="Disable cache")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--num-latents", type=int, default=0)
    parser.add_argument("--octree-resolution", type=int, default=0)
    parser.add_argument("--decode-chunk-size", type=int, default=0)
    parser.add_argument("--low-vram", choices=("Auto", "On", "Off"), default="Auto")
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--vram-interval", type=float, default=0.5)
    parser.add_argument("--cancel-after", type=float, default=2)
    parser.add_argument("--cancel-start-timeout", type=float, default=900)
    parser.add_argument("--trellis-python", type=Path, default=Path.home() / ".ce/.pixi/envs/trellis2-nodes/bin/python")
    parser.add_argument("--ultrashape-source", type=Path, default=Path("/content/UltraShape-1.0"))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run one staged case in the foreground.")
    add_run_options(run)
    run.set_defaults(func=execute_case)
    launch = commands.add_parser("launch", help="Launch one staged case in a detached process group.")
    add_run_options(launch)
    launch.set_defaults(func=launch_case)
    status = commands.add_parser("status", help="Print machine-readable case state.")
    status.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    status.add_argument("--case", choices=sorted(CASES))
    status.set_defaults(func=status_command)
    cancel = commands.add_parser("cancel", help="Stop a detached case and interrupt ComfyUI.")
    cancel.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    cancel.add_argument("--base-url", default=DEFAULT_BASE_URL)
    cancel.add_argument("--case", choices=sorted(CASES), required=True)
    cancel.set_defaults(func=cancel_command)
    merge = commands.add_parser("merge", help="Merge passed case records into the release validation JSON.")
    merge.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    merge.add_argument("--template", type=Path, default=Path("docs/3d-validation.json"))
    merge.add_argument("--output", type=Path, default=Path("docs/3d-validation.json"))
    merge.set_defaults(func=merge_command, require_geometry_evidence=True)
    listing = commands.add_parser("list-cases", help="List independently runnable validation cases.")
    listing.set_defaults(func=lambda _args: print("\n".join(sorted(CASES))) or 0)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"live_3d_g4_validation: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
