#!/usr/bin/env python3
"""Stage and record live ComfyColab 3D validation runs on a Colab G4.

The runner intentionally uses ComfyUI's HTTP API rather than a websocket.  A
case can therefore be launched into its own process group and continue after a
Colab CLI/websocket client disconnects.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
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


STATE_SCHEMA = "comfycolab-3d-live-run-state-v1"
CASE_SCHEMA = "comfycolab-3d-live-case-v1"
EVENT_PREFIX = "COMFYCOLAB_LIVE3D="
SHAPE_METRICS = re.compile(
    r"ComfyColab shape metrics:\s*(?P<tokens>\d+)\s+tokens\s+at\s+resolution\s+(?P<resolution>\d+)"
)
DEFAULT_STATE_DIR = Path("/content/.comfycolab/live-3d-validation")
DEFAULT_COMFY_ROOT = Path("/content/ComfyUI")
DEFAULT_LOG = Path("/content/.comfycolab/comfyui.log")
DEFAULT_BASE_URL = "http://127.0.0.1:8188"


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
    octree_resolution: int = 384
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
        "trellis_1536_default_cap", "strict1536", resolution="1536_cascade",
        actual_resolution=1536, quality="1536 — Maximum", texture_size=4096,
        require_textured=True,
    ),
    "ultrashape_384": CaseSpec(
        "ultrashape_384", "ultrashape", "ultrashape_384_refinement",
        "ultrashape_384", actual_resolution=384, detail="Fast", octree_resolution=384,
    ),
    "ultrashape_512": CaseSpec(
        "ultrashape_512", "ultrashape", "ultrashape_512_refinement",
        "ultrashape_512", actual_resolution=512, detail="Fast", octree_resolution=512,
    ),
    "ultrashape_1024_run_1": CaseSpec(
        "ultrashape_1024_run_1", "ultrashape", "ultrashape_1024_run_1",
        "ultrashape_1024_run_1", actual_resolution=1024, detail="Detailed", octree_resolution=1024,
    ),
    "ultrashape_1024_run_2": CaseSpec(
        "ultrashape_1024_run_2", "ultrashape", "ultrashape_1024_run_2",
        "ultrashape_1024_run_2", actual_resolution=1024, detail="Detailed", octree_resolution=1024,
    ),
    "full_workflow_hard_surface": CaseSpec(
        "full_workflow_hard_surface", "full", "full_workflow_hard_surface",
        resolution="512", actual_resolution=512, retexture=True, require_textured=True,
    ),
    "full_workflow_organic": CaseSpec(
        "full_workflow_organic", "full", "full_workflow_organic",
        resolution="512", actual_resolution=512, retexture=True, require_textured=True,
    ),
    "full_workflow_thin": CaseSpec(
        "full_workflow_thin", "full", "full_workflow_thin",
        resolution="512", actual_resolution=512, retexture=True, require_textured=True,
    ),
    "full_workflow_holed": CaseSpec(
        "full_workflow_holed", "full", "full_workflow_holed",
        resolution="512", actual_resolution=512, retexture=True, require_textured=True,
    ),
    "full_workflow_transparent_background": CaseSpec(
        "full_workflow_transparent_background", "full",
        "full_workflow_transparent_background", resolution="512", actual_resolution=512,
        retexture=True, require_textured=True,
    ),
    "cache_hit_no_inference": CaseSpec(
        "cache_hit_no_inference", "cache", "cache_hit_no_inference",
        resolution="512", actual_resolution=512, require_textured=True,
    ),
    "cancellation_cleanup": CaseSpec(
        "cancellation_cleanup", "cancel", "cancellation_cleanup",
        actual_resolution=1024, detail="Detailed", octree_resolution=1024,
    ),
    "advanced_trellis_workflow": CaseSpec(
        "advanced_trellis_workflow", "advanced", "advanced_trellis_workflow",
        resolution="512", actual_resolution=512, require_textured=True,
    ),
    "combined_environment_cuda_probes": CaseSpec(
        "combined_environment_cuda_probes", "probe", "combined_environment_cuda_probes",
    ),
}


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


def add_preview_and_save(prompt: dict[str, Any], source: str, prefix: str) -> None:
    prompt["90"] = {"class_type": "Preview3D", "inputs": {"model_file": [source, 0]}}
    prompt["91"] = {
        "class_type": "SaveGLB",
        "inputs": {"mesh": [source, 0], "filename_prefix": prefix},
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
    elif spec.kind == "advanced":
        prompt.update(build_advanced_nodes(args))
        output_node = "9"
    else:
        raise ValueError(f"Case {spec.name} does not use a ComfyUI prompt")
    prefix = f"3d/validation/{run_id}-{spec.name}"
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


def check_object_info(api: ApiClient, prompt: dict[str, Any], source_node: str) -> dict[str, Any]:
    info = api.get("/object_info")
    required = {value["class_type"] for value in prompt.values()}
    missing = sorted(node for node in required if node not in info)
    if missing:
        raise RuntimeError(f"ComfyUI is missing required nodes: {', '.join(missing)}")
    facade_type = prompt[source_node]["class_type"]
    facade_outputs = info.get(facade_type, {}).get("output", [])
    preview_input = info.get("Preview3D", {}).get("input", {}).get("required", {}).get("model_file")
    save_input = info.get("SaveGLB", {}).get("input", {}).get("required", {}).get("mesh")
    preview_type = preview_input[0] if isinstance(preview_input, list) and preview_input else preview_input
    save_type = save_input[0] if isinstance(save_input, list) and save_input else save_input
    output_type = facade_outputs[0] if facade_outputs else None
    if output_type != "FILE_3D_GLB":
        raise RuntimeError(f"{facade_type} did not expose FILE_3D_GLB (got {output_type!r})")
    for label, accepted in (("Preview3D", preview_type), ("SaveGLB", save_type)):
        if "FILE_3D_GLB" not in str(accepted):
            raise RuntimeError(f"{label} does not accept FILE_3D_GLB (got {accepted!r})")
    if prompt["90"]["inputs"]["model_file"] != [source_node, 0]:
        raise RuntimeError("Preview3D is not connected to the facade output")
    if prompt["91"]["inputs"]["mesh"] != [source_node, 0]:
        raise RuntimeError("SaveGLB is not connected to the facade output")
    return {
        "facade": facade_type,
        "outputType": output_type,
        "previewAcceptedType": preview_type,
        "saveAcceptedType": save_type,
        "previewNode": "90",
        "saveNode": "91",
    }


def queue_prompt(api: ApiClient, prompt: dict[str, Any], client_id: str) -> str:
    response = api.post("/prompt", {"prompt": prompt, "client_id": client_id})
    prompt_id = response.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise RuntimeError(f"ComfyUI did not return a prompt ID: {response}")
    return prompt_id


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
    for path in output_root.rglob("*.glb"):
        try:
            stat = path.stat()
            result[path.resolve()] = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            pass
    return result


def changed_glbs(before: dict[Path, tuple[int, int]], output_root: Path) -> list[Path]:
    after = output_snapshot(output_root)
    return sorted(
        (path for path, value in after.items() if before.get(path) != value),
        key=lambda item: after[item][0],
    )


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


def inspect_glb(path: Path, *, require_textured: bool) -> dict[str, Any]:
    document, binary = _parse_glb(path)
    accessors = document.get("accessors") or []
    materials = document.get("materials") or []
    textures = document.get("textures") or []
    images = document.get("images") or []
    faces = vertices = primitives = 0
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
            if actual_vertex_count != vertex_count or not all(
                isinstance(value, (int, float)) and float("-inf") < float(value) < float("inf")
                for row in position_rows for value in row
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
        if require_shape_marker and SHAPE_METRICS.search(text):
            return text, end
        if now - last_change >= settle_seconds:
            return text, end
        if now >= deadline:
            return text, end
        time.sleep(0.25)


def source_node_for(spec: CaseSpec) -> str:
    if spec.kind in {"trellis", "cache", "strict1536"}:
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
    requested_resolution: int | None = None,
    texture_size: int | None = None,
) -> dict[str, Any] | None:
    if not spec.benchmark:
        return None
    if runtime <= 0 or peak_vram <= 0 or int(glb.get("bytes", 0)) <= 0 or int(glb.get("faces", 0)) <= 0:
        raise RuntimeError(
            "Benchmark metrics are incomplete; runtime, peak VRAM, GLB bytes, and faces must be positive"
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
    else:
        actual_resolution = requested_resolution or spec.actual_resolution
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
    }
    if spec.kind == "trellis":
        benchmark.update(tokens=tokens, textureSize=texture_size or spec.texture_size)
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
    prompt = build_prompt(spec, args, image_name, run_id, cache_mode=cache_mode)
    source_node = source_node_for(spec)
    proof = check_object_info(api, prompt, source_node)
    output_root = Path(args.comfy_root) / "output"
    before = output_snapshot(output_root)
    log_offset = Path(args.comfy_log).stat().st_size if Path(args.comfy_log).exists() else 0
    started = time.monotonic()
    with VramSampler(args.vram_interval) as sampler:
        prompt_id = queue_prompt(api, prompt, f"comfycolab-live3d-{run_id}-{spec.name}")
        recorder.event("queued", promptId=prompt_id)
        history = wait_prompt(api, prompt_id, args.timeout, recorder)
    runtime = time.monotonic() - started
    log_text, _ = read_settled_log_since(
        Path(args.comfy_log),
        log_offset,
        require_shape_marker=spec.kind == "trellis",
    )
    files = changed_glbs(before, output_root)
    if not files:
        raise RuntimeError("ComfyUI completed but produced no new or changed GLB")
    validated = [inspect_glb(path, require_textured=spec.require_textured) for path in files]
    proof.update(historyCompleted=True, saveArtifactValidated=True)
    primary = max(validated, key=lambda item: item["bytes"])
    return {
        "promptId": prompt_id,
        "runtimeSeconds": round(runtime, 3),
        "peakVramBytes": sampler.peak_bytes,
        "historyStatus": history.get("status"),
        "previewSaveProof": proof,
        "glb": primary,
        "resultFiles": validated,
        "logExcerpt": log_text[-12000:],
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
    return {
        **second,
        "cacheProof": {
            "firstPromptId": first["promptId"],
            "secondPromptId": second["promptId"],
            "secondRunShapeMetricCount": 0,
            "firstRuntimeSeconds": first["runtimeSeconds"],
            "secondRuntimeSeconds": second["runtimeSeconds"],
            "noModelInference": True,
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
    validated = [inspect_glb(path, require_textured=True) for path in files]
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


def worker_pids() -> list[int]:
    try:
        result = subprocess.run(
            ["pgrep", "-f", "worker/ultrashape/worker_main.py"], capture_output=True, text=True, timeout=5
        )
        return [int(line) for line in result.stdout.splitlines() if line.strip().isdigit() and int(line) != os.getpid()]
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return []


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
    deadline = time.monotonic() + args.cancel_start_timeout
    started_worker = False
    while time.monotonic() < deadline:
        if worker_pids():
            started_worker = True
            break
        time.sleep(1)
    if not started_worker:
        raise RuntimeError("UltraShape worker did not start before the cancellation deadline")
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
    retained_workers = worker_pids()
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
        elif spec.kind == "cache":
            result = run_cache_case(spec, args, run["runId"], image_name, recorder)
        elif spec.kind == "strict1536":
            result = run_strict_1536_default_case(spec, args, run["runId"], image_name, recorder)
        elif spec.kind == "cancel":
            result = run_cancellation_case(spec, args, run["runId"], image_name, recorder)
        else:
            result = run_prompt_once(spec, args, run["runId"], image_name, recorder)
        benchmark = None
        if spec.benchmark:
            benchmark = benchmark_from(
                spec,
                float(result["runtimeSeconds"]),
                int(result["peakVramBytes"]),
                result["glb"],
                result.get("logExcerpt", ""),
                requested_resolution=args.octree_resolution or spec.octree_resolution,
                texture_size=args.texture_size or spec.texture_size,
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
    excluded = {"command", "func"}
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
    for name in CASES:
        record = read_json(state_dir / "cases" / name / "record.json")
        if isinstance(record, dict) and record.get("status") == "passed" and record.get("runId") == run.get("runId"):
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
            if record.get("kind") in {"trellis", "advanced", "ultrashape", "full"}
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
    benchmarks_passed = all(
        isinstance(value, dict) and value.get("status") == "passed" and value.get("glbValidated") is True
        for value in template.get("benchmarks", {}).values()
    )
    template["runId"] = run.get("runId")
    template["status"] = "passed" if gates_passed and benchmarks_passed else "pending"
    template["completedAt"] = utc_now() if template["status"] == "passed" else None
    atomic_json(output_path, template)
    print(json.dumps({"status": template["status"], "output": str(output_path), "passedCases": sorted(passed)}))
    return 0


def add_run_options(parser: argparse.ArgumentParser) -> None:
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
    merge.set_defaults(func=merge_command)
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
