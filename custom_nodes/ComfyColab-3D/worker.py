from __future__ import annotations

import json
import hashlib
import math
import os
import queue
import signal
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .cache import atomic_write_bytes
from .geometry_quality import GEOMETRY_QUALITY_SCHEMA, validate_volumetric_glb


TRANSFORM_SCHEMA = "comfycolab-3d-transform-v1"
GEOMETRY_CACHE_SCHEMA = "comfycolab-ultrashape-geometry-cache-v2"


class UltraShapeNoDecodableSurfaceError(RuntimeError):
    """UltraShape exhausted its adaptive decode without finding a surface."""


@dataclass(frozen=True)
class UltraShapeCommand:
    python: str
    worker_script: str
    source_dir: str
    checkpoint: str
    dinov2_dir: str
    input_mesh: str
    reference_image: str
    output_mesh: str
    metadata_output: str
    steps: int
    num_latents: int
    octree_resolution: int
    decode_chunk_size: int
    seed: int
    low_vram: str
    checkpoint_sha256: str = ""

    def argv(
        self,
        output_override: str | None = None,
        metadata_override: str | None = None,
    ) -> list[str]:
        output = output_override or self.output_mesh
        metadata = metadata_override or self.metadata_output
        return [
            self.python,
            self.worker_script,
            "--source-dir", self.source_dir,
            "--checkpoint", self.checkpoint,
            "--checkpoint-sha256", self.checkpoint_sha256,
            "--dinov2-dir", self.dinov2_dir,
            "--input-mesh", self.input_mesh,
            "--reference-image", self.reference_image,
            "--output-mesh", output,
            "--metadata-output", metadata,
            "--steps", str(self.steps),
            "--num-latents", str(self.num_latents),
            "--octree-resolution", str(self.octree_resolution),
            "--decode-chunk-size", str(self.decode_chunk_size),
            "--seed", str(self.seed),
            "--low-vram", self.low_vram,
        ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_transform_metadata(path: str | Path) -> dict:
    path = Path(path)
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("UltraShape transform metadata is missing or invalid") from error
    if metadata.get("schema") != TRANSFORM_SCHEMA:
        raise ValueError("UltraShape transform metadata has an unsupported schema")
    if metadata.get("output_space") != "gltf-y-up-restored-world":
        raise ValueError("UltraShape transform metadata has an unexpected output space")
    if metadata.get("geometry_only") is not True:
        raise ValueError("UltraShape geometry metadata must mark the result geometry-only")
    normalization = metadata.get("ultrashape_normalization")
    if not isinstance(normalization, dict) or normalization.get("schema") != TRANSFORM_SCHEMA:
        raise ValueError("UltraShape normalization metadata is missing")
    for name in ("forward", "inverse"):
        matrix = normalization.get(name)
        if (
            not isinstance(matrix, list)
            or len(matrix) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
            or any(not math.isfinite(float(value)) for row in matrix for value in row)
        ):
            raise ValueError(f"UltraShape {name} transform is invalid")
    validation = metadata.get("validation")
    if not isinstance(validation, dict) or any(
        int(validation.get(name, 0)) <= 0 for name in ("bytes", "vertices", "faces")
    ):
        raise ValueError("UltraShape geometry validation metadata is incomplete")
    return metadata


def write_geometry_cache_record(
    directory: str | Path,
    cache_key: str,
    *,
    stage: str = "worker-refined",
) -> dict:
    directory = Path(directory)
    geometry = directory / "geometry.glb"
    transform = directory / "transform.json"
    metrics = validate_volumetric_glb(
        geometry,
        stage="UltraShape geometry cache write",
        require_material=True,
    )
    validate_transform_metadata(transform)
    record = {
        "schema": GEOMETRY_CACHE_SCHEMA,
        "cache_key": cache_key,
        "geometry_sha256": _sha256_file(geometry),
        "transform_sha256": _sha256_file(transform),
        "transform_schema": TRANSFORM_SCHEMA,
        "geometry_only": True,
        "stage": stage,
        "geometry_quality_schema": GEOMETRY_QUALITY_SCHEMA,
        "geometry_metrics": metrics.to_dict() if hasattr(metrics, "to_dict") else metrics,
    }
    atomic_write_bytes(
        directory / "record.json",
        (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return record


def validate_geometry_cache_record(
    directory: str | Path,
    cache_key: str,
    *,
    required_stage: str | None = None,
) -> bool:
    directory = Path(directory)
    try:
        record = json.loads((directory / "record.json").read_text(encoding="utf-8"))
        if (
            record.get("schema") != GEOMETRY_CACHE_SCHEMA
            or record.get("cache_key") != cache_key
            or record.get("transform_schema") != TRANSFORM_SCHEMA
            or record.get("geometry_only") is not True
            or record.get("geometry_quality_schema") != GEOMETRY_QUALITY_SCHEMA
            or (required_stage is not None and record.get("stage") != required_stage)
        ):
            return False
        geometry = directory / "geometry.glb"
        transform = directory / "transform.json"
        validate_volumetric_glb(
            geometry,
            stage="cached UltraShape geometry",
            require_material=True,
        )
        validate_transform_metadata(transform)
        return (
            _sha256_file(geometry) == record.get("geometry_sha256")
            and _sha256_file(transform) == record.get("transform_sha256")
        )
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return False


def atomic_replace_cache_directory(staging: str | Path, destination: str | Path) -> None:
    staging = Path(staging)
    destination = Path(destination)
    backup = destination.with_name(f".{destination.name}.previous")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists() and not destination.exists():
        os.replace(backup, destination)
    elif backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        shutil.rmtree(backup, ignore_errors=True)


def _reader(stream, output: queue.Queue[str | None]) -> None:
    try:
        for line in iter(stream.readline, ""):
            output.put(line.rstrip("\n"))
    finally:
        output.put(None)


def _terminate_group(process: subprocess.Popen, timeout: float = 5.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=timeout)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=timeout)


def _worker_reported_error(result: dict, *, fallback_seed: int | None = None) -> RuntimeError:
    error_type = str(result.get("error_type", ""))
    error_code = str(result.get("error_code", ""))
    if error_code == "no_decodable_surface" or error_type == "NoDecodableSurface":
        resolution = result.get("octree_resolution", "unknown")
        depth = result.get("octree_depth", "unknown")
        preceding = result.get("preceding_active_points", "unknown")
        seed = result.get("seed", fallback_seed if fallback_seed is not None else "unknown")
        return UltraShapeNoDecodableSurfaceError(
            "UltraShape could not decode a surface "
            f"(requested_resolution={resolution}, decode_stage_resolution={depth}, "
            f"preceding_active_points={preceding}, seed={seed}). "
            "Use a volumetric input mesh, try the conservative 512 detail tier, or choose another seed."
        )
    message = str(result.get("error") or "unknown worker failure")
    return RuntimeError(f"UltraShape worker failed: {error_type or 'RuntimeError'}: {message}")


def run_ultrashape_worker(
    command: UltraShapeCommand,
    *,
    is_cancelled: Callable[[], bool] = lambda: False,
    on_progress: Callable[[dict], None] = lambda _event: None,
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    poll_interval: float = 0.1,
) -> dict:
    output = Path(command.output_mesh)
    partial = output.with_name(f".{output.stem}.{os.getpid()}.partial.glb")
    metadata = Path(command.metadata_output)
    partial_metadata = metadata.with_name(f".{metadata.stem}.{os.getpid()}.partial.json")
    partial_artifacts = (
        output.with_suffix(output.suffix + ".partial"),
        output.with_name(output.stem + ".partial" + output.suffix),
        output.with_name(output.name + ".partial"),
        partial_metadata,
        metadata.with_suffix(metadata.suffix + ".partial"),
        metadata.with_name(metadata.stem + ".partial" + metadata.suffix),
        metadata.with_name(metadata.name + ".partial"),
    )
    lines: queue.Queue[str | None] = queue.Queue()
    result: dict = {}
    process = popen_factory(
        command.argv(str(partial), str(partial_metadata)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    if process.stdout is None:
        raise RuntimeError("UltraShape worker stdout pipe is unavailable")
    reader = threading.Thread(target=_reader, args=(process.stdout, lines), daemon=True)
    reader.start()
    tail: list[str] = []
    try:
        stream_closed = False
        while process.poll() is None or not stream_closed:
            if is_cancelled():
                _terminate_group(process)
                raise InterruptedError("UltraShape refinement was cancelled")
            try:
                line = lines.get(timeout=poll_interval)
            except queue.Empty:
                continue
            if line is None:
                stream_closed = True
                continue
            tail = (tail + [line])[-40:]
            if line.startswith("COMFYCOLAB_PROGRESS="):
                on_progress(json.loads(line.split("=", 1)[1]))
            elif line.startswith("COMFYCOLAB_RESULT="):
                result = json.loads(line.split("=", 1)[1])
        return_code = process.wait()
        if result and result.get("status") != "ok":
            raise _worker_reported_error(result, fallback_seed=command.seed)
        if return_code:
            raise RuntimeError(f"UltraShape worker exited with {return_code}: {' | '.join(tail)}")
        if not result:
            raise RuntimeError("UltraShape worker exited without COMFYCOLAB_RESULT")
        if Path(str(result.get("output_mesh", ""))).resolve() != partial.resolve():
            raise RuntimeError("UltraShape worker reported an unexpected output path")
        if Path(str(result.get("metadata_output", ""))).resolve() != partial_metadata.resolve():
            raise RuntimeError("UltraShape worker reported an unexpected metadata path")
        validate_volumetric_glb(
            partial,
            stage="UltraShape worker output",
            require_material=True,
        )
        validate_transform_metadata(partial_metadata)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, output)
        os.replace(partial_metadata, metadata)
        result["metadata_output"] = str(metadata)
        result["output_mesh"] = str(output)
        return result
    except BaseException:
        _terminate_group(process)
        for artifact in (metadata, *partial_artifacts):
            artifact.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        raise
    finally:
        partial.unlink(missing_ok=True)
