from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any


def _stable_value(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "numpy"):
        array = value.detach().cpu().contiguous().numpy()
        return {
            "kind": "array",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and Path(value).is_file():
        path = Path(value)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return {"kind": "file", "sha256": digest.hexdigest(), "size": path.stat().st_size}
    if isinstance(value, dict):
        return {str(key): _stable_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def deterministic_cache_key(stage: str, **inputs: Any) -> str:
    payload = json.dumps(
        {"stage": stage, "inputs": _stable_value(inputs)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def trellis_cache_key(
    image: Any,
    *,
    settings: Any,
    seed: int,
    remove_background: str,
    comfyui_ref: str,
    trellis_ref: str,
    trellis_patch_id: str,
    birefnet_ref: str,
) -> str:
    return deterministic_cache_key(
        "trellis",
        image=image,
        mask_background_policy=remove_background,
        seed=seed,
        resolution=settings.resolution,
        sampling_steps=settings.sampling_steps,
        target_face_count=settings.target_face_count,
        texture_size=settings.texture_size,
        max_tokens=settings.max_tokens,
        comfyui_ref=comfyui_ref,
        trellis_ref=trellis_ref,
        trellis_patch_id=trellis_patch_id,
        birefnet_ref=birefnet_ref,
    )


def ultrashape_geometry_cache_key(
    source_geometry_digest: str,
    reference_image: Any,
    *,
    detail: str,
    seed: int,
    steps: int,
    num_latents: int,
    octree_resolution: int,
    decode_chunk_size: int,
    low_vram: str,
    worker_ref: str,
    checkpoint_ref: str,
    dinov2_ref: str,
    transform_schema: str,
) -> str:
    return deterministic_cache_key(
        "ultrashape",
        canonical_geometry=source_geometry_digest,
        reference_image=reference_image,
        detail=detail,
        seed=seed,
        steps=steps,
        num_latents=num_latents,
        octree_resolution=octree_resolution,
        decode_chunk_size=decode_chunk_size,
        low_vram=low_vram,
        worker_ref=worker_ref,
        checkpoint_ref=checkpoint_ref,
        dinov2_ref=dinov2_ref,
        transform_schema=transform_schema,
    )


def texture_cache_key(
    refined_geometry_digest: str,
    reference_image: Any,
    *,
    seed: int,
    target_face_count: int,
    texture_size: int,
    texture_sampling_steps: int,
    trellis_ref: str,
) -> str:
    return deterministic_cache_key(
        "texture",
        canonical_refined_geometry=refined_geometry_digest,
        reference_image=reference_image,
        seed=seed,
        target_face_count=target_face_count,
        texture_size=texture_size,
        texture_sampling_steps=texture_sampling_steps,
        trellis_ref=trellis_ref,
    )


def canonical_trimesh_digest(mesh: Any) -> str:
    """Hash geometry only; material/texture encoding and object repr never affect the key."""
    digest = hashlib.sha256()
    for label, array in ((b"vertices", mesh.vertices), (b"faces", mesh.faces)):
        canonical = array.copy()
        if label == b"vertices":
            canonical = canonical.round(7)
        canonical = canonical.astype("<f8" if label == b"vertices" else "<i8", copy=False)
        digest.update(label)
        digest.update(str(tuple(canonical.shape)).encode())
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def canonical_glb_geometry_digest(path: str | Path) -> str:
    trimesh = importlib.import_module("trimesh")
    from .file3d import bake_scene_mesh

    loaded = trimesh.load(str(path), force="scene")
    mesh = bake_scene_mesh(loaded, trimesh)
    return canonical_trimesh_digest(mesh)


def atomic_write_bytes(destination: str | Path, data: bytes) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    try:
        with partial.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def cache_path(root: str | Path, stage: str, key: str, filename: str = "model.glb") -> Path:
    if not key or any(character not in "0123456789abcdef" for character in key):
        raise ValueError("Cache keys must be lowercase hexadecimal")
    safe_stage = "".join(character for character in stage if character.isalnum() or character in "-_")
    if not safe_stage:
        raise ValueError("Cache stage must contain a safe character")
    if Path(filename).name != filename:
        raise ValueError("Cache filename must not contain a directory")
    return Path(root) / safe_stage / key / filename
