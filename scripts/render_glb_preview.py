#!/usr/bin/env python3
"""Render deterministic multi-angle GLB previews without a GPU renderer."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


_COMPONENT_DTYPES = {
    5120: np.dtype("<i1"),
    5121: np.dtype("<u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _parse_glb(path: Path) -> tuple[dict, bytes]:
    payload = path.read_bytes()
    if len(payload) < 20:
        raise ValueError("GLB is truncated")
    magic, version, length = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or length != len(payload):
        raise ValueError("GLB header is invalid")
    document = None
    binary = b""
    offset = 12
    while offset < len(payload):
        chunk_length, chunk_type = struct.unpack_from("<I4s", payload, offset)
        offset += 8
        chunk = payload[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == b"JSON":
            document = json.loads(chunk.decode("utf-8").rstrip(" \t\r\n\x00"))
        elif chunk_type == b"BIN\x00":
            binary = chunk
    if not isinstance(document, dict) or not binary:
        raise ValueError("GLB must contain JSON and embedded binary chunks")
    return document, binary


def _accessor(document: dict, binary: bytes, index: int) -> np.ndarray:
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    dtype = _COMPONENT_DTYPES[int(accessor["componentType"])]
    width = _WIDTHS[str(accessor["type"])]
    count = int(accessor["count"])
    packed = dtype.itemsize * width
    stride = int(view.get("byteStride", packed))
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    if stride == packed:
        return np.frombuffer(binary, dtype=dtype, count=count * width, offset=offset).reshape(
            count, width
        )
    rows = np.empty((count, width), dtype=dtype)
    for row in range(count):
        rows[row] = np.frombuffer(
            binary, dtype=dtype, count=width, offset=offset + row * stride
        )
    return rows


def load_geometry(path: Path) -> tuple[np.ndarray, np.ndarray]:
    document, binary = _parse_glb(path)
    vertices = []
    faces = []
    vertex_offset = 0
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            positions = _accessor(document, binary, primitive["attributes"]["POSITION"]).astype(
                np.float64
            )
            indices = _accessor(document, binary, primitive["indices"]).reshape(-1).astype(
                np.int64
            )
            if len(indices) % 3:
                raise ValueError("GLB triangle index count is not divisible by three")
            vertices.append(positions)
            faces.append(indices.reshape(-1, 3) + vertex_offset)
            vertex_offset += len(positions)
    if not vertices:
        raise ValueError("GLB contains no triangle geometry")
    return np.concatenate(vertices), np.concatenate(faces)


def _camera_basis(yaw_degrees: float, pitch_degrees: float) -> np.ndarray:
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    direction = np.array(
        [
            math.sin(yaw) * math.cos(pitch),
            -math.cos(yaw) * math.cos(pitch),
            math.sin(pitch),
        ],
        dtype=np.float64,
    )
    forward = -direction / np.linalg.norm(direction)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1.0e-8:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack((right, up, forward), axis=1)


def render(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    yaw: float,
    pitch: float,
    size: int,
    max_faces: int,
) -> Image.Image:
    centered = vertices - (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    basis = _camera_basis(yaw, pitch)
    camera_vertices = centered @ basis
    triangles = camera_vertices[faces]
    if len(triangles) > max_faces:
        step = math.ceil(len(triangles) / max_faces)
        triangles = triangles[::step]
    span = max(
        float(np.ptp(camera_vertices[:, 0])),
        float(np.ptp(camera_vertices[:, 1])),
        1.0e-8,
    )
    scale = size * 0.82 / span
    projected = triangles[:, :, :2] * np.array([scale, -scale])
    projected += size * 0.5
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edges_a, edges_b)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1.0e-12
    normals[valid] /= lengths[valid, None]
    light = np.array([-0.35, 0.45, 0.82])
    light /= np.linalg.norm(light)
    brightness = np.clip(np.abs(normals @ light) * 0.72 + 0.25, 0.18, 0.97)
    depth = triangles[:, :, 2].mean(axis=1)
    order = np.argsort(depth)
    image = Image.new("RGB", (size, size), (238, 241, 245))
    draw = ImageDraw.Draw(image)
    for index in order:
        if not valid[index]:
            continue
        shade = int(brightness[index] * 210)
        color = (min(255, shade + 28), min(255, shade + 12), max(0, shade - 35))
        draw.polygon([tuple(point) for point in projected[index]], fill=color)
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--max-faces", type=int, default=80_000)
    args = parser.parse_args()
    vertices, faces = load_geometry(args.input)
    views = [
        ("front", 0.0, 0.0),
        ("left", -90.0, 0.0),
        ("back", 180.0, 0.0),
        ("right", 90.0, 0.0),
        ("iso", 35.0, 18.0),
    ]
    previews = [
        render(
            vertices,
            faces,
            yaw=yaw,
            pitch=pitch,
            size=args.size,
            max_faces=args.max_faces,
        )
        for _name, yaw, pitch in views
    ]
    sheet = Image.new("RGB", (args.size * len(previews), args.size), "white")
    draw = ImageDraw.Draw(sheet)
    for column, ((name, _yaw, _pitch), preview) in enumerate(zip(views, previews)):
        sheet.paste(preview, (column * args.size, 0))
        draw.text((column * args.size + 18, 18), name, fill=(20, 24, 31))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "vertices": int(len(vertices)),
                "faces": int(len(faces)),
                "views": [name for name, _yaw, _pitch in views],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
