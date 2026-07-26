from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def large_planar_component_face_mask(
    vertices: Any,
    faces: Any,
    *,
    singular_ratio_limit: float = 0.01,
    minimum_span_fraction: float = 0.5,
    minimum_face_count: int = 512,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Reject only large, independently connected sheet-like components."""

    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
        raise ValueError("Mesh cleanup vertices must have shape [N, 3]")
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise ValueError("Mesh cleanup faces must have shape [M, 3]")
    if face_array.size == 0:
        raise ValueError("Mesh cleanup requires at least one face")
    if face_array.min() < 0 or face_array.max() >= len(vertex_array):
        raise ValueError("Mesh cleanup faces contain an out-of-range vertex index")

    referenced = np.unique(face_array.reshape(-1))
    global_extent = float(np.ptp(vertex_array[referenced], axis=0).max())
    if not np.isfinite(global_extent) or global_extent <= 0.0:
        raise ValueError("Mesh cleanup requires finite non-collapsed vertices")

    parent = np.arange(len(vertex_array), dtype=np.int64)
    component_size = np.ones(len(vertex_array), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return
        if component_size[first_root] < component_size[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        component_size[first_root] += component_size[second_root]

    for first, second, third in face_array:
        union(int(first), int(second))
        union(int(second), int(third))

    components: dict[int, list[int]] = defaultdict(list)
    for face_index, face in enumerate(face_array):
        components[find(int(face[0]))].append(face_index)

    keep = np.ones(len(face_array), dtype=bool)
    removed = []
    for face_indices in components.values():
        if len(face_indices) < int(minimum_face_count):
            continue
        component_faces = face_array[np.asarray(face_indices, dtype=np.int64)]
        component_vertex_indices = np.unique(component_faces.reshape(-1))
        points = vertex_array[component_vertex_indices]
        extents = np.ptp(points, axis=0)
        centered = points - points.mean(axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        singular_ratio = (
            float(singular_values[-1] / singular_values[0])
            if singular_values[0] > 0.0
            else 0.0
        )
        second_largest_extent = float(np.sort(extents)[-2])
        if (
            singular_ratio < float(singular_ratio_limit)
            and second_largest_extent
            >= global_extent * float(minimum_span_fraction)
        ):
            keep[np.asarray(face_indices, dtype=np.int64)] = False
            removed.append(
                {
                    "faces": len(face_indices),
                    "vertices": len(component_vertex_indices),
                    "extents": extents.tolist(),
                    "singular_ratio": singular_ratio,
                }
            )
    return keep, removed


__all__ = ["large_planar_component_face_mask"]
