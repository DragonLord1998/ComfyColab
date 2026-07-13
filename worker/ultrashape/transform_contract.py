"""Pure-Python transform helpers shared by the worker and local tests.

Matrices use row-major storage and multiply column vectors. Keeping this module
free of NumPy, trimesh, torch, and ComfyUI imports makes the transform contract
testable on the Mac without initializing CUDA.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


TRANSFORM_SCHEMA = "comfycolab-3d-transform-v1"


def identity_matrix() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def apply_matrix_to_point(
    matrix: Sequence[Sequence[float]], point: Sequence[float]
) -> tuple[float, float, float]:
    if len(point) != 3:
        raise ValueError("A 3D point must contain exactly three values.")
    values = [float(point[0]), float(point[1]), float(point[2]), 1.0]
    result = [sum(float(matrix[row][col]) * values[col] for col in range(4)) for row in range(4)]
    if not math.isfinite(result[3]) or abs(result[3]) < 1e-12:
        raise ValueError("Transform produced an invalid homogeneous coordinate.")
    return tuple(result[index] / result[3] for index in range(3))


def multiply_matrices(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> list[list[float]]:
    return [
        [
            sum(float(left[row][index]) * float(right[index][column]) for index in range(4))
            for column in range(4)
        ]
        for row in range(4)
    ]


def normalization_from_bounds(
    minimum: Iterable[float],
    maximum: Iterable[float],
    *,
    normalize_scale: float = 0.99,
) -> dict[str, object]:
    """Return UltraShape's normalization and its exact inverse.

    Upstream UltraShape centers a mesh and scales its longest bounding-box side
    to ``2 * normalize_scale``. This function records the same operation before
    upstream mutates its private mesh copy.
    """

    lower = tuple(float(value) for value in minimum)
    upper = tuple(float(value) for value in maximum)
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError("Mesh bounds must contain three minimum and maximum values.")
    if not 0.0 < normalize_scale <= 1.0:
        raise ValueError("normalize_scale must be greater than zero and at most one.")
    if not all(math.isfinite(value) for value in lower + upper):
        raise ValueError("Mesh bounds must be finite.")

    extents = tuple(upper[index] - lower[index] for index in range(3))
    if any(extent < 0 for extent in extents):
        raise ValueError("Mesh maximum bounds must not be below minimum bounds.")
    longest_extent = max(extents)
    if longest_extent <= 1e-12:
        raise ValueError("Cannot normalize a mesh with zero spatial extent.")

    center = tuple((lower[index] + upper[index]) / 2.0 for index in range(3))
    scale = (2.0 * normalize_scale) / longest_extent
    inverse_scale = 1.0 / scale
    forward = identity_matrix()
    inverse = identity_matrix()
    for axis in range(3):
        forward[axis][axis] = scale
        forward[axis][3] = -center[axis] * scale
        inverse[axis][axis] = inverse_scale
        inverse[axis][3] = center[axis]

    return {
        "schema": TRANSFORM_SCHEMA,
        "minimum": list(lower),
        "maximum": list(upper),
        "center": list(center),
        "extent": list(extents),
        "longest_extent": longest_extent,
        "normalize_scale": normalize_scale,
        "forward": forward,
        "inverse": inverse,
    }


def y_up_to_z_up_matrix() -> list[list[float]]:
    """Rotate glTF Y-up geometry +90 degrees around X into Z-up."""

    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def z_up_to_y_up_matrix() -> list[list[float]]:
    """Rotate internal Z-up geometry -90 degrees around X into glTF Y-up."""

    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matrices_are_inverse(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    *,
    tolerance: float = 1e-9,
) -> bool:
    product = multiply_matrices(left, right)
    identity = identity_matrix()
    return all(
        abs(product[row][column] - identity[row][column]) <= tolerance
        for row in range(4)
        for column in range(4)
    )
