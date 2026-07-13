from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


def z_up_to_y_up(vertices: Iterable[Sequence[float]]) -> list[tuple[float, float, float]]:
    return [(float(x), float(z), -float(y)) for x, y, z in vertices]


def y_up_to_z_up(vertices: Iterable[Sequence[float]]) -> list[tuple[float, float, float]]:
    return [(float(x), -float(z), float(y)) for x, y, z in vertices]


@dataclass(frozen=True)
class Normalization:
    center: tuple[float, float, float]
    scale: float


def normalization_for(vertices: Iterable[Sequence[float]], normalize_scale: float = 0.99999) -> Normalization:
    rows = [tuple(map(float, row)) for row in vertices]
    if not rows or any(len(row) != 3 for row in rows):
        raise ValueError("Expected at least one XYZ vertex")
    if not all(math.isfinite(component) for row in rows for component in row):
        raise ValueError("Mesh vertices must be finite")
    minimum = tuple(min(row[axis] for row in rows) for axis in range(3))
    maximum = tuple(max(row[axis] for row in rows) for axis in range(3))
    extent = max(maximum[axis] - minimum[axis] for axis in range(3))
    if extent <= 0:
        raise ValueError("Cannot normalize a zero-extent mesh")
    center = tuple((minimum[axis] + maximum[axis]) / 2.0 for axis in range(3))
    return Normalization(center, normalize_scale / extent)


def apply_normalization(vertices: Iterable[Sequence[float]], transform: Normalization) -> list[tuple[float, float, float]]:
    return [tuple((float(row[axis]) - transform.center[axis]) * transform.scale for axis in range(3)) for row in vertices]


def invert_normalization(vertices: Iterable[Sequence[float]], transform: Normalization) -> list[tuple[float, float, float]]:
    if transform.scale == 0:
        raise ValueError("Normalization scale cannot be zero")
    return [tuple(float(row[axis]) / transform.scale + transform.center[axis] for axis in range(3)) for row in vertices]
