"""Dependency-free semantic geometry checks shared by the 3D pipeline.

The checks in this module deliberately distinguish numerical dimensional
collapse from ordinary thin geometry.  A genuinely rank-3 mesh can therefore
pass even when one dimension is much smaller than the other two, while a plane
is rejected regardless of its orientation.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Iterable


GEOMETRY_QUALITY_SCHEMA = "comfycolab-3d-geometry-quality-v1"
ANALYSIS_MODE_EXACT = "exact"
ANALYSIS_MODE_RAW = "raw"
SINGULAR_COLLAPSE_RATIO = 1.0e-8
THIN_GEOMETRY_WARNING_RATIO = 1.0e-3
_SINGULAR_COLLAPSE_RATIO = SINGULAR_COLLAPSE_RATIO
_THIN_GEOMETRY_WARNING_RATIO = THIN_GEOMETRY_WARNING_RATIO
_EIGENVALUE_ROUNDOFF_RATIO = 1.0e-14
_FACE_AREA_ROUNDOFF_FACTOR = 128.0 * math.ulp(1.0)
_VECTORIZED_CHUNK_ROWS = 262_144


@dataclass(frozen=True, slots=True)
class GeometryQuality:
    """Versioned, JSON-serializable geometry measurements."""

    stage: str
    analysis_mode: str
    vertex_count: int
    referenced_vertex_count: int
    face_count: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    extents: tuple[float, float, float]
    centroid: tuple[float, float, float]
    singular_values: tuple[float, float, float]
    pca_variances: tuple[float, float, float]
    explained_variance_ratio: tuple[float, float, float]
    smallest_to_largest_singular_ratio: float
    numerical_rank: int
    connected_component_count: int
    connected_components_exact: bool
    nondegenerate_face_count: int
    nondegenerate_face_ratio: float
    surface_area: float
    is_numerically_collapsed: bool
    collapse_reasons: tuple[str, ...]
    is_very_thin: bool
    warnings: tuple[str, ...]
    schema: str = GEOMETRY_QUALITY_SCHEMA

    @property
    def passes_volumetric_validation(self) -> bool:
        return not self.is_numerically_collapsed

    @property
    def is_collapsed(self) -> bool:
        """Compatibility-friendly alias for callers making a gate decision."""

        return self.is_numerically_collapsed

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "stage": self.stage,
            "analysis_mode": self.analysis_mode,
            "vertex_count": self.vertex_count,
            "referenced_vertex_count": self.referenced_vertex_count,
            "face_count": self.face_count,
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
            "extents": list(self.extents),
            "centroid": list(self.centroid),
            "singular_values": list(self.singular_values),
            "pca_variances": list(self.pca_variances),
            "explained_variance_ratio": list(self.explained_variance_ratio),
            "smallest_to_largest_singular_ratio": self.smallest_to_largest_singular_ratio,
            "numerical_rank": self.numerical_rank,
            "connected_component_count": self.connected_component_count,
            "connected_components_exact": self.connected_components_exact,
            "singular_collapse_ratio": _SINGULAR_COLLAPSE_RATIO,
            "thin_geometry_warning_ratio": _THIN_GEOMETRY_WARNING_RATIO,
            "nondegenerate_face_count": self.nondegenerate_face_count,
            "nondegenerate_face_ratio": self.nondegenerate_face_ratio,
            "surface_area": self.surface_area,
            "is_numerically_collapsed": self.is_numerically_collapsed,
            "passes_volumetric_validation": self.passes_volumetric_validation,
            "collapse_reasons": list(self.collapse_reasons),
            "is_very_thin": self.is_very_thin,
            "warnings": list(self.warnings),
        }


def _rows(values: Iterable[Any], *, width: int, label: str, cast) -> list[tuple[Any, ...]]:
    rows = []
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise ValueError(f"{label} must be an iterable of {width}-component rows") from exc
    for row_index, row in enumerate(iterator):
        try:
            converted = tuple(cast(value) for value in row)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label} row {row_index} contains an invalid value") from exc
        if len(converted) != width:
            raise ValueError(f"{label} row {row_index} must contain exactly {width} values")
        rows.append(converted)
    return rows


def _symmetric_eigenvalues(matrix: list[list[float]]) -> tuple[float, float, float]:
    """Return descending eigenvalues for a real symmetric 3x3 matrix."""

    values = [row[:] for row in matrix]
    for _ in range(64):
        p, q = max(((0, 1), (0, 2), (1, 2)), key=lambda pair: abs(values[pair[0]][pair[1]]))
        off_diagonal = values[p][q]
        diagonal_scale = max(abs(values[0][0]), abs(values[1][1]), abs(values[2][2]))
        if abs(off_diagonal) <= max(diagonal_scale * 1.0e-15, 1.0e-300):
            break
        tau = (values[q][q] - values[p][p]) / (2.0 * off_diagonal)
        sign = 1.0 if tau >= 0.0 else -1.0
        tangent = sign / (abs(tau) + math.sqrt(1.0 + tau * tau))
        cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
        sine = tangent * cosine
        app = values[p][p]
        aqq = values[q][q]
        values[p][p] = app - tangent * off_diagonal
        values[q][q] = aqq + tangent * off_diagonal
        values[p][q] = values[q][p] = 0.0
        for other in range(3):
            if other in (p, q):
                continue
            aop = values[other][p]
            aoq = values[other][q]
            values[other][p] = values[p][other] = cosine * aop - sine * aoq
            values[other][q] = values[q][other] = sine * aop + cosine * aoq
    eigenvalues = sorted((values[0][0], values[1][1], values[2][2]), reverse=True)
    largest = max(eigenvalues[0], 0.0)
    roundoff_floor = largest * _EIGENVALUE_ROUNDOFF_RATIO
    cleaned = [0.0 if value <= roundoff_floor else value for value in eigenvalues]
    return cleaned[0], cleaned[1], cleaned[2]


def _build_quality(
    *,
    stage: str,
    analysis_mode: str,
    vertex_count: int,
    referenced_vertex_count: int,
    face_count: int,
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    centroid: tuple[float, float, float],
    scatter: list[list[float]],
    connected_component_count: int,
    connected_components_exact: bool,
    nondegenerate_face_count: int,
    surface_area: float,
) -> GeometryQuality:
    extents = tuple(bounds_max[axis] - bounds_min[axis] for axis in range(3))
    eigenvalues = _symmetric_eigenvalues(scatter)
    singular_values = tuple(math.sqrt(value) for value in eigenvalues)
    largest_singular = singular_values[0]
    singular_ratio = (
        singular_values[2] / largest_singular if largest_singular > 0.0 else 0.0
    )
    numerical_rank = (
        sum(
            value > largest_singular * _SINGULAR_COLLAPSE_RATIO
            for value in singular_values
        )
        if largest_singular > 0.0
        else 0
    )
    variance_divisor = max(referenced_vertex_count - 1, 1)
    pca_variances = tuple(value / variance_divisor for value in eigenvalues)
    total_variance = math.fsum(pca_variances)
    explained_variance = tuple(
        value / total_variance if total_variance > 0.0 else 0.0
        for value in pca_variances
    )
    nondegenerate_ratio = nondegenerate_face_count / face_count

    reasons = []
    if numerical_rank < 3:
        reasons.append("vertex_rank_below_3")
    if nondegenerate_face_count == 0:
        reasons.append("no_nondegenerate_faces")
    warnings = []
    if not reasons and singular_ratio < _THIN_GEOMETRY_WARNING_RATIO:
        warnings.append("very_thin_rank_3_geometry")

    return GeometryQuality(
        stage=stage,
        analysis_mode=analysis_mode,
        vertex_count=vertex_count,
        referenced_vertex_count=referenced_vertex_count,
        face_count=face_count,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        extents=extents,
        centroid=centroid,
        singular_values=singular_values,
        pca_variances=pca_variances,
        explained_variance_ratio=explained_variance,
        smallest_to_largest_singular_ratio=singular_ratio,
        numerical_rank=numerical_rank,
        connected_component_count=connected_component_count,
        connected_components_exact=connected_components_exact,
        nondegenerate_face_count=nondegenerate_face_count,
        nondegenerate_face_ratio=nondegenerate_ratio,
        surface_area=surface_area,
        is_numerically_collapsed=bool(reasons),
        collapse_reasons=tuple(reasons),
        is_very_thin=bool(warnings),
        warnings=tuple(warnings),
    )


def _analyze_geometry_python(
    vertices,
    faces,
    *,
    stage: str,
    analysis_mode: str,
    exact_components: bool,
) -> GeometryQuality:
    """Dependency-free path for ordinary processed/final meshes and small inputs."""

    vertex_rows = _rows(vertices, width=3, label="vertices", cast=float)
    face_rows = _rows(faces, width=3, label="faces", cast=int)
    if len(vertex_rows) < 3:
        raise ValueError("Geometry quality analysis requires at least three vertices")
    if not face_rows:
        raise ValueError("Geometry quality analysis requires at least one triangle")
    if any(index < 0 or index >= len(vertex_rows) for row in face_rows for index in row):
        raise ValueError("Geometry faces contain an out-of-range vertex index")

    referenced_vertices = {index for row in face_rows for index in row}
    referenced_rows = [vertex_rows[index] for index in sorted(referenced_vertices)]
    if not all(math.isfinite(value) for row in referenced_rows for value in row):
        raise ValueError("Geometry vertices must be finite")

    bounds_min = tuple(min(row[axis] for row in referenced_rows) for axis in range(3))
    bounds_max = tuple(max(row[axis] for row in referenced_rows) for axis in range(3))
    centroid = tuple(
        math.fsum(row[axis] for row in referenced_rows) / len(referenced_rows)
        for axis in range(3)
    )
    scatter = [[0.0, 0.0, 0.0] for _ in range(3)]
    for row in referenced_rows:
        centered = tuple(row[axis] - centroid[axis] for axis in range(3))
        for first in range(3):
            for second in range(first, 3):
                scatter[first][second] += centered[first] * centered[second]
    for first in range(3):
        for second in range(first):
            scatter[first][second] = scatter[second][first]

    extents = tuple(bounds_max[axis] - bounds_min[axis] for axis in range(3))
    maximum_extent = max(extents)
    twice_area_tolerance = maximum_extent * maximum_extent * _FACE_AREA_ROUNDOFF_FACTOR
    nondegenerate_faces = 0
    surface_area = 0.0
    for first_index, second_index, third_index in face_rows:
        first = vertex_rows[first_index]
        second = vertex_rows[second_index]
        third = vertex_rows[third_index]
        ab = tuple(second[axis] - first[axis] for axis in range(3))
        ac = tuple(third[axis] - first[axis] for axis in range(3))
        cross = (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        twice_area = math.sqrt(math.fsum(value * value for value in cross))
        surface_area += 0.5 * twice_area
        if twice_area > twice_area_tolerance:
            nondegenerate_faces += 1

    connected_component_count = -1
    if exact_components:
        parent = list(range(len(vertex_rows)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(first: int, second: int) -> None:
            first_root, second_root = find(first), find(second)
            if first_root != second_root:
                parent[second_root] = first_root

        for first_index, second_index, third_index in face_rows:
            union(first_index, second_index)
            union(second_index, third_index)
        connected_component_count = len(
            {find(index) for index in referenced_vertices}
        )

    return _build_quality(
        stage=stage,
        analysis_mode=analysis_mode,
        vertex_count=len(vertex_rows),
        referenced_vertex_count=len(referenced_vertices),
        face_count=len(face_rows),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        centroid=centroid,
        scatter=scatter,
        connected_component_count=connected_component_count,
        connected_components_exact=exact_components,
        nondegenerate_face_count=nondegenerate_faces,
        surface_area=surface_area,
    )


def _has_array_shape(value: Any) -> bool:
    return getattr(value, "shape", None) is not None and hasattr(value, "__getitem__")


def _as_numpy_array(value, numpy, *, dtype=None):
    """Convert one already-bounded chunk without using Python row materialization."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return numpy.asarray(value, dtype=dtype)


def _take_numpy_rows(values, indices, numpy, *, dtype=None):
    """Gather a bounded NumPy or Torch-like chunk and return it on the CPU."""

    if hasattr(values, "detach") and hasattr(values, "new_tensor"):
        backend_indices = values.new_tensor(indices).long()
        selected = values[backend_indices]
    else:
        selected = values[indices]
    return _as_numpy_array(selected, numpy, dtype=dtype)


def _analyze_geometry_vectorized_raw(vertices, faces, *, stage: str, numpy) -> GeometryQuality:
    """Chunked NumPy path that avoids Python rows and exact topology on raw meshes."""

    try:
        vertex_shape = tuple(vertices.shape)
        face_shape = tuple(faces.shape)
    except (AttributeError, TypeError) as exc:
        raise ValueError("Raw geometry arrays must expose two-dimensional shapes") from exc
    if len(vertex_shape) != 2 or vertex_shape[1] != 3:
        raise ValueError("vertices must be an array with shape (N, 3)")
    if len(face_shape) != 2 or face_shape[1] != 3:
        raise ValueError("faces must be an array with shape (N, 3)")
    vertex_count, face_count = int(vertex_shape[0]), int(face_shape[0])
    if vertex_count < 3:
        raise ValueError("Geometry quality analysis requires at least three vertices")
    if face_count < 1:
        raise ValueError("Geometry quality analysis requires at least one triangle")

    referenced_mask = numpy.zeros(vertex_count, dtype=numpy.bool_)
    for start in range(0, face_count, _VECTORIZED_CHUNK_ROWS):
        stop = min(start + _VECTORIZED_CHUNK_ROWS, face_count)
        face_chunk = _as_numpy_array(faces[start:stop], numpy)
        if face_chunk.shape != (stop - start, 3):
            raise ValueError("faces must be an array with shape (N, 3)")
        if not numpy.issubdtype(face_chunk.dtype, numpy.integer):
            raise ValueError("Geometry face indices must be integers")
        if bool(numpy.any(face_chunk < 0)) or bool(numpy.any(face_chunk >= vertex_count)):
            raise ValueError("Geometry faces contain an out-of-range vertex index")
        referenced_mask[face_chunk.reshape(-1)] = True

    referenced_indices = numpy.flatnonzero(referenced_mask)
    referenced_vertex_count = int(referenced_indices.size)
    bounds_min_array = numpy.full(3, numpy.inf, dtype=numpy.float64)
    bounds_max_array = numpy.full(3, -numpy.inf, dtype=numpy.float64)
    coordinate_sum = numpy.zeros(3, dtype=numpy.float64)
    for start in range(0, referenced_vertex_count, _VECTORIZED_CHUNK_ROWS):
        index_chunk = referenced_indices[start : start + _VECTORIZED_CHUNK_ROWS]
        vertex_chunk = _take_numpy_rows(
            vertices,
            index_chunk,
            numpy,
            dtype=numpy.float64,
        )
        if vertex_chunk.shape != (len(index_chunk), 3):
            raise ValueError("vertices must be an array with shape (N, 3)")
        if not bool(numpy.all(numpy.isfinite(vertex_chunk))):
            raise ValueError("Geometry vertices must be finite")
        bounds_min_array = numpy.minimum(bounds_min_array, numpy.min(vertex_chunk, axis=0))
        bounds_max_array = numpy.maximum(bounds_max_array, numpy.max(vertex_chunk, axis=0))
        coordinate_sum += numpy.sum(vertex_chunk, axis=0, dtype=numpy.float64)
    centroid_array = coordinate_sum / referenced_vertex_count

    scatter_array = numpy.zeros((3, 3), dtype=numpy.float64)
    for start in range(0, referenced_vertex_count, _VECTORIZED_CHUNK_ROWS):
        index_chunk = referenced_indices[start : start + _VECTORIZED_CHUNK_ROWS]
        vertex_chunk = _take_numpy_rows(
            vertices,
            index_chunk,
            numpy,
            dtype=numpy.float64,
        )
        centered = vertex_chunk - centroid_array
        scatter_array += centered.T @ centered

    bounds_min = tuple(float(bounds_min_array[axis]) for axis in range(3))
    bounds_max = tuple(float(bounds_max_array[axis]) for axis in range(3))
    centroid = tuple(float(centroid_array[axis]) for axis in range(3))
    extents = tuple(bounds_max[axis] - bounds_min[axis] for axis in range(3))
    maximum_extent = max(extents)
    twice_area_tolerance = maximum_extent * maximum_extent * _FACE_AREA_ROUNDOFF_FACTOR
    nondegenerate_faces = 0
    surface_area = 0.0
    for start in range(0, face_count, _VECTORIZED_CHUNK_ROWS):
        stop = min(start + _VECTORIZED_CHUNK_ROWS, face_count)
        face_chunk = _as_numpy_array(faces[start:stop], numpy)
        triangles = _take_numpy_rows(
            vertices,
            face_chunk,
            numpy,
            dtype=numpy.float64,
        )
        first_edges = triangles[:, 1, :] - triangles[:, 0, :]
        second_edges = triangles[:, 2, :] - triangles[:, 0, :]
        crosses = numpy.cross(first_edges, second_edges)
        twice_areas = numpy.sqrt(numpy.einsum("ij,ij->i", crosses, crosses))
        surface_area += 0.5 * float(numpy.sum(twice_areas, dtype=numpy.float64))
        nondegenerate_faces += int(numpy.count_nonzero(twice_areas > twice_area_tolerance))

    scatter = [
        [float(scatter_array[first, second]) for second in range(3)]
        for first in range(3)
    ]
    return _build_quality(
        stage=stage,
        analysis_mode=ANALYSIS_MODE_RAW,
        vertex_count=vertex_count,
        referenced_vertex_count=referenced_vertex_count,
        face_count=face_count,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        centroid=centroid,
        scatter=scatter,
        connected_component_count=-1,
        connected_components_exact=False,
        nondegenerate_face_count=nondegenerate_faces,
        surface_area=surface_area,
    )


def analyze_geometry(
    vertices,
    faces,
    *,
    stage: str,
    analysis_mode: str = ANALYSIS_MODE_EXACT,
) -> GeometryQuality:
    """Measure dimensional rank and triangle quality for vertex/face data.

    ``raw`` mode uses bounded-memory vectorized chunks when array-backed data is
    available and deliberately skips exact connected components.  Both modes
    calculate rank and bounds from face-referenced vertices only.
    """

    stage = str(stage).strip()
    if not stage:
        raise ValueError("Geometry quality stage must be a non-empty string")
    analysis_mode = str(analysis_mode).strip().lower()
    if analysis_mode not in {ANALYSIS_MODE_EXACT, ANALYSIS_MODE_RAW}:
        raise ValueError("Geometry analysis mode must be 'exact' or 'raw'")
    if (
        analysis_mode == ANALYSIS_MODE_RAW
        and _has_array_shape(vertices)
        and _has_array_shape(faces)
    ):
        try:
            numpy = importlib.import_module("numpy")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Raw array-backed geometry validation requires NumPy"
            ) from exc
        else:
            return _analyze_geometry_vectorized_raw(vertices, faces, stage=stage, numpy=numpy)
    return _analyze_geometry_python(
        vertices,
        faces,
        stage=stage,
        analysis_mode=analysis_mode,
        exact_components=analysis_mode == ANALYSIS_MODE_EXACT,
    )


def validate_volumetric_mesh(
    mesh,
    *,
    stage: str,
    analysis_mode: str = ANALYSIS_MODE_EXACT,
) -> GeometryQuality:
    """Analyze a trimesh-like object and reject numerical dimensional collapse."""

    try:
        vertices = mesh.vertices
        faces = mesh.faces
    except AttributeError as exc:
        raise ValueError("Volumetric validation requires a mesh with vertices and faces") from exc
    metrics = analyze_geometry(
        vertices,
        faces,
        stage=stage,
        analysis_mode=analysis_mode,
    )
    if metrics.is_numerically_collapsed:
        reasons = ", ".join(metrics.collapse_reasons)
        raise ValueError(
            f"{stage} geometry is numerically collapsed ({reasons}); "
            f"PCA rank={metrics.numerical_rank}, "
            f"smallest/largest singular ratio={metrics.smallest_to_largest_singular_ratio:.3g}"
        )
    return metrics


def validate_volumetric_glb(
    path,
    *,
    stage: str,
    require_material: bool = False,
    require_texture: bool = False,
    require_uv: bool = False,
) -> GeometryQuality:
    """Expose the file-backed gate alongside the shared in-memory contract."""

    file3d = importlib.import_module(f"{__package__}.file3d")
    return file3d.validate_volumetric_glb(
        path,
        stage=stage,
        require_material=require_material,
        require_texture=require_texture,
        require_uv=require_uv,
    )


__all__ = [
    "ANALYSIS_MODE_EXACT",
    "ANALYSIS_MODE_RAW",
    "GEOMETRY_QUALITY_SCHEMA",
    "SINGULAR_COLLAPSE_RATIO",
    "THIN_GEOMETRY_WARNING_RATIO",
    "GeometryQuality",
    "analyze_geometry",
    "validate_volumetric_glb",
    "validate_volumetric_mesh",
]
