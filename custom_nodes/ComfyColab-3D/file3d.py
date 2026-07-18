from __future__ import annotations

import importlib
import json
import math
import os
import shutil
import struct
from pathlib import Path
from typing import Any, Iterator


_COMPONENT_FORMATS = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}
_TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
_TEXTURE_SOURCE_EXTENSIONS = ("EXT_texture_webp", "KHR_texture_basisu")


def _texture_image_index(texture: dict[str, Any]) -> int | None:
    source = texture.get("source")
    if isinstance(source, int):
        return source
    extensions = texture.get("extensions")
    if not isinstance(extensions, dict):
        return None
    for name in _TEXTURE_SOURCE_EXTENSIONS:
        extension = extensions.get(name)
        if isinstance(extension, dict) and isinstance(extension.get("source"), int):
            return extension["source"]
    return None


def _texture_image_index(texture: dict[str, Any]) -> int | None:
    """Resolve core or extension-backed glTF texture image sources."""

    extensions = texture.get("extensions")
    if isinstance(extensions, dict):
        for name in ("EXT_texture_webp", "KHR_texture_basisu"):
            extension = extensions.get(name)
            if isinstance(extension, dict) and isinstance(extension.get("source"), int):
                return int(extension["source"])
    source = texture.get("source")
    return int(source) if isinstance(source, int) else None


def _iter_accessor(
    document: dict[str, Any], binary: bytes, index: int
) -> tuple[int, Iterator[tuple[Any, ...]]]:
    accessors = document.get("accessors") or []
    views = document.get("bufferViews") or []
    if index < 0 or index >= len(accessors):
        raise ValueError(f"GLB accessor index {index} is invalid")
    accessor = accessors[index]
    view_index = accessor.get("bufferView")
    if not isinstance(view_index, int) or view_index < 0 or view_index >= len(views):
        raise ValueError(f"GLB accessor {index} has no valid buffer view")
    view = views[view_index]
    if int(view.get("buffer", 0)) != 0:
        raise ValueError("GLB uses an unsupported external buffer")
    component_type = int(accessor.get("componentType", 0))
    component = _COMPONENT_FORMATS.get(component_type)
    width = _TYPE_COMPONENTS.get(str(accessor.get("type", "")))
    count = int(accessor.get("count", 0))
    if component is None or width is None or count <= 0:
        raise ValueError(f"GLB accessor {index} has an unsupported representation")
    format_code, component_bytes = component
    packed_bytes = component_bytes * width
    stride = int(view.get("byteStride", packed_bytes))
    if stride < packed_bytes:
        raise ValueError(f"GLB accessor {index} has an invalid byte stride")
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    view_end = int(view.get("byteOffset", 0)) + int(view.get("byteLength", 0))
    final_end = offset + (count - 1) * stride + packed_bytes
    if offset < 0 or final_end > min(view_end, len(binary)):
        raise ValueError(f"GLB accessor {index} exceeds its embedded buffer")
    unpack = struct.Struct("<" + format_code * width).unpack_from
    return count, (unpack(binary, offset + item * stride) for item in range(count))


def _validate_glb_impl(
    path: str | Path,
    *,
    require_material: bool = False,
    require_texture: bool = False,
    require_uv: bool = False,
) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12:
            raise ValueError("GLB header is truncated")
        magic, version, declared_length = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2:
            raise ValueError("Expected a glTF 2.0 binary file")
        if declared_length != path.stat().st_size:
            raise ValueError("GLB declared length does not match file size")
        chunk_header = stream.read(8)
        if len(chunk_header) != 8:
            raise ValueError("GLB JSON chunk is missing")
        chunk_length, chunk_type = struct.unpack("<I4s", chunk_header)
        if chunk_type != b"JSON":
            raise ValueError("GLB first chunk must contain JSON")
        try:
            document = json.loads(stream.read(chunk_length).decode("utf-8").rstrip(" \t\r\n\x00"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("GLB JSON chunk is invalid") from exc
        binary = b""
        while stream.tell() < declared_length:
            next_header = stream.read(8)
            if len(next_header) != 8:
                raise ValueError("GLB chunk header is truncated")
            next_length, next_type = struct.unpack("<I4s", next_header)
            payload = stream.read(next_length)
            if len(payload) != next_length:
                raise ValueError("GLB chunk is truncated")
            if next_type == b"BIN\x00":
                if binary:
                    raise ValueError("GLB contains multiple binary chunks")
                binary = payload
    meshes = document.get("meshes") or []
    accessors = document.get("accessors") or []
    if not meshes:
        raise ValueError("GLB does not contain a mesh")
    if (require_material or require_texture or require_uv) and not binary:
        raise ValueError("GLB validation requires embedded mesh buffers")
    materials = document.get("materials") or []
    textured_materials: set[int] = set()
    for material_index, material in enumerate(materials):
        texture = (material.get("pbrMetallicRoughness") or {}).get("baseColorTexture")
        if isinstance(texture, dict) and isinstance(texture.get("index"), int):
            textured_materials.add(material_index)
    primitive_count = 0
    for mesh in meshes:
        for primitive in mesh.get("primitives") or []:
            primitive_count += 1
            attributes = primitive.get("attributes") or {}
            position = attributes.get("POSITION")
            if not isinstance(position, int) or position < 0 or position >= len(accessors):
                raise ValueError("GLB mesh primitive has no valid POSITION accessor")
            position_accessor = accessors[position]
            if (
                position_accessor.get("type") != "VEC3"
                or int(position_accessor.get("componentType", 0)) != 5126
            ):
                raise ValueError("GLB POSITION accessor must be FLOAT VEC3")
            position_count = int(position_accessor.get("count", 0))
            if position_count <= 0:
                raise ValueError("GLB mesh primitive contains no vertices")
            indices_index = primitive.get("indices")
            index_accessor = accessors[indices_index] if isinstance(indices_index, int) and 0 <= indices_index < len(accessors) else {}
            if (
                not isinstance(indices_index, int)
                or indices_index < 0
                or indices_index >= len(accessors)
                or index_accessor.get("type") != "SCALAR"
                or int(index_accessor.get("componentType", 0)) not in {5121, 5123, 5125}
            ):
                raise ValueError("GLB mesh primitive has no valid index accessor")
            if int(primitive.get("mode", 4)) != 4:
                raise ValueError("GLB mesh primitive must use TRIANGLES mode")
            declared_index_count = int(index_accessor.get("count", 0))
            if declared_index_count < 3 or declared_index_count % 3:
                raise ValueError("GLB triangle index count must be a positive multiple of three")
            if binary:
                position_count, positions = _iter_accessor(document, binary, position)
                if not all(math.isfinite(float(value)) for row in positions for value in row):
                    raise ValueError("GLB mesh primitive contains non-finite vertices")
                index_count, index_rows = _iter_accessor(document, binary, indices_index)
                minimum_index = None
                maximum_index = None
                for row in index_rows:
                    value = int(row[0])
                    minimum_index = value if minimum_index is None else min(minimum_index, value)
                    maximum_index = value if maximum_index is None else max(maximum_index, value)
                if (
                    index_count <= 0
                    or minimum_index is None
                    or minimum_index < 0
                    or maximum_index is None
                    or maximum_index >= position_count
                ):
                    raise ValueError("GLB mesh primitive contains invalid indices")
            material_index = primitive.get("material")
            if require_material and (
                not isinstance(material_index, int)
                or material_index < 0
                or material_index >= len(materials)
            ):
                raise ValueError("GLB mesh primitive has no material")
            if require_uv:
                uv_index = attributes.get("TEXCOORD_0")
                if not isinstance(uv_index, int) or uv_index < 0 or uv_index >= len(accessors):
                    raise ValueError("GLB mesh primitive has no valid UV accessor")
                uv_accessor = accessors[uv_index]
                uv_component = int(uv_accessor.get("componentType", 0))
                if uv_accessor.get("type") != "VEC2" or uv_component not in {5121, 5123, 5126}:
                    raise ValueError("GLB UV accessor must use a supported VEC2 representation")
                if uv_component in {5121, 5123} and uv_accessor.get("normalized") is not True:
                    raise ValueError("Integer GLB UV accessors must be normalized")
                uv_count, uv_rows = _iter_accessor(document, binary, uv_index)
                if uv_count != position_count:
                    raise ValueError("GLB UV count does not match its vertex count")
                if not all(math.isfinite(float(value)) for row in uv_rows for value in row):
                    raise ValueError("GLB mesh primitive contains non-finite UV coordinates")
            if require_texture and material_index not in textured_materials:
                raise ValueError("GLB mesh primitive has no base-color texture")
    if primitive_count == 0:
        raise ValueError("GLB meshes contain no primitives")
    if require_texture:
        textures = document.get("textures") or []
        images = document.get("images") or []
        for material_index in textured_materials:
            texture_index = materials[material_index]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
            if texture_index < 0 or texture_index >= len(textures):
                raise ValueError("GLB material references an invalid texture")
            image_index = _texture_image_index(textures[texture_index])
            if not isinstance(image_index, int) or image_index < 0 or image_index >= len(images):
                raise ValueError("GLB texture references an invalid image")
            image = images[image_index]
            uri = image.get("uri")
            image_view = image.get("bufferView")
            if image_view is not None:
                views = document.get("bufferViews") or []
                if not isinstance(image_view, int) or image_view < 0 or image_view >= len(views):
                    raise ValueError("GLB texture image references an invalid buffer view")
                view = views[image_view]
                offset = int(view.get("byteOffset", 0))
                length = int(view.get("byteLength", 0))
                if (
                    int(view.get("buffer", 0)) != 0
                    or offset < 0
                    or length <= 0
                    or offset + length > len(binary)
                ):
                    raise ValueError("GLB embedded texture exceeds its binary buffer")
            elif not (
                isinstance(uri, str) and uri.startswith("data:")
            ):
                raise ValueError("GLB texture image is not embedded")
    return document


def validate_glb(
    path: str | Path,
    *,
    require_material: bool = False,
    require_texture: bool = False,
    require_uv: bool = False,
) -> dict[str, Any]:
    """Validate a GLB and normalize malformed JSON schema errors to ValueError."""

    try:
        return _validate_glb_impl(
            path,
            require_material=require_material,
            require_texture=require_texture,
            require_uv=require_uv,
        )
    except ValueError:
        raise
    except (AttributeError, IndexError, KeyError, OverflowError, struct.error, TypeError) as exc:
        raise ValueError("GLB JSON structure is invalid") from exc


def _read_embedded_binary_chunk(path: Path) -> bytes:
    with path.open("rb") as stream:
        stream.seek(12)
        while stream.tell() < path.stat().st_size:
            chunk_header = stream.read(8)
            if len(chunk_header) != 8:
                raise ValueError("GLB chunk header is truncated")
            chunk_length, chunk_type = struct.unpack("<I4s", chunk_header)
            payload = stream.read(chunk_length)
            if len(payload) != chunk_length:
                raise ValueError("GLB chunk is truncated")
            if chunk_type == b"BIN\x00":
                return payload
    return b""


_IDENTITY_MATRIX = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _matrix_multiply(first, second):
    return tuple(
        tuple(
            math.fsum(first[row][inner] * second[inner][column] for inner in range(4))
            for column in range(4)
        )
        for row in range(4)
    )


def _node_matrix(node: dict[str, Any]):
    if "matrix" in node:
        values = tuple(float(value) for value in node["matrix"])
        if len(values) != 16 or not all(math.isfinite(value) for value in values):
            raise ValueError("GLB node matrix must contain 16 finite values")
        # glTF serializes matrices in column-major order.
        return tuple(
            tuple(values[column * 4 + row] for column in range(4))
            for row in range(4)
        )

    translation = tuple(float(value) for value in node.get("translation", (0.0, 0.0, 0.0)))
    rotation = tuple(float(value) for value in node.get("rotation", (0.0, 0.0, 0.0, 1.0)))
    scale = tuple(float(value) for value in node.get("scale", (1.0, 1.0, 1.0)))
    if len(translation) != 3 or len(rotation) != 4 or len(scale) != 3:
        raise ValueError("GLB node TRS transform has an invalid width")
    if not all(math.isfinite(value) for value in (*translation, *rotation, *scale)):
        raise ValueError("GLB node TRS transform must be finite")
    x, y, z, w = rotation
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("GLB node quaternion must be non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    sx, sy, sz = scale
    tx, ty, tz = translation
    return (
        ((1.0 - 2.0 * (y * y + z * z)) * sx, 2.0 * (x * y - z * w) * sy, 2.0 * (x * z + y * w) * sz, tx),
        (2.0 * (x * y + z * w) * sx, (1.0 - 2.0 * (x * x + z * z)) * sy, 2.0 * (y * z - x * w) * sz, ty),
        (2.0 * (x * z - y * w) * sx, 2.0 * (y * z + x * w) * sy, (1.0 - 2.0 * (x * x + y * y)) * sz, tz),
        (0.0, 0.0, 0.0, 1.0),
    )


def _scene_mesh_instances(document: dict[str, Any]) -> list[tuple[int, tuple]]:
    meshes = document.get("meshes") or []
    nodes = document.get("nodes") or []
    if not nodes:
        return [(index, _IDENTITY_MATRIX) for index in range(len(meshes))]

    scenes = document.get("scenes") or []
    if scenes:
        scene_index = document.get("scene", 0)
        if not isinstance(scene_index, int) or scene_index < 0 or scene_index >= len(scenes):
            raise ValueError("GLB default scene index is invalid")
        scene = scenes[scene_index]
        if not isinstance(scene, dict):
            raise ValueError("GLB scene must be an object")
        roots = scene.get("nodes") or []
    else:
        children = {
            child
            for node in nodes
            if isinstance(node, dict)
            for child in (node.get("children") or [])
            if isinstance(child, int)
        }
        roots = [index for index in range(len(nodes)) if index not in children]

    instances: list[tuple[int, tuple]] = []

    def visit(node_index: int, parent_matrix, ancestry: frozenset[int]) -> None:
        if not isinstance(node_index, int) or node_index < 0 or node_index >= len(nodes):
            raise ValueError("GLB scene references an invalid node")
        if node_index in ancestry:
            raise ValueError("GLB node graph contains a cycle")
        node = nodes[node_index]
        if not isinstance(node, dict):
            raise ValueError("GLB node must be an object")
        world_matrix = _matrix_multiply(parent_matrix, _node_matrix(node))
        mesh_index = node.get("mesh")
        if mesh_index is not None:
            if not isinstance(mesh_index, int) or mesh_index < 0 or mesh_index >= len(meshes):
                raise ValueError("GLB node references an invalid mesh")
            instances.append((mesh_index, world_matrix))
        next_ancestry = ancestry | {node_index}
        for child in node.get("children") or []:
            visit(child, world_matrix, next_ancestry)

    for root in roots:
        visit(root, _IDENTITY_MATRIX, frozenset())
    if not instances:
        raise ValueError("GLB default scene contains no mesh geometry")
    return instances


def _transform_position(position, matrix) -> tuple[float, float, float]:
    x, y, z = position
    transformed = tuple(
        math.fsum((matrix[row][0] * x, matrix[row][1] * y, matrix[row][2] * z, matrix[row][3]))
        for row in range(4)
    )
    if not all(math.isfinite(value) for value in transformed):
        raise ValueError("GLB node transform produced non-finite geometry")
    if abs(transformed[3]) <= 1.0e-300:
        raise ValueError("GLB node transform produced an invalid homogeneous coordinate")
    return (
        transformed[0] / transformed[3],
        transformed[1] / transformed[3],
        transformed[2] / transformed[3],
    )


def _glb_vertex_face_data(
    path: Path, document: dict[str, Any]
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Return the baked geometry from the default GLB scene."""

    binary = _read_embedded_binary_chunk(path)
    if not binary:
        raise ValueError("Volumetric GLB validation requires embedded mesh buffers")
    mesh_cache: dict[int, tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]] = {}

    def mesh_data(mesh_index: int):
        cached = mesh_cache.get(mesh_index)
        if cached is not None:
            return cached
        mesh_vertices: list[tuple[float, float, float]] = []
        mesh_faces: list[tuple[int, int, int]] = []
        mesh = document["meshes"][mesh_index]
        for primitive in mesh.get("primitives") or []:
            attributes = primitive.get("attributes") or {}
            position_index = int(attributes["POSITION"])
            indices_index = int(primitive["indices"])
            _, position_rows = _iter_accessor(document, binary, position_index)
            primitive_vertices = [
                (float(row[0]), float(row[1]), float(row[2])) for row in position_rows
            ]
            _, index_rows = _iter_accessor(document, binary, indices_index)
            primitive_indices = [int(row[0]) for row in index_rows]
            vertex_offset = len(mesh_vertices)
            mesh_vertices.extend(primitive_vertices)
            mesh_faces.extend(
                (
                    vertex_offset + primitive_indices[offset],
                    vertex_offset + primitive_indices[offset + 1],
                    vertex_offset + primitive_indices[offset + 2],
                )
                for offset in range(0, len(primitive_indices), 3)
            )
        cached = (mesh_vertices, mesh_faces)
        mesh_cache[mesh_index] = cached
        return cached

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    try:
        instances = _scene_mesh_instances(document)
        for mesh_index, world_matrix in instances:
            instance_vertices, instance_faces = mesh_data(mesh_index)
            vertex_offset = len(vertices)
            vertices.extend(
                _transform_position(vertex, world_matrix) for vertex in instance_vertices
            )
            faces.extend(
                tuple(vertex_offset + index for index in face) for face in instance_faces
            )
    except ValueError:
        raise
    except (AttributeError, IndexError, KeyError, OverflowError, TypeError) as exc:
        raise ValueError("GLB scene geometry structure is invalid") from exc
    return vertices, faces


def validate_volumetric_mesh(mesh: Any, *, stage: str):
    """Reject dimensional collapse in a trimesh-like object."""

    quality = importlib.import_module(f"{__package__}.geometry_quality")
    return quality.validate_volumetric_mesh(mesh, stage=stage)


def validate_volumetric_glb(
    path: str | Path,
    *,
    stage: str,
    require_material: bool = False,
    require_texture: bool = False,
    require_uv: bool = False,
):
    """Run structural GLB validation, then reject numerical planar collapse."""

    path = Path(path)
    document = validate_glb(
        path,
        require_material=require_material,
        require_texture=require_texture,
        require_uv=require_uv,
    )
    vertices, faces = _glb_vertex_face_data(path, document)
    quality = importlib.import_module(f"{__package__}.geometry_quality")
    metrics = quality.analyze_geometry(vertices, faces, stage=stage)
    if metrics.is_numerically_collapsed:
        reasons = ", ".join(metrics.collapse_reasons)
        raise ValueError(
            f"{stage} GLB geometry is numerically collapsed ({reasons}); "
            f"PCA rank={metrics.numerical_rank}, "
            f"smallest/largest singular ratio={metrics.smallest_to_largest_singular_ratio:.3g}"
        )
    return metrics


def materialize_file3d(path: str | Path):
    path = Path(path)
    validate_glb(path)
    latest = importlib.import_module("comfy_api.latest")
    return latest.Types.File3D(str(path), file_format="glb")


def publish_glb(path: str | Path, key: str) -> Path:
    source = Path(path)
    validate_glb(source)
    if not key or Path(key).name != key or key in {".", ".."} or "\\" in key:
        raise ValueError("Published GLB keys must be non-empty path-safe filenames")
    output_root = Path(os.environ.get("COMFYCOLAB_3D_OUTPUT", "/content/ComfyUI/output/3d"))
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / f"{key}.glb"
    partial = output_root / f".{key}.{os.getpid()}.partial.glb"
    try:
        try:
            os.link(source, partial)
        except OSError:
            shutil.copyfile(source, partial)
        validate_glb(partial)
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def copy_file3d_to(model_3d: Any, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    try:
        if hasattr(model_3d, "save_to"):
            model_3d.save_to(str(partial))
        elif hasattr(model_3d, "get_data"):
            data = model_3d.get_data()
            if hasattr(data, "read"):
                if hasattr(data, "seek"):
                    data.seek(0)
                data = data.read()
            partial.write_bytes(data if isinstance(data, bytes) else bytes(data))
        else:
            shutil.copyfile(str(model_3d), partial)
        validate_glb(partial)
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def export_trimesh_atomic(
    mesh: Any,
    destination: str | Path,
    *,
    require_material: bool = False,
    require_texture: bool = False,
    require_uv: bool = False,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.stem}.{os.getpid()}.partial.glb")
    try:
        mesh.export(str(partial), file_type="glb")
        validate_glb(
            partial,
            require_material=require_material,
            require_texture=require_texture,
            require_uv=require_uv,
        )
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def bake_scene_mesh(loaded, trimesh_module):
    if not isinstance(loaded, trimesh_module.Scene):
        return loaded.copy()
    geometries = []
    for node_name in loaded.graph.nodes_geometry:
        transform, geometry_name = loaded.graph.get(node_name)
        geometry = loaded.geometry[geometry_name].copy()
        geometry.apply_transform(transform)
        geometries.append(geometry)
    if not geometries:
        raise ValueError("GLB scene contains no mesh geometry")
    return trimesh_module.util.concatenate(geometries)


def load_glb_trimesh(path: str | Path):
    trimesh = importlib.import_module("trimesh")
    numpy = importlib.import_module("numpy")
    loaded = trimesh.load(str(path), force="scene")
    mesh = bake_scene_mesh(loaded, trimesh)
    # glTF Y-up -> TRELLIS Z-up. Scene.to_geometry() bakes scene transforms first.
    mesh.apply_transform(numpy.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]))
    return mesh
