from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "custom_nodes" / "ComfyColab-3D"
PACKAGE_NAME = "comfycolab_geometry_quality_test"


PLANE_VERTICES = [
    (-1.0, -1.0, 0.0),
    (1.0, -1.0, 0.0),
    (1.0, 1.0, 0.0),
    (-1.0, 1.0, 0.0),
]
PLANE_FACES = [(0, 1, 2), (0, 2, 3)]
ROTATED_PLANE_VERTICES = [(x, y, x + 2.0 * y) for x, y, _ in PLANE_VERTICES]
THIN_BOX_VERTICES = [
    (x, y, z)
    for z in (-1.0e-6, 1.0e-6)
    for y in (-1.0, 1.0)
    for x in (-1.0, 1.0)
]
THIN_BOX_FACES = [
    (0, 1, 3), (0, 3, 2),
    (4, 6, 7), (4, 7, 5),
    (0, 4, 5), (0, 5, 1),
    (2, 3, 7), (2, 7, 6),
    (0, 2, 6), (0, 6, 4),
    (1, 5, 7), (1, 7, 3),
]


def load_modules():
    for module_name in list(sys.modules):
        if module_name == PACKAGE_NAME or module_name.startswith(PACKAGE_NAME + "."):
            del sys.modules[module_name]
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(PACKAGE_DIR)]
    sys.modules[PACKAGE_NAME] = package
    loaded = []
    for name in ("geometry_quality", "file3d"):
        qualified_name = f"{PACKAGE_NAME}.{name}"
        spec = importlib.util.spec_from_file_location(qualified_name, PACKAGE_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        assert spec.loader
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


def write_glb(path: Path, vertices, faces):
    position_values = [coordinate for vertex in vertices for coordinate in vertex]
    index_values = [index for face in faces for index in face]
    positions = struct.pack(f"<{len(position_values)}f", *position_values)
    indices = struct.pack(f"<{len(index_values)}H", *index_values)
    indices += b"\x00" * ((4 - len(indices) % 4) % 4)
    binary = positions + indices
    document = {
        "asset": {"version": "2.0"},
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(vertices),
                "type": "VEC3",
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": len(index_values),
                "type": "SCALAR",
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {
                "buffer": 0,
                "byteOffset": len(positions),
                "byteLength": len(index_values) * 2,
            },
        ],
        "buffers": [{"byteLength": len(binary)}],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}
        ],
    }
    json_chunk = json.dumps(document, separators=(",", ":")).encode()
    json_chunk += b" " * ((4 - len(json_chunk) % 4) % 4)
    body = (
        struct.pack("<I4s", len(json_chunk), b"JSON")
        + json_chunk
        + struct.pack("<I4s", len(binary), b"BIN\x00")
        + binary
    )
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)


class GeometryQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.quality, cls.file3d = load_modules()

    @classmethod
    def tearDownClass(cls):
        for module_name in list(sys.modules):
            if module_name == PACKAGE_NAME or module_name.startswith(PACKAGE_NAME + "."):
                del sys.modules[module_name]

    def test_xy_plane_is_rejected_as_rank_two(self):
        metrics = self.quality.analyze_geometry(
            PLANE_VERTICES, PLANE_FACES, stage="trellis-source"
        )
        self.assertEqual(metrics.schema, "comfycolab-3d-geometry-quality-v1")
        self.assertEqual(metrics.numerical_rank, 2)
        self.assertEqual(metrics.nondegenerate_face_ratio, 1.0)
        self.assertTrue(metrics.is_numerically_collapsed)
        self.assertIn("vertex_rank_below_3", metrics.collapse_reasons)
        self.assertEqual(metrics.to_dict()["stage"], "trellis-source")

    def test_rotated_plane_is_rejected_without_axis_heuristics(self):
        metrics = self.quality.analyze_geometry(
            ROTATED_PLANE_VERTICES, PLANE_FACES, stage="rotated-plane"
        )
        self.assertEqual(metrics.numerical_rank, 2)
        self.assertTrue(metrics.is_numerically_collapsed)
        self.assertEqual(metrics.smallest_to_largest_singular_ratio, 0.0)

    def test_unused_off_plane_vertex_cannot_make_a_planar_mesh_pass(self):
        metrics = self.quality.analyze_geometry(
            [*PLANE_VERTICES, (0.0, 0.0, 10.0)],
            PLANE_FACES,
            stage="plane-with-unused-depth",
        )
        self.assertEqual(metrics.vertex_count, 5)
        self.assertEqual(metrics.referenced_vertex_count, 4)
        self.assertEqual(metrics.extents[2], 0.0)
        self.assertEqual(metrics.numerical_rank, 2)
        self.assertTrue(metrics.is_numerically_collapsed)

    def test_very_thin_rank_three_box_is_accepted(self):
        metrics = self.quality.analyze_geometry(
            THIN_BOX_VERTICES, THIN_BOX_FACES, stage="thin-box"
        )
        self.assertEqual(metrics.numerical_rank, 3)
        self.assertFalse(metrics.is_numerically_collapsed)
        self.assertTrue(metrics.passes_volumetric_validation)
        self.assertTrue(metrics.is_very_thin)
        self.assertIn("very_thin_rank_3_geometry", metrics.warnings)
        self.assertAlmostEqual(metrics.extents[2], 2.0e-6)
        self.assertGreater(metrics.surface_area, 8.0)

    def test_nondegenerate_face_ratio_and_surface_area_are_reported(self):
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        metrics = self.quality.analyze_geometry(
            vertices, [(0, 1, 2), (3, 3, 0)], stage="mixed-faces"
        )
        self.assertEqual(metrics.nondegenerate_face_count, 1)
        self.assertEqual(metrics.nondegenerate_face_ratio, 0.5)
        self.assertAlmostEqual(metrics.surface_area, 0.5)
        self.assertFalse(metrics.is_numerically_collapsed)

    def test_connected_components_are_reported(self):
        vertices = [
            (0, 0, 0), (1, 0, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 2),
        ]
        metrics = self.quality.analyze_geometry(
            vertices, [(0, 1, 2), (3, 4, 5)], stage="two-components"
        )
        self.assertEqual(metrics.connected_component_count, 2)
        self.assertTrue(metrics.connected_components_exact)
        self.assertEqual(metrics.to_dict()["connected_component_count"], 2)

    def test_raw_array_path_is_chunked_and_never_iterates_or_materializes_rows(self):
        try:
            numpy = importlib.import_module("numpy")
        except ModuleNotFoundError:
            self.skipTest("NumPy is unavailable")

        class VectorizedOnly:
            def __init__(self, values, dtype):
                self._values = numpy.asarray(values, dtype=dtype)
                self.shape = self._values.shape
                self.slice_count = 0

            def __getitem__(self, index):
                self.slice_count += 1
                return self._values[index]

            def __iter__(self):
                raise AssertionError("raw validation must not iterate array rows")

            def tolist(self):
                raise AssertionError("raw validation must not materialize arrays as lists")

        vertices = VectorizedOnly(
            [*THIN_BOX_VERTICES, (0.0, 0.0, 100.0)],
            numpy.float64,
        )
        faces = VectorizedOnly(THIN_BOX_FACES, numpy.int64)
        with mock.patch.object(self.quality, "_VECTORIZED_CHUNK_ROWS", 2):
            metrics = self.quality.analyze_geometry(
                vertices,
                faces,
                stage="raw-shape",
                analysis_mode="raw",
            )
        self.assertEqual(metrics.analysis_mode, "raw")
        self.assertEqual(metrics.vertex_count, 9)
        self.assertEqual(metrics.referenced_vertex_count, 8)
        self.assertAlmostEqual(metrics.bounds_max[2], 1.0e-6)
        self.assertEqual(metrics.numerical_rank, 3)
        self.assertFalse(metrics.is_numerically_collapsed)
        self.assertEqual(metrics.connected_component_count, -1)
        self.assertFalse(metrics.connected_components_exact)
        self.assertGreater(vertices.slice_count, 2)
        self.assertGreater(faces.slice_count, 2)

    def test_raw_small_list_path_remains_dependency_free(self):
        with mock.patch.object(
            self.quality.importlib,
            "import_module",
            side_effect=AssertionError("small lists must not import NumPy"),
        ):
            metrics = self.quality.analyze_geometry(
                THIN_BOX_VERTICES,
                THIN_BOX_FACES,
                stage="raw-small-list",
                analysis_mode="raw",
            )
        self.assertEqual(metrics.numerical_rank, 3)
        self.assertEqual(metrics.connected_component_count, -1)
        self.assertFalse(metrics.connected_components_exact)

    def test_trimesh_like_validation_uses_vertices_and_faces_only(self):
        mesh = types.SimpleNamespace(vertices=THIN_BOX_VERTICES, faces=THIN_BOX_FACES)
        metrics = self.quality.validate_volumetric_mesh(mesh, stage="mesh")
        self.assertEqual(metrics.numerical_rank, 3)
        planar = types.SimpleNamespace(vertices=PLANE_VERTICES, faces=PLANE_FACES)
        with self.assertRaisesRegex(ValueError, "numerically collapsed"):
            self.quality.validate_volumetric_mesh(planar, stage="mesh")

    def test_glb_gate_is_opt_in_and_preserves_structural_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            plane_path = Path(temporary) / "plane.glb"
            box_path = Path(temporary) / "thin-box.glb"
            write_glb(plane_path, PLANE_VERTICES, PLANE_FACES)
            write_glb(box_path, THIN_BOX_VERTICES, THIN_BOX_FACES)
            self.assertTrue(self.file3d.validate_glb(plane_path)["meshes"])
            with self.assertRaisesRegex(ValueError, "numerically collapsed"):
                self.file3d.validate_volumetric_glb(plane_path, stage="trellis-source")
            metrics = self.quality.validate_volumetric_glb(box_path, stage="trellis-source")
            self.assertEqual(metrics.numerical_rank, 3)
            self.assertEqual(metrics.stage, "trellis-source")


if __name__ == "__main__":
    unittest.main()
