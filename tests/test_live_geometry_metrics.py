from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "live_3d_g4_validation.py"


def load_module():
    name = "comfycolab_live_geometry_metrics"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_glb(
    path: Path,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
) -> None:
    positions = b"".join(struct.pack("<3f", *vertex) for vertex in vertices)
    padding = b"\x00" * ((-len(positions)) % 4)
    index_offset = len(positions) + len(padding)
    indices = b"".join(struct.pack("<3I", *face) for face in faces)
    binary = positions + padding + indices
    binary += b"\x00" * ((-len(binary)) % 4)
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": len(indices)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(vertices), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5125, "count": len(faces) * 3, "type": "SCALAR"},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
    }
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * ((-len(json_chunk)) % 4)
    payload = (
        struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_chunk) + 8 + len(binary))
        + struct.pack("<I4s", len(json_chunk), b"JSON")
        + json_chunk
        + struct.pack("<I4s", len(binary), b"BIN\x00")
        + binary
    )
    path.write_bytes(payload)


def box(
    *,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    size: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    ox, oy, oz = origin
    sx, sy, sz = size
    vertices = [
        (ox, oy, oz),
        (ox + sx, oy, oz),
        (ox + sx, oy + sy, oz),
        (ox, oy + sy, oz),
        (ox, oy, oz + sz),
        (ox + sx, oy, oz + sz),
        (ox + sx, oy + sy, oz + sz),
        (ox, oy + sy, oz + sz),
    ]
    faces = [
        (0, 2, 1), (0, 3, 2),
        (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4),
        (2, 3, 7), (2, 7, 6),
        (0, 4, 7), (0, 7, 3),
        (1, 2, 6), (1, 6, 5),
    ]
    return vertices, faces


class LiveGeometryMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def inspect(self, vertices, faces, *, strict=True):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.glb"
            write_glb(path, vertices, faces)
            return self.module.inspect_glb(
                path,
                require_textured=False,
                require_noncollapsed=strict,
            )

    def test_cube_records_complete_noncollapsed_geometry_evidence(self) -> None:
        vertices, faces = box()
        record = self.inspect(vertices, faces)
        metrics = record["geometryMetrics"]
        self.assertEqual(metrics["schema"], self.module.GEOMETRY_METRICS_SCHEMA)
        self.assertEqual(metrics["bounds"]["minimum"], [0.0, 0.0, 0.0])
        self.assertEqual(metrics["bounds"]["maximum"], [1.0, 1.0, 1.0])
        self.assertEqual(metrics["bounds"]["extents"], [1.0, 1.0, 1.0])
        self.assertEqual(metrics["centeredSvd"]["intrinsicRank"], 3)
        self.assertEqual(metrics["connectedComponents"], 1)
        self.assertEqual(metrics["nondegenerateFaceRatio"], 1.0)
        self.assertAlmostEqual(metrics["surfaceArea"], 6.0)
        self.assertTrue(metrics["nonCollapsed"])
        self.assertTrue(record["nonCollapsedGeometryValidated"])

    def test_xy_and_rotated_planes_fail_the_intrinsic_rank_gate(self) -> None:
        faces = [(0, 1, 2), (1, 3, 2)]
        planes = (
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
            [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 2)],
        )
        for vertices in planes:
            with self.subTest(vertices=vertices):
                legacy = self.inspect(vertices, faces, strict=False)
                self.assertEqual(legacy["geometryMetrics"]["centeredSvd"]["intrinsicRank"], 2)
                self.assertFalse(legacy["nonCollapsedGeometryValidated"])
                with self.assertRaisesRegex(ValueError, "geometrically collapsed.*intrinsicRank=2"):
                    self.inspect(vertices, faces, strict=True)

    def test_thin_but_rank_three_box_is_not_rejected(self) -> None:
        vertices, faces = box(size=(1.0, 1.0, 1e-4))
        record = self.inspect(vertices, faces)
        metrics = record["geometryMetrics"]
        self.assertEqual(metrics["centeredSvd"]["intrinsicRank"], 3)
        self.assertGreater(
            metrics["centeredSvd"]["singularValueRatios"][2],
            self.module.INTRINSIC_RANK_RELATIVE_THRESHOLD,
        )
        self.assertTrue(metrics["nonCollapsed"])

    def test_disconnected_components_are_recorded(self) -> None:
        first_vertices, first_faces = box()
        second_vertices, second_faces = box(origin=(3.0, 0.0, 0.0))
        offset = len(first_vertices)
        vertices = first_vertices + second_vertices
        faces = first_faces + [tuple(index + offset for index in face) for face in second_faces]
        record = self.inspect(vertices, faces)
        self.assertEqual(record["geometryMetrics"]["connectedComponents"], 2)

    def test_live_metrics_use_shared_contract_and_stage_events_are_parseable(self) -> None:
        vertices, faces = box()
        metrics = self.module.compute_geometry_metrics(vertices, faces)
        self.assertEqual(
            metrics["contractSchema"],
            self.module._GEOMETRY_QUALITY.GEOMETRY_QUALITY_SCHEMA,
        )
        runtime_metrics = self.module._GEOMETRY_QUALITY.analyze_geometry(
            vertices,
            faces,
            stage="TRELLIS raw shape",
        ).to_dict()
        line = "COMFYCOLAB_GEOMETRY_QUALITY=" + json.dumps(runtime_metrics)
        self.assertEqual(self.module.geometry_quality_events(line), [runtime_metrics])
        compact = self.module.compact_log_evidence(line + "\n" + ("x" * 20_000))
        self.assertIn(line, compact)

    def test_benchmark_strict_path_requires_geometry_metrics(self) -> None:
        spec = self.module.CASES["trellis_512"]
        marker = "ComfyColab shape metrics: 3964 tokens at resolution 512"
        with self.assertRaisesRegex(RuntimeError, "lacks passing non-collapsed"):
            self.module.benchmark_from(
                spec,
                1.0,
                1024,
                {"bytes": 100, "faces": 12},
                marker,
                require_geometry_metrics=True,
            )
        vertices, faces = box()
        glb = self.inspect(vertices, faces)
        benchmark = self.module.benchmark_from(
            spec,
            1.0,
            1024,
            glb,
            marker,
            require_geometry_metrics=True,
        )
        self.assertTrue(benchmark["nonCollapsedGeometryValidated"])
        self.assertTrue(benchmark["geometryMetrics"]["nonCollapsed"])

    def test_strict_merge_ignores_legacy_structural_only_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            self.module.atomic_json(
                state / "run.json",
                {"schema": self.module.STATE_SCHEMA, "runId": "run-1"},
            )
            legacy_record = {
                "schema": self.module.CASE_SCHEMA,
                "status": "passed",
                "case": "trellis_512",
                "kind": "trellis",
                "gate": "trellis_512_textured_glb",
                "benchmarkName": "trellis_512",
                "runId": "run-1",
                "evidence": "legacy",
                "benchmark": {"status": "passed", "glbValidated": True},
                "previewSaveProof": {"saveArtifactValidated": True},
                "glb": {"bytes": 100, "faces": 12},
            }
            self.module.atomic_json(
                state / "cases/trellis_512/record.json",
                legacy_record,
            )
            template = {
                "schema": "comfycolab-3d-live-validation-v1",
                "status": "pending",
                "gates": {
                    "trellis_512_textured_glb": {"status": "pending", "evidence": None},
                },
                "benchmarks": {"trellis_512": {"status": "pending"}},
            }
            template_path = root / "template.json"
            output_path = root / "output.json"
            self.module.atomic_json(template_path, template)
            args = argparse.Namespace(
                state_dir=state,
                template=template_path,
                output=output_path,
                require_geometry_evidence=True,
            )
            self.assertEqual(self.module.merge_command(args), 0)
            merged = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(merged["gates"]["trellis_512_textured_glb"]["status"], "pending")
            self.assertEqual(merged["benchmarks"]["trellis_512"]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
