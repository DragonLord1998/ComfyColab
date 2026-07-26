from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "worker" / "pixal3d" / "mesh_cleanup.py"
POSTPROCESS_PATH = ROOT / "worker" / "pixal3d" / "ovoxel_postprocess.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "comfycolab_pixal3d_mesh_cleanup",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


class Pixal3DMeshCleanupTests(unittest.TestCase):
    def test_planar_cleanup_runs_only_after_attribute_sampling(self):
        source = POSTPROCESS_PATH.read_text()
        cleanup_call = source.index(
            "large_planar_component_face_mask(vertices_np, faces_np)"
        )

        self.assertEqual(
            source.count("large_planar_component_face_mask("),
            1,
        )
        self.assertGreater(cleanup_call, source.index("grid_sample_3d("))
        self.assertGreater(cleanup_call, source.index("vertices_np ="))

    def test_removes_only_large_disconnected_planar_component(self):
        module = load_module()
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-2.0, -2.0, -2.0],
                [2.0, -2.0, -2.0],
                [2.0, 2.0, -2.0],
                [-2.0, 2.0, -2.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray(
            [
                [0, 2, 1],
                [0, 1, 3],
                [0, 3, 2],
                [1, 2, 3],
                [4, 5, 6],
                [4, 6, 7],
            ],
            dtype=np.int64,
        )

        keep, removed = module.large_planar_component_face_mask(
            vertices,
            faces,
            minimum_face_count=2,
        )

        self.assertEqual(keep.tolist(), [True, True, True, True, False, False])
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]["faces"], 2)
        self.assertEqual(removed[0]["singular_ratio"], 0.0)

    def test_preserves_small_thin_detail(self):
        module = load_module()
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.1, 0.1, 0.1],
                [0.2, 0.1, 0.1],
                [0.2, 0.2, 0.1],
                [0.1, 0.2, 0.1],
            ],
            dtype=np.float64,
        )
        faces = np.asarray(
            [
                [0, 2, 1],
                [0, 1, 3],
                [0, 3, 2],
                [1, 2, 3],
                [4, 5, 6],
                [4, 6, 7],
            ],
            dtype=np.int64,
        )

        keep, removed = module.large_planar_component_face_mask(
            vertices,
            faces,
            minimum_face_count=2,
        )

        self.assertTrue(keep.all())
        self.assertEqual(removed, [])


if __name__ == "__main__":
    unittest.main()
