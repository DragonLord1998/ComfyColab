from __future__ import annotations

import contextlib
import io
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TRANSFORMS = ROOT / "worker" / "ultrashape" / "transform_contract.py"
SEEDS = ROOT / "worker" / "ultrashape" / "seed_contract.py"
WORKER = ROOT / "worker" / "ultrashape" / "worker_main.py"


def load_transforms():
    spec = importlib.util.spec_from_file_location("comfycolab_transform_contract", TRANSFORMS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_seeds():
    spec = importlib.util.spec_from_file_location("comfycolab_seed_contract", SEEDS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_worker():
    module_name = "comfycolab_ultrashape_worker_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, WORKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class UltraShapeTransformTests(unittest.TestCase):
    def test_asymmetric_bounds_round_trip_exactly(self) -> None:
        transforms = load_transforms()
        contract = transforms.normalization_from_bounds(
            (-7.0, 2.0, -0.5), (5.0, 8.0, 3.5), normalize_scale=0.99
        )
        point = (4.25, 3.5, 2.75)
        normalized = transforms.apply_matrix_to_point(contract["forward"], point)
        restored = transforms.apply_matrix_to_point(contract["inverse"], normalized)
        for actual, expected in zip(restored, point):
            self.assertAlmostEqual(actual, expected, places=10)
        self.assertTrue(
            transforms.matrices_are_inverse(contract["forward"], contract["inverse"])
        )

    def test_y_up_and_z_up_are_inverse_without_reflection(self) -> None:
        transforms = load_transforms()
        y_up = (2.0, 5.0, -3.0)
        z_up = transforms.apply_matrix_to_point(transforms.y_up_to_z_up_matrix(), y_up)
        self.assertEqual(z_up, (2.0, 3.0, 5.0))
        restored = transforms.apply_matrix_to_point(transforms.z_up_to_y_up_matrix(), z_up)
        self.assertEqual(restored, y_up)
        self.assertTrue(
            transforms.matrices_are_inverse(
                transforms.y_up_to_z_up_matrix(), transforms.z_up_to_y_up_matrix()
            )
        )

    def test_zero_extent_is_rejected(self) -> None:
        transforms = load_transforms()
        with self.assertRaisesRegex(ValueError, "zero spatial extent"):
            transforms.normalization_from_bounds((1, 1, 1), (1, 1, 1))

    def test_rotation_preserves_winding_and_transforms_asymmetric_normal(self) -> None:
        transforms = load_transforms()
        points = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 1.0, 3.0)]
        rotated = [
            transforms.apply_matrix_to_point(transforms.y_up_to_z_up_matrix(), point)
            for point in points
        ]

        def cross(left, right):
            return (
                left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0],
            )

        original_normal = cross(
            tuple(points[1][axis] - points[0][axis] for axis in range(3)),
            tuple(points[2][axis] - points[0][axis] for axis in range(3)),
        )
        rotated_normal = cross(
            tuple(rotated[1][axis] - rotated[0][axis] for axis in range(3)),
            tuple(rotated[2][axis] - rotated[0][axis] for axis in range(3)),
        )
        expected = transforms.apply_matrix_to_point(
            transforms.y_up_to_z_up_matrix(), original_normal
        )
        self.assertEqual(rotated_normal, expected)


class UltraShapeSeedTests(unittest.TestCase):
    def test_numpy_sampling_generator_is_repeatable_and_seed_sensitive(self) -> None:
        seeds = load_seeds()
        first = seeds.make_numpy_rng(8128).random(32)
        repeated = seeds.make_numpy_rng(8128).random(32)
        different = seeds.make_numpy_rng(8129).random(32)
        self.assertTrue((first == repeated).all())
        self.assertFalse((first == different).all())

    def test_torch_global_seed_is_set_before_voxelization(self) -> None:
        source = WORKER.read_text(encoding="utf-8")
        seed_position = source.index("torch.manual_seed(args.seed)")
        voxel_position = source.index("voxelize_from_point(", seed_position)
        self.assertLess(seed_position, voxel_position)


class UltraShapeWorkerCliTests(unittest.TestCase):
    def test_help_does_not_import_cuda_dependencies(self) -> None:
        result = subprocess.run(
            [sys.executable, str(WORKER), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--octree-resolution", result.stdout)
        self.assertIn("--metadata-output", result.stdout)

    def test_failed_worker_removes_stale_and_partial_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "refined.glb"
            partial_output = root / "refined.partial.glb"
            metadata = root / "transform.json"
            partial_metadata = root / "transform.json.partial"
            for path in (output, partial_output, metadata, partial_metadata):
                path.write_bytes(b"stale")
            result = subprocess.run(
                [
                    sys.executable,
                    str(WORKER),
                    "--source-dir",
                    str(root / "missing-source"),
                    "--checkpoint",
                    str(root / "missing-checkpoint"),
                    "--dinov2-dir",
                    str(root / "missing-dino"),
                    "--input-mesh",
                    str(root / "missing.glb"),
                    "--reference-image",
                    str(root / "missing.png"),
                    "--output-mesh",
                    str(output),
                    "--metadata-output",
                    str(metadata),
                    "--steps",
                    "12",
                    "--num-latents",
                    "8192",
                    "--octree-resolution",
                    "384",
                    "--decode-chunk-size",
                    "2048",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn('"status": "error"', result.stdout)
            for path in (output, partial_output, metadata, partial_metadata):
                self.assertFalse(path.exists(), path)

    def test_worker_preflight_rejects_planar_mesh(self) -> None:
        worker = load_worker()
        planar = SimpleNamespace(
            vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            faces=[(0, 1, 2)],
        )
        with self.assertRaisesRegex(ValueError, "UltraShape input mesh.*PCA rank=2"):
            worker._validate_volumetric_mesh(planar, stage="UltraShape input mesh")

    def test_worker_emits_structured_empty_surface_error(self) -> None:
        worker = load_worker()

        class NoDecodableSurface(RuntimeError):
            requested_resolution = 1024
            octree_depth = 1008
            preceding_active_points = 17

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = [
                "--source-dir", str(root / "source"),
                "--checkpoint", str(root / "checkpoint"),
                "--dinov2-dir", str(root / "dino"),
                "--input-mesh", str(root / "input.glb"),
                "--reference-image", str(root / "image.png"),
                "--output-mesh", str(root / "output.glb"),
                "--metadata-output", str(root / "transform.json"),
                "--steps", "24",
                "--num-latents", "16384",
                "--octree-resolution", "1024",
                "--decode-chunk-size", "4096",
                "--seed", "9",
            ]
            output = io.StringIO()
            error_output = io.StringIO()
            with mock.patch.object(
                worker, "run", side_effect=NoDecodableSurface("empty")
            ), contextlib.redirect_stdout(output), contextlib.redirect_stderr(error_output):
                return_code = worker.main(arguments)
        self.assertEqual(return_code, 1)
        result_line = next(
            line for line in output.getvalue().splitlines()
            if line.startswith(worker.RESULT_PREFIX)
        )
        payload = json.loads(result_line.removeprefix(worker.RESULT_PREFIX))
        self.assertEqual(payload["error_code"], "no_decodable_surface")
        self.assertEqual(payload["octree_resolution"], 1024)
        self.assertEqual(payload["octree_depth"], 1008)
        self.assertEqual(payload["preceding_active_points"], 17)
        self.assertEqual(payload["seed"], 9)


if __name__ == "__main__":
    unittest.main()
