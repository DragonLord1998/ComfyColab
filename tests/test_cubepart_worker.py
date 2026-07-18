from __future__ import annotations

import importlib
import importlib.util
import io as stdio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_3d_node_pack import load_package, write_glb


ROOT = Path(__file__).resolve().parents[1]


def load_worker_main():
    path = ROOT / "worker/cubepart/worker_main.py"
    spec = importlib.util.spec_from_file_location("comfycolab_cubepart_worker_main_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CubePartWorkerProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        load_package()
        self.worker = importlib.import_module("comfycolab_3d_test.cubepart_worker")

    def _command(self, root: Path, *, request_id: str = "cube-request-001", part_names=None):
        return self.worker.CubePartWorkerCommand(
            python="cached-python",
            worker_script="worker_main.py",
            source_dir="/content/cube/cubepart",
            weights_dir="/content/.comfycolab/models/3d/cubepart/cubepart-28431d124e77",
            input_mesh=str(root / "input.glb"),
            output_dir=str(root / "parts-output"),
            request_id=request_id,
            part_names=part_names if part_names is not None else ("body", "wheel"),
            accept_research_license=True,
        )

    def _write_result(self, output_dir: Path, request_id: str, part_names=("body", "wheel")) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_glb(output_dir / "parts.glb", volumetric=True)
        parts = []
        for index, name in enumerate(part_names):
            file_name = f"part_{index:02d}.glb"
            write_glb(output_dir / file_name, volumetric=True)
            parts.append({"index": index, "name": name, "file": file_name})
        (output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "comfycolab-cubepart-result-v1",
                    "request_id": request_id,
                    "part_names": list(part_names),
                    "parts": parts,
                }
            ),
            encoding="utf-8",
        )

    def test_request_requires_non_empty_ordered_part_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self._command(Path(directory), part_names=" , \n")
            with self.assertRaisesRegex(ValueError, "non-empty ordered part_names"):
                self.worker.build_cubepart_request(command)

    def test_request_requires_research_license_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self._command(Path(directory), part_names=["body"])
            blocked = self.worker.CubePartWorkerCommand(
                **{**command.__dict__, "accept_research_license": False}
            )
            with self.assertRaisesRegex(PermissionError, "research-only"):
                self.worker.build_cubepart_request(blocked)

    def test_request_records_pinned_revisions_and_license_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = self.worker.build_cubepart_request(
                self._command(Path(directory), part_names="body, front wheel")
            )

        self.assertEqual(request["part_names"], ["body", "front wheel"])
        self.assertEqual(request["revisions"]["source"], self.worker.CUBEPART_SOURCE_REF)
        self.assertEqual(request["revisions"]["model"], self.worker.CUBEPART_MODEL_REF)
        self.assertEqual(request["license"]["weights_repo"], "Roblox/cubepart")
        self.assertEqual(request["license"]["required_acceptance"], "accept_research_license")

    def test_manifest_requires_one_glb_per_requested_part(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "parts-output"
            self._write_result(output, "request", part_names=("body",))
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["part_names"] = ["body", "wheel"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "every requested part"):
                self.worker.validate_cubepart_output(output, ("body", "wheel"))

    def test_importing_comfyui_client_does_not_import_torch_or_trimesh(self) -> None:
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("trimesh", sys.modules)

    def test_worker_main_import_does_not_import_torch_or_trimesh(self) -> None:
        sys.modules.pop("torch", None)
        sys.modules.pop("trimesh", None)
        load_worker_main()
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("trimesh", sys.modules)

    def test_pool_validates_result_manifest_and_reuses_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = self._command(root)
            launches: list[list[str]] = []
            test_case = self

            class ReusableProcess:
                pid = 12345

                def __init__(self, argv, **_kwargs):
                    launches.append(argv)

                    class InputSink(stdio.StringIO):
                        def write(self, value):
                            request = json.loads(value)
                            test_case._write_result(
                                root / "parts-output",
                                request["request_id"],
                                tuple(request["part_names"]),
                            )
                            return super().write(value)

                    self.stdin = InputSink()
                    self.stdout = stdio.StringIO(
                        'COMFYCOLAB_CUBEPART_READY={"protocol":1}\n'
                        f'COMFYCOLAB_CUBEPART_RESULT={{"request_id":"{command.request_id}",'
                        f'"status":"ok","output_dir":"{root / "parts-output"}"}}\n'
                        f'COMFYCOLAB_CUBEPART_RESULT={{"request_id":"second",'
                        f'"status":"ok","output_dir":"{root / "parts-output"}"}}\n'
                    )

                def poll(self):
                    return None

            with mock.patch.object(self.worker, "validate_volumetric_glb", return_value=None):
                pool = self.worker.CubePartWorkerPool(
                    popen_factory=ReusableProcess,
                    poll_interval=0.001,
                )
                try:
                    first = pool.run(command)
                    second = self.worker.CubePartWorkerCommand(
                        **{**command.__dict__, "request_id": "second"}
                    )
                    pool.run(second)
                finally:
                    pool.close()

        self.assertEqual(len(launches), 1)
        self.assertEqual(first["manifest"]["part_names"], ["body", "wheel"])

    def test_cancellation_terminates_worker_group_and_removes_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = self._command(root)
            output_dir = root / "parts-output"
            output_dir.mkdir()
            (output_dir / "stale.txt").write_text("stale", encoding="utf-8")

            class RunningProcess:
                pid = 24680

                def __init__(self, _argv, **_kwargs):
                    self.stdin = stdio.StringIO()
                    self.stdout = stdio.StringIO('COMFYCOLAB_CUBEPART_READY={"protocol":1}\n')

                def poll(self):
                    return None

                def wait(self, timeout=None):
                    return -15

            with mock.patch.object(self.worker.os, "killpg") as killpg:
                pool = self.worker.CubePartWorkerPool(
                    popen_factory=RunningProcess,
                    poll_interval=0.001,
                )
                try:
                    with self.assertRaisesRegex(InterruptedError, "cancelled"):
                        pool.run(command, is_cancelled=lambda: True)
                finally:
                    pool.close()

        killpg.assert_called_with(24680, self.worker.signal.SIGTERM)
        self.assertFalse(output_dir.exists())


class CubePartWorkerMainValidationTests(unittest.TestCase):
    def test_worker_main_rejects_empty_schema_before_pipeline_load(self) -> None:
        worker_main = load_worker_main()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_glb(root / "input.glb", volumetric=True)
            request = {
                "protocol": worker_main.PROTOCOL_VERSION,
                "request_id": "req",
                "input_mesh": str(root / "input.glb"),
                "output_dir": str(root / "out"),
                "part_names": [],
                "accept_research_license": True,
                "license": worker_main._license_metadata(),
                "seed": 0,
                "num_inference_steps": 1,
                "num_samples": 1,
                "scheduler": "dpm_solver",
            }
            with self.assertRaisesRegex(ValueError, "non-empty ordered part_names"):
                worker_main._validate_request(request)

    def test_worker_main_rejects_missing_license_acceptance(self) -> None:
        worker_main = load_worker_main()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_glb(root / "input.glb", volumetric=True)
            request = {
                "protocol": worker_main.PROTOCOL_VERSION,
                "request_id": "req",
                "input_mesh": str(root / "input.glb"),
                "output_dir": str(root / "out"),
                "part_names": ["body"],
                "accept_research_license": False,
                "license": worker_main._license_metadata(),
                "seed": 0,
                "num_inference_steps": 1,
                "num_samples": 1,
                "scheduler": "dpm_solver",
            }
            with self.assertRaisesRegex(PermissionError, "accept_research_license"):
                worker_main._validate_request(request)


if __name__ == "__main__":
    unittest.main()
