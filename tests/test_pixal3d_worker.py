from __future__ import annotations

import importlib
import importlib.util
import io as stdio
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_3d_node_pack import load_package, write_glb


def load_worker_main():
    path = Path(__file__).resolve().parents[1] / "worker/pixal3d/worker_main.py"
    spec = importlib.util.spec_from_file_location("comfycolab_pixal3d_worker_main_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Pixal3DWorkerProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        load_package()
        self.worker = importlib.import_module("comfycolab_3d_test.pixal3d_worker")

    def _command(self, root: Path, *, request_id: str = "pixal-request-001"):
        return self.worker.Pixal3DWorkerCommand(
            python="cached-python",
            worker_script="pixal3d_worker_main.py",
            source_dir="/content/Pixal3D",
            checkpoint_dir="/content/.cache/pixal3d",
            image_path=str(root / "input.png"),
            output_mesh=str(root / "model.glb"),
            metadata_output=str(root / "model.json"),
            request_id=request_id,
            seed=17,
            guidance_scale=5.5,
            inference_steps=30,
            camera_fov_degrees=49.1343426412,
            texture_size=2048,
            pipeline_type="1024_cascade",
            max_tokens=49_152,
        )

    def test_command_argv_carries_request_id_and_public_camera_fov(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self._command(Path(directory), request_id="req-abc")

        argv = command.argv()

        self.assertIn("--request-id", argv)
        self.assertEqual(argv[argv.index("--request-id") + 1], "req-abc")
        self.assertIn("--camera-fov-degrees", argv)
        self.assertEqual(
            argv[argv.index("--camera-fov-degrees") + 1],
            "49.1343426412",
        )

    def test_worker_request_converts_camera_fov_degrees_to_official_radians(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self._command(Path(directory))

        request = self.worker.build_pixal3d_request(command)

        self.assertNotIn("camera_fov_degrees", request["camera_params"])
        self.assertAlmostEqual(
            request["camera_params"]["camera_angle_x"],
            math.radians(command.camera_fov_degrees),
            places=10,
        )
        self.assertIn("naf_checkpoint", request["revisions"])

    def test_external_preprocess_crops_rgba_without_loading_rmbg(self) -> None:
        worker_main = load_worker_main()
        image_module = importlib.import_module("PIL.Image")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.png"
            image = image_module.new("RGBA", (20, 10), (0, 0, 0, 0))
            for x in range(8, 12):
                for y in range(3, 7):
                    image.putpixel((x, y), (255, 0, 0, 255))
            image.save(path)
            prepared = worker_main._prepare_image_without_rmbg(path)

        self.assertEqual(prepared.mode, "RGB")
        self.assertEqual(prepared.width, prepared.height)
        self.assertGreater(prepared.getpixel((prepared.width // 2, prepared.height // 2))[0], 0)

    def test_external_preprocess_rejects_empty_alpha(self) -> None:
        worker_main = load_worker_main()
        image_module = importlib.import_module("PIL.Image")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.png"
            image_module.new("RGBA", (8, 8), (0, 0, 0, 0)).save(path)
            with self.assertRaisesRegex(ValueError, "no visible foreground"):
                worker_main._prepare_image_without_rmbg(path)

    def test_reuses_running_worker_for_multiple_request_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._command(root, request_id="req-1")
            second = self._command(root, request_id="req-2")
            launches: list[list[str]] = []

            class ReusableProcess:
                pid = 12345

                def __init__(self, argv, **_kwargs):
                    launches.append(argv)
                    self.stdin = stdio.StringIO()
                    self.stdout = stdio.StringIO(
                        'COMFYCOLAB_PIXAL3D_READY={"protocol":1}\n'
                        'COMFYCOLAB_PIXAL3D_RESULT={"request_id":"req-1","status":"ok",'
                        f'"output_mesh":"{root / "model.glb"}","metadata_output":"{root / "model.json"}"}}\n'
                        'COMFYCOLAB_PIXAL3D_RESULT={"request_id":"req-2","status":"ok",'
                        f'"output_mesh":"{root / "model.glb"}","metadata_output":"{root / "model.json"}"}}\n'
                    )

                def poll(self):
                    return None

            write_glb(root / "model.glb", volumetric=True, textured=True)
            (root / "model.json").write_text("{}", encoding="utf-8")
            pool = self.worker.Pixal3DWorkerPool(
                popen_factory=ReusableProcess,
                poll_interval=0.001,
            )
            try:
                pool.run(first)
                pool.run(second)
            finally:
                pool.close()

        self.assertEqual(len(launches), 1)

    def test_restarts_worker_after_protocol_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = self._command(root)
            launches = 0

            class ExitedProcess:
                pid = 12345

                def __init__(self, _argv, **_kwargs):
                    nonlocal launches
                    launches += 1
                    write_glb(root / "model.glb", volumetric=True, textured=True)
                    (root / "model.json").write_text("{}", encoding="utf-8")
                    self.stdin = stdio.StringIO()
                    self.stdout = stdio.StringIO(
                        'COMFYCOLAB_PIXAL3D_READY={"protocol":1}\n'
                        f'COMFYCOLAB_PIXAL3D_RESULT={{"request_id":"{command.request_id}",'
                        f'"status":"ok","output_mesh":"{root / "model.glb"}",'
                        f'"metadata_output":"{root / "model.json"}"}}\n'
                    )

                def poll(self):
                    return 0

            pool = self.worker.Pixal3DWorkerPool(
                popen_factory=ExitedProcess,
                poll_interval=0.001,
            )
            try:
                pool.run(command)
                pool.run(command)
            finally:
                pool.close()

        self.assertEqual(launches, 2)

    def test_cancellation_terminates_worker_group_and_removes_partials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = self._command(root)
            partial = root / ".model.partial.glb"
            partial.write_bytes(b"partial")

            class RunningProcess:
                pid = 24680

                def __init__(self, _argv, **_kwargs):
                    self.stdin = stdio.StringIO()
                    self.stdout = stdio.StringIO('COMFYCOLAB_PIXAL3D_READY={"protocol":1}\n')

                def poll(self):
                    return None

                def wait(self, timeout=None):
                    return -15

            with mock.patch.object(self.worker.os, "killpg") as killpg:
                pool = self.worker.Pixal3DWorkerPool(
                    popen_factory=RunningProcess,
                    poll_interval=0.001,
                )
                try:
                    with self.assertRaisesRegex(InterruptedError, "cancelled"):
                        pool.run(command, is_cancelled=lambda: True)
                finally:
                    pool.close()

        killpg.assert_called_with(24680, self.worker.signal.SIGTERM)
        self.assertFalse(partial.exists())

    def test_failure_removes_output_metadata_and_partial_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = self._command(root)
            output = Path(command.output_mesh)
            metadata = Path(command.metadata_output)

            class FailedProcess:
                pid = 13579

                def __init__(self, _argv, **_kwargs):
                    output.write_bytes(b"partial")
                    output.with_suffix(".glb.partial").write_bytes(b"partial")
                    metadata.write_text("partial", encoding="utf-8")
                    metadata.with_suffix(".json.partial").write_text("partial", encoding="utf-8")
                    self.stdin = stdio.StringIO()
                    self.stdout = stdio.StringIO(
                        'COMFYCOLAB_PIXAL3D_READY={"protocol":1}\n'
                        f'COMFYCOLAB_PIXAL3D_RESULT={{"request_id":"{command.request_id}",'
                        '"status":"error","error_type":"RuntimeError","error":"boom"}\n'
                    )

                def poll(self):
                    return 1

                def wait(self, timeout=None):
                    return 1

            pool = self.worker.Pixal3DWorkerPool(
                popen_factory=FailedProcess,
                poll_interval=0.001,
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    pool.run(command)
            finally:
                pool.close()

            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".glb.partial").exists())
            self.assertFalse(metadata.exists())
            self.assertFalse(metadata.with_suffix(".json.partial").exists())

    def test_ignores_result_for_unrelated_request_id_until_matching_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = self._command(root, request_id="wanted")

            class MixedResponseProcess:
                pid = 97531

                def __init__(self, _argv, **_kwargs):
                    write_glb(root / "model.glb", volumetric=True, textured=True)
                    (root / "model.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
                    self.stdin = stdio.StringIO()
                    self.stdout = stdio.StringIO(
                        'COMFYCOLAB_PIXAL3D_READY={"protocol":1}\n'
                        f'COMFYCOLAB_PIXAL3D_RESULT={{"request_id":"other","status":"ok",'
                        f'"output_mesh":"{root / "wrong.glb"}","metadata_output":"{root / "wrong.json"}"}}\n'
                        f'COMFYCOLAB_PIXAL3D_RESULT={{"request_id":"wanted","status":"ok",'
                        f'"output_mesh":"{root / "model.glb"}","metadata_output":"{root / "model.json"}"}}\n'
                    )

                def poll(self):
                    return None

            pool = self.worker.Pixal3DWorkerPool(
                popen_factory=MixedResponseProcess,
                poll_interval=0.001,
            )
            try:
                result = pool.run(command)
            finally:
                pool.close()

        self.assertEqual(result["request_id"], "wanted")
        self.assertEqual(Path(result["output_mesh"]).name, "model.glb")


if __name__ == "__main__":
    unittest.main()
