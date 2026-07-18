from __future__ import annotations

import importlib
import importlib.util
import io as stdio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_3d_node_pack import load_package, rewrite_glb_document, write_glb


def write_rigged_glb(path: Path, *, textured: bool = True) -> None:
    write_glb(path, material=True, textured=textured, volumetric=True)

    def mutate(document):
        nodes = document.setdefault("nodes", [])
        nodes.extend(
            [
                {"mesh": 0, "skin": 0, "name": "mesh"},
                {"name": "root"},
                {"name": "joint"},
            ]
        )
        document["skins"] = [{"joints": [1, 2], "skeleton": 1}]
        document["scenes"] = [{"nodes": [0, 1]}]
        document["scene"] = 0

    rewrite_glb_document(path, mutate)


def load_worker_main():
    path = Path(__file__).resolve().parents[1] / "worker/skintokens/worker_main.py"
    spec = importlib.util.spec_from_file_location("comfycolab_skintokens_worker_main_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SkinTokensWorkerProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        load_package()
        self.worker = importlib.import_module("comfycolab_3d_test.skintokens_worker")

    def _command(self, root: Path, *, request_id: str = "skin-request-001"):
        return self.worker.SkinTokensWorkerCommand(
            python="cached-python",
            worker_script="skintokens_worker_main.py",
            source_dir="/content/SkinTokens",
            model_dir="/content/.comfycolab/models/3d/skintokens/skintokens-79736cad0fd8",
            qwen_dir="/content/.comfycolab/models/3d/skintokens/qwen3-0.6b-c1899de289a0",
            checkpoint="/content/.comfycolab/models/3d/skintokens/skintokens-79736cad0fd8/experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
            input_glb=str(root / "input.glb"),
            output_glb=str(root / "rigged.glb"),
            metadata_output=str(root / "rigged.json"),
            request_id=request_id,
            source_ref="273b691d35989d71cd17ff2895fdc735097b92d1",
            model_ref="79736cad0fd84de384d5eede659b4ebd24effe33",
            qwen_ref="c1899de289a04d12100db370d81485cdf75e47ca",
            environment_ref="g4-linux64-py311-torch270-cu128-skintokens-v1",
        )

    def _write_metadata(self, path: Path, *, requested: bool = True) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema": "comfycolab-skintokens-worker-result-v1",
                    "texture_preservation": {"requested": requested, "transfer_enabled": True},
                }
            ),
            encoding="utf-8",
        )

    def test_command_request_defaults_preserve_texture_and_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = self._command(Path(directory), request_id="req-abc")

        request = self.worker.build_skintokens_request(command)

        self.assertEqual(request["request_id"], "req-abc")
        self.assertTrue(request["preserve_texture"])
        self.assertTrue(request["use_transfer"])
        self.assertFalse(request["use_postprocess"])
        self.assertEqual(request["revisions"]["source"], command.source_ref)

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
                        'COMFYCOLAB_SKINTOKENS_READY={"protocol":1}\n'
                        'COMFYCOLAB_SKINTOKENS_RESULT={"request_id":"req-1","status":"ok",'
                        f'"output_glb":"{root / "rigged.glb"}","metadata_output":"{root / "rigged.json"}"}}\n'
                        'COMFYCOLAB_SKINTOKENS_RESULT={"request_id":"req-2","status":"ok",'
                        f'"output_glb":"{root / "rigged.glb"}","metadata_output":"{root / "rigged.json"}"}}\n'
                    )

                def poll(self):
                    return None

            write_rigged_glb(root / "rigged.glb")
            self._write_metadata(root / "rigged.json")
            pool = self.worker.SkinTokensWorkerPool(
                popen_factory=ReusableProcess,
                poll_interval=0.001,
            )
            try:
                pool.run(first)
                pool.run(second)
            finally:
                pool.close()

        self.assertEqual(len(launches), 1)

    def test_rejects_output_without_skin_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_glb(root / "unrigged.glb", material=True, textured=True, volumetric=True)
            with self.assertRaisesRegex(ValueError, "does not contain a skin"):
                self.worker.validate_skintokens_output(root / "unrigged.glb", preserve_texture=True)

    def test_cancellation_terminates_worker_group_and_removes_partials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = self._command(root)
            partial = root / ".rigged.partial.glb"
            partial.write_bytes(b"partial")

            class RunningProcess:
                pid = 24680

                def __init__(self, _argv, **_kwargs):
                    self.stdin = stdio.StringIO()
                    self.stdout = stdio.StringIO('COMFYCOLAB_SKINTOKENS_READY={"protocol":1}\n')

                def poll(self):
                    return None

                def wait(self, timeout=None):
                    return -15

            with mock.patch.object(self.worker.os, "killpg") as killpg:
                pool = self.worker.SkinTokensWorkerPool(
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
            output = Path(command.output_glb)
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
                        'COMFYCOLAB_SKINTOKENS_READY={"protocol":1}\n'
                        f'COMFYCOLAB_SKINTOKENS_RESULT={{"request_id":"{command.request_id}",'
                        '"status":"error","error_type":"RuntimeError","error":"boom"}\n'
                    )

                def poll(self):
                    return 1

                def wait(self, timeout=None):
                    return 1

            pool = self.worker.SkinTokensWorkerPool(
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

    def test_worker_main_validates_skin_contract(self) -> None:
        worker_main = load_worker_main()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rigged.glb"
            write_rigged_glb(path, textured=False)
            document = self.worker.validate_glb(path)

        contract = worker_main._validate_rig_contract(document)

        self.assertEqual(contract["skins"], 1)
        self.assertEqual(contract["joints"], 2)


if __name__ == "__main__":
    unittest.main()
