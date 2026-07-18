from __future__ import annotations

import importlib
import importlib.util
import io as stdio
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_3d_node_pack import load_package, rewrite_glb_document, write_glb


def write_rigged_glb(
    path: Path,
    *,
    textured: bool = True,
    weight_sum: float = 1.0,
    joint_index: int = 0,
) -> None:
    binary = bytearray()
    buffer_views: list[dict[str, int]] = []

    def append_view(payload: bytes) -> int:
        offset = len(binary)
        binary.extend(payload)
        binary.extend(b"\x00" * ((4 - len(binary) % 4) % 4))
        buffer_views.append(
            {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
        )
        return len(buffer_views) - 1

    positions = struct.pack(
        "<12f",
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    indices = struct.pack("<12H", 0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3)
    uv_values = struct.pack("<8f", 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    joints = struct.pack("<16H", *([joint_index, 0, 0, 0] * 4))
    weights = struct.pack("<16f", *([weight_sum, 0.0, 0.0, 0.0] * 4))
    identity = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    inverse_bind = struct.pack("<32f", *(identity * 2))

    position_view = append_view(positions)
    index_view = append_view(indices)
    uv_view = append_view(uv_values) if textured else None
    image_view = append_view(b"fake-png") if textured else None
    joint_view = append_view(joints)
    weight_view = append_view(weights)
    inverse_bind_view = append_view(inverse_bind)

    accessors = [
        {"bufferView": position_view, "componentType": 5126, "count": 4, "type": "VEC3"},
        {"bufferView": index_view, "componentType": 5123, "count": 12, "type": "SCALAR"},
    ]
    attributes = {"POSITION": 0}
    if textured and uv_view is not None:
        accessors.append(
            {"bufferView": uv_view, "componentType": 5126, "count": 4, "type": "VEC2"}
        )
        attributes["TEXCOORD_0"] = 2
    joint_accessor = len(accessors)
    accessors.append(
        {"bufferView": joint_view, "componentType": 5123, "count": 4, "type": "VEC4"}
    )
    weight_accessor = len(accessors)
    accessors.append(
        {"bufferView": weight_view, "componentType": 5126, "count": 4, "type": "VEC4"}
    )
    inverse_bind_accessor = len(accessors)
    accessors.append(
        {
            "bufferView": inverse_bind_view,
            "componentType": 5126,
            "count": 2,
            "type": "MAT4",
        }
    )
    attributes["JOINTS_0"] = joint_accessor
    attributes["WEIGHTS_0"] = weight_accessor
    primitive = {"attributes": attributes, "indices": 1, "material": 0}
    document = {
        "asset": {"version": "2.0"},
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(binary)}],
        "meshes": [{"primitives": [primitive]}],
        "nodes": [
            {"mesh": 0, "skin": 0, "name": "mesh"},
            {"name": "root", "children": [2]},
            {"name": "joint"},
        ],
        "skins": [
            {
                "joints": [1, 2],
                "skeleton": 1,
                "inverseBindMatrices": inverse_bind_accessor,
            }
        ],
        "scenes": [{"nodes": [0, 1]}],
        "scene": 0,
        "materials": [{"pbrMetallicRoughness": {}}],
    }
    if textured and image_view is not None:
        document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
        document["textures"] = [{"source": 0}]
        document["images"] = [{"bufferView": image_view, "mimeType": "image/png"}]
    chunk = json.dumps(document, separators=(",", ":")).encode()
    chunk += b" " * ((4 - len(chunk) % 4) % 4)
    body = (
        struct.pack("<I4s", len(chunk), b"JSON")
        + chunk
        + struct.pack("<I4s", len(binary), b"BIN\x00")
        + bytes(binary)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)


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
            environment_ref=(
                "g4-linux64-py31115-torch270-cu128-bpy4222-skintokens-v2"
            ),
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
        self.assertEqual(request["max_generation_attempts"], 4)
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

    def test_rejects_skin_without_joint_and_weight_accessors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unweighted.glb"
            write_rigged_glb(path)

            def remove_weights(document):
                attributes = document["meshes"][0]["primitives"][0]["attributes"]
                attributes.pop("WEIGHTS_0")

            rewrite_glb_document(path, remove_weights)
            with self.assertRaisesRegex(ValueError, "JOINTS_0, and WEIGHTS_0"):
                self.worker.validate_skintokens_output(
                    path, preserve_texture=True
                )

    def test_rejects_skin_with_unnormalized_weight_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-weights.glb"
            write_rigged_glb(path, weight_sum=0.5)

            with self.assertRaisesRegex(ValueError, "not normalized"):
                self.worker.validate_skintokens_output(path, preserve_texture=True)

    def test_rejects_skin_with_unbound_joint_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-joints.glb"
            write_rigged_glb(path, joint_index=2)

            with self.assertRaisesRegex(ValueError, "unbound joint"):
                self.worker.validate_skintokens_output(path, preserve_texture=True)

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
                        '"status":"error","error_type":"RuntimeError","error":"boom",'
                        '"traceback":"worker stack sentinel"}\n'
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
                with self.assertRaisesRegex(RuntimeError, "worker stack sentinel"):
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

            contract = worker_main._validate_rig_contract(path, preserve_texture=False)

        self.assertEqual(contract["skins"], 1)
        self.assertEqual(contract["joints"], 2)
        self.assertEqual(contract["skinned_primitives"], 1)
        self.assertEqual(contract["weighted_vertices"], 4)
        self.assertEqual(contract["maximum_joint_index"], 0)
        self.assertAlmostEqual(contract["minimum_weight_sum"], 1.0)
        self.assertAlmostEqual(contract["maximum_weight_sum"], 1.0)

    def test_worker_main_attests_environment_from_python_marker(self) -> None:
        worker_main = load_worker_main()
        versions = {
            "python": "3.11.15",
            "bpy": "4.2.22",
            "torch": "2.7.0+cu128",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "env"
            python = root / "bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("", encoding="utf-8")
            marker = root / ".comfycolab-environment.json"
            marker.write_text(
                json.dumps(
                    {
                        "schema": "comfycolab-skintokens-artifacts-v1",
                        "environment_ref": "env-ref",
                        "python": str(python),
                        "python_version": "3.11.15",
                        "versions": versions,
                    }
                ),
                encoding="utf-8",
            )

            attestation = worker_main._load_environment_attestation(python)

        self.assertEqual(attestation["environment_ref"], "env-ref")
        self.assertEqual(attestation["versions"], versions)
        self.assertEqual(attestation["python_version"], "3.11.15")

    def test_worker_main_uses_active_venv_prefix_for_environment_marker(self) -> None:
        worker_main = load_worker_main()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "env"
            python = root / "bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to("/managed/python3.11")
            (root / ".comfycolab-environment.json").write_text(
                json.dumps(
                    {
                        "schema": "comfycolab-skintokens-artifacts-v1",
                        "environment_ref": "measured-env-ref",
                        "python_version": "3.11.15",
                        "versions": {"python": "3.11.15"},
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(
                worker_main.sys, "prefix", str(root)
            ), mock.patch.object(
                worker_main.sys, "executable", str(python)
            ):
                attestation = worker_main._load_environment_attestation()

        self.assertEqual(attestation["environment_ref"], "measured-env-ref")
        self.assertEqual(attestation["marker"], str(root / ".comfycolab-environment.json"))

    def test_worker_main_rejects_unmeasured_skip_environment_marker(self) -> None:
        worker_main = load_worker_main()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "env"
            python = root / "bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("", encoding="utf-8")
            (root / ".comfycolab-environment.json").write_text(
                json.dumps(
                    {
                        "schema": "comfycolab-skintokens-artifacts-v1",
                        "environment_ref": "env-ref",
                        "skipped": True,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "not a measured runtime"):
                worker_main._load_environment_attestation(python)

    def test_worker_main_ignores_spoofed_environment_variable_for_revisions(self) -> None:
        worker_main = load_worker_main()
        runtime = worker_main.SkinTokensRuntime(
            mock.Mock(
                source_dir=Path("/source"),
                model_dir=Path("/model"),
                qwen_dir=Path("/qwen"),
            )
        )
        request = {
            "revisions": {
                "source": "source-ref",
                "model": "model-ref",
                "qwen": "qwen-ref",
                "environment": "measured-env-ref",
            }
        }

        with mock.patch.object(
            worker_main, "_git_revision", return_value="source-ref"
        ), mock.patch.object(
            worker_main, "_snapshot_revision", side_effect=["model-ref", "qwen-ref"]
        ), mock.patch.object(
            runtime,
            "environment_attestation",
            return_value={"environment_ref": "measured-env-ref"},
        ), mock.patch.dict(
            worker_main.os.environ,
            {"COMFYCOLAB_SKINTOKENS_ENVIRONMENT_REF": "spoofed-env-ref"},
        ):
            revisions = runtime.resolved_revisions(request)

        self.assertEqual(revisions["environment"], "measured-env-ref")

    def test_worker_main_retries_malformed_skeleton_tokens_with_a_safe_fallback(
        self,
    ) -> None:
        worker_main = load_worker_main()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_glb = root / "input.glb"
            input_glb.write_bytes(b"stable input")
            partial_output = root / ".rigged.partial.glb"
            runtime = worker_main.SkinTokensRuntime(
                mock.Mock(source_dir=root, checkpoint=root / "model.ckpt")
            )
            run_settings: list[tuple] = []

            def run_rig(*args):
                run_settings.append(args[1:6])
                if len(run_settings) == 1:
                    raise ValueError("all input arrays must have the same shape")
                Path(args[9][0]).write_bytes(b"rigged")

            request = {
                "request_id": "retry-request",
                "top_k": 5,
                "top_p": 0.95,
                "temperature": 1.0,
                "repetition_penalty": 2.0,
                "num_beams": 10,
                "use_skeleton": False,
                "use_transfer": True,
                "use_postprocess": False,
                "max_generation_attempts": 2,
                "revisions": {
                    "source": "source",
                    "model": "model",
                    "qwen": "qwen",
                    "environment": "environment",
                },
            }

            with mock.patch.object(worker_main, "_seed_generation") as seed_generation:
                generation = runtime.run_rig_with_retries(
                    mock.Mock(run_rig=run_rig),
                    input_glb=input_glb,
                    partial_output=partial_output,
                    request=request,
                )

        self.assertEqual(generation["attempt_count"], 2)
        self.assertEqual(generation["retry_count"], 1)
        self.assertEqual(generation["attempts"][0]["status"], "retryable_error")
        self.assertEqual(generation["attempts"][1]["status"], "ok")
        self.assertEqual(run_settings[0], (5, 0.95, 1.0, 2.0, 10))
        self.assertEqual(run_settings[1], (1, 1.0, 0.7, 1.1, 1))
        self.assertEqual(seed_generation.call_count, 2)
        self.assertEqual(
            seed_generation.call_args_list[1].args[0],
            (seed_generation.call_args_list[0].args[0] + 1) % (2**32),
        )

    def test_worker_main_does_not_retry_non_generation_failures(self) -> None:
        worker_main = load_worker_main()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_glb = root / "input.glb"
            input_glb.write_bytes(b"stable input")
            partial_output = root / ".rigged.partial.glb"
            runtime = worker_main.SkinTokensRuntime(
                mock.Mock(source_dir=root, checkpoint=root / "model.ckpt")
            )
            demo = mock.Mock()
            demo.run_rig.side_effect = RuntimeError("Blender server unavailable")
            request = {
                "request_id": "nonretry-request",
                "top_k": 5,
                "top_p": 0.95,
                "temperature": 1.0,
                "repetition_penalty": 2.0,
                "num_beams": 10,
                "use_skeleton": False,
                "use_transfer": True,
                "use_postprocess": False,
                "max_generation_attempts": 4,
                "revisions": {},
            }

            with mock.patch.object(worker_main, "_seed_generation"):
                with self.assertRaisesRegex(RuntimeError, "Blender server unavailable"):
                    runtime.run_rig_with_retries(
                        demo,
                        input_glb=input_glb,
                        partial_output=partial_output,
                        request=request,
                    )

        self.assertEqual(demo.run_rig.call_count, 1)

    def test_worker_main_recognizes_missing_generated_skin_tokens_as_retryable(
        self,
    ) -> None:
        worker_main = load_worker_main()
        namespace: dict = {}
        exec(
            compile(
                "def predict_step():\n    assert False\n",
                "/content/SkinTokens/src/model/tokenrig.py",
                "exec",
            ),
            namespace,
        )
        try:
            namespace["predict_step"]()
        except AssertionError as error:
            self.assertTrue(worker_main._recoverable_generation_error(error))
        else:
            self.fail("synthetic TokenRig assertion did not fail")


if __name__ == "__main__":
    unittest.main()
