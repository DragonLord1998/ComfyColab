from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "live_3d_g4_validation.py"


def load_module():
    name = "comfycolab_live_3d_g4_validation"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def options(**overrides):
    values = {
        "seed": 7,
        "sampling_steps": 0,
        "target_face_count": 0,
        "texture_size": 0,
        "max_tokens": 49152,
        "remove_background": "Auto",
        "cache_mode": "Disable cache",
        "steps": 0,
        "num_latents": 0,
        "octree_resolution": 0,
        "decode_chunk_size": 0,
        "low_vram": "Auto",
        "model": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def minimal_glb(path: Path, *, textured: bool = True) -> None:
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": 48}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6},
            {"buffer": 0, "byteOffset": 42, "byteLength": 6},
            {"buffer": 0, "byteOffset": 44, "byteLength": 4},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5121, "count": 3, "type": "VEC2", "normalized": True},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
    }
    if textured:
        primitive = document["meshes"][0]["primitives"][0]
        primitive["attributes"]["TEXCOORD_0"] = 2
        primitive["material"] = 0
        document["materials"] = [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
        document["textures"] = [{"source": 0}]
        document["images"] = [{"bufferView": 3, "mimeType": "image/png"}]
    json_chunk = json.dumps(document, separators=(",", ":")).encode()
    json_chunk += b" " * ((-len(json_chunk)) % 4)
    binary = bytes(48)
    payload = (
        struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_chunk) + 8 + len(binary))
        + struct.pack("<I4s", len(json_chunk), b"JSON")
        + json_chunk
        + struct.pack("<I4s", len(binary), b"BIN\x00")
        + binary
    )
    path.write_bytes(payload)


class Live3DG4ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_trellis_prompt_is_staged_through_preview_and_save(self) -> None:
        spec = self.module.CASES["trellis_512"]
        prompt = self.module.build_prompt(spec, options(), "input/example.png", "run-1")
        self.assertEqual(prompt["1"]["class_type"], "LoadImage")
        self.assertEqual(prompt["2"]["class_type"], "ComfyColabTrellisImageTo3D")
        self.assertEqual(prompt["2"]["inputs"]["exact_resolution"], "512")
        self.assertEqual(prompt["90"]["inputs"]["model_file"], ["2", 0])
        self.assertEqual(prompt["91"]["inputs"]["mesh"], ["2", 0])
        self.assertIn("run-1-trellis_512", prompt["91"]["inputs"]["filename_prefix"])

    def test_cli_keeps_public_token_default_and_stages_strict_1536_separately(self) -> None:
        args = self.module.parser().parse_args(
            ["run", "--case", "trellis_1536_default_cap", "--image", "/content/input.png"]
        )
        self.assertEqual(args.max_tokens, 49152)
        spec = self.module.CASES[args.case]
        prompt = self.module.build_prompt(spec, args, "input.png", "strict-run")
        self.assertEqual(prompt["2"]["inputs"]["exact_resolution"], "1536_cascade")
        self.assertEqual(prompt["2"]["inputs"]["max_tokens"], 49152)

    def test_ultrashape_prompt_uses_native_file3d_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "input.glb"
            model.write_bytes(b"placeholder")
            spec = self.module.CASES["ultrashape_512"]
            prompt = self.module.build_prompt(
                spec, options(model=model), "reference.png", "run-2"
            )
        self.assertEqual(prompt["2"]["class_type"], "ComfyColab3DPathToFile3D")
        self.assertEqual(prompt["3"]["class_type"], "ComfyColabUltraShapeRefine")
        self.assertEqual(prompt["3"]["inputs"]["model_3d"], ["2", 0])
        self.assertEqual(prompt["3"]["inputs"]["octree_resolution"], 512)
        self.assertEqual(prompt["90"]["inputs"]["model_file"], ["3", 0])

    def test_advanced_prompt_uses_hexadecimal_adapter_cache_key(self) -> None:
        spec = self.module.CASES["advanced_trellis_workflow"]
        prompt = self.module.build_prompt(spec, options(), "input.png", "advanced-run")
        cache_key = prompt["9"]["inputs"]["cache_key"]
        self.assertRegex(cache_key, r"^[0-9a-f]+$")
        self.assertEqual(prompt["9"]["inputs"]["cache_mode"], "Disable cache")

    def test_object_info_proves_native_preview_and_save_compatibility(self) -> None:
        spec = self.module.CASES["trellis_512"]
        prompt = self.module.build_prompt(spec, options(), "input.png", "run-3")
        info = {
            node["class_type"]: {"output": []}
            for node in prompt.values()
        }
        info["ComfyColabTrellisImageTo3D"]["output"] = ["FILE_3D_GLB"]
        info["Preview3D"]["input"] = {
            "required": {"model_file": ["STRING,FILE_3D_GLB,FILE_3D", {}]}
        }
        info["SaveGLB"]["input"] = {
            "required": {"mesh": ["MESH,FILE_3D_GLB,FILE_3D", {}]}
        }
        api = mock.Mock()
        api.get.return_value = info
        proof = self.module.check_object_info(api, prompt, "2")
        self.assertEqual(proof["outputType"], "FILE_3D_GLB")
        self.assertEqual(proof["previewNode"], "90")
        self.assertEqual(proof["saveNode"], "91")
        info["Preview3D"]["input"]["required"]["model_file"][0] = "STRING"
        with self.assertRaisesRegex(RuntimeError, "does not accept"):
            self.module.check_object_info(api, prompt, "2")

    def test_shape_metrics_require_actual_resolution_and_reject_downgrade(self) -> None:
        spec = self.module.CASES["trellis_1536_cascade"]
        glb = {"bytes": 123, "faces": 45}
        valid = self.module.benchmark_from(
            spec, 1.5, 2048, glb,
            "ComfyColab shape metrics: 70000 tokens at resolution 1536",
        )
        self.assertEqual(valid["actualResolution"], 1536)
        self.assertEqual(valid["tokens"], 70000)
        with self.assertRaisesRegex(RuntimeError, "silent downgrade rejected"):
            self.module.benchmark_from(
                spec, 1.5, 2048, glb,
                "ComfyColab shape metrics: 48000 tokens at resolution 1408",
            )

    def test_log_reader_waits_for_late_isolated_node_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "comfy.log"
            log.write_text("prompt complete\n", encoding="utf-8")

            def append_marker() -> None:
                time.sleep(0.05)
                with log.open("a", encoding="utf-8") as stream:
                    stream.write("[worker] ComfyColab shape metrics: 3964 tokens at resolution 512\n")

            writer = threading.Thread(target=append_marker)
            writer.start()
            text, _ = self.module.read_settled_log_since(
                log, 0, require_shape_marker=True, timeout=1.0, settle_seconds=0.1
            )
            writer.join()
        self.assertIn("3964 tokens at resolution 512", text)

    def test_ultrashape_benchmark_rejects_resolution_override(self) -> None:
        spec = self.module.CASES["ultrashape_1024_run_1"]
        with self.assertRaisesRegex(RuntimeError, "requires octree resolution 1024"):
            self.module.benchmark_from(
                spec,
                1.5,
                2048,
                {"bytes": 123, "faces": 45},
                "",
                requested_resolution=512,
            )

    def test_glb_inspection_requires_embedded_material_texture_and_uv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            textured = root / "textured.glb"
            neutral = root / "neutral.glb"
            minimal_glb(textured, textured=True)
            minimal_glb(neutral, textured=False)
            record = self.module.inspect_glb(textured, require_textured=True)
            self.assertEqual(record["faces"], 1)
            self.assertEqual(record["vertices"], 3)
            self.assertTrue(record["embeddedTextureValidated"])
            with self.assertRaisesRegex(ValueError, "UV accessor"):
                self.module.inspect_glb(neutral, require_textured=True)

    def test_detached_launch_builds_run_subcommand_and_persists_pid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state"
            args = options(
                command="launch",
                func=self.module.launch_case,
                case="trellis_512",
                state_dir=state,
                base_url="http://127.0.0.1:8188",
                comfy_root=Path("/content/ComfyUI"),
                comfy_log=Path("/content/log"),
                image=Path("/content/input.png"),
                timeout=50.0,
                vram_interval=0.5,
                cancel_after=2.0,
                cancel_start_timeout=50.0,
                trellis_python=Path("/content/python"),
                ultrashape_source=Path("/content/UltraShape"),
            )
            process = mock.Mock(pid=4321)
            with mock.patch.object(self.module.subprocess, "Popen", return_value=process) as popen:
                self.assertEqual(self.module.launch_case(args), 0)
            argv = popen.call_args.args[0]
            self.assertEqual(argv[2], "run")
            self.assertIn("--case", argv)
            self.assertEqual(argv[argv.index("--case") + 1], "trellis_512")
            current = json.loads((state / "cases/trellis_512/current.json").read_text())
            self.assertEqual(current["pid"], 4321)
            self.assertEqual(current["status"], "launching")

    def test_merge_writes_digest_evidence_without_glb_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state"
            run = {"schema": self.module.STATE_SCHEMA, "runId": "g4-test", "createdAt": "now"}
            self.module.atomic_json(state / "run.json", run)
            template = {
                "schema": "comfycolab-3d-live-validation-v1",
                "status": "pending",
                "runId": None,
                "completedAt": None,
                "gates": {
                    "trellis_512_textured_glb": {"status": "pending", "evidence": None},
                    "preview_and_save_native_file3d": {"status": "pending", "evidence": None},
                },
                "benchmarks": {
                    "trellis_512": {"status": "pending"},
                },
            }
            template_path = root / "template.json"
            output = root / "output.json"
            self.module.atomic_json(template_path, template)
            record = {
                "schema": self.module.CASE_SCHEMA,
                "status": "passed",
                "case": "trellis_512",
                "kind": "trellis",
                "gate": "trellis_512_textured_glb",
                "benchmarkName": "trellis_512",
                "runId": "g4-test",
                "evidence": "live-g4:g4-test:trellis_512:digest",
                "benchmark": {"status": "passed", "glbValidated": True},
                "previewSaveProof": {"saveArtifactValidated": True},
                "resultFiles": [{"path": "/content/secret/model.glb"}],
            }
            self.module.atomic_json(state / "cases/trellis_512/record.json", record)
            args = argparse.Namespace(state_dir=state, template=template_path, output=output)
            self.assertEqual(self.module.merge_command(args), 0)
            merged_text = output.read_text()
            merged = json.loads(merged_text)
            self.assertEqual(merged["gates"]["trellis_512_textured_glb"]["status"], "passed")
            self.assertNotIn("/content/secret/model.glb", merged_text)
            self.assertEqual(merged["status"], "pending")


if __name__ == "__main__":
    unittest.main()
