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
        "camera_fov_degrees": 0.0,
        "keep_worker_loaded": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def minimal_glb(
    path: Path,
    *,
    textured: bool = True,
    texture_source_extension: str | None = None,
) -> None:
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
        if texture_source_extension:
            document["textures"] = [
                {"extensions": {texture_source_extension: {"source": 0}}}
            ]
            document["extensionsUsed"] = [texture_source_extension]
        else:
            document["textures"] = [{"source": 0}]
        document["images"] = [
            {
                "bufferView": 3,
                "mimeType": (
                    "image/webp"
                    if texture_source_extension == "EXT_texture_webp"
                    else "image/png"
                ),
            }
        ]
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


def minimal_rigged_glb(path: Path, *, weight_sum: float = 1.0) -> None:
    binary = bytearray()
    views: list[dict[str, int]] = []

    def append_view(payload: bytes) -> int:
        offset = len(binary)
        binary.extend(payload)
        views.append(
            {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": len(payload),
            }
        )
        binary.extend(b"\x00" * ((-len(binary)) % 4))
        return len(views) - 1

    position_view = append_view(
        struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    )
    index_view = append_view(struct.pack("<3H", 0, 1, 2))
    uv_view = append_view(struct.pack("<6f", 0, 0, 1, 0, 0, 1))
    joints_view = append_view(struct.pack("<12H", *([0, 0, 0, 0] * 3)))
    weights_view = append_view(
        struct.pack("<12f", *([weight_sum, 0.0, 0.0, 0.0] * 3))
    )
    identity = (
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    inverse_bind_view = append_view(struct.pack("<32f", *(identity * 2)))
    image_view = append_view(b"fake-png")
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": [
            {
                "bufferView": position_view,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
            },
            {
                "bufferView": index_view,
                "componentType": 5123,
                "count": 3,
                "type": "SCALAR",
            },
            {
                "bufferView": uv_view,
                "componentType": 5126,
                "count": 3,
                "type": "VEC2",
            },
            {
                "bufferView": joints_view,
                "componentType": 5123,
                "count": 3,
                "type": "VEC4",
            },
            {
                "bufferView": weights_view,
                "componentType": 5126,
                "count": 3,
                "type": "VEC4",
            },
            {
                "bufferView": inverse_bind_view,
                "componentType": 5126,
                "count": 2,
                "type": "MAT4",
            },
        ],
        "materials": [
            {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
        ],
        "textures": [{"source": 0}],
        "images": [{"bufferView": image_view, "mimeType": "image/png"}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "TEXCOORD_0": 2,
                            "JOINTS_0": 3,
                            "WEIGHTS_0": 4,
                        },
                        "indices": 1,
                        "material": 0,
                    }
                ]
            }
        ],
        "nodes": [
            {"mesh": 0, "skin": 0},
            {"name": "root", "children": [2]},
            {"name": "tip"},
        ],
        "skins": [
            {
                "joints": [1, 2],
                "skeleton": 1,
                "inverseBindMatrices": 5,
            }
        ],
        "scenes": [{"nodes": [0, 1]}],
        "scene": 0,
    }
    json_chunk = json.dumps(document, separators=(",", ":")).encode()
    json_chunk += b" " * ((-len(json_chunk)) % 4)
    payload = (
        struct.pack(
            "<4sII",
            b"glTF",
            2,
            12 + 8 + len(json_chunk) + 8 + len(binary),
        )
        + struct.pack("<I4s", len(json_chunk), b"JSON")
        + json_chunk
        + struct.pack("<I4s", len(binary), b"BIN\x00")
        + binary
    )
    path.write_bytes(payload)


def minimal_3dgs_ply(path: Path, *, vertices: int = 2) -> None:
    properties = [
        ("float", "x"),
        ("float", "y"),
        ("float", "z"),
        ("float", "f_dc_0"),
        ("float", "f_dc_1"),
        ("float", "f_dc_2"),
        ("float", "opacity"),
        ("float", "scale_0"),
        ("float", "scale_1"),
        ("float", "scale_2"),
        ("float", "rot_0"),
        ("float", "rot_1"),
        ("float", "rot_2"),
        ("float", "rot_3"),
    ]
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {vertices}",
        *[f"property {kind} {name}" for kind, name in properties],
        "end_header",
    ]
    row = struct.pack(
        "<" + "f" * len(properties),
        0.0, 1.0, 2.0,
        0.1, 0.2, 0.3,
        0.9,
        -3.0, -3.0, -3.0,
        1.0, 0.0, 0.0, 0.0,
    )
    path.write_bytes(("\n".join(header) + "\n").encode("ascii") + row * vertices)


def stage_binary(node_id: str, text: str) -> bytes:
    encoded_node = node_id.encode("utf-8")
    return (
        struct.pack(">II", 3, len(encoded_node))
        + encoded_node
        + text.encode("utf-8")
    )


def completed_stage_verifier(module, *, prompt_id: str = "prompt-1"):
    verifier = module.FiveStageVerifier("2")
    verifier.prompt_id = prompt_id
    for index, text in enumerate(module.FIVE_STAGE_TEXTS):
        verifier.record_binary(stage_binary("2", text))
        if index > 0:
            verifier.record_json(
                {
                    "type": "progress_state",
                    "data": {
                        "prompt_id": prompt_id,
                        "nodes": {
                            "2": {
                                "value": index,
                                "max": 5,
                            }
                        },
                    },
                }
            )
        if index == 2:
            verifier.record_json(
                {
                    "type": "executed",
                    "data": {
                        "prompt_id": prompt_id,
                        "node": "2:early-preview",
                        "display_node": "90",
                        "output": {"result": ["preview3d_early.glb", None, None]},
                    },
                }
            )
    verifier.record_json(
        {
            "type": "executed",
            "data": {
                "prompt_id": prompt_id,
                "node": "90",
                "display_node": "90",
                "output": {"result": ["preview3d_final.glb", None, None]},
            },
        }
    )
    return verifier


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

    def test_five_stage_verifier_requires_ordered_stages_progress_and_two_previews(self) -> None:
        verifier = completed_stage_verifier(self.module)
        proof = verifier.verify()
        self.assertEqual(proof["status"], "passed")
        self.assertTrue(all(proof["checks"].values()))
        self.assertEqual(
            [event["text"] for event in proof["stageEvents"]],
            list(self.module.FIVE_STAGE_TEXTS),
        )
        self.assertEqual(
            [event["value"] for event in proof["progressEvents"]],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(len(proof["previewEvents"]), 2)

    def test_five_stage_verifier_rejects_missing_final_preview(self) -> None:
        verifier = completed_stage_verifier(self.module)
        verifier.preview_events.pop()
        with self.assertRaisesRegex(RuntimeError, "finalPreview"):
            verifier.verify()

    def test_history_output_paths_and_glb_classifier_separate_preview_from_save(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            neutral = output / "neutral.glb"
            textured = output / "saved.glb"
            minimal_glb(neutral, textured=False)
            minimal_glb(textured, textured=True)
            history = {
                "outputs": {
                    "91": {
                        "3d": [
                            {"filename": "saved.glb", "subfolder": "", "type": "output"},
                            {"filename": "ignored.glb", "subfolder": "", "type": "temp"},
                        ]
                    }
                }
            }
            self.assertEqual(
                self.module.history_output_paths(history, "91", output),
                [textured.resolve()],
            )
            self.assertEqual(self.module.classify_glb(neutral)["artifactKind"], "geometry")
            self.assertEqual(self.module.classify_glb(textured)["artifactKind"], "textured")
            event = {"glbs": ["neutral.glb"]}
            self.assertEqual(
                self.module.preview_event_paths(event, output),
                [neutral.resolve()],
            )
            with self.assertRaisesRegex(RuntimeError, "escaped"):
                self.module.preview_event_paths({"glbs": ["../outside.glb"]}, output)

    def test_inspect_glb_accepts_embedded_webp_extension_texture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "webp.glb"
            minimal_glb(
                path,
                texture_source_extension="EXT_texture_webp",
            )
            inspection = self.module.inspect_glb(path, require_textured=True)

        self.assertTrue(inspection["embeddedTextureValidated"])
        self.assertEqual(inspection["textureCount"], 1)

    def test_fresh_trellis_run_wires_five_stage_and_explicit_save_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            neutral = output / "preview3d_early.glb"
            final_preview = output / "preview3d_final.glb"
            textured = output / "saved.glb"
            minimal_glb(neutral, textured=False)
            minimal_glb(final_preview, textured=True)
            minimal_glb(textured, textured=True)
            verifier = completed_stage_verifier(self.module)
            history = {
                "status": {"completed": True},
                "outputs": {
                    "91": {
                        "3d": [
                            {"filename": "saved.glb", "subfolder": "", "type": "output"}
                        ]
                    }
                },
            }
            args = options(
                base_url="http://127.0.0.1:8188",
                comfy_root=root,
                comfy_log=root / "comfy.log",
                timeout=60.0,
                vram_interval=0.01,
            )
            recorder = mock.Mock()
            marker = "ComfyColab shape metrics: 3964 tokens at resolution 512"
            with mock.patch.object(
                self.module,
                "check_object_info",
                return_value={"previewNode": "90", "saveNode": "91"},
            ), mock.patch.object(
                self.module,
                "queue_and_capture_five_stage_events",
                return_value=("prompt-1", verifier),
            ) as capture, mock.patch.object(
                self.module,
                "wait_prompt",
                return_value=history,
            ), mock.patch.object(
                self.module,
                "changed_glbs",
                return_value=[neutral, final_preview, textured],
            ), mock.patch.object(
                self.module,
                "read_settled_log_since",
                return_value=(marker, 0),
            ), mock.patch.object(self.module.VramSampler, "sample", return_value=1):
                result = self.module.run_prompt_once(
                    self.module.CASES["trellis_512"],
                    args,
                    "run-1",
                    "input.png",
                    recorder,
                )

            capture.assert_called_once()
            self.assertEqual(result["glb"]["path"], str(textured.resolve()))
            stage_proof = result["previewSaveProof"]["fiveStageProof"]
            self.assertEqual(stage_proof["status"], "passed")
            self.assertTrue(stage_proof["checks"]["explicitSaveGLBArtifactValidated"])
            self.assertEqual(stage_proof["artifacts"]["geometryPreviewCount"], 1)
            self.assertEqual(stage_proof["artifacts"]["finalPreviewCount"], 1)

    def test_cli_keeps_public_token_default_and_stages_strict_1536_separately(self) -> None:
        args = self.module.parser().parse_args(
            ["run", "--case", "trellis_1536_default_cap", "--image", "/content/input.png"]
        )
        self.assertEqual(args.max_tokens, 49152)
        spec = self.module.CASES[args.case]
        self.assertEqual(spec.gate, "trellis_1536_default_cap_no_downgrade")
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
        self.assertEqual(prompt["3"]["inputs"]["detail"], "Conservative")
        self.assertEqual(
            prompt["3"]["inputs"]["octree_resolution"],
            0,
            "the live 512 gate must exercise the preset/default path",
        )
        self.assertEqual(prompt["90"]["inputs"]["model_file"], ["3", 0])

    def test_triposplat_fast_prompt_uses_file3d_ply_preview_and_save(self) -> None:
        spec = self.module.CASES["triposplat_fast_65k"]
        prompt = self.module.build_prompt(spec, options(seed=123), "input.png", "splat-run")
        self.assertEqual(prompt["1"]["class_type"], "LoadImage")
        self.assertEqual(prompt["2"]["class_type"], "ComfyColabTripoSplatImageToGaussianSplat")
        self.assertEqual(prompt["2"]["inputs"]["image"], ["1", 0])
        self.assertEqual(prompt["2"]["inputs"]["quality"], "Fast — 65K")
        self.assertEqual(prompt["2"]["inputs"]["seed"], 123)
        self.assertEqual(prompt["2"]["inputs"]["remove_background"], True)
        self.assertEqual(prompt["2"]["inputs"]["enable_sampling_preview"], True)
        self.assertEqual(prompt["2"]["inputs"]["output_format"], "ply")
        self.assertEqual(prompt["90"]["inputs"]["model_file"], ["2", 1])
        self.assertEqual(prompt["91"]["class_type"], "SaveGLB")
        self.assertEqual(prompt["91"]["inputs"]["mesh"], ["2", 1])
        self.assertIn("splat-run-triposplat_fast_65k", prompt["91"]["inputs"]["filename_prefix"])

    def test_pixal3d_prompt_is_single_image_facade_with_preview_and_save(self) -> None:
        spec = self.module.CASES["pixal3d_cold_1024"]
        prompt = self.module.build_prompt(spec, options(), "input.png", "pixal-run")
        self.assertEqual(prompt["1"]["class_type"], "LoadImage")
        self.assertEqual(prompt["2"]["class_type"], "ComfyColabPixal3DImageTo3D")
        self.assertEqual(prompt["2"]["inputs"]["image"], ["1", 0])
        self.assertEqual(prompt["2"]["inputs"]["quality"], "1024 — Stable")
        self.assertEqual(prompt["2"]["inputs"]["camera_fov_degrees"], 0.0)
        self.assertEqual(prompt["2"]["inputs"]["keep_worker_loaded"], True)
        self.assertNotIn("mode", prompt["2"]["inputs"])
        self.assertNotIn("num_views", prompt["2"]["inputs"])
        self.assertEqual(prompt["90"]["inputs"]["model_file"], ["2", 0])
        self.assertEqual(prompt["91"]["inputs"]["mesh"], ["2", 0])
        self.assertIn("pixal-run-pixal3d_cold_1024", prompt["91"]["inputs"]["filename_prefix"])

    def test_skintokens_prompt_uses_file3d_adapter_without_an_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "input.glb"
            model.write_bytes(b"placeholder")
            spec = self.module.CASES["skintokens_auto_rig"]
            prompt = self.module.build_prompt(
                spec,
                options(model=model, keep_worker_loaded=False),
                "",
                "skin-run",
            )

        self.assertFalse(self.module.required_image(spec))
        self.assertEqual(prompt["1"]["class_type"], "ComfyColab3DPathToFile3D")
        self.assertEqual(prompt["2"]["class_type"], "ComfyColabSkinTokensAutoRig")
        self.assertEqual(prompt["2"]["inputs"]["model_3d"], ["1", 0])
        self.assertTrue(prompt["2"]["inputs"]["preserve_texture"])
        self.assertFalse(prompt["2"]["inputs"]["use_postprocess"])
        self.assertFalse(prompt["2"]["inputs"]["keep_worker_loaded"])
        self.assertEqual(prompt["90"]["inputs"]["model_file"], ["2", 0])
        self.assertEqual(prompt["91"]["inputs"]["mesh"], ["2", 0])
        self.assertNotIn("LoadImage", {node["class_type"] for node in prompt.values()})

    def test_skintokens_inspection_reads_real_joint_and_weight_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rigged = root / "rigged.glb"
            minimal_rigged_glb(rigged)
            contract = self.module.inspect_skinning(rigged)
            self.assertEqual(contract["skins"], 1)
            self.assertEqual(contract["joints"], 2)
            self.assertEqual(contract["skinnedPrimitives"], 1)
            self.assertEqual(contract["weightedVertices"], 3)
            self.assertEqual(contract["minimumWeightSum"], 1.0)
            self.assertEqual(contract["maximumWeightSum"], 1.0)
            self.assertEqual(contract["inverseBindMatrices"], 1)

            invalid = root / "invalid-weights.glb"
            minimal_rigged_glb(invalid, weight_sum=0.5)
            with self.assertRaisesRegex(ValueError, "not normalized"):
                self.module.inspect_skinning(invalid)

    def test_skintokens_run_records_skin_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            model = root / "input.glb"
            model.write_bytes(b"placeholder")
            rigged = output / "rigged.glb"
            minimal_rigged_glb(rigged)
            history = {
                "status": {"completed": True},
                "outputs": {
                    "91": {
                        "glbs": [
                            {
                                "filename": rigged.name,
                                "subfolder": "",
                                "type": "output",
                            }
                        ]
                    }
                },
            }
            worker_result = {
                "schema": "comfycolab-skintokens-worker-result-v1",
                "request_id": "skin-request",
                "revisions": {
                    "source": "source-ref",
                    "model": "model-ref",
                    "qwen": "qwen-ref",
                    "environment": "environment-ref",
                },
                "environment": {
                    "environment_ref": "environment-ref",
                    "versions": {
                        "python": "3.11.15",
                        "bpy": "4.2.22",
                    },
                },
                "environment_versions": {
                    "python": "3.11.15",
                    "bpy": "4.2.22",
                },
                "generation": {
                    "attempt_count": 1,
                    "retry_count": 0,
                    "base_seed": 7,
                    "selected_seed": 7,
                    "selected_settings": {"top_k": 5},
                    "attempts": [
                        {
                            "attempt": 1,
                            "seed": 7,
                            "settings": {"top_k": 5},
                            "status": "ok",
                        }
                    ],
                },
                "rig_contract": {"skins": 1, "joints": 1},
                "runtime_seconds": 1.0,
                "model_load_count": 1,
                "bytes": rigged.stat().st_size,
                "sha256": self.module.sha256_file(rigged),
            }
            args = options(
                base_url="http://127.0.0.1:8188",
                comfy_root=root,
                comfy_log=root / "comfy.log",
                timeout=60.0,
                vram_interval=0.01,
                model=model,
            )
            recorder = mock.Mock()
            spec = self.module.CASES["skintokens_auto_rig"]
            with mock.patch.object(
                self.module,
                "check_object_info",
                return_value={"previewNode": "90", "saveNode": "91"},
            ), mock.patch.object(
                self.module,
                "queue_prompt",
                return_value="prompt-skin",
            ), mock.patch.object(
                self.module,
                "wait_prompt",
                return_value=history,
            ), mock.patch.object(
                self.module,
                "changed_glbs",
                return_value=[rigged.resolve()],
            ), mock.patch.object(
                self.module,
                "read_settled_log_since",
                return_value=(
                    "COMFYCOLAB_SKINTOKENS_RESULT="
                    + json.dumps(worker_result, sort_keys=True),
                    0,
                ),
            ), mock.patch.object(self.module.VramSampler, "sample", return_value=4096):
                result = self.module.run_prompt_once(
                    spec,
                    args,
                    "skin-run",
                    "",
                    recorder,
                )

        self.assertEqual(result["glb"]["path"], str(rigged.resolve()))
        self.assertEqual(result["glb"]["skinContract"]["skins"], 1)
        self.assertEqual(result["glb"]["skinContract"]["weightedVertices"], 3)
        self.assertTrue(result["previewSaveProof"]["saveArtifactValidated"])
        self.assertEqual(
            result["workerSkinTokensResult"]["revisions"]["environment"],
            "environment-ref",
        )

    def test_skintokens_requires_saveglb_history_and_worker_provenance(self) -> None:
        marker = {
            "revisions": {
                "source": "source",
                "model": "model",
                "qwen": "qwen",
                "environment": "environment",
            },
            "environment": {
                "environment_ref": "environment",
                "versions": {"python": "3.11.15"},
            },
        }
        line = "COMFYCOLAB_SKINTOKENS_RESULT=" + json.dumps(marker)
        self.assertEqual(self.module.skintokens_worker_result_events(line), [marker])
        compacted = self.module.compact_log_evidence(line + "\n" + ("verbose\n" * 5000))
        self.assertIn(line, compacted)

    def test_pixal3d_benchmark_requires_machine_resolution_and_tokens(self) -> None:
        spec = self.module.CASES["pixal3d_cold_1024"]
        glb = {"bytes": 123, "faces": 45}
        worker = {
            "actual_resolution": 1024,
            "token_count": 42000,
            "peak_vram_bytes": 1024,
            "pipeline_load_count": 1,
            "worker_pid": 4321,
        }
        benchmark = self.module.benchmark_from(
            spec,
            1.5,
            2048,
            glb,
            "",
            pixal3d_worker_result=worker,
        )
        self.assertEqual(benchmark["actualResolution"], 1024)
        self.assertEqual(benchmark["tokens"], 42000)
        self.assertEqual(benchmark["workerPid"], 4321)
        with self.assertRaisesRegex(RuntimeError, "silent downgrade rejected"):
            self.module.benchmark_from(
                spec,
                1.5,
                2048,
                glb,
                "",
                pixal3d_worker_result={**worker, "actual_resolution": 896},
            )

    def test_pixal3d_reuse_case_requires_same_worker_and_single_pipeline_load(self) -> None:
        spec = self.module.CASES["pixal3d_worker_reuse_1024"]
        first = {
            "promptId": "first",
            "workerPixal3DResult": {"worker_pid": 321, "pipeline_load_count": 1},
        }
        second = {
            "promptId": "second",
            "workerPixal3DResult": {"worker_pid": 321, "pipeline_load_count": 1},
        }
        with mock.patch.object(
            self.module, "run_prompt_once", side_effect=[first, second]
        ) as run:
            result = self.module.run_pixal3d_reuse_case(
                spec, options(seed=7), "run", "input.png", mock.Mock()
            )
        self.assertTrue(result["reuseProof"]["workerReused"])
        self.assertEqual(result["reuseProof"]["secondSeed"], 8)
        self.assertEqual(run.call_args_list[1].args[1].seed, 8)

    def test_pixal3d_cancellation_matches_actual_worker_command(self) -> None:
        spec = self.module.CASES["pixal3d_cancellation_cleanup"]
        self.assertEqual(
            self.module.worker_pattern_for(spec),
            "worker/pixal3d/worker_main.py",
        )

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

    def test_triposplat_object_info_proves_file3d_preview_and_save_compatibility(self) -> None:
        spec = self.module.CASES["triposplat_fast_65k"]
        prompt = self.module.build_prompt(spec, options(), "input.png", "splat-run")
        info = {node["class_type"]: {"output": []} for node in prompt.values()}
        info["ComfyColabTripoSplatImageToGaussianSplat"]["output"] = [
            "SPLAT",
            "FILE_3D_SPLAT_ANY",
        ]
        info["Preview3D"]["input"] = {
            "required": {"model_file": ["STRING,FILE_3D_SPLAT_ANY,FILE_3D", {}]}
        }
        info["SaveGLB"]["input"] = {
            "required": {"mesh": ["MESH,FILE_3D_SPLAT_ANY,FILE_3D", {}]}
        }
        api = mock.Mock()
        api.get.return_value = info
        proof = self.module.check_object_info(api, prompt, "2", spec)
        self.assertEqual(proof["outputType"], "FILE_3D_SPLAT_ANY")
        self.assertEqual(proof["saveNodeType"], "SaveGLB")
        self.assertEqual(proof["saveInput"], "mesh")
        info["ComfyColabTripoSplatImageToGaussianSplat"]["output"] = ["SPLAT"]
        with self.assertRaisesRegex(RuntimeError, "did not expose FILE_3D_SPLAT_ANY"):
            self.module.check_object_info(api, prompt, "2", spec)

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

    def test_log_compaction_preserves_shape_marker_before_verbose_tail(self) -> None:
        marker = "ComfyColab shape metrics: 3964 tokens at resolution 512"
        compacted = self.module.compact_log_evidence(marker + "\n" + ("mesh-log\n" * 5000))
        self.assertIn(marker, compacted)
        self.assertLess(len(compacted), 12_100)

    def test_ultrashape_benchmark_rejects_resolution_override(self) -> None:
        spec = self.module.CASES["ultrashape_1024_run_1"]
        with self.assertRaisesRegex(RuntimeError, "lacks machine-observed"):
            self.module.benchmark_from(
                spec,
                1.5,
                2048,
                {"bytes": 123, "faces": 45},
                "",
            )
        with self.assertRaisesRegex(RuntimeError, "requires octree resolution 1024"):
            self.module.benchmark_from(
                spec,
                1.5,
                2048,
                {"bytes": 123, "faces": 45},
                "",
                observed_resolution=512,
            )

    def test_ultrashape_machine_settings_are_parsed_and_retained_in_log_evidence(self) -> None:
        resolved = {
            "detail": "Conservative",
            "steps": 24,
            "num_latents": 16384,
            "octree_resolution": 512,
            "decode_chunk_size": 4096,
            "seed": 9,
        }
        worker = {**resolved, "low_vram": "auto"}
        resolved_line = "COMFYCOLAB_ULTRASHAPE_SETTINGS=" + json.dumps(resolved)
        worker_line = "COMFYCOLAB_ULTRASHAPE_WORKER_SETTINGS=" + json.dumps(worker)
        text = resolved_line + "\n" + worker_line + "\n" + ("verbose\n" * 5000)
        self.assertEqual(
            self.module.ultrashape_resolved_settings_events(text),
            [resolved],
        )
        self.assertEqual(self.module.ultrashape_worker_settings_events(text), [worker])
        compacted = self.module.compact_log_evidence(text)
        self.assertIn(resolved_line, compacted)
        self.assertIn(worker_line, compacted)

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

    def test_3dgs_ply_inspection_requires_binary_little_endian_and_gaussian_properties(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ply = root / "splat.ply"
            minimal_3dgs_ply(ply, vertices=3)
            record = self.module.inspect_3dgs_ply(ply)
            self.assertEqual(record["artifactKind"], "3dgs-ply")
            self.assertEqual(record["gaussianCount"], 3)
            self.assertEqual(record["format"], "binary_little_endian")
            self.assertTrue(record["plyValidated"])
            self.assertIn("f_dc_0", record["properties"])
            bad = root / "bad.ply"
            bad.write_bytes(ply.read_bytes().replace(b"property float opacity\n", b""))
            with self.assertRaisesRegex(ValueError, "required 3DGS properties"):
                self.module.inspect_3dgs_ply(bad)

    def test_triposplat_run_wires_file3d_save_and_records_ply_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            splat = output / "saved.ply"
            minimal_3dgs_ply(splat, vertices=4)
            history = {
                "status": {"completed": True},
                "outputs": {
                    "91": {
                        "3d": [
                            {"filename": "saved.ply", "subfolder": "", "type": "output"}
                        ]
                    }
                },
            }
            args = options(
                base_url="http://127.0.0.1:8188",
                comfy_root=root,
                comfy_log=root / "comfy.log",
                timeout=60.0,
                vram_interval=0.01,
            )
            recorder = mock.Mock()
            spec = self.module.CASES["triposplat_fast_65k"]
            with mock.patch.object(
                self.module,
                "check_object_info",
                return_value={"previewNode": "90", "saveNode": "91"},
            ) as object_info, mock.patch.object(
                self.module,
                "queue_prompt",
                return_value="prompt-splat",
            ), mock.patch.object(
                self.module,
                "wait_prompt",
                return_value=history,
            ), mock.patch.object(
                self.module,
                "changed_artifacts",
                return_value=[splat.resolve()],
            ), mock.patch.object(
                self.module,
                "read_settled_log_since",
                return_value=("", 0),
            ), mock.patch.object(self.module.VramSampler, "sample", return_value=4096):
                result = self.module.run_prompt_once(
                    spec,
                    args,
                    "run-splat",
                    "input.png",
                    recorder,
                )
            object_info.assert_called_once()
            self.assertEqual(result["glb"]["path"], str(splat.resolve()))
            self.assertEqual(result["glb"]["gaussianCount"], 4)
            self.assertTrue(result["previewSaveProof"]["saveArtifactValidated"])
            benchmark = self.module.benchmark_from(
                spec,
                1.25,
                result["peakVramBytes"],
                result["glb"],
                "",
            )
            self.assertEqual(benchmark["gaussianCount"], 4)
            self.assertEqual(benchmark["outputFormat"], "ply")
            self.assertEqual(benchmark["plySha256"], result["glb"]["sha256"])

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
