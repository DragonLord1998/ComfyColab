from __future__ import annotations

import importlib
import base64
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from tests.test_3d_node_pack import load_package


class MeshFlowWorkerTests(unittest.TestCase):
    def test_dinov3_adapter_uses_pre_final_norm_tokens(self):
        worker_main = importlib.import_module("worker.meshflow.worker_main")
        prenorm = object()
        outputs = types.SimpleNamespace(
            last_hidden_state=object(),
            hidden_states=(object(), prenorm),
        )

        self.assertIs(worker_main._dinov3_prenorm_tokens(outputs), prenorm)
        with self.assertRaisesRegex(
            RuntimeError,
            "pre-final-normalization hidden states",
        ):
            worker_main._dinov3_prenorm_tokens(
                types.SimpleNamespace(hidden_states=None)
            )

    def test_command_is_pinned_and_carries_geometry_controls(self):
        load_package()
        worker = importlib.import_module("comfycolab_3d_test.meshflow_worker")
        command = worker.MeshFlowWorkerCommand(
            python="/env/python",
            worker_script="/repo/worker.py",
            source_dir="/content/meshflow",
            checkpoint_dir="/models/meshflow",
            dinov3_model_dir="/models/dinov3",
            input_mesh="/input.glb",
            reference_images=(
                "/reference-front.png",
                "/reference-back.png",
            ),
            output_mesh="/output.glb",
            metadata_output="/output.json",
            steps=28,
            num_verts=4096,
            guidance_scale=2.5,
            seed=9,
            dtype="fp16",
            compile_models=True,
            source_ref="source-revision",
            model_ref="model-revision",
            appearance_mesh="/appearance.glb",
        )

        argv = command.argv()

        self.assertEqual(argv[:2], ["/env/python", "/repo/worker.py"])
        self.assertIn("/content/meshflow", argv)
        self.assertIn("/models/meshflow", argv)
        self.assertIn("--dinov3-model-dir", argv)
        self.assertIn("/models/dinov3", argv)
        self.assertIn("--reference-image", argv)
        self.assertIn("--appearance-mesh", argv)
        self.assertIn("/appearance.glb", argv)
        self.assertIn("/reference-front.png", argv)
        self.assertIn("/reference-back.png", argv)
        self.assertEqual(argv.count("--reference-image"), 2)
        self.assertIn("--guidance-scale", argv)
        self.assertIn("2.5", argv)
        self.assertIn("source-revision", argv)
        self.assertIn("model-revision", argv)
        self.assertIn("--compile", argv)

    def test_artifact_download_is_revision_pinned_and_reused(self):
        artifacts = importlib.import_module("worker.meshflow.artifacts")
        calls = []

        def snapshot_download(**kwargs):
            calls.append(kwargs)
            destination = Path(kwargs["local_dir"]) / "meshflow"
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "config.yaml").write_text("system: {}\n")
            (destination / "model.pth").write_bytes(b"weights")

        fake_hub = types.SimpleNamespace(snapshot_download=snapshot_download)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules, {"huggingface_hub": fake_hub}
        ):
            first = artifacts.ensure_meshflow_artifacts(directory)
            second = artifacts.ensure_meshflow_artifacts(directory)

        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["repo_id"], "facebook/meshflow")
        self.assertEqual(calls[0]["revision"], artifacts.MESHFLOW_MODEL_REF)
        self.assertEqual(
            calls[0]["allow_patterns"],
            ["meshflow/config.yaml", "meshflow/model.pth"],
        )

    def test_worker_main_passes_reference_guidance_and_truthful_metadata(self):
        worker_main = importlib.import_module("worker.meshflow.worker_main")
        pipeline_calls = []
        exported_meshes = []

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def run(self, **kwargs):
                pipeline_calls.append(kwargs)

                class ResultMesh:
                    v_nrm = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]

                    def to_trimesh(self):
                        return FakeTrimesh(
                            vertices=[
                                [0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                            ],
                            faces=[[0, 1, 2]],
                        )

                return ResultMesh()

        class FakeColorVisuals:
            def __init__(self, mesh, vertex_colors):
                self.mesh = mesh
                self.vertex_colors = vertex_colors

        class FakeTrimesh:
            def __init__(self, vertices, faces, vertex_normals=None, process=False):
                self.vertices = vertices
                self.faces = faces
                self.vertex_normals_value = vertex_normals
                self.visual = types.SimpleNamespace(
                    to_color=lambda: types.SimpleNamespace(
                        vertex_colors=[
                            [220, 120, 40, 255],
                            [220, 120, 40, 255],
                            [220, 120, 40, 255],
                        ]
                    )
                )

            @property
            def vertex_normals(self):
                if self.vertex_normals_value is not None:
                    return self.vertex_normals_value
                return [[0.0, 0.0, 1.0] for _ in self.vertices]

            def export(self, path, **_kwargs):
                exported_meshes.append(self)
                Path(path).write_bytes(b"glb")

        fake_meshflow = types.ModuleType("meshflow")
        fake_pipelines = types.ModuleType("meshflow.pipelines")
        fake_pipelines.MeshFlowPipeline = FakePipeline
        class FakeOpenedImage:
            mode = "RGB"

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def convert(self, _mode):
                return self

            def copy(self):
                return self

        fake_pil_image = types.SimpleNamespace(
            open=lambda _path: FakeOpenedImage(),
        )
        fake_pil = types.SimpleNamespace(Image=fake_pil_image)
        fake_trimesh = types.SimpleNamespace(
            Trimesh=FakeTrimesh,
            load=lambda *args, **kwargs: FakeTrimesh(
                vertices=[
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                faces=[[0, 1, 2]],
            ),
            util=types.SimpleNamespace(concatenate=lambda meshes: meshes[0]),
            visual=types.SimpleNamespace(ColorVisuals=FakeColorVisuals),
        )

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules,
            {
                "meshflow": fake_meshflow,
                "meshflow.pipelines": fake_pipelines,
                "PIL": fake_pil,
                "PIL.Image": fake_pil_image,
                "trimesh": fake_trimesh,
            },
        ), mock.patch.object(
            worker_main,
            "_install_transformers_dinov3_adapter",
        ), mock.patch.object(
            worker_main,
            "_load_geometry_points",
            return_value=[[0.0, 0.0, 0.0]] * 4096,
        ), mock.patch.object(
            worker_main,
            "_sanitize_candidate_mesh",
            side_effect=lambda mesh: (
                mesh,
                {
                    "cleanup_applied": False,
                    "raw_component_count": 1,
                    "dominant_area_fraction": 1.0,
                    "dominant_face_fraction": 1.0,
                },
            ),
        ), mock.patch.object(
            worker_main,
            "_export_host_glb",
            side_effect=lambda mesh, path: mesh.export(path),
        ), mock.patch.object(
            worker_main,
            "_candidate_metrics",
            side_effect=[
                {
                    "vertices": 3,
                    "faces": 1,
                    "face_vertex_ratio": 1 / 3,
                    "connected_components": 1,
                    "unique_edges": 3,
                    "boundary_edges": 3,
                    "boundary_edge_ratio": 1.0,
                    "nonmanifold_edges": 0,
                    "nonmanifold_edge_ratio": 0.0,
                    "duplicate_faces": 0,
                    "degenerate_faces": 0,
                    "symmetric_chamfer_mse": 0.2,
                    "accepted": True,
                    "selection_score": 0.2,
                },
                {
                    "vertices": 3,
                    "faces": 1,
                    "face_vertex_ratio": 1 / 3,
                    "connected_components": 1,
                    "unique_edges": 3,
                    "boundary_edges": 3,
                    "boundary_edge_ratio": 1.0,
                    "nonmanifold_edges": 0,
                    "nonmanifold_edge_ratio": 0.0,
                    "duplicate_faces": 0,
                    "degenerate_faces": 0,
                    "symmetric_chamfer_mse": 0.1,
                    "accepted": True,
                    "selection_score": 0.1,
                },
            ],
        ) as candidate_metrics:
            root = Path(directory)
            source_dir = root / "source"
            checkpoint_dir = root / "checkpoint"
            source_dir.mkdir()
            checkpoint_dir.mkdir()
            (checkpoint_dir / "config.yaml").write_text(
                "denoiser_model:\n  use_proj_cond_on_temb: false\n",
                encoding="utf-8",
            )
            (checkpoint_dir / "model.pth").write_bytes(b"weights")
            input_mesh = root / "input.glb"
            input_mesh.write_bytes(b"input")
            reference = root / "reference.png"
            reference.write_bytes(
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwC"
                    "AAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
                )
            )
            reference_back = root / "reference-back.png"
            reference_back.write_bytes(reference.read_bytes())
            dinov3 = root / "dinov3"
            dinov3.mkdir()
            (dinov3 / "config.json").write_text("{}\n", encoding="utf-8")
            (dinov3 / "model.safetensors").write_bytes(b"weights")
            output = root / "output.glb"
            metadata = root / "metadata.json"

            result = worker_main.run(
                types.SimpleNamespace(
                    source_dir=str(source_dir),
                    checkpoint_dir=str(checkpoint_dir),
                    dinov3_model_dir=str(dinov3),
                    input_mesh=str(input_mesh),
                    appearance_mesh=str(input_mesh),
                    reference_image=[str(reference), str(reference_back)],
                    output_mesh=str(output),
                    metadata_output=str(metadata),
                    steps=28,
                    num_verts=4096,
                    guidance_scale=2.75,
                    seed=19,
                    dtype="fp16",
                    compile=False,
                    source_ref="source-ref",
                    model_ref="model-ref",
                )
            )
            candidate_metrics.side_effect = [
                {
                    "vertices": 3,
                    "faces": 1,
                    "face_vertex_ratio": 1 / 3,
                    "connected_components": 1,
                    "unique_edges": 3,
                    "boundary_edges": 3,
                    "boundary_edge_ratio": 1.0,
                    "nonmanifold_edges": 0,
                    "nonmanifold_edge_ratio": 0.0,
                    "duplicate_faces": 0,
                    "degenerate_faces": 0,
                    "symmetric_chamfer_mse": 0.15,
                    "accepted": True,
                    "selection_score": 0.15,
                }
            ]
            geometry_result = worker_main.run(
                types.SimpleNamespace(
                    source_dir=str(source_dir),
                    checkpoint_dir=str(checkpoint_dir),
                    dinov3_model_dir="",
                    input_mesh=str(input_mesh),
                    appearance_mesh=str(input_mesh),
                    reference_image=[],
                    output_mesh=str(root / "geometry-output.glb"),
                    metadata_output=str(root / "geometry-metadata.json"),
                    steps=28,
                    num_verts=4096,
                    guidance_scale=2.75,
                    seed=19,
                    dtype="fp16",
                    compile=False,
                    source_ref="source-ref",
                    model_ref="model-ref",
                )
            )

        self.assertEqual(len(pipeline_calls), 3)
        self.assertEqual(
            [call["seed"] for call in pipeline_calls],
            [19, 20, 19],
        )
        self.assertEqual(pipeline_calls[0]["image"].mode, "RGB")
        self.assertEqual(pipeline_calls[0]["guidance_scale"], 2.75)
        self.assertEqual(result["actual_vertices"], 3)
        self.assertEqual(result["actual_faces"], 1)
        self.assertFalse(result["num_verts_control_supported"])
        self.assertTrue(result["num_verts_warning"])
        self.assertTrue(result["image_conditioned"])
        self.assertEqual(result["conditioning_view_count"], 2)
        self.assertEqual(
            result["conditioning_fusion"],
            "separate_candidates_best_valid",
        )
        self.assertEqual(result["selected_view"], "back")
        self.assertEqual(
            result["candidate_selection"]["policy"],
            "topology_gates_then_symmetric_chamfer_v1",
        )
        for candidate in result["candidate_selection"]["candidates"]:
            self.assertNotIn("output_mesh", candidate)
            self.assertIn("candidate_filename", candidate)
            self.assertFalse(candidate["candidate_artifact_persisted"])
            self.assertIn("cleanup", candidate)
        self.assertEqual(result["appearance_method"], "nearest_source_vertex_color")
        self.assertTrue(exported_meshes[0].visual.vertex_colors is not None)
        self.assertEqual(result["guidance_scale"], 2.75)
        self.assertFalse(result["geometry_only"])
        self.assertTrue(geometry_result["geometry_only"])
        self.assertFalse(geometry_result["image_conditioned"])
        self.assertIsNone(pipeline_calls[2]["image"])
