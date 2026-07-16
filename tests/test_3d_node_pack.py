from __future__ import annotations

import asyncio
import importlib
import importlib.util
import io as stdio
import json
import math
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "custom_nodes" / "ComfyColab-3D"


def load_package():
    name = "comfycolab_3d_test"
    for module in list(sys.modules):
        if module == name or module.startswith(name + "."):
            del sys.modules[module]
    spec = importlib.util.spec_from_file_location(
        name, PACKAGE_DIR / "__init__.py", submodule_search_locations=[str(PACKAGE_DIR)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    assert spec.loader
    spec.loader.exec_module(package)
    return package


def write_glb(
    path: Path,
    *,
    material: bool = True,
    textured: bool = False,
    uv_count: int | None = None,
    uv_values=None,
    empty_primitives: bool = False,
    invalid_image_view: bool = False,
    volumetric: bool = False,
):
    if volumetric:
        position_values = (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1)
        index_values = (0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3)
    else:
        position_values = (0, 0, 0, 1, 0, 0, 0, 1, 0)
        index_values = (0, 1, 2)
    positions = struct.pack(f"<{len(position_values)}f", *position_values)
    raw_indices = struct.pack(f"<{len(index_values)}H", *index_values)
    indices = raw_indices + b"\x00" * ((4 - len(raw_indices) % 4) % 4)
    binary = positions + indices
    buffer_views = [
        {"buffer": 0, "byteOffset": 0, "byteLength": len(positions)},
        {"buffer": 0, "byteOffset": len(positions), "byteLength": len(raw_indices)},
    ]
    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": len(position_values) // 3, "type": "VEC3"},
        {"bufferView": 1, "componentType": 5123, "count": len(index_values), "type": "SCALAR"},
    ]
    attributes = {"POSITION": 0}
    primitive = {"attributes": attributes, "indices": 1}
    if textured:
        if uv_values is None:
            uv_values = (0, 0, 1, 0, 0, 1, 1, 1) if volumetric else (0, 0, 1, 0, 0, 1)
        if uv_count is None:
            uv_count = len(position_values) // 3
        uvs = struct.pack(f"<{len(uv_values)}f", *uv_values)
        uv_offset = len(binary)
        binary += uvs
        buffer_views.append({"buffer": 0, "byteOffset": uv_offset, "byteLength": len(uvs)})
        accessors.append(
            {"bufferView": 2, "componentType": 5126, "count": uv_count, "type": "VEC2"}
        )
        attributes["TEXCOORD_0"] = 2
        image_offset = len(binary)
        image_payload = b"fake-png"
        binary += image_payload
        binary += b"\x00" * ((4 - len(binary) % 4) % 4)
        buffer_views.append(
            {"buffer": 0, "byteOffset": image_offset, "byteLength": len(image_payload)}
        )
        if invalid_image_view:
            buffer_views[-1]["byteOffset"] = len(binary) + 1024
    if material or textured:
        primitive["material"] = 0
    document = {
        "asset": {"version": "2.0"},
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(binary)}],
        "meshes": [{"primitives": [] if empty_primitives else [primitive]}],
    }
    if material or textured:
        document["materials"] = [{"pbrMetallicRoughness": {}}]
    if textured:
        document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
        document["textures"] = [{"source": 0}]
        document["images"] = [{"bufferView": 3, "mimeType": "image/png"}]
    chunk = json.dumps(document, separators=(",", ":")).encode()
    chunk += b" " * ((4 - len(chunk) % 4) % 4)
    body = (
        struct.pack("<I4s", len(chunk), b"JSON")
        + chunk
        + struct.pack("<I4s", len(binary), b"BIN\x00")
        + binary
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)


def write_transform_metadata(path: Path):
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    path.write_text(json.dumps({
        "schema": "comfycolab-3d-transform-v1",
        "output_space": "gltf-y-up-restored-world",
        "geometry_only": True,
        "ultrashape_normalization": {
            "schema": "comfycolab-3d-transform-v1",
            "forward": identity,
            "inverse": identity,
        },
        "validation": {"bytes": 100, "vertices": 3, "faces": 1},
    }))


def rewrite_glb_document(path: Path, mutate):
    payload = path.read_bytes()
    json_length = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20:20 + json_length].decode().rstrip(" \t\r\n\x00"))
    binary_header = 20 + json_length
    binary_length = struct.unpack_from("<I", payload, binary_header)[0]
    binary = payload[binary_header + 8:binary_header + 8 + binary_length]
    mutate(document)
    chunk = json.dumps(document, separators=(",", ":")).encode()
    chunk += b" " * ((4 - len(chunk) % 4) % 4)
    body = (
        struct.pack("<I4s", len(chunk), b"JSON")
        + chunk
        + struct.pack("<I4s", len(binary), b"BIN\x00")
        + binary
    )
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)


class PortFactory:
    def __init__(self, io_type=None):
        self.io_type = io_type

    def Input(self, name, **kwargs):
        return {"direction": "input", "name": name, "io_type": self.io_type, **kwargs}

    def Output(self, name=None, **kwargs):
        return {"direction": "output", "name": name, "io_type": self.io_type, **kwargs}


class FakeIO:
    class ComfyNode:
        pass

    Image = Mask = Combo = Int = Float = Boolean = String = AnyType = PortFactory()
    File3DGLB = PortFactory("FILE_3D_GLB")

    class Hidden:
        unique_id = "UNIQUE_ID"
        prompt = "PROMPT"

    @staticmethod
    def Custom(name):
        return PortFactory(name)

    @staticmethod
    def Schema(**kwargs):
        return types.SimpleNamespace(**kwargs)

    @staticmethod
    def NodeOutput(*values, **kwargs):
        return types.SimpleNamespace(values=values, **kwargs)


class Link:
    def __init__(self, node_id, index):
        self.node_id, self.index = node_id, index


class GraphNode:
    def __init__(self, index, class_type, inputs):
        self.index, self.class_type, self.inputs = index, class_type, inputs
        self.override_display_id = None

    def out(self, index):
        return Link(self.index, index)

    def set_override_display_id(self, node_id):
        self.override_display_id = node_id


class GraphBuilder:
    last = None

    def __init__(self):
        self.nodes = []
        GraphBuilder.last = self

    def node(self, class_type, **inputs):
        node = GraphNode(len(self.nodes), class_type, inputs)
        self.nodes.append(node)
        return node

    def finalize(self):
        result = []
        for node in self.nodes:
            item = {"class_type": node.class_type, "inputs": node.inputs}
            if node.override_display_id is not None:
                item["override_display_id"] = node.override_display_id
            result.append(item)
        return result


class ThreeDNodePackTests(unittest.TestCase):
    def setUp(self):
        self.saved_modules = {
            name: sys.modules.get(name)
            for name in ("comfy_api", "comfy_api.latest", "comfy_execution", "comfy_execution.graph_utils")
        }
        latest = types.ModuleType("comfy_api.latest")
        latest.io = FakeIO
        latest.Types = types.SimpleNamespace(File3D=lambda path, file_format: (path, file_format))
        latest.ComfyExtension = type("ComfyExtension", (), {})
        api = types.ModuleType("comfy_api")
        api.latest = latest
        execution = types.ModuleType("comfy_execution")
        graph_utils = types.ModuleType("comfy_execution.graph_utils")
        graph_utils.GraphBuilder = GraphBuilder
        sys.modules.update({
            "comfy_api": api,
            "comfy_api.latest": latest,
            "comfy_execution": execution,
            "comfy_execution.graph_utils": graph_utils,
        })

    def tearDown(self):
        for name, module in self.saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_import_is_lazy_and_exactly_three_nodes_are_public(self):
        before = set(sys.modules)
        package = load_package()
        imported = set(sys.modules) - before
        self.assertFalse({"torch", "trimesh", "numpy", "PIL", "diffusers"} & imported)
        extension = asyncio.run(package.comfy_entrypoint())
        node_classes = asyncio.run(extension.get_node_list())
        schemas = [node.define_schema() for node in node_classes]
        public = [schema.node_id for schema in schemas if not getattr(schema, "is_dev_only", False)]
        self.assertEqual(
            public,
            [
                "ComfyColabTrellisImageTo3D",
                "ComfyColabUltraShapeRefine",
                "ComfyColabPixal3DImageTo3D",
            ],
        )
        trellis, ultra, pixal = schemas[:3]
        self.assertEqual(trellis.display_name, "ComfyColab TRELLIS.2 — Image to 3D")
        self.assertIn("early untextured geometry preview", trellis.description)
        self.assertEqual(ultra.display_name, "ComfyColab UltraShape — Refine Geometry")
        self.assertEqual(pixal.display_name, "ComfyColab Pixal3D — Image to 3D")
        self.assertEqual(trellis.outputs[0]["name"], "model_3d")
        self.assertEqual(ultra.outputs[0]["name"], "refined_model_3d")
        self.assertEqual(pixal.outputs[0]["name"], "model_3d")
        self.assertTrue(trellis.enable_expand)
        self.assertTrue(ultra.enable_expand)
        self.assertTrue(pixal.enable_expand)
        trellis_inputs = {item["name"]: item for item in trellis.inputs}
        self.assertIn("exact_resolution", trellis_inputs)
        self.assertNotIn("resolution", trellis_inputs)
        self.assertEqual(trellis_inputs["max_tokens"]["default"], 49_152)
        self.assertEqual(trellis_inputs["seed"]["max"], (2**31) - 1)
        ultra_inputs = {item["name"]: item for item in ultra.inputs}
        self.assertEqual(ultra_inputs["seed"]["max"], (2**31) - 1)
        self.assertEqual(ultra_inputs["detail"]["default"], "Conservative")
        pixal_inputs = {item["name"]: item for item in pixal.inputs}
        self.assertEqual(pixal_inputs["quality"]["default"], "1024 — Stable")
        self.assertEqual(
            pixal_inputs["quality"]["options"],
            ["1024 — Stable", "1536 — Experimental"],
        )
        self.assertEqual(pixal_inputs["camera_fov_degrees"]["default"], 0.0)
        self.assertEqual(pixal_inputs["sampling_steps"]["default"], 0)
        self.assertEqual(pixal_inputs["target_face_count"]["default"], 0)
        self.assertEqual(pixal_inputs["texture_size"]["default"], 0)
        self.assertEqual(pixal_inputs["max_tokens"]["default"], 49_152)
        self.assertEqual(pixal_inputs["keep_worker_loaded"]["default"], True)
        self.assertEqual(pixal_inputs["seed"]["max"], (2**31) - 1)
        self.assertEqual(pixal_inputs["cache_mode"]["default"], "Use cache")
        self.assertNotIn("mode", pixal_inputs)
        self.assertNotIn("num_views", pixal_inputs)
        encoded_schema = next(schema for schema in schemas if schema.node_id == "ComfyColab3DEncodedMeshToTrimesh")
        self.assertEqual(encoded_schema.inputs[0]["io_type"], "TRELLIS2_SHAPE_LATENT")

    def test_pixal3d_schema_outputs_native_file3d_and_uses_public_category(self):
        load_package()
        nodes = importlib.import_module("comfycolab_3d_test.nodes")
        schema = nodes.ComfyColabPixal3DImageTo3D.define_schema()
        inputs = {item["name"]: item for item in schema.inputs}

        self.assertEqual(schema.category, "ComfyColab/3D")
        self.assertFalse(getattr(schema, "is_dev_only", False))
        self.assertEqual(schema.outputs[0]["name"], "model_3d")
        self.assertEqual(schema.outputs[0]["io_type"], "FILE_3D_GLB")
        self.assertEqual(list(inputs), [
            "image",
            "quality",
            "seed",
            "remove_background",
            "camera_fov_degrees",
            "sampling_steps",
            "target_face_count",
            "texture_size",
            "max_tokens",
            "keep_worker_loaded",
            "cache_mode",
        ])
        self.assertEqual(inputs["texture_size"]["default"], 0)

    def test_pixal3d_quality_preserves_1536_experimental_without_downgrade(self):
        load_package()
        nodes = importlib.import_module("comfycolab_3d_test.nodes")
        sentinel = object()
        with mock.patch.object(nodes, "build_pixal3d_graph", return_value=sentinel) as build:
            self.assertIs(
                nodes.ComfyColabPixal3DImageTo3D.execute(
                    "image", quality="1536 — Experimental", cache_mode="Disable cache"
                ),
                sentinel,
            )

        settings = build.call_args.args[1]
        self.assertEqual(settings.pipeline_type, "1536_cascade")
        self.assertEqual(settings.texture_size, 4096)

    def test_pixal3d_graph_receives_resolved_settings_and_passes_public_fov_to_worker(self):
        load_package()
        graph = importlib.import_module("comfycolab_3d_test.graph")
        presets = importlib.import_module("comfycolab_3d_test.presets")
        settings = presets.resolve_pixal3d_settings(
            "1024 — Stable",
            sampling_steps=30,
            target_face_count=500_000,
            texture_size=2048,
            max_tokens=49_152,
        )
        result = graph.build_pixal3d_graph(
            "image",
            settings,
            seed=3,
            remove_background="Auto",
            camera_fov_degrees=60.0,
            keep_worker_loaded=True,
            cache_mode="Disable cache",
            cache_key="b" * 64,
        )
        worker = next(
            item for item in result.expand if item["class_type"] == "ComfyColab3DPixal3DWorker"
        )

        self.assertEqual(worker["inputs"]["camera_fov_degrees"], 60.0)
        self.assertEqual(worker["inputs"]["pipeline_type"], "1024_cascade")
        self.assertEqual(worker["inputs"]["sampling_steps"], 30)
        self.assertEqual(worker["inputs"]["target_face_count"], 500_000)
        self.assertEqual(worker["inputs"]["texture_size"], 2048)
        self.assertEqual(worker["inputs"]["max_tokens"], 49_152)

    def test_pixal3d_cache_key_includes_model_revision_and_suppresses_worker_on_hit(self):
        load_package()
        nodes = importlib.import_module("comfycolab_3d_test.nodes")
        cache = importlib.import_module("comfycolab_3d_test.cache")
        presets = importlib.import_module("comfycolab_3d_test.presets")
        image = "pixal-cache-image"
        settings = presets.resolve_pixal3d_settings("1024 — Stable")
        current = cache.pixal3d_cache_key(
            image,
            settings=settings,
            seed=9,
            remove_background="Auto",
            camera_fov_degrees=0.0,
            source_ref="pixal-source-a",
            model_ref="pixal-model-a",
            dinov3_ref="dinov3-a",
            moge_ref="moge-a",
            naf_ref="naf-a",
            environment_ref="env-a",
        )
        changed_revision = cache.pixal3d_cache_key(
            image,
            settings=settings,
            seed=9,
            remove_background="Auto",
            camera_fov_degrees=0.0,
            source_ref="pixal-source-b",
            model_ref="pixal-model-a",
            dinov3_ref="dinov3-a",
            moge_ref="moge-a",
            naf_ref="naf-a",
            environment_ref="env-a",
        )
        self.assertNotEqual(current, changed_revision)

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            "os.environ",
            {
                "COMFYCOLAB_3D_CACHE": str(Path(directory) / "cache"),
                "COMFYCOLAB_3D_OUTPUT": str(Path(directory) / "output"),
            },
        ):
            destination = cache.cache_path(Path(directory) / "cache", "pixal3d", current)
            write_glb(destination, textured=True, volumetric=True)
            with mock.patch.object(
                nodes,
                "pixal3d_cache_key",
                return_value=current,
            ), mock.patch.object(
                nodes,
                "build_pixal3d_graph",
                side_effect=AssertionError("cache hit expanded Pixal3D graph"),
            ):
                result = nodes.ComfyColabPixal3DImageTo3D.execute(
                    image, quality="1024 — Stable", seed=9
                )

        self.assertEqual(result.values[0][1], "glb")
        self.assertTrue(result.values[0][0].endswith(f"{current}.glb"))

    def test_pixal3d_rejects_invalid_cached_glb_before_cache_hit(self):
        load_package()
        nodes = importlib.import_module("comfycolab_3d_test.nodes")
        cache = importlib.import_module("comfycolab_3d_test.cache")
        key = "d" * 64
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            "os.environ", {"COMFYCOLAB_3D_CACHE": str(Path(directory) / "cache")}
        ):
            destination = cache.cache_path(Path(directory) / "cache", "pixal3d", key)
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"not a glb")
            with mock.patch.object(nodes, "pixal3d_cache_key", return_value=key), mock.patch.object(
                nodes, "build_pixal3d_graph", return_value=object()
            ) as build:
                nodes.ComfyColabPixal3DImageTo3D.execute("image")

        build.assert_called_once()
        self.assertFalse(destination.exists())

    def test_pixal3d_finalizer_rejects_path_escaping_cache_key_when_cache_disabled(self):
        load_package()
        nodes = importlib.import_module("comfycolab_3d_test.nodes")
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            "os.environ",
            {
                "COMFYCOLAB_3D_CACHE": str(Path(directory) / "cache"),
                "COMFYCOLAB_3D_OUTPUT": str(Path(directory) / "output"),
            },
        ):
            source = Path(directory) / "model.glb"
            write_glb(source, textured=True, volumetric=True)
            with self.assertRaisesRegex(ValueError, "Cache keys"):
                nodes.ComfyColab3DPixal3DPathToFile3D.execute(
                    str(source), "../escape", cache_mode="Disable cache"
                )
            self.assertFalse((Path(directory) / "escape.glb").exists())

    def test_trellis_facade_expands_to_exact_modular_nodes(self):
        package = load_package()
        nodes = sys.modules.get("comfycolab_3d_test.nodes") or __import__("comfycolab_3d_test.nodes", fromlist=["*"])
        result = nodes.NODE_CLASS_MAPPINGS["ComfyColabTrellisImageTo3D"].execute(
            object(), quality="1024 — Quality", seed=7,
        )
        node_ids = [item["class_type"] for item in result.expand]
        self.assertEqual(node_ids, [
            "LoadTrellis2Models",
            "Trellis2RemoveBackground",
            "Trellis2GetConditioning",
            "Trellis2ImageToShape",
            "ComfyColab3DValidateMesh",
            "Trellis2ProcessMesh",
            "ComfyColab3DValidateMesh",
            "Trellis2ShapeToTexturedMesh",
            "Trellis2RasterizePBR",
            "ComfyColab3DValidateMesh",
            "ComfyColab3DTrimeshToFile3D",
        ])
        self.assertNotIn("Trellis2ExportGLB", node_ids)
        shape = result.expand[3]["inputs"]
        self.assertEqual(shape["ss_sampling_steps"], 12)
        self.assertEqual(shape["shape_sampling_steps"], 12)
        processed = result.expand[5]["inputs"]
        self.assertEqual(processed["remesh"], "on")
        self.assertEqual(processed["remesh.remesh_band"], 1.0)
        self.assertIs(processed["remesh.remove_inner_faces"], True)
        self.assertNotIsInstance(processed["remesh"], dict)

    def test_mesh_gate_rejects_planar_stage_before_downstream_nodes(self):
        load_package()
        nodes = importlib.import_module("comfycolab_3d_test.nodes")
        planar = types.SimpleNamespace(
            vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            faces=[(0, 1, 2)],
        )
        with self.assertRaisesRegex(ValueError, "TRELLIS raw shape.*PCA rank=2"):
            nodes.ComfyColab3DValidateMesh.execute(planar, "TRELLIS raw shape")

    def test_trellis_facade_reports_stages_and_previews_shape_before_texturing(self):
        package = load_package()
        nodes = sys.modules.get("comfycolab_3d_test.nodes") or __import__(
            "comfycolab_3d_test.nodes", fromlist=["*"]
        )
        facade = nodes.NODE_CLASS_MAPPINGS["ComfyColabTrellisImageTo3D"]
        facade.hidden = types.SimpleNamespace(
            unique_id="2",
            prompt={
                "2": {"class_type": "ComfyColabTrellisImageTo3D", "inputs": {}},
                "3": {
                    "class_type": "Preview3DAdvanced",
                    "inputs": {
                        "model_3d": ["2", 0],
                        "viewport_state": {"camera_info": {"position": [0, 0, 3]}},
                        "width": 1024,
                        "height": 768,
                    },
                },
            },
        )
        try:
            with mock.patch.object(nodes, "_send_progress_text") as send_progress_text:
                result = facade.execute(object(), quality="1024 — Quality", seed=7)
        finally:
            del facade.hidden

        send_progress_text.assert_called_once_with(
            "2", "Stage 1/5 - Preparing models and input..."
        )

        node_ids = [item["class_type"] for item in result.expand]
        self.assertEqual(node_ids.count("ComfyColab3DProgressCheckpoint"), 5)
        self.assertIn("ComfyColab3DNeutralMeshToFile3D", node_ids)
        self.assertIn("Preview3DAdvanced", node_ids)
        checkpoints = [
            item["inputs"]
            for item in result.expand
            if item["class_type"] == "ComfyColab3DProgressCheckpoint"
        ]
        self.assertEqual(
            [
                (item["completed"], item["total"], item["status"])
                for item in checkpoints
            ],
            [
                (1, 5, "Stage 2/5 - Generating 3D shape..."),
                (2, 5, "Stage 3/5 - Building geometry preview..."),
                (3, 5, "Stage 4/5 - Geometry preview ready; generating texture..."),
                (4, 5, "Stage 5/5 - Baking PBR material and final GLB..."),
                (5, 5, "Complete - 3D model ready"),
            ],
        )

        preview_index = node_ids.index("Preview3DAdvanced")
        texture_index = node_ids.index("Trellis2ShapeToTexturedMesh")
        self.assertLess(preview_index, texture_index)
        self.assertEqual(result.expand[preview_index]["override_display_id"], "3")
        preview_inputs = result.expand[preview_index]["inputs"]
        self.assertIn("model_3d", preview_inputs)
        self.assertNotIn("model_file", preview_inputs)
        self.assertEqual(preview_inputs["width"], 1024)
        self.assertEqual(preview_inputs["height"], 768)
        self.assertEqual(
            preview_inputs["viewport_state"],
            {"camera_info": {"position": [0, 0, 3]}},
        )

        texture = result.expand[texture_index]
        checkpoint = result.expand[texture["inputs"]["shape_slat"].node_id]
        self.assertEqual(
            checkpoint["inputs"]["status"],
            "Stage 4/5 - Geometry preview ready; generating texture...",
        )

    def test_trellis_early_preview_preserves_standard_preview_contract(self):
        load_package()
        nodes = importlib.import_module("comfycolab_3d_test.nodes")
        facade = nodes.NODE_CLASS_MAPPINGS["ComfyColabTrellisImageTo3D"]
        facade.hidden = types.SimpleNamespace(
            unique_id="2",
            prompt={
                "2": {"class_type": "ComfyColabTrellisImageTo3D", "inputs": {}},
                "90": {
                    "class_type": "Preview3D",
                    "inputs": {"model_file": ["2", 0], "camera_info": "camera"},
                },
            },
        )
        try:
            result = facade.execute(object(), quality="512 — Fast", seed=7)
        finally:
            del facade.hidden
        preview = next(item for item in result.expand if item["class_type"] == "Preview3D")
        self.assertEqual(preview["override_display_id"], "90")
        self.assertEqual(preview["inputs"]["camera_info"], "camera")
        self.assertIn("model_file", preview["inputs"])
        self.assertNotIn("model_3d", preview["inputs"])

    def test_trellis_progress_without_a_viewer_does_not_export_preview_glb(self):
        load_package()
        nodes = importlib.import_module("comfycolab_3d_test.nodes")
        facade = nodes.NODE_CLASS_MAPPINGS["ComfyColabTrellisImageTo3D"]
        facade.hidden = types.SimpleNamespace(
            unique_id="2",
            prompt={"2": {"class_type": "ComfyColabTrellisImageTo3D", "inputs": {}}},
        )
        try:
            result = facade.execute(object(), quality="512 — Fast", seed=7)
        finally:
            del facade.hidden
        node_ids = [item["class_type"] for item in result.expand]
        self.assertEqual(node_ids.count("ComfyColab3DProgressCheckpoint"), 5)
        self.assertNotIn("ComfyColab3DNeutralMeshToFile3D", node_ids)
        stage4 = next(
            item
            for item in result.expand
            if item["class_type"] == "ComfyColab3DProgressCheckpoint"
            and item["inputs"]["completed"] == 3
        )
        self.assertIn("wait_for", stage4["inputs"])

    def test_progress_checkpoint_reports_text_and_native_progress_on_facade(self):
        load_package()
        nodes = importlib.import_module("comfycolab_3d_test.nodes")
        latest = sys.modules["comfy_api.latest"]
        set_progress = mock.AsyncMock()
        latest.ComfyAPI = lambda: types.SimpleNamespace(
            execution=types.SimpleNamespace(set_progress=set_progress)
        )
        send_progress_text = mock.Mock()
        previous_server = sys.modules.get("server")
        sys.modules["server"] = types.SimpleNamespace(
            PromptServer=types.SimpleNamespace(
                instance=types.SimpleNamespace(send_progress_text=send_progress_text)
            )
        )
        try:
            result = asyncio.run(
                nodes.ComfyColab3DProgressCheckpoint.execute(
                    "value",
                    "2",
                    3,
                    5,
                    "Stage 4/5 - Geometry preview ready; generating texture...",
                )
            )
        finally:
            if previous_server is None:
                sys.modules.pop("server", None)
            else:
                sys.modules["server"] = previous_server
        self.assertEqual(result.values, ("value",))
        send_progress_text.assert_called_once_with(
            "Stage 4/5 - Geometry preview ready; generating texture...", "2"
        )
        set_progress.assert_awaited_once_with(
            value=3.0,
            max_value=5.0,
            node_id="2",
        )

    def test_ultrashape_geometry_only_graph_masks_background_and_applies_face_target(self):
        load_package()
        graph = importlib.import_module("comfycolab_3d_test.graph")
        result = graph.build_ultrashape_graph(
            "input.glb",
            "reference-image",
            detail="Detailed",
            seed=17,
            retexture=False,
            steps=24,
            num_latents=16_384,
            octree_resolution=1024,
            decode_chunk_size=4096,
            target_face_count=321_000,
            texture_size=2048,
            low_vram="auto",
            cache_mode="Use cache",
            geometry_cache_key="a" * 64,
        )
        node_ids = [item["class_type"] for item in result.expand]
        self.assertEqual(node_ids, [
            "Trellis2RemoveBackground",
            "ComfyColab3DUltraShapeWorker",
            "ComfyColab3DGLBToTrellisMesh",
            "Trellis2ProcessMesh",
            "ComfyColab3DNeutralMeshToFile3D",
        ])
        worker_inputs = result.expand[1]["inputs"]
        self.assertEqual(worker_inputs["reference_image"].node_id, 0)
        self.assertEqual(worker_inputs["reference_image"].index, 0)
        self.assertEqual(worker_inputs["reference_mask"].node_id, 0)
        self.assertEqual(worker_inputs["reference_mask"].index, 1)
        self.assertEqual(result.expand[3]["inputs"]["target_face_count"], 321_000)

    def test_preset_override_and_coordinate_round_trip(self):
        load_package()
        presets = importlib.import_module("comfycolab_3d_test.presets")
        transforms = importlib.import_module("comfycolab_3d_test.transforms")
        settings = presets.resolve_trellis_settings("512 — Fast", resolution="1536_cascade", texture_size=4096)
        self.assertEqual((settings.resolution, settings.texture_size), ("1536_cascade", 4096))
        vertices = [(1, 2, 3), (-4, 5, -6)]
        restored = transforms.y_up_to_z_up(transforms.z_up_to_y_up(vertices))
        self.assertEqual(restored, [(1.0, 2.0, 3.0), (-4.0, 5.0, -6.0)])
        asymmetric = [(-4.0, 2.0, 1.0), (6.0, 5.0, 3.0), (1.0, -1.0, 2.0)]
        transform = transforms.normalization_for(asymmetric)
        normalized = transforms.apply_normalization(asymmetric, transform)
        extent = max(max(row[axis] for row in normalized) - min(row[axis] for row in normalized) for axis in range(3))
        self.assertAlmostEqual(extent, 0.99999, places=7)

        fast = presets.resolve_ultrashape_settings("Fast")
        conservative = presets.resolve_ultrashape_settings("Conservative")
        detailed = presets.resolve_ultrashape_settings("Detailed")
        ultra = presets.resolve_ultrashape_settings("Ultra")
        self.assertEqual(fast.octree_resolution, 512)
        self.assertEqual(conservative.octree_resolution, 512)
        self.assertEqual(detailed.octree_resolution, 1024)
        self.assertEqual(ultra.octree_resolution, 1024)
        self.assertEqual(
            presets.ULTRASHAPE_EXPERIMENTAL_PRESETS,
            {"Detailed", "Ultra"},
        )
        pixal_stable = presets.resolve_pixal3d_settings("1024 — Stable")
        pixal_1536 = presets.resolve_pixal3d_settings("1536 — Experimental")
        self.assertEqual(pixal_stable.pipeline_type, "1024_cascade")
        self.assertEqual((pixal_stable.sampling_steps, pixal_stable.target_face_count, pixal_stable.texture_size), (12, 200_000, 2048))
        self.assertEqual(pixal_1536.pipeline_type, "1536_cascade")
        self.assertEqual(pixal_1536.texture_size, 4096)
        inverted = transforms.invert_normalization(normalized, transform)
        for expected, actual in zip(asymmetric, inverted):
            for expected_value, actual_value in zip(expected, actual):
                self.assertAlmostEqual(expected_value, actual_value, places=7)
        with self.assertRaisesRegex(ValueError, "sampling_steps"):
            presets.resolve_trellis_settings("512 — Fast", sampling_steps=51)
        with self.assertRaisesRegex(ValueError, "at least 1000"):
            presets.resolve_trellis_settings("512 — Fast", target_face_count=999)
        with self.assertRaisesRegex(ValueError, "at least 512"):
            presets.resolve_trellis_settings("512 — Fast", texture_size=511)

    def test_cache_keys_are_deterministic_and_atomic_write_cleans_partial(self):
        load_package()
        cache = importlib.import_module("comfycolab_3d_test.cache")
        self.assertEqual(
            cache.deterministic_cache_key("shape", seed=1, options={"b": 2, "a": 1}),
            cache.deterministic_cache_key("shape", options={"a": 1, "b": 2}, seed=1),
        )
        key = "a" * 64
        self.assertEqual(
            cache.cache_path("/content/.comfycolab/cache/3d", "shape", key),
            Path("/content/.comfycolab/cache/3d") / "shape" / key / "model.glb",
        )
        self.assertEqual(
            cache.cache_path("/content/.comfycolab/cache/3d", "ultrashape", key, "geometry.glb"),
            Path("/content/.comfycolab/cache/3d") / "ultrashape" / key / "geometry.glb",
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "value.bin"
            cache.atomic_write_bytes(target, b"complete")
            self.assertEqual(target.read_bytes(), b"complete")
            self.assertEqual(list(target.parent.glob("*.partial")), [])

    def test_cache_schema_versions_change_result_keys(self):
        load_package()
        cache = importlib.import_module("comfycolab_3d_test.cache")
        settings = types.SimpleNamespace(
            resolution="512", sampling_steps=10, target_face_count=200_000,
            texture_size=1024, max_tokens=49_152,
        )
        inputs = dict(
            settings=settings, seed=0, remove_background="Auto",
            comfyui_ref="comfy", trellis_ref="trellis",
            trellis_patch_id="patch", birefnet_ref="birefnet",
        )
        current = cache.trellis_cache_key("image", **inputs)
        legacy = cache.trellis_cache_key(
            "image", **inputs, result_schema="comfycolab-trellis-result-v1"
        )
        self.assertNotEqual(current, legacy)
        self.assertEqual(cache.TRELLIS_RESULT_SCHEMA, "comfycolab-trellis-result-v2")
        self.assertEqual(
            cache.ULTRASHAPE_GEOMETRY_SCHEMA,
            "comfycolab-ultrashape-geometry-v2",
        )

    def test_planar_cached_result_is_invalidated(self):
        load_package()
        nodes = importlib.import_module("comfycolab_3d_test.nodes")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "planar.glb"
            write_glb(path, textured=True)
            self.assertFalse(nodes._valid_cached_glb(path, require_textured=True))
            self.assertFalse(path.exists())

    def test_scene_transform_is_baked_before_volumetric_validation(self):
        load_package()
        file3d = importlib.import_module("comfycolab_3d_test.file3d")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flattened = root / "flattened.glb"
            write_glb(flattened, volumetric=True)
            rewrite_glb_document(
                flattened,
                lambda document: document.update(
                    nodes=[{"mesh": 0, "scale": [1.0, 1.0, 0.0]}],
                    scenes=[{"nodes": [0]}],
                    scene=0,
                ),
            )
            with self.assertRaisesRegex(ValueError, "numerically collapsed"):
                file3d.validate_volumetric_glb(flattened, stage="flattened scene")

            assembly = root / "assembly.glb"
            write_glb(assembly)
            rewrite_glb_document(
                assembly,
                lambda document: document.update(
                    nodes=[
                        {"mesh": 0},
                        {"mesh": 0, "translation": [0.0, 0.0, 1.0]},
                    ],
                    scenes=[{"nodes": [0, 1]}],
                    scene=0,
                ),
            )
            metrics = file3d.validate_volumetric_glb(assembly, stage="scene assembly")
            self.assertEqual(metrics.numerical_rank, 3)

    def test_malformed_glb_schema_becomes_cache_miss_instead_of_attribute_error(self):
        load_package()
        nodes = importlib.import_module("comfycolab_3d_test.nodes")
        worker = importlib.import_module("comfycolab_3d_test.worker")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trellis_path = root / "trellis.glb"
            write_glb(trellis_path, textured=True, volumetric=True)
            rewrite_glb_document(
                trellis_path,
                lambda document: document["meshes"][0].update(primitives=["bad"]),
            )
            self.assertFalse(nodes._valid_cached_glb(trellis_path, require_textured=True))
            self.assertFalse(trellis_path.exists())

            key = "c" * 64
            cache_dir = root / key
            cache_dir.mkdir()
            write_glb(cache_dir / "geometry.glb", volumetric=True)
            write_transform_metadata(cache_dir / "transform.json")
            worker.write_geometry_cache_record(cache_dir, key)
            rewrite_glb_document(
                cache_dir / "geometry.glb",
                lambda document: document["meshes"][0].update(primitives=["bad"]),
            )
            self.assertFalse(worker.validate_geometry_cache_record(cache_dir, key))

    def test_ultrashape_geometry_cache_record_and_refresh_rollback_are_atomic(self):
        load_package()
        worker = importlib.import_module("comfycolab_3d_test.worker")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = "b" * 64
            destination = root / key
            destination.mkdir()
            write_glb(destination / "geometry.glb", volumetric=True)
            write_transform_metadata(destination / "transform.json")
            worker.write_geometry_cache_record(destination, key)
            self.assertTrue(worker.validate_geometry_cache_record(destination, key))
            (destination / "transform.json").write_text("{}")
            self.assertFalse(worker.validate_geometry_cache_record(destination, key))
            (destination / "old-marker").write_text("preserve me")

            staging = root / ".staging"
            staging.mkdir()
            (staging / "new-marker").write_text("new")
            real_replace = worker.os.replace

            def fail_new_install(source, target):
                if Path(source) == staging:
                    raise OSError("simulated rename failure")
                return real_replace(source, target)

            with mock.patch.object(worker.os, "replace", side_effect=fail_new_install):
                with self.assertRaisesRegex(OSError, "simulated"):
                    worker.atomic_replace_cache_directory(staging, destination)
            self.assertEqual((destination / "old-marker").read_text(), "preserve me")
            self.assertTrue(staging.exists())

    def test_glb_validation_and_string_backed_file3d(self):
        load_package()
        file3d = importlib.import_module("comfycolab_3d_test.file3d")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.glb"
            write_glb(path)
            self.assertIn("meshes", file3d.validate_glb(path))
            materialized = file3d.materialize_file3d(path)
            self.assertEqual(materialized, (str(path), "glb"))
            output_root = Path(directory) / "ComfyUI" / "output" / "3d"
            with mock.patch.dict("os.environ", {"COMFYCOLAB_3D_OUTPUT": str(output_root)}):
                published = file3d.publish_glb(path, "published-key")
            self.assertEqual(published, output_root / "published-key.glb")
            self.assertTrue(published.is_file())

    def test_glb_validation_rejects_nonfinite_vertices_and_invalid_indices(self):
        load_package()
        file3d = importlib.import_module("comfycolab_3d_test.file3d")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.glb"
            write_glb(path, textured=True)
            payload = bytearray(path.read_bytes())
            json_length = struct.unpack_from("<I", payload, 12)[0]
            binary_offset = 12 + 8 + json_length + 8
            struct.pack_into("<f", payload, binary_offset, float("nan"))
            path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                file3d.validate_glb(path)

            write_glb(path, textured=True)
            payload = bytearray(path.read_bytes())
            json_length = struct.unpack_from("<I", payload, 12)[0]
            binary_offset = 12 + 8 + json_length + 8
            struct.pack_into("<H", payload, binary_offset + 36, 99)
            path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "invalid indices"):
                file3d.validate_glb(path)

    def test_glb_validation_rejects_invalid_uvs_textures_and_empty_primitives(self):
        load_package()
        file3d = importlib.import_module("comfycolab_3d_test.file3d")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.glb"
            write_glb(path, textured=True, uv_count=2)
            with self.assertRaisesRegex(ValueError, "UV count"):
                file3d.validate_glb(path, require_texture=True, require_uv=True)
            write_glb(
                path,
                textured=True,
                uv_values=(0, 0, float("nan"), 0, 0, 1),
            )
            with self.assertRaisesRegex(ValueError, "non-finite UV"):
                file3d.validate_glb(path, require_texture=True, require_uv=True)
            write_glb(path, textured=True, invalid_image_view=True)
            with self.assertRaisesRegex(ValueError, "embedded texture"):
                file3d.validate_glb(path, require_texture=True, require_uv=True)
            write_glb(path, empty_primitives=True)
            with self.assertRaisesRegex(ValueError, "no primitives"):
                file3d.validate_glb(path)

    def test_glb_validation_enforces_triangle_accessor_semantics(self):
        load_package()
        file3d = importlib.import_module("comfycolab_3d_test.file3d")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.glb"
            write_glb(path, textured=True)
            rewrite_glb_document(
                path, lambda document: document["accessors"][0].update(type="VEC2")
            )
            with self.assertRaisesRegex(ValueError, "POSITION.*FLOAT VEC3"):
                file3d.validate_glb(path, require_texture=True, require_uv=True)
            write_glb(path, textured=True)
            rewrite_glb_document(
                path, lambda document: document["accessors"][1].update(count=1)
            )
            with self.assertRaisesRegex(ValueError, "multiple of three"):
                file3d.validate_glb(path, require_texture=True, require_uv=True)
            write_glb(path, textured=True)
            rewrite_glb_document(
                path, lambda document: document["meshes"][0]["primitives"][0].update(mode=1)
            )
            with self.assertRaisesRegex(ValueError, "TRIANGLES"):
                file3d.validate_glb(path, require_texture=True, require_uv=True)

    def test_scene_baking_applies_every_instance_transform_before_concatenation(self):
        load_package()
        file3d = importlib.import_module("comfycolab_3d_test.file3d")

        class Geometry:
            def __init__(self, name):
                self.name, self.transforms = name, []

            def copy(self):
                return Geometry(self.name)

            def apply_transform(self, transform):
                self.transforms.append(transform)

        class Scene:
            def __init__(self):
                self.geometry = {"shared": Geometry("shared")}
                self.graph = types.SimpleNamespace(
                    nodes_geometry=["instance-a", "instance-b"],
                    get=lambda node: (f"transform-{node}", "shared"),
                )

        fake_trimesh = types.SimpleNamespace(
            Scene=Scene,
            util=types.SimpleNamespace(concatenate=lambda geometries: list(geometries)),
        )
        baked = file3d.bake_scene_mesh(Scene(), fake_trimesh)
        self.assertEqual([mesh.transforms for mesh in baked], [["transform-instance-a"], ["transform-instance-b"]])

    def test_labeled_glb_full_transform_pipeline_preserves_geometry_contract(self):
        """Regression for scene -> TRELLIS -> UltraShape -> glTF materialization."""
        load_package()
        file3d = importlib.import_module("comfycolab_3d_test.file3d")
        transforms = importlib.import_module("comfycolab_3d_test.transforms")
        transform_path = ROOT / "worker" / "ultrashape" / "transform_contract.py"
        transform_spec = importlib.util.spec_from_file_location("cc3d_ultra_transform_regression", transform_path)
        ultra = importlib.util.module_from_spec(transform_spec)
        assert transform_spec.loader
        transform_spec.loader.exec_module(ultra)

        def apply_matrix(matrix, point):
            return ultra.apply_matrix_to_point(matrix, point)

        def subtract(left, right):
            return tuple(left[index] - right[index] for index in range(3))

        def cross(left, right):
            return (
                left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0],
            )

        def normalized(vector):
            length = math.sqrt(sum(value * value for value in vector))
            return tuple(value / length for value in vector)

        def face_normals(vertices, faces):
            return [
                normalized(cross(subtract(vertices[b], vertices[a]), subtract(vertices[c], vertices[a])))
                for a, b, c in faces
            ]

        def distance(left, right):
            return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))

        labels = ["nose", "right-fin", "crown", "lower-keel"]
        source_vertices = [
            (-2.0, 1.0, 0.5),
            (3.0, 1.25, -0.25),
            (-1.0, 4.0, 2.0),
            (0.5, -2.0, 1.0),
        ]
        source_faces = [(0, 1, 2), (0, 3, 1)]
        scene_matrix = [
            [0.0, 0.0, 1.0, 4.0],
            [0.0, 1.0, 0.0, -3.0],
            [-1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        class NumericMesh:
            def __init__(self, vertices, faces, vertex_labels):
                self.vertices = [tuple(map(float, row)) for row in vertices]
                self.faces = [tuple(face) for face in faces]
                self.labels = list(vertex_labels)

            def copy(self):
                return NumericMesh(self.vertices, self.faces, self.labels)

            def apply_transform(self, matrix):
                self.vertices = [apply_matrix(matrix, vertex) for vertex in self.vertices]

            @property
            def normals(self):
                return face_normals(self.vertices, self.faces)

            def export(self, path, file_type):
                self.assert_export_type = file_type
                position_bytes = b"".join(struct.pack("<fff", *vertex) for vertex in self.vertices)
                index_bytes = b"".join(struct.pack("<H", index) for face in self.faces for index in face)
                index_offset = len(position_bytes)
                binary = position_bytes + index_bytes
                binary += b"\x00" * ((4 - len(binary) % 4) % 4)
                document = {
                    "asset": {"version": "2.0"},
                    "buffers": [{"byteLength": len(binary)}],
                    "bufferViews": [
                        {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes)},
                        {"buffer": 0, "byteOffset": index_offset, "byteLength": len(index_bytes)},
                    ],
                    "accessors": [
                        {"bufferView": 0, "componentType": 5126, "count": len(self.vertices), "type": "VEC3"},
                        {"bufferView": 1, "componentType": 5123, "count": len(self.faces) * 3, "type": "SCALAR"},
                    ],
                    "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
                    "extras": {
                        "labels": self.labels,
                        "positions": self.vertices,
                        "indices": self.faces,
                        "normals": self.normals,
                    },
                }
                chunk = json.dumps(document, separators=(",", ":")).encode()
                chunk += b" " * ((4 - len(chunk) % 4) % 4)
                body = (
                    struct.pack("<I4s", len(chunk), b"JSON")
                    + chunk
                    + struct.pack("<I4s", len(binary), b"BIN\x00")
                    + binary
                )
                Path(path).write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)

        class Scene:
            def __init__(self, mesh):
                self.geometry = {"labeled": mesh}
                self.graph = types.SimpleNamespace(
                    nodes_geometry=["labeled-instance"],
                    get=lambda _node: (scene_matrix, "labeled"),
                )

        fake_trimesh = types.SimpleNamespace(
            Scene=Scene,
            util=types.SimpleNamespace(concatenate=lambda meshes: list(meshes)[0]),
        )
        baked = file3d.bake_scene_mesh(
            Scene(NumericMesh(source_vertices, source_faces, labels)), fake_trimesh,
        )
        expected_baked = [apply_matrix(scene_matrix, vertex) for vertex in source_vertices]
        self.assertEqual(baked.vertices, expected_baked)

        z_up = transforms.y_up_to_z_up(baked.vertices)
        minimum = tuple(min(vertex[axis] for vertex in z_up) for axis in range(3))
        maximum = tuple(max(vertex[axis] for vertex in z_up) for axis in range(3))
        normalization = ultra.normalization_from_bounds(minimum, maximum, normalize_scale=0.99)
        normalized_vertices = [apply_matrix(normalization["forward"], vertex) for vertex in z_up]
        restored_z_up = [apply_matrix(normalization["inverse"], vertex) for vertex in normalized_vertices]
        restored_y_up = transforms.z_up_to_y_up(restored_z_up)

        self.assertTrue(all(math.isfinite(value) for vertex in restored_y_up for value in vertex))
        for actual, expected in zip(restored_y_up, expected_baked):
            for actual_value, expected_value in zip(actual, expected):
                self.assertAlmostEqual(actual_value, expected_value, places=10)
        self.assertAlmostEqual(
            distance(restored_y_up[0], restored_y_up[2]),
            distance(expected_baked[0], expected_baked[2]),
            places=10,
        )

        final_mesh = NumericMesh(restored_y_up, source_faces, labels)
        expected_normals = face_normals(expected_baked, source_faces)
        for actual, expected in zip(final_mesh.normals, expected_normals):
            for actual_value, expected_value in zip(actual, expected):
                self.assertAlmostEqual(actual_value, expected_value, places=10)

        with tempfile.TemporaryDirectory() as directory:
            exported = Path(directory) / "labeled.glb"
            file3d.export_trimesh_atomic(final_mesh, exported)
            document = file3d.validate_glb(exported)
        self.assertEqual(document["extras"]["labels"], labels)
        self.assertEqual([tuple(face) for face in document["extras"]["indices"]], source_faces)
        self.assertEqual(document["meshes"][0]["primitives"][0]["indices"], 1)
        for actual, expected in zip(document["extras"]["normals"], expected_normals):
            for actual_value, expected_value in zip(actual, expected):
                self.assertAlmostEqual(actual_value, expected_value, places=10)

    def test_pinned_ultrashape_defaults_and_cache_root(self):
        load_package()
        nodes = importlib.import_module("comfycolab_3d_test.nodes")
        self.assertEqual(nodes._cache_root(), Path("/content/.comfycolab/cache/3d"))
        self.assertEqual(nodes.DEFAULT_ULTRASHAPE_SOURCE, "/content/UltraShape-1.0")
        self.assertTrue(nodes.DEFAULT_ULTRASHAPE_PYTHON.endswith("/.ce/.pixi/envs/trellis2-nodes/bin/python"))
        self.assertEqual(nodes.ULTRASHAPE_SOURCE_REF, "5e8dcef05df101ab00ab6cd5fdd0ed0c74fbca66")

    def test_export_flips_texture_v_once_without_mutating_input(self):
        load_package()
        nodes = importlib.import_module("comfycolab_3d_test.nodes")

        class Column(list):
            def __rsub__(self, value):
                return [value - item for item in self]

        class UVArray:
            def __init__(self, rows):
                self.rows = [list(row) for row in rows]

            def copy(self):
                return UVArray(self.rows)

            def __getitem__(self, key):
                row_selector, column = key
                self.assert_full_slice(row_selector)
                return Column(row[column] for row in self.rows)

            def __setitem__(self, key, values):
                row_selector, column = key
                self.assert_full_slice(row_selector)
                for row, value in zip(self.rows, values):
                    row[column] = value

            @staticmethod
            def assert_full_slice(value):
                if not isinstance(value, slice) or value != slice(None):
                    raise AssertionError("expected a full UV row slice")

        class Mesh:
            def __init__(self, uv):
                self.visual = types.SimpleNamespace(uv=uv)
                self.vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 1)]
                self.faces = [(0, 1, 2)]
                self.transforms = []

            def copy(self):
                return Mesh(self.visual.uv.copy())

            def apply_transform(self, matrix):
                self.transforms.append(matrix)

        source = Mesh(UVArray([(0.1, 0.25), (0.5, 0.75), (0.9, 0.0)]))
        fake_numpy = types.SimpleNamespace(array=lambda value: value)
        real_import = nodes.importlib.import_module

        def import_module(name):
            return fake_numpy if name == "numpy" else real_import(name)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            nodes.importlib, "import_module", side_effect=import_module
        ), mock.patch.object(nodes, "validate_volumetric_mesh"), mock.patch.object(
            nodes, "validate_volumetric_glb"
        ), mock.patch.object(nodes, "export_trimesh_atomic") as export:
            nodes._export_z_up_mesh(
                source,
                Path(directory) / "mesh.glb",
                require_textured=True,
            )

        exported = export.call_args.args[0]
        self.assertEqual([row[1] for row in source.visual.uv.rows], [0.25, 0.75, 0.0])
        self.assertEqual([row[1] for row in exported.visual.uv.rows], [0.75, 0.25, 1.0])

    def test_temporary_cleanup_cannot_delete_an_arbitrary_parent(self):
        load_package()
        nodes = importlib.import_module("comfycolab_3d_test.nodes")
        with tempfile.TemporaryDirectory() as directory:
            victim = Path(directory) / "victim" / "refined.glb"
            victim.parent.mkdir()
            victim.write_bytes(b"not-a-glb")
            nodes._remove_owned_ultrashape_temp(victim)
            self.assertTrue(victim.exists())

        owned = Path(tempfile.mkdtemp(prefix="comfycolab-ultrashape-"))
        output = owned / "refined.glb"
        output.write_bytes(b"temporary")
        nodes._remove_owned_ultrashape_temp(output)
        self.assertFalse(owned.exists())

    def test_trellis_public_cache_hit_skips_graph_and_inference(self):
        load_package()
        nodes = importlib.import_module("comfycolab_3d_test.nodes")
        presets = importlib.import_module("comfycolab_3d_test.presets")
        cache = importlib.import_module("comfycolab_3d_test.cache")
        image = "cache-test-image"
        settings = presets.resolve_trellis_settings("1024 — Quality")
        key = cache.trellis_cache_key(
            image,
            settings=settings,
            seed=11,
            remove_background="Auto",
            comfyui_ref="8b099de36acd81acd1afa3b5442951dc847e0a52",
            trellis_ref="9b878516f2dc2fd873f4f6cceadba403dd12d83e",
            trellis_patch_id="trellis2-strict-1536-birefnet-pin-metrics-v4",
            birefnet_ref="e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            "os.environ",
            {
                "COMFYCOLAB_3D_CACHE": str(Path(directory) / "cache"),
                "COMFYCOLAB_3D_OUTPUT": str(Path(directory) / "output"),
            },
        ):
            destination = cache.cache_path(Path(directory) / "cache", "trellis", key)
            write_glb(destination, textured=True, volumetric=True)
            with mock.patch.object(
                nodes, "build_trellis_graph", side_effect=AssertionError("cache hit expanded graph")
            ):
                result = nodes.ComfyColabTrellisImageTo3D.execute(
                    image, quality="1024 — Quality", seed=11,
                )
        self.assertEqual(result.values[0][1], "glb")
        self.assertTrue(result.values[0][0].endswith(f"{key}.glb"))

    def test_trellis_refresh_disable_and_corruption_expand_instead_of_hitting_cache(self):
        load_package()
        nodes = importlib.import_module("comfycolab_3d_test.nodes")
        cache = importlib.import_module("comfycolab_3d_test.cache")
        presets = importlib.import_module("comfycolab_3d_test.presets")
        image = "cache-test-image"
        settings = presets.resolve_trellis_settings("512 — Fast")
        key = cache.trellis_cache_key(
            image,
            settings=settings,
            seed=0,
            remove_background="Auto",
            comfyui_ref="8b099de36acd81acd1afa3b5442951dc847e0a52",
            trellis_ref="9b878516f2dc2fd873f4f6cceadba403dd12d83e",
            trellis_patch_id="trellis2-strict-1536-birefnet-pin-metrics-v4",
            birefnet_ref="e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
        )
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            "os.environ", {"COMFYCOLAB_3D_CACHE": directory}
        ):
            destination = cache.cache_path(directory, "trellis", key)
            write_glb(destination, textured=True)
            with mock.patch.object(nodes, "build_trellis_graph", return_value=sentinel) as build:
                self.assertIs(
                    nodes.ComfyColabTrellisImageTo3D.execute(
                        image, quality="512 — Fast", cache_mode="Refresh this node"
                    ),
                    sentinel,
                )
                self.assertIs(
                    nodes.ComfyColabTrellisImageTo3D.execute(
                        image, quality="512 — Fast", cache_mode="Disable cache"
                    ),
                    sentinel,
                )
                self.assertEqual(build.call_count, 2)
            destination.write_bytes(b"corrupt")
            with mock.patch.object(nodes, "build_trellis_graph", return_value=sentinel):
                self.assertIs(
                    nodes.ComfyColabTrellisImageTo3D.execute(image, quality="512 — Fast"),
                    sentinel,
                )
            self.assertFalse(destination.exists())

    def test_missing_upstream_nodes_raise_actionable_dependency_error(self):
        load_package()
        nodes_3d = importlib.import_module("comfycolab_3d_test.nodes")
        fake_registry = types.ModuleType("nodes")
        fake_registry.NODE_CLASS_MAPPINGS = {"LoadTrellis2Models": object()}
        with mock.patch.dict(sys.modules, {"nodes": fake_registry}):
            with self.assertRaisesRegex(RuntimeError, "comfycolab start --refresh"):
                nodes_3d.ComfyColabTrellisImageTo3D.execute(
                    "image", quality="512 — Fast", cache_mode="Disable cache"
                )

    def test_trellis_seed_boundary_rejects_values_upstream_cannot_accept(self):
        load_package()
        nodes_3d = importlib.import_module("comfycolab_3d_test.nodes")
        with self.assertRaisesRegex(ValueError, "2147483647"):
            nodes_3d.ComfyColabTrellisImageTo3D.execute(
                "image", quality="512 — Fast", seed=2**31
            )

    def test_ultrashape_texture_cache_hit_skips_worker_and_texture_inference(self):
        load_package()
        nodes = importlib.import_module("comfycolab_3d_test.nodes")
        cache = importlib.import_module("comfycolab_3d_test.cache")
        artifact_module = types.SimpleNamespace(
            ULTRASHAPE_REVISION="checkpoint-ref",
            DINOV2_REVISION="dinov2-ref",
        )
        image = "cache-test-image"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.glb"
            write_glb(source, volumetric=True)
            geometry_key = cache.ultrashape_geometry_cache_key(
                "source-geometry",
                image,
                detail="Conservative",
                seed=5,
                steps=24,
                num_latents=16_384,
                octree_resolution=512,
                decode_chunk_size=4096,
                low_vram="auto",
                worker_ref=nodes.ULTRASHAPE_SOURCE_REF,
                checkpoint_ref="checkpoint-ref",
                dinov2_ref="dinov2-ref",
                transform_schema=nodes.TRANSFORM_SCHEMA,
            )
            geometry_path = cache.cache_path(root / "cache", "ultrashape", geometry_key, "geometry.glb")
            write_glb(geometry_path, volumetric=True)
            write_transform_metadata(geometry_path.parent / "transform.json")
            worker = importlib.import_module("comfycolab_3d_test.worker")
            worker.write_geometry_cache_record(geometry_path.parent, geometry_key)
            texture_key = cache.texture_cache_key(
                "refined-geometry",
                image,
                seed=5,
                target_face_count=500_000,
                texture_size=2048,
                texture_sampling_steps=12,
                trellis_ref="9b878516f2dc2fd873f4f6cceadba403dd12d83e",
            )
            texture_path = cache.cache_path(root / "cache", "texture", texture_key)
            write_glb(texture_path, textured=True, volumetric=True)

            def geometry_digest(path):
                return "refined-geometry" if Path(path) == geometry_path else "source-geometry"

            with mock.patch.dict(
                "os.environ",
                {
                    "COMFYCOLAB_3D_CACHE": str(root / "cache"),
                    "COMFYCOLAB_3D_OUTPUT": str(root / "output"),
                },
            ), mock.patch.object(
                nodes, "_load_artifact_provisioner", return_value=artifact_module
            ), mock.patch.object(
                nodes, "canonical_glb_geometry_digest", side_effect=geometry_digest
            ), mock.patch.object(
                nodes, "build_ultrashape_graph", side_effect=AssertionError("cache hit expanded graph")
            ):
                result = nodes.ComfyColabUltraShapeRefine.execute(
                    source, image, detail="Conservative", seed=5,
                )
        self.assertEqual(result.values[0][1], "glb")
        self.assertTrue(result.values[0][0].endswith(f"{texture_key}.glb"))

    def test_ultrashape_geometry_cache_hit_skips_birefnet_and_worker_models(self):
        load_package()
        nodes = importlib.import_module("comfycolab_3d_test.nodes")
        cache = importlib.import_module("comfycolab_3d_test.cache")
        worker = importlib.import_module("comfycolab_3d_test.worker")
        artifacts = types.SimpleNamespace(
            ULTRASHAPE_REVISION="checkpoint-ref",
            DINOV2_REVISION="dinov2-ref",
        )
        image = "geometry-cache-image"
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.glb"
            write_glb(source, volumetric=True)
            key = cache.ultrashape_geometry_cache_key(
                "source-geometry",
                image,
                detail="Conservative",
                seed=5,
                steps=24,
                num_latents=16_384,
                octree_resolution=512,
                decode_chunk_size=4096,
                low_vram="auto",
                worker_ref=nodes.ULTRASHAPE_SOURCE_REF,
                checkpoint_ref="checkpoint-ref",
                dinov2_ref="dinov2-ref",
                transform_schema=nodes.TRANSFORM_SCHEMA,
            )
            geometry = cache.cache_path(root / "cache", "ultrashape", key, "geometry.glb")
            write_glb(geometry, volumetric=True)
            write_transform_metadata(geometry.parent / "transform.json")
            worker.write_geometry_cache_record(geometry.parent, key)
            with mock.patch.dict(
                "os.environ", {"COMFYCOLAB_3D_CACHE": str(root / "cache")}
            ), mock.patch.object(
                nodes, "_load_artifact_provisioner", return_value=artifacts
            ), mock.patch.object(
                nodes, "canonical_glb_geometry_digest", return_value="source-geometry"
            ), mock.patch.object(
                nodes, "build_ultrashape_cached_geometry_graph", return_value=sentinel
            ) as cached_graph, mock.patch.object(
                nodes, "build_ultrashape_graph", side_effect=AssertionError("model graph expanded")
            ):
                result = nodes.ComfyColabUltraShapeRefine.execute(
                    source, image, detail="Conservative", seed=5, retexture=False
                )
            self.assertIs(result, sentinel)
            cached_graph.assert_called_once_with(str(geometry), target_face_count=500_000)

    def test_ultrashape_rejects_downstream_invalid_seed_and_postprocess_overrides(self):
        load_package()
        nodes = importlib.import_module("comfycolab_3d_test.nodes")
        with self.assertRaisesRegex(ValueError, "2147483647"):
            nodes.ComfyColabUltraShapeRefine.execute(
                "model.glb", "image", seed=2**31
            )
        with self.assertRaisesRegex(ValueError, "at least 1000"):
            nodes.ComfyColabUltraShapeRefine.execute(
                "model.glb", "image", target_face_count=999
            )
        with self.assertRaisesRegex(ValueError, "at least 512"):
            nodes.ComfyColabUltraShapeRefine.execute(
                "model.glb", "image", texture_size=511
            )

    def test_ultrashape_corrupt_cache_regenerates_and_cleans_input_workdir(self):
        load_package()
        nodes = importlib.import_module("comfycolab_3d_test.nodes")
        with tempfile.TemporaryDirectory() as directory:
            cache_file = Path(directory) / "ultrashape" / ("a" * 64) / "geometry.glb"
            cache_file.parent.mkdir(parents=True)
            cache_file.write_bytes(b"corrupt")
            observed = {}

            artifacts = types.SimpleNamespace(
                ULTRASHAPE_REVISION="checkpoint-ref",
                DINOV2_REVISION="dinov2-ref",
                ensure_ultrashape_artifacts=lambda root, progress: types.SimpleNamespace(
                    checkpoint=Path(directory) / "checkpoint.pt",
                    dinov2_dir=Path(directory) / "dinov2",
                ),
            )

            def copy_input(_model, destination):
                write_glb(Path(destination), volumetric=True)
                return Path(destination)

            def run_worker(command, **_kwargs):
                observed["command"] = command
                observed["workdir"] = Path(command.input_mesh).parent
                self.assertTrue(cache_file.exists(), "refresh must preserve the old cache until success")
                write_glb(Path(command.output_mesh), volumetric=True)
                write_transform_metadata(Path(command.metadata_output))
                return {
                    "status": "ok",
                    "output_mesh": command.output_mesh,
                    "metadata_output": command.metadata_output,
                    "settings": {
                        "steps": command.steps,
                        "num_latents": command.num_latents,
                        "octree_resolution": command.octree_resolution,
                        "decode_chunk_size": command.decode_chunk_size,
                        "seed": command.seed,
                        "low_vram": command.low_vram,
                    },
                }

            model_management = types.SimpleNamespace(throw_exception_if_processing_interrupted=lambda: None)
            utils = types.SimpleNamespace(ProgressBar=lambda _total: types.SimpleNamespace(update_absolute=lambda *_args: None))
            real_import = importlib.import_module

            def fake_import(name):
                if name == "comfy.model_management":
                    return model_management
                if name == "comfy.utils":
                    return utils
                return real_import(name)

            with mock.patch.object(nodes, "_load_artifact_provisioner", return_value=artifacts), mock.patch.object(
                nodes, "copy_file3d_to", side_effect=copy_input
            ), mock.patch.object(nodes, "_save_reference_image", side_effect=lambda _image, _mask, path: Path(path).write_bytes(b"png")), mock.patch.object(
                nodes, "canonical_glb_geometry_digest", return_value="geometry-digest"
            ), mock.patch.object(nodes, "cache_path", return_value=cache_file), mock.patch.object(
                nodes, "run_ultrashape_worker", side_effect=run_worker
            ), mock.patch.object(nodes.importlib, "import_module", side_effect=fake_import):
                result = nodes.ComfyColab3DUltraShapeWorker.execute(
                    "input.glb", object(), object(), "Detailed", 3, 24, 16_384, 1024, 4096, "auto", "Use cache",
                )

            self.assertEqual(result.values, (str(cache_file),))
            self.assertTrue(cache_file.is_file())
            self.assertFalse(observed["workdir"].exists())
            self.assertEqual(observed["command"].source_dir, "/content/UltraShape-1.0")
            self.assertTrue(observed["command"].python.endswith("/.ce/.pixi/envs/trellis2-nodes/bin/python"))
            self.assertEqual(Path(observed["command"].metadata_output).name, "transform.json")

    def test_ultrashape_worker_rejects_planar_input_before_artifact_loading(self):
        load_package()
        nodes = importlib.import_module("comfycolab_3d_test.nodes")

        def copy_planar(_model, destination):
            write_glb(Path(destination))
            return Path(destination)

        with mock.patch.object(
            nodes, "copy_file3d_to", side_effect=copy_planar
        ), mock.patch.object(
            nodes,
            "_load_artifact_provisioner",
            side_effect=AssertionError("planar input reached artifact loading"),
        ):
            with self.assertRaisesRegex(ValueError, "UltraShape worker input GLB.*PCA rank=2"):
                nodes.ComfyColab3DUltraShapeWorker.execute(
                    "input.glb", object(), object(), "Detailed", 3, 24,
                    16_384, 512, 4096, "auto", "Disable cache",
                )

    def test_worker_contract_parses_progress_and_atomically_promotes_output(self):
        load_package()
        worker = importlib.import_module("comfycolab_3d_test.worker")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result.glb"
            metadata = Path(directory) / "result.json"
            command = worker.UltraShapeCommand(
                python="cached-python", worker_script="worker_main.py", source_dir="source", checkpoint="checkpoint",
                dinov2_dir="dinov2", input_mesh="input.glb", reference_image="image.png",
                output_mesh=str(destination), metadata_output=str(metadata), steps=24, num_latents=16384,
                octree_resolution=1024, decode_chunk_size=1024, seed=9, low_vram="auto",
            )
            observed = []

            class FakeProcess:
                pid = 99999

                def __init__(self, argv, **kwargs):
                    self.argv, self.kwargs = argv, kwargs
                    output = Path(argv[argv.index("--output-mesh") + 1])
                    metadata_output = Path(argv[argv.index("--metadata-output") + 1])
                    write_glb(output, volumetric=True)
                    write_transform_metadata(metadata_output)
                    self.stdout = stdio.StringIO(
                        'COMFYCOLAB_PROGRESS={"stage":"decode","current":1,"total":2}\n'
                        f'COMFYCOLAB_RESULT={{"status":"ok","output_mesh":"{output}",'
                        f'"metadata_output":"{metadata_output}"}}\n'
                    )

                def poll(self):
                    return 0

                def wait(self, timeout=None):
                    return 0

            result = worker.run_ultrashape_worker(
                command, on_progress=observed.append, popen_factory=FakeProcess, poll_interval=0.001,
            )
            self.assertTrue(destination.exists())
            self.assertEqual(result["status"], "ok")
            self.assertEqual(observed[0]["stage"], "decode")
            self.assertIn("--source-dir", command.argv())
            self.assertIn("--dinov2-dir", command.argv())

    def test_worker_failure_removes_metadata_and_partial_outputs(self):
        load_package()
        worker = importlib.import_module("comfycolab_3d_test.worker")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result.glb"
            metadata = Path(directory) / "result.json"
            command = worker.UltraShapeCommand(
                python="cached-python", worker_script="worker_main.py", source_dir="source", checkpoint="checkpoint",
                dinov2_dir="dinov2", input_mesh="input.glb", reference_image="image.png",
                output_mesh=str(destination), metadata_output=str(metadata), steps=24, num_latents=16384,
                octree_resolution=1024, decode_chunk_size=4096, seed=9, low_vram="auto",
            )

            class FailedProcess:
                pid = 99999

                def __init__(self, argv, **kwargs):
                    output = Path(argv[argv.index("--output-mesh") + 1])
                    output.write_bytes(b"partial")
                    destination.with_suffix(".glb.partial").write_bytes(b"worker partial")
                    metadata.write_text("partial metadata")
                    metadata.with_suffix(".json.partial").write_text("partial metadata")
                    self.stdout = stdio.StringIO("fatal worker error\n")

                def poll(self):
                    return 1

                def wait(self, timeout=None):
                    return 1

            with self.assertRaises(RuntimeError):
                worker.run_ultrashape_worker(command, popen_factory=FailedProcess, poll_interval=0.001)
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix(".glb.partial").exists())
            self.assertFalse(metadata.exists())
            self.assertFalse(metadata.with_suffix(".json.partial").exists())

    def test_worker_translates_empty_surface_result_to_domain_error(self):
        load_package()
        worker = importlib.import_module("comfycolab_3d_test.worker")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result.glb"
            metadata = Path(directory) / "result.json"
            command = worker.UltraShapeCommand(
                python="cached-python", worker_script="worker_main.py", source_dir="source",
                checkpoint="checkpoint", dinov2_dir="dinov2", input_mesh="input.glb",
                reference_image="image.png", output_mesh=str(destination),
                metadata_output=str(metadata), steps=24, num_latents=16384,
                octree_resolution=1024, decode_chunk_size=4096, seed=9, low_vram="auto",
            )

            class EmptySurfaceProcess:
                pid = 99999

                def __init__(self, _argv, **_kwargs):
                    self.stdout = stdio.StringIO(
                        'COMFYCOLAB_RESULT={"status":"error",'
                        '"error_type":"NoDecodableSurface",'
                        '"error_code":"no_decodable_surface",'
                        '"octree_resolution":1024,"octree_depth":5,'
                        '"preceding_active_points":17,"seed":9}\n'
                    )

                def poll(self):
                    return 1

                def wait(self, timeout=None):
                    return 1

            with self.assertRaisesRegex(
                worker.UltraShapeNoDecodableSurfaceError,
                "requested_resolution=1024.*decode_stage_resolution=5.*"
                "preceding_active_points=17.*seed=9.*conservative 512",
            ):
                worker.run_ultrashape_worker(
                    command,
                    popen_factory=EmptySurfaceProcess,
                    poll_interval=0.001,
                )
            self.assertFalse(destination.exists())
            self.assertFalse(metadata.exists())

    def test_worker_rejects_zero_exit_without_machine_result(self):
        load_package()
        worker = importlib.import_module("comfycolab_3d_test.worker")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "result.glb"
            metadata = Path(directory) / "transform.json"
            command = worker.UltraShapeCommand(
                python="cached-python", worker_script="worker_main.py", source_dir="source",
                checkpoint="checkpoint", dinov2_dir="dinov2", input_mesh="input.glb",
                reference_image="image.png", output_mesh=str(destination),
                metadata_output=str(metadata), steps=12, num_latents=8192,
                octree_resolution=384, decode_chunk_size=2048, seed=0, low_vram="auto",
            )

            class MisleadingProcess:
                pid = 99999

                def __init__(self, argv, **kwargs):
                    write_glb(Path(argv[argv.index("--output-mesh") + 1]), volumetric=True)
                    self.stdout = stdio.StringIO("looks successful but has no result sentinel\n")

                def poll(self):
                    return 0

                def wait(self, timeout=None):
                    return 0

            with self.assertRaisesRegex(RuntimeError, "without COMFYCOLAB_RESULT"):
                worker.run_ultrashape_worker(
                    command, popen_factory=MisleadingProcess, poll_interval=0.001
                )
            self.assertFalse(destination.exists())

    def test_worker_cancellation_terminates_process_group_and_cleans_partials(self):
        load_package()
        worker = importlib.import_module("comfycolab_3d_test.worker")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "result.glb"
            metadata = root / "transform.json"
            command = worker.UltraShapeCommand(
                python="cached-python", worker_script="worker_main.py", source_dir="source",
                checkpoint="checkpoint", dinov2_dir="dinov2", input_mesh="input.glb",
                reference_image="image.png", output_mesh=str(destination),
                metadata_output=str(metadata), steps=24, num_latents=16384,
                octree_resolution=1024, decode_chunk_size=4096, seed=9, low_vram="auto",
            )

            class RunningProcess:
                pid = 24680

                def __init__(self, argv, **kwargs):
                    self.return_code = None
                    output = Path(argv[argv.index("--output-mesh") + 1])
                    output.write_bytes(b"partial")
                    metadata.write_text("partial")
                    self.stdout = stdio.StringIO("")

                def poll(self):
                    return self.return_code

                def wait(self, timeout=None):
                    self.return_code = -15
                    return self.return_code

            with mock.patch.object(worker.os, "killpg") as killpg:
                with self.assertRaisesRegex(InterruptedError, "cancelled"):
                    worker.run_ultrashape_worker(
                        command,
                        is_cancelled=lambda: True,
                        popen_factory=RunningProcess,
                        poll_interval=0.001,
                    )
            killpg.assert_called_once_with(24680, worker.signal.SIGTERM)
            self.assertFalse(destination.exists())
            self.assertFalse(metadata.exists())


if __name__ == "__main__":
    unittest.main()
