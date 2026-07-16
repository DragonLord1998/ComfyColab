from __future__ import annotations

import asyncio
import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "custom_nodes" / "ComfyColab-Triposplat"

PUBLIC_NODE_ID = "ComfyColabTripoSplatImageToGaussianSplat"
DISPLAY_NAME = "ComfyColab TripoSplat \u2014 Image to Gaussian Splat"

REQUIRED_NATIVE_NODES = {
    "LoadBackgroundRemovalModel",
    "RemoveBackground",
    "TripoSplatPreprocessImage",
    "CLIPVisionLoader",
    "VAELoader",
    "TripoSplatConditioning",
    "UNETLoader",
    "TripoSplatSamplingPreview",
    "KSampler",
    "VAEDecodeTripoSplat",
    "SplatToFile3D",
}


def load_package():
    name = "comfycolab_triposplat_test"
    for module in list(sys.modules):
        if module == name or module.startswith(name + "."):
            del sys.modules[module]
    spec = importlib.util.spec_from_file_location(
        name,
        PACKAGE_DIR / "__init__.py",
        submodule_search_locations=[str(PACKAGE_DIR)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    assert spec.loader
    spec.loader.exec_module(package)
    return package


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

    Image = Combo = Int = Float = Boolean = Mask = PortFactory()
    Splat = PortFactory("SPLAT")
    File3D = PortFactory("FILE_3D")
    File3DSplatAny = PortFactory("FILE_3D_SPLAT_ANY")
    AnyType = PortFactory("ANY")

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
        self.node_id = node_id
        self.index = index

    def __eq__(self, other):
        return (
            isinstance(other, Link)
            and self.node_id == other.node_id
            and self.index == other.index
        )

    def __repr__(self):
        return f"Link({self.node_id!r}, {self.index!r})"


class GraphNode:
    def __init__(self, index, class_type, inputs):
        self.index = index
        self.class_type = class_type
        self.inputs = inputs
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
        items = []
        for node in self.nodes:
            item = {"class_type": node.class_type, "inputs": node.inputs}
            if node.override_display_id is not None:
                item["override_display_id"] = node.override_display_id
            items.append(item)
        return items


class TriposplatNodePackTests(unittest.TestCase):
    def setUp(self):
        self.saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "comfy_api",
                "comfy_api.latest",
                "comfy_execution",
                "comfy_execution.graph_utils",
                "nodes",
                "folder_paths",
            )
        }
        latest = types.ModuleType("comfy_api.latest")
        latest.io = FakeIO
        latest.ComfyExtension = type("ComfyExtension", (), {})
        api = types.ModuleType("comfy_api")
        api.latest = latest
        execution = types.ModuleType("comfy_execution")
        graph_utils = types.ModuleType("comfy_execution.graph_utils")
        graph_utils.GraphBuilder = GraphBuilder
        comfy_nodes = types.ModuleType("nodes")
        comfy_nodes.NODE_CLASS_MAPPINGS = {
            node_id: object for node_id in REQUIRED_NATIVE_NODES
        }
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_folder_paths = lambda key: [f"/tmp/comfy-models/{key}"]
        sys.modules.update(
            {
                "comfy_api": api,
                "comfy_api.latest": latest,
                "comfy_execution": execution,
                "comfy_execution.graph_utils": graph_utils,
                "nodes": comfy_nodes,
                "folder_paths": folder_paths,
            }
        )

    def tearDown(self):
        for name, module in self.saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def _nodes_module(self):
        load_package()
        return importlib.import_module("comfycolab_triposplat_test.nodes")

    def _graph_module(self):
        load_package()
        return importlib.import_module("comfycolab_triposplat_test.graph")

    def _patch_model_downloads(self, nodes_module=None, graph_module=None):
        patches = []
        for module in (nodes_module, graph_module):
            if module is None:
                continue
            for name in (
                "ensure_triposplat_models",
                "ensure_models",
                "download_triposplat_models",
                "ensure_model_assets",
            ):
                if hasattr(module, name):
                    patcher = mock.patch.object(module, name, return_value=None)
                    patches.append(patcher)
                    patcher.start()
        self.addCleanup(lambda: [patcher.stop() for patcher in reversed(patches)])

    def test_import_is_lazy_and_exposes_exactly_one_public_facade(self):
        before = set(sys.modules)
        package = load_package()
        imported = set(sys.modules) - before
        self.assertFalse({"torch", "numpy", "PIL", "diffusers", "transformers"} & imported)

        extension = asyncio.run(package.comfy_entrypoint())
        node_classes = asyncio.run(extension.get_node_list())
        schemas = [node.define_schema() for node in node_classes]
        public = [
            schema.node_id
            for schema in schemas
            if not getattr(schema, "is_dev_only", False)
        ]

        self.assertEqual(public, [PUBLIC_NODE_ID])
        self.assertEqual(len(set(public)), 1)

    def test_facade_schema_defaults_outputs_and_input_order(self):
        nodes = self._nodes_module()
        schema = nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID].define_schema()
        inputs = {item["name"]: item for item in schema.inputs}

        self.assertEqual(schema.node_id, PUBLIC_NODE_ID)
        self.assertEqual(schema.display_name, DISPLAY_NAME)
        self.assertEqual(schema.category, "ComfyColab/3D")
        self.assertTrue(schema.enable_expand)
        self.assertEqual(
            [item["name"] for item in schema.outputs],
            ["splat", "model_3d"],
        )
        self.assertEqual(schema.outputs[0]["io_type"], "SPLAT")
        self.assertEqual(schema.outputs[1]["io_type"], "FILE_3D_SPLAT_ANY")
        self.assertEqual(
            list(inputs),
            [
                "image",
                "quality",
                "seed",
                "remove_background",
                "sampling_steps",
                "guidance_scale",
                "enable_sampling_preview",
                "output_format",
            ],
        )
        self.assertEqual(
            inputs["quality"]["options"],
            ["Fast \u2014 65K", "Balanced \u2014 131K", "Quality \u2014 262K"],
        )
        self.assertEqual(inputs["quality"]["default"], "Quality \u2014 262K")
        self.assertEqual(inputs["seed"]["default"], 0)
        self.assertEqual(inputs["seed"]["min"], 0)
        self.assertEqual(inputs["seed"]["max"], (2**63) - 1)
        self.assertIs(inputs["remove_background"]["default"], True)
        self.assertEqual(inputs["sampling_steps"]["default"], 20)
        self.assertEqual(inputs["sampling_steps"]["min"], 1)
        self.assertEqual(inputs["sampling_steps"]["max"], 50)
        self.assertEqual(inputs["guidance_scale"]["default"], 3.0)
        self.assertEqual(inputs["guidance_scale"]["min"], 1.0)
        self.assertEqual(inputs["guidance_scale"]["max"], 10.0)
        self.assertIs(inputs["enable_sampling_preview"]["default"], True)
        self.assertEqual(inputs["output_format"]["options"], ["ply", "spz", "ksplat"])
        self.assertEqual(inputs["output_format"]["default"], "ply")

    def test_facade_expands_to_exact_native_graph_with_background_and_preview(self):
        nodes = self._nodes_module()
        graph = importlib.import_module("comfycolab_triposplat_test.graph")
        self._patch_model_downloads(nodes, graph)

        result = nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID].execute(
            "image",
            quality="Quality \u2014 262K",
            seed=7,
            remove_background=True,
            sampling_steps=20,
            guidance_scale=3.0,
            enable_sampling_preview=True,
            output_format="ply",
        )
        node_ids = [item["class_type"] for item in result.expand]

        self.assertEqual(
            node_ids,
            [
                "LoadBackgroundRemovalModel",
                "RemoveBackground",
                "TripoSplatPreprocessImage",
                "CLIPVisionLoader",
                "VAELoader",
                "TripoSplatConditioning",
                "UNETLoader",
                "VAELoader",
                "TripoSplatSamplingPreview",
                "KSampler",
                "VAEDecodeTripoSplat",
                "SplatToFile3D",
            ],
        )
        remove_background = result.expand[1]["inputs"]
        preprocess = result.expand[2]["inputs"]
        sampler = result.expand[9]["inputs"]
        decode = result.expand[10]["inputs"]
        export = result.expand[11]["inputs"]

        self.assertEqual(remove_background["bg_removal_model"], Link(0, 0))
        self.assertEqual(remove_background["image"], "image")
        self.assertEqual(preprocess["image"], "image")
        self.assertEqual(preprocess["mask"], Link(1, 0))
        self.assertEqual(sampler["seed"], 7)
        self.assertEqual(sampler["steps"], 20)
        self.assertEqual(sampler["cfg"], 3.0)
        self.assertEqual(decode["num_gaussians"], 262_144)
        self.assertEqual(export["format"], "ply")
        self.assertEqual(result.values, (Link(10, 0), Link(11, 0)))

    def test_background_off_uses_opaque_mask_and_preview_can_be_disabled(self):
        nodes = self._nodes_module()
        graph = importlib.import_module("comfycolab_triposplat_test.graph")
        self._patch_model_downloads(nodes, graph)

        result = nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID].execute(
            "image",
            quality="Fast \u2014 65K",
            seed=0,
            remove_background=False,
            sampling_steps=8,
            guidance_scale=2.5,
            enable_sampling_preview=False,
            output_format="spz",
        )
        node_ids = [item["class_type"] for item in result.expand]

        self.assertEqual(
            node_ids,
            [
                "ComfyColabTripoSplatOpaqueMask",
                "TripoSplatPreprocessImage",
                "CLIPVisionLoader",
                "VAELoader",
                "TripoSplatConditioning",
                "UNETLoader",
                "VAELoader",
                "KSampler",
                "VAEDecodeTripoSplat",
                "SplatToFile3D",
            ],
        )
        self.assertNotIn("TripoSplatSamplingPreview", node_ids)
        self.assertEqual(result.expand[8]["inputs"]["num_gaussians"], 65_536)
        self.assertEqual(result.expand[-1]["inputs"]["format"], "spz")
        self.assertEqual(result.values, (Link(8, 0), Link(9, 0)))

    def test_quality_presets_and_output_formats_are_contract_covered(self):
        nodes = self._nodes_module()
        graph = importlib.import_module("comfycolab_triposplat_test.graph")
        self._patch_model_downloads(nodes, graph)

        expectations = {
            "Fast \u2014 65K": 65_536,
            "Balanced \u2014 131K": 131_072,
            "Quality \u2014 262K": 262_144,
        }
        for index, (quality, expected_gaussians) in enumerate(expectations.items()):
            with self.subTest(quality=quality):
                result = nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID].execute(
                    "image",
                    quality=quality,
                    seed=index,
                    output_format="ksplat",
                )
                decode = next(
                    item
                    for item in result.expand
                    if item["class_type"] == "VAEDecodeTripoSplat"
                )
                export = result.expand[-1]
                self.assertEqual(decode["inputs"]["num_gaussians"], expected_gaussians)
                self.assertEqual(export["class_type"], "SplatToFile3D")
                self.assertEqual(export["inputs"]["format"], "ksplat")

    def test_missing_upstream_nodes_raise_refresh_diagnostic(self):
        nodes = self._nodes_module()
        graph = importlib.import_module("comfycolab_triposplat_test.graph")
        self._patch_model_downloads(nodes, graph)
        sys.modules["nodes"].NODE_CLASS_MAPPINGS = {
            node_id: object
            for node_id in REQUIRED_NATIVE_NODES
            if node_id != "VAEDecodeTripoSplat"
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "pinned commit.*VAEDecodeTripoSplat.*latest ComfyColab bootstrap",
        ):
            nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID].execute("image")

    def test_no_real_weights_are_required_for_local_contract_execution(self):
        nodes = self._nodes_module()
        graph = importlib.import_module("comfycolab_triposplat_test.graph")
        self._patch_model_downloads(nodes, graph)

        result = nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID].execute(
            "image",
            quality="Quality \u2014 262K",
        )

        self.assertTrue(result.expand)
        self.assertFalse(Path("/tmp/comfy-models").exists())


if __name__ == "__main__":
    unittest.main()
