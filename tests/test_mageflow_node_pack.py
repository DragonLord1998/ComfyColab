from __future__ import annotations

import asyncio
import ast
import contextlib
import inspect
import importlib
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "custom_nodes" / "ComfyColab-MageFlow"

PUBLIC_NODE_IDS = [
    "ComfyColabMageFlow",
    "ComfyColabMageFlowTurbo",
    "ComfyColabMageFlowEdit",
    "ComfyColabMageFlowEditTurbo",
    "ComfyColabMageFlowEmptyLatent",
]
FLOW_NODE_IDS = set(PUBLIC_NODE_IDS) - {"ComfyColabMageFlowEmptyLatent"}
EDIT_NODE_IDS = {"ComfyColabMageFlowEdit", "ComfyColabMageFlowEditTurbo"}
FORBIDDEN_SCHEMA_INPUTS = {
    "content_screening",
    "prompt_screening",
    "screen_prompt",
    "screen_image",
    "safety_checker",
    "gaussian_shading",
    "watermark",
    "watermark_key",
}
FORBIDDEN_NODE_TOKENS = (
    "Screen",
    "Safety",
    "Moderation",
    "Watermark",
    "GaussianShading",
    "Gaussian-Shading",
)


def load_package():
    name = "comfycolab_mageflow_test"
    for module_name in list(sys.modules):
        if module_name == name or module_name.startswith(name + "."):
            del sys.modules[module_name]
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
        return {
            "direction": "input",
            "name": name,
            "io_type": self.io_type,
            **kwargs,
        }

    def Output(self, name=None, **kwargs):
        return {
            "direction": "output",
            "name": name,
            "io_type": self.io_type,
            **kwargs,
        }


class FakeIO:
    class ComfyNode:
        pass

    String = PortFactory("STRING")
    Combo = PortFactory("COMBO")
    Int = PortFactory("INT")
    Float = PortFactory("FLOAT")
    Boolean = PortFactory("BOOLEAN")
    Image = PortFactory("IMAGE")
    Mask = PortFactory("MASK")
    Latent = PortFactory("LATENT")
    Conditioning = PortFactory("CONDITIONING")
    Model = PortFactory("MODEL")
    Clip = PortFactory("CLIP")
    Vae = PortFactory("VAE")
    AnyType = PortFactory("ANY")

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
        result = []
        for node in self.nodes:
            item = {"class_type": node.class_type, "inputs": node.inputs}
            if node.override_display_id is not None:
                item["override_display_id"] = node.override_display_id
            result.append(item)
        return result


REQUIRED_NATIVE_NODES = {
    "BasicGuider",
    "CFGGuider",
    "CLIPLoader",
    "CLIPTextEncode",
    "ConditioningZeroOut",
    "DifferentialDiffusion",
    "DualCLIPLoader",
    "EmptySD3LatentImage",
    "FluxGuidance",
    "ImageScale",
    "KSampler",
    "KSamplerSelect",
    "LoadImage",
    "ModelSamplingFlux",
    "ModelSamplingSD3",
    "RandomNoise",
    "SamplerCustomAdvanced",
    "SetLatentNoiseMask",
    "UNETLoader",
    "VAEDecode",
    "VAEEncode",
    "VAELoader",
}


class MageFlowNodePackTests(unittest.TestCase):
    def setUp(self):
        self.saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "comfy_api",
                "comfy_api.latest",
                "comfy_execution",
                "comfy_execution.graph_utils",
                "folder_paths",
                "nodes",
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
        folder_paths.get_filename_list = lambda _key: []
        folder_paths.get_folder_paths = lambda key: [f"/tmp/comfy-models/{key}"]
        folder_paths.get_full_path = (
            lambda key, filename: f"/tmp/comfy-models/{key}/{filename}"
        )
        sys.modules.update(
            {
                "comfy_api": api,
                "comfy_api.latest": latest,
                "comfy_execution": execution,
                "comfy_execution.graph_utils": graph_utils,
                "folder_paths": folder_paths,
                "nodes": comfy_nodes,
            }
        )

    def tearDown(self):
        for name, module in self.saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def _modules(self):
        package = load_package()
        nodes = importlib.import_module("comfycolab_mageflow_test.nodes")
        return package, nodes

    def _module_candidates(self):
        package, nodes = self._modules()
        modules = [nodes]
        for module_name in (
            "comfycolab_mageflow_test.graph",
            "comfycolab_mageflow_test.models",
            "comfycolab_mageflow_test.download",
        ):
            try:
                modules.append(importlib.import_module(module_name))
            except ModuleNotFoundError:
                pass
        return package, nodes, modules

    @contextlib.contextmanager
    def _mock_model_downloads(self, modules):
        with contextlib.ExitStack() as stack:
            for module in modules:
                for name in (
                    "ensure_mageflow_model_assets",
                    "ensure_model_assets",
                    "ensure_runtime_assets",
                    "download_mageflow_models",
                ):
                    if hasattr(module, name):
                        stack.enter_context(mock.patch.object(module, name, return_value={}))
            yield

    def test_import_is_lazy_and_exposes_four_facades_plus_native_empty_latent(self):
        before = set(sys.modules)
        package = load_package()
        imported = set(sys.modules) - before
        self.assertFalse(
            {"torch", "numpy", "PIL", "diffusers", "transformers"} & imported
        )

        extension = asyncio.run(package.comfy_entrypoint())
        node_classes = asyncio.run(extension.get_node_list())
        schemas = [node.define_schema() for node in node_classes]
        public = [
            schema.node_id
            for schema in schemas
            if not getattr(schema, "is_dev_only", False)
        ]

        self.assertEqual(public, PUBLIC_NODE_IDS)
        self.assertEqual(len(set(public)), 5)
        self.assertFalse(any("Base" in node_id for node_id in public))

    def test_public_schemas_have_no_screening_or_watermark_controls(self):
        _, nodes = self._modules()
        for node_id in FLOW_NODE_IDS:
            with self.subTest(node_id=node_id):
                schema = nodes.NODE_CLASS_MAPPINGS[node_id].define_schema()
                input_names = [item["name"] for item in schema.inputs]
                normalized = {name.lower() for name in input_names}

                self.assertEqual(schema.category, "ComfyColab/Image")
                self.assertTrue(schema.enable_expand)
                self.assertIn("prompt", normalized)
                self.assertIn("seed", normalized)
                self.assertFalse(FORBIDDEN_SCHEMA_INPUTS & normalized)
                self.assertFalse(any("screen" in name for name in normalized))
                self.assertFalse(any("watermark" in name for name in normalized))
                self.assertFalse(any("gaussian_shading" in name for name in normalized))
                if node_id in EDIT_NODE_IDS:
                    self.assertIn("image", normalized)
                else:
                    self.assertNotIn("image", normalized)

                outputs = {item["io_type"] for item in schema.outputs}
                self.assertEqual(outputs, {"IMAGE", "MODEL", "CLIP", "VAE"})

    def test_public_mappings_do_not_expose_base_or_auxiliary_nodes(self):
        _, nodes = self._modules()
        public_keys = [
            key
            for key, cls in nodes.NODE_CLASS_MAPPINGS.items()
            if not getattr(cls.define_schema(), "is_dev_only", False)
        ]

        self.assertEqual(public_keys, PUBLIC_NODE_IDS)
        self.assertFalse(any("Base" in key for key in nodes.NODE_CLASS_MAPPINGS))

    def test_internal_mage_vae_encoder_is_dev_only_and_returns_latent(self):
        _, nodes = self._modules()
        encoder = nodes.NODE_CLASS_MAPPINGS["ComfyColabMageVAEEncode"]
        schema = encoder.define_schema()
        self.assertTrue(schema.is_dev_only)
        self.assertEqual([item["io_type"] for item in schema.outputs], ["LATENT"])
        self.assertEqual(
            [item["name"] for item in schema.inputs],
            ["image", "vae_name", "tile_size", "tile_overlap", "keep_worker_loaded"],
        )

    def test_component_loader_exposes_standard_sampler_types(self):
        _, nodes = self._modules()
        loader = nodes.NODE_CLASS_MAPPINGS["ComfyColabMageFlowComponents"]
        schema = loader.define_schema()
        self.assertTrue(schema.is_dev_only)
        self.assertEqual(
            [item["io_type"] for item in schema.outputs],
            ["MODEL", "CLIP", "VAE"],
        )
        self.assertEqual(
            [item["name"] for item in schema.inputs],
            ["image", "variant", "keep_worker_loaded"],
        )
        native = types.ModuleType("comfycolab_mageflow_test.native")
        native.build_components = mock.Mock(
            return_value=("MODEL_OBJECT", "CLIP_OBJECT", "VAE_OBJECT")
        )
        with mock.patch.dict(
            sys.modules,
            {"comfycolab_mageflow_test.native": native},
        ), mock.patch.object(
            nodes,
            "_source_dir",
            return_value=Path("/source"),
        ), mock.patch.object(
            nodes,
            "_worker_script",
            return_value=Path("/worker.py"),
        ), mock.patch.object(
            nodes,
            "_worker_site_packages",
            return_value="/packages",
        ), mock.patch.object(
            nodes,
            "_runtime_root",
            return_value=Path("/runtime"),
        ):
            result = loader.execute(
                image="reference",
                variant="flow",
                keep_worker_loaded=True,
            )
        self.assertEqual(
            result.values,
            ("MODEL_OBJECT", "CLIP_OBJECT", "VAE_OBJECT"),
        )
        self.assertEqual(
            native.build_components.call_args.kwargs["reference_image"],
            "reference",
        )

    def test_native_cache_keys_do_not_repeat_the_variant_argument(self):
        tree = ast.parse(
            (PACKAGE_DIR / "native.py").read_text(encoding="utf-8")
        )
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_request_key"
        ]
        self.assertGreaterEqual(len(calls), 2)
        for call in calls:
            self.assertNotIn(
                "variant",
                {keyword.arg for keyword in call.keywords},
            )

    def test_empty_latent_uses_mage_128_channel_16x_contract(self):
        _, nodes = self._modules()
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is required")
        latent = nodes.ComfyColabMageFlowEmptyLatent.execute(
            width=512,
            height=768,
            batch_size=2,
        ).values[0]["samples"]
        self.assertEqual(tuple(latent.shape), (2, 128, 48, 32))

    def test_public_facades_expand_with_seeded_random_noise_and_no_policy_nodes(self):
        _, nodes, modules = self._module_candidates()
        with self._mock_model_downloads(modules):
            for node_id in FLOW_NODE_IDS:
                with self.subTest(node_id=node_id):
                    facade = nodes.NODE_CLASS_MAPPINGS[node_id]
                    signature = inspect.signature(facade.execute)
                    defaults = {
                        "prompt": "A compact studio product render on a clean background.",
                        "image": "image",
                        "width": 1024,
                        "height": 1024,
                        "steps": 4 if "Turbo" in node_id else 20,
                        "guidance": 1.5 if "Turbo" in node_id else 4.0,
                        "edit_strength": 0.65,
                        "seed": 123,
                        "force_redownload": False,
                    }
                    kwargs = {
                        name: value
                        for name, value in defaults.items()
                        if name in signature.parameters
                    }
                    result = facade.execute(**kwargs)
                    expanded = getattr(result, "expand", result)
                    node_types = [
                        item.get("class_type") or item.get("type")
                        for item in expanded
                        if isinstance(item, dict)
                    ]
                    joined = " ".join(str(node_type) for node_type in node_types)

                    self.assertIn("RandomNoise", node_types)
                    self.assertIn("ComfyColabMageFlowComponents", node_types)
                    for token in FORBIDDEN_NODE_TOKENS:
                        self.assertNotIn(token, joined)

                    random_noise = next(
                        item
                        for item in expanded
                        if isinstance(item, dict)
                        and (item.get("class_type") or item.get("type")) == "RandomNoise"
                    )
                    self.assertEqual(random_noise["inputs"]["noise_seed"], 123)
                    self.assertEqual(len(result.values), 4)
                    components = next(
                        item
                        for item in expanded
                        if isinstance(item, dict)
                        and (item.get("class_type") or item.get("type"))
                        == "ComfyColabMageFlowComponents"
                    )
                    self.assertEqual(
                        components["inputs"]["image"],
                        "image" if node_id in EDIT_NODE_IDS else None,
                    )

    def test_worker_compatibility_packages_are_isolated_and_cached(self):
        _, nodes = self._modules()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            nodes, "_runtime_root", return_value=Path(directory)
        ), mock.patch.dict(
            nodes.os.environ,
            {
                "COMFYCOLAB_MAGEFLOW_PYTHON": "",
                "COMFYCOLAB_MAGE_FLOW_PYTHON": "",
                "COMFYCOLAB_MAGEFLOW_SITE_PACKAGES": "",
            },
            clear=False,
        ), mock.patch.object(nodes.subprocess, "check_call") as install:
            first = nodes._worker_site_packages()
            second = nodes._worker_site_packages()

        self.assertEqual(first, second)
        self.assertTrue(first.endswith("python-packages"))
        install.assert_called_once()
        argv = install.call_args.args[0]
        self.assertIn("--target", argv)
        self.assertIn("transformers==5.5.0", argv)
        self.assertIn("loguru==0.7.3", argv)

    def test_legacy_path_self_provisions_exact_mage_source_revision(self):
        _, nodes = self._modules()
        with tempfile.TemporaryDirectory() as directory:
            runtime_root = Path(directory) / "runtime"

            def fake_git(argv):
                if argv[1] == "clone":
                    target = Path(argv[-1])
                    (target / ".git").mkdir(parents=True)
                    (target / "mage_flow").mkdir()
                    (target / "mage_flow" / "pipeline.py").write_text("# pinned\n")

            with mock.patch.object(nodes, "_runtime_root", return_value=runtime_root), mock.patch.object(
                nodes, "_repo_root", return_value=Path(directory) / "repo"
            ), mock.patch.dict(
                nodes.os.environ,
                {"COMFYCOLAB_MAGEFLOW_SOURCE": "", "COMFYCOLAB_MAGE_FLOW_SOURCE_DIR": ""},
                clear=False,
            ), mock.patch.object(
                nodes.subprocess, "check_call", side_effect=fake_git
            ) as git, mock.patch.object(
                nodes.subprocess,
                "check_output",
                return_value=nodes.MAGE_FLOW_SOURCE_REF + "\n",
            ):
                source = nodes._source_dir()

        self.assertEqual(source, runtime_root / "source" / "Mage")
        self.assertEqual(git.call_args_list[0].args[0][0:3], ["git", "clone", "--filter=blob:none"])
        self.assertIn(nodes.MAGE_FLOW_SOURCE_REF, git.call_args_list[1].args[0])


if __name__ == "__main__":
    unittest.main()
