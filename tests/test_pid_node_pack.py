from __future__ import annotations

import asyncio
import contextlib
import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "custom_nodes" / "ComfyColab-PiD"

PUBLIC_NODE_ID = "ComfyColabPiDUpscale"
DISPLAY_NAME = "ComfyColab PiD — Image Upscaler"
MAGE_VAE = "Mage-VAE (experimental)"
BACKBONES = ["FLUX.1", "FLUX.2", "Qwen Image", MAGE_VAE]
SCALES = ["4x", "Experimental 16x (tiled)"]
PID_FILENAMES = {
    "FLUX.1": "pid_1.5_flux1_1024_to_4096_4step_bf16.safetensors",
    "FLUX.2": "pid_1.5_flux2_1024_to_4096_4step_bf16.safetensors",
    "Qwen Image": "pid_1.5_qwenimage_1024_to_4096_4step_bf16.safetensors",
    MAGE_VAE: "pid_1.5_flux2_1024_to_4096_4step_bf16.safetensors",
}
VAE_FILENAMES = {
    "FLUX.1": "ae.safetensors",
    "FLUX.2": "flux2-dev-vae.safetensors",
    "Qwen Image": "qwen_image_vae.safetensors",
    MAGE_VAE: "mage-vae.safetensors",
}


def load_package():
    name = "comfycolab_pid_test"
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
    Latent = PortFactory("LATENT")

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

    def out(self, index):
        return Link(self.index, index)


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
        return [
            {"class_type": node.class_type, "inputs": node.inputs}
            for node in self.nodes
        ]


class FakeImage:
    def __init__(self, width=512, height=512):
        self.shape = (1, height, width, 3)


REQUIRED_NATIVE_NODES = {
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "VAEEncode",
    "VAEEncodeTiled",
    "CLIPTextEncode",
    "PiDConditioning",
    "EmptyChromaRadianceLatentImage",
    "KSamplerSelect",
    "ManualSigmas",
    "SamplerCustom",
    "VAEDecode",
    "ContextWindowsManual",
    "ImageScale",
    "ComfyColabMageVAEEncode",
}


class PiDNodePackTests(unittest.TestCase):
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
        folder_paths.get_folder_paths = lambda key: [f"/tmp/comfy-models/{key}"]
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
        nodes = importlib.import_module("comfycolab_pid_test.nodes")
        graph = importlib.import_module("comfycolab_pid_test.graph")
        models = importlib.import_module("comfycolab_pid_test.models")
        return package, nodes, graph, models

    @contextlib.contextmanager
    def _mock_model_downloads(self, modules):
        def ensure(backbone, force_redownload=False):
            return {
                "model": PID_FILENAMES[backbone],
                "text_encoder": "gemma_2_2b_it_elm_bf16.safetensors",
                "vae": VAE_FILENAMES[backbone],
            }

        with contextlib.ExitStack() as stack:
            patched = False
            for module in modules:
                if hasattr(module, "ensure_pid_assets"):
                    stack.enter_context(
                        mock.patch.object(module, "ensure_pid_assets", side_effect=ensure)
                    )
                    patched = True
            self.assertTrue(
                patched,
                "The PiD pack must expose a mockable model-provisioning boundary.",
            )
            yield

    @staticmethod
    def _execute(node, *, backbone="FLUX.1", scale="4x", image=None, **kwargs):
        return node.execute(
            image=image or FakeImage(),
            vae_family=backbone,
            prompt="restore clean fine photographic detail",
            scale=scale,
            seed=123,
            degrade_sigma=0.0,
            tile_size=1536,
            tile_overlap=384,
            accept_nvidia_noncommercial_license=True,
            **kwargs,
        )

    def test_import_is_lazy_and_exposes_one_public_facade_with_exact_schema(self):
        before = set(sys.modules)
        package = load_package()
        imported = set(sys.modules) - before
        self.assertFalse(
            {"torch", "transformers", "diffusers", "numpy", "PIL"} & imported
        )

        extension = asyncio.run(package.comfy_entrypoint())
        node_classes = asyncio.run(extension.get_node_list())
        schemas = [node.define_schema() for node in node_classes]
        public = [
            schema.node_id
            for schema in schemas
            if not getattr(schema, "is_dev_only", False)
        ]
        self.assertEqual(public, [PUBLIC_NODE_ID])

        schema = next(item for item in schemas if item.node_id == PUBLIC_NODE_ID)
        inputs = {item["name"]: item for item in schema.inputs}
        self.assertEqual(schema.display_name, DISPLAY_NAME)
        self.assertEqual(schema.category, "ComfyColab/Image")
        self.assertTrue(schema.enable_expand)
        self.assertEqual([item["name"] for item in schema.outputs], ["image"])
        self.assertEqual([item["io_type"] for item in schema.outputs], ["IMAGE"])
        self.assertEqual(inputs["vae_family"]["options"], BACKBONES)
        self.assertEqual(inputs["vae_family"]["default"], "FLUX.1")
        self.assertEqual(inputs["scale"]["options"], SCALES)
        self.assertEqual(inputs["scale"]["default"], "4x")
        self.assertEqual(inputs["seed"]["default"], 0)
        self.assertEqual(inputs["degrade_sigma"]["default"], 0.0)
        self.assertEqual(inputs["tile_size"]["default"], 1536)
        self.assertEqual(inputs["tile_overlap"]["default"], 384)
        self.assertFalse(inputs["accept_nvidia_noncommercial_license"]["default"])
        self.assertIn("force_redownload", inputs)

    def test_mage_vae_catalog_is_revision_and_digest_pinned(self):
        self._modules()
        catalog = importlib.import_module("comfycolab_pid_test.catalog")
        assets = catalog.selected_assets(MAGE_VAE)
        self.assertEqual(
            assets["model"]["filename"],
            PID_FILENAMES[MAGE_VAE],
        )
        self.assertEqual(assets["vae"]["filename"], "mage-vae.safetensors")
        self.assertIn(catalog.MAGE_FLOW_REVISION, assets["vae"]["url"])
        self.assertEqual(
            assets["vae"]["sha256"],
            "34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0",
        )
        self.assertEqual(assets["vae"]["size_bytes"], 345053056)

    def test_license_gate_blocks_before_any_model_download(self):
        _, nodes, _, _ = self._modules()
        facade = nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID]
        with mock.patch.object(nodes, "ensure_pid_assets") as ensure:
            with self.assertRaises(PermissionError):
                facade.execute(
                    image=FakeImage(),
                    prompt="detail",
                    accept_nvidia_noncommercial_license=False,
                )
        ensure.assert_not_called()

    def test_4x_graph_uses_matching_bf16_pid_gemma_vae_for_each_backbone(self):
        _, nodes, graph, models = self._modules()
        facade = nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID]
        with self._mock_model_downloads((nodes, graph, models)):
            for backbone in BACKBONES:
                with self.subTest(backbone=backbone):
                    result = self._execute(facade, backbone=backbone)
                    expanded = result.expand
                    node_types = [item["class_type"] for item in expanded]
                    self.assertEqual(node_types.count("UNETLoader"), 1)
                    self.assertEqual(node_types.count("CLIPLoader"), 1)
                    self.assertEqual(
                        node_types.count("VAELoader"),
                        1 if backbone == MAGE_VAE else 2,
                    )
                    self.assertEqual(
                        node_types.count("VAEEncode"),
                        0 if backbone == MAGE_VAE else 1,
                    )
                    self.assertEqual(
                        node_types.count("ComfyColabMageVAEEncode"),
                        1 if backbone == MAGE_VAE else 0,
                    )
                    self.assertEqual(node_types.count("VAEEncodeTiled"), 0)
                    self.assertEqual(node_types.count("PiDConditioning"), 1)
                    self.assertEqual(node_types.count("SamplerCustom"), 1)
                    self.assertEqual(node_types.count("ContextWindowsManual"), 0)

                    unet = next(item for item in expanded if item["class_type"] == "UNETLoader")
                    clip = next(item for item in expanded if item["class_type"] == "CLIPLoader")
                    vaes = [
                        item for item in expanded if item["class_type"] == "VAELoader"
                    ]
                    latent = next(
                        item
                        for item in expanded
                        if item["class_type"] == "EmptyChromaRadianceLatentImage"
                    )
                    sampler = next(
                        item for item in expanded if item["class_type"] == "KSamplerSelect"
                    )
                    sigmas = next(item for item in expanded if item["class_type"] == "ManualSigmas")
                    custom = next(
                        item for item in expanded if item["class_type"] == "SamplerCustom"
                    )
                    self.assertEqual(unet["inputs"]["unet_name"], PID_FILENAMES[backbone])
                    self.assertEqual(clip["inputs"]["clip_name"], "gemma_2_2b_it_elm_bf16.safetensors")
                    self.assertEqual(clip["inputs"]["type"], "pixeldit")
                    expected_vaes = {"pixel_space"}
                    if backbone != MAGE_VAE:
                        expected_vaes.add(VAE_FILENAMES[backbone])
                    self.assertEqual(
                        {item["inputs"]["vae_name"] for item in vaes},
                        expected_vaes,
                    )
                    self.assertEqual(latent["inputs"]["width"], 2048)
                    self.assertEqual(latent["inputs"]["height"], 2048)
                    self.assertEqual(sampler["inputs"]["sampler_name"], "lcm")
                    self.assertEqual(sigmas["inputs"]["sigmas"], "0.999,0.866,0.634,0.342,0")
                    self.assertEqual(custom["inputs"]["cfg"], 1.0)
                    self.assertNotIn("noise", custom["inputs"])
                    pid = next(
                        item
                        for item in expanded
                        if item["class_type"] == "PiDConditioning"
                    )
                    self.assertEqual(
                        pid["inputs"]["latent_format"],
                        "qwenimage" if backbone == "Qwen Image" else "flux",
                    )
                    pixel_vae_index = next(
                        index
                        for index, item in enumerate(expanded)
                        if item["class_type"] == "VAELoader"
                        and item["inputs"]["vae_name"] == "pixel_space"
                    )
                    decoded = next(
                        item
                        for item in expanded
                        if item["class_type"] == "VAEDecode"
                    )
                    self.assertEqual(
                        decoded["inputs"]["vae"],
                        Link(pixel_vae_index, 0),
                    )
                    if backbone == MAGE_VAE:
                        mage_encode = next(
                            item
                            for item in expanded
                            if item["class_type"] == "ComfyColabMageVAEEncode"
                        )
                        self.assertEqual(
                            mage_encode["inputs"]["vae_name"],
                            "mage-vae.safetensors",
                        )
                        self.assertEqual(mage_encode["inputs"]["tile_size"], 1536)
                        self.assertEqual(mage_encode["inputs"]["tile_overlap"], 384)

    def test_experimental_16x_is_two_4x_passes_and_tiles_second_pass(self):
        _, nodes, graph, models = self._modules()
        facade = nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID]
        with self._mock_model_downloads((nodes, graph, models)):
            result = self._execute(
                facade,
                backbone="FLUX.2",
                scale="Experimental 16x (tiled)",
                image=FakeImage(width=256, height=384),
            )
        expanded = result.expand
        node_types = [item["class_type"] for item in expanded]
        self.assertEqual(node_types.count("PiDConditioning"), 2)
        self.assertEqual(node_types.count("SamplerCustom"), 2)
        self.assertEqual(node_types.count("VAEEncode"), 1)
        self.assertEqual(node_types.count("VAEEncodeTiled"), 1)
        context = next(
            item for item in expanded if item["class_type"] == "ContextWindowsManual"
        )
        self.assertEqual(context["inputs"]["dim"], 2)
        self.assertEqual(context["inputs"]["context_length"], 1536)
        self.assertEqual(context["inputs"]["context_overlap"], 384)
        self.assertEqual(context["inputs"]["context_schedule"], "standard_static")
        self.assertEqual(context["inputs"]["fuse_method"], "pyramid")
        self.assertFalse(context["inputs"]["freenoise"])
        latents = [
            item
            for item in expanded
            if item["class_type"] == "EmptyChromaRadianceLatentImage"
        ]
        self.assertEqual(
            [(item["inputs"]["width"], item["inputs"]["height"]) for item in latents],
            [(1024, 1536), (4096, 6144)],
        )
        second_encode = next(
            item for item in expanded if item["class_type"] == "VAEEncodeTiled"
        )
        self.assertEqual(second_encode["inputs"]["tile_size"], 1536)
        self.assertEqual(second_encode["inputs"]["overlap"], 384)

    def test_mage_vae_16x_uses_isolated_tiled_encoder_for_both_passes(self):
        _, nodes, graph, models = self._modules()
        facade = nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID]
        with self._mock_model_downloads((nodes, graph, models)):
            result = self._execute(
                facade,
                backbone=MAGE_VAE,
                scale="Experimental 16x (tiled)",
                image=FakeImage(width=256, height=384),
            )
        expanded = result.expand
        node_types = [item["class_type"] for item in expanded]
        self.assertEqual(node_types.count("ComfyColabMageVAEEncode"), 2)
        self.assertEqual(node_types.count("VAEEncode"), 0)
        self.assertEqual(node_types.count("VAEEncodeTiled"), 0)
        encoders = [
            item
            for item in expanded
            if item["class_type"] == "ComfyColabMageVAEEncode"
        ]
        self.assertTrue(
            all(item["inputs"]["tile_size"] == 1536 for item in encoders)
        )
        self.assertTrue(
            all(item["inputs"]["tile_overlap"] == 384 for item in encoders)
        )

    def test_arbitrary_image_dimensions_are_exactly_resized_after_pid(self):
        _, nodes, graph, models = self._modules()
        facade = nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID]
        with self._mock_model_downloads((nodes, graph, models)):
            result = self._execute(
                facade,
                image=FakeImage(width=513, height=511),
            )
        resize = next(
            item for item in result.expand if item["class_type"] == "ImageScale"
        )
        self.assertEqual(resize["inputs"]["width"], 2052)
        self.assertEqual(resize["inputs"]["height"], 2044)

    def test_mage_vae_alignment_drives_pid_target_then_resizes_exactly(self):
        _, nodes, graph, models = self._modules()
        facade = nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID]
        with self._mock_model_downloads((nodes, graph, models)):
            result = self._execute(
                facade,
                backbone=MAGE_VAE,
                image=FakeImage(width=257, height=259),
            )
        latent = next(
            item
            for item in result.expand
            if item["class_type"] == "EmptyChromaRadianceLatentImage"
        )
        self.assertEqual(latent["inputs"]["width"], 1088)
        self.assertEqual(latent["inputs"]["height"], 1088)
        resize = next(
            item for item in result.expand if item["class_type"] == "ImageScale"
        )
        self.assertEqual(resize["inputs"]["width"], 1028)
        self.assertEqual(resize["inputs"]["height"], 1036)

    def test_mage_vae_accepts_small_images_after_16_pixel_alignment(self):
        _, nodes, graph, models = self._modules()
        facade = nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID]
        with self._mock_model_downloads((nodes, graph, models)):
            result = self._execute(
                facade,
                backbone=MAGE_VAE,
                image=FakeImage(width=128, height=96),
            )
        encoder = next(
            item
            for item in result.expand
            if item["class_type"] == "ComfyColabMageVAEEncode"
        )
        self.assertEqual(encoder["inputs"]["vae_name"], "mage-vae.safetensors")

    def test_output_caps_are_validated_before_downloads(self):
        _, nodes, _, _ = self._modules()
        facade = nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID]
        with mock.patch.object(nodes, "ensure_pid_assets") as ensure:
            with self.assertRaisesRegex(ValueError, "capped"):
                self._execute(
                    facade,
                    scale="Experimental 16x (tiled)",
                    image=FakeImage(width=1024, height=512),
                )
        ensure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
