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
PACKAGE_DIR = ROOT / "custom_nodes" / "ComfyColab-LTXVideo"

PUBLIC_NODE_ID = "ComfyColabLTX23Video"
DISPLAY_NAME = "ComfyColab LTX-2.3 — Text/Image to Video"
GGUF_OPTIONS = ["Q3_K_S", "Q4_K_S", "Q4_K_M"]
FPS_OPTIONS = ["24", "48"]
SPATIAL_OPTIONS = ["None", "1.5x", "2x"]

GGUF_FILENAMES = {
    "Q3_K_S": "ltx-2.3-22b-distilled-1.1-Q3_K_S.gguf",
    "Q4_K_S": "ltx-2.3-22b-distilled-1.1-Q4_K_S.gguf",
    "Q4_K_M": "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf",
}
SPATIAL_FILENAMES = {
    "1.5x": "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors",
    "2x": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
}
TEMPORAL_FILENAME = "ltx-2.3-temporal-upscaler-x2-1.0.safetensors"


def load_package():
    name = "comfycolab_ltx_test"
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
    Video = PortFactory("VIDEO")
    Audio = PortFactory("AUDIO")
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
    "UnetLoaderGGUF",
    "DualCLIPLoaderGGUF",
    "VAELoader",
    "LTXVAudioVAELoader",
    "LTXAVTextEncoderLoader",
    "CLIPTextEncode",
    "LTXVConditioning",
    "EmptyLTXVLatentVideo",
    "LTXVEmptyLatentAudio",
    "LTXVConcatAVLatent",
    "LTXVSeparateAVLatent",
    "LTXVPreprocess",
    "LTXVImgToVideoConditionOnly",
    "RandomNoise",
    "BasicGuider",
    "KSamplerSelect",
    "ManualSigmas",
    "SamplerCustomAdvanced",
    "LatentUpscaleModelLoader",
    "LTXVLatentUpsampler",
    "LTXVTiledVAEDecode",
    "LTXVAudioVAEDecode",
    "CreateVideo",
}


class LTXNodePackTests(unittest.TestCase):
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
        nodes = importlib.import_module("comfycolab_ltx_test.nodes")
        graph = importlib.import_module("comfycolab_ltx_test.graph")
        models = importlib.import_module("comfycolab_ltx_test.models")
        return package, nodes, graph, models

    @staticmethod
    def _filenames(
        gguf_model: str,
        spatial_upscaler: str,
        fps: str | int,
    ) -> dict[str, str | None]:
        fps = str(fps)
        return {
            "model": GGUF_FILENAMES[gguf_model],
            "gguf_model": GGUF_FILENAMES[gguf_model],
            "connector": (
                "ltx-2.3-22b-distilled_embeddings_connectors.safetensors"
            ),
            "embedding_connector": (
                "ltx-2.3-22b-distilled_embeddings_connectors.safetensors"
            ),
            "text_encoder": "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf",
            "gemma": "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf",
            "video_vae": "ltx-2.3-22b-distilled_video_vae.safetensors",
            "audio_vae": "ltx-2.3-22b-distilled_audio_vae.safetensors",
            "spatial_upscaler": SPATIAL_FILENAMES.get(spatial_upscaler),
            "temporal_2x": TEMPORAL_FILENAME if fps == "48" else None,
            "temporal_upscaler": TEMPORAL_FILENAME if fps == "48" else None,
        }

    @contextlib.contextmanager
    def _mock_model_downloads(self, modules):
        def ensure(
            gguf_model,
            spatial_upscaler,
            fps,
            force_redownload=False,
        ):
            return self._filenames(gguf_model, spatial_upscaler, fps)

        with contextlib.ExitStack() as stack:
            patched = False
            for module in modules:
                for name in (
                    "ensure_model_assets",
                    "ensure_ltx23_model_assets",
                    "ensure_runtime_assets",
                ):
                    if hasattr(module, name):
                        stack.enter_context(
                            mock.patch.object(module, name, side_effect=ensure)
                        )
                        patched = True
            self.assertTrue(
                patched,
                "The LTX pack must expose a mockable model-provisioning boundary.",
            )
            yield

    @staticmethod
    def _execute(
        node,
        *,
        gguf_model="Q3_K_S",
        fps="24",
        spatial_upscaler="2x",
        image=None,
        width=960,
        height=544,
    ):
        return node.execute(
            prompt="A quiet tea ceremony with synchronized room ambience.",
            gguf_model=gguf_model,
            fps=fps,
            spatial_upscaler=spatial_upscaler,
            width=width,
            height=height,
            frame_count=121,
            seed=0,
            image_strength=0.7,
            image=image,
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
        self.assertEqual(schema.category, "ComfyColab/Video")
        self.assertTrue(schema.enable_expand)
        self.assertEqual(
            [item["name"] for item in schema.outputs],
            ["video", "frames", "audio"],
        )
        self.assertEqual(
            [item["io_type"] for item in schema.outputs],
            ["VIDEO", "IMAGE", "AUDIO"],
        )

        self.assertEqual(inputs["gguf_model"]["options"], GGUF_OPTIONS)
        self.assertEqual(inputs["gguf_model"]["default"], "Q3_K_S")
        self.assertEqual(inputs["fps"]["options"], FPS_OPTIONS)
        self.assertEqual(str(inputs["fps"]["default"]), "24")
        self.assertEqual(inputs["spatial_upscaler"]["options"], SPATIAL_OPTIONS)
        self.assertEqual(inputs["spatial_upscaler"]["default"], "2x")
        self.assertEqual(inputs["width"]["default"], 960)
        self.assertEqual(inputs["height"]["default"], 544)
        self.assertEqual(inputs["frame_count"]["default"], 121)
        self.assertEqual(inputs["seed"]["default"], 0)
        self.assertEqual(inputs["image_strength"]["default"], 0.7)
        self.assertTrue(inputs["image"]["optional"])
        self.assertIn("prompt", inputs)
        self.assertNotIn("negative_prompt", inputs)

    def test_graph_matrix_honors_gguf_fps_and_spatial_choices(self):
        _, nodes, graph, models = self._modules()
        facade = nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID]
        cases = (
            ("Q3_K_S", "24", "None"),
            ("Q4_K_S", "48", "None"),
            ("Q4_K_M", "24", "1.5x"),
            ("Q3_K_S", "48", "2x"),
        )
        with self._mock_model_downloads((nodes, graph, models)):
            for gguf_model, fps, spatial_upscaler in cases:
                with self.subTest(
                    gguf_model=gguf_model,
                    fps=fps,
                    spatial_upscaler=spatial_upscaler,
                ):
                    result = self._execute(
                        facade,
                        gguf_model=gguf_model,
                        fps=fps,
                        spatial_upscaler=spatial_upscaler,
                        height=576 if spatial_upscaler == "1.5x" else 544,
                    )
                    expanded = result.expand
                    node_types = [item["class_type"] for item in expanded]

                    self.assertNotIn("CheckpointLoaderSimple", node_types)
                    self.assertNotIn("LoraLoaderModelOnly", node_types)
                    gguf_loader = next(
                        item
                        for item in expanded
                        if item["class_type"] == "UnetLoaderGGUF"
                    )
                    self.assertEqual(
                        gguf_loader["inputs"]["unet_name"],
                        GGUF_FILENAMES[gguf_model],
                    )

                    latent_video = next(
                        item
                        for item in expanded
                        if item["class_type"] == "EmptyLTXVLatentVideo"
                    )
                    latent_audio = next(
                        item
                        for item in expanded
                        if item["class_type"] == "LTXVEmptyLatentAudio"
                    )
                    conditioning = next(
                        item
                        for item in expanded
                        if item["class_type"] == "LTXVConditioning"
                    )
                    encoded_prompts = [
                        item["inputs"]["text"]
                        for item in expanded
                        if item["class_type"] == "CLIPTextEncode"
                    ]
                    create_video = next(
                        item
                        for item in expanded
                        if item["class_type"] == "CreateVideo"
                    )
                    self.assertEqual(latent_video["inputs"]["length"], 121)
                    self.assertEqual(latent_audio["inputs"]["frames_number"], 121)
                    self.assertEqual(int(latent_audio["inputs"]["frame_rate"]), 24)
                    self.assertEqual(int(conditioning["inputs"]["frame_rate"]), 24)
                    self.assertEqual(
                        encoded_prompts,
                        [
                            "A quiet tea ceremony with synchronized room ambience."
                        ],
                    )
                    self.assertEqual(int(create_video["inputs"]["fps"]), int(fps))

                    loaders = [
                        (index, item)
                        for index, item in enumerate(expanded)
                        if item["class_type"]
                        in {
                            "LatentUpscaleModelLoader",
                            "LowVRAMLatentUpscaleModelLoader",
                        }
                    ]
                    loaded_names = {
                        item["inputs"].get("model_name")
                        or item["inputs"].get("upscale_model")
                        for _, item in loaders
                    }
                    expected_spatial = SPATIAL_FILENAMES.get(spatial_upscaler)
                    if expected_spatial is None:
                        self.assertFalse(
                            loaded_names & set(SPATIAL_FILENAMES.values())
                        )
                    else:
                        self.assertIn(expected_spatial, loaded_names)

                    if fps == "48":
                        self.assertIn(TEMPORAL_FILENAME, loaded_names)
                    else:
                        self.assertNotIn(TEMPORAL_FILENAME, loaded_names)

                    expected_sampler_count = 1 if spatial_upscaler == "None" else 2
                    self.assertEqual(
                        node_types.count("SamplerCustomAdvanced"),
                        expected_sampler_count,
                    )
                    self.assertEqual(
                        node_types.count("BasicGuider"),
                        expected_sampler_count,
                    )
                    self.assertNotIn("CFGGuider", node_types)
                    sampler_names = [
                        item["inputs"]["sampler_name"]
                        for item in expanded
                        if item["class_type"] == "KSamplerSelect"
                    ]
                    self.assertEqual(
                        sampler_names,
                        ["euler"] * expected_sampler_count,
                    )
                    sigma_schedules = [
                        item["inputs"]["sigmas"]
                        for item in expanded
                        if item["class_type"] == "ManualSigmas"
                    ]
                    self.assertEqual(
                        sigma_schedules[0],
                        (
                            "1.0, 0.99375, 0.9875, 0.98125, 0.975, "
                            "0.909375, 0.725, 0.421875, 0.0"
                        ),
                    )
                    if spatial_upscaler == "None":
                        self.assertEqual(len(sigma_schedules), 1)
                    else:
                        self.assertEqual(
                            sigma_schedules[1],
                            "0.909375, 0.725, 0.421875, 0.0",
                        )

                    for loader_index, loader in loaders:
                        model_name = (
                            loader["inputs"].get("model_name")
                            or loader["inputs"].get("upscale_model")
                        )
                        if model_name not in {
                            *SPATIAL_FILENAMES.values(),
                            TEMPORAL_FILENAME,
                        }:
                            continue
                        upsampler = next(
                            item
                            for item in expanded
                            if item["class_type"] == "LTXVLatentUpsampler"
                            and item["inputs"]["upscale_model"]
                            == Link(loader_index, 0)
                        )
                        source = upsampler["inputs"]["samples"]
                        self.assertIsInstance(source, Link)
                        self.assertEqual(
                            expanded[source.node_id]["class_type"],
                            "LTXVSeparateAVLatent",
                        )
                        self.assertEqual(
                            source.index,
                            0,
                            "spatial and temporal upscalers must consume video only",
                        )

                    self.assertEqual(len(result.values), 3)
                    self.assertEqual(
                        expanded[result.values[0].node_id]["class_type"],
                        "CreateVideo",
                    )
                    self.assertIn(
                        expanded[result.values[1].node_id]["class_type"],
                        {"LTXVTiledVAEDecode", "VAEDecode"},
                    )
                    self.assertEqual(
                        expanded[result.values[2].node_id]["class_type"],
                        "LTXVAudioVAEDecode",
                    )

    def test_optional_image_adds_image_conditioning_with_public_strength(self):
        _, nodes, graph, models = self._modules()
        facade = nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID]
        image = object()
        with self._mock_model_downloads((nodes, graph, models)):
            text_result = self._execute(facade, image=None)
            image_result = self._execute(facade, image=image)

        self.assertNotIn(
            "LTXVImgToVideoConditionOnly",
            [item["class_type"] for item in text_result.expand],
        )
        preprocess = next(
            item
            for item in image_result.expand
            if item["class_type"] == "LTXVPreprocess"
        )
        self.assertIs(preprocess["inputs"]["image"], image)
        preprocess_index = image_result.expand.index(preprocess)
        conditioners = [
            item
            for item in image_result.expand
            if item["class_type"] == "LTXVImgToVideoConditionOnly"
        ]
        self.assertTrue(conditioners)
        self.assertTrue(
            all(
                conditioner["inputs"]["image"] == Link(preprocess_index, 0)
                for conditioner in conditioners
            )
        )
        self.assertIn(0.7, {item["inputs"]["strength"] for item in conditioners})

    def test_missing_required_ltx_node_fails_before_any_download(self):
        _, nodes, _, models = self._modules()
        facade = nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID]
        sys.modules["nodes"].NODE_CLASS_MAPPINGS.pop("LTXVLatentUpsampler")

        with mock.patch.object(
            models,
            "ensure_model_assets",
            side_effect=AssertionError("download ran before compatibility check"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "LTXVLatentUpsampler.*(?:bootstrap|refresh)|"
                "(?:bootstrap|refresh).*LTXVLatentUpsampler",
            ):
                self._execute(facade, spatial_upscaler="2x")

    def test_one_point_five_spatial_requires_exactly_scalable_dimensions(self):
        _, nodes, _, _ = self._modules()
        facade = nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID]

        with mock.patch.object(nodes, "ensure_model_assets") as ensure:
            with self.assertRaisesRegex(ValueError, "divisible by 64"):
                facade.execute(
                    prompt="A slow camera move through a quiet greenhouse.",
                    gguf_model="Q3_K_S",
                    fps="24",
                    spatial_upscaler="1.5x",
                    width=960,
                    height=544,
                    frame_count=121,
                    seed=0,
                )
        ensure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
