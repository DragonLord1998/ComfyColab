from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "custom_nodes" / "ComfyColab-LTXVideo"

H3_NODE_IDS = {
    "ComfyColabMiniMaxH3BundleLoader",
    "ComfyColabMiniMaxH3Video",
    "ComfyColabMiniMaxH3ReferenceVideo",
}
H3_NATIVE_NODES = {
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "MiniMaxH3ImageToVideo",
    "MiniMaxH3ReferenceToVideo",
    "RandomNoise",
    "BasicGuider",
    "KSamplerSelect",
    "BasicScheduler",
    "SamplerCustomAdvanced",
    "VAEDecode",
    "VAEDecodeAudio",
    "CreateVideo",
}


def load_package():
    name = "comfycolab_h3_test"
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
    Model = PortFactory("MODEL")
    Clip = PortFactory("CLIP")
    Vae = PortFactory("VAE")

    class Autogrow:
        Type = dict

        class TemplatePrefix:
            def __init__(self, input, prefix, min=0, max=100):
                self.input = input
                self.prefix = prefix
                self.min = min
                self.max = max

        @staticmethod
        def Input(name, template):
            return {
                "direction": "input",
                "name": name,
                "io_type": "COMFY_AUTOGROW_V3",
                "template": template,
            }

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


class H3NodePackTests(unittest.TestCase):
    def setUp(self):
        self.saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "comfy_api",
                "comfy_api.latest",
                "comfy_execution",
                "comfy_execution.graph_utils",
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
            node_id: object for node_id in H3_NATIVE_NODES
        }
        sys.modules.update(
            {
                "comfy_api": api,
                "comfy_api.latest": latest,
                "comfy_execution": execution,
                "comfy_execution.graph_utils": graph_utils,
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
        nodes = importlib.import_module("comfycolab_h3_test.nodes")
        catalog = importlib.import_module("comfycolab_h3_test.catalog")
        graph_h3 = importlib.import_module("comfycolab_h3_test.graph_h3")
        return package, nodes, catalog, graph_h3

    def _bundle(self, variant="FL2VA"):
        return {
            "family": "minimax_h3",
            "variant": variant,
            "model": "MODEL",
            "text_encoder": "CLIP",
            "video_vae": "VIDEO_VAE",
            "audio_vae": "AUDIO_VAE",
            "filenames": {},
        }

    def test_public_h3_nodes_are_registered_with_expected_display_names(self):
        package, nodes, _catalog, _graph_h3 = self._modules()
        self.assertTrue(H3_NODE_IDS <= set(nodes.PUBLIC_NODE_CLASS_MAPPINGS))
        self.assertTrue(H3_NODE_IDS <= set(nodes.NODE_CLASS_MAPPINGS))
        self.assertEqual(
            nodes.NODE_DISPLAY_NAME_MAPPINGS["ComfyColabMiniMaxH3BundleLoader"],
            "MiniMax H3 Bundle Loader",
        )
        self.assertEqual(
            nodes.NODE_DISPLAY_NAME_MAPPINGS["ComfyColabMiniMaxH3Video"],
            "ComfyColab MiniMax H3 — Text/Image to Video",
        )
        self.assertEqual(
            nodes.NODE_DISPLAY_NAME_MAPPINGS["ComfyColabMiniMaxH3ReferenceVideo"],
            "ComfyColab MiniMax H3 — Reference to Video",
        )

    def test_h3_schema_contracts_expose_typed_bundle_and_media_outputs(self):
        _package, nodes, _catalog, _graph_h3 = self._modules()
        loader_schema = nodes.ComfyColabMiniMaxH3BundleLoader.define_schema()
        self.assertEqual(loader_schema.category, "ComfyColab/loaders")
        self.assertEqual(
            [output["io_type"] for output in loader_schema.outputs],
            ["MINIMAX_H3_BUNDLE", "MODEL", "CLIP", "VAE", "VAE"],
        )
        self.assertFalse(
            next(
                input_
                for input_ in loader_schema.inputs
                if input_["name"] == "accept_h3_license_and_territory"
            )["default"]
        )

        fl2va_schema = nodes.ComfyColabMiniMaxH3Video.define_schema()
        self.assertTrue(fl2va_schema.enable_expand)
        self.assertEqual(fl2va_schema.category, "ComfyColab/Video")
        self.assertEqual(
            [output["io_type"] for output in fl2va_schema.outputs],
            ["VIDEO", "IMAGE", "AUDIO"],
        )

        ref_schema = nodes.ComfyColabMiniMaxH3ReferenceVideo.define_schema()
        ref_inputs = {input_["name"]: input_ for input_ in ref_schema.inputs}
        self.assertEqual(ref_inputs["bundle"]["io_type"], "MINIMAX_H3_BUNDLE")
        self.assertEqual(ref_inputs["ref_image_size"]["default"], "match")
        self.assertEqual(ref_inputs["scheduler"]["default"], "beta")
        self.assertEqual(ref_inputs["ref_images"]["io_type"], "COMFY_AUTOGROW_V3")
        self.assertEqual(ref_inputs["ref_images"]["template"].max, 9)
        self.assertEqual(ref_inputs["ref_images"]["template"].prefix, "ref_image_")
        self.assertEqual(ref_inputs["ref_images"]["template"].input["name"], "ref_image")
        self.assertEqual(ref_inputs["ref_videos"]["template"].max, 3)
        self.assertEqual(ref_inputs["ref_videos"]["template"].prefix, "ref_video_")
        self.assertEqual(ref_inputs["ref_videos"]["template"].input["name"], "ref_video")
        self.assertEqual(ref_inputs["ref_video_audios"]["template"].max, 3)
        self.assertEqual(
            ref_inputs["ref_video_audios"]["template"].prefix,
            "ref_video_audio_",
        )
        self.assertEqual(
            ref_inputs["ref_video_audios"]["template"].input["name"],
            "ref_video_audio",
        )
        self.assertEqual(ref_inputs["ref_audios"]["template"].max, 3)
        self.assertEqual(ref_inputs["ref_audios"]["template"].prefix, "ref_audio_")
        self.assertEqual(ref_inputs["ref_audios"]["template"].input["name"], "ref_audio")

    def test_h3_catalog_pins_official_optimized_assets_and_totals(self):
        _package, _nodes, catalog, _graph_h3 = self._modules()
        h3 = catalog.load_h3_catalog()
        self.assertEqual(h3["repo_id"], "Comfy-Org/MiniMax-H3")
        self.assertEqual(
            h3["revision"],
            "0543966fbdce5ba05709a8f2031c94bdba629b4a",
        )
        fl2va = catalog.h3_assets_for("FL2VA")
        ref2va = catalog.h3_assets_for("Ref2VA")
        self.assertEqual(
            set(fl2va),
            {"model", "text_encoder", "video_vae", "audio_vae"},
        )
        self.assertEqual(fl2va["model"]["folder_key"], "diffusion_models")
        self.assertEqual(ref2va["model"]["folder_key"], "diffusion_models")
        self.assertEqual(fl2va["text_encoder"]["folder_key"], "text_encoders")
        self.assertEqual(fl2va["video_vae"]["folder_key"], "vae")
        self.assertEqual(fl2va["audio_vae"]["folder_key"], "vae")
        self.assertEqual(
            sum(asset["size_bytes"] for asset in fl2va.values()),
            42470585471,
        )
        shared_total = (
            fl2va["text_encoder"]["size_bytes"]
            + fl2va["video_vae"]["size_bytes"]
            + fl2va["audio_vae"]["size_bytes"]
        )
        self.assertEqual(shared_total, 21500205855)
        self.assertEqual(
            fl2va["model"]["size_bytes"] + ref2va["model"]["size_bytes"] + shared_total,
            63440965087,
        )
        self.assertEqual(
            fl2va["model"]["sha256"],
            "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a",
        )
        for asset in fl2va.values():
            self.assertIn("/resolve/0543966fbdce5ba05709a8f2031c94bdba629b4a/", asset["url"])
            self.assertRegex(asset["sha256"], r"^[0-9a-f]{64}$")

    def test_loader_license_gate_blocks_before_provisioning(self):
        _package, nodes, _catalog, _graph_h3 = self._modules()
        with mock.patch.object(nodes, "ensure_h3_model_assets") as ensure:
            with self.assertRaisesRegex(PermissionError, "accept_h3_license_and_territory"):
                nodes.ComfyColabMiniMaxH3BundleLoader.execute(
                    "FL2VA — Text / First / Last Frame",
                    accept_h3_license_and_territory=False,
                )
        ensure.assert_not_called()

    def test_loader_preflights_full_h3_runtime_before_provisioning(self):
        _package, nodes, _catalog, _graph_h3 = self._modules()
        sys.modules["nodes"].NODE_CLASS_MAPPINGS.pop("MiniMaxH3ImageToVideo")
        with mock.patch.object(nodes, "ensure_h3_model_assets") as ensure:
            with self.assertRaisesRegex(RuntimeError, "MiniMaxH3ImageToVideo"):
                nodes.ComfyColabMiniMaxH3BundleLoader.execute(
                    "FL2VA — Text / First / Last Frame",
                    accept_h3_license_and_territory=True,
                )
        ensure.assert_not_called()

    def test_loader_returns_typed_bundle_and_raw_components(self):
        _package, nodes, _catalog, _graph_h3 = self._modules()
        calls = []

        class Unet:
            def load_unet(self, filename, weight_dtype="default"):
                calls.append(("unet", filename, weight_dtype))
                return ("MODEL_OBJECT",)

        class Clip:
            def load_clip(self, filename, type):
                calls.append(("clip", filename, type))
                return ("CLIP_OBJECT",)

        class Vae:
            def load_vae(self, filename):
                calls.append(("vae", filename))
                return (f"VAE_OBJECT:{filename}",)

        sys.modules["nodes"].NODE_CLASS_MAPPINGS.update(
            {"UNETLoader": Unet, "CLIPLoader": Clip, "VAELoader": Vae}
        )
        filenames = {
            "variant": "Ref2VA",
            "model": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "text_encoder": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "video_vae": "minimax_h3_video_vae_fp16.safetensors",
            "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
        }
        with mock.patch.object(nodes, "ensure_h3_model_assets", return_value=filenames):
            outputs = nodes.ComfyColabMiniMaxH3BundleLoader.execute(
                "Ref2VA — Reference Images / Video / Audio",
                accept_h3_license_and_territory=True,
                force_redownload=True,
            )
        bundle, model, clip, video_vae, audio_vae = outputs
        self.assertEqual(bundle["family"], "minimax_h3")
        self.assertEqual(bundle["variant"], "Ref2VA")
        self.assertEqual(bundle["model"], "MODEL_OBJECT")
        self.assertEqual(model, "MODEL_OBJECT")
        self.assertEqual(clip, "CLIP_OBJECT")
        self.assertEqual(video_vae, "VAE_OBJECT:minimax_h3_video_vae_fp16.safetensors")
        self.assertEqual(audio_vae, "VAE_OBJECT:minimax_h3_audio_vae_fp32.safetensors")
        self.assertIn(("clip", filenames["text_encoder"], "minimax"), calls)

    def test_fl2va_graph_uses_official_native_contract_and_frame_snap(self):
        _package, nodes, _catalog, _graph_h3 = self._modules()
        result = nodes.ComfyColabMiniMaxH3Video.execute(
            self._bundle("FL2VA"),
            prompt="A quiet market with matching sound.",
            first_frame="FIRST",
            last_frame="LAST",
            duration_seconds=5.0,
            width=864,
            height=480,
            seed=123,
        )
        expanded = result.expand
        node_types = [node["class_type"] for node in expanded]
        self.assertNotIn("EmptyMiniMaxH3LatentAV", node_types)
        h3_node = next(node for node in expanded if node["class_type"] == "MiniMaxH3ImageToVideo")
        self.assertNotIn("model", h3_node["inputs"])
        self.assertNotIn("fps", h3_node["inputs"])
        self.assertEqual(h3_node["inputs"]["clip"], "CLIP")
        self.assertEqual(h3_node["inputs"]["vae"], "VIDEO_VAE")
        self.assertEqual(h3_node["inputs"]["length"], 124)
        self.assertEqual(h3_node["inputs"]["first_frame"], "FIRST")
        self.assertEqual(h3_node["inputs"]["last_frame"], "LAST")
        sampler = next(node for node in expanded if node["class_type"] == "KSamplerSelect")
        self.assertEqual(sampler["inputs"]["sampler_name"], "res_multistep")
        scheduler = next(node for node in expanded if node["class_type"] == "BasicScheduler")
        self.assertEqual(scheduler["inputs"]["scheduler"], "simple")
        self.assertEqual(scheduler["inputs"]["steps"], 20)
        sampled_index = node_types.index("SamplerCustomAdvanced")
        video_decode = next(node for node in expanded if node["class_type"] == "VAEDecode")
        audio_decode = next(node for node in expanded if node["class_type"] == "VAEDecodeAudio")
        self.assertEqual(video_decode["inputs"]["samples"], Link(sampled_index, 0))
        self.assertEqual(audio_decode["inputs"]["samples"], Link(sampled_index, 0))
        self.assertNotIn("sample_rate", audio_decode["inputs"])
        create_video = next(node for node in expanded if node["class_type"] == "CreateVideo")
        self.assertEqual(create_video["inputs"]["fps"], 24.0)
        self.assertNotIn("frame_count", create_video["inputs"])

    def test_ref2va_graph_uses_reference_contract_and_beta_default(self):
        _package, nodes, _catalog, _graph_h3 = self._modules()
        result = nodes.ComfyColabMiniMaxH3ReferenceVideo.execute(
            self._bundle("Ref2VA"),
            prompt="Use <Picture 1>, <Video 1>, and <Audio 1>.",
            ref_images={"ref_image_0": "IMAGE"},
            ref_videos={"ref_video_0": ["frame"] * 48},
            ref_video_audios={
                "ref_video_audio_0": {"waveform": [0.0] * 64000, "sample_rate": 32000}
            },
            ref_audios={
                "ref_audio_0": {"waveform": [0.0] * 64000, "sample_rate": 32000}
            },
        )
        expanded = result.expand
        h3_node = next(node for node in expanded if node["class_type"] == "MiniMaxH3ReferenceToVideo")
        self.assertNotIn("model", h3_node["inputs"])
        self.assertNotIn("fps", h3_node["inputs"])
        self.assertEqual(h3_node["inputs"]["clip"], "CLIP")
        self.assertEqual(h3_node["inputs"]["vae"], "VIDEO_VAE")
        self.assertEqual(h3_node["inputs"]["audio_vae"], "AUDIO_VAE")
        self.assertEqual(h3_node["inputs"]["ref_image_size"], "match")
        self.assertEqual(h3_node["inputs"]["ref_images"], {"ref_image_0": "IMAGE"})
        self.assertEqual(h3_node["inputs"]["ref_videos"], {"ref_video_0": ["frame"] * 48})
        self.assertEqual(
            h3_node["inputs"]["ref_video_audios"],
            {"ref_video_audio_0": {"waveform": [0.0] * 64000, "sample_rate": 32000}},
        )
        self.assertEqual(
            h3_node["inputs"]["ref_audios"],
            {"ref_audio_0": {"waveform": [0.0] * 64000, "sample_rate": 32000}},
        )
        scheduler = next(node for node in expanded if node["class_type"] == "BasicScheduler")
        self.assertEqual(scheduler["inputs"]["scheduler"], "beta")

    def test_h3_validation_rejects_wrong_variant_invalid_canvas_and_audio_only(self):
        _package, nodes, _catalog, _graph_h3 = self._modules()
        with self.assertRaisesRegex(ValueError, "Reference Video node"):
            nodes.ComfyColabMiniMaxH3Video.execute(
                self._bundle("Ref2VA"),
                prompt="valid",
            )
        with self.assertRaisesRegex(ValueError, "768 x 1344"):
            nodes.ComfyColabMiniMaxH3Video.execute(
                self._bundle("FL2VA"),
                prompt="valid",
                width=1344,
                height=1024,
            )
        with self.assertRaisesRegex(ValueError, "reference image or video"):
            nodes.ComfyColabMiniMaxH3ReferenceVideo.execute(
                self._bundle("Ref2VA"),
                prompt="valid",
                ref_audios={
                    "ref_audio_0": {"waveform": [0.0] * 64000, "sample_rate": 32000}
                },
            )

    def test_h3_frame_snapping_uses_official_round_and_modulo_formula(self):
        _package, _nodes, _catalog, graph_h3 = self._modules()
        self.assertEqual(graph_h3.snap_h3_frames(4.0), 107)
        self.assertEqual(graph_h3.snap_h3_frames(5.0), 124)
        self.assertEqual(graph_h3.snap_h3_frames(15.0), 362)
        self.assertEqual(
            graph_h3.snap_h3_frames(4.4625),
            107,
            "round-based snapping must not jump to 124 at this fractional boundary",
        )

    def test_ref2va_duration_boundaries_are_validated_before_graph_construction(self):
        _package, nodes, _catalog, _graph_h3 = self._modules()
        valid_audio = {"waveform": [0.0] * 64000, "sample_rate": 32000}
        valid_video = ["frame"] * 48
        nodes.ComfyColabMiniMaxH3ReferenceVideo.execute(
            self._bundle("Ref2VA"),
            prompt="valid",
            ref_images={"ref_image_0": "IMAGE"},
            ref_videos={"ref_video_0": valid_video},
            ref_video_audios={"ref_video_audio_0": valid_audio},
        )
        with self.assertRaisesRegex(ValueError, "reference videos must be 2-15 seconds"):
            nodes.ComfyColabMiniMaxH3ReferenceVideo.execute(
                self._bundle("Ref2VA"),
                prompt="valid",
                ref_videos={"ref_video_0": ["frame"] * 47},
            )
        with self.assertRaisesRegex(ValueError, "reference audio clips must be 2-15 seconds"):
            nodes.ComfyColabMiniMaxH3ReferenceVideo.execute(
                self._bundle("Ref2VA"),
                prompt="valid",
                ref_images={"ref_image_0": "IMAGE"},
                ref_audios={
                    "ref_audio_0": {"waveform": [0.0] * 63999, "sample_rate": 32000}
                },
            )
        with self.assertRaisesRegex(ValueError, "reference videos must be 2-15 seconds"):
            nodes.ComfyColabMiniMaxH3ReferenceVideo.execute(
                self._bundle("Ref2VA"),
                prompt="valid",
                ref_videos={
                    "ref_video_0": {
                        "duration_seconds": 3.0,
                        "frames": ["frame"] * 47,
                    }
                },
            )
        with self.assertRaisesRegex(ValueError, "reference audio clips must be 2-15 seconds"):
            nodes.ComfyColabMiniMaxH3ReferenceVideo.execute(
                self._bundle("Ref2VA"),
                prompt="valid",
                ref_images={"ref_image_0": "IMAGE"},
                ref_audios={
                    "ref_audio_0": {
                        "duration_seconds": 3.0,
                        "waveform": [0.0] * 63999,
                        "sample_rate": 32000,
                    }
                },
            )
        with self.assertRaisesRegex(ValueError, "video duration is unavailable"):
            nodes.ComfyColabMiniMaxH3ReferenceVideo.execute(
                self._bundle("Ref2VA"),
                prompt="valid",
                ref_videos={"ref_video_0": {"duration_seconds": 3.0}},
            )
        with self.assertRaisesRegex(ValueError, "waveform and sample_rate"):
            nodes.ComfyColabMiniMaxH3ReferenceVideo.execute(
                self._bundle("Ref2VA"),
                prompt="valid",
                ref_images={"ref_image_0": "IMAGE"},
                ref_audios={"ref_audio_0": {"duration_seconds": 3.0}},
            )
        with self.assertRaisesRegex(ValueError, "total reference-video duration"):
            nodes.ComfyColabMiniMaxH3ReferenceVideo.execute(
                self._bundle("Ref2VA"),
                prompt="valid",
                ref_videos={
                    "ref_video_0": ["frame"] * 120,
                    "ref_video_1": ["frame"] * 120,
                    "ref_video_2": ["frame"] * 121,
                },
            )
        with self.assertRaisesRegex(ValueError, "total reference-audio duration"):
            nodes.ComfyColabMiniMaxH3ReferenceVideo.execute(
                self._bundle("Ref2VA"),
                prompt="valid",
                ref_images={"ref_image_0": "IMAGE"},
                ref_audios={
                    "ref_audio_0": {"waveform": [0.0] * 160000, "sample_rate": 32000},
                    "ref_audio_1": {"waveform": [0.0] * 160000, "sample_rate": 32000},
                    "ref_audio_2": {"waveform": [0.0] * 160001, "sample_rate": 32000},
                },
            )

    def test_ref2va_paired_audio_matches_official_underscore_indexed_video_key(self):
        _package, nodes, _catalog, _graph_h3 = self._modules()
        nodes.ComfyColabMiniMaxH3ReferenceVideo.execute(
            self._bundle("Ref2VA"),
            prompt="valid",
            ref_videos={"ref_video_0": ["frame"] * 48},
            ref_video_audios={
                "ref_video_audio_0": {"waveform": [0.0] * 64000, "sample_rate": 32000}
            },
        )


if __name__ == "__main__":
    unittest.main()
