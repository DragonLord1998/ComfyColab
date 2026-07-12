from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
NODE_ROOT = ROOT / "custom_nodes" / "ComfyColab-ZImage"


def load_node_package():
    name = "comfycolab_zimage_loading_test"
    spec = importlib.util.spec_from_file_location(
        name,
        NODE_ROOT / "__init__.py",
        submodule_search_locations=[str(NODE_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class NodeLoadingTests(unittest.TestCase):
    def test_bundle_delegates_to_existing_comfy_loaders(self) -> None:
        package = load_node_package()
        node_module = sys.modules["comfycolab_zimage_loading_test.nodes"]

        calls: list[tuple] = []

        class Unet:
            def load_unet(self, filename):
                calls.append(("gguf_model", filename))
                return ("MODEL_OBJECT",)

        class NativeUnet:
            def load_unet(self, filename, weight_dtype):
                calls.append(("native_model", filename, weight_dtype))
                return ("MODEL_OBJECT",)

        class Clip:
            def load_clip(self, filename, type):
                calls.append(("clip", filename, type))
                return ("CLIP_OBJECT",)

        class Vae:
            def load_vae(self, filename):
                calls.append(("vae", filename))
                return ("VAE_OBJECT",)

        with tempfile.TemporaryDirectory() as directory:
            roots = {
                "unet_gguf": [str(Path(directory) / "diffusion_models")],
                "clip_gguf": [str(Path(directory) / "text_encoders")],
                "diffusion_models": [str(Path(directory) / "diffusion_models")],
                "text_encoders": [str(Path(directory) / "text_encoders")],
                "vae": [str(Path(directory) / "vae")],
            }
            folder_paths = types.SimpleNamespace(get_folder_paths=lambda key: roots[key])
            comfy_nodes = types.SimpleNamespace(
                NODE_CLASS_MAPPINGS={
                    "UnetLoaderGGUF": Unet,
                    "UNETLoader": NativeUnet,
                    "CLIPLoaderGGUF": Clip,
                    "CLIPLoader": Clip,
                    "VAELoader": Vae,
                }
            )

            def fake_import(name):
                if name == "folder_paths":
                    return folder_paths
                if name == "nodes":
                    return comfy_nodes
                raise ImportError(name)

            def fake_download(*, destination, **kwargs):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"stub")
                return destination

            with mock.patch.object(node_module.importlib, "import_module", side_effect=fake_import), mock.patch.object(
                node_module, "download_file", side_effect=fake_download
            ):
                loader = package.NODE_CLASS_MAPPINGS["ComfyColabZImageTurboBundleLoader"]()
                z_outputs = loader.load_bundle("Q4_K_M")
                qwen_loader = package.NODE_CLASS_MAPPINGS[
                    "ComfyColabQwenImageEdit2511BundleLoader"
                ]()
                qwen_outputs = qwen_loader.load_bundle("Q4_K_M")
                krea_loader = package.NODE_CLASS_MAPPINGS[
                    "ComfyColabKrea2BundleLoader"
                ]()
                krea_outputs = krea_loader.load_bundle("Turbo FP8")
                flux_4b_loader = package.NODE_CLASS_MAPPINGS[
                    "ComfyColabFlux2Klein4BBundleLoader"
                ]()
                flux_4b_outputs = flux_4b_loader.load_bundle("Q4_K_M")
                flux_9b_loader = package.NODE_CLASS_MAPPINGS[
                    "ComfyColabFlux2Klein9BBundleLoader"
                ]()
                flux_9b_outputs = flux_9b_loader.load_bundle("Q4_K_M")
                flux_dev_loader = package.NODE_CLASS_MAPPINGS[
                    "ComfyColabFlux2DevBundleLoader"
                ]()
                flux_dev_outputs = flux_dev_loader.load_bundle("Q4_K_M")

        expected = ("MODEL_OBJECT", "CLIP_OBJECT", "VAE_OBJECT")
        self.assertEqual(z_outputs, expected)
        self.assertEqual(qwen_outputs, expected)
        self.assertEqual(krea_outputs, expected)
        self.assertEqual(flux_4b_outputs, expected)
        self.assertEqual(flux_9b_outputs, expected)
        self.assertEqual(flux_dev_outputs, expected)
        self.assertIn(("clip", "Qwen3-4B-Q4_K_M.gguf", "lumina2"), calls)
        self.assertIn(
            ("gguf_model", "qwen-image-edit-2511-Q4_K_M.gguf"),
            calls,
        )
        self.assertIn(
            ("clip", "qwen_2.5_vl_7b_fp8_scaled.safetensors", "qwen_image"),
            calls,
        )
        self.assertIn(
            ("native_model", "krea2_turbo_fp8_scaled.safetensors", "default"),
            calls,
        )
        self.assertIn(
            ("clip", "qwen3vl_4b_fp8_scaled.safetensors", "krea2"),
            calls,
        )
        self.assertIn(
            ("gguf_model", "flux-2-klein-4b-Q4_K_M.gguf"),
            calls,
        )
        self.assertIn(
            ("clip", "qwen_3_4b.safetensors", "flux2"),
            calls,
        )
        self.assertIn(
            ("gguf_model", "flux-2-klein-9b-Q4_K_M.gguf"),
            calls,
        )
        self.assertIn(
            ("clip", "qwen_3_8b_fp8mixed.safetensors", "flux2"),
            calls,
        )
        self.assertIn(("gguf_model", "flux2-dev-Q4_K_M.gguf"), calls)
        self.assertIn(
            ("clip", "mistral_3_small_flux2_fp8.safetensors", "flux2"),
            calls,
        )
        self.assertIn(("vae", "flux2-dev-vae.safetensors"), calls)


if __name__ == "__main__":
    unittest.main()
