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
                calls.append(("model", filename))
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
                "vae": [str(Path(directory) / "vae")],
            }
            folder_paths = types.SimpleNamespace(get_folder_paths=lambda key: roots[key])
            comfy_nodes = types.SimpleNamespace(
                NODE_CLASS_MAPPINGS={
                    "UnetLoaderGGUF": Unet,
                    "CLIPLoaderGGUF": Clip,
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
                outputs = loader.load_bundle("Q4_K_M")

        self.assertEqual(outputs, ("MODEL_OBJECT", "CLIP_OBJECT", "VAE_OBJECT"))
        self.assertIn(("clip", "Qwen3-4B-Q4_K_M.gguf", "lumina2"), calls)


if __name__ == "__main__":
    unittest.main()
