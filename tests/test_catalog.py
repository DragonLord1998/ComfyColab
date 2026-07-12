from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NODE_ROOT = ROOT / "custom_nodes" / "ComfyColab-ZImage"


def load_node_package():
    name = "comfycolab_zimage_test"
    if name in sys.modules:
        return sys.modules[name]
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


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.package = load_node_package()
        cls.catalog = sys.modules["comfycolab_zimage_test.catalog"]

    def test_supported_quantizations_are_curated(self) -> None:
        self.assertEqual(
            self.catalog.quantization_names(),
            ["Q4_K_M", "Q3_K_M", "Q5_K_M", "Q8_0"],
        )

        self.assertEqual(
            self.catalog.qwen_edit_quantization_names(),
            ["Q4_K_M", "Q3_K_M", "Q5_K_M", "Q8_0"],
        )
        self.assertEqual(
            self.catalog.krea_2_variants(),
            ["Turbo FP8", "Raw FP8 (training/base)"],
        )

    def test_every_component_has_https_url_and_checksum(self) -> None:
        selections = (
            (self.catalog.quantization_names(), self.catalog.bundle_for),
            (
                self.catalog.qwen_edit_quantization_names(),
                self.catalog.qwen_edit_bundle_for,
            ),
            (self.catalog.krea_2_variants(), self.catalog.krea_2_bundle_for),
        )
        for names, bundle_for in selections:
            for name in names:
                bundle = bundle_for(name)
                self.assertEqual(set(bundle), {"model", "text_encoder", "vae"})
                for specification in bundle.values():
                    self.assertTrue(
                        specification["url"].startswith("https://huggingface.co/")
                    )
                    self.assertEqual(len(specification["sha256"]), 64)
                    self.assertGreater(specification["size_bytes"], 0)

    def test_node_contract_has_three_comfy_outputs(self) -> None:
        loader_names = (
            "ComfyColabZImageTurboBundleLoader",
            "ComfyColabQwenImageEdit2511BundleLoader",
            "ComfyColabKrea2BundleLoader",
        )
        for loader_name in loader_names:
            loader = self.package.NODE_CLASS_MAPPINGS[loader_name]
            self.assertEqual(loader.RETURN_TYPES, ("MODEL", "CLIP", "VAE"))
            self.assertEqual(loader.RETURN_NAMES, ("model", "text_encoder", "vae"))

        qwen = self.package.NODE_CLASS_MAPPINGS[
            "ComfyColabQwenImageEdit2511BundleLoader"
        ]
        self.assertEqual(qwen.IS_CHANGED("Q4_K_M", False), "Q4_K_M")
        forced = qwen.IS_CHANGED("Q4_K_M", True)
        self.assertNotEqual(forced, forced)

        krea = self.package.NODE_CLASS_MAPPINGS["ComfyColabKrea2BundleLoader"]
        self.assertEqual(krea.IS_CHANGED("Turbo FP8", False), "Turbo FP8")


if __name__ == "__main__":
    unittest.main()
