from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "custom_nodes" / "ComfyColab-LTXVideo"

GGUF_FILENAMES = {
    "Q3_K_S": "ltx-2.3-22b-distilled-1.1-Q3_K_S.gguf",
    "Q4_K_S": "ltx-2.3-22b-distilled-1.1-Q4_K_S.gguf",
    "Q4_K_M": "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf",
}
CONNECTOR_FILENAME = "ltx-2.3-22b-distilled_embeddings_connectors.safetensors"
VIDEO_VAE_FILENAME = "ltx-2.3-22b-distilled_video_vae.safetensors"
AUDIO_VAE_FILENAME = "ltx-2.3-22b-distilled_audio_vae.safetensors"
SPATIAL_FILENAMES = {
    "1.5x": "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors",
    "2x": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
}
TEMPORAL_FILENAME = "ltx-2.3-temporal-upscaler-x2-1.0.safetensors"


def load_package_module(module_name):
    package_name = "comfycolab_ltx_models_test_package"
    for loaded_name in list(sys.modules):
        if loaded_name == package_name or loaded_name.startswith(package_name + "."):
            del sys.modules[loaded_name]
    package = types.ModuleType(package_name)
    package.__path__ = [str(PACKAGE_DIR)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.{module_name}",
        PACKAGE_DIR / f"{module_name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def all_assets(catalog_module):
    catalog = catalog_module.load_catalog()
    assets = list(catalog["ggufs"].values())
    assets.extend(catalog["shared"].values())
    assets.extend(
        asset
        for asset in catalog["spatial_upscalers"].values()
        if asset is not None
    )
    return assets


def asset_field(asset, name, *aliases):
    names = (name, *aliases)
    if isinstance(asset, dict):
        for candidate in names:
            if candidate in asset:
                return asset[candidate]
        return None
    for candidate in names:
        if hasattr(asset, candidate):
            return getattr(asset, candidate)
    return None


class LTXModelTests(unittest.TestCase):
    def test_catalog_pins_curated_distilled_11_and_latest_upscalers(self):
        catalog = load_package_module("catalog")
        assets = all_assets(catalog)
        by_filename = {
            asset_field(asset, "filename"): asset
            for asset in assets
        }
        expected_fixed = {
            *GGUF_FILENAMES.values(),
            CONNECTOR_FILENAME,
            VIDEO_VAE_FILENAME,
            AUDIO_VAE_FILENAME,
            *SPATIAL_FILENAMES.values(),
            TEMPORAL_FILENAME,
        }
        self.assertTrue(expected_fixed <= set(by_filename))

        gemma_assets = [
            asset
            for filename, asset in by_filename.items()
            if filename and "gemma" in filename.casefold()
        ]
        self.assertEqual(
            len(gemma_assets),
            1,
            "The curated bundle should pin one shared Gemma text encoder.",
        )
        self.assertFalse(
            any("lora" in str(filename).casefold() for filename in by_filename)
        )

        expected_folders = {
            **{filename: "unet_gguf" for filename in GGUF_FILENAMES.values()},
            CONNECTOR_FILENAME: "text_encoders",
            VIDEO_VAE_FILENAME: "vae",
            AUDIO_VAE_FILENAME: "vae",
            **{
                filename: "latent_upscale_models"
                for filename in (
                    *SPATIAL_FILENAMES.values(),
                    TEMPORAL_FILENAME,
                )
            },
        }
        for filename, asset in by_filename.items():
            with self.subTest(filename=filename):
                url = str(asset_field(asset, "url") or "")
                sha256 = str(asset_field(asset, "sha256") or "")
                size = asset_field(asset, "size_bytes", "size")
                folder = asset_field(asset, "folder_key", "folder")

                self.assertRegex(
                    url,
                    r"^https://huggingface\.co/.+/resolve/[0-9a-f]{40}/",
                )
                self.assertNotIn("/resolve/main/", url)
                self.assertNotIn("/resolve/master/", url)
                self.assertRegex(sha256, r"^[0-9a-f]{64}$")
                self.assertIsInstance(size, int)
                self.assertGreater(size, 0)
                if filename in expected_folders:
                    self.assertEqual(folder, expected_folders[filename])
                elif "gemma" in str(filename).casefold():
                    self.assertEqual(folder, "text_encoders")

    def test_ensure_model_assets_downloads_only_the_selected_runtime_bundle(self):
        models = load_package_module("models")
        catalog_module = sys.modules[
            "comfycolab_ltx_models_test_package.catalog"
        ]
        assets = all_assets(catalog_module)
        gemma_filename = next(
            asset_field(asset, "filename")
            for asset in assets
            if "gemma" in str(asset_field(asset, "filename")).casefold()
        )
        common = {
            CONNECTOR_FILENAME,
            VIDEO_VAE_FILENAME,
            AUDIO_VAE_FILENAME,
            gemma_filename,
        }
        cases = (
            (
                "Q3_K_S",
                "None",
                "24",
                {GGUF_FILENAMES["Q3_K_S"], *common},
            ),
            (
                "Q4_K_S",
                "1.5x",
                24,
                {
                    GGUF_FILENAMES["Q4_K_S"],
                    SPATIAL_FILENAMES["1.5x"],
                    *common,
                },
            ),
            (
                "Q4_K_M",
                "2x",
                "48",
                {
                    GGUF_FILENAMES["Q4_K_M"],
                    SPATIAL_FILENAMES["2x"],
                    TEMPORAL_FILENAME,
                    *common,
                },
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots = {
                key: [str(root / key)]
                for key in (
                    "unet_gguf",
                    "text_encoders",
                    "vae",
                    "latent_upscale_models",
                )
            }
            folder_paths = types.SimpleNamespace(
                get_folder_paths=lambda key: roots[key]
            )
            previous = sys.modules.get("folder_paths")
            sys.modules["folder_paths"] = folder_paths
            try:
                for gguf_model, spatial_upscaler, fps, expected in cases:
                    with self.subTest(
                        gguf_model=gguf_model,
                        spatial_upscaler=spatial_upscaler,
                        fps=fps,
                    ):
                        calls = []

                        def fake_download(*, destination, **kwargs):
                            calls.append((destination, kwargs))
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            destination.write_bytes(b"stub")
                            return destination

                        with mock.patch.object(
                            models,
                            "download_file",
                            side_effect=fake_download,
                        ):
                            result = models.ensure_model_assets(
                                gguf_model,
                                spatial_upscaler,
                                fps,
                                force_redownload=True,
                            )

                        downloaded = {destination.name for destination, _ in calls}
                        self.assertEqual(downloaded, expected)
                        self.assertEqual(
                            {
                                Path(value).name
                                for value in result.values()
                                if value is not None
                            },
                            expected,
                        )
                        for destination, kwargs in calls:
                            self.assertIn(
                                destination.parent.name,
                                {
                                    "unet_gguf",
                                    "text_encoders",
                                    "vae",
                                    "latent_upscale_models",
                                },
                            )
                            force = kwargs.get(
                                "force",
                                kwargs.get("force_redownload"),
                            )
                            self.assertIs(force, True)
                            self.assertRegex(
                                str(kwargs.get("expected_sha256", "")),
                                r"^[0-9a-f]{64}$",
                            )
            finally:
                if previous is None:
                    sys.modules.pop("folder_paths", None)
                else:
                    sys.modules["folder_paths"] = previous

    def test_unknown_public_choices_fail_before_downloading(self):
        models = load_package_module("models")
        previous = sys.modules.get("folder_paths")
        sys.modules["folder_paths"] = types.SimpleNamespace(
            get_folder_paths=lambda key: [f"/tmp/comfy-models/{key}"]
        )
        try:
            with mock.patch.object(models, "download_file") as download:
                for arguments in (
                    ("Q8_0", "2x", "24"),
                    ("Q3_K_S", "4x", "24"),
                    ("Q3_K_S", "2x", "30"),
                ):
                    with self.subTest(arguments=arguments):
                        with self.assertRaises((KeyError, ValueError, RuntimeError)):
                            models.ensure_model_assets(*arguments)
            download.assert_not_called()
        finally:
            if previous is None:
                sys.modules.pop("folder_paths", None)
            else:
                sys.modules["folder_paths"] = previous


if __name__ == "__main__":
    unittest.main()
