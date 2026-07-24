from __future__ import annotations

import hashlib
import http.client
import importlib.util
import io
import os
import sys
import tempfile
import types
import unittest
import urllib.error
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "custom_nodes" / "ComfyColab-Triposplat"
MODEL_REVISION = "de3b99ab2627d565a8d5fc40f2db52557b82b974"
EXPECTED_FILENAMES = {
    "birefnet.safetensors",
    "triposplat_fp16.safetensors",
    "dino_v3_vit_h.safetensors",
    "flux2-vae.safetensors",
    "triposplat_vae_decoder_fp16.safetensors",
}
EXPECTED_FOLDERS = {
    "birefnet.safetensors": "background_removal",
    "triposplat_fp16.safetensors": "diffusion_models",
    "dino_v3_vit_h.safetensors": "clip_vision",
    "flux2-vae.safetensors": "vae",
    "triposplat_vae_decoder_fp16.safetensors": "vae",
}
EXPECTED_SHA256 = {
    "birefnet.safetensors": "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154",
    "dino_v3_vit_h.safetensors": "a29ef35101a16966972a0d50732a6f3a608ff7cfffb2afa9bbe9007cb842cc53",
    "triposplat_fp16.safetensors": "c870b97ac1d6bc9177608a5ec625e19ef9f3c5019aa68f64b0fb7803abcd6d20",
    "flux2-vae.safetensors": "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5",
    "triposplat_vae_decoder_fp16.safetensors": "ed0d0c3d43b599e326845d0ec70f3cf77be9a55e2d97627ac3b34d2830763cc8",
}
EXPECTED_SIZES = {
    "birefnet.safetensors": 444_473_596,
    "dino_v3_vit_h.safetensors": 1_681_247_696,
    "triposplat_fp16.safetensors": 741_106_994,
    "flux2-vae.safetensors": 336_213_556,
    "triposplat_vae_decoder_fp16.safetensors": 576_148_442,
}


def load_module(module_name: str):
    name = f"comfycolab_triposplat_{module_name}_test"
    package_name = "comfycolab_triposplat_download_package"
    for module in list(sys.modules):
        if module == package_name or module.startswith(package_name + ".") or module == name:
            del sys.modules[module]
    package = types.ModuleType(package_name)
    package.__path__ = [str(PACKAGE_DIR)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.{module_name}",
        PACKAGE_DIR / f"{module_name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{package_name}.{module_name}"] = module
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def asset_filename(asset):
    if isinstance(asset, dict):
        return asset["filename"]
    return getattr(asset, "filename")


def asset_folder(asset):
    if isinstance(asset, dict):
        return asset.get("folder") or asset.get("folder_key")
    return getattr(asset, "folder", getattr(asset, "folder_key", None))


def asset_revision(asset):
    if isinstance(asset, dict):
        return asset.get("revision")
    return getattr(asset, "revision", None)


def asset_sha256(asset):
    if isinstance(asset, dict):
        return asset.get("sha256")
    return getattr(asset, "sha256", None)


def asset_size(asset):
    if isinstance(asset, dict):
        return asset.get("size_bytes") or asset.get("size")
    return getattr(asset, "size_bytes", getattr(asset, "size", None))


def asset_url(asset):
    if isinstance(asset, dict):
        return asset.get("url", "")
    return getattr(asset, "url", "")


def all_assets(models_module):
    for name in (
        "TRIPOSPLAT_MODEL_ASSETS",
        "MODEL_ASSETS",
        "ASSETS",
        "REQUIRED_ASSETS",
    ):
        if hasattr(models_module, name):
            assets = getattr(models_module, name)
            return list(assets.values() if isinstance(assets, dict) else assets)
    raise AssertionError("Triposplat models module does not expose a model asset catalog")


class FakeResponse(io.BytesIO):
    def __init__(
        self,
        content: bytes,
        *,
        status: int = 200,
        declared_length: int | None = None,
    ):
        super().__init__(content)
        self.status = status
        self.headers = {
            "Content-Length": str(
                len(content) if declared_length is None else declared_length
            )
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class TriposplatDownloadTests(unittest.TestCase):
    def setUp(self):
        self.download = load_module("download")

    def test_hub_primary_uses_token_and_high_performance_xet(self):
        content = b"triposplat-xet-primary"
        digest = hashlib.sha256(content).hexdigest()
        calls = []

        def hf_hub_download(**kwargs):
            calls.append(kwargs)
            destination = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
            destination.write_bytes(content)
            return str(destination)

        fake_hub = types.SimpleNamespace(hf_hub_download=hf_hub_download)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules,
            {"huggingface_hub": fake_hub},
        ), mock.patch.dict(
            os.environ,
            {"HF_TOKEN": "test-token"},
            clear=False,
        ), mock.patch.object(
            self.download.urllib.request,
            "urlopen",
        ) as urlopen:
            destination = Path(directory) / "model.safetensors"
            result = self.download.download_file(
                url=(
                    "https://huggingface.co/VAST-AI/TripoSplat/resolve/"
                    f"{MODEL_REVISION}/model.safetensors?download=true"
                ),
                destination=destination,
                expected_sha256=digest,
                expected_size=len(content),
            )
            self.assertEqual(os.environ["HF_XET_HIGH_PERFORMANCE"], "1")
            self.assertEqual(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"], "120")
            self.assertEqual(result.read_bytes(), content)

        self.assertEqual(calls[0]["repo_id"], "VAST-AI/TripoSplat")
        self.assertEqual(calls[0]["revision"], MODEL_REVISION)
        self.assertEqual(calls[0]["filename"], "model.safetensors")
        self.assertEqual(calls[0]["token"], "test-token")
        urlopen.assert_not_called()

    def test_model_catalog_uses_immutable_official_revision_and_expected_folders(self):
        models = load_module("models")
        assets = all_assets(models)
        by_filename = {asset_filename(asset): asset for asset in assets}

        self.assertEqual(models.HF_REVISION, MODEL_REVISION)
        self.assertEqual(models.HF_REPO_ID, "VAST-AI/TripoSplat")
        self.assertEqual(set(by_filename), EXPECTED_FILENAMES)
        for filename, asset in by_filename.items():
            with self.subTest(filename=filename):
                self.assertEqual(asset_folder(asset), EXPECTED_FOLDERS[filename])
                self.assertIn(asset_revision(asset), (None, MODEL_REVISION))
                self.assertEqual(asset_sha256(asset), EXPECTED_SHA256[filename])
                self.assertEqual(asset_size(asset), EXPECTED_SIZES[filename])
                url = asset_url(asset)
                self.assertIn("VAST-AI/TripoSplat", url)
                self.assertIn(f"/resolve/{MODEL_REVISION}/", url)
                self.assertNotIn("/resolve/main/", url)
                self.assertNotIn("/resolve/master/", url)

    def test_ensure_models_downloads_to_comfy_folders_without_real_network(self):
        models = load_module("models")
        calls = []

        with tempfile.TemporaryDirectory() as directory:
            roots = {
                folder: [str(Path(directory) / folder)]
                for folder in set(EXPECTED_FOLDERS.values())
            }
            folder_paths = types.SimpleNamespace(get_folder_paths=lambda key: roots[key])

            def fake_download(
                *,
                url,
                destination,
                expected_sha256,
                expected_size,
                force=False,
                progress=None,
            ):
                calls.append((url, destination, expected_sha256, expected_size, force, progress))
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"stub")
                return destination

            previous = sys.modules.get("folder_paths")
            sys.modules["folder_paths"] = folder_paths
            try:
                with mock.patch.object(models, "download_file", side_effect=fake_download):
                    if hasattr(models, "ensure_model_assets"):
                        result = models.ensure_model_assets()
                    elif hasattr(models, "ensure_triposplat_models"):
                        result = models.ensure_triposplat_models(folder_paths)
                    elif hasattr(models, "ensure_models"):
                        result = models.ensure_models(folder_paths)
                    else:
                        raise AssertionError(
                            "models module has no ensure_model_assets/ensure_triposplat_models/ensure_models"
                        )
            finally:
                if previous is None:
                    sys.modules.pop("folder_paths", None)
                else:
                    sys.modules["folder_paths"] = previous

        destinations = {destination.name: destination for _, destination, *_ in calls}
        self.assertEqual(set(destinations), EXPECTED_FILENAMES)
        self.assertEqual(
            {path.parent.name for path in destinations.values()},
            set(EXPECTED_FOLDERS.values()),
        )
        for _, destination, expected_sha256, expected_size, *_ in calls:
            self.assertEqual(expected_sha256, EXPECTED_SHA256[destination.name])
            self.assertEqual(expected_size, EXPECTED_SIZES[destination.name])
        self.assertIsNotNone(result)

    def test_download_is_atomic_and_checksum_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            content = b"triposplat model bytes" * 1024
            digest = hashlib.sha256(content).hexdigest()
            destination = Path(directory) / "triposplat_fp16.safetensors"
            with mock.patch.object(
                self.download,
                "_download_with_hub",
                return_value=False,
            ), mock.patch.object(
                self.download.urllib.request,
                "urlopen",
                return_value=FakeResponse(content),
            ):
                result = self.download.download_file(
                    url="https://huggingface.co/VAST-AI/TripoSplat/resolve/revision/model",
                    destination=destination,
                    expected_sha256=digest,
                    expected_size=len(content),
                )

            self.assertEqual(result, destination)
            self.assertEqual(destination.read_bytes(), content)
            self.assertFalse(destination.with_suffix(".safetensors.part").exists())
            marker = destination.with_suffix(".safetensors.sha256").read_text(
                encoding="ascii"
            )
            self.assertEqual(marker, f"{digest} {len(content)}\n")

    def test_existing_verified_file_skips_network(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "cached.safetensors"
            content = b"cached public asset"
            digest = hashlib.sha256(content).hexdigest()
            destination.write_bytes(content)
            destination.with_suffix(".safetensors.sha256").write_text(
                f"{digest} {len(content)}\n",
                encoding="ascii",
            )

            with mock.patch.object(self.download.urllib.request, "urlopen") as urlopen:
                result = self.download.download_file(
                    url="https://invalid.example.test/cached.safetensors",
                    destination=destination,
                    expected_sha256=digest,
                    expected_size=len(content),
                )

            self.assertEqual(result.read_bytes(), content)
            urlopen.assert_not_called()

    def test_same_size_corruption_is_rehashed_despite_valid_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "cached.safetensors"
            content = b"verified public asset"
            digest = hashlib.sha256(content).hexdigest()
            destination.write_bytes(b"x" * len(content))
            destination.with_suffix(".safetensors.sha256").write_text(
                f"{digest} {len(content)}\n",
                encoding="ascii",
            )

            with mock.patch.object(
                self.download.urllib.request,
                "urlopen",
                return_value=FakeResponse(content),
            ) as urlopen:
                result = self.download.download_file(
                    url="https://example.test/cached.safetensors",
                    destination=destination,
                    expected_sha256=digest,
                    expected_size=len(content),
                )

            self.assertEqual(result.read_bytes(), content)
            urlopen.assert_called_once()

    def test_transient_failure_preserves_partial_and_resumes_with_range(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "resumable.safetensors"
            partial = destination.with_suffix(".safetensors.part")
            partial.write_bytes(b"first-")
            content = b"first-second"
            digest = hashlib.sha256(content).hexdigest()
            requests = []

            def open_request(request, timeout):
                self.assertEqual(timeout, 120)
                requests.append(request)
                if len(requests) == 1:
                    raise urllib.error.HTTPError(
                        request.full_url,
                        503,
                        "Service Unavailable",
                        {},
                        None,
                    )
                return FakeResponse(b"second", status=206)

            with mock.patch.object(
                self.download.urllib.request,
                "urlopen",
                side_effect=open_request,
            ), mock.patch.object(self.download.time, "sleep"):
                result = self.download.download_file(
                    url="https://example.test/resumable.safetensors",
                    destination=destination,
                    expected_sha256=digest,
                    expected_size=len(content),
                )

            self.assertEqual(result.read_bytes(), content)
            self.assertEqual(
                [request.get_header("Range") for request in requests],
                ["bytes=6-", "bytes=6-"],
            )

    def test_short_response_and_incomplete_read_are_resumable(self):
        class InterruptedResponse:
            status = 200
            headers = {"Content-Length": "12"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self, _size):
                raise http.client.IncompleteRead(b"first-", 6)

        for first_response in (
            FakeResponse(b"first-", declared_length=12),
            InterruptedResponse(),
        ):
            with self.subTest(first_response=type(first_response).__name__), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "resume.safetensors"
                content = b"first-second"
                digest = hashlib.sha256(content).hexdigest()
                requests = []

                def open_request(request, timeout):
                    requests.append(request)
                    if len(requests) == 1:
                        return first_response
                    return FakeResponse(b"second", status=206)

                with mock.patch.object(
                    self.download.urllib.request,
                    "urlopen",
                    side_effect=open_request,
                ), mock.patch.object(self.download.time, "sleep"):
                    result = self.download.download_file(
                        url="https://example.test/resume.safetensors",
                        destination=destination,
                        expected_sha256=digest,
                        expected_size=len(content),
                    )

                self.assertEqual(result.read_bytes(), content)
                self.assertEqual(requests[1].get_header("Range"), "bytes=6-")

    def test_stale_hf_token_retries_public_assets_anonymously(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "public.safetensors"
            content = b"public model bytes"
            digest = hashlib.sha256(content).hexdigest()
            requests = []

            def open_request(request, timeout):
                requests.append(request)
                if len(requests) == 1:
                    raise urllib.error.HTTPError(
                        request.full_url,
                        403,
                        "Forbidden",
                        {},
                        None,
                    )
                return FakeResponse(content)

            with mock.patch.dict(
                os.environ,
                {"HF_TOKEN": "stale-token"},
                clear=False,
            ), mock.patch.object(
                self.download,
                "_download_with_hub",
                return_value=False,
            ), mock.patch.object(
                self.download.urllib.request,
                "urlopen",
                side_effect=open_request,
            ), mock.patch.object(self.download.time, "sleep"):
                result = self.download.download_file(
                    url="https://huggingface.co/VAST-AI/TripoSplat/resolve/revision/public",
                    destination=destination,
                    expected_sha256=digest,
                    expected_size=len(content),
                )

            self.assertEqual(result.read_bytes(), content)
            self.assertEqual(requests[0].get_header("Authorization"), "Bearer stale-token")
            self.assertIsNone(requests[1].get_header("Authorization"))

    def test_checksum_mismatch_never_promotes_partial_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "bad.safetensors"
            with mock.patch.object(
                self.download.urllib.request,
                "urlopen",
                return_value=FakeResponse(b"corrupt"),
            ), mock.patch.object(self.download.time, "sleep"):
                with self.assertRaisesRegex(self.download.DownloadError, "Checksum mismatch"):
                    self.download.download_file(
                        url="https://example.test/bad.safetensors",
                        destination=destination,
                        expected_sha256="0" * 64,
                        expected_size=len(b"corrupt"),
                        attempts=1,
                    )

            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix(".safetensors.sha256").exists())
            self.assertFalse(destination.with_suffix(".safetensors.part").exists())


if __name__ == "__main__":
    unittest.main()
