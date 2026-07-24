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
NODE_ROOT = ROOT / "custom_nodes" / "ComfyColab-ZImage"


def load_download_module():
    name = "comfycolab_download_test"
    spec = importlib.util.spec_from_file_location(name, NODE_ROOT / "download.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.download = load_download_module()

    def test_download_is_atomic_and_checksum_verified(self) -> None:
        with tempfile.TemporaryDirectory() as target_directory:
            content = b"verified model bytes" * 1024
            digest = hashlib.sha256(content).hexdigest()
            destination = Path(target_directory) / "model.gguf"
            with mock.patch.object(
                self.download.urllib.request,
                "urlopen",
                return_value=FakeResponse(content),
            ):
                result = self.download.download_file(
                    url="https://example.test/model.gguf",
                    destination=destination,
                    expected_sha256=digest,
                )

            self.assertEqual(result.read_bytes(), content)
            self.assertFalse(destination.with_suffix(".gguf.part").exists())
            marker = destination.with_suffix(".gguf.sha256").read_text(encoding="ascii")
            self.assertIn(digest, marker)

    def test_huggingface_download_uses_hub_xet_primary_without_urllib(self) -> None:
        with tempfile.TemporaryDirectory() as target_directory:
            directory = Path(target_directory)
            content = b"hub cached model bytes"
            digest = hashlib.sha256(content).hexdigest()
            cache_file = directory / "cache" / "model.gguf"
            cache_file.parent.mkdir()
            cache_file.write_bytes(content)
            destination = directory / "model.gguf"
            destination.with_suffix(".gguf.part").write_bytes(b"stale partial")
            calls = []

            fake_hub = types.ModuleType("huggingface_hub")

            def hf_hub_download(**kwargs):
                calls.append(kwargs)
                return str(cache_file)

            fake_hub.hf_hub_download = hf_hub_download
            with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}), mock.patch.dict(
                os.environ, {"HF_TOKEN": "valid-token"}, clear=False
            ), mock.patch.object(
                self.download.urllib.request,
                "urlopen",
                side_effect=AssertionError("urllib fallback should not run"),
            ):
                result = self.download.download_file(
                    url=(
                        "https://huggingface.co/org/model/resolve/"
                        "0123456789abcdef0123456789abcdef01234567/nested/model.gguf"
                        "?download=true"
                    ),
                    destination=destination,
                    expected_sha256=digest,
                )
                self.assertEqual(os.environ["HF_XET_HIGH_PERFORMANCE"], "1")
                self.assertEqual(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"], "120")
                self.assertEqual(os.environ["HF_HUB_ETAG_TIMEOUT"], "60")

            self.assertEqual(result.read_bytes(), content)
            self.assertFalse(destination.with_suffix(".gguf.part").exists())
            self.assertEqual(
                calls,
                [
                    {
                        "repo_id": "org/model",
                        "filename": "nested/model.gguf",
                        "revision": "0123456789abcdef0123456789abcdef01234567",
                        "token": "valid-token",
                        "force_download": False,
                        "local_dir": str(directory / ".model.gguf.hf-xet"),
                    }
                ],
            )

    def test_existing_verified_file_skips_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "cached.gguf"
            content = b"cached"
            destination.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            destination.with_suffix(".gguf.sha256").write_text(
                f"{digest} {len(content)}\n",
                encoding="ascii",
            )
            result = self.download.download_file(
                url="https://invalid.example.test/cached.gguf",
                destination=destination,
                expected_sha256=digest,
            )
            self.assertEqual(result.read_bytes(), content)

    def test_hub_stale_token_retries_anonymously_before_urllib_fallback(self) -> None:
        class StaleTokenError(RuntimeError):
            response = types.SimpleNamespace(status_code=403)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "public.gguf"
            content = b"public model bytes"
            digest = hashlib.sha256(content).hexdigest()
            calls = []

            def hub_download(**kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    raise StaleTokenError("stale")
                destination.with_suffix(".gguf.part").write_bytes(content)
                destination.with_suffix(".gguf.part").replace(destination)
                return destination

            with mock.patch.dict(os.environ, {"HF_TOKEN": "stale-token"}, clear=False), mock.patch.object(
                self.download._hf_download,
                "_download_with_hub",
                side_effect=hub_download,
            ), mock.patch.object(
                self.download.urllib.request,
                "urlopen",
                side_effect=AssertionError("urllib fallback should not run"),
            ):
                result = self.download.download_file(
                    url="https://huggingface.co/public/model/resolve/revision/model.gguf",
                    destination=destination,
                    expected_sha256=digest,
                )

            self.assertEqual(result.read_bytes(), content)
            self.assertEqual([call["include_auth"] for call in calls], [True, False])

    def test_urllib_fallback_retries_stale_token_anonymously(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "public.gguf"
            content = b"public model bytes"
            digest = hashlib.sha256(content).hexdigest()
            requests = []

            def open_request(request, timeout):
                self.assertEqual(timeout, 120)
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

            with mock.patch.dict(os.environ, {"HF_TOKEN": "stale-token"}, clear=False), mock.patch.object(
                self.download._hf_download,
                "_download_with_hub",
                side_effect=ImportError("missing huggingface_hub"),
            ), mock.patch.object(
                self.download.urllib.request,
                "urlopen",
                side_effect=open_request,
            ), mock.patch.object(self.download.time, "sleep") as sleep:
                result = self.download.download_file(
                    url="https://huggingface.co/public/model/resolve/revision/model.gguf",
                    destination=destination,
                    expected_sha256=digest,
                )

            self.assertEqual(result.read_bytes(), content)
            self.assertEqual(requests[0].get_header("Authorization"), "Bearer stale-token")
            self.assertIsNone(requests[1].get_header("Authorization"))
            self.assertEqual(requests[1].get_header("Cache-control"), "no-cache")
            sleep.assert_called_once_with(2)

    def test_transient_failure_preserves_and_resumes_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "resumable.gguf"
            partial = destination.with_suffix(".gguf.part")
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
                    url="https://example.test/resumable.gguf",
                    destination=destination,
                    expected_sha256=digest,
                )

            self.assertEqual(result.read_bytes(), content)
            self.assertEqual(
                [request.get_header("Range") for request in requests],
                ["bytes=6-", "bytes=6-"],
            )

    def test_permanent_http_error_fails_without_retrying(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "missing.gguf"
            error = urllib.error.HTTPError(
                "https://example.test/missing.gguf",
                404,
                "Not Found",
                {},
                None,
            )
            with mock.patch.object(
                self.download.urllib.request,
                "urlopen",
                side_effect=error,
            ) as urlopen, mock.patch.object(self.download.time, "sleep") as sleep:
                with self.assertRaisesRegex(self.download.DownloadError, "not retryable"):
                    self.download.download_file(
                        url="https://example.test/missing.gguf",
                        destination=destination,
                        expected_sha256="0" * 64,
                    )
            urlopen.assert_called_once()
            sleep.assert_not_called()

    def test_unverifiable_range_restarts_from_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "range-reset.gguf"
            partial = destination.with_suffix(".gguf.part")
            partial.write_bytes(b"stale-partial")
            content = b"current model"
            digest = hashlib.sha256(content).hexdigest()
            requests = []

            def open_request(request, timeout):
                self.assertEqual(timeout, 120)
                requests.append(request)
                if len(requests) == 1:
                    raise urllib.error.HTTPError(
                        request.full_url,
                        416,
                        "Range Not Satisfiable",
                        {},
                        None,
                    )
                return FakeResponse(content)

            with mock.patch.object(
                self.download.urllib.request,
                "urlopen",
                side_effect=open_request,
            ), mock.patch.object(self.download.time, "sleep"):
                result = self.download.download_file(
                    url="https://example.test/range-reset.gguf",
                    destination=destination,
                    expected_sha256=digest,
                )

            self.assertEqual(result.read_bytes(), content)
            self.assertEqual(requests[0].get_header("Range"), "bytes=13-")
            self.assertIsNone(requests[1].get_header("Range"))

    def test_short_declared_response_resumes_retained_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "short-response.gguf"
            content = b"first-second"
            digest = hashlib.sha256(content).hexdigest()
            requests = []

            def open_request(request, timeout):
                self.assertEqual(timeout, 120)
                requests.append(request)
                if len(requests) == 1:
                    return FakeResponse(b"first-", declared_length=len(content))
                return FakeResponse(b"second", status=206)

            with mock.patch.object(
                self.download.urllib.request,
                "urlopen",
                side_effect=open_request,
            ), mock.patch.object(self.download.time, "sleep"):
                result = self.download.download_file(
                    url="https://example.test/short-response.gguf",
                    destination=destination,
                    expected_sha256=digest,
                )

            self.assertEqual(result.read_bytes(), content)
            self.assertIsNone(requests[0].get_header("Range"))
            self.assertEqual(requests[1].get_header("Range"), "bytes=6-")

    def test_incomplete_read_bytes_are_retained_for_resume(self) -> None:
        class InterruptedResponse:
            status = 200
            headers = {"Content-Length": "12"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self, _size):
                raise http.client.IncompleteRead(b"first-", 6)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "incomplete-read.gguf"
            content = b"first-second"
            digest = hashlib.sha256(content).hexdigest()
            requests = []

            def open_request(request, timeout):
                self.assertEqual(timeout, 120)
                requests.append(request)
                if len(requests) == 1:
                    return InterruptedResponse()
                return FakeResponse(b"second", status=206)

            with mock.patch.object(
                self.download.urllib.request,
                "urlopen",
                side_effect=open_request,
            ), mock.patch.object(self.download.time, "sleep"):
                result = self.download.download_file(
                    url="https://example.test/incomplete-read.gguf",
                    destination=destination,
                    expected_sha256=digest,
                )

            self.assertEqual(result.read_bytes(), content)
            self.assertEqual(requests[1].get_header("Range"), "bytes=6-")


if __name__ == "__main__":
    unittest.main()
