from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
import tempfile
import unittest
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
    status = 200

    def __init__(self, content: bytes):
        super().__init__(content)
        self.headers = {"Content-Length": str(len(content))}

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


if __name__ == "__main__":
    unittest.main()
