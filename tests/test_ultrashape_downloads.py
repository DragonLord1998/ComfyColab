from __future__ import annotations

import hashlib
import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "worker" / "ultrashape" / "artifacts.py"


def load_artifacts():
    name = "comfycolab_ultrashape_artifacts_test"
    spec = importlib.util.spec_from_file_location(name, ARTIFACTS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, payload: bytes, *, status: int, headers: dict[str, str]):
        self._payload = io.BytesIO(payload)
        self.status = status
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int) -> bytes:
        return self._payload.read(size)


class UltraShapeArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifacts = load_artifacts()

    def test_model_revisions_and_checkpoint_checksum_are_pinned(self) -> None:
        artifacts = self.artifacts
        self.assertEqual(
            artifacts.ULTRASHAPE_REVISION,
            "5aeb21a7185d39f042d02b2695802f125a6f5159",
        )
        self.assertEqual(
            artifacts.DINOV2_REVISION,
            "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
        )
        self.assertEqual(
            artifacts.ULTRASHAPE_CHECKPOINT.expected_sha256,
            "c96ae010c4169597fd0006dcb08056bf6104a1fca249b10fed7ddded324c3f0f",
        )
        self.assertIn(artifacts.ULTRASHAPE_REVISION, artifacts.ULTRASHAPE_CHECKPOINT.url)
        self.assertEqual(
            tuple(item.filename for item in artifacts.DINOV2_ARTIFACTS),
            ("config.json", "model.safetensors", "preprocessor_config.json"),
        )
        self.assertEqual(artifacts.ULTRASHAPE_CHECKPOINT.expected_size, 7_366_231_254)
        self.assertEqual(
            artifacts.DINOV2_ARTIFACTS[1].expected_sha256,
            "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23",
        )
        self.assertEqual(artifacts.DINOV2_ARTIFACTS[1].expected_size, 1_217_522_888)

    def test_default_progress_protocol_is_worker_machine_readable(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.artifacts.print_progress(
                {"stage": "artifact_download", "percent": 50.0}
            )
        self.assertTrue(output.getvalue().startswith("COMFYCOLAB_PROGRESS="))

    def test_resumes_partial_and_reports_percentage_speed_and_eta(self) -> None:
        artifacts = self.artifacts
        payload = b"already-" + b"remaining"
        expected = hashlib.sha256(payload).hexdigest()
        spec = artifacts.ArtifactSpec(
            name="fixture",
            repository="example/model",
            revision="a" * 40,
            filename="fixture.bin",
            expected_sha256=expected,
            expected_size=len(payload),
        )
        events: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / spec.filename
            destination.with_suffix(".bin.partial").write_bytes(b"already-")

            def urlopen(request, timeout):
                self.assertEqual(request.get_header("Range"), "bytes=8-")
                self.assertEqual(timeout, 0.25)
                return FakeResponse(
                    b"remaining",
                    status=206,
                    headers={
                        "Content-Length": "9",
                        "Content-Range": "bytes 8-16/17",
                    },
                )

            with mock.patch.object(artifacts.urllib.request, "urlopen", side_effect=urlopen):
                result = artifacts.download_artifact(
                    spec,
                    destination,
                    progress=events.append,
                    timeout=0.25,
                    progress_interval=0,
                    retry_delay=0,
                )

            self.assertEqual(result.read_bytes(), payload)
            self.assertFalse(destination.with_suffix(".bin.partial").exists())
            self.assertTrue(destination.with_suffix(".bin.sha256").is_file())
        complete = events[-1]
        self.assertEqual(complete["status"], "complete")
        self.assertEqual(complete["percent"], 100.0)
        self.assertIsInstance(complete["bytes_per_second"], float)
        self.assertEqual(complete["eta_seconds"], 0.0)

    def test_retries_an_early_eof_from_the_retained_partial(self) -> None:
        artifacts = self.artifacts
        payload = b"abcdefghij"
        spec = artifacts.ArtifactSpec(
            name="stall-fixture",
            repository="example/model",
            revision="b" * 40,
            filename="stall.bin",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_size=len(payload),
        )
        requests = []
        responses = [
            FakeResponse(b"abcd", status=200, headers={"Content-Length": "10"}),
            FakeResponse(
                b"efghij",
                status=206,
                headers={"Content-Length": "6", "Content-Range": "bytes 4-9/10"},
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / spec.filename

            def urlopen(request, timeout):
                requests.append(request)
                return responses.pop(0)

            events: list[dict[str, object]] = []
            with mock.patch.object(artifacts.urllib.request, "urlopen", side_effect=urlopen):
                artifacts.download_artifact(
                    spec,
                    destination,
                    progress=events.append,
                    progress_interval=0,
                    retry_delay=0,
                )

            self.assertEqual(destination.read_bytes(), payload)
        self.assertIsNone(requests[0].get_header("Range"))
        self.assertEqual(requests[1].get_header("Range"), "bytes=4-")
        self.assertIn("retrying", [event["status"] for event in events])

    def test_corrupt_published_file_is_replaced_and_new_digest_is_recorded(self) -> None:
        artifacts = self.artifacts
        payload = b"valid-artifact"
        spec = artifacts.ArtifactSpec(
            name="corruption-fixture",
            repository="example/model",
            revision="c" * 40,
            filename="model.bin",
            expected_sha256=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / spec.filename
            destination.write_bytes(b"corrupt")
            destination.with_suffix(".bin.sha256").write_text(
                f"{'0' * 64} 7\n", encoding="ascii"
            )
            response = FakeResponse(
                payload,
                status=200,
                headers={"Content-Length": str(len(payload))},
            )
            with mock.patch.object(artifacts.urllib.request, "urlopen", return_value=response):
                artifacts.download_artifact(
                    spec,
                    destination,
                    progress=lambda _event: None,
                    retry_delay=0,
                )

            self.assertEqual(destination.read_bytes(), payload)
            recorded = destination.with_suffix(".bin.sha256").read_text(encoding="ascii")
            self.assertTrue(recorded.startswith(hashlib.sha256(payload).hexdigest()))

    def test_unverifiable_range_error_discards_partial_before_retry(self) -> None:
        artifacts = self.artifacts
        payload = b"fresh-copy"
        spec = artifacts.ArtifactSpec(
            name="range-fixture",
            repository="example/model",
            revision="d" * 40,
            filename="range.bin",
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / spec.filename
            destination.with_suffix(".bin.partial").write_bytes(b"bad-partial")
            range_error = artifacts.urllib.error.HTTPError(
                spec.url,
                416,
                "Range Not Satisfiable",
                {},
                None,
            )
            response = FakeResponse(
                payload,
                status=200,
                headers={"Content-Length": str(len(payload))},
            )
            requests = []

            def urlopen(request, timeout):
                requests.append(request)
                if len(requests) == 1:
                    raise range_error
                return response

            with mock.patch.object(artifacts.urllib.request, "urlopen", side_effect=urlopen):
                artifacts.download_artifact(
                    spec,
                    destination,
                    progress=lambda _event: None,
                    retry_delay=0,
                )

            self.assertEqual(destination.read_bytes(), payload)
        self.assertEqual(requests[0].get_header("Range"), "bytes=11-")
        self.assertIsNone(requests[1].get_header("Range"))

    def test_ensure_returns_worker_ready_paths_at_exact_revisions(self) -> None:
        artifacts = self.artifacts
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def fake_download(specification, destination, **_kwargs):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"fixture")
                return destination

            with mock.patch.object(
                artifacts, "download_artifact", side_effect=fake_download
            ) as call:
                result = artifacts.ensure_ultrashape_artifacts(root, progress=lambda _event: None)

            self.assertEqual(call.call_count, 4)
            self.assertEqual(result.checkpoint.name, "ultrashape_v1.pt")
            self.assertIn(artifacts.ULTRASHAPE_REVISION, result.checkpoint.parts)
            self.assertIn(artifacts.DINOV2_REVISION, result.dinov2_dir.parts)
            self.assertTrue((result.dinov2_dir / "model.safetensors").is_file())


if __name__ == "__main__":
    unittest.main()
