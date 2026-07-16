from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "worker/pixal3d/artifacts.py"


def load_artifacts():
    name = "comfycolab_pixal3d_artifacts_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, ARTIFACTS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Pixal3DArtifactTests(unittest.TestCase):
    def test_snapshot_manifest_rejects_and_repairs_corrupted_non_sentinel_file(self) -> None:
        artifacts = load_artifacts()
        calls: list[tuple[str, str]] = []

        def snapshot_download(*, repo_id, revision, local_dir, **_kwargs):
            calls.append((repo_id, revision))
            root = Path(local_dir)
            root.mkdir(parents=True, exist_ok=True)
            (root / "pipeline.json").write_text("{}", encoding="utf-8")
            (root / "weights.bin").write_bytes(b"verified-weights")
            return str(root)

        fake_hub = types.SimpleNamespace(snapshot_download=snapshot_download)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules, {"huggingface_hub": fake_hub}
        ):
            destination = Path(directory) / "snapshot"
            artifacts._ensure_snapshot(
                repo_id="owner/model",
                revision="a" * 40,
                destination=destination,
                sentinel="pipeline.json",
                progress=lambda _event: None,
            )
            (destination / "weights.bin").write_bytes(b"corrupt")
            artifacts._ensure_snapshot(
                repo_id="owner/model",
                revision="a" * 40,
                destination=destination,
                sentinel="pipeline.json",
                progress=lambda _event: None,
            )
            artifacts._ensure_snapshot(
                repo_id="owner/model",
                revision="a" * 40,
                destination=destination,
                sentinel="pipeline.json",
                progress=lambda _event: None,
            )
            self.assertEqual((destination / "weights.bin").read_bytes(), b"verified-weights")

        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
