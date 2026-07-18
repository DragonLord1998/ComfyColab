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
    def test_moge_snapshot_accepts_model_checkpoint_without_config_json(self) -> None:
        artifacts = load_artifacts()

        def snapshot_download(*, repo_id, revision, local_dir, **_kwargs):
            root = Path(local_dir)
            root.mkdir(parents=True, exist_ok=True)
            if repo_id == artifacts.PIXAL3D_MODEL_REPO:
                (root / "pipeline.json").write_text("{}", encoding="utf-8")
            elif repo_id == artifacts.DINOV3_MODEL_REPO:
                (root / "config.json").write_text("{}", encoding="utf-8")
            elif repo_id == artifacts.MOGE_MODEL_REPO:
                (root / "model.pt").write_bytes(b"checkpoint-with-embedded-config")
            else:
                self.fail(f"Unexpected snapshot repository: {repo_id}")
            return str(root)

        fake_hub = types.SimpleNamespace(snapshot_download=snapshot_download)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules, {"huggingface_hub": fake_hub}
        ), mock.patch.object(
            artifacts,
            "_ensure_naf_source",
            return_value=Path(directory) / "naf-source",
        ), mock.patch.object(
            artifacts,
            "_download_verified",
            return_value=Path(directory) / "naf_release.pth",
        ), mock.patch.object(
            artifacts.shutil,
            "disk_usage",
            return_value=types.SimpleNamespace(free=artifacts.MIN_FREE_BYTES),
        ):
            provisioned = artifacts.ensure_pixal3d_artifacts(Path(directory) / "models")
            self.assertTrue((provisioned.moge_dir / "model.pt").is_file())
            self.assertFalse((provisioned.moge_dir / "config.json").exists())

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
