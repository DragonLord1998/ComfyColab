from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "worker/skintokens/artifacts.py"


def load_artifacts():
    name = "comfycolab_skintokens_artifacts_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, ARTIFACTS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SkinTokensArtifactTests(unittest.TestCase):
    def test_snapshot_manifest_rejects_corrupted_checkpoint_and_redownloads(self) -> None:
        artifacts = load_artifacts()
        calls: list[tuple[str, str]] = []

        def snapshot_download(*, repo_id, revision, local_dir, allow_patterns=None, **_kwargs):
            calls.append((repo_id, revision))
            root = Path(local_dir)
            root.mkdir(parents=True, exist_ok=True)
            files = allow_patterns or ["config.json"]
            for relative in files:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{repo_id}@{revision}:{relative}".encode())
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
                allow_patterns=["experiments/model.ckpt"],
                ignore_patterns=None,
                sentinel="experiments/model.ckpt",
                progress=lambda _event: None,
            )
            (destination / "experiments/model.ckpt").write_bytes(b"corrupt")
            artifacts._ensure_snapshot(
                repo_id="owner/model",
                revision="a" * 40,
                destination=destination,
                allow_patterns=["experiments/model.ckpt"],
                ignore_patterns=None,
                sentinel="experiments/model.ckpt",
                progress=lambda _event: None,
            )
            self.assertIn("owner/model", (destination / "experiments/model.ckpt").read_text())

        self.assertEqual(len(calls), 2)

    def test_environment_skip_uses_current_python_without_main_env_mutation(self) -> None:
        artifacts = load_artifacts()
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            artifacts.os.environ, {"COMFYCOLAB_SKINTOKENS_SKIP_ENV_INSTALL": "1"}
        ), mock.patch.object(artifacts.subprocess, "check_call") as check_call:
            env_dir = Path(directory) / "env"
            python = artifacts._ensure_environment(Path(directory), env_dir, lambda _event: None)
            marker = json.loads((env_dir / ".comfycolab-environment.json").read_text())

        self.assertEqual(python, Path(sys.executable))
        self.assertTrue(marker["skipped"])
        check_call.assert_not_called()

    def test_constants_pin_license_and_revisions(self) -> None:
        artifacts = load_artifacts()

        self.assertEqual(
            artifacts.SKINTOKENS_SOURCE_REF,
            "273b691d35989d71cd17ff2895fdc735097b92d1",
        )
        self.assertEqual(
            artifacts.SKINTOKENS_MODEL_REF,
            "79736cad0fd84de384d5eede659b4ebd24effe33",
        )
        self.assertEqual(artifacts.SKINTOKENS_LICENSE["name"], "MIT")


if __name__ == "__main__":
    unittest.main()
