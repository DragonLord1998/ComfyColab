from __future__ import annotations

import importlib.util
import json
import os
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
        calls: list[tuple[str, str, object]] = []

        def snapshot_download(
            *,
            repo_id,
            revision,
            local_dir,
            allow_patterns=None,
            token=None,
            **_kwargs,
        ):
            calls.append((repo_id, revision, token))
            root = Path(local_dir)
            root.mkdir(parents=True, exist_ok=True)
            files = allow_patterns or ["config.json"]
            for relative in files:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{repo_id}@{revision}:{relative}".encode())
            return str(root)

        fake_hub = types.ModuleType("huggingface_hub")

        def hub_getattr(name):
            if name == "snapshot_download":
                self.assertEqual(os.environ["HF_XET_HIGH_PERFORMANCE"], "1")
                self.assertEqual(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"], "120")
                return snapshot_download
            raise AttributeError(name)

        fake_hub.__getattr__ = hub_getattr
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules, {"huggingface_hub": fake_hub}
        ), mock.patch.dict(
            os.environ,
            {"HF_TOKEN": "test-token"},
            clear=False,
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
            self.assertEqual(os.environ["HF_XET_HIGH_PERFORMANCE"], "1")
            self.assertEqual(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"], "120")

        self.assertEqual(len(calls), 2)
        self.assertEqual([call[2] for call in calls], ["test-token", "test-token"])

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

    def test_environment_uses_real_managed_python311_and_pinned_bpy(self) -> None:
        artifacts = load_artifacts()
        calls: list[tuple[list[str], dict]] = []
        versions = {
            "python": "3.11.15",
            "bpy": "4.2.22",
            "diffusers": "0.37.1",
            "flash_attn": "2.8.3+cu12torch2.7cxx11abiTRUE",
            "numpy": "1.26.4",
            "torch": "2.7.0+cu128",
            "transformers": "4.57.3",
        }

        def check_call(argv, **kwargs):
            values = [str(value) for value in argv]
            calls.append((values, kwargs))
            if values[:2] == ["/usr/local/bin/uv", "venv"]:
                python = Path(values[-1]) / "bin/python"
                python.parent.mkdir(parents=True, exist_ok=True)
                python.write_text("", encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            artifacts.shutil, "which", return_value="/usr/local/bin/uv"
        ), mock.patch.object(
            artifacts.subprocess, "check_call", side_effect=check_call
        ), mock.patch.object(
            artifacts, "_probe_environment", return_value=versions
        ) as probe_environment:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "requirements.txt").write_text("bpy>=4.2\n", encoding="utf-8")
            env_dir = root / "envs/skintokens"
            stale_python = env_dir / "bin/python"
            stale_python.parent.mkdir(parents=True)
            stale_python.write_text("", encoding="utf-8")
            (env_dir / ".comfycolab-environment.json").write_text(
                json.dumps(
                    {
                        "schema": artifacts.ARTIFACT_SCHEMA,
                        "environment_ref": artifacts.SKINTOKENS_ENVIRONMENT_REF,
                        "python_version": artifacts.SKINTOKENS_PYTHON_VERSION,
                        "versions": {"python": "0.0.0"},
                    }
                ),
                encoding="utf-8",
            )
            python = artifacts._ensure_environment(
                source, env_dir, lambda _event: None
            )
            marker = json.loads(
                (env_dir / ".comfycolab-environment.json").read_text()
            )

        self.assertEqual(python, env_dir / "bin/python")
        self.assertEqual(marker["versions"], versions)
        self.assertEqual(marker["python_version"], "3.11.15")
        self.assertGreaterEqual(probe_environment.call_count, 2)
        uv_call = calls[0][0]
        self.assertEqual(uv_call[:2], ["/usr/local/bin/uv", "venv"])
        self.assertIn("3.11.15", uv_call)
        self.assertIn("--managed-python", uv_call)
        self.assertTrue(
            any("bpy==4.2.22" in argv for argv, _kwargs in calls)
        )
        flash_call = next(
            (argv, kwargs)
            for argv, kwargs in calls
            if artifacts.SKINTOKENS_FLASH_ATTN_PACKAGE in argv
        )
        self.assertIn("--no-deps", flash_call[0])
        self.assertNotIn("--no-build-isolation", flash_call[0])
        self.assertNotIn("env", flash_call[1])

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
        self.assertEqual(artifacts.SKINTOKENS_PYTHON_VERSION, "3.11.15")
        self.assertEqual(
            artifacts.SKINTOKENS_ENVIRONMENT_REF,
            "g4-linux64-py31115-torch270-cu128-bpy4222-skintokens-v3",
        )
        self.assertIn("bpy==4.2.22", artifacts.SKINTOKENS_RUNTIME_PINS)
        self.assertTrue(
            artifacts.SKINTOKENS_FLASH_ATTN_PACKAGE.endswith(
                f"#sha256={artifacts.SKINTOKENS_FLASH_ATTN_WHEEL_SHA256}"
            )
        )
        self.assertEqual(artifacts.SKINTOKENS_LICENSE["name"], "MIT")


if __name__ == "__main__":
    unittest.main()
