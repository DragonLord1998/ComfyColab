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


def load_artifacts():
    path = ROOT / "worker/cubepart/artifacts.py"
    spec = importlib.util.spec_from_file_location("comfycolab_cubepart_artifacts_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CubePartArtifactProvisioningTests(unittest.TestCase):
    def test_model_snapshot_uses_authenticated_high_performance_xet(self) -> None:
        artifacts = load_artifacts()
        calls: list[dict[str, object]] = []

        def snapshot_download(**kwargs):
            calls.append(kwargs)
            weights = Path(str(kwargs["local_dir"]))
            weights.mkdir(parents=True, exist_ok=True)
            (weights / artifacts.CUBEPART_CHECKPOINT).write_bytes(b"dit")
            (weights / artifacts.CUBEPART_VAE_CHECKPOINT).write_bytes(b"vae")
            return str(weights)

        fake_hub = types.ModuleType("huggingface_hub")

        def hub_getattr(name):
            if name == "snapshot_download":
                self.assertEqual(os.environ["HF_XET_HIGH_PERFORMANCE"], "1")
                self.assertEqual(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"], "120")
                return snapshot_download
            raise AttributeError(name)

        fake_hub.__getattr__ = hub_getattr
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cube" / "cubepart"
            source.mkdir(parents=True)
            (source / "pyproject.toml").write_text(
                "[project]\nname='cube_part'\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                artifacts,
                "_git_revision",
                return_value=artifacts.CUBEPART_SOURCE_REF,
            ), mock.patch.object(
                artifacts.shutil,
                "disk_usage",
                return_value=types.SimpleNamespace(free=artifacts.MIN_FREE_BYTES),
            ), mock.patch.dict(
                sys.modules,
                {"huggingface_hub": fake_hub},
            ), mock.patch.dict(
                os.environ,
                {
                    "HF_TOKEN": "test-token",
                    "COMFYCOLAB_CUBEPART_SKIP_ENV_INSTALL": "1",
                },
                clear=False,
            ):
                artifacts.ensure_cubepart_artifacts(
                    accept_research_license=True,
                    source_dir=source,
                    environment_dir=root / "env",
                    weights_root=root / "models",
                )
                self.assertEqual(os.environ["HF_XET_HIGH_PERFORMANCE"], "1")
                self.assertEqual(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"], "120")

        self.assertEqual(calls[0]["token"], "test-token")
        self.assertIn(
            "huggingface_hub[hf_xet]>=0.36.0,<2",
            artifacts.CUBEPART_RUNTIME_REQUIREMENTS,
        )

    def test_license_gate_runs_before_any_provisioning(self) -> None:
        artifacts = load_artifacts()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(artifacts.Path, "mkdir") as mkdir:
                with self.assertRaisesRegex(PermissionError, "Research-Only"):
                    artifacts.ensure_cubepart_artifacts(
                        accept_research_license=False,
                        source_dir=root / "source",
                        environment_dir=root / "env",
                        weights_root=root / "models",
                    )
        mkdir.assert_not_called()

    def test_license_metadata_records_code_and_weights_terms(self) -> None:
        artifacts = load_artifacts()
        metadata = artifacts.cubepart_license_metadata()

        self.assertEqual(metadata["source_ref"], artifacts.CUBEPART_SOURCE_REF)
        self.assertEqual(metadata["weights_repo"], "Roblox/cubepart")
        self.assertEqual(metadata["weights_ref"], artifacts.CUBEPART_MODEL_REF)
        self.assertIn("RAIL", metadata["source_license"])
        self.assertEqual(metadata["required_acceptance"], "accept_research_license")

    def test_environment_skip_returns_worker_python_after_source_and_pin_validation(self) -> None:
        artifacts = load_artifacts()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cube" / "cubepart"
            source.mkdir(parents=True)
            (source / "pyproject.toml").write_text("[project]\nname='cube_part'\n")
            weights = root / "models" / f"cubepart-{artifacts.CUBEPART_MODEL_REF[:12]}"
            weights.mkdir(parents=True)
            (weights / artifacts.CUBEPART_CHECKPOINT).write_bytes(b"dit")
            (weights / artifacts.CUBEPART_VAE_CHECKPOINT).write_bytes(b"vae")
            with mock.patch.object(
                artifacts, "_git_revision", return_value=artifacts.CUBEPART_SOURCE_REF
            ), mock.patch.object(
                artifacts, "_marker_valid", return_value=True
            ), mock.patch.dict(
                os.environ, {"COMFYCOLAB_CUBEPART_SKIP_ENV_INSTALL": "1"}
            ):
                resolved = artifacts.ensure_cubepart_artifacts(
                    accept_research_license=True,
                    source_dir=source,
                    environment_dir=root / "env",
                    weights_root=root / "models",
                )

        self.assertEqual(resolved.python, Path(sys.executable))

    def test_environment_installs_required_runtime_versions_before_cube_part(self) -> None:
        artifacts = load_artifacts()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cube" / "cubepart"
            source.mkdir(parents=True)
            (source / "pyproject.toml").write_text("[project]\nname='cube_part'\n")
            environment = root / "env"
            base_python = root / "trellis-python"
            base_python.touch()
            calls: list[list[str]] = []

            def fake_check_call(command):
                calls.append(command)
                if "-m" in command and "venv" in command:
                    (environment / "bin").mkdir(parents=True)
                    (environment / "bin" / "python").touch()

            with mock.patch.object(
                artifacts, "_git_revision", return_value=artifacts.CUBEPART_SOURCE_REF
            ), mock.patch.object(
                artifacts.subprocess, "check_call", side_effect=fake_check_call
            ), mock.patch.dict(
                os.environ,
                {"COMFYCOLAB_CUBEPART_BASE_PYTHON": str(base_python)},
            ):
                python = artifacts._ensure_environment(source, environment)

            self.assertEqual(python, environment / "bin" / "python")
            self.assertEqual(
                calls[2],
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    *artifacts.CUBEPART_RUNTIME_REQUIREMENTS,
                ],
            )
            self.assertEqual(
                calls[3],
                [str(python), "-m", "pip", "install", "-e", str(source)],
            )
            marker = json.loads(
                (environment / ".comfycolab-cubepart-env.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                marker["runtime_requirements"],
                list(artifacts.CUBEPART_RUNTIME_REQUIREMENTS),
            )
            self.assertTrue(marker["environment_ref"].endswith("-v2"))


if __name__ == "__main__":
    unittest.main()
