from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
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


if __name__ == "__main__":
    unittest.main()
