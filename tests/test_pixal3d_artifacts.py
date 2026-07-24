from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import subprocess
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
    def test_snapshot_uses_xet_token_then_retries_public_repo_anonymously(self) -> None:
        artifacts = load_artifacts()
        calls: list[object] = []

        def snapshot_download(*, local_dir, token, **_kwargs):
            calls.append(token)
            if token:
                raise RuntimeError("stale token")
            root = Path(local_dir)
            root.mkdir(parents=True, exist_ok=True)
            (root / "config.json").write_text("{}", encoding="utf-8")
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
            sys.modules,
            {"huggingface_hub": fake_hub},
        ), mock.patch.dict(
            os.environ,
            {"HF_TOKEN": "test-token"},
            clear=False,
        ):
            artifacts._ensure_snapshot(
                repo_id="owner/model",
                revision="a" * 40,
                destination=Path(directory) / "snapshot",
                sentinel="config.json",
                progress=lambda _event: None,
            )
            self.assertEqual(os.environ["HF_XET_HIGH_PERFORMANCE"], "1")
            self.assertEqual(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"], "120")

        self.assertEqual(calls, ["test-token", False])

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

    def test_vggt_omega_snapshot_is_revision_pinned_and_checkpoint_only(self) -> None:
        artifacts = load_artifacts()
        calls = []

        def snapshot_download(**kwargs):
            calls.append(kwargs)
            root = Path(kwargs["local_dir"])
            root.mkdir(parents=True, exist_ok=True)
            (root / artifacts.VGGT_OMEGA_CHECKPOINT).write_bytes(
                b"small-test-checkpoint"
            )
            return str(root)

        fake_hub = types.SimpleNamespace(snapshot_download=snapshot_download)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            sys.modules,
            {"huggingface_hub": fake_hub},
        ):
            destination = Path(directory) / "omega"
            artifacts._ensure_snapshot(
                repo_id=artifacts.VGGT_OMEGA_MODEL_REPO,
                revision=artifacts.VGGT_OMEGA_MODEL_REF,
                destination=destination,
                sentinel=artifacts.VGGT_OMEGA_CHECKPOINT,
                progress=lambda _event: None,
                allow_patterns=[artifacts.VGGT_OMEGA_CHECKPOINT],
            )
            artifacts._ensure_snapshot(
                repo_id=artifacts.VGGT_OMEGA_MODEL_REPO,
                revision=artifacts.VGGT_OMEGA_MODEL_REF,
                destination=destination,
                sentinel=artifacts.VGGT_OMEGA_CHECKPOINT,
                progress=lambda _event: None,
                allow_patterns=[artifacts.VGGT_OMEGA_CHECKPOINT],
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["repo_id"], artifacts.VGGT_OMEGA_MODEL_REPO)
        self.assertEqual(calls[0]["revision"], artifacts.VGGT_OMEGA_MODEL_REF)
        self.assertEqual(
            calls[0]["allow_patterns"],
            [artifacts.VGGT_OMEGA_CHECKPOINT],
        )

    def test_vggt_omega_checkpoint_prefers_official_snapshot(self) -> None:
        artifacts = load_artifacts()
        payload = b"official-checkpoint"
        calls: list[tuple[str, str]] = []
        progress: list[dict] = []

        def ensure_snapshot(*, repo_id, revision, destination, **_kwargs):
            calls.append((repo_id, revision))
            destination.mkdir(parents=True, exist_ok=True)
            (destination / artifacts.VGGT_OMEGA_CHECKPOINT).write_bytes(payload)
            return destination

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            artifacts,
            "VGGT_OMEGA_CHECKPOINT_BYTES",
            len(payload),
        ), mock.patch.object(
            artifacts,
            "VGGT_OMEGA_CHECKPOINT_SHA256",
            hashlib.sha256(payload).hexdigest(),
        ), mock.patch.object(
            artifacts,
            "_ensure_snapshot",
            side_effect=ensure_snapshot,
        ):
            checkpoint, repo_id, revision, used_fallback = (
                artifacts._ensure_vggt_omega_checkpoint(
                    Path(directory),
                    progress=progress.append,
                )
            )
            checkpoint_payload = checkpoint.read_bytes()

        self.assertEqual(
            calls,
            [(artifacts.VGGT_OMEGA_MODEL_REPO, artifacts.VGGT_OMEGA_MODEL_REF)],
        )
        self.assertEqual(repo_id, artifacts.VGGT_OMEGA_MODEL_REPO)
        self.assertEqual(revision, artifacts.VGGT_OMEGA_MODEL_REF)
        self.assertFalse(used_fallback)
        self.assertEqual(checkpoint_payload, payload)
        self.assertFalse(any(event["stage"] == "snapshot_fallback" for event in progress))

    def test_vggt_omega_checkpoint_uses_pinned_public_mirror_after_gated_failure(
        self,
    ) -> None:
        artifacts = load_artifacts()
        payload = b"byte-identical-mirror-checkpoint"
        calls: list[tuple[str, str]] = []
        progress: list[dict] = []

        def ensure_snapshot(*, repo_id, revision, destination, **_kwargs):
            calls.append((repo_id, revision))
            if repo_id == artifacts.VGGT_OMEGA_MODEL_REPO:
                raise RuntimeError("official repository is gated")
            destination.mkdir(parents=True, exist_ok=True)
            (destination / artifacts.VGGT_OMEGA_CHECKPOINT).write_bytes(payload)
            return destination

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            artifacts,
            "VGGT_OMEGA_CHECKPOINT_BYTES",
            len(payload),
        ), mock.patch.object(
            artifacts,
            "VGGT_OMEGA_CHECKPOINT_SHA256",
            hashlib.sha256(payload).hexdigest(),
        ), mock.patch.object(
            artifacts,
            "_ensure_snapshot",
            side_effect=ensure_snapshot,
        ):
            checkpoint, repo_id, revision, used_fallback = (
                artifacts._ensure_vggt_omega_checkpoint(
                    Path(directory),
                    progress=progress.append,
                )
            )
            checkpoint_payload = checkpoint.read_bytes()

        self.assertEqual(
            calls,
            [
                (artifacts.VGGT_OMEGA_MODEL_REPO, artifacts.VGGT_OMEGA_MODEL_REF),
                (
                    artifacts.VGGT_OMEGA_FALLBACK_MODEL_REPO,
                    artifacts.VGGT_OMEGA_FALLBACK_MODEL_REF,
                ),
            ],
        )
        self.assertEqual(repo_id, artifacts.VGGT_OMEGA_FALLBACK_MODEL_REPO)
        self.assertEqual(revision, artifacts.VGGT_OMEGA_FALLBACK_MODEL_REF)
        self.assertTrue(used_fallback)
        self.assertEqual(checkpoint_payload, payload)
        fallback_event = next(
            event for event in progress if event["stage"] == "snapshot_fallback"
        )
        self.assertEqual(
            fallback_event["fallback_repo"],
            artifacts.VGGT_OMEGA_FALLBACK_MODEL_REPO,
        )

    def test_vggt_omega_public_mirror_bypasses_failed_xet_transport(self) -> None:
        artifacts = load_artifacts()
        payload = b"byte-identical-direct-checkpoint"
        progress: list[dict] = []

        def ensure_snapshot(**_kwargs):
            raise RuntimeError("Xet token endpoint returned 401")

        def ensure_direct(*, destination, filename, **kwargs):
            self.assertEqual(
                kwargs["url"],
                artifacts.VGGT_OMEGA_FALLBACK_CHECKPOINT_URL,
            )
            destination.mkdir(parents=True, exist_ok=True)
            (destination / filename).write_bytes(payload)
            return destination

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            artifacts,
            "VGGT_OMEGA_CHECKPOINT_BYTES",
            len(payload),
        ), mock.patch.object(
            artifacts,
            "VGGT_OMEGA_CHECKPOINT_SHA256",
            hashlib.sha256(payload).hexdigest(),
        ), mock.patch.object(
            artifacts,
            "_ensure_snapshot",
            side_effect=ensure_snapshot,
        ), mock.patch.object(
            artifacts,
            "_ensure_direct_hf_snapshot",
            side_effect=ensure_direct,
        ) as direct:
            checkpoint, repo_id, revision, used_fallback = (
                artifacts._ensure_vggt_omega_checkpoint(
                    Path(directory),
                    progress=progress.append,
                )
            )
            checkpoint_payload = checkpoint.read_bytes()

        self.assertEqual(checkpoint_payload, payload)
        self.assertEqual(repo_id, artifacts.VGGT_OMEGA_FALLBACK_MODEL_REPO)
        self.assertEqual(revision, artifacts.VGGT_OMEGA_FALLBACK_MODEL_REF)
        self.assertTrue(used_fallback)
        direct.assert_called_once()
        transport_event = next(
            event
            for event in progress
            if event["stage"] == "snapshot_transport_fallback"
        )
        self.assertEqual(
            transport_event["fallback_transport"],
            "immutable_direct_resolve",
        )

    def test_direct_hf_snapshot_writes_and_reuses_verified_manifest(self) -> None:
        artifacts = load_artifacts()
        payload = b"direct-download-test"

        class Response:
            status = 200
            headers: dict[str, str] = {}

            def __init__(self):
                self._stream = io.BytesIO(payload)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, size=-1):
                return self._stream.read(size)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            artifacts.urllib.request,
            "urlopen",
            return_value=Response(),
        ) as urlopen:
            destination = Path(directory) / "snapshot"
            kwargs = {
                "url": "https://example.invalid/pinned.pt",
                "repo_id": "owner/model",
                "revision": "a" * 40,
                "destination": destination,
                "filename": "pinned.pt",
                "expected_bytes": len(payload),
                "expected_sha256": hashlib.sha256(payload).hexdigest(),
                "progress": lambda _event: None,
            }
            artifacts._ensure_direct_hf_snapshot(**kwargs)
            artifacts._ensure_direct_hf_snapshot(**kwargs)

            marker = json.loads(
                (destination / ".comfycolab-artifact.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual((destination / "pinned.pt").read_bytes(), payload)
            self.assertEqual(marker["repo"], "owner/model")
            self.assertEqual(marker["revision"], "a" * 40)
            self.assertEqual(marker["transport"], "immutable-direct-resolve")
            urlopen.assert_called_once()

    def test_vggt_omega_public_mirror_must_match_official_checkpoint_digest(
        self,
    ) -> None:
        artifacts = load_artifacts()
        expected = b"official-checkpoint"
        mirrored = b"different-checkpoint"

        def ensure_snapshot(*, repo_id, destination, **_kwargs):
            if repo_id == artifacts.VGGT_OMEGA_MODEL_REPO:
                raise RuntimeError("official repository is gated")
            destination.mkdir(parents=True, exist_ok=True)
            (destination / artifacts.VGGT_OMEGA_CHECKPOINT).write_bytes(mirrored)
            return destination

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            artifacts,
            "VGGT_OMEGA_CHECKPOINT_BYTES",
            len(mirrored),
        ), mock.patch.object(
            artifacts,
            "VGGT_OMEGA_CHECKPOINT_SHA256",
            hashlib.sha256(expected).hexdigest(),
        ), mock.patch.object(
            artifacts,
            "_ensure_snapshot",
            side_effect=ensure_snapshot,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "public mirror.*SHA-256 mismatch",
            ):
                artifacts._ensure_vggt_omega_checkpoint(
                    Path(directory),
                    progress=lambda _event: None,
                )

    def test_vggt_omega_source_checkout_failure_is_a_fallback_safe_runtime_error(
        self,
    ) -> None:
        artifacts = load_artifacts()

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            artifacts,
            "_git",
            side_effect=subprocess.CalledProcessError(128, ["git", "clone"]),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Unable to provision pinned source.*vggt-omega",
            ):
                artifacts._ensure_git_source(
                    repository=artifacts.VGGT_OMEGA_SOURCE_REPO,
                    revision=artifacts.VGGT_OMEGA_SOURCE_REF,
                    destination=Path(directory) / "vggt-omega",
                    sentinel="vggt_omega/models/vggt_omega.py",
                )

    def test_vggt_omega_notice_pins_source_model_and_noncommercial_terms(
        self,
    ) -> None:
        artifacts = load_artifacts()
        notice = (
            Path(__file__).resolve().parents[1] / "THIRD_PARTY_NOTICES.md"
        ).read_text(encoding="utf-8")

        self.assertIn(artifacts.VGGT_OMEGA_SOURCE_REF, notice)
        self.assertIn(artifacts.VGGT_OMEGA_MODEL_REF, notice)
        self.assertIn(artifacts.VGGT_OMEGA_FALLBACK_MODEL_REF, notice)
        self.assertIn(artifacts.VGGT_OMEGA_CHECKPOINT_SHA256, notice)
        self.assertIn("FAIR Noncommercial Research License", notice)
        self.assertIn("CC BY-NC 4.0", notice)
        self.assertIn("not a grant of rights", notice)
        self.assertIn("does not redistribute VGGT-Ω source or weights", notice)


if __name__ == "__main__":
    unittest.main()
