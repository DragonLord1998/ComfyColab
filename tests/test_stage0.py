from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from comfycolab import stage0_runtime
from comfycolab.config import ConfigError, CoreStage0ConfigV1
from comfycolab.stage0 import CONFIG_MARKER, render_stage0


CORE_COMMIT = "a" * 40
STAGE1_SHA = "b" * 64
LOCK = b'{"comfyui":{"commit":"' + (b"c" * 40) + b'"},"packs":[],"schema":1}'


class Stage0ConfigTests(unittest.TestCase):
    def config(self) -> CoreStage0ConfigV1:
        return CoreStage0ConfigV1.create(
            core_repository="https://github.com/example/ComfyColab.git",
            core_commit=CORE_COMMIT,
            stage1_entrypoint="src/comfycolab/runtime.py",
            stage1_sha256=STAGE1_SHA,
            lock_bytes=LOCK,
            port=8188,
            refresh=True,
            colab_proxy=True,
        )

    def test_round_trip_and_lock_digest(self) -> None:
        config = self.config()
        restored = CoreStage0ConfigV1.from_dict(config.to_dict())
        self.assertEqual(restored, config)
        self.assertEqual(restored.lock_bytes(), LOCK)
        self.assertEqual(restored.lock_sha256, hashlib.sha256(LOCK).hexdigest())

    def test_rejects_mutable_core_ref_and_unsafe_entrypoint(self) -> None:
        with self.assertRaisesRegex(ConfigError, "40-character"):
            CoreStage0ConfigV1.create(
                core_repository="https://github.com/example/ComfyColab.git",
                core_commit="main",
                stage1_entrypoint="src/comfycolab/runtime.py",
                stage1_sha256=STAGE1_SHA,
                lock_bytes=LOCK,
            )
        with self.assertRaisesRegex(ConfigError, "safe relative"):
            CoreStage0ConfigV1.create(
                core_repository="https://github.com/example/ComfyColab.git",
                core_commit=CORE_COMMIT,
                stage1_entrypoint="../runtime.py",
                stage1_sha256=STAGE1_SHA,
                lock_bytes=LOCK,
            )

    def test_rejects_lock_digest_tampering(self) -> None:
        payload = self.config().to_dict()
        payload["lock_b64"] = base64.b64encode(b'{"schema":1,"packs":[1]}').decode()
        with self.assertRaisesRegex(ConfigError, "digest"):
            CoreStage0ConfigV1.from_dict(payload)

    def test_renderer_embeds_only_encoded_canonical_config(self) -> None:
        config = self.config()
        source = render_stage0(config)
        self.assertNotIn(CONFIG_MARKER, source)
        self.assertNotIn(config.core_repository, source)
        match = re.search(r'CONFIG_B64 = "([A-Za-z0-9+/=]+)"', source)
        self.assertIsNotNone(match)
        assert match is not None
        decoded = base64.b64decode(match.group(1))
        self.assertEqual(decoded, config.canonical_bytes())
        self.assertEqual(json.loads(decoded), config.to_dict())

    def test_stage1_failure_includes_combined_child_output_tail(self) -> None:
        process = mock.Mock()
        process.stdout = io.StringIO(
            "[comfycolab] installing\n"
            "ipython 7.34.0 requires jedi, which is not installed.\n"
        )
        process.wait.return_value = 1

        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory) / "state"
            log_path = state_dir / "stage1.log"
            with (
                mock.patch.object(stage0_runtime, "STATE_DIR", state_dir),
                mock.patch.object(stage0_runtime, "STAGE1_LOG_FILE", log_path),
                mock.patch.object(
                    stage0_runtime.subprocess,
                    "Popen",
                    return_value=process,
                ),
                redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(
                    RuntimeError,
                    "(?s)stage-1 exited with status 1.*requires jedi",
                ),
            ):
                stage0_runtime.run_stage1(
                    ["python", "-m", "comfycolab.runtime"],
                    cwd=Path(directory),
                    env={},
                )

            self.assertIn("requires jedi", log_path.read_text(encoding="utf-8"))
            process.wait.assert_called_once_with()

    def test_stage1_success_streams_and_persists_output(self) -> None:
        process = mock.Mock()
        process.stdout = io.StringIO("COMFYCOLAB_READY={}\n")
        process.wait.return_value = 0

        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory) / "state"
            log_path = state_dir / "stage1.log"
            output = io.StringIO()
            with (
                mock.patch.object(stage0_runtime, "STATE_DIR", state_dir),
                mock.patch.object(stage0_runtime, "STAGE1_LOG_FILE", log_path),
                mock.patch.object(
                    stage0_runtime.subprocess,
                    "Popen",
                    return_value=process,
                ),
                redirect_stdout(output),
            ):
                stage0_runtime.run_stage1(
                    ["python", "-m", "comfycolab.runtime"],
                    cwd=Path(directory),
                    env={},
                )

            self.assertIn("COMFYCOLAB_READY={}", output.getvalue())
            self.assertEqual(
                log_path.read_text(encoding="utf-8"),
                "COMFYCOLAB_READY={}\n",
            )

    def test_stage1_stream_failure_terminates_child(self) -> None:
        class BrokenStream:
            def __iter__(self):
                return self

            def __next__(self):
                raise OSError("stream failed")

        process = mock.Mock()
        process.stdout = BrokenStream()
        process.poll.return_value = None
        process.wait.return_value = 0

        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory) / "state"
            with (
                mock.patch.object(stage0_runtime, "STATE_DIR", state_dir),
                mock.patch.object(
                    stage0_runtime,
                    "STAGE1_LOG_FILE",
                    state_dir / "stage1.log",
                ),
                mock.patch.object(
                    stage0_runtime.subprocess,
                    "Popen",
                    return_value=process,
                ),
                redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(
                    RuntimeError,
                    "stage-1 launch/output failed: stream failed",
                ),
            ):
                stage0_runtime.run_stage1(
                    ["python", "-m", "comfycolab.runtime"],
                    cwd=Path(directory),
                    env={},
                )

            process.terminate.assert_called_once_with()
            process.wait.assert_called_once_with(timeout=5)


if __name__ == "__main__":
    unittest.main()
