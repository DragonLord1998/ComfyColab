from __future__ import annotations

import base64
import hashlib
import json
import re
import unittest

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


if __name__ == "__main__":
    unittest.main()
