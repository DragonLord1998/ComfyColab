from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from comfycolab.state import (
    RuntimeStateError,
    normalize_runtime_state,
    read_runtime_state,
    verify_lock_digest,
    write_runtime_state,
)


class RuntimeStateTests(unittest.TestCase):
    def test_legacy_ready_payload_is_migrated_without_dropping_aliases(self) -> None:
        state = normalize_runtime_state(
            {
                "comfyUrl": "https://example.test",
                "repositoryCommit": "a" * 40,
            },
            session="demo",
            gpu="G4",
        )
        self.assertEqual(state["schema"], 1)
        self.assertEqual(state["status"], "ready")
        self.assertEqual(state["core"], {})
        self.assertEqual(state["packs"], {})
        self.assertEqual(state["repositoryCommit"], "a" * 40)
        self.assertEqual(state["session"], "demo")

    def test_round_trip_is_atomic_and_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            write_runtime_state(path, {"status": "ready", "packs": {"image": {}}})
            state = read_runtime_state(path)
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state["schema"], 1)
            self.assertEqual(state["packs"], {"image": {}})
            self.assertEqual(json.loads(path.read_text())["core"], {})

    def test_lock_digest_must_match(self) -> None:
        verify_lock_digest({"lockSha256": "a" * 64}, "a" * 64)
        with self.assertRaisesRegex(RuntimeStateError, "mismatch"):
            verify_lock_digest({"lockSha256": "b" * 64}, "a" * 64)

    def test_invalid_nested_state_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeStateError, "packs"):
            normalize_runtime_state({"schema": 1, "packs": []})


if __name__ == "__main__":
    unittest.main()
