from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from comfycolab.artifacts import discover_glb_artifacts, sync_glb_artifacts


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class ArtifactSyncTests(unittest.TestCase):
    def test_discovers_only_safe_glb_history_outputs(self):
        history = {
            "outputs": [
                {
                    "filename": "pixal.glb",
                    "subfolder": "3d/Pixal3DMV_Advanced",
                    "type": "output",
                },
                {
                    "filename": "../escape.glb",
                    "subfolder": "",
                    "type": "output",
                },
                {
                    "filename": "preview.png",
                    "subfolder": "3d",
                    "type": "output",
                },
            ]
        }

        artifacts = discover_glb_artifacts(history)

        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].filename, "pixal.glb")
        self.assertEqual(artifacts[0].subfolder, "3d/Pixal3DMV_Advanced")

    def test_sync_downloads_each_remote_stage_once(self):
        history = {
            "outputs": [
                {
                    "filename": "stage_pixal3d_00001_.glb",
                    "subfolder": "3d/Pixal3DMV_Advanced",
                    "type": "output",
                },
                {
                    "filename": "stage_meshflow_00001_.glb",
                    "subfolder": "3d/Pixal3DMV_Advanced",
                    "type": "output",
                },
            ]
        }
        calls: list[str] = []

        def opener(url, timeout):
            del timeout
            calls.append(url)
            if url.endswith("/history"):
                return Response(json.dumps(history).encode())
            return Response(b"glb-stage")

        with tempfile.TemporaryDirectory() as directory:
            saved = sync_glb_artifacts(
                "https://example.invalid",
                directory,
                opener=opener,
            )
            repeated = sync_glb_artifacts(
                "https://example.invalid",
                directory,
                opener=opener,
            )

            self.assertEqual(len(saved), 2)
            self.assertEqual(repeated, [])
            self.assertTrue(all(path.read_bytes() == b"glb-stage" for path in saved))
            state = json.loads(
                (Path(directory) / ".comfycolab-artifacts.json").read_text()
            )
            self.assertEqual(len(state["downloaded"]), 2)
            self.assertEqual(sum("/view?" in call for call in calls), 2)
