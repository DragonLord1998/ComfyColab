from __future__ import annotations

import base64
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from comfycolab.bootstrap import CONFIG_MARKER, render_bootstrap
from comfycolab import remote_bootstrap


class BootstrapRenderingTests(unittest.TestCase):
    def test_failed_tunnel_start_cleans_both_managed_processes(self) -> None:
        class Process:
            def __init__(self, pid: int):
                self.pid = pid

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_dir = root / "state"
            stopped: list[int] = []
            with mock.patch.multiple(
                remote_bootstrap,
                CONFIG={
                    "port": 8188,
                    "refresh": True,
                    "repository_url": "https://example.test/repo.git",
                    "repository_ref": "main",
                },
                STATE_DIR=state_dir,
                STATE_FILE=state_dir / "runtime.json",
                COMFY_LOG=state_dir / "comfy.log",
                TUNNEL_LOG=state_dir / "tunnel.log",
                COMFY_DIR=root / "ComfyUI",
                REPO_DIR=root / "ComfyColab",
                GGUF_DIR=root / "ComfyUI-GGUF",
                NODE_TARGET=root / "node",
            ), mock.patch.object(remote_bootstrap, "load_state", return_value={}), mock.patch.object(
                remote_bootstrap, "http_ready", return_value=False
            ), mock.patch.object(remote_bootstrap, "clone_or_update"), mock.patch.object(
                remote_bootstrap, "install_node_pack"
            ), mock.patch.object(remote_bootstrap, "install_dependencies"), mock.patch.object(
                remote_bootstrap, "cloudflared_path", return_value=Path("/tmp/cloudflared")
            ), mock.patch.object(remote_bootstrap, "wait_for_comfy"), mock.patch.object(
                remote_bootstrap, "wait_for_tunnel", side_effect=RuntimeError("no tunnel")
            ), mock.patch.object(
                remote_bootstrap.subprocess,
                "Popen",
                side_effect=[Process(101), Process(202)],
            ), mock.patch.object(
                remote_bootstrap,
                "stop_managed_process",
            ), mock.patch.object(
                remote_bootstrap,
                "stop_started_process",
                side_effect=lambda process: stopped.append(process.pid),
            ):
                with self.assertRaisesRegex(RuntimeError, "no tunnel"):
                    remote_bootstrap.main()

            self.assertEqual(stopped, [202, 101])
            self.assertFalse((state_dir / "runtime.json").exists())

    def test_template_module_is_safe_to_import_locally(self) -> None:
        self.assertEqual(
            remote_bootstrap.CONFIG,
            {
                "repository_url": "https://github.com/DragonLord1998/ComfyColab.git",
                "repository_ref": "main",
                "port": 8188,
                "refresh": False,
            },
        )

    def test_configuration_is_embedded_without_raw_interpolation(self) -> None:
        source = render_bootstrap(
            repository_url="https://example.com/org/repo.git",
            repository_ref="release-test",
            port=9000,
            refresh=True,
        )
        self.assertNotIn(CONFIG_MARKER, source)
        match = re.search(r'CONFIG_B64 = "([A-Za-z0-9+/=]+)"', source)
        self.assertIsNotNone(match)
        config = json.loads(base64.b64decode(match.group(1)).decode("utf-8"))
        self.assertEqual(
            config,
            {
                "repository_url": "https://example.com/org/repo.git",
                "repository_ref": "release-test",
                "port": 9000,
                "refresh": True,
            },
        )
        self.assertIn("8b099de36acd81acd1afa3b5442951dc847e0a52", source)
        self.assertIn("6ea2651e7df66d7585f6ffee804b20e92fb38b8a", source)


if __name__ == "__main__":
    unittest.main()
