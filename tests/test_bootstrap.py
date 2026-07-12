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
    def test_working_colab_proxy_survives_cloudflare_failure(self) -> None:
        class Process:
            def __init__(self, pid: int):
                self.pid = pid

        proxy_url = "https://abc-8188.colab.googleusercontent.com/"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_dir = root / "state"
            stopped: list[int] = []
            with mock.patch.multiple(
                remote_bootstrap,
                CONFIG={
                    "port": 8188,
                    "refresh": True,
                    "colab_proxy": True,
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
                remote_bootstrap, "request_colab_proxy_url", return_value=proxy_url
            ), mock.patch.object(
                remote_bootstrap, "git_commit", return_value="abc123"
            ), mock.patch.object(
                remote_bootstrap.subprocess,
                "Popen",
                side_effect=[Process(101), Process(202)],
            ), mock.patch.object(
                remote_bootstrap, "stop_managed_process"
            ), mock.patch.object(
                remote_bootstrap,
                "stop_started_process",
                side_effect=lambda process: stopped.append(process.pid),
            ):
                remote_bootstrap.main()

            payload = json.loads((state_dir / "runtime.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["comfyUrl"], proxy_url)
            self.assertIsNone(payload["cloudflareUrl"])
            self.assertEqual(stopped, [202])

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
                    "colab_proxy": False,
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
                "colab_proxy": False,
            },
        )

    def test_configuration_is_embedded_without_raw_interpolation(self) -> None:
        source = render_bootstrap(
            repository_url="https://example.com/org/repo.git",
            repository_ref="release-test",
            port=9000,
            refresh=True,
            colab_proxy=True,
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
                "colab_proxy": True,
            },
        )
        self.assertIn("8b099de36acd81acd1afa3b5442951dc847e0a52", source)
        self.assertIn("6ea2651e7df66d7585f6ffee804b20e92fb38b8a", source)

    def test_colab_proxy_accepts_googleusercontent_url(self) -> None:
        url = "https://abc-8188.colab.googleusercontent.com/"
        with mock.patch.object(
            remote_bootstrap,
            "CONFIG",
            {"colab_proxy": True},
        ), mock.patch.object(
            remote_bootstrap,
            "eval_colab_js",
            return_value=url,
        ) as eval_js, mock.patch.object(
            remote_bootstrap,
            "probe_colab_proxy_url",
            return_value=True,
        ) as probe:
            self.assertEqual(remote_bootstrap.request_colab_proxy_url(8188), url)
        eval_js.assert_called_once_with("google.colab.kernel.proxyPort(8188)", 15)
        probe.assert_called_once_with(url)

    def test_colab_proxy_rejects_untrusted_url_and_falls_back(self) -> None:
        with mock.patch.object(
            remote_bootstrap,
            "CONFIG",
            {"colab_proxy": True},
        ), mock.patch.object(
            remote_bootstrap,
            "eval_colab_js",
            return_value="https://example.test/steal",
        ):
            self.assertIsNone(remote_bootstrap.request_colab_proxy_url(8188, attempts=1))

    def test_colab_proxy_falls_back_when_comfy_websocket_probe_fails(self) -> None:
        url = "https://abc-8188.colab.googleusercontent.com/"
        with mock.patch.object(
            remote_bootstrap,
            "CONFIG",
            {"colab_proxy": True},
        ), mock.patch.object(
            remote_bootstrap,
            "eval_colab_js",
            return_value=url,
        ), mock.patch.object(
            remote_bootstrap,
            "probe_colab_proxy_url",
            return_value=False,
        ):
            self.assertIsNone(remote_bootstrap.request_colab_proxy_url(8188, attempts=1))

    def test_colab_proxy_probe_checks_http_and_websocket(self) -> None:
        with mock.patch.object(remote_bootstrap, "eval_colab_js", return_value=True) as eval_js:
            self.assertTrue(
                remote_bootstrap.probe_colab_proxy_url(
                    "https://abc-8188.colab.googleusercontent.com/"
                )
            )
        expression, timeout = eval_js.call_args.args
        self.assertIn('new URL("system_stats", baseUrl)', expression)
        self.assertIn("new WebSocket(socketUrl)", expression)
        self.assertEqual(timeout, 15)

    def test_colab_proxy_retries_while_browser_attaches(self) -> None:
        url = "https://abc-8188.colab.googleusercontent.com/"
        with mock.patch.object(
            remote_bootstrap,
            "CONFIG",
            {"colab_proxy": True},
        ), mock.patch.object(
            remote_bootstrap,
            "eval_colab_js",
            side_effect=[TimeoutError("frontend loading"), url],
        ) as eval_js, mock.patch.object(
            remote_bootstrap,
            "probe_colab_proxy_url",
            return_value=True,
        ), mock.patch.object(remote_bootstrap.time, "sleep") as sleep:
            self.assertEqual(remote_bootstrap.request_colab_proxy_url(8188), url)
        self.assertEqual(eval_js.call_count, 2)
        sleep.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
