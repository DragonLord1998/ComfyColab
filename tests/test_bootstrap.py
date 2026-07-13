from __future__ import annotations

import base64
import hashlib
import io
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
                TRELLIS_DIR=root / "ComfyUI-TRELLIS2",
                GEOMETRY_DIR=root / "ComfyUI-GeometryPack",
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
            self.assertEqual(payload["trellisCommit"], "abc123")
            self.assertEqual(payload["geometryCommit"], "abc123")
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
                TRELLIS_DIR=root / "ComfyUI-TRELLIS2",
                GEOMETRY_DIR=root / "ComfyUI-GeometryPack",
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
        self.assertIn("9b878516f2dc2fd873f4f6cceadba403dd12d83e", source)
        self.assertIn("c67199de05705642258e727fa118f412877b4ebf", source)

    def test_trellis_dependencies_use_isolated_upstream_installer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comfy_dir = root / "ComfyUI"
            gguf_dir = comfy_dir / "custom_nodes" / "ComfyUI-GGUF"
            trellis_dir = comfy_dir / "custom_nodes" / "ComfyUI-TRELLIS2"
            comfy_dir.mkdir(parents=True)
            gguf_dir.mkdir(parents=True)
            trellis_dir.mkdir(parents=True)
            (gguf_dir / "requirements.txt").write_text("gguf\n", encoding="utf-8")
            (trellis_dir / "requirements.txt").write_text(
                "comfy-env==0.3.89\n", encoding="utf-8"
            )
            commands: list[tuple[list[str], Path | None]] = []
            with mock.patch.multiple(
                remote_bootstrap,
                COMFY_DIR=comfy_dir,
                GGUF_DIR=gguf_dir,
                TRELLIS_DIR=trellis_dir,
            ), mock.patch.object(
                remote_bootstrap,
                "run",
                side_effect=lambda command, cwd=None: commands.append((command, cwd)),
            ):
                remote_bootstrap.install_dependencies()

            self.assertEqual(commands[0][1], comfy_dir)
            self.assertEqual(commands[1][0][-1], str(gguf_dir / "requirements.txt"))
            self.assertEqual(
                commands[2][0][-3:],
                ["-r", str(trellis_dir / "requirements.txt"), "--upgrade"],
            )
            self.assertEqual(
                commands[3],
                ([remote_bootstrap.sys.executable, "install.py"], trellis_dir),
            )

    def test_trellis_cache_profile_is_versioned_and_checksum_pinned(self) -> None:
        cache = remote_bootstrap.TRELLIS_CACHE
        self.assertEqual(
            cache["profile"],
            "g4-linux64-py31213-torch2110-cu128-sm120-glibc235-v1",
        )
        self.assertTrue(str(cache["release_base"]).endswith("/trellis2-cache-v1"))
        self.assertRegex(str(cache["archive_sha256"]), r"^[0-9a-f]{64}$")
        self.assertEqual(len(cache["parts"]), 3)
        for part in cache["parts"]:
            self.assertRegex(part["name"], r"\.part-\d{3}$")
            self.assertRegex(part["sha256"], r"^[0-9a-f]{64}$")

    def test_trellis_cache_part_download_is_checksum_verified(self) -> None:
        payload = b"verified-cache-part"
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "part-000"
            with mock.patch.object(
                remote_bootstrap.urllib.request,
                "urlopen",
                return_value=io.BytesIO(payload),
            ):
                remote_bootstrap.download_cache_part(
                    {"url": "https://example.test/part", "sha256": expected},
                    destination,
                )
            self.assertEqual(destination.read_bytes(), payload)

    def test_trellis_cache_part_rejects_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "part-000"
            with mock.patch.object(
                remote_bootstrap.urllib.request,
                "urlopen",
                return_value=io.BytesIO(b"corrupt"),
            ):
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    remote_bootstrap.download_cache_part(
                        {"url": "https://example.test/part", "sha256": "0" * 64},
                        destination,
                    )
            self.assertFalse(destination.exists())

    def test_incompatible_runtime_skips_trellis_cache_download(self) -> None:
        with mock.patch.object(
            remote_bootstrap,
            "trellis_cache_compatible",
            return_value=False,
        ), mock.patch.object(remote_bootstrap, "download_cache_part") as download:
            self.assertFalse(remote_bootstrap.restore_trellis_cache())
        download.assert_not_called()

    def test_trellis_workspace_requires_both_environments_and_pinned_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / ".pixi" / "envs" / "trellis2-nodes" / "bin").mkdir(
                parents=True
            )
            (workspace / ".pixi" / "envs" / "trellis2-nodes" / "bin" / "python").touch()
            (workspace / "pixi.toml").write_bytes(b"toml")
            (workspace / "pixi.lock").write_bytes(b"lock")
            (workspace / "install.hash").write_text("pinned\n", encoding="utf-8")
            cache = {
                **remote_bootstrap.TRELLIS_CACHE,
                "pixi_toml_sha256": hashlib.sha256(b"toml").hexdigest(),
                "pixi_lock_sha256": hashlib.sha256(b"lock").hexdigest(),
                "install_hash": "pinned",
            }
            with mock.patch.object(remote_bootstrap, "TRELLIS_CACHE", cache):
                self.assertFalse(
                    remote_bootstrap.trellis_workspace_metadata_valid(workspace)
                )
                geometry_python = (
                    workspace
                    / ".pixi"
                    / "envs"
                    / "geometrypack-nodes"
                    / "bin"
                    / "python"
                )
                geometry_python.parent.mkdir(parents=True)
                geometry_python.touch()
                self.assertTrue(
                    remote_bootstrap.trellis_workspace_metadata_valid(workspace)
                )
                (workspace / "install.hash").write_text("stale\n", encoding="utf-8")
                self.assertFalse(
                    remote_bootstrap.trellis_workspace_metadata_valid(workspace)
                )

    def test_trellis_archive_rejects_path_traversal(self) -> None:
        completed = remote_bootstrap.subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "-rw-r--r-- root/root 1 2026-07-13 00:00 .ce/pixi.toml\n"
                "-rw-r--r-- root/root 1 2026-07-13 00:00 ../escape\n"
            ),
            stderr="",
        )
        with mock.patch.object(
            remote_bootstrap.subprocess,
            "run",
            return_value=completed,
        ):
            with self.assertRaisesRegex(RuntimeError, "Unsafe TRELLIS cache"):
                remote_bootstrap.validate_trellis_archive(Path("cache.tar.zst"))

    def test_trellis_archive_rejects_escaping_hard_link(self) -> None:
        completed = remote_bootstrap.subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "hrw-r--r-- root/root 0 2026-07-13 00:00 "
                ".ce/link link to ../outside\n"
            ),
            stderr="",
        )
        with mock.patch.object(remote_bootstrap.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "hard-link target"):
                remote_bootstrap.validate_trellis_archive(Path("cache.tar.zst"))

    def test_trellis_archive_rejects_special_file(self) -> None:
        completed = remote_bootstrap.subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="prw-r--r-- root/root 0 2026-07-13 00:00 .ce/pipe\n",
            stderr="",
        )
        with mock.patch.object(remote_bootstrap.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "Unsupported TRELLIS cache"):
                remote_bootstrap.validate_trellis_archive(Path("cache.tar.zst"))

    def test_cache_failure_forces_normal_installer_revalidation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comfy_dir = root / "ComfyUI"
            trellis_dir = comfy_dir / "custom_nodes" / "ComfyUI-TRELLIS2"
            workspace = root / ".ce"
            comfy_dir.mkdir(parents=True)
            trellis_dir.mkdir(parents=True)
            (trellis_dir / "requirements.txt").write_text(
                "comfy-env==0.3.89\n", encoding="utf-8"
            )
            workspace.mkdir()
            install_hash = workspace / "install.hash"
            install_hash.write_text("stale\n", encoding="utf-8")
            with mock.patch.multiple(
                remote_bootstrap,
                COMFY_DIR=comfy_dir,
                GGUF_DIR=root / "missing-gguf",
                TRELLIS_DIR=trellis_dir,
            ), mock.patch.object(
                remote_bootstrap.Path,
                "home",
                return_value=root,
            ), mock.patch.object(
                remote_bootstrap,
                "restore_trellis_cache",
                side_effect=RuntimeError("bad cache"),
            ), mock.patch.object(remote_bootstrap, "run") as run:
                remote_bootstrap.install_dependencies()
            self.assertFalse(install_hash.exists())
            self.assertEqual(
                run.call_args_list[-1],
                mock.call([remote_bootstrap.sys.executable, "install.py"], cwd=trellis_dir),
            )

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
        expression, timeout = eval_js.call_args.args
        self.assertIn("google.colab.kernel.accessAllowed", expression)
        self.assertIn("google.colab.kernel.proxyPort(8188)", expression)
        self.assertIn('new URL("/", proxy).toString()', expression)
        self.assertEqual(timeout, 15)
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

    def test_colab_proxy_retries_none_frontend_reply(self) -> None:
        url = "https://abc-8188.colab.googleusercontent.com/"
        with mock.patch.object(
            remote_bootstrap,
            "CONFIG",
            {"colab_proxy": True},
        ), mock.patch.object(
            remote_bootstrap,
            "eval_colab_js",
            side_effect=[None, url],
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
