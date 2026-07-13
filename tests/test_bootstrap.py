from __future__ import annotations

import base64
import errno
import hashlib
import io
import json
import re
import socket
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from comfycolab.bootstrap import CONFIG_MARKER, render_bootstrap
from comfycolab import remote_bootstrap


class BootstrapRenderingTests(unittest.TestCase):
    def test_patch_specs_pin_expected_upstream_revisions(self) -> None:
        root = Path(__file__).resolve().parents[1]
        trellis = json.loads(
            (root / "patches" / "trellis2-no-1536-downgrade.json").read_text(
                encoding="utf-8"
            )
        )
        ultrashape = json.loads(
            (root / "patches" / "ultrashape-inference-only-imports.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(trellis["patch_id"], remote_bootstrap.TRELLIS_PATCH_ID)
        self.assertEqual(trellis["revision"], remote_bootstrap.TRELLIS_REF)
        birefnet = next(
            item for item in trellis["files"] if item["path"] == "nodes/rembg/BiRefNet.py"
        )
        birefnet_patch = "\n".join(
            line
            for replacement in birefnet["replacements"]
            for line in replacement["after_lines"]
        )
        self.assertIn(remote_bootstrap.BIREFNET_MODEL_REF, birefnet_patch)
        self.assertIn("code_revision=BIREFNET_MODEL_REVISION", birefnet_patch)
        inference = next(
            item for item in trellis["files"] if item["path"] == "nodes/nodes_inference.py"
        )
        inference_patch = "\n".join(
            line
            for replacement in inference["replacements"]
            for line in replacement["after_lines"]
        )
        self.assertIn("ComfyColab shape metrics", inference_patch)
        self.assertIn("shape_slat_data['_resolution']", inference_patch)
        self.assertIn("shape_slat_data['feats'].shape[0]", inference_patch)
        self.assertEqual(ultrashape["patch_id"], remote_bootstrap.ULTRASHAPE_PATCH_ID)
        self.assertEqual(ultrashape["revision"], remote_bootstrap.ULTRASHAPE_REF)
        surface = next(
            item for item in ultrashape["files"]
            if item["path"] == "ultrashape/surface_loaders.py"
        )
        rendered = "\n".join(
            line
            for replacement in surface["replacements"]
            for line in replacement["after_lines"]
        )
        self.assertIn("trimesh.sample.sample_surface(mesh, num, seed=rng)", rendered)
        self.assertIn("normalize_scale=normalize_scale, rng=rng", rendered)
        self.assertNotIn("np.random.rand", rendered)

    def test_patched_isolated_nodes_invalidate_prebuilt_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            expected = []
            for name in ("trellis2-nodes", "geometrypack-nodes"):
                metadata = workspace / ".pixi" / "envs" / name / ".metadata_cache.pkl"
                metadata.parent.mkdir(parents=True)
                metadata.write_bytes(b"stale")
                expected.append(metadata)
            unrelated = workspace / ".pixi" / "envs" / "other" / ".metadata_cache.pkl"
            unrelated.parent.mkdir(parents=True)
            unrelated.write_bytes(b"keep")

            removed = remote_bootstrap.invalidate_comfyenv_metadata_cache(workspace)

            self.assertEqual(removed, expected)
            self.assertTrue(unrelated.is_file())
            self.assertTrue(all(not path.exists() for path in expected))

    def test_sm120_ultrashape_validation_executes_a_cubvh_kernel(self) -> None:
        completed = mock.Mock(returncode=0)
        with mock.patch.object(remote_bootstrap.subprocess, "run", return_value=completed) as run:
            remote_bootstrap.validate_ultrashape_imports(
                Path("/cached/python"), require_sm120=True
            )
        command = run.call_args.args[0]
        source = command[2]
        self.assertIn("cubvh.cuBVH", source)
        self.assertIn("unsigned_distance", source)
        self.assertIn("torch.cuda.synchronize", source)
        self.assertIn("get_device_capability() == (12, 0)", source)

    def test_malformed_combined_manifest_falls_back_to_trellis_cache(self) -> None:
        with mock.patch.object(
            remote_bootstrap,
            "combined_cache_specification",
            side_effect=RuntimeError("malformed manifest"),
        ), mock.patch.object(
            remote_bootstrap, "restore_trellis_cache", return_value=True
        ) as restore:
            self.assertEqual(
                remote_bootstrap.restore_3d_environment_cache(),
                remote_bootstrap.TRELLIS_CACHE["profile"],
            )
        restore.assert_called_once_with()

    def test_ready_combined_manifest_rejects_cubvh_or_comfy_env_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            manifest = {
                "schema": 1,
                "status": "ready",
                "profile": "test",
                "sources": {
                    "comfy": remote_bootstrap.COMFY_REF,
                    "trellis": remote_bootstrap.TRELLIS_REF,
                    "geometry": remote_bootstrap.GEOMETRY_REF,
                    "ultrashape": remote_bootstrap.ULTRASHAPE_REF,
                    "cubvh": "wrong",
                    "birefnet": remote_bootstrap.BIREFNET_MODEL_REF,
                    "comfyEnv": remote_bootstrap.COMFY_ENV_VERSION,
                },
                "patches": {
                    "trellis": remote_bootstrap.TRELLIS_PATCH_ID,
                    "ultrashape": remote_bootstrap.ULTRASHAPE_PATCH_ID,
                },
            }
            (cache_dir / remote_bootstrap.COMBINED_CACHE_MANIFEST).write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with mock.patch.object(remote_bootstrap, "REPO_DIR", root):
                with self.assertRaisesRegex(RuntimeError, "does not match pinned sources"):
                    remote_bootstrap.combined_cache_specification()
            manifest["sources"]["cubvh"] = remote_bootstrap.ULTRASHAPE_CUBVH_REF
            manifest["sources"]["comfyEnv"] = "0.0.0"
            (cache_dir / remote_bootstrap.COMBINED_CACHE_MANIFEST).write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with mock.patch.object(remote_bootstrap, "REPO_DIR", root):
                with self.assertRaisesRegex(RuntimeError, "does not match pinned sources"):
                    remote_bootstrap.combined_cache_specification()

    def test_ultrashape_overlay_reruns_all_trellis_and_cuda_probes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            python = home / ".ce" / ".pixi" / "envs" / "trellis2-nodes" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("", encoding="utf-8")
            with mock.patch.object(remote_bootstrap.Path, "home", return_value=home), mock.patch.object(
                remote_bootstrap, "run"
            ), mock.patch.object(remote_bootstrap, "validate_trellis_cache") as validate:
                remote_bootstrap.install_ultrashape_overlay()
        validate.assert_called_once_with(home / ".ce", validate_ultrashape=True)

    def test_content_addressed_patch_is_strict_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repository"
            repository.mkdir()
            target = repository / "target.py"
            target.write_text("original\n", encoding="utf-8")
            before = hashlib.sha256(b"original\n").hexdigest()
            after = hashlib.sha256(b"patched\n").hexdigest()
            specification = Path(directory) / "patch.json"
            specification.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "patch_id": "test-v1",
                        "revision": "abc123",
                        "files": [
                            {
                                "path": "target.py",
                                "before_sha256": before,
                                "after_sha256": after,
                                "replacements": [
                                    {
                                        "before_lines": ["original"],
                                        "after_lines": ["patched"],
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(remote_bootstrap, "git_commit", return_value="abc123"):
                self.assertEqual(
                    remote_bootstrap.apply_pinned_patch(repository, specification),
                    "test-v1",
                )
                self.assertEqual(target.read_text(encoding="utf-8"), "patched\n")
                self.assertEqual(
                    remote_bootstrap.apply_pinned_patch(repository, specification),
                    "test-v1",
                )
            target.write_text("unexpected\n", encoding="utf-8")
            with mock.patch.object(remote_bootstrap, "git_commit", return_value="abc123"):
                with self.assertRaisesRegex(RuntimeError, "refused unexpected content"):
                    remote_bootstrap.apply_pinned_patch(repository, specification)
            with mock.patch.object(remote_bootstrap, "git_commit", return_value="wrong"):
                with self.assertRaisesRegex(RuntimeError, "requires revision"):
                    remote_bootstrap.apply_pinned_patch(repository, specification)

    def test_install_node_pack_links_image_and_3d_facades(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repo"
            image_source = repository / "custom_nodes" / "ComfyColab-ZImage"
            three_d_source = repository / "custom_nodes" / "ComfyColab-3D"
            image_source.mkdir(parents=True)
            three_d_source.mkdir(parents=True)
            image_target = root / "custom_nodes" / "ComfyColab-ZImage"
            three_d_target = root / "custom_nodes" / "ComfyColab-3D"
            image_target.mkdir(parents=True)
            three_d_target.mkdir(parents=True)
            with mock.patch.multiple(
                remote_bootstrap,
                REPO_DIR=repository,
                NODE_TARGET=image_target,
                NODE_3D_TARGET=three_d_target,
            ):
                remote_bootstrap.install_node_pack()
            self.assertEqual(image_target.resolve(), image_source.resolve())
            self.assertEqual(three_d_target.resolve(), three_d_source.resolve())

    def test_awaiting_combined_cache_falls_back_to_trellis_profile(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with mock.patch.object(remote_bootstrap, "REPO_DIR", root):
            self.assertIsNone(remote_bootstrap.combined_cache_specification())
        with mock.patch.object(
            remote_bootstrap, "combined_cache_specification", return_value=None
        ), mock.patch.object(remote_bootstrap, "restore_trellis_cache", return_value=True):
            self.assertEqual(
                remote_bootstrap.restore_3d_environment_cache(),
                remote_bootstrap.TRELLIS_CACHE["profile"],
            )

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
                ULTRASHAPE_DIR=root / "UltraShape-1.0",
                NODE_TARGET=root / "node",
                NODE_3D_TARGET=root / "node-3d",
            ), mock.patch.object(remote_bootstrap, "load_state", return_value={}), mock.patch.object(
                remote_bootstrap, "http_ready", return_value=False
            ), mock.patch.object(remote_bootstrap, "clone_or_update"), mock.patch.object(
                remote_bootstrap, "install_node_pack"
            ), mock.patch.object(
                remote_bootstrap,
                "apply_pinned_patch",
                side_effect=[
                    remote_bootstrap.TRELLIS_PATCH_ID,
                    remote_bootstrap.ULTRASHAPE_PATCH_ID,
                ],
            ), mock.patch.object(
                remote_bootstrap,
                "install_dependencies",
                return_value="combined-test-cache",
            ), mock.patch.object(
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
            self.assertEqual(payload["ultrashapeCommit"], "abc123")
            self.assertEqual(payload["trellisPatch"], remote_bootstrap.TRELLIS_PATCH_ID)
            self.assertEqual(payload["ultrashapePatch"], remote_bootstrap.ULTRASHAPE_PATCH_ID)
            self.assertEqual(payload["environmentCacheProfile"], "combined-test-cache")
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
                ULTRASHAPE_DIR=root / "UltraShape-1.0",
                NODE_TARGET=root / "node",
                NODE_3D_TARGET=root / "node-3d",
            ), mock.patch.object(remote_bootstrap, "load_state", return_value={}), mock.patch.object(
                remote_bootstrap, "http_ready", return_value=False
            ), mock.patch.object(remote_bootstrap, "clone_or_update"), mock.patch.object(
                remote_bootstrap, "install_node_pack"
            ), mock.patch.object(
                remote_bootstrap,
                "apply_pinned_patch",
                side_effect=[
                    remote_bootstrap.TRELLIS_PATCH_ID,
                    remote_bootstrap.ULTRASHAPE_PATCH_ID,
                ],
            ), mock.patch.object(
                remote_bootstrap,
                "install_dependencies",
                return_value="combined-test-cache",
            ), mock.patch.object(
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
        self.assertIn("5e8dcef05df101ab00ab6cd5fdd0ed0c74fbca66", source)

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
            ), mock.patch.object(remote_bootstrap, "install_ultrashape_overlay"):
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
            self.assertGreater(part["bytes"], 0)
            self.assertRegex(part["sha256"], r"^[0-9a-f]{64}$")

    def test_trellis_cache_progress_reports_percentage_speed_and_eta(self) -> None:
        parts = [{"name": "part-000", "bytes": 1_000_000_000}]
        with mock.patch.object(
            remote_bootstrap.time,
            "monotonic",
            side_effect=[0.0, 10.0],
        ), mock.patch("builtins.print") as output:
            progress = remote_bootstrap.CacheDownloadProgress(parts, report_interval=0)
            progress.advance("part-000", 500_000_000)

        message = output.call_args.args[0]
        self.assertIn("0.50 GB/1.00 GB (50.0%)", message)
        self.assertIn("50.0 MB/s", message)
        self.assertIn("ETA 0m 10s", message)

    def test_trellis_cache_progress_reports_during_a_stall(self) -> None:
        progress = remote_bootstrap.CacheDownloadProgress(
            [{"name": "part-000", "bytes": 1_000_000_000}],
            report_interval=0.01,
        )
        with mock.patch("builtins.print") as output:
            progress.start()
            deadline = remote_bootstrap.time.monotonic() + 0.2
            while output.call_count < 2 and remote_bootstrap.time.monotonic() < deadline:
                remote_bootstrap.time.sleep(0.005)
            reports_before_stop = output.call_count
            progress.stop()

        self.assertGreaterEqual(reports_before_stop, 2)

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
                        max_attempts=1,
                    )
            self.assertFalse(destination.exists())

    def test_trellis_cache_part_resumes_after_stall(self) -> None:
        class Response(io.BytesIO):
            def __init__(self, payload: bytes, *, status: int, content_range: str | None = None):
                super().__init__(payload)
                self.status = status
                self.headers = {}
                if content_range is not None:
                    self.headers["Content-Range"] = content_range

        class StalledResponse(Response):
            def __init__(self):
                super().__init__(b"abc", status=200)
                self.stalled = False

            def read(self, size: int = -1) -> bytes:
                chunk = super().read(size)
                if chunk:
                    return chunk
                if not self.stalled:
                    self.stalled = True
                    raise socket.timeout("stalled")
                return b""

        payload = b"abcdef"
        expected = hashlib.sha256(payload).hexdigest()
        responses = [
            StalledResponse(),
            Response(b"def", status=206, content_range="bytes 3-5/6"),
        ]
        requests = []

        def open_request(request, timeout):
            requests.append((request, timeout))
            return responses.pop(0)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            remote_bootstrap.urllib.request,
            "urlopen",
            side_effect=open_request,
        ), mock.patch.object(remote_bootstrap.time, "sleep"):
            destination = Path(directory) / "part-000"
            remote_bootstrap.download_cache_part(
                {
                    "name": "part-000",
                    "url": "https://example.test/part",
                    "bytes": len(payload),
                    "sha256": expected,
                },
                destination,
                stall_timeout=7,
            )
            downloaded = destination.read_bytes()

        self.assertEqual(downloaded, payload)
        self.assertIsNone(requests[0][0].get_header("Range"))
        self.assertEqual(requests[1][0].get_header("Range"), "bytes=3-")
        self.assertEqual([timeout for _, timeout in requests], [7, 7])

    def test_trellis_cache_checksum_failure_retries_from_zero(self) -> None:
        payload = b"good"
        expected = hashlib.sha256(payload).hexdigest()
        requests = []

        def open_request(request, timeout):
            requests.append(request)
            return io.BytesIO(b"evil" if len(requests) == 1 else payload)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            remote_bootstrap.urllib.request,
            "urlopen",
            side_effect=open_request,
        ), mock.patch.object(remote_bootstrap.time, "sleep"):
            destination = Path(directory) / "part-000"
            remote_bootstrap.download_cache_part(
                {
                    "name": "part-000",
                    "url": "https://example.test/part",
                    "bytes": len(payload),
                    "sha256": expected,
                },
                destination,
            )
            downloaded = destination.read_bytes()

        self.assertEqual(downloaded, payload)
        self.assertEqual(len(requests), 2)
        self.assertIsNone(requests[1].get_header("Range"))

    def test_trellis_cache_complete_partial_is_promoted_without_network(self) -> None:
        payload = b"complete"
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            remote_bootstrap.urllib.request,
            "urlopen",
        ) as urlopen:
            destination = Path(directory) / "part-000"
            destination.with_suffix(".partial").write_bytes(payload)
            remote_bootstrap.download_cache_part(
                {
                    "name": "part-000",
                    "url": "https://example.test/part",
                    "bytes": len(payload),
                    "sha256": expected,
                },
                destination,
            )
            self.assertEqual(destination.read_bytes(), payload)
        urlopen.assert_not_called()

    def test_trellis_cache_restarts_when_server_ignores_range(self) -> None:
        class Response(io.BytesIO):
            status = 200
            headers = {}

        payload = b"abcdef"
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            remote_bootstrap.urllib.request,
            "urlopen",
            return_value=Response(payload),
        ) as urlopen:
            destination = Path(directory) / "part-000"
            destination.with_suffix(".partial").write_bytes(b"abc")
            remote_bootstrap.download_cache_part(
                {
                    "name": "part-000",
                    "url": "https://example.test/part",
                    "bytes": len(payload),
                    "sha256": expected,
                },
                destination,
            )
            self.assertEqual(destination.read_bytes(), payload)
        self.assertEqual(urlopen.call_args.args[0].get_header("Range"), "bytes=3-")

    def test_trellis_cache_oversized_response_retries_from_zero(self) -> None:
        payload = b"good"
        expected = hashlib.sha256(payload).hexdigest()
        requests = []

        def open_request(request, timeout):
            requests.append(request)
            return io.BytesIO(b"too-long" if len(requests) == 1 else payload)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            remote_bootstrap.urllib.request,
            "urlopen",
            side_effect=open_request,
        ), mock.patch.object(remote_bootstrap.time, "sleep"):
            destination = Path(directory) / "part-000"
            remote_bootstrap.download_cache_part(
                {
                    "name": "part-000",
                    "url": "https://example.test/part",
                    "bytes": len(payload),
                    "sha256": expected,
                },
                destination,
            )
            self.assertEqual(destination.read_bytes(), payload)

        self.assertEqual(len(requests), 2)
        self.assertIsNone(requests[1].get_header("Range"))

    def test_trellis_cache_permanent_http_error_does_not_retry(self) -> None:
        error = urllib.error.HTTPError(
            "https://example.test/missing",
            404,
            "missing",
            hdrs=None,
            fp=None,
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            remote_bootstrap.urllib.request,
            "urlopen",
            side_effect=error,
        ) as urlopen:
            with self.assertRaisesRegex(RuntimeError, "HTTP 404"):
                remote_bootstrap.download_cache_part(
                    {
                        "name": "part-000",
                        "url": "https://example.test/missing",
                        "bytes": 4,
                        "sha256": "0" * 64,
                    },
                    Path(directory) / "part-000",
                )
        urlopen.assert_called_once()

    def test_trellis_cache_retryable_http_error_exhausts_attempts(self) -> None:
        error = urllib.error.HTTPError(
            "https://example.test/busy",
            503,
            "busy",
            hdrs=None,
            fp=None,
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            remote_bootstrap.urllib.request,
            "urlopen",
            side_effect=error,
        ) as urlopen, mock.patch.object(remote_bootstrap.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                remote_bootstrap.download_cache_part(
                    {
                        "name": "part-000",
                        "url": "https://example.test/busy",
                        "bytes": 4,
                        "sha256": "0" * 64,
                    },
                    Path(directory) / "part-000",
                    max_attempts=3,
                )
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(sleep.call_args_list, [mock.call(2), mock.call(4)])

    def test_trellis_cache_disk_error_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            remote_bootstrap.urllib.request,
            "urlopen",
            return_value=io.BytesIO(b"data"),
        ) as urlopen, mock.patch.object(
            remote_bootstrap.Path,
            "open",
            side_effect=OSError(errno.ENOSPC, "disk full"),
        ), mock.patch.object(remote_bootstrap.time, "sleep") as sleep:
            with self.assertRaisesRegex(OSError, "disk full"):
                remote_bootstrap.download_cache_part(
                    {
                        "name": "part-000",
                        "url": "https://example.test/part",
                        "bytes": 4,
                        "sha256": "0" * 64,
                    },
                    Path(directory) / "part-000",
                )
        urlopen.assert_called_once()
        sleep.assert_not_called()

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
            ), mock.patch.object(remote_bootstrap, "run") as run, mock.patch.object(
                remote_bootstrap, "install_ultrashape_overlay"
            ):
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
