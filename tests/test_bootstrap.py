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
        trellis_categories = json.loads(
            (root / "patches" / "trellis2-advanced-categories.json").read_text(
                encoding="utf-8"
            )
        )
        trellis_multiview = json.loads(
            (root / "patches" / "trellis2-multiview-weight-cache.json").read_text(
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
        self.assertEqual(
            trellis_categories["patch_id"],
            remote_bootstrap.TRELLIS_CATEGORY_PATCH_ID,
        )
        self.assertEqual(trellis_categories["revision"], remote_bootstrap.TRELLIS_REF)
        self.assertEqual(
            trellis_multiview["patch_id"], remote_bootstrap.TRELLIS_MULTIVIEW_PATCH_ID
        )
        self.assertEqual(trellis_multiview["revision"], remote_bootstrap.TRELLIS_REF)
        multiview_source = "\n".join(
            line
            for file_spec in trellis_multiview["files"]
            for replacement in file_spec["replacements"]
            for line in replacement["after_lines"]
        )
        self.assertIn("view_weights=view_weights", multiview_source)
        self.assertIn("Compute spatial blend weights once per sampler run", multiview_source)
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
        decoder = next(
            item for item in ultrashape["files"]
            if item["path"] == "ultrashape/models/autoencoders/volume_decoders.py"
        )
        decoder_patch = "\n".join(
            line
            for replacement in decoder["replacements"]
            for line in replacement["after_lines"]
        )
        self.assertIn("class NoDecodableSurface", decoder_patch)
        self.assertIn("if next_points.numel() == 0", decoder_patch)
        self.assertIn("requested_resolution=requested_octree_resolution", decoder_patch)

    def test_pixal3d_bootstrap_runs_after_trellis_and_before_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_dir = root / "state"
            calls: list[str] = []

            class Process:
                def __init__(self, pid: int):
                    self.pid = pid

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
                PIXAL3D_DIR=root / "Pixal3D",
                NODE_TARGET=root / "node",
                NODE_3D_TARGET=root / "node-3d",
            ), mock.patch.object(
                remote_bootstrap, "load_state", return_value={}
            ), mock.patch.object(
                remote_bootstrap, "http_ready", return_value=False
            ), mock.patch.object(
                remote_bootstrap, "clone_or_update", side_effect=lambda *_args: calls.append("clone")
            ), mock.patch.object(
                remote_bootstrap, "install_node_pack", side_effect=lambda: calls.append("install_node_pack")
            ), mock.patch.object(
                remote_bootstrap,
                "validate_triposplat_core_support",
                side_effect=lambda: calls.append("validate_triposplat_core_support"),
            ), mock.patch.object(
                remote_bootstrap,
                "apply_pinned_patch",
                side_effect=[
                    remote_bootstrap.TRELLIS_PATCH_ID,
                    remote_bootstrap.TRELLIS_CATEGORY_PATCH_ID,
                    remote_bootstrap.TRELLIS_MULTIVIEW_PATCH_ID,
                    remote_bootstrap.ULTRASHAPE_PATCH_ID,
                ],
            ), mock.patch.object(
                remote_bootstrap,
                "install_dependencies",
                side_effect=lambda: calls.append("install_dependencies") or "combined-test-cache",
            ), mock.patch.object(
                remote_bootstrap,
                "install_pixal3d",
                side_effect=lambda: calls.append("install_pixal3d") or "pixal3d-test-cache",
            ), mock.patch.object(
                remote_bootstrap,
                "validate_pixal3d_runtime",
                side_effect=lambda: calls.append("validate_pixal3d_runtime"),
            ), mock.patch.object(
                remote_bootstrap, "cloudflared_path", return_value=Path("/tmp/cloudflared")
            ), mock.patch.object(
                remote_bootstrap, "wait_for_comfy"
            ), mock.patch.object(
                remote_bootstrap, "wait_for_tunnel", side_effect=RuntimeError("no tunnel")
            ), mock.patch.object(
                remote_bootstrap, "request_colab_proxy_url", return_value="https://abc-8188.colab.googleusercontent.com/"
            ), mock.patch.object(
                remote_bootstrap, "git_commit", return_value="abc123"
            ), mock.patch.object(
                remote_bootstrap.subprocess, "Popen", side_effect=[Process(101), Process(202)]
            ), mock.patch.object(
                remote_bootstrap, "stop_managed_process"
            ), mock.patch.object(
                remote_bootstrap, "stop_started_process"
            ):
                remote_bootstrap.main()

            self.assertLess(calls.index("install_dependencies"), calls.index("install_pixal3d"))
            self.assertLess(calls.index("validate_triposplat_core_support"), calls.index("install_pixal3d"))
            self.assertLess(calls.index("install_pixal3d"), calls.index("validate_pixal3d_runtime"))
            payload = json.loads((state_dir / "runtime.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["pixal3dCommit"], "abc123")
            self.assertEqual(payload["pixal3dCacheProfile"], "pixal3d-test-cache")

    def test_pixal3d_probe_requires_import_cuda_and_file3d_export(self) -> None:
        completed = mock.Mock(returncode=0)
        with mock.patch.object(remote_bootstrap.subprocess, "run", return_value=completed) as run:
            remote_bootstrap.validate_pixal3d_runtime(Path("/cached/python"))

        command = run.call_args.args[0]
        source = command[2]
        self.assertIn("import pixal3d", source)
        self.assertIn("torch.cuda.is_available()", source)
        self.assertIn("export_glb", source)

    def test_trellis_category_patch_changes_categories_only(self) -> None:
        root = Path(__file__).resolve().parents[1]
        specification = json.loads(
            (root / "patches" / "trellis2-advanced-categories.json").read_text(
                encoding="utf-8"
            )
        )
        replacements = [
            replacement
            for file_specification in specification["files"]
            for replacement in file_specification["replacements"]
        ]

        self.assertEqual(
            {file_specification["path"] for file_specification in specification["files"]},
            {
                "nodes/nodes_loader.py",
                "nodes/nodes_native_sampling.py",
                "nodes/nodes_export.py",
                "nodes/nodes_inference.py",
                "nodes/nodes_unwrap.py",
            },
        )
        self.assertEqual(
            sum(replacement["occurrences"] for replacement in replacements),
            24,
        )
        allowed_changes = {
            (
                'category="TRELLIS2",',
                'category="TRELLIS2 / Advanced",',
            ),
            (
                'category="TRELLIS2/Native",',
                'category="TRELLIS2 / Advanced/Native",',
            ),
        }
        for replacement in replacements:
            self.assertEqual(len(replacement["before_lines"]), 1)
            self.assertEqual(len(replacement["after_lines"]), 1)
            self.assertIn(
                (
                    replacement["before_lines"][0].strip(),
                    replacement["after_lines"][0].strip(),
                ),
                allowed_changes,
            )

        rendered = json.dumps(specification, sort_keys=True)
        for protected_contract_token in (
            "class ",
            "schema=io.Schema",
            "node_id=",
            "inputs=[",
            "outputs=[",
        ):
            self.assertNotIn(protected_contract_token, rendered)

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

    def test_non_sm120_ultrashape_validation_still_executes_a_cubvh_kernel(self) -> None:
        completed = mock.Mock(returncode=0)
        with mock.patch.object(remote_bootstrap.subprocess, "run", return_value=completed) as run:
            remote_bootstrap.validate_ultrashape_imports(Path("/cached/python"))
        source = run.call_args.args[0][2]
        self.assertIn("cubvh.cuBVH", source)
        self.assertIn("unsigned_distance", source)
        self.assertIn("torch.cuda.synchronize", source)
        self.assertNotIn("get_device_capability() == (12, 0)", source)

    def test_source_trellis_validation_does_not_require_sm120(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            for name in ("trellis2-nodes", "geometrypack-nodes"):
                python = workspace / ".pixi" / "envs" / name / "bin" / "python"
                python.parent.mkdir(parents=True)
                python.touch()
            completed = mock.Mock(returncode=0)
            with mock.patch.object(
                remote_bootstrap.subprocess,
                "run",
                return_value=completed,
            ) as run:
                remote_bootstrap.validate_trellis_cache(workspace)
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            source = call.args[0][2]
            self.assertIn("torch.cuda.synchronize", source)
            self.assertNotIn("get_device_capability() == (12, 0)", source)

    def test_sm120_trellis_cache_validation_retains_capability_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            for name in ("trellis2-nodes", "geometrypack-nodes"):
                python = workspace / ".pixi" / "envs" / name / "bin" / "python"
                python.parent.mkdir(parents=True)
                python.touch()
            completed = mock.Mock(returncode=0)
            with mock.patch.object(
                remote_bootstrap.subprocess,
                "run",
                return_value=completed,
            ) as run:
                remote_bootstrap.validate_trellis_cache(
                    workspace,
                    require_sm120=True,
                )
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            source = call.args[0][2]
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

    def test_pixal3d_skips_sm120_worker_on_other_gpus(self) -> None:
        with mock.patch.object(
            remote_bootstrap,
            "sm120_gpu_available",
            return_value=False,
        ), mock.patch.object(remote_bootstrap, "restore_pixal3d_cache") as restore, mock.patch.object(
            remote_bootstrap, "install_pixal3d_source"
        ) as source_install:
            self.assertEqual(remote_bootstrap.install_pixal3d(), "unavailable")
        restore.assert_not_called()
        source_install.assert_not_called()

    def test_pixal3d_source_install_remains_available_when_sm120_cache_is_incompatible(
        self,
    ) -> None:
        with mock.patch.object(
            remote_bootstrap,
            "sm120_gpu_available",
            return_value=True,
        ), mock.patch.object(
            remote_bootstrap,
            "restore_pixal3d_cache",
            return_value=False,
        ) as restore, mock.patch.object(
            remote_bootstrap,
            "install_pixal3d_source",
            return_value="source-install",
        ) as source_install:
            self.assertEqual(remote_bootstrap.install_pixal3d(), "source-install")
        restore.assert_called_once_with()
        source_install.assert_called_once_with()

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

    def test_install_node_pack_links_all_first_party_facades(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repo"
            image_source = repository / "custom_nodes" / "ComfyColab-ZImage"
            three_d_source = repository / "custom_nodes" / "ComfyColab-3D"
            triposplat_source = repository / "custom_nodes" / "ComfyColab-Triposplat"
            ltx_source = repository / "custom_nodes" / "ComfyColab-LTXVideo"
            image_source.mkdir(parents=True)
            three_d_source.mkdir(parents=True)
            triposplat_source.mkdir(parents=True)
            ltx_source.mkdir(parents=True)
            image_target = root / "custom_nodes" / "ComfyColab-ZImage"
            three_d_target = root / "custom_nodes" / "ComfyColab-3D"
            triposplat_target = root / "custom_nodes" / "ComfyColab-Triposplat"
            ltx_target = root / "custom_nodes" / "ComfyColab-LTXVideo"
            image_target.mkdir(parents=True)
            three_d_target.mkdir(parents=True)
            triposplat_target.mkdir(parents=True)
            ltx_target.mkdir(parents=True)
            with mock.patch.multiple(
                remote_bootstrap,
                REPO_DIR=repository,
                NODE_TARGET=image_target,
                NODE_3D_TARGET=three_d_target,
                NODE_TRIPOSPLAT_TARGET=triposplat_target,
                NODE_LTX_TARGET=ltx_target,
            ):
                remote_bootstrap.install_node_pack()
            self.assertEqual(image_target.resolve(), image_source.resolve())
            self.assertEqual(three_d_target.resolve(), three_d_source.resolve())
            self.assertEqual(triposplat_target.resolve(), triposplat_source.resolve())
            self.assertEqual(ltx_target.resolve(), ltx_source.resolve())

    def test_install_node_pack_links_configured_daughter_facades(self) -> None:
        targets = {
            "3d": "ComfyColab-3D",
            "3dgs": "ComfyColab-Triposplat",
            "image": "ComfyColab-ZImage",
            "video": "ComfyColab-LTXVideo",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comfy = root / "ComfyUI"
            configured: dict[str, tuple[Path, dict[str, object]]] = {}
            for pack_id, target_name in targets.items():
                pack_root = root / "packs" / pack_id
                (pack_root / "custom_nodes" / target_name).mkdir(parents=True)
                configured[pack_id] = (
                    pack_root,
                    {
                        "node_roots": [
                            {
                                "source": f"custom_nodes/{target_name}",
                                "target": target_name,
                            }
                        ]
                    },
                )
            with mock.patch.object(remote_bootstrap, "COMFY_DIR", comfy):
                remote_bootstrap.install_node_pack(configured)
            for pack_id, target_name in targets.items():
                target = comfy / "custom_nodes" / target_name
                self.assertTrue(target.is_symlink())
                self.assertEqual(
                    target.resolve(),
                    (
                        configured[pack_id][0]
                        / "custom_nodes"
                        / target_name
                    ).resolve(),
                )

    def test_prepare_configured_node_pack_verifies_manifest_digest(self) -> None:
        manifest_bytes = json.dumps(
            {
                "schema": 1,
                "id": "image",
                "version": "0.1.0",
                "node_roots": [
                    {
                        "source": "custom_nodes/ComfyColab-ZImage",
                        "target": "ComfyColab-ZImage",
                    }
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        commit = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            packs_dir = Path(directory) / "packs"
            stale_root = packs_dir / "image"
            stale_root.mkdir(parents=True)
            stale_node = stale_root / "custom_nodes" / "stale.py"
            stale_node.parent.mkdir(parents=True)
            stale_node.write_text("modified = True\n", encoding="utf-8")

            def clone(_url: str, destination: Path, _ref: str) -> None:
                destination.mkdir(parents=True)
                (destination / "comfycolab-pack.json").write_bytes(manifest_bytes)

            with (
                mock.patch.multiple(
                    remote_bootstrap,
                    PACKS_DIR=packs_dir,
                    CONFIG={
                        "packs": [
                            {
                                "id": "image",
                                "repository": "https://github.com/example/image.git",
                                "commit": commit,
                                "manifest_sha256": hashlib.sha256(
                                    manifest_bytes
                                ).hexdigest(),
                            }
                        ]
                    },
                ),
                mock.patch.object(
                    remote_bootstrap,
                    "clone_or_update",
                    side_effect=clone,
                ),
                mock.patch.object(
                    remote_bootstrap,
                    "git_commit",
                    return_value=commit,
                ),
            ):
                prepared = remote_bootstrap.prepare_configured_node_packs()
            self.assertEqual(set(prepared), {"image"})
            self.assertEqual(prepared["image"][1]["id"], "image")
            self.assertFalse(stale_node.exists())

    def test_cubepart_source_requires_explicit_license_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cube_dir = Path(directory) / "cube"
            cube_dir.mkdir()
            (cube_dir / "stale.py").write_text("stale = True\n", encoding="utf-8")
            with (
                mock.patch.multiple(
                    remote_bootstrap,
                    CONFIG={"accepted_licenses": []},
                    CUBE_DIR=cube_dir,
                ),
                mock.patch.object(remote_bootstrap, "clone_or_update") as clone,
            ):
                self.assertFalse(remote_bootstrap.prepare_cubepart_source())
            clone.assert_not_called()
            self.assertFalse(cube_dir.exists())

        with (
            mock.patch.object(
                remote_bootstrap,
                "CONFIG",
                {"accepted_licenses": ["accept_research_license"]},
            ),
            mock.patch.object(remote_bootstrap, "clone_or_update") as clone,
        ):
            self.assertTrue(remote_bootstrap.prepare_cubepart_source())
        clone.assert_called_once_with(
            "https://github.com/Roblox/cube.git",
            remote_bootstrap.CUBE_DIR,
            remote_bootstrap.CUBEPART_REF,
        )

    def test_legacy_cloudflared_is_versioned_and_digest_verified(self) -> None:
        payload = b"verified-cloudflared"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            with (
                mock.patch.object(remote_bootstrap, "STATE_DIR", state_dir),
                mock.patch.object(
                    remote_bootstrap,
                    "CLOUDFLARED_ASSETS",
                    {"amd64": ("cloudflared-linux-amd64", digest)},
                ),
                mock.patch.object(
                    remote_bootstrap.platform,
                    "machine",
                    return_value="x86_64",
                ),
                mock.patch.object(
                    remote_bootstrap.urllib.request,
                    "urlopen",
                    return_value=io.BytesIO(payload),
                ) as urlopen,
            ):
                path = remote_bootstrap.cloudflared_path()
            self.assertEqual(path.read_bytes(), payload)
            url = urlopen.call_args.args[0]
            self.assertIn(
                f"/download/{remote_bootstrap.CLOUDFLARED_VERSION}/",
                url,
            )
            self.assertNotIn("/latest/", url)

            path.unlink()
            with (
                mock.patch.object(remote_bootstrap, "STATE_DIR", state_dir),
                mock.patch.object(
                    remote_bootstrap,
                    "CLOUDFLARED_ASSETS",
                    {"amd64": ("cloudflared-linux-amd64", "0" * 64)},
                ),
                mock.patch.object(
                    remote_bootstrap.platform,
                    "machine",
                    return_value="x86_64",
                ),
                mock.patch.object(
                    remote_bootstrap.urllib.request,
                    "urlopen",
                    return_value=io.BytesIO(payload),
                ),
                self.assertRaisesRegex(RuntimeError, "digest mismatch"),
            ):
                remote_bootstrap.cloudflared_path()
            self.assertEqual(list(state_dir.glob("*.part")), [])

    def test_configured_daughter_nodes_must_appear_in_object_info(self) -> None:
        configured = {
            "image": (
                Path("/content/.comfycolab/packs/image"),
                {
                    "probes": [
                        {
                            "phase": "post_start",
                            "type": "comfy_node_ids",
                            "values": ["ComfyColabZImageTurboBundleLoader"],
                        }
                    ]
                },
            )
        }
        response = io.BytesIO(
            json.dumps(
                {"ComfyColabZImageTurboBundleLoader": {"display_name": "Z-Image"}}
            ).encode()
        )
        with mock.patch.object(
            remote_bootstrap.urllib.request,
            "urlopen",
            return_value=response,
        ):
            self.assertEqual(
                remote_bootstrap.validate_configured_node_discovery(
                    8188,
                    configured,
                ),
                {"image": ["ComfyColabZImageTurboBundleLoader"]},
            )

        missing_response = io.BytesIO(b"{}")
        with (
            mock.patch.object(
                remote_bootstrap.urllib.request,
                "urlopen",
                return_value=missing_response,
            ),
            self.assertRaisesRegex(RuntimeError, "missing ComfyUI nodes"),
        ):
            remote_bootstrap.validate_configured_node_discovery(
                8188,
                configured,
            )

    def test_triposplat_core_support_requires_native_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            comfy = Path(directory)
            for relative_path, symbols in remote_bootstrap.TRIPOSPLAT_CORE_REQUIREMENTS.items():
                path = comfy / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n".join(symbols), encoding="utf-8")
            with mock.patch.object(remote_bootstrap, "COMFY_DIR", comfy):
                remote_bootstrap.validate_triposplat_core_support()
                missing_path = comfy / "comfy_extras" / "nodes_triposplat.py"
                missing_path.write_text("TripoSplatConditioning\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "native TripoSplat"):
                    remote_bootstrap.validate_triposplat_core_support()

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
                LTX_VIDEO_DIR=root / "ComfyUI-LTXVideo",
                TRELLIS_DIR=root / "ComfyUI-TRELLIS2",
                GEOMETRY_DIR=root / "ComfyUI-GeometryPack",
                ULTRASHAPE_DIR=root / "UltraShape-1.0",
                NODE_TARGET=root / "node",
                NODE_3D_TARGET=root / "node-3d",
                NODE_TRIPOSPLAT_TARGET=root / "node-triposplat",
            ), mock.patch.object(remote_bootstrap, "load_state", return_value={}), mock.patch.object(
                remote_bootstrap, "http_ready", return_value=False
            ), mock.patch.object(remote_bootstrap, "clone_or_update"), mock.patch.object(
                remote_bootstrap, "validate_triposplat_core_support"
            ), mock.patch.object(
                remote_bootstrap, "install_node_pack"
            ), mock.patch.object(
                remote_bootstrap,
                "apply_pinned_patch",
                side_effect=[
                    remote_bootstrap.TRELLIS_PATCH_ID,
                    remote_bootstrap.TRELLIS_CATEGORY_PATCH_ID,
                    remote_bootstrap.TRELLIS_MULTIVIEW_PATCH_ID,
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
            ) as popen, mock.patch.object(
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
            self.assertEqual(payload["ltxVideoCommit"], "abc123")
            self.assertEqual(payload["geometryCommit"], "abc123")
            self.assertEqual(payload["ultrashapeCommit"], "abc123")
            self.assertEqual(payload["skinTokensModelRef"], remote_bootstrap.SKINTOKENS_MODEL_REF)
            self.assertEqual(payload["skinTokensQwenRef"], remote_bootstrap.SKINTOKENS_QWEN_REF)
            self.assertEqual(payload["cubePartModelRef"], remote_bootstrap.CUBEPART_MODEL_REF)
            self.assertEqual(payload["triposplatCoreRef"], remote_bootstrap.COMFY_REF)
            self.assertTrue(payload["triposplatCoreReady"])
            self.assertEqual(payload["trellisPatch"], remote_bootstrap.TRELLIS_PATCH_ID)
            self.assertEqual(
                payload["trellisCategoryPatch"],
                remote_bootstrap.TRELLIS_CATEGORY_PATCH_ID,
            )
            self.assertEqual(
                payload["trellisMultiviewPatch"],
                remote_bootstrap.TRELLIS_MULTIVIEW_PATCH_ID,
            )
            self.assertEqual(payload["ultrashapePatch"], remote_bootstrap.ULTRASHAPE_PATCH_ID)
            self.assertEqual(
                payload["comfyEnvTimeoutPatch"],
                remote_bootstrap.COMFY_ENV_TIMEOUT_PATCH_ID,
            )
            self.assertEqual(
                payload["isolatedCallTimeoutSeconds"],
                remote_bootstrap.COMFY_ENV_CALL_TIMEOUT_SECONDS,
            )
            self.assertEqual(payload["environmentCacheProfile"], "combined-test-cache")
            comfy_environment = popen.call_args_list[0].kwargs["env"]
            self.assertEqual(
                comfy_environment["COMFY_ENV_CALL_TIMEOUT"],
                str(remote_bootstrap.COMFY_ENV_CALL_TIMEOUT_SECONDS),
            )
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
                NODE_TRIPOSPLAT_TARGET=root / "node-triposplat",
            ), mock.patch.object(remote_bootstrap, "load_state", return_value={}), mock.patch.object(
                remote_bootstrap, "http_ready", return_value=False
            ), mock.patch.object(remote_bootstrap, "clone_or_update"), mock.patch.object(
                remote_bootstrap, "validate_triposplat_core_support"
            ), mock.patch.object(
                remote_bootstrap, "install_node_pack"
            ), mock.patch.object(
                remote_bootstrap,
                "apply_pinned_patch",
                side_effect=[
                    remote_bootstrap.TRELLIS_PATCH_ID,
                    remote_bootstrap.TRELLIS_CATEGORY_PATCH_ID,
                    remote_bootstrap.TRELLIS_MULTIVIEW_PATCH_ID,
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
        self.assertIn("aceeae9635f6d493f2893ba3c411a1c36031788a", source)
        self.assertIn("9b878516f2dc2fd873f4f6cceadba403dd12d83e", source)
        self.assertIn("c67199de05705642258e727fa118f412877b4ebf", source)
        self.assertIn("5e8dcef05df101ab00ab6cd5fdd0ed0c74fbca66", source)

    def test_trellis_dependencies_use_isolated_upstream_installer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comfy_dir = root / "ComfyUI"
            gguf_dir = comfy_dir / "custom_nodes" / "ComfyUI-GGUF"
            ltx_video_dir = comfy_dir / "custom_nodes" / "ComfyUI-LTXVideo"
            trellis_dir = comfy_dir / "custom_nodes" / "ComfyUI-TRELLIS2"
            comfy_dir.mkdir(parents=True)
            gguf_dir.mkdir(parents=True)
            ltx_video_dir.mkdir(parents=True)
            trellis_dir.mkdir(parents=True)
            (gguf_dir / "requirements.txt").write_text("gguf\n", encoding="utf-8")
            (ltx_video_dir / "requirements.txt").write_text(
                "diffusers\n", encoding="utf-8"
            )
            (trellis_dir / "requirements.txt").write_text(
                "comfy-env==0.3.89\n", encoding="utf-8"
            )
            commands: list[tuple[list[str], Path | None]] = []
            with mock.patch.multiple(
                remote_bootstrap,
                COMFY_DIR=comfy_dir,
                GGUF_DIR=gguf_dir,
                LTX_VIDEO_DIR=ltx_video_dir,
                TRELLIS_DIR=trellis_dir,
            ), mock.patch.object(
                remote_bootstrap,
                "run",
                side_effect=lambda command, cwd=None: commands.append((command, cwd)),
            ), mock.patch.object(remote_bootstrap, "install_ultrashape_overlay"), mock.patch.object(
                remote_bootstrap, "patch_comfyenv_call_timeout"
            ) as timeout_patch:
                remote_bootstrap.install_dependencies()

            self.assertEqual(commands[0][1], comfy_dir)
            self.assertEqual(commands[1][0][-1], str(gguf_dir / "requirements.txt"))
            self.assertEqual(
                commands[2][0][-1],
                str(ltx_video_dir / "requirements.txt"),
            )
            self.assertEqual(
                commands[3][0][-3:],
                ["-r", str(trellis_dir / "requirements.txt"), "--upgrade"],
            )
            self.assertEqual(
                commands[4],
                ([remote_bootstrap.sys.executable, "install.py"], trellis_dir),
            )
            timeout_patch.assert_called_once_with()

    def test_comfyenv_timeout_patch_is_exact_idempotent_and_cache_preserving(self) -> None:
        source = (
            "import os\n"
            "def proxy(worker):\n"
            "    try:\n"
            "        return worker.call_method(\n"
            "                    timeout=600.0,\n"
            "        )\n"
            "            except (RuntimeError, ConnectionError):\n"
            "                raise\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "metadata.py"
            metadata.write_text(source, encoding="utf-8")
            self.assertEqual(
                remote_bootstrap.patch_comfyenv_call_timeout(
                    metadata,
                    installed_version=remote_bootstrap.COMFY_ENV_VERSION,
                ),
                remote_bootstrap.COMFY_ENV_TIMEOUT_PATCH_ID,
            )
            patched = metadata.read_text(encoding="utf-8")
            self.assertIn(
                'timeout=float(os.environ.get("COMFY_ENV_CALL_TIMEOUT", "600.0"))',
                patched,
            )
            self.assertIn(
                "except (RuntimeError, ConnectionError, TimeoutError):",
                patched,
            )
            self.assertNotIn("force_download", patched)
            self.assertNotIn("unlink", patched)
            self.assertEqual(
                remote_bootstrap.patch_comfyenv_call_timeout(
                    metadata,
                    installed_version=remote_bootstrap.COMFY_ENV_VERSION,
                ),
                remote_bootstrap.COMFY_ENV_TIMEOUT_PATCH_ID,
            )

    def test_comfyenv_timeout_patch_rejects_version_and_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "metadata.py"
            metadata.write_text("timeout=601.0\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unexpected comfy-env version"):
                remote_bootstrap.patch_comfyenv_call_timeout(
                    metadata,
                    installed_version="0.3.90",
                )
            with self.assertRaisesRegex(RuntimeError, "drifted or partially patched"):
                remote_bootstrap.patch_comfyenv_call_timeout(
                    metadata,
                    installed_version=remote_bootstrap.COMFY_ENV_VERSION,
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
            ltx_video_dir = comfy_dir / "custom_nodes" / "ComfyUI-LTXVideo"
            trellis_dir = comfy_dir / "custom_nodes" / "ComfyUI-TRELLIS2"
            workspace = root / ".ce"
            comfy_dir.mkdir(parents=True)
            ltx_video_dir.mkdir(parents=True)
            trellis_dir.mkdir(parents=True)
            (ltx_video_dir / "requirements.txt").write_text(
                "diffusers\n", encoding="utf-8"
            )
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
                LTX_VIDEO_DIR=ltx_video_dir,
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
            ), mock.patch.object(
                remote_bootstrap, "patch_comfyenv_call_timeout"
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

    def test_legacy_launch_uses_reserved_prod_colab_proxy_and_cors(self) -> None:
        proxy = (
            "https://8188-gpu-g4-s-kkb-use5c1-3m501x4q87ilk-c."
            "us-east5-1.prod.colab.dev/"
        )
        with mock.patch.dict(
            remote_bootstrap.os.environ,
            {
                "COMFYCOLAB_PROXY_URL": proxy,
                "COMFYCOLAB_CORS_ORIGIN": proxy.removesuffix("/"),
            },
            clear=True,
        ):
            self.assertEqual(
                remote_bootstrap.request_colab_proxy_url(8188),
                None,
            )
            with mock.patch.object(
                remote_bootstrap,
                "CONFIG",
                {"colab_proxy": True},
            ):
                self.assertEqual(
                    remote_bootstrap.request_colab_proxy_url(8188),
                    proxy,
                )
                self.assertEqual(
                    remote_bootstrap.comfy_launch_command(
                        8188,
                        colab_proxy=True,
                    )[-2:],
                    ["--enable-cors-header", proxy.removesuffix("/")],
                )

    def test_legacy_reuse_requires_same_lock_cors_and_runtime_mode(self) -> None:
        state = {
            "lockSha256": "a" * 64,
            "corsOrigin": "https://proxy.prod.colab.dev",
            "runtimeMode": "legacy-full",
            "acceptedLicenses": [],
            "comfyUrl": "https://proxy.prod.colab.dev/",
            "comfyPid": 42,
        }
        with (
            mock.patch.object(remote_bootstrap, "pid_alive", return_value=True),
            mock.patch.object(remote_bootstrap, "http_ready", return_value=True),
        ):
            self.assertTrue(
                remote_bootstrap.running_comfy_matches(
                    state,
                    lock_sha256="a" * 64,
                    cors_origin="https://proxy.prod.colab.dev",
                    runtime_mode="legacy-full",
                    accepted_licenses=[],
                    port=8188,
                    refresh=False,
                )
            )
            for field, value in (
                ("lock_sha256", "b" * 64),
                ("cors_origin", "https://other.prod.colab.dev"),
                ("runtime_mode", "generic"),
            ):
                arguments = {
                    "lock_sha256": "a" * 64,
                    "cors_origin": "https://proxy.prod.colab.dev",
                    "runtime_mode": "legacy-full",
                    "accepted_licenses": [],
                    "port": 8188,
                    "refresh": False,
                }
                arguments[field] = value
                self.assertFalse(
                    remote_bootstrap.running_comfy_matches(state, **arguments)
                )
            self.assertFalse(
                remote_bootstrap.running_comfy_matches(
                    state,
                    lock_sha256="a" * 64,
                    cors_origin="https://proxy.prod.colab.dev",
                    runtime_mode="legacy-full",
                    accepted_licenses=["accept_research_license"],
                    port=8188,
                    refresh=False,
                )
            )

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
