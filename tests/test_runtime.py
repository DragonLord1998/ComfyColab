from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from comfycolab import runtime


def canonical(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


class RuntimeContractTests(unittest.TestCase):
    def core_lock(self) -> dict[str, object]:
        return {
            "schema": 1,
            "core": {
                "version": "0.2.0",
                "repository": "https://github.com/example/ComfyColab.git",
                "commit": "a" * 40,
            },
            "comfyui": {
                "repository": "https://github.com/Comfy-Org/ComfyUI.git",
                "commit": "b" * 40,
            },
            "packs": [],
            "dependencies": [],
            "patches": [],
            "environments": [],
            "runtime_env": [],
        }

    def full_node_lock(self) -> dict[str, object]:
        lock = self.core_lock()
        lock["packs"] = [
            {
                "id": pack_id,
                "repository": f"https://github.com/example/{pack_id}.git",
                "commit": character * 40,
                "manifest_sha256": character * 64,
            }
            for pack_id, character in (
                ("3d", "c"),
                ("3dgs", "d"),
                ("image", "e"),
                ("video", "f"),
            )
        ]
        return lock

    def test_load_lock_requires_digest_and_canonical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock.json"
            data = canonical(self.core_lock())
            path.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            self.assertEqual(runtime.load_lock(path, digest), self.core_lock())
            with self.assertRaisesRegex(runtime.RuntimeContractError, "digest mismatch"):
                runtime.load_lock(path, "0" * 64)
            path.write_text(json.dumps(self.core_lock(), indent=2), encoding="utf-8")
            pretty_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(runtime.RuntimeContractError, "canonical"):
                runtime.load_lock(path, pretty_digest)

    def test_comfy_launch_command_preserves_non_proxy_arguments(self) -> None:
        self.assertEqual(
            runtime.comfy_launch_command(
                8188,
                colab_proxy=False,
                inherited_environment={
                    "COMFYCOLAB_CORS_ORIGIN": "https://untrusted.example"
                },
            ),
            [
                runtime.sys.executable,
                "main.py",
                "--listen",
                "127.0.0.1",
                "--port",
                "8188",
            ],
        )

    def test_comfy_launch_command_uses_default_colab_cors_origin(self) -> None:
        self.assertEqual(
            runtime.comfy_launch_command(
                8188,
                colab_proxy=True,
                inherited_environment={},
            ),
            [
                runtime.sys.executable,
                "main.py",
                "--listen",
                "127.0.0.1",
                "--port",
                "8188",
                "--enable-cors-header",
                "https://colab.research.google.com",
            ],
        )

    def test_comfy_launch_command_accepts_trusted_proxy_origins(self) -> None:
        origins = {
            (
                "https://8188-gpu-g4-s-kkb-use5c1-3m501x4q87ilk-c."
                "us-east5-1.prod.colab.dev/"
            ): (
                "https://8188-gpu-g4-s-kkb-use5c1-3m501x4q87ilk-c."
                "us-east5-1.prod.colab.dev"
            ),
            "https://abc-8188.colab.googleusercontent.com": (
                "https://abc-8188.colab.googleusercontent.com"
            ),
        }
        for origin, expected in origins.items():
            with self.subTest(origin=origin):
                command = runtime.comfy_launch_command(
                    8188,
                    colab_proxy=True,
                    inherited_environment={"COMFYCOLAB_CORS_ORIGIN": origin},
                )
                self.assertEqual(
                    command[-2:],
                    ["--enable-cors-header", expected],
                )

    def test_comfy_launch_command_rejects_untrusted_cors_origins(self) -> None:
        for origin in (
            "http://colab.research.google.com",
            "https://prod.colab.dev",
            "https://googleusercontent.com",
            "https://colab.research.google.com.evil.example",
            "https://user@colab.research.google.com",
            "https://colab.research.google.com/path",
            "https://colab.research.google.com?token=secret",
            "https://colab.research.google.com?",
            "https://colab.research.google.com#",
            "https://abc.prod.colab.dev:8443",
            "https://[broken",
            " https://colab.research.google.com",
        ):
            with self.subTest(origin=origin), self.assertRaisesRegex(
                runtime.RuntimeContractError,
                "COMFYCOLAB_CORS_ORIGIN",
            ):
                runtime.comfy_launch_command(
                    8188,
                    colab_proxy=True,
                    inherited_environment={"COMFYCOLAB_CORS_ORIGIN": origin},
                )

    def test_validate_colab_proxy_url_supports_current_and_legacy_hosts(self) -> None:
        current = (
            "https://8188-gpu-g4-s-kkb-use5c1-3m501x4q87ilk-c."
            "us-east5-1.prod.colab.dev/"
        )
        legacy = "https://abc-8188.colab.googleusercontent.com"
        self.assertEqual(runtime.validate_colab_proxy_url(current), current)
        self.assertEqual(runtime.validate_colab_proxy_url(legacy), legacy + "/")

        for value in (
            "https://colab.research.google.com/",
            "https://prod.colab.dev/",
            "https://googleusercontent.com/",
            "https://colab.googleusercontent.com/",
            "https://attacker.googleusercontent.com/",
            "https://abc.prod.colab.dev.evil.example/",
            "https://user@abc.prod.colab.dev/",
            "https://abc.prod.colab.dev/path",
        ):
            with self.subTest(value=value), self.assertRaisesRegex(
                runtime.RuntimeContractError,
                "COMFYCOLAB_PROXY_URL",
            ):
                runtime.validate_colab_proxy_url(value)

    def test_proxy_reuse_records_tunnel_failure_without_resetting_install(self) -> None:
        proxy_url = (
            "https://8188-gpu-g4-s-kkb-use5c1-3m501x4q87ilk-c."
            "us-east5-1.prod.colab.dev/"
        )
        proxy_origin = proxy_url.removesuffix("/")
        config = {
            "accepted_licenses": [],
            "port": 8188,
            "refresh": False,
            "colab_proxy": True,
            "lock_sha256": "a" * 64,
        }
        previous = {
            "schema": 1,
            "status": "ready",
            "lockSha256": "a" * 64,
            "comfyPid": 101,
            "tunnelPid": None,
            "comfyUrl": proxy_url,
            "cloudflareUrl": None,
            "colabProxyUrl": proxy_url,
            "corsOrigin": proxy_origin,
        }
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory) / "state"
            with (
                mock.patch.dict(
                    runtime.os.environ,
                    {
                        "COMFYCOLAB_PROXY_URL": proxy_url,
                        "COMFYCOLAB_CORS_ORIGIN": proxy_origin,
                    },
                    clear=True,
                ),
                mock.patch.object(runtime, "STATE_DIR", state_dir),
                mock.patch.object(
                    runtime,
                    "load_or_create_pip_baseline",
                    return_value=(),
                ),
                mock.patch.object(runtime, "load_state", return_value=previous),
                mock.patch.object(runtime, "pid_alive", return_value=True),
                mock.patch.object(runtime, "http_ready", return_value=True),
                mock.patch.object(
                    runtime,
                    "start_cloudflare_tunnel",
                    return_value=(None, None, "RuntimeError: tunnel failed"),
                ) as start_tunnel,
                mock.patch.object(runtime, "request_colab_proxy_url") as request_proxy,
                mock.patch.object(runtime, "stop_managed_process"),
                mock.patch.object(runtime, "save_state") as save_state,
                mock.patch.object(runtime, "emit_ready") as emit_ready,
                mock.patch.object(runtime, "reset_installation_roots") as reset,
            ):
                runtime.execute(config, self.core_lock())

        payload = save_state.call_args.args[0]
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["comfyUrl"], proxy_url)
        self.assertEqual(payload["colabProxyUrl"], proxy_url)
        self.assertIsNone(payload["cloudflareUrl"])
        self.assertIsNone(payload["tunnelPid"])
        self.assertEqual(payload["tunnelError"], "RuntimeError: tunnel failed")
        start_tunnel.assert_called_once_with(8188)
        request_proxy.assert_not_called()
        reset.assert_not_called()
        emit_ready.assert_called_once_with(payload)

    def test_non_proxy_reuse_without_tunnel_still_fails(self) -> None:
        config = {
            "accepted_licenses": [],
            "port": 8188,
            "refresh": False,
            "colab_proxy": False,
            "lock_sha256": "a" * 64,
        }
        previous = {
            "schema": 1,
            "status": "ready",
            "lockSha256": "a" * 64,
            "comfyPid": 101,
            "tunnelPid": None,
            "comfyUrl": None,
            "cloudflareUrl": None,
            "colabProxyUrl": None,
            "corsOrigin": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory) / "state"
            with (
                mock.patch.dict(runtime.os.environ, {}, clear=True),
                mock.patch.object(runtime, "STATE_DIR", state_dir),
                mock.patch.object(
                    runtime,
                    "load_or_create_pip_baseline",
                    return_value=(),
                ),
                mock.patch.object(runtime, "load_state", return_value=previous),
                mock.patch.object(runtime, "pid_alive", return_value=True),
                mock.patch.object(runtime, "http_ready", return_value=True),
                mock.patch.object(
                    runtime,
                    "start_cloudflare_tunnel",
                    return_value=(None, None, "RuntimeError: tunnel failed"),
                ),
                mock.patch.object(runtime, "stop_managed_process"),
                mock.patch.object(runtime, "save_state") as save_state,
                mock.patch.object(runtime, "emit_ready") as emit_ready,
                mock.patch.object(runtime, "reset_installation_roots") as reset,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Cloudflare fallback unavailable.*proxy mode is disabled",
                ):
                    runtime.execute(config, self.core_lock())

        save_state.assert_not_called()
        emit_ready.assert_not_called()
        reset.assert_not_called()

    def test_same_lock_cors_restart_reuses_installation_when_tunnel_fails(
        self,
    ) -> None:
        config = {
            "accepted_licenses": [],
            "port": 8188,
            "refresh": False,
            "colab_proxy": False,
            "lock_sha256": "a" * 64,
        }
        previous = {
            "schema": 1,
            "status": "ready",
            "lockSha256": "a" * 64,
            "comfyPid": 101,
            "tunnelPid": None,
            "comfyUrl": "https://old.prod.colab.dev/",
            "cloudflareUrl": None,
            "colabProxyUrl": "https://old.prod.colab.dev/",
            "corsOrigin": "https://old.prod.colab.dev",
        }
        comfy_process = mock.Mock(pid=303)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_dir = root / "state"
            comfy_dir = root / "ComfyUI"
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(runtime.os.environ, {}, clear=True))
                stack.enter_context(
                    mock.patch.multiple(
                        runtime,
                        STATE_DIR=state_dir,
                        STATE_FILE=state_dir / "runtime.json",
                        COMFY_DIR=comfy_dir,
                        COMFY_LOG=state_dir / "comfyui.log",
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        runtime,
                        "load_or_create_pip_baseline",
                        return_value=(),
                    )
                )
                stack.enter_context(
                    mock.patch.object(runtime, "load_state", return_value=previous)
                )
                stack.enter_context(
                    mock.patch.object(runtime, "http_ready", return_value=False)
                )
                stop_managed = stack.enter_context(
                    mock.patch.object(runtime, "stop_managed_process")
                )
                load_existing = stack.enter_context(
                    mock.patch.object(
                        runtime,
                        "load_existing_installation",
                        return_value=({}, {}, {"comfyui": str(comfy_dir)}),
                    )
                )
                stack.enter_context(
                    mock.patch.object(runtime, "validate_manifest_compatibility")
                )
                stack.enter_context(
                    mock.patch.object(runtime, "validate_minimax_h3_core_support")
                )
                stack.enter_context(
                    mock.patch.object(runtime, "apply_patches", return_value=[])
                )
                stack.enter_context(
                    mock.patch.object(runtime, "run_post_clone_probes")
                )
                stack.enter_context(mock.patch.object(runtime, "link_node_roots"))
                stack.enter_context(
                    mock.patch.object(
                        runtime,
                        "run_prestart_hooks",
                        return_value=({}, {}),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        runtime,
                        "resolved_runtime_environment",
                        return_value={},
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        runtime,
                        "pip_check_conflicts",
                        return_value=(),
                    )
                )
                stack.enter_context(
                    mock.patch.object(runtime, "reject_new_pip_conflicts")
                )
                popen = stack.enter_context(
                    mock.patch.object(
                        runtime.subprocess,
                        "Popen",
                        return_value=comfy_process,
                    )
                )
                stack.enter_context(mock.patch.object(runtime, "wait_for_comfy"))
                stack.enter_context(
                    mock.patch.object(
                        runtime,
                        "validate_post_start_nodes",
                        return_value={},
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        runtime,
                        "start_cloudflare_tunnel",
                        return_value=(None, None, "RuntimeError: tunnel failed"),
                    )
                )
                stop_started = stack.enter_context(
                    mock.patch.object(runtime, "stop_started_process")
                )
                clone_packs = stack.enter_context(
                    mock.patch.object(runtime, "clone_packs")
                )
                reset = stack.enter_context(
                    mock.patch.object(runtime, "reset_installation_roots")
                )
                save_state = stack.enter_context(
                    mock.patch.object(runtime, "save_state")
                )
                emit_ready = stack.enter_context(
                    mock.patch.object(runtime, "emit_ready")
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Cloudflare fallback unavailable.*proxy mode is disabled",
                ):
                    runtime.execute(config, self.core_lock())

        load_existing.assert_called_once_with(self.core_lock())
        stop_managed.assert_any_call(101)
        stop_started.assert_called_once_with(comfy_process)
        command = popen.call_args.args[0]
        self.assertEqual(
            command,
            [
                runtime.sys.executable,
                "main.py",
                "--listen",
                "127.0.0.1",
                "--port",
                "8188",
            ],
        )
        clone_packs.assert_not_called()
        reset.assert_not_called()
        save_state.assert_not_called()
        emit_ready.assert_not_called()

    def test_unsupported_installers_fail_local_runtime_preflight(self) -> None:
        with self.assertRaisesRegex(runtime.RuntimeContractError, "environment-TOML"):
            runtime.validate_runtime_support(
                {
                    "dependencies": [
                        {
                            "kind": "git",
                            "id": "three-d",
                            "requirements_file": "environment.toml",
                            "requirements_format": "comfycolab-environment-toml",
                        }
                    ],
                    "environments": [],
                }
            )
        with self.assertRaisesRegex(runtime.RuntimeContractError, "system manager"):
            runtime.validate_runtime_support(
                {
                    "dependencies": [],
                    "environments": [
                        {
                            "id": "worker",
                            "kind": "isolated",
                            "python_requirements": [],
                            "system_dependencies": [
                                {"manager": "pixi", "name": "cuda"}
                            ],
                        }
                    ],
                }
            )

    def test_legacy_full_requires_exact_node_bearing_daughter_set(self) -> None:
        packs = runtime.validate_legacy_full_lock(self.full_node_lock())
        self.assertEqual(
            [pack["id"] for pack in packs],
            ["3d", "3dgs", "image", "video"],
        )
        incomplete = self.full_node_lock()
        incomplete["packs"] = list(incomplete["packs"])[:-1]
        with self.assertRaisesRegex(
            runtime.RuntimeContractError,
            "requires exactly the node-bearing daughter packs",
        ):
            runtime.validate_legacy_full_lock(incomplete)

    def test_legacy_full_dispatches_before_generic_runtime_preflight(self) -> None:
        config = {
            "runtime_mode": "legacy-full",
            "accepted_licenses": [],
            "port": 8188,
            "refresh": False,
            "colab_proxy": True,
            "lock_sha256": "a" * 64,
        }
        lock = self.full_node_lock()
        with (
            mock.patch.object(runtime, "execute_legacy_full") as legacy,
            mock.patch.object(runtime, "validate_runtime_support") as generic,
        ):
            runtime.execute(config, lock)
        legacy.assert_called_once_with(config, lock)
        generic.assert_not_called()

    def test_pack_license_gate_is_enforced_before_runtime_mutation(self) -> None:
        lock = {
            "packs": [
                {
                    "id": "research",
                    "license_gate": "accept_research_terms",
                }
            ]
        }
        with self.assertRaisesRegex(runtime.RuntimeContractError, "accept-license"):
            runtime.validate_pack_license_gates(
                lock,
                accepted_licenses=set(),
            )
        runtime.validate_pack_license_gates(
            lock,
            accepted_licenses={"accept_research_terms"},
        )

    def test_pip_check_records_preexisting_colab_conflicts(self) -> None:
        result = subprocess.CompletedProcess(
            args=["python", "-m", "pip", "check"],
            returncode=1,
            stdout="ipython 7.34.0 requires jedi, which is not installed.\n",
            stderr="",
        )
        with mock.patch.object(runtime, "run", return_value=result) as run:
            conflicts = runtime.pip_check_conflicts()

        self.assertEqual(
            conflicts,
            ("ipython 7.34.0 requires jedi, which is not installed.",),
        )
        run.assert_called_once_with(
            [runtime.sys.executable, "-m", "pip", "check"],
            capture_output=True,
            check=False,
        )

    def test_unchanged_preexisting_pip_conflict_is_not_fatal(self) -> None:
        baseline = ("ipython 7.34.0 requires jedi, which is not installed.",)
        runtime.reject_new_pip_conflicts(baseline, baseline)

    def test_new_pip_conflict_is_fatal(self) -> None:
        baseline = ("ipython 7.34.0 requires jedi, which is not installed.",)
        current = baseline + (
            "example 2.0 has requirement dependency<2, but you have dependency 3.0.",
        )
        with self.assertRaisesRegex(
            runtime.RuntimeContractError,
            "(?s)introduced Python package conflicts.*example 2.0",
        ):
            runtime.reject_new_pip_conflicts(baseline, current)

    def test_pip_check_execution_error_is_fatal(self) -> None:
        result = subprocess.CompletedProcess(
            args=["python", "-m", "pip", "check"],
            returncode=2,
            stdout="",
            stderr="pip check failed internally",
        )
        with (
            mock.patch.object(runtime, "run", return_value=result),
            self.assertRaisesRegex(
                runtime.RuntimeContractError,
                "pip check could not complete.*failed internally",
            ),
        ):
            runtime.pip_check_conflicts()

    def test_original_pip_baseline_is_reused_across_failed_attempts(self) -> None:
        original = ("ipython 7.34.0 requires jedi, which is not installed.",)
        introduced = original + (
            "example 2.0 has requirement dependency<2, but you have dependency 3.0.",
        )
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "pip-baseline.json"
            with (
                mock.patch.object(runtime, "PIP_BASELINE_FILE", baseline_path),
                mock.patch.object(
                    runtime,
                    "pip_check_conflicts",
                    return_value=original,
                ) as pip_check,
            ):
                first = runtime.load_or_create_pip_baseline()
            self.assertEqual(first, original)
            pip_check.assert_called_once_with()

            with (
                mock.patch.object(runtime, "PIP_BASELINE_FILE", baseline_path),
                mock.patch.object(
                    runtime,
                    "pip_check_conflicts",
                    side_effect=AssertionError("must not re-baseline a dirty retry"),
                ),
            ):
                retry = runtime.load_or_create_pip_baseline()

            self.assertEqual(retry, original)
            with self.assertRaisesRegex(
                runtime.RuntimeContractError,
                "introduced Python package conflicts",
            ):
                runtime.reject_new_pip_conflicts(retry, introduced)

    def test_corrupt_pip_baseline_requires_a_clean_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = Path(directory) / "pip-baseline.json"
            baseline_path.write_text('{"schema":1}', encoding="utf-8")
            with (
                mock.patch.object(runtime, "PIP_BASELINE_FILE", baseline_path),
                self.assertRaisesRegex(
                    runtime.RuntimeContractError,
                    "pip baseline is invalid.*Restart the Colab runtime",
                ),
            ):
                runtime.load_or_create_pip_baseline()

    def test_dependency_destination_rejects_escape(self) -> None:
        with self.assertRaisesRegex(runtime.RuntimeContractError, "unsafe"):
            runtime.dependency_destination(
                {
                    "id": "escape",
                    "destination": "../outside",
                    "scope": "comfyui",
                }
            )

    def test_runtime_pack_ids_match_the_public_schema(self) -> None:
        self.assertEqual(runtime.safe_pack_id("image_v2"), "image_v2")
        self.assertEqual(runtime.safe_pack_id("world.models"), "world.models")
        for value in ("Image", "image--v2", ".image", "image/escape"):
            with self.subTest(value=value):
                with self.assertRaises(runtime.RuntimeContractError):
                    runtime.safe_pack_id(value)

    def test_link_node_roots_preserves_declared_legacy_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comfy = root / "ComfyUI"
            pack = root / "pack"
            source = pack / "custom_nodes" / "ComfyColab-ZImage"
            source.mkdir(parents=True)
            with mock.patch.object(runtime, "COMFY_DIR", comfy):
                runtime.link_node_roots(
                    {
                        "image": {
                            "node_roots": [
                                {
                                    "source": "custom_nodes/ComfyColab-ZImage",
                                    "target": "ComfyColab-ZImage",
                                }
                            ]
                        }
                    },
                    pack_roots={"image": pack},
                )
            target = comfy / "custom_nodes" / "ComfyColab-ZImage"
            self.assertTrue(target.is_symlink())
            self.assertEqual(target.resolve(), source.resolve())

    def test_link_node_roots_links_all_full_node_daughters(self) -> None:
        targets = {
            "3d": "ComfyColab-3D",
            "3dgs": "ComfyColab-Triposplat",
            "image": "ComfyColab-ZImage",
            "video": "ComfyColab-LTXVideo",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comfy = root / "ComfyUI"
            pack_roots: dict[str, Path] = {}
            manifests: dict[str, dict[str, object]] = {}
            for pack_id, target_name in targets.items():
                pack_root = root / "packs" / pack_id
                source = pack_root / "custom_nodes" / target_name
                source.mkdir(parents=True)
                pack_roots[pack_id] = pack_root
                manifests[pack_id] = {
                    "node_roots": [
                        {
                            "source": f"custom_nodes/{target_name}",
                            "target": target_name,
                        }
                    ]
                }
            with mock.patch.object(runtime, "COMFY_DIR", comfy):
                runtime.link_node_roots(manifests, pack_roots=pack_roots)
            for pack_id, target_name in targets.items():
                target = comfy / "custom_nodes" / target_name
                self.assertTrue(target.is_symlink())
                self.assertEqual(
                    target.resolve(),
                    (pack_roots[pack_id] / "custom_nodes" / target_name).resolve(),
                )

    def test_duplicate_node_targets_are_rejected_before_second_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comfy = root / "ComfyUI"
            first = root / "first"
            second = root / "second"
            for pack in (first, second):
                (pack / "node").mkdir(parents=True)
            manifests = {
                "one": {"node_roots": [{"source": "node", "target": "same"}]},
                "two": {"node_roots": [{"source": "node", "target": "same"}]},
            }
            with mock.patch.object(runtime, "COMFY_DIR", comfy):
                with self.assertRaisesRegex(runtime.RuntimeContractError, "duplicate"):
                    runtime.link_node_roots(
                        manifests,
                        pack_roots={"one": first, "two": second},
                    )

    def test_license_gated_environment_requires_explicit_acceptance(self) -> None:
        lock = {
            "environments": [
                {
                    "id": "research",
                    "kind": "isolated",
                    "scope": "worker",
                    "python_requirements": [],
                    "system_dependencies": [],
                    "license_gate": "accept_research_license",
                }
            ]
        }
        with self.assertRaisesRegex(runtime.RuntimeContractError, "accept-license"):
            runtime.install_environments(lock, accepted_licenses=set())

    def test_duplicate_environment_ids_fail_before_runtime_mutation(self) -> None:
        lock = {
            "environments": [
                {
                    "id": "worker",
                    "kind": "isolated",
                    "scope": "worker",
                    "python_requirements": [],
                    "system_dependencies": [],
                    "owner": owner,
                }
                for owner in ("image", "video")
            ]
        }
        with mock.patch.object(runtime, "run") as run:
            with self.assertRaisesRegex(runtime.RuntimeContractError, "duplicate"):
                runtime.install_environments(lock, accepted_licenses=set())
        run.assert_not_called()

    def test_pack_requirements_with_multiple_owners_fail_before_install(self) -> None:
        dependency = {
            "kind": "git",
            "id": "shared",
            "repository": "https://github.com/example/shared.git",
            "ref": "a" * 40,
            "destination": "custom_nodes/shared",
            "scope": "comfyui",
            "requirements_file": "requirements.txt",
            "requirements_source": "pack",
            "requested_by": ["image", "video"],
        }
        with mock.patch.object(runtime, "install_dependency") as install:
            with self.assertRaisesRegex(runtime.RuntimeContractError, "exactly one owner"):
                runtime.install_dependencies(
                    {"dependencies": [dependency]},
                    pack_roots={
                        "image": Path("/tmp/image"),
                        "video": Path("/tmp/video"),
                    },
                    accepted_licenses=set(),
                )
        install.assert_not_called()

    def test_reset_installation_roots_removes_previous_lock_owned_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            managed = {
                "COMFY_DIR": root / "ComfyUI",
                "PACKS_DIR": root / "state" / "packs",
                "DEPENDENCIES_DIR": root / "state" / "dependencies",
                "ENVIRONMENTS_DIR": root / "state" / "environments",
                "PACK_STATE_DIR": root / "state" / "pack-state",
            }
            for path in managed.values():
                path.mkdir(parents=True)
                (path / "stale").write_text("old lock", encoding="utf-8")
            with mock.patch.multiple(runtime, **managed):
                runtime.reset_installation_roots()
            self.assertTrue(all(not path.exists() for path in managed.values()))

    def test_resolved_runtime_environment_is_generic(self) -> None:
        environment = runtime.resolved_runtime_environment(
            {
                "runtime_env": [
                    {
                        "name": "COMFYCOLAB_WORKER",
                        "value": "/content/worker",
                        "requested_by": ["example"],
                    }
                ]
            }
        )
        self.assertEqual(environment, {"COMFYCOLAB_WORKER": "/content/worker"})

    def test_lazy_dependency_records_path_without_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(runtime, "DEPENDENCIES_DIR", root / "dependencies"):
                resolved: dict[str, str] = {}
                runtime.install_dependency(
                    {
                        "id": "model",
                        "kind": "huggingface",
                        "repository": "example/model",
                        "ref": "a" * 40,
                        "destination": "models/example",
                        "scope": "isolated",
                        "install_phase": "lazy",
                    },
                    resolved_paths=resolved,
                    pack_roots={},
                    accepted_licenses=set(),
                )
            self.assertEqual(
                resolved["model"],
                str(root / "dependencies" / "models" / "example"),
            )
            self.assertFalse((root / "dependencies" / "models" / "example").exists())

    def test_eager_huggingface_dependency_verifies_declared_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dependencies = root / "dependencies"
            payload = b"pinned model bytes"
            expected = hashlib.sha256(payload).hexdigest()

            calls: list[dict[str, object]] = []

            def fake_snapshot_download(**kwargs: object) -> str:
                calls.append(kwargs)
                destination = Path(str(kwargs["local_dir"]))
                (destination / "weights").mkdir(parents=True, exist_ok=True)
                (destination / "weights" / "model.bin").write_bytes(payload)
                return str(destination)

            hub = types.ModuleType("huggingface_hub")
            hub.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]

            with (
                mock.patch.object(runtime, "DEPENDENCIES_DIR", dependencies),
                mock.patch.dict(sys.modules, {"huggingface_hub": hub}),
                mock.patch.dict(os.environ, {"HF_TOKEN": "test-token"}, clear=False),
            ):
                resolved: dict[str, str] = {}
                runtime.install_dependency(
                    {
                        "id": "model",
                        "kind": "huggingface",
                        "repository": "example/model",
                        "ref": "a" * 40,
                        "destination": "models/example",
                        "scope": "isolated",
                        "artifacts": [
                            {
                                "path": "weights/model.bin",
                                "bytes": len(payload),
                                "sha256": expected,
                            }
                        ],
                    },
                    resolved_paths=resolved,
                    pack_roots={},
                    accepted_licenses=set(),
                )
                self.assertEqual(os.environ["HF_XET_HIGH_PERFORMANCE"], "1")

            self.assertEqual(
                resolved["model"],
                str(dependencies / "models" / "example"),
            )
            self.assertEqual(
                calls,
                [
                    {
                        "repo_id": "example/model",
                        "revision": "a" * 40,
                        "local_dir": str(dependencies / "models" / "example"),
                        "token": "test-token",
                    }
                ],
            )

    def test_huggingface_dependency_retries_stale_token_anonymously(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dependencies = Path(directory) / "dependencies"
            calls: list[object] = []

            def fake_snapshot_download(**kwargs: object) -> str:
                calls.append(kwargs["token"])
                if kwargs["token"]:
                    raise RuntimeError("stale token")
                return str(kwargs["local_dir"])

            hub = types.ModuleType("huggingface_hub")
            hub.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
            with (
                mock.patch.object(runtime, "DEPENDENCIES_DIR", dependencies),
                mock.patch.dict(sys.modules, {"huggingface_hub": hub}),
                mock.patch.dict(
                    os.environ,
                    {"HF_TOKEN": "stale-token"},
                    clear=False,
                ),
            ):
                runtime.install_dependency(
                    {
                        "id": "model",
                        "kind": "huggingface",
                        "repository": "example/model",
                        "ref": "a" * 40,
                        "destination": "models/example",
                        "scope": "isolated",
                    },
                    resolved_paths={},
                    pack_roots={},
                    accepted_licenses=set(),
                )

            self.assertEqual(calls, ["stale-token", False])

    def test_core_requirements_install_hf_xet_transport(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            comfy_dir = Path(directory)
            requirements = comfy_dir / "requirements.txt"
            requirements.write_text("torch\n", encoding="utf-8")
            with (
                mock.patch.object(runtime, "COMFY_DIR", comfy_dir),
                mock.patch.object(runtime, "run") as run,
                mock.patch.object(
                    runtime, "install_minimax_h3_cuda_runtime"
                ) as install_cuda,
                mock.patch.object(
                    runtime, "validate_minimax_h3_cuda_runtime"
                ) as validate_cuda,
            ):
                runtime.install_core_requirements()
            self.assertEqual(
                run.call_args_list,
                [
                    mock.call(
                        [
                            runtime.sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "-r",
                            str(requirements),
                        ]
                    ),
                    mock.call(
                        [
                            runtime.sys.executable,
                            "-m",
                            "pip",
                            "install",
                            runtime.HUGGINGFACE_HUB_REQUIREMENT,
                        ]
                    ),
                    mock.call(
                        [
                            runtime.sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "--no-build-isolation",
                            runtime.SAGE_ATTENTION_REQUIREMENT,
                        ],
                        env=mock.ANY,
                    ),
                ],
            )
            sage_environment = run.call_args_list[2].kwargs["env"]
            self.assertEqual(
                {key: sage_environment[key] for key in runtime.SAGE_ATTENTION_BUILD_ENV},
                runtime.SAGE_ATTENTION_BUILD_ENV,
            )
            self.assertEqual(
                sage_environment["CUDA_HOME"],
                str(runtime.MINIMAX_H3_CUDA_HOME),
            )
            self.assertTrue(
                runtime.SAGE_ATTENTION_REQUIREMENT.endswith(
                    f"@{runtime.SAGE_ATTENTION_SOURCE_REF}"
                )
            )
            install_cuda.assert_called_once_with()
            validate_cuda.assert_called_once_with()

    def test_minimax_h3_cuda_runtime_installs_pinned_cuda13_stack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cuda_home = Path(directory) / "cuda-13.0"
            calls: list[list[str]] = []

            def fake_run(command, **_kwargs):
                calls.append(command)
                if command[:3] == ["apt-get", "install", "-y"]:
                    (cuda_home / "bin").mkdir(parents=True)
                    (cuda_home / "bin" / "nvcc").touch()

            with (
                mock.patch.object(runtime, "MINIMAX_H3_CUDA_HOME", cuda_home),
                mock.patch.object(runtime, "run", side_effect=fake_run),
            ):
                runtime.install_minimax_h3_cuda_runtime()

            self.assertEqual(calls[0], ["apt-get", "update", "-qq"])
            self.assertEqual(
                calls[1],
                [
                    "apt-get",
                    "install",
                    "-y",
                    "-qq",
                    runtime.MINIMAX_H3_CUDA_COMPILER_PACKAGE,
                ],
            )
            self.assertEqual(
                calls[2],
                [
                    runtime.sys.executable,
                    "-m",
                    "pip",
                    "install",
                    *runtime.MINIMAX_H3_TORCH_REQUIREMENTS,
                    "--index-url",
                    runtime.MINIMAX_H3_TORCH_INDEX_URL,
                ],
            )
            self.assertTrue(
                all(
                    requirement.endswith("+cu130")
                    for requirement in runtime.MINIMAX_H3_TORCH_REQUIREMENTS
                )
            )

    def test_huggingface_artifact_digest_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            (destination / "model.bin").write_bytes(b"actual")
            with self.assertRaisesRegex(
                runtime.RuntimeContractError,
                "digest mismatch",
            ):
                runtime.verify_huggingface_artifacts(
                    "model",
                    destination,
                    [
                        {
                            "path": "model.bin",
                            "bytes": len(b"actual"),
                            "sha256": hashlib.sha256(b"expected").hexdigest(),
                        }
                    ],
                )

    def test_hook_sandbox_blocks_undeclared_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "pack"
            pack.mkdir()
            hook = pack / "hook.py"
            hook.write_text(
                "from pathlib import Path\n"
                "Path('forbidden.txt').write_text('no')\n"
                "print('{\"status\":\"ok\"}')\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(runtime, "PACK_STATE_DIR", root / "pack-state"),
                self.assertRaisesRegex(runtime.RuntimeContractError, "filesystem write blocked"),
            ):
                runtime.run_hook(
                    "example",
                    pack,
                    "configure",
                    {
                        "path": "hook.py",
                        "network": "none",
                        "write_roots": [],
                        "timeout_seconds": 30,
                    },
                    {"schema": 1},
                )
            self.assertFalse((pack / "forbidden.txt").exists())

    def test_hook_sandbox_allows_declared_pack_state_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "pack"
            pack.mkdir()
            hook = pack / "hook.py"
            hook.write_text(
                "import json, sys\n"
                "from pathlib import Path\n"
                "context = json.load(sys.stdin)\n"
                "target = Path(context['write_roots']['pack_state']) / 'configured.json'\n"
                "target.write_text('{}')\n"
                "print(json.dumps({'status': 'ok', 'writes': "
                "[{'root': 'pack_state', 'path': 'configured.json'}]}))\n",
                encoding="utf-8",
            )
            pack_state = root / "pack-state"
            with mock.patch.object(runtime, "PACK_STATE_DIR", pack_state):
                result = runtime.run_hook(
                    "example",
                    pack,
                    "configure",
                    {
                        "path": "hook.py",
                        "network": "none",
                        "write_roots": ["pack_state"],
                        "timeout_seconds": 30,
                    },
                    {"schema": 1},
                )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(
                (pack_state / "example" / "configured.json").read_text(
                    encoding="utf-8"
                ),
                "{}",
            )

    def test_post_clone_file_symbols_can_target_comfyui(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comfy = root / "ComfyUI"
            target = comfy / "comfy_extras" / "nodes_example.py"
            target.parent.mkdir(parents=True)
            target.write_text("class RequiredNode:\n    pass\n", encoding="utf-8")
            with mock.patch.object(runtime, "COMFY_DIR", comfy):
                runtime.run_post_clone_probes(
                    {
                        "example": {
                            "probes": [
                                {
                                    "phase": "post_clone",
                                    "type": "file_symbols",
                                    "target": "comfyui",
                                    "path": "comfy_extras/nodes_example.py",
                                    "symbols": ["RequiredNode"],
                                }
                            ]
                        }
                    },
                    pack_roots={"example": root / "pack"},
                    resolved_paths={},
                )

    def test_minimax_h3_core_support_requires_native_nodes_and_clip_type(self) -> None:
        self.assertIn(
            "comfy_extras/nodes_audio.py",
            runtime.MINIMAX_H3_CORE_REQUIREMENTS,
        )
        self.assertNotIn("nodes.py", runtime.MINIMAX_H3_CORE_REQUIREMENTS)
        with tempfile.TemporaryDirectory() as directory:
            comfy = Path(directory)
            for (
                relative_path,
                symbols,
            ) in runtime.MINIMAX_H3_CORE_REQUIREMENTS.items():
                path = comfy / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n".join(symbols), encoding="utf-8")
            with mock.patch.object(runtime, "COMFY_DIR", comfy):
                runtime.validate_minimax_h3_core_support()
                missing_path = comfy / "comfy_extras" / "nodes_minimax_h3.py"
                missing_path.write_text("MiniMaxH3ImageToVideo\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    runtime.RuntimeContractError,
                    "native MiniMax H3",
                ):
                    runtime.validate_minimax_h3_core_support()

    def test_post_start_probe_and_health_command_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = root / "pack"
            (pack / "runtime").mkdir(parents=True)
            (pack / "runtime" / "doctor.py").write_text(
                "import json\n"
                "print(json.dumps({'status': 'ok', 'checked': True}))\n",
                encoding="utf-8",
            )
            manifests = {
                "image": {
                    "probes": [
                        {
                            "phase": "post_start",
                            "type": "comfy_node_ids",
                            "values": ["ExampleNode"],
                        }
                    ],
                    "health_checks": {
                        "node_ids": ["ExampleNode"],
                        "command": ["python", "runtime/doctor.py"],
                    },
                }
            }
            with (
                mock.patch.object(runtime, "COMFY_DIR", root / "ComfyUI"),
                mock.patch.object(runtime, "PACK_STATE_DIR", root / "pack-state"),
                mock.patch.object(
                    runtime,
                    "object_info",
                    return_value={"ExampleNode": {}},
                ),
            ):
                result = runtime.validate_post_start_nodes(
                    8188,
                    manifests,
                    lock={"schema": 1},
                    lock_sha256="a" * 64,
                    pack_roots={"image": pack},
                    resolved_paths={"comfyui": str(root / "ComfyUI")},
                )
            self.assertEqual(result["image"]["nodeIds"], ["ExampleNode"])
            self.assertTrue(result["image"]["command"]["checked"])

    def test_cloudflared_download_is_versioned_and_digest_verified(self) -> None:
        payload = b"authenticated cloudflared"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            response = io.BytesIO(payload)
            with (
                mock.patch.object(runtime, "STATE_DIR", state),
                mock.patch.object(runtime.platform, "machine", return_value="x86_64"),
                mock.patch.dict(
                    runtime.CLOUDFLARED_ASSETS,
                    {"amd64": ("cloudflared-linux-amd64", digest)},
                    clear=False,
                ),
                mock.patch.object(
                    runtime.urllib.request,
                    "urlopen",
                    return_value=response,
                ) as download,
            ):
                path = runtime.cloudflared_path()
            self.assertEqual(path.read_bytes(), payload)
            self.assertIn(
                f"/download/{runtime.CLOUDFLARED_VERSION}/",
                download.call_args.args[0],
            )
            self.assertNotIn("/latest/", download.call_args.args[0])

    def test_manifest_compatibility_fails_before_engine_install(self) -> None:
        with self.assertRaisesRegex(runtime.RuntimeContractError, "incompatible"):
            runtime.validate_manifest_compatibility(
                {
                    "image": {
                        "compatibility": {
                            "comfyui": {"compatible_refs": ["a" * 40]}
                        }
                    }
                },
                comfyui_commit="b" * 40,
            )


if __name__ == "__main__":
    unittest.main()
