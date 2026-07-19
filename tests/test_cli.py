from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from comfycolab import cli
from comfycolab.packs.lock import ComfyColabLockV1


LOCK_SHA256 = "d" * 64


def arguments(state: Path):
    return types.SimpleNamespace(
        repo_url="https://example.test/ComfyColab.git",
        repo_ref="main",
        session="comfycolab",
        gpu="G4",
        port=8188,
        bootstrap_timeout=1800,
        refresh=False,
        colab_proxy=False,
        state=str(state),
        auth="adc",
        config="/tmp/sessions.json",
        colab_bin="colab",
        pack=[],
        profile="core",
        pack_ref=[],
        accept_license=[],
        lock_dir=str(state.parent / "locks"),
    )


class FakeClient:
    def __init__(self, *, bootstrap_error: Exception | None = None):
        self.bootstrap_error = bootstrap_error
        self.stopped: list[str] = []
        self.opened: list[str] = []

    def session_exists(self, session):
        return False

    def new(self, session, gpu):
        return subprocess.CompletedProcess([], 0, "[colab] Session READY.\n", "")

    def exec_bootstrap(self, **kwargs):
        if self.bootstrap_error:
            raise self.bootstrap_error
        return subprocess.CompletedProcess(
            [],
            0,
            (
                'COMFYCOLAB_READY={"comfyUrl":"https://demo.trycloudflare.com",'
                f'"lockSha256":"{LOCK_SHA256}"}}\n'
            ),
            "",
        )

    def open_url(self, session):
        self.opened.append(session)
        return subprocess.CompletedProcess([], 0, "https://colab.example/notebook\n", "")

    def stop(self, session):
        self.stopped.append(session)


class FakeStatusClient:
    def __init__(self, output: str, returncode: int = 0):
        self.output = output
        self.returncode = returncode

    def status(self, session):
        return subprocess.CompletedProcess([], self.returncode, self.output, "")


class CliLifecycleTests(unittest.TestCase):
    @staticmethod
    def prepared():
        lock = mock.Mock()
        lock.sha256 = LOCK_SHA256
        lock.canonical_bytes.return_value = b'{"packs":[],"schema":1}'
        lock.to_dict.return_value = {"dependencies": [], "environments": []}
        return types.SimpleNamespace(source="bootstrap", lock=lock)

    @staticmethod
    def real_lock(comfyui_commit: str) -> ComfyColabLockV1:
        return ComfyColabLockV1.from_dict(
            {
                "schema": 1,
                "core": {
                    "version": "0.2.0.dev1",
                    "repository": "https://github.com/example/ComfyColab.git",
                    "commit": "a" * 40,
                },
                "comfyui": {
                    "repository": "https://github.com/Comfy-Org/ComfyUI.git",
                    "commit": comfyui_commit,
                },
                "packs": [],
                "dependencies": [],
                "patches": [],
                "environments": [],
                "runtime_env": [],
            }
        )

    def test_start_defaults_to_g4(self) -> None:
        args = cli.build_parser().parse_args(["start"])
        self.assertEqual(args.gpu, "G4")
        self.assertFalse(args.colab_proxy)
        self.assertEqual(args.profile, "core")
        self.assertEqual(args.pack, [])

    def test_repository_defaults_are_canonical_and_overridable(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                cli.resolve_repository_url(None),
                "https://github.com/DragonLord1998/ComfyColab.git",
            )
        with mock.patch.dict(
            os.environ,
            {"COMFYCOLAB_REPO_URL": "https://github.com/example/fork.git"},
            clear=True,
        ):
            self.assertEqual(
                cli.resolve_repository_url(None),
                "https://github.com/example/fork.git",
            )
        self.assertEqual(
            cli.resolve_repository_url("https://github.com/example/explicit.git"),
            "https://github.com/example/explicit.git",
        )

    def test_notebook_defaults_to_reproducible_core_profile(self) -> None:
        args = cli.build_parser().parse_args(["notebook"])
        self.assertEqual(args.profile, "core")
        self.assertTrue(args.colab_proxy)
        self.assertEqual(args.output, "ComfyColab.ipynb")
        self.assertFalse(args.legacy_full)

    def test_notebook_legacy_full_mode_is_explicit(self) -> None:
        args = cli.build_parser().parse_args(
            ["notebook", "--profile", "legacy-full", "--legacy-full"]
        )
        self.assertEqual(args.profile, "legacy-full")
        self.assertTrue(args.legacy_full)

    def test_notebook_runtime_resolve_main_mode_is_explicit(self) -> None:
        args = cli.build_parser().parse_args(
            [
                "notebook",
                "--runtime-resolve-main",
                "--profile",
                "legacy-full",
                "--legacy-full",
                "--accept-license",
                "accept_research_license",
            ]
        )
        self.assertTrue(args.runtime_resolve_main)
        self.assertEqual(args.repo_ref, "main")
        self.assertEqual(args.profile, "legacy-full")
        self.assertTrue(args.legacy_full)
        self.assertTrue(args.colab_proxy)
        self.assertEqual(args.accept_license, ["accept_research_license"])

    def test_runtime_resolve_main_notebook_writes_without_local_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ComfyColab.ipynb"
            args = cli.build_parser().parse_args(
                [
                    "notebook",
                    "--runtime-resolve-main",
                    "--profile",
                    "legacy-full",
                    "--legacy-full",
                    "--accept-license",
                    "accept_research_license",
                    "--lock-dir",
                    str(Path(directory) / "locks"),
                    "--output",
                    str(output),
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                result = cli._render_notebook(args)

            self.assertEqual(result, 0)
            self.assertTrue(output.is_file())
            self.assertFalse(cli._lock_path(args).exists())
            rendered = output.read_text(encoding="utf-8")
            self.assertIn('"runtime-resolved-comfycolab-bootstrap"', rendered)
            self.assertIn("resolved inside Colab", stdout.getvalue())

    def test_runtime_resolve_main_rejects_local_pack_ref_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pack_ref = Path(directory) / "pack-ref.json"
            pack_ref.write_text("{}", encoding="utf-8")
            args = cli.build_parser().parse_args(
                [
                    "notebook",
                    "--runtime-resolve-main",
                    "--pack-ref",
                    str(pack_ref),
                ]
            )
            with self.assertRaisesRegex(ValueError, "local --pack-ref"):
                cli._render_notebook(args)

    def test_colab_proxy_opens_attached_page_and_embeds_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "runtime.json"
            args = arguments(state)
            args.colab_proxy = True
            client = FakeClient()
            with mock.patch.object(cli, "_client", return_value=client), mock.patch.object(
                cli, "_prepare_launch", return_value=self.prepared()
            ) as prepare, contextlib.redirect_stdout(io.StringIO()):
                result = cli._start(args)

            self.assertEqual(result, 0)
            self.assertEqual(client.opened, ["comfycolab"])
            self.assertTrue(prepare.call_args.args[0].colab_proxy)

    def test_status_transport_error_preserves_saved_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "runtime.json"
            original = '{"session":"comfycolab","comfyUrl":"https://saved.trycloudflare.com"}\n'
            state.write_text(original, encoding="utf-8")
            args = arguments(state)
            client = FakeStatusClient("", returncode=2)
            with mock.patch.object(cli, "_client", return_value=client), contextlib.redirect_stdout(
                io.StringIO()
            ):
                result = cli._status(args)
            self.assertEqual(result, 2)
            self.assertEqual(state.read_text(encoding="utf-8"), original)

    def test_missing_remote_session_removes_stale_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "runtime.json"
            state.write_text(
                '{"session":"comfycolab","comfyUrl":"https://stale.trycloudflare.com"}\n',
                encoding="utf-8",
            )
            args = arguments(state)
            client = FakeStatusClient("[colab] Session 'comfycolab' not found.\n")
            with mock.patch.object(cli, "_client", return_value=client), contextlib.redirect_stdout(
                io.StringIO()
            ):
                result = cli._status(args)
            self.assertEqual(result, 1)
            self.assertFalse(state.exists())

    def test_start_persists_ready_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "runtime.json"
            client = FakeClient()
            with mock.patch.object(cli, "_client", return_value=client), mock.patch.object(
                cli, "_prepare_launch", return_value=self.prepared()
            ), contextlib.redirect_stdout(io.StringIO()):
                result = cli._start(arguments(state))

            self.assertEqual(result, 0)
            payload = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(payload["comfyUrl"], "https://demo.trycloudflare.com")
            self.assertEqual(payload["session"], "comfycolab")
            self.assertEqual(client.stopped, [])

    def test_failed_bootstrap_stops_newly_created_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient(bootstrap_error=RuntimeError("bootstrap failed"))
            with mock.patch.object(cli, "_client", return_value=client), mock.patch.object(
                cli, "_prepare_launch", return_value=self.prepared()
            ), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "bootstrap failed"):
                    cli._start(arguments(Path(directory) / "runtime.json"))
            self.assertEqual(client.stopped, ["comfycolab"])

    def test_unsupported_lock_fails_before_colab_client_or_lock_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "runtime.json"
            args = arguments(state)
            prepared = self.prepared()
            prepared.lock.to_dict.return_value = {
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
            with mock.patch.object(
                cli, "_prepare_launch", return_value=prepared
            ), mock.patch.object(cli, "_client") as client:
                with self.assertRaisesRegex(RuntimeError, "environment-TOML"):
                    cli._start(args)
            client.assert_not_called()
            self.assertFalse(cli._lock_path(args).exists())

    def test_unaccepted_pack_gate_fails_before_colab_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(Path(directory) / "runtime.json")
            prepared = self.prepared()
            prepared.lock.to_dict.return_value = {
                "dependencies": [],
                "environments": [],
                "packs": [
                    {
                        "id": "research",
                        "license_gate": "accept_research_terms",
                    }
                ],
            }
            with mock.patch.object(
                cli, "_prepare_launch", return_value=prepared
            ), mock.patch.object(cli, "_client") as client:
                with self.assertRaisesRegex(RuntimeError, "accept-license"):
                    cli._start(args)
            client.assert_not_called()

    def test_lock_updates_preserve_and_restore_one_previous_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_dir = Path(directory)
            args = types.SimpleNamespace(
                session="comfycolab",
                lock_dir=str(lock_dir),
            )
            path = cli._lock_path(args)
            first = self.real_lock("b" * 40)
            second = self.real_lock("c" * 40)
            cli._write_lock(path, first)
            cli._write_lock(path, second)
            previous_path = cli._previous_lock_path(path)
            self.assertEqual(
                ComfyColabLockV1.from_bytes(previous_path.read_bytes()).sha256,
                first.sha256,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli._pack_rollback(args), 0)
            self.assertEqual(
                ComfyColabLockV1.from_bytes(path.read_bytes()).sha256,
                first.sha256,
            )
            self.assertEqual(
                ComfyColabLockV1.from_bytes(previous_path.read_bytes()).sha256,
                second.sha256,
            )


if __name__ == "__main__":
    unittest.main()
