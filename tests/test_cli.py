from __future__ import annotations

import contextlib
import io
import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from comfycolab import cli


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
            'COMFYCOLAB_READY={"comfyUrl":"https://demo.trycloudflare.com"}\n',
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
    def test_start_defaults_to_g4(self) -> None:
        args = cli.build_parser().parse_args(["start"])
        self.assertEqual(args.gpu, "G4")
        self.assertFalse(args.colab_proxy)

    def test_colab_proxy_opens_attached_page_and_embeds_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "runtime.json"
            args = arguments(state)
            args.colab_proxy = True
            client = FakeClient()
            with mock.patch.object(cli, "_client", return_value=client), mock.patch.object(
                cli, "render_bootstrap", return_value="bootstrap"
            ) as render, contextlib.redirect_stdout(io.StringIO()):
                result = cli._start(args)

            self.assertEqual(result, 0)
            self.assertEqual(client.opened, ["comfycolab"])
            self.assertTrue(render.call_args.kwargs["colab_proxy"])

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
                cli, "render_bootstrap", return_value="bootstrap"
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
                cli, "render_bootstrap", return_value="bootstrap"
            ), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "bootstrap failed"):
                    cli._start(arguments(Path(directory) / "runtime.json"))
            self.assertEqual(client.stopped, ["comfycolab"])


if __name__ == "__main__":
    unittest.main()
