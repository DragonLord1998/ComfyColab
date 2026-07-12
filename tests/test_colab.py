from __future__ import annotations

import subprocess
import tempfile
import unittest
import contextlib
import io
from pathlib import Path

from comfycolab.colab import ColabClient, parse_ready_payload


class ColabClientTests(unittest.TestCase):
    def test_streaming_run_prints_and_accumulates_output(self) -> None:
        class Process:
            stdout = iter(["setup\n", 'COMFYCOLAB_READY={"comfyUrl":"https://demo.trycloudflare.com"}\n'])

            def wait(self):
                return 0

            def kill(self):
                pass

        def popen(command, **kwargs):
            return Process()

        client = ColabClient(
            "colab",
            "adc",
            Path("/tmp/sessions.json"),
            popen_factory=popen,
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = client.run_streaming("exec", timeout=None)
        self.assertIn("setup", output.getvalue())
        self.assertIn("COMFYCOLAB_READY=", result.stdout)

    def test_global_options_precede_subcommand(self) -> None:
        calls: list[list[str]] = []

        def runner(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, "[demo] endpoint | Status: IDLE\n", "")

        with tempfile.TemporaryDirectory() as directory:
            client = ColabClient(
                executable="/venv/bin/colab",
                auth="adc",
                config_path=Path(directory) / "sessions.json",
                runner=runner,
            )
            self.assertTrue(client.session_exists("demo"))

        self.assertEqual(
            calls[0],
            [
                "/venv/bin/colab",
                "--config",
                calls[0][2],
                "--auth",
                "adc",
                "status",
                "--session",
                "demo",
            ],
        )

    def test_missing_session_is_detected_from_stdout(self) -> None:
        def runner(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, "[colab] Session 'missing' not found.\n", "")

        client = ColabClient("colab", "oauth2", Path("/tmp/sessions.json"), runner)
        self.assertFalse(client.session_exists("missing"))

    def test_open_url_uses_colab_browser_flag(self) -> None:
        calls: list[list[str]] = []

        def runner(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, "https://colab.example/notebook\n", "")

        client = ColabClient("colab", "adc", Path("/tmp/sessions.json"), runner)
        client.open_url("demo")
        self.assertEqual(calls[0][-4:], ["url", "--session", "demo", "--open"])

    def test_parse_ready_payload_uses_last_sentinel(self) -> None:
        output = (
            "setup log\n"
            "COMFYCOLAB_READY={\"comfyUrl\":\"https://first.trycloudflare.com\"}\n"
            "more log\n"
            "COMFYCOLAB_READY={\"comfyUrl\":\"https://final.trycloudflare.com\",\"port\":8188}\n"
        )
        payload = parse_ready_payload(output)
        self.assertEqual(payload["comfyUrl"], "https://final.trycloudflare.com")


if __name__ == "__main__":
    unittest.main()
