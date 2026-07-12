from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


READY_PREFIX = "COMFYCOLAB_READY="


class ColabCommandError(RuntimeError):
    def __init__(self, command: Sequence[str], result: subprocess.CompletedProcess[str]):
        self.command = list(command)
        self.result = result
        detail = (result.stderr or "").strip() or (result.stdout or "").strip() or "unknown error"
        super().__init__(f"Command failed ({result.returncode}): {' '.join(command)}\n{detail}")


Runner = Callable[..., subprocess.CompletedProcess[str]]
PopenFactory = Callable[..., subprocess.Popen[str]]


@dataclass(frozen=True)
class ColabClient:
    executable: str
    auth: str
    config_path: Path
    runner: Runner = subprocess.run
    popen_factory: PopenFactory = subprocess.Popen

    @classmethod
    def create(
        cls,
        *,
        executable: str | None,
        auth: str,
        config_path: Path,
        runner: Runner = subprocess.run,
        popen_factory: PopenFactory = subprocess.Popen,
    ) -> "ColabClient":
        resolved = executable or shutil.which("colab")
        if not resolved:
            raise RuntimeError(
                "The 'colab' executable is unavailable. Install this project in a virtual "
                "environment so its google-colab-cli dependency is on PATH."
            )
        return cls(resolved, auth, config_path.expanduser(), runner, popen_factory)

    def base_command(self) -> list[str]:
        return [
            self.executable,
            "--config",
            str(self.config_path),
            "--auth",
            self.auth,
        ]

    def run(
        self,
        *args: str,
        timeout: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        command = [*self.base_command(), *args]
        result = self.runner(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
        if check and result.returncode != 0:
            raise ColabCommandError(command, result)
        return result

    def session_exists(self, session: str) -> bool:
        result = self.run("status", "--session", session, check=False, timeout=60)
        marker = f"[{session}]"
        return result.returncode == 0 and marker in result.stdout

    def authenticate_interactively(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        command = [*self.base_command(), "sessions"]
        result = self.runner(
            command,
            text=True,
            timeout=300,
            env=os.environ.copy(),
        )
        if result.returncode != 0:
            raise ColabCommandError(command, result)

    def run_streaming(
        self,
        *args: str,
        timeout: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        command = [*self.base_command(), *args]
        process = self.popen_factory(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            bufsize=1,
        )
        timed_out = threading.Event()

        def expire() -> None:
            timed_out.set()
            process.kill()

        timer = threading.Timer(timeout, expire) if timeout is not None else None
        if timer:
            timer.start()
        chunks: list[str] = []
        try:
            assert process.stdout is not None
            for line in process.stdout:
                chunks.append(line)
                print(line, end="", flush=True)
            return_code = process.wait()
        finally:
            if timer:
                timer.cancel()

        if timed_out.is_set():
            raise subprocess.TimeoutExpired(command, timeout, output="".join(chunks))
        result = subprocess.CompletedProcess(command, return_code, "".join(chunks), "")
        if check and result.returncode != 0:
            raise ColabCommandError(command, result)
        return result

    def new(self, session: str, gpu: str) -> subprocess.CompletedProcess[str]:
        return self.run_streaming("new", "--session", session, "--gpu", gpu, timeout=300)

    def exec_bootstrap(
        self,
        *,
        session: str,
        source: str,
        remote_timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix="-comfycolab-bootstrap.py",
                delete=False,
            ) as handle:
                handle.write(source)
                path = handle.name
            return self.run_streaming(
                "exec",
                "--session",
                session,
                "--file",
                path,
                "--timeout",
                str(remote_timeout),
                timeout=remote_timeout + 120,
            )
        finally:
            if path:
                Path(path).unlink(missing_ok=True)

    def status(self, session: str) -> subprocess.CompletedProcess[str]:
        return self.run("status", "--session", session, check=False, timeout=60)

    def stop(self, session: str) -> subprocess.CompletedProcess[str]:
        return self.run("stop", "--session", session, check=False, timeout=120)


def parse_ready_payload(output: str) -> dict[str, object]:
    for line in reversed(output.splitlines()):
        if not line.startswith(READY_PREFIX):
            continue
        payload = json.loads(line[len(READY_PREFIX) :])
        if not isinstance(payload, dict) or not payload.get("comfyUrl"):
            raise ValueError("Bootstrap readiness payload is missing 'comfyUrl'.")
        return payload
    raise ValueError("Bootstrap completed without a ComfyColab readiness payload.")
