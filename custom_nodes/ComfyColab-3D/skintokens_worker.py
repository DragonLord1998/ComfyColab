from __future__ import annotations

import atexit
import json
import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

from .file3d import validate_glb


READY_PREFIX = "COMFYCOLAB_SKINTOKENS_READY="
PROGRESS_PREFIX = "COMFYCOLAB_SKINTOKENS_PROGRESS="
RESULT_PREFIX = "COMFYCOLAB_SKINTOKENS_RESULT="
PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class SkinTokensWorkerCommand:
    python: str
    worker_script: str
    source_dir: str
    model_dir: str
    qwen_dir: str
    checkpoint: str
    input_glb: str
    output_glb: str
    metadata_output: str
    request_id: str
    source_ref: str
    model_ref: str
    qwen_ref: str
    environment_ref: str
    preserve_texture: bool = True
    use_transfer: bool = True
    use_skeleton: bool = False
    use_postprocess: bool = False
    top_k: int = 5
    top_p: float = 0.95
    temperature: float = 1.0
    repetition_penalty: float = 2.0
    num_beams: int = 10
    keep_worker_loaded: bool = True

    def server_argv(self) -> list[str]:
        return [
            self.python,
            self.worker_script,
            "--server",
            "--source-dir",
            self.source_dir,
            "--model-dir",
            self.model_dir,
            "--qwen-dir",
            self.qwen_dir,
            "--checkpoint",
            self.checkpoint,
        ]

    def argv(self) -> list[str]:
        values = self.server_argv() + ["--one-shot"]
        request = build_skintokens_request(self)
        for name, value in (
            ("--request-id", request["request_id"]),
            ("--input-glb", request["input_glb"]),
            ("--output-glb", request["output_glb"]),
            ("--metadata-output", request["metadata_output"]),
        ):
            values.extend((name, str(value)))
        return values


def build_skintokens_request(command: SkinTokensWorkerCommand) -> dict:
    return {
        "protocol": PROTOCOL_VERSION,
        "request_id": command.request_id,
        "input_glb": command.input_glb,
        "output_glb": command.output_glb,
        "metadata_output": command.metadata_output,
        "preserve_texture": bool(command.preserve_texture),
        "use_transfer": bool(command.use_transfer),
        "use_skeleton": bool(command.use_skeleton),
        "use_postprocess": bool(command.use_postprocess),
        "top_k": int(command.top_k),
        "top_p": float(command.top_p),
        "temperature": float(command.temperature),
        "repetition_penalty": float(command.repetition_penalty),
        "num_beams": int(command.num_beams),
        "revisions": {
            "source": command.source_ref,
            "model": command.model_ref,
            "qwen": command.qwen_ref,
            "environment": command.environment_ref,
        },
    }


def _reader(stream: TextIO, output: queue.Queue[str | None]) -> None:
    try:
        for line in iter(stream.readline, ""):
            output.put(line.rstrip("\n"))
    finally:
        output.put(None)


def _terminate_process(process, timeout: float = 5.0) -> None:
    try:
        if process.poll() is not None:
            return
    except (AttributeError, OSError):
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            try:
                terminate()
            except OSError:
                pass
    wait = getattr(process, "wait", None)
    if not callable(wait):
        return
    try:
        wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError, PermissionError, OSError):
            kill = getattr(process, "kill", None)
            if callable(kill):
                try:
                    kill()
                except OSError:
                    pass
        try:
            wait(timeout=timeout)
        except (subprocess.TimeoutExpired, OSError):
            pass


def _owned_artifacts(command: SkinTokensWorkerCommand) -> tuple[Path, ...]:
    output = Path(command.output_glb)
    metadata = Path(command.metadata_output)
    return (
        output,
        metadata,
        output.with_suffix(output.suffix + ".partial"),
        output.with_name(output.stem + ".partial" + output.suffix),
        output.with_name("." + output.stem + ".partial" + output.suffix),
        metadata.with_suffix(metadata.suffix + ".partial"),
        metadata.with_name(metadata.stem + ".partial" + metadata.suffix),
        metadata.with_name("." + metadata.stem + ".partial" + metadata.suffix),
    )


def _cleanup(command: SkinTokensWorkerCommand, *, include_final: bool = True) -> None:
    artifacts = _owned_artifacts(command)
    if not include_final:
        artifacts = artifacts[2:]
    for artifact in artifacts:
        artifact.unlink(missing_ok=True)


def validate_skintokens_output(path: str | Path, *, preserve_texture: bool) -> dict:
    document = validate_glb(
        path,
        require_material=preserve_texture,
        require_texture=preserve_texture,
        require_uv=preserve_texture,
    )
    skins = document.get("skins") or []
    nodes = document.get("nodes") or []
    meshes = document.get("meshes") or []
    if not skins:
        raise ValueError("SkinTokens output GLB does not contain a skin")
    joint_count = 0
    for skin in skins:
        joints = skin.get("joints") or []
        if not joints:
            raise ValueError("SkinTokens output GLB skin has no joints")
        for joint in joints:
            if not isinstance(joint, int) or joint < 0 or joint >= len(nodes):
                raise ValueError("SkinTokens output GLB skin references an invalid joint")
        joint_count += len(joints)
    skinned_nodes = [
        node for node in nodes if isinstance(node, dict) and isinstance(node.get("skin"), int)
    ]
    if not skinned_nodes:
        raise ValueError("SkinTokens output GLB has no skinned mesh node")
    for node in skinned_nodes:
        mesh = node.get("mesh")
        skin = node.get("skin")
        if not isinstance(mesh, int) or mesh < 0 or mesh >= len(meshes):
            raise ValueError("SkinTokens output GLB skinned node references an invalid mesh")
        if not isinstance(skin, int) or skin < 0 or skin >= len(skins):
            raise ValueError("SkinTokens output GLB skinned node references an invalid skin")
    return {"skins": len(skins), "joints": joint_count, "skinned_nodes": len(skinned_nodes)}


class SkinTokensWorkerPool:
    """Serialize requests through one long-lived isolated SkinTokens process."""

    def __init__(
        self,
        *,
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        poll_interval: float = 0.1,
        startup_timeout: float = 120.0,
    ) -> None:
        self._popen_factory = popen_factory
        self._poll_interval = poll_interval
        self._startup_timeout = startup_timeout
        self._process = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._signature: tuple[str, ...] | None = None
        self._lock = threading.RLock()

    def _process_running(self) -> bool:
        if self._process is None:
            return False
        try:
            return self._process.poll() is None
        except (AttributeError, OSError):
            return True

    def _launch(self, command: SkinTokensWorkerCommand) -> None:
        self.close()
        self._lines = queue.Queue()
        argv = command.server_argv()
        env = os.environ.copy()
        env["COMFYCOLAB_SKINTOKENS_ENVIRONMENT_REF"] = command.environment_ref
        self._process = self._popen_factory(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=env,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self.close()
            raise RuntimeError("SkinTokens worker pipes are unavailable")
        self._reader_thread = threading.Thread(
            target=_reader, args=(self._process.stdout, self._lines), daemon=True
        )
        self._reader_thread.start()
        self._signature = tuple(argv)
        deadline = time.monotonic() + self._startup_timeout
        tail: list[str] = []
        while time.monotonic() < deadline:
            try:
                line = self._lines.get(timeout=self._poll_interval)
            except queue.Empty:
                if not self._process_running():
                    break
                continue
            if line is None:
                break
            tail = (tail + [line])[-40:]
            if line.startswith(READY_PREFIX):
                payload = json.loads(line.split("=", 1)[1])
                if int(payload.get("protocol", -1)) != PROTOCOL_VERSION:
                    self.close()
                    raise RuntimeError("SkinTokens worker protocol version mismatch")
                return
        self.close()
        raise RuntimeError(
            "SkinTokens worker failed to become ready"
            + (f": {' | '.join(tail)}" if tail else "")
        )

    def _ensure_process(self, command: SkinTokensWorkerCommand) -> None:
        signature = tuple(command.server_argv())
        if not self._process_running() or self._signature != signature:
            self._launch(command)

    def run(
        self,
        command: SkinTokensWorkerCommand,
        *,
        is_cancelled: Callable[[], bool] = lambda: False,
        on_progress: Callable[[dict], None] = lambda _event: None,
    ) -> dict:
        with self._lock:
            _cleanup(command, include_final=False)
            try:
                self._ensure_process(command)
                process = self._process
                if process is None or process.stdin is None:
                    raise RuntimeError("SkinTokens worker is unavailable")
                process.stdin.write(json.dumps(build_skintokens_request(command), sort_keys=True) + "\n")
                process.stdin.flush()
                tail: list[str] = []
                while True:
                    if is_cancelled():
                        self.close()
                        raise InterruptedError("SkinTokens rigging was cancelled")
                    try:
                        line = self._lines.get(timeout=self._poll_interval)
                    except queue.Empty:
                        if not self._process_running():
                            raise RuntimeError(
                                "SkinTokens worker exited before returning a matching result"
                                + (f": {' | '.join(tail)}" if tail else "")
                            )
                        continue
                    if line is None:
                        if not self._process_running():
                            raise RuntimeError("SkinTokens worker output closed unexpectedly")
                        continue
                    tail = (tail + [line])[-40:]
                    if line.startswith(PROGRESS_PREFIX):
                        event = json.loads(line.split("=", 1)[1])
                        if event.get("request_id") == command.request_id:
                            on_progress(event)
                        continue
                    if not line.startswith(RESULT_PREFIX):
                        continue
                    result = json.loads(line.split("=", 1)[1])
                    if result.get("request_id") != command.request_id:
                        continue
                    if result.get("status") != "ok":
                        error_type = str(result.get("error_type") or "RuntimeError")
                        message = str(result.get("error") or "unknown worker failure")
                        raise RuntimeError(f"SkinTokens worker failed: {error_type}: {message}")
                    output = Path(str(result.get("output_glb", "")))
                    metadata = Path(str(result.get("metadata_output", "")))
                    if output.resolve() != Path(command.output_glb).resolve():
                        raise RuntimeError("SkinTokens worker reported an unexpected output path")
                    if metadata.resolve() != Path(command.metadata_output).resolve():
                        raise RuntimeError("SkinTokens worker reported an unexpected metadata path")
                    contract = validate_skintokens_output(
                        output, preserve_texture=command.preserve_texture
                    )
                    if not metadata.is_file():
                        raise RuntimeError("SkinTokens worker metadata is missing")
                    metadata_payload = json.loads(metadata.read_text(encoding="utf-8"))
                    if not metadata_payload.get("texture_preservation", {}).get("requested") == command.preserve_texture:
                        raise RuntimeError("SkinTokens metadata omitted the requested texture policy")
                    if not command.keep_worker_loaded:
                        self.close()
                    return {**result, "rig_contract": contract}
            except BaseException:
                _cleanup(command)
                self.close()
                raise

    def close(self) -> None:
        process, self._process = self._process, None
        self._signature = None
        if process is not None:
            _terminate_process(process)


_GLOBAL_POOL: SkinTokensWorkerPool | None = None
_GLOBAL_POOL_LOCK = threading.Lock()


def global_skintokens_worker_pool() -> SkinTokensWorkerPool:
    global _GLOBAL_POOL
    with _GLOBAL_POOL_LOCK:
        if _GLOBAL_POOL is None:
            _GLOBAL_POOL = SkinTokensWorkerPool()
        return _GLOBAL_POOL


def _close_global_pool() -> None:
    if _GLOBAL_POOL is not None:
        _GLOBAL_POOL.close()


atexit.register(_close_global_pool)


__all__ = [
    "SkinTokensWorkerCommand",
    "SkinTokensWorkerPool",
    "build_skintokens_request",
    "global_skintokens_worker_pool",
    "validate_skintokens_output",
]
