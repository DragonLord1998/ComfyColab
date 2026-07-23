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
from typing import Callable


READY_PREFIX = "COMFYCOLAB_MAGE_FLOW_READY="
PROGRESS_PREFIX = "COMFYCOLAB_MAGE_FLOW_PROGRESS="
RESULT_PREFIX = "COMFYCOLAB_MAGE_FLOW_RESULT="
PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class MageFlowWorkerCommand:
    python: str
    worker_script: str
    source_dir: str
    model_id: str
    model_revision: str
    mode: str
    prompt: str
    negative_prompt: str
    output_image: str
    metadata_output: str
    request_id: str
    seed: int
    width: int
    height: int
    steps: int
    guidance_scale: float
    num_images: int = 1
    input_image: str = ""
    input_latent: str = ""
    output_latent: str = ""
    output_tensor: str = ""
    prompts: tuple[str, ...] = ()
    sigmas: tuple[float, ...] = ()
    strength: float = 0.75
    tile_size: int = 1536
    tile_overlap: int = 384
    source_ref: str = ""
    keep_worker_loaded: bool = True
    site_packages: str = ""

    def server_argv(self) -> list[str]:
        server_mode = "native" if self.mode.startswith("native_") else self.mode
        return [
            self.python,
            self.worker_script,
            "--server",
            "--source-dir",
            self.source_dir,
            "--model-id",
            self.model_id,
            "--model-revision",
            self.model_revision,
            "--mode",
            server_mode,
        ]

    def argv(self) -> list[str]:
        values = self.server_argv() + ["--one-shot"]
        request = build_mage_flow_request(self)
        for name, value in (
            ("--request-id", request["request_id"]),
            ("--prompt", request["prompt"]),
            ("--negative-prompt", request["negative_prompt"]),
            ("--output-image", request["output_image"]),
            ("--metadata-output", request["metadata_output"]),
            ("--seed", request["seed"]),
            ("--width", request["width"]),
            ("--height", request["height"]),
            ("--steps", request["steps"]),
            ("--guidance-scale", request["guidance_scale"]),
            ("--num-images", request["num_images"]),
            ("--input-image", request["input_image"]),
            ("--input-latent", request["input_latent"]),
            ("--output-latent", request["output_latent"]),
            ("--output-tensor", request["output_tensor"]),
            ("--strength", request["strength"]),
            ("--tile-size", request["tile_size"]),
            ("--tile-overlap", request["tile_overlap"]),
            ("--prompts-json", json.dumps(request["prompts"])),
            ("--sigmas-json", json.dumps(request["sigmas"])),
        ):
            values.extend((name, str(value)))
        return values


def build_mage_flow_request(command: MageFlowWorkerCommand) -> dict:
    mode = str(command.mode)
    valid_modes = {
        "text",
        "edit",
        "vae_encode",
        "native_denoise",
        "native_vae_encode",
        "native_vae_decode",
    }
    if mode not in valid_modes:
        raise ValueError(f"Unsupported MageFlow mode: {mode}")
    seed = int(command.seed)
    if seed < 0 or seed > (2**31) - 1:
        raise ValueError("MageFlow seed must be between 0 and 2147483647")
    width = int(command.width)
    height = int(command.height)
    minimum = 16 if mode in {"vae_encode", "native_denoise", "native_vae_encode", "native_vae_decode"} else 256
    if width < minimum or height < minimum or width > 2048 or height > 2048:
        raise ValueError(
            f"MageFlow {mode} width and height must be between {minimum} and 2048"
        )
    if width % 16 or height % 16:
        raise ValueError("MageFlow width and height must be multiples of 16")
    steps = int(command.steps)
    if steps < 1 or steps > 100:
        raise ValueError("MageFlow steps must be between 1 and 100")
    guidance = float(command.guidance_scale)
    if not 0.0 <= guidance <= 30.0:
        raise ValueError("MageFlow guidance_scale must be between 0 and 30")
    num_images = int(command.num_images)
    if num_images != 1:
        raise ValueError("ComfyColab MageFlow currently returns exactly one image")
    strength = float(command.strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("MageFlow edit strength must be between 0 and 1")
    input_image = str(command.input_image)
    if mode in {"edit", "vae_encode"} and not input_image:
        raise ValueError(f"MageFlow {mode} mode requires input_image")
    input_latent = str(command.input_latent)
    if mode in {"native_denoise", "native_vae_encode", "native_vae_decode"} and not input_latent:
        raise ValueError(f"MageFlow {mode} mode requires input_latent")
    if mode == "native_denoise" and "Edit" in str(command.model_id) and not input_image:
        raise ValueError("MageFlow native edit sampling requires input_image")
    output_latent = str(command.output_latent)
    if mode in {"vae_encode", "native_denoise", "native_vae_encode"} and not output_latent:
        raise ValueError(f"MageFlow {mode} mode requires output_latent")
    output_tensor = str(command.output_tensor)
    if mode == "native_vae_decode" and not output_tensor:
        raise ValueError("MageFlow native_vae_decode mode requires output_tensor")
    prompts = tuple(str(value) for value in command.prompts)
    sigmas = tuple(float(value) for value in command.sigmas)
    if mode == "native_denoise":
        if not prompts or len(prompts) != len(sigmas):
            raise ValueError(
                "MageFlow native_denoise requires one prompt and sigma per latent batch"
            )
    tile_size = int(command.tile_size)
    tile_overlap = int(command.tile_overlap)
    if tile_size < 64 or tile_size > 4096 or tile_size % 16:
        raise ValueError("MageFlow tile_size must be a multiple of 16 between 64 and 4096")
    if tile_overlap < 0 or tile_overlap >= tile_size or tile_overlap % 16:
        raise ValueError(
            "MageFlow tile_overlap must be a non-negative multiple of 16 "
            "smaller than tile_size"
        )
    return {
        "protocol": PROTOCOL_VERSION,
        "request_id": str(command.request_id),
        "model_id": str(command.model_id),
        "mode": mode,
        "prompt": str(command.prompt),
        "negative_prompt": str(command.negative_prompt),
        "output_image": str(command.output_image),
        "metadata_output": str(command.metadata_output),
        "seed": seed,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance,
        "num_images": num_images,
        "input_image": input_image,
        "input_latent": input_latent,
        "output_latent": output_latent,
        "output_tensor": output_tensor,
        "prompts": list(prompts),
        "sigmas": list(sigmas),
        "strength": strength,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "revisions": {
            "source": str(command.source_ref),
            "model": str(command.model_revision),
        },
    }


def _reader(stream, output: queue.Queue[str | None]) -> None:
    try:
        for line in iter(stream.readline, ""):
            output.put(line.rstrip("\n"))
    finally:
        output.put(None)


def _terminate_process(process: subprocess.Popen, timeout: float = 5.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=timeout)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=timeout)
    except OSError:
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout)


def _cleanup(command: MageFlowWorkerCommand, *, include_final: bool = True) -> None:
    paths = [Path(command.metadata_output)]
    if include_final:
        if command.mode in {"vae_encode", "native_denoise", "native_vae_encode"}:
            final_path = command.output_latent
        elif command.mode == "native_vae_decode":
            final_path = command.output_tensor
        else:
            final_path = command.output_image
        if final_path:
            paths.append(Path(final_path))
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass


class MageFlowWorkerPool:
    """Serialize Mage-Flow requests through one long-lived isolated process."""

    def __init__(
        self,
        *,
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        poll_interval: float = 0.1,
        startup_timeout: float = 180.0,
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

    def _launch(self, command: MageFlowWorkerCommand) -> None:
        self.close()
        self._lines = queue.Queue()
        argv = command.server_argv()
        env = os.environ.copy()
        env["COMFYCOLAB_MAGE_FLOW_SOURCE_REF"] = command.source_ref
        env["COMFYCOLAB_MAGE_FLOW_MODEL_REF"] = command.model_revision
        env["VF_HF_ATTN_IMPL"] = "sdpa"
        if command.site_packages:
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = command.site_packages + (
                os.pathsep + existing_pythonpath if existing_pythonpath else ""
            )
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
            raise RuntimeError("MageFlow worker pipes are unavailable")
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
                    raise RuntimeError("MageFlow worker protocol version mismatch")
                return
        self.close()
        raise RuntimeError(
            "MageFlow worker failed to become ready"
            + (f": {' | '.join(tail)}" if tail else "")
        )

    def _ensure_process(self, command: MageFlowWorkerCommand) -> None:
        signature = tuple(command.server_argv())
        if not self._process_running() or self._signature != signature:
            self._launch(command)

    def run(
        self,
        command: MageFlowWorkerCommand,
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
                    raise RuntimeError("MageFlow worker is unavailable")
                request = build_mage_flow_request(command)
                process.stdin.write(json.dumps(request, sort_keys=True) + "\n")
                process.stdin.flush()
                tail: list[str] = []
                while True:
                    if is_cancelled():
                        self.close()
                        raise InterruptedError("MageFlow generation was cancelled")
                    try:
                        line = self._lines.get(timeout=self._poll_interval)
                    except queue.Empty:
                        if not self._process_running():
                            raise RuntimeError(
                                "MageFlow worker exited before returning a matching result"
                                + (f": {' | '.join(tail)}" if tail else "")
                            )
                        continue
                    if line is None:
                        if not self._process_running():
                            raise RuntimeError("MageFlow worker output closed unexpectedly")
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
                        worker_traceback = str(result.get("traceback") or "").strip()
                        detail = (
                            f"\nWorker traceback:\n{worker_traceback}"
                            if worker_traceback
                            else ""
                        )
                        raise RuntimeError(
                            f"MageFlow worker failed: {error_type}: {message}{detail}"
                        )
                    if command.mode in {
                        "vae_encode",
                        "native_denoise",
                        "native_vae_encode",
                    }:
                        output_key = "output_latent"
                        expected_output = command.output_latent
                    elif command.mode == "native_vae_decode":
                        output_key = "output_tensor"
                        expected_output = command.output_tensor
                    else:
                        output_key = "output_image"
                        expected_output = command.output_image
                    output = Path(str(result.get(output_key, "")))
                    metadata = Path(str(result.get("metadata_output", "")))
                    if output.resolve() != Path(expected_output).resolve():
                        raise RuntimeError("MageFlow worker reported an unexpected output path")
                    if metadata.resolve() != Path(command.metadata_output).resolve():
                        raise RuntimeError("MageFlow worker reported an unexpected metadata path")
                    if not output.is_file():
                        raise RuntimeError(f"MageFlow worker {output_key} is missing")
                    if not metadata.is_file():
                        raise RuntimeError("MageFlow worker metadata is missing")
                    if not command.keep_worker_loaded:
                        self.close()
                    return result
            except BaseException:
                _cleanup(command)
                self.close()
                raise

    def close(self) -> None:
        process, self._process = self._process, None
        self._signature = None
        if process is not None:
            _terminate_process(process)


_GLOBAL_POOL: MageFlowWorkerPool | None = None
_GLOBAL_POOL_LOCK = threading.Lock()


def global_mage_flow_worker_pool() -> MageFlowWorkerPool:
    global _GLOBAL_POOL
    with _GLOBAL_POOL_LOCK:
        if _GLOBAL_POOL is None:
            _GLOBAL_POOL = MageFlowWorkerPool()
        return _GLOBAL_POOL


def _close_global_pool() -> None:
    if _GLOBAL_POOL is not None:
        _GLOBAL_POOL.close()


atexit.register(_close_global_pool)


__all__ = [
    "MageFlowWorkerCommand",
    "MageFlowWorkerPool",
    "build_mage_flow_request",
    "global_mage_flow_worker_pool",
]
