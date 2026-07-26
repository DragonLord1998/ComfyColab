from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .geometry_quality import validate_volumetric_glb


RESULT_PREFIX = "COMFYCOLAB_RESULT="
PROGRESS_PREFIX = "COMFYCOLAB_PROGRESS="


@dataclass(frozen=True)
class MeshFlowWorkerCommand:
    python: str
    worker_script: str
    source_dir: str
    checkpoint_dir: str
    dinov3_model_dir: str
    input_mesh: str
    reference_images: tuple[str, ...]
    output_mesh: str
    metadata_output: str
    steps: int
    num_verts: int
    guidance_scale: float
    seed: int
    dtype: str
    compile_models: bool
    source_ref: str
    model_ref: str
    appearance_mesh: str = ""

    def argv(
        self,
        output_override: str | None = None,
        metadata_override: str | None = None,
    ) -> list[str]:
        argv = [
            self.python,
            self.worker_script,
            "--source-dir",
            self.source_dir,
            "--checkpoint-dir",
            self.checkpoint_dir,
            "--input-mesh",
            self.input_mesh,
            "--output-mesh",
            output_override or self.output_mesh,
            "--metadata-output",
            metadata_override or self.metadata_output,
            "--steps",
            str(self.steps),
            "--num-verts",
            str(self.num_verts),
            "--guidance-scale",
            str(float(self.guidance_scale)),
            "--seed",
            str(self.seed),
            "--dtype",
            self.dtype,
            "--source-ref",
            self.source_ref,
            "--model-ref",
            self.model_ref,
        ]
        if self.dinov3_model_dir:
            argv.extend(["--dinov3-model-dir", self.dinov3_model_dir])
        if self.appearance_mesh:
            argv.extend(["--appearance-mesh", self.appearance_mesh])
        for reference_image in self.reference_images:
            argv.extend(["--reference-image", reference_image])
        if self.compile_models:
            argv.append("--compile")
        return argv


def _reader(stream, lines: queue.Queue[str | None]) -> None:
    try:
        for line in iter(stream.readline, ""):
            lines.put(line.rstrip("\n"))
    finally:
        lines.put(None)


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)


def run_meshflow_worker(
    command: MeshFlowWorkerCommand,
    *,
    is_cancelled: Callable[[], bool] = lambda: False,
    on_progress: Callable[[dict], None] = lambda _event: None,
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> dict:
    output = Path(command.output_mesh)
    metadata = Path(command.metadata_output)
    partial = output.with_name(f".{output.stem}.{os.getpid()}.partial.glb")
    partial_metadata = metadata.with_name(
        f".{metadata.stem}.{os.getpid()}.partial.json"
    )
    process = popen_factory(
        command.argv(str(partial), str(partial_metadata)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    if process.stdout is None:
        raise RuntimeError("MeshFlow worker stdout pipe is unavailable")
    lines: queue.Queue[str | None] = queue.Queue()
    threading.Thread(
        target=_reader, args=(process.stdout, lines), daemon=True
    ).start()
    result: dict = {}
    tail: list[str] = []
    try:
        closed = False
        while process.poll() is None or not closed:
            if is_cancelled():
                _terminate(process)
                raise InterruptedError("MeshFlow generation was cancelled")
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                closed = True
                continue
            tail = (tail + [line])[-40:]
            if line.startswith(PROGRESS_PREFIX):
                on_progress(json.loads(line[len(PROGRESS_PREFIX) :]))
            elif line.startswith(RESULT_PREFIX):
                result = json.loads(line[len(RESULT_PREFIX) :])
        return_code = process.wait()
        if result.get("status") == "error":
            raise RuntimeError(
                f"MeshFlow worker failed: {result.get('error_type')}: "
                f"{result.get('error')}"
            )
        if return_code:
            raise RuntimeError(
                f"MeshFlow worker exited with {return_code}: {' | '.join(tail)}"
            )
        if not result:
            raise RuntimeError("MeshFlow worker exited without COMFYCOLAB_RESULT")
        if Path(str(result.get("output_mesh", ""))).resolve() != partial.resolve():
            raise RuntimeError("MeshFlow worker reported an unexpected output path")
        validate_volumetric_glb(
            partial,
            stage="MeshFlow worker output",
            require_material=False,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, output)
        os.replace(partial_metadata, metadata)
        result["output_mesh"] = str(output)
        result["metadata_output"] = str(metadata)
        return result
    except BaseException:
        _terminate(process)
        output.unlink(missing_ok=True)
        metadata.unlink(missing_ok=True)
        raise
    finally:
        partial.unlink(missing_ok=True)
        partial_metadata.unlink(missing_ok=True)
