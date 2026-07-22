from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .mage_flow_worker import (
    MageFlowWorkerCommand,
    global_mage_flow_worker_pool,
)


MAGE_FLOW_SOURCE_REF = "1c4727a6daea1200488d9c68544ebea2e784c765"
MAGE_FLOW_SOURCE_REPOSITORY = "https://github.com/microsoft/Mage.git"
MAGE_FLOW_MODELS = {
    "flow": {
        "model_id": "microsoft/Mage-Flow",
        "revision": "d272c957b204b92040be6e6edfac8912823a0e15",
        "environment": "COMFYCOLAB_MAGEFLOW_MODEL",
        "mode": "text",
        "default_steps": 20,
        "default_guidance": 5.0,
    },
    "flow_turbo": {
        "model_id": "microsoft/Mage-Flow-Turbo",
        "revision": "09ecbcc42576c88b25c7c160fc57b3ec412ecb60",
        "environment": "COMFYCOLAB_MAGEFLOW_TURBO_MODEL",
        "mode": "text",
        "default_steps": 4,
        "default_guidance": 1.0,
    },
    "edit": {
        "model_id": "microsoft/Mage-Flow-Edit",
        "revision": "79de4e0869566c20960ffa1c9ac8154a840f6eca",
        "environment": "COMFYCOLAB_MAGEFLOW_EDIT_MODEL",
        "mode": "edit",
        "default_steps": 30,
        "default_guidance": 5.0,
    },
    "edit_turbo": {
        "model_id": "microsoft/Mage-Flow-Edit-Turbo",
        "revision": "286ac4b5429ab803e63073c79c10c9a205793c86",
        "environment": "COMFYCOLAB_MAGEFLOW_EDIT_TURBO_MODEL",
        "mode": "edit",
        "default_steps": 4,
        "default_guidance": 1.0,
    },
}
MAGE_FLOW_WORKER_REQUIREMENTS = ("transformers==5.5.0", "loguru==0.7.3")
_WORKER_DEPENDENCY_LOCK = threading.Lock()


def _io():
    return importlib.import_module("comfy_api.latest").io


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime_root() -> Path:
    root = Path(os.environ.get("COMFYCOLAB_CACHE_DIR", "")) if os.environ.get("COMFYCOLAB_CACHE_DIR") else _repo_root() / ".cache" / "comfycolab"
    path = root / "mage_flow"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _source_dir() -> Path:
    configured = os.environ.get("COMFYCOLAB_MAGEFLOW_SOURCE") or os.environ.get(
        "COMFYCOLAB_MAGE_FLOW_SOURCE_DIR"
    )
    candidates = [Path(configured)] if configured else []
    candidates.append(_repo_root() / "repositories" / "Mage-Flow")
    target = _runtime_root() / "source" / "Mage"
    candidates.append(target)
    for candidate in candidates:
        if (candidate / "mage_flow" / "pipeline.py").is_file():
            return candidate

    with _WORKER_DEPENDENCY_LOCK:
        if not (target / ".git").is_dir():
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.check_call(
                ["git", "clone", "--filter=blob:none", MAGE_FLOW_SOURCE_REPOSITORY, str(target)]
            )
        subprocess.check_call(
            ["git", "-C", str(target), "fetch", "origin", MAGE_FLOW_SOURCE_REF, "--depth", "1"]
        )
        subprocess.check_call(
            ["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"]
        )
        actual = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != MAGE_FLOW_SOURCE_REF:
            raise RuntimeError(
                f"Mage-Flow source revision mismatch: expected {MAGE_FLOW_SOURCE_REF}, got {actual}"
            )
    return target


def _worker_script() -> Path:
    configured = os.environ.get("COMFYCOLAB_MAGEFLOW_WORKER")
    if configured:
        return Path(configured)
    return _repo_root() / "worker" / "mage_flow" / "worker_main.py"


def _python() -> str:
    return (
        os.environ.get("COMFYCOLAB_MAGEFLOW_PYTHON")
        or os.environ.get("COMFYCOLAB_MAGE_FLOW_PYTHON")
        or os.environ.get("PYTHON")
        or sys.executable
    )


def _worker_site_packages() -> str:
    configured = os.environ.get("COMFYCOLAB_MAGEFLOW_SITE_PACKAGES")
    if configured:
        return configured
    if os.environ.get("COMFYCOLAB_MAGEFLOW_PYTHON") or os.environ.get(
        "COMFYCOLAB_MAGE_FLOW_PYTHON"
    ):
        return ""

    target = _runtime_root() / "python-packages"
    marker = target / ".comfycolab-mageflow-requirements.json"
    expected = {
        "python": sys.version.split()[0],
        "requirements": list(MAGE_FLOW_WORKER_REQUIREMENTS),
    }
    with _WORKER_DEPENDENCY_LOCK:
        if marker.is_file():
            try:
                if json.loads(marker.read_text(encoding="utf-8")) == expected:
                    return str(target)
            except (OSError, json.JSONDecodeError):
                pass
        target.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--target",
                str(target),
                *MAGE_FLOW_WORKER_REQUIREMENTS,
            ]
        )
        marker.write_text(json.dumps(expected, sort_keys=True) + "\n", encoding="utf-8")
    return str(target)


def _stable_value(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        array = value.detach().cpu().contiguous().numpy()
        return {
            "kind": "tensor",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _stable_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _request_key(variant: str, **inputs: Any) -> str:
    payload = json.dumps(
        {"variant": variant, "inputs": _stable_value(inputs)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _image_tensor_to_png(image: Any, path: Path) -> None:
    import numpy as np
    from PIL import Image

    tensor = image
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu()
    if hasattr(tensor, "numpy"):
        array = tensor.numpy()
    else:
        array = np.asarray(tensor)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError("MageFlow edit input image must be an HWC or BHWC tensor")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] > 3:
        array = array[..., :3]
    if array.dtype.kind == "f":
        array = np.clip(array, 0.0, 1.0) * 255.0
    array = np.clip(array, 0, 255).astype("uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def _png_to_image_tensor(path: Path):
    import numpy as np
    import torch
    from PIL import Image

    image = Image.open(path).convert("RGB")
    array = np.asarray(image).astype("float32") / 255.0
    return torch.from_numpy(array)[None,]


def _send_progress_text(node_id: Any, text: str) -> None:
    if not node_id:
        return
    try:
        server = importlib.import_module("server").PromptServer.instance
        server.send_sync(
            "progress",
            {"node": node_id, "value": 0, "max": 1, "text": text},
        )
    except Exception:
        return


def _builder():
    return importlib.import_module("comfy_execution.graph_utils").GraphBuilder()


def _finish(graph, image):
    return _io().NodeOutput(image, expand=graph.finalize())


@dataclass(frozen=True)
class _ResolvedSettings:
    width: int
    height: int
    steps: int
    guidance_scale: float
    strength: float
    keep_worker_loaded: bool


def _resolve_settings(
    defaults: dict[str, Any],
    *,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    strength: float,
    keep_worker_loaded: bool,
) -> _ResolvedSettings:
    resolved_steps = int(steps) if int(steps) > 0 else int(defaults["default_steps"])
    resolved_guidance = (
        float(guidance_scale)
        if float(guidance_scale) >= 0.0
        else float(defaults["default_guidance"])
    )
    return _ResolvedSettings(
        width=int(width),
        height=int(height),
        steps=resolved_steps,
        guidance_scale=resolved_guidance,
        strength=float(strength),
        keep_worker_loaded=bool(keep_worker_loaded),
    )


def _run_mage_flow(
    variant: str,
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    keep_worker_loaded: bool,
    cache_mode: str,
    progress_node_id: Any = None,
    image: Any = None,
    strength: float = 0.75,
):
    defaults = MAGE_FLOW_MODELS[variant]
    settings = _resolve_settings(
        defaults,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        strength=strength,
        keep_worker_loaded=keep_worker_loaded,
    )
    seed = int(seed)
    if seed < 0 or seed > (2**31) - 1:
        raise ValueError("MageFlow seed must be between 0 and 2147483647")
    if cache_mode not in {"Use cache", "Refresh cache"}:
        raise ValueError("MageFlow cache_mode must be Use cache or Refresh cache")
    key = _request_key(
        variant,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        width=settings.width,
        height=settings.height,
        steps=settings.steps,
        guidance_scale=settings.guidance_scale,
        strength=settings.strength,
        image=image,
        source_ref=MAGE_FLOW_SOURCE_REF,
        model_ref=defaults["revision"],
        noise="seeded_torch_randn",
        screening="disabled",
        watermark="disabled",
        attn_type="sdpa",
    )
    case_dir = _runtime_root() / key
    output_image = case_dir / "image.png"
    metadata_output = case_dir / "metadata.json"
    input_image = ""
    if image is not None:
        input_path = case_dir / "input.png"
        _image_tensor_to_png(image, input_path)
        input_image = str(input_path)
    if cache_mode == "Use cache" and output_image.is_file() and metadata_output.is_file():
        _send_progress_text(progress_node_id, "Complete - Loaded cached MageFlow image")
        return _io().NodeOutput(_png_to_image_tensor(output_image))
    case_dir.mkdir(parents=True, exist_ok=True)
    _send_progress_text(progress_node_id, "Stage 1/1 - Running isolated MageFlow worker...")
    command = MageFlowWorkerCommand(
        python=_python(),
        worker_script=str(_worker_script()),
        source_dir=str(_source_dir()),
        model_id=str(defaults["model_id"]),
        model_revision=str(defaults["revision"]),
        mode=str(defaults["mode"]),
        prompt=str(prompt),
        negative_prompt=str(negative_prompt),
        output_image=str(output_image),
        metadata_output=str(metadata_output),
        request_id=uuid.uuid4().hex,
        seed=seed,
        width=settings.width,
        height=settings.height,
        steps=settings.steps,
        guidance_scale=settings.guidance_scale,
        input_image=input_image,
        strength=settings.strength,
        source_ref=MAGE_FLOW_SOURCE_REF,
        keep_worker_loaded=settings.keep_worker_loaded,
        site_packages=_worker_site_packages(),
    )
    result = global_mage_flow_worker_pool().run(
        command,
        on_progress=lambda event: _send_progress_text(
            progress_node_id,
            str(event.get("message") or event.get("stage") or "MageFlow progress"),
        ),
    )
    print(
        "COMFYCOLAB_MAGE_FLOW_RESULT="
        + json.dumps(
            {
                "variant": variant,
                "worker_pid": result.get("worker_pid"),
                "model_revision": defaults["revision"],
                "source_revision": MAGE_FLOW_SOURCE_REF,
                "screening": "disabled",
                "watermark": "disabled",
                "noise": "seeded_torch_randn",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    _send_progress_text(progress_node_id, "Complete - MageFlow image generated")
    return _io().NodeOutput(_png_to_image_tensor(output_image))


def build_mage_flow_graph(
    *,
    variant: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    keep_worker_loaded: bool,
    cache_mode: str,
    image: Any = None,
):
    graph = _builder()
    noise = graph.node("RandomNoise", noise_seed=int(seed))
    worker = graph.node(
        "ComfyColabMageFlowWorker",
        image=image,
        noise=noise.out(0),
        variant=variant,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=int(seed),
        width=int(width),
        height=int(height),
        steps=int(steps),
        guidance=float(guidance),
        keep_worker_loaded=bool(keep_worker_loaded),
        cache_mode=cache_mode,
    )
    return _finish(graph, worker.out(0))


class _DevNode:
    @classmethod
    def _schema(cls, inputs, outputs):
        return _io().Schema(
            node_id=cls.__name__,
            display_name=cls.__name__,
            category="ComfyColab/Image",
            is_dev_only=True,
            inputs=inputs,
            outputs=outputs,
        )


class ComfyColabMageFlowWorker(_DevNode):
    @classmethod
    def define_schema(cls):
        io = _io()
        return cls._schema(
            [
                io.Image.Input("image", optional=True),
                io.Custom("NOISE").Input("noise", optional=True),
                io.Combo.Input("variant", options=list(MAGE_FLOW_MODELS)),
                io.String.Input("prompt", multiline=True, default=""),
                io.String.Input("negative_prompt", multiline=True, default=""),
                io.Int.Input("seed", default=0, min=0, max=(2**31) - 1),
                io.Int.Input("width", default=1024, min=256, max=2048, step=16),
                io.Int.Input("height", default=1024, min=256, max=2048, step=16),
                io.Int.Input("steps", default=30, min=1, max=100),
                io.Float.Input("guidance", default=5.0, min=0.0, max=30.0, step=0.1),
                io.Boolean.Input("keep_worker_loaded", default=True),
                io.Combo.Input("cache_mode", options=["Use cache", "Refresh cache"], default="Use cache"),
            ],
            [io.Image.Output("image")],
        )

    @classmethod
    def execute(
        cls,
        image=None,
        noise=None,
        variant="flow",
        prompt="",
        negative_prompt="",
        seed=0,
        width=1024,
        height=1024,
        steps=30,
        guidance=5.0,
        keep_worker_loaded=True,
        cache_mode="Use cache",
    ):
        del noise
        return _run_mage_flow(
            variant,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance,
            keep_worker_loaded=keep_worker_loaded,
            cache_mode=cache_mode,
            image=image,
            strength=0.75,
        )


class _MageFlowTextBase:
    VARIANT = "flow"

    @classmethod
    def define_schema(cls):
        io = _io()
        defaults = MAGE_FLOW_MODELS[cls.VARIANT]
        return io.Schema(
            node_id=cls.__name__,
            display_name=NODE_DISPLAY_NAME_MAPPINGS[cls.__name__],
            category="ComfyColab/Image",
            description=(
                "Runs the pinned Microsoft Mage-Flow text-to-image model in an isolated "
                "persistent worker with deterministic seeded noise."
            ),
            enable_expand=True,
            inputs=[
                io.String.Input("prompt", multiline=True, default=""),
                io.String.Input("negative_prompt", multiline=True, default="", advanced=True),
                io.Int.Input("seed", default=0, min=0, max=(2**31) - 1),
                io.Int.Input("width", default=1024, min=256, max=2048, step=16),
                io.Int.Input("height", default=1024, min=256, max=2048, step=16),
                io.Int.Input(
                    "steps",
                    default=defaults["default_steps"],
                    min=1,
                    max=100,
                    advanced=True,
                ),
                io.Float.Input(
                    "guidance",
                    default=defaults["default_guidance"],
                    min=0.0,
                    max=30.0,
                    step=0.1,
                    advanced=True,
                ),
                io.Boolean.Input("keep_worker_loaded", default=True, advanced=True),
                io.Combo.Input(
                    "cache_mode",
                    options=["Use cache", "Refresh cache"],
                    default="Use cache",
                    advanced=True,
                ),
            ],
            outputs=[io.Image.Output("image")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        prompt: str,
        negative_prompt: str = "",
        seed: int = 0,
        width: int = 1024,
        height: int = 1024,
        steps: int | None = None,
        guidance: float | None = None,
        keep_worker_loaded: bool = True,
        cache_mode: str = "Use cache",
    ):
        defaults = MAGE_FLOW_MODELS[cls.VARIANT]
        return build_mage_flow_graph(
            variant=cls.VARIANT,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            steps=int(defaults["default_steps"] if steps is None else steps),
            guidance=float(defaults["default_guidance"] if guidance is None else guidance),
            keep_worker_loaded=keep_worker_loaded,
            cache_mode=cache_mode,
        )


class _MageFlowEditBase:
    VARIANT = "edit"

    @classmethod
    def define_schema(cls):
        io = _io()
        defaults = MAGE_FLOW_MODELS[cls.VARIANT]
        return io.Schema(
            node_id=cls.__name__,
            display_name=NODE_DISPLAY_NAME_MAPPINGS[cls.__name__],
            category="ComfyColab/Image",
            description=(
                "Runs the pinned Microsoft Mage-Flow edit model in an isolated persistent "
                "worker with deterministic seeded noise."
            ),
            enable_expand=True,
            inputs=[
                io.Image.Input("image"),
                io.String.Input("prompt", multiline=True, default=""),
                io.String.Input("negative_prompt", multiline=True, default="", advanced=True),
                io.Int.Input("seed", default=0, min=0, max=(2**31) - 1),
                io.Int.Input("width", default=1024, min=256, max=2048, step=16, advanced=True),
                io.Int.Input("height", default=1024, min=256, max=2048, step=16, advanced=True),
                io.Int.Input(
                    "steps",
                    default=defaults["default_steps"],
                    min=1,
                    max=100,
                    advanced=True,
                ),
                io.Float.Input(
                    "guidance",
                    default=defaults["default_guidance"],
                    min=0.0,
                    max=30.0,
                    step=0.1,
                    advanced=True,
                ),
                io.Boolean.Input("keep_worker_loaded", default=True, advanced=True),
                io.Combo.Input(
                    "cache_mode",
                    options=["Use cache", "Refresh cache"],
                    default="Use cache",
                    advanced=True,
                ),
            ],
            outputs=[io.Image.Output("image")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        image,
        prompt: str,
        negative_prompt: str = "",
        seed: int = 0,
        width: int = 1024,
        height: int = 1024,
        steps: int | None = None,
        guidance: float | None = None,
        keep_worker_loaded: bool = True,
        cache_mode: str = "Use cache",
    ):
        defaults = MAGE_FLOW_MODELS[cls.VARIANT]
        return build_mage_flow_graph(
            variant=cls.VARIANT,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            steps=int(defaults["default_steps"] if steps is None else steps),
            guidance=float(defaults["default_guidance"] if guidance is None else guidance),
            keep_worker_loaded=keep_worker_loaded,
            cache_mode=cache_mode,
            image=image,
        )


class ComfyColabMageFlow(_MageFlowTextBase):
    VARIANT = "flow"


class ComfyColabMageFlowTurbo(_MageFlowTextBase):
    VARIANT = "flow_turbo"


class ComfyColabMageFlowEdit(_MageFlowEditBase):
    VARIANT = "edit"


class ComfyColabMageFlowEditTurbo(_MageFlowEditBase):
    VARIANT = "edit_turbo"


NODE_CLASS_MAPPINGS = {
    "ComfyColabMageFlow": ComfyColabMageFlow,
    "ComfyColabMageFlowTurbo": ComfyColabMageFlowTurbo,
    "ComfyColabMageFlowEdit": ComfyColabMageFlowEdit,
    "ComfyColabMageFlowEditTurbo": ComfyColabMageFlowEditTurbo,
    "ComfyColabMageFlowWorker": ComfyColabMageFlowWorker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyColabMageFlow": "ComfyColab Mage-Flow",
    "ComfyColabMageFlowTurbo": "ComfyColab Mage-Flow Turbo",
    "ComfyColabMageFlowEdit": "ComfyColab Mage-Flow Edit",
    "ComfyColabMageFlowEditTurbo": "ComfyColab Mage-Flow Edit Turbo",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
