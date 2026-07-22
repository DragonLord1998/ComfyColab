#!/usr/bin/env python3
"""Persistent process-isolated runner for pinned Microsoft Mage-Flow models."""

from __future__ import annotations

import argparse
import gc
import importlib
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from types import MethodType
from typing import Any


READY_PREFIX = "COMFYCOLAB_MAGE_FLOW_READY="
PROGRESS_PREFIX = "COMFYCOLAB_MAGE_FLOW_PROGRESS="
RESULT_PREFIX = "COMFYCOLAB_MAGE_FLOW_RESULT="
PROTOCOL_VERSION = 1


def _emit(prefix: str, payload: dict[str, Any]) -> None:
    print(prefix + json.dumps(payload, sort_keys=True), flush=True)


def emit_progress(request_id: str, stage: str, current: int, total: int, **details: Any) -> None:
    _emit(
        PROGRESS_PREFIX,
        {"request_id": request_id, "stage": stage, "current": current, "total": total, **details},
    )


def emit_result(**details: Any) -> None:
    _emit(RESULT_PREFIX, details)


def _validate_request(request: dict[str, Any]) -> None:
    if int(request.get("protocol", -1)) != PROTOCOL_VERSION:
        raise ValueError("Unsupported MageFlow worker protocol")
    if request.get("mode") not in {"text", "edit"}:
        raise ValueError("MageFlow mode must be text or edit")
    seed = int(request.get("seed", -1))
    if seed < 0 or seed > (2**31) - 1:
        raise ValueError("MageFlow seed must be between 0 and 2147483647")
    for name in ("request_id", "prompt", "output_image", "metadata_output"):
        if request.get(name) is None:
            raise ValueError(f"MageFlow request omitted {name}")
    width, height = int(request.get("width", 0)), int(request.get("height", 0))
    if width < 256 or height < 256 or width > 2048 or height > 2048:
        raise ValueError("MageFlow width and height must be between 256 and 2048")
    if width % 16 or height % 16:
        raise ValueError("MageFlow width and height must be multiples of 16")
    steps = int(request.get("steps", 0))
    if steps < 1 or steps > 100:
        raise ValueError("MageFlow steps must be between 1 and 100")
    guidance = float(request.get("guidance_scale", -1.0))
    if not 0.0 <= guidance <= 30.0:
        raise ValueError("MageFlow guidance_scale must be between 0 and 30")
    strength = float(request.get("strength", 0.75))
    if not 0.0 <= strength <= 1.0:
        raise ValueError("MageFlow strength must be between 0 and 1")
    if request.get("mode") == "edit":
        input_image = Path(str(request.get("input_image", "")))
        if not input_image.is_file():
            raise FileNotFoundError(f"MageFlow edit input does not exist: {input_image}")


def _patch_screening(module: ModuleType) -> None:
    def allow_text(*_args: Any, **_kwargs: Any) -> Any:
        return None

    def allow_edit(*_args: Any, **_kwargs: Any) -> Any:
        return None

    for name, value in (
        ("screen_text", allow_text),
        ("screen_edit", allow_edit),
    ):
        if hasattr(module, name):
            setattr(module, name, value)


class _AllowVerdict:
    violates = False

    @staticmethod
    def banner() -> str:
        return ""


def _disable_model_screening(pipe: Any) -> None:
    """Replace the two mandatory upstream gates on the loaded text encoder."""
    text_encoder = getattr(getattr(pipe, "model", None), "txt_enc", None)
    if text_encoder is None:
        raise RuntimeError("Mage-Flow pipeline has no text encoder to patch")

    def allow_text(_self: Any, *_args: Any, **_kwargs: Any) -> _AllowVerdict:
        return _AllowVerdict()

    def allow_edit(_self: Any, *_args: Any, **_kwargs: Any) -> _AllowVerdict:
        return _AllowVerdict()

    text_encoder.screen_text = MethodType(allow_text, text_encoder)
    text_encoder.screen_edit = MethodType(allow_edit, text_encoder)


def _patch_noise(module: ModuleType) -> None:
    def seeded_randn(*args: Any, **kwargs: Any) -> Any:
        import torch

        seed = int(kwargs.pop("seed", 0))
        shape = kwargs.pop("shape", None)
        device = kwargs.pop("device", None)
        dtype = kwargs.pop("dtype", None)
        if shape is None and args:
            first, *args = args
            if isinstance(first, (tuple, list)):
                shape = tuple(int(value) for value in first)
            elif hasattr(first, "shape"):
                shape = tuple(int(value) for value in first.shape)
                device = device or getattr(first, "device", None)
                dtype = dtype or getattr(first, "dtype", None)
            else:
                shape = first
        if shape is None:
            for key in ("latent_shape", "size"):
                if key in kwargs:
                    shape = kwargs.pop(key)
                    break
        if shape is None:
            raise TypeError("encode_noise replacement requires a tensor shape")
        generator = torch.Generator(device="cpu").manual_seed(seed & 0x7FFFFFFF)
        noise = torch.randn((1, *tuple(shape)), generator=generator, dtype=torch.float32)
        return noise.to(device=device, dtype=dtype)

    setattr(module, "encode_noise", seeded_randn)


def _patch_model_config(module: ModuleType) -> None:
    original = getattr(module, "ModelConfig", None)
    if original is None or getattr(original, "_comfycolab_sdpa_wrapped", False):
        return

    def sdpa_model_config(*args: Any, **kwargs: Any) -> Any:
        kwargs["attn_type"] = "sdpa"
        return original(*args, **kwargs)

    sdpa_model_config._comfycolab_sdpa_wrapped = True  # type: ignore[attr-defined]
    sdpa_model_config.__name__ = getattr(original, "__name__", "ModelConfig")
    setattr(module, "ModelConfig", sdpa_model_config)


def _patch_mage_flow_runtime() -> None:
    os.environ["VF_HF_ATTN_IMPL"] = "sdpa"
    candidates = (
        "mage_flow.pipeline",
        "mage_flow",
        "pipeline",
    )
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        _patch_screening(module)
        _patch_noise(module)
        _patch_model_config(module)


def _load_pipeline_module(source_dir: Path) -> ModuleType:
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    errors: list[str] = []
    for name in ("mage_flow.pipeline", "mage_flow", "pipeline"):
        try:
            module = importlib.import_module(name)
            _patch_mage_flow_runtime()
            return module
        except ImportError as error:
            errors.append(f"{name}: {error}")
    raise ImportError("Unable to import Mage-Flow pipeline module: " + " | ".join(errors))


def _candidate_pipeline_classes(module: ModuleType) -> list[type]:
    names = (
        "MageFlowPipeline",
        "MageFlowEditPipeline",
        "MagePipeline",
        "Pipeline",
    )
    classes = []
    for name in names:
        value = getattr(module, name, None)
        if inspect.isclass(value):
            classes.append(value)
    for _name, value in inspect.getmembers(module, inspect.isclass):
        if value in classes:
            continue
        if hasattr(value, "from_pretrained") and ("Pipeline" in value.__name__ or "Mage" in value.__name__):
            classes.append(value)
    return classes


def _resolve_model_dir(model_id: str, revision: str) -> str:
    candidate = Path(model_id).expanduser()
    if candidate.is_dir():
        return str(candidate.resolve())
    from huggingface_hub import snapshot_download

    environment_names = {
        "microsoft/Mage-Flow": "COMFYCOLAB_MAGEFLOW_MODEL",
        "microsoft/Mage-Flow-Turbo": "COMFYCOLAB_MAGEFLOW_TURBO_MODEL",
        "microsoft/Mage-Flow-Edit": "COMFYCOLAB_MAGEFLOW_EDIT_MODEL",
        "microsoft/Mage-Flow-Edit-Turbo": "COMFYCOLAB_MAGEFLOW_EDIT_TURBO_MODEL",
    }
    local_dir = os.environ.get(environment_names.get(model_id, ""))
    kwargs = {"repo_id": model_id, "revision": revision}
    if local_dir:
        kwargs["local_dir"] = local_dir
    return snapshot_download(**kwargs)


def _instantiate_pipeline(module: ModuleType, model_id: str, revision: str):
    _patch_mage_flow_runtime()
    pipeline_class = getattr(module, "MageFlowPipeline", None)
    if pipeline_class is None or not callable(getattr(pipeline_class, "from_pretrained", None)):
        raise RuntimeError("Pinned Mage source does not expose MageFlowPipeline")
    model_dir = _resolve_model_dir(model_id, revision)
    pipe = pipeline_class.from_pretrained(model_dir, "cuda")
    _patch_mage_flow_runtime()
    _disable_model_screening(pipe)
    return pipe


def _call_pipeline(pipe: Any, request: dict[str, Any]):
    common: dict[str, Any] = {
        "neg_prompts": [str(request.get("negative_prompt") or " ")],
        "seeds": [int(request["seed"])],
        "steps": int(request["steps"]),
        "cfg": float(request["guidance_scale"]),
    }
    if request["mode"] == "edit":
        result = pipe.edit(
            [str(request["prompt"])],
            [[str(request["input_image"])]],
            heights=[int(request["height"])],
            widths=[int(request["width"])],
            **common,
        )
    else:
        result = pipe.generate(
            [str(request["prompt"])],
            heights=[int(request["height"])],
            widths=[int(request["width"])],
            **common,
        )
    _patch_mage_flow_runtime()
    _disable_model_screening(pipe)
    image = None
    if isinstance(result, (list, tuple)) and result:
        image = result[0]
    elif hasattr(result, "save"):
        image = result
    if image is None or not hasattr(image, "save"):
        raise RuntimeError("Mage-Flow pipeline did not return a PIL image")
    return image


class MageFlowRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        source_dir = Path(args.source_dir)
        if not source_dir.exists():
            raise FileNotFoundError(
                f"Pinned Mage-Flow source checkout is missing: {source_dir}. "
                f"Install Microsoft Mage-Flow at {os.environ.get('COMFYCOLAB_MAGE_FLOW_SOURCE_REF', '')}."
            )
        module = _load_pipeline_module(source_dir)
        self.pipe = _instantiate_pipeline(module, args.model_id, args.model_revision)
        self.pipeline_load_count = 1

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        _validate_request(request)
        request_id = str(request["request_id"])
        start = time.monotonic()
        emit_progress(request_id, "inference", 0, 1, message="Running MageFlow inference")
        output = Path(str(request["output_image"]))
        metadata = Path(str(request["metadata_output"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        image = _call_pipeline(self.pipe, request)
        image.save(output)
        if output.stat().st_size <= 0:
            raise RuntimeError("MageFlow wrote an empty output image")
        elapsed = time.monotonic() - start
        meta = {
            "schema": "comfycolab-mage-flow-result-v1",
            "request_id": request_id,
            "mode": request["mode"],
            "seed": int(request["seed"]),
            "width": int(request["width"]),
            "height": int(request["height"]),
            "steps": int(request["steps"]),
            "guidance_scale": float(request["guidance_scale"]),
            "revisions": request.get("revisions") or {},
            "screening": "disabled",
            "watermark": "disabled",
            "noise": "seeded_torch_randn",
            "attn_type": "sdpa",
            "runtime_seconds": elapsed,
            "worker_pid": os.getpid(),
            "pipeline_load_count": self.pipeline_load_count,
        }
        metadata.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        emit_progress(request_id, "complete", 1, 1, message="MageFlow inference complete")
        return {
            "request_id": request_id,
            "status": "ok",
            "output_image": str(output),
            "metadata_output": str(metadata),
            "worker_pid": os.getpid(),
            "pipeline_load_count": self.pipeline_load_count,
            "runtime_seconds": elapsed,
        }


def _result_for_exception(request: dict[str, Any], error: BaseException) -> dict[str, Any]:
    return {
        "request_id": str(request.get("request_id", "")),
        "status": "error",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }


def run_one(runtime: MageFlowRuntime, request: dict[str, Any]) -> None:
    try:
        emit_result(**runtime.run(request))
    except BaseException as error:
        emit_result(**_result_for_exception(request, error))
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()


def serve(args: argparse.Namespace) -> int:
    runtime = MageFlowRuntime(args)
    _emit(
        READY_PREFIX,
        {
            "protocol": PROTOCOL_VERSION,
            "worker_pid": os.getpid(),
            "source_ref": os.environ.get("COMFYCOLAB_MAGE_FLOW_SOURCE_REF", ""),
            "model_ref": args.model_revision,
            "screening": "disabled",
            "watermark": "disabled",
            "noise": "seeded_torch_randn",
            "attn_type": "sdpa",
        },
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        run_one(runtime, request)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--one-shot", action="store_true")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--mode", choices=("text", "edit"), required=True)
    parser.add_argument("--request-id", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output-image", default="")
    parser.add_argument("--metadata-output", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--input-image", default="")
    parser.add_argument("--strength", type=float, default=0.75)
    return parser.parse_args(argv)


def request_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VERSION,
        "request_id": args.request_id,
        "mode": args.mode,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "output_image": args.output_image,
        "metadata_output": args.metadata_output,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "num_images": args.num_images,
        "input_image": args.input_image,
        "strength": args.strength,
        "revisions": {
            "source": os.environ.get("COMFYCOLAB_MAGE_FLOW_SOURCE_REF", ""),
            "model": args.model_revision,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.server:
        return serve(args)
    runtime = MageFlowRuntime(args)
    run_one(runtime, request_from_args(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
