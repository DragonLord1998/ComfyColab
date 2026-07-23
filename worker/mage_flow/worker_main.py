#!/usr/bin/env python3
"""Persistent process-isolated runner for pinned Microsoft Mage-Flow models."""

from __future__ import annotations

import argparse
import contextlib
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
    valid_modes = {
        "text",
        "edit",
        "vae_encode",
        "native_denoise",
        "native_vae_encode",
        "native_vae_decode",
    }
    if request.get("mode") not in valid_modes:
        raise ValueError(f"Unsupported MageFlow mode: {request.get('mode')}")
    seed = int(request.get("seed", -1))
    if seed < 0 or seed > (2**31) - 1:
        raise ValueError("MageFlow seed must be between 0 and 2147483647")
    for name in ("request_id", "prompt", "output_image", "metadata_output"):
        if request.get(name) is None:
            raise ValueError(f"MageFlow request omitted {name}")
    width, height = int(request.get("width", 0)), int(request.get("height", 0))
    minimum = (
        16
        if request.get("mode")
        in {"vae_encode", "native_denoise", "native_vae_encode", "native_vae_decode"}
        else 256
    )
    if width < minimum or height < minimum or width > 2048 or height > 2048:
        raise ValueError(
            f"MageFlow {request.get('mode')} width and height must be "
            f"between {minimum} and 2048"
        )
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
    if request.get("mode") in {"edit", "vae_encode"}:
        input_image = Path(str(request.get("input_image", "")))
        if not input_image.is_file():
            raise FileNotFoundError(
                f"MageFlow {request.get('mode')} input does not exist: {input_image}"
            )
    if request.get("mode") in {
        "native_denoise",
        "native_vae_encode",
        "native_vae_decode",
    }:
        input_latent = Path(str(request.get("input_latent", "")))
        if not input_latent.is_file():
            raise FileNotFoundError(
                f"MageFlow {request.get('mode')} input does not exist: {input_latent}"
            )
    if request.get("mode") == "native_vae_decode" and not str(
        request.get("output_tensor", "")
    ):
        raise ValueError("MageFlow native_vae_decode request omitted output_tensor")
    if (
        request.get("mode") == "native_denoise"
        and "Edit" in str(request.get("model_id", ""))
    ):
        input_image = Path(str(request.get("input_image", "")))
        if not input_image.is_file():
            raise FileNotFoundError(
                f"MageFlow native edit input does not exist: {input_image}"
            )
    if request.get("mode") in {
        "vae_encode",
        "native_vae_encode",
        "native_vae_decode",
    }:
        if not str(request.get("output_latent", "")):
            if request.get("mode") != "native_vae_decode":
                raise ValueError(
                    f"MageFlow {request.get('mode')} request omitted output_latent"
                )
        tile_size = int(request.get("tile_size", 0))
        tile_overlap = int(request.get("tile_overlap", 0))
        if tile_size < 64 or tile_size > 4096 or tile_size % 16:
            raise ValueError(
                "MageFlow tile_size must be a multiple of 16 between 64 and 4096"
            )
        if tile_overlap < 0 or tile_overlap >= tile_size or tile_overlap % 16:
            raise ValueError(
                "MageFlow tile_overlap must be a non-negative multiple of 16 "
                "smaller than tile_size"
            )
    if request.get("mode") == "native_denoise":
        prompts = request.get("prompts")
        sigmas = request.get("sigmas")
        if (
            not isinstance(prompts, list)
            or not isinstance(sigmas, list)
            or not prompts
            or len(prompts) != len(sigmas)
        ):
            raise ValueError(
                "MageFlow native_denoise requires one prompt and sigma per latent batch"
            )
        if not str(request.get("output_latent", "")):
            raise ValueError("MageFlow native_denoise request omitted output_latent")


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


def _instantiate_mage_vae(source_dir: Path, checkpoint: str):
    import torch

    source = source_dir / "mage_flow" / "models" / "modules" / "mage_vae.py"
    if not source.is_file():
        raise FileNotFoundError(f"Pinned Mage-VAE source is missing: {source}")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pinned Mage-VAE checkpoint is missing: {checkpoint_path}")
    module_name = "_comfycolab_pinned_mage_vae"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Mage-VAE source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    vae = module.MageVAE(str(checkpoint_path), sample_posterior=False)
    vae.decoder_model = None
    vae.to(device="cuda", dtype=torch.bfloat16)
    vae.eval()
    return vae


def _tile_positions(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    positions = list(range(0, length - tile + 1, stride))
    final = length - tile
    if positions[-1] != final:
        positions.append(final)
    return positions


def _axis_blend(
    torch,
    length: int,
    overlap: int,
    *,
    starts_at_edge: bool,
    ends_at_edge: bool,
    device,
) -> Any:
    weights = torch.ones(length, device=device, dtype=torch.float32)
    ramp_length = min(overlap, length // 2)
    if ramp_length:
        ramp = torch.linspace(0.0, 1.0, ramp_length + 2, device=device)[1:-1]
        if not starts_at_edge:
            weights[:ramp_length] = ramp
        if not ends_at_edge:
            weights[-ramp_length:] = ramp.flip(0)
    return weights


def _encode_mage_vae_tensor(vae: Any, pixels: Any, request: dict[str, Any]):
    import numpy as np
    import torch

    width = int(request["width"])
    height = int(request["height"])
    image = pixels.detach().cpu() if hasattr(pixels, "detach") else torch.as_tensor(
        np.asarray(pixels)
    )
    if image.ndim != 4 or image.shape[-1] not in {1, 3, 4}:
        raise ValueError("Mage-VAE encode expects a BHWC image tensor")
    image = image[..., :3].float()
    if image.shape[-1] == 1:
        image = image.repeat(1, 1, 1, 3)
    image = image.permute(0, 3, 1, 2)
    if tuple(image.shape[-2:]) != (height, width):
        image = torch.nn.functional.interpolate(
            image,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
    image = image.mul(2.0).sub(1.0)
    image = image.to(device=vae.device, dtype=vae.dtype, memory_format=torch.contiguous_format)

    tile_size = int(request["tile_size"])
    overlap = int(request["tile_overlap"])
    if height <= tile_size and width <= tile_size:
        with torch.no_grad():
            return vae.encode(image).contiguous()

    scale = int(vae.downsample_factor)
    tile_height = min(tile_size, height)
    tile_width = min(tile_size, width)
    y_positions = _tile_positions(height, tile_height, overlap)
    x_positions = _tile_positions(width, tile_width, overlap)
    latent_height = height // scale
    latent_width = width // scale
    accumulator = torch.zeros(
        (image.shape[0], int(vae.latent_channels), latent_height, latent_width),
        device=vae.device,
        dtype=torch.float32,
    )
    weights = torch.zeros(
        (1, 1, latent_height, latent_width),
        device=vae.device,
        dtype=torch.float32,
    )
    overlap_latent = overlap // scale
    for y in y_positions:
        for x in x_positions:
            tile = image[:, :, y : y + tile_height, x : x + tile_width]
            with torch.no_grad():
                encoded = vae.encode(tile).float()
            ly, lx = y // scale, x // scale
            lh, lw = encoded.shape[-2:]
            wy = _axis_blend(
                torch,
                lh,
                overlap_latent,
                starts_at_edge=y == 0,
                ends_at_edge=y + tile_height == height,
                device=vae.device,
            )
            wx = _axis_blend(
                torch,
                lw,
                overlap_latent,
                starts_at_edge=x == 0,
                ends_at_edge=x + tile_width == width,
                device=vae.device,
            )
            blend = wy[:, None] * wx[None, :]
            blend = blend[None, None]
            accumulator[:, :, ly : ly + lh, lx : lx + lw] += encoded * blend
            weights[:, :, ly : ly + lh, lx : lx + lw] += blend
    return (accumulator / weights.clamp_min(1e-6)).to(dtype=vae.dtype).contiguous()


def _encode_mage_vae(vae: Any, image_path: Path, request: dict[str, Any]):
    import numpy as np
    import torch
    from PIL import Image

    pil = Image.open(image_path).convert("RGB")
    array = np.asarray(pil, dtype=np.float32) / 255.0
    pixels = torch.from_numpy(array).unsqueeze(0)
    return _encode_mage_vae_tensor(vae, pixels, request)


def _decode_mage_vae_tensor(vae: Any, latent_path: Path, request: dict[str, Any]):
    import torch

    payload = torch.load(latent_path, map_location="cpu", weights_only=True)
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, torch.Tensor) or samples.ndim != 4:
        raise RuntimeError("Mage-Flow native VAE decode received an invalid latent")
    samples = samples.to(device=vae.device, dtype=vae.dtype)
    tile_size = int(request["tile_size"])
    overlap = int(request["tile_overlap"])
    scale = int(vae.downsample_factor)
    latent_tile = max(1, tile_size // scale)
    latent_overlap = overlap // scale
    latent_height, latent_width = samples.shape[-2:]

    def decode(tile):
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if getattr(vae.device, "type", None) == "cuda"
            else contextlib.nullcontext()
        )
        with torch.no_grad(), autocast:
            return vae.decode(tile.float()).float()

    if latent_height <= latent_tile and latent_width <= latent_tile:
        decoded = decode(samples)
    else:
        tile_height = min(latent_tile, latent_height)
        tile_width = min(latent_tile, latent_width)
        y_positions = _tile_positions(latent_height, tile_height, latent_overlap)
        x_positions = _tile_positions(latent_width, tile_width, latent_overlap)
        output_height = latent_height * scale
        output_width = latent_width * scale
        accumulator = torch.zeros(
            (samples.shape[0], 3, output_height, output_width),
            device=vae.device,
            dtype=torch.float32,
        )
        weights = torch.zeros(
            (1, 1, output_height, output_width),
            device=vae.device,
            dtype=torch.float32,
        )
        output_overlap = latent_overlap * scale
        for y in y_positions:
            for x in x_positions:
                tile = samples[:, :, y : y + tile_height, x : x + tile_width]
                decoded_tile = decode(tile)
                py, px = y * scale, x * scale
                ph, pw = decoded_tile.shape[-2:]
                wy = _axis_blend(
                    torch,
                    ph,
                    output_overlap,
                    starts_at_edge=y == 0,
                    ends_at_edge=y + tile_height == latent_height,
                    device=vae.device,
                )
                wx = _axis_blend(
                    torch,
                    pw,
                    output_overlap,
                    starts_at_edge=x == 0,
                    ends_at_edge=x + tile_width == latent_width,
                    device=vae.device,
                )
                blend = (wy[:, None] * wx[None, :])[None, None]
                accumulator[:, :, py : py + ph, px : px + pw] += decoded_tile * blend
                weights[:, :, py : py + ph, px : px + pw] += blend
        decoded = accumulator / weights.clamp_min(1e-6)
    return decoded.clamp(-1, 1).add(1.0).mul(0.5).permute(0, 2, 3, 1).cpu()


def _native_velocity(
    pipe: Any,
    module: ModuleType,
    request: dict[str, Any],
    native_cache: dict[Any, Any],
):
    import torch

    payload = torch.load(
        Path(str(request["input_latent"])),
        map_location="cpu",
        weights_only=True,
    )
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, torch.Tensor) or samples.ndim != 4:
        raise RuntimeError("Mage-Flow native sampler received an invalid latent")
    prompts = [str(value) for value in request["prompts"]]
    sigmas = [float(value) for value in request["sigmas"]]
    if samples.shape[0] != len(prompts):
        raise ValueError("Mage-Flow prompt batch does not match the latent batch")
    if max(sigmas) - min(sigmas) > 1e-6:
        raise ValueError("Mage-Flow native sampler requires a shared batch sigma")

    model = pipe.model
    device = next(model.transformer.parameters()).device
    dtype = next(model.transformer.parameters()).dtype
    samples = samples.to(device=device, dtype=dtype)
    batch, channels, latent_height, latent_width = samples.shape
    target_length = latent_height * latent_width
    target_tokens = samples.permute(0, 2, 3, 1).reshape(
        batch, latent_height * latent_width, channels
    )

    ids = torch.zeros(latent_height, latent_width, 3, device=device, dtype=dtype)
    ids[..., 1] = torch.arange(latent_height, device=device, dtype=dtype)[:, None]
    ids[..., 2] = torch.arange(latent_width, device=device, dtype=dtype)[None, :]
    target_ids = ids.reshape(1, target_length, 3)
    is_edit = "Edit" in str(request.get("model_id", ""))
    template_name = "mage-flow-edit" if is_edit else "mage-flow"
    template_info = module._template_info(template_name)

    if is_edit:
        from PIL import Image

        reference_path = Path(str(request["input_image"]))
        reference_key = (
            "reference",
            str(reference_path.resolve()),
            reference_path.stat().st_mtime_ns,
            latent_height,
            latent_width,
        )
        cached_reference = native_cache.get(reference_key)
        if cached_reference is None:
            reference_pil = Image.open(reference_path).convert("RGB")
            reference_tensor = module._preprocess_ref_image(
                reference_pil,
                latent_height * 16,
                latent_width * 16,
                device,
            )
            ref_tokens, ref_shapes, ref_ids = model.compute_vae_encodings(
                [reference_tensor],
                with_ids=True,
            )
            cached_reference = (
                reference_pil,
                ref_tokens.to(device=device, dtype=dtype),
                ref_shapes,
                ref_ids.to(device=device, dtype=dtype),
            )
            native_cache[reference_key] = cached_reference
        reference_pil, ref_tokens, ref_shapes, ref_ids = cached_reference
        reference_length = int(ref_tokens.shape[1])
        lengths = [target_length + reference_length] * batch
        packed_tokens = []
        packed_ids = []
        shapes = []
        target_indices = []
        offset = 0
        for index in range(batch):
            packed_tokens.extend((target_tokens[index : index + 1], ref_tokens))
            packed_ids.extend((target_ids, ref_ids))
            shapes.append((1, latent_height, latent_width))
            shapes.extend(shape[0] for shape in ref_shapes)
            target_indices.append(
                torch.arange(offset, offset + target_length, device=device)
            )
            offset += target_length + reference_length
        image_tokens = torch.cat(packed_tokens, dim=1)
        image_ids = torch.cat(packed_ids, dim=1)
        target_index = torch.cat(target_indices)
        edit_refs = [[module._resize_long_edge(reference_pil, 384)] for _ in prompts]
        text_key = (
            "edit_text",
            reference_key,
            tuple(prompts),
        )
        cached_text = native_cache.get(text_key)
        if cached_text is None:
            cached_text = module._encode_edits_packed(
                model,
                edit_refs,
                prompts,
                template_info["template"],
                template_info["start_idx"],
                device,
            )
            native_cache[text_key] = cached_text
        image_shapes = [shapes]
    else:
        lengths = [target_length] * batch
        image_tokens = target_tokens.reshape(
            1,
            batch * target_length,
            channels,
        )
        image_ids = target_ids.repeat(1, batch, 1)
        target_index = None
        image_shapes = [[(1, latent_height, latent_width)] * batch]
        text_key = ("text", template_name, tuple(prompts))
        cached_text = native_cache.get(text_key)
        if cached_text is None:
            cached_text = module._encode_texts_packed(
                model,
                prompts,
                template_info["template"],
                template_info["start_idx"],
                device,
            )
            native_cache[text_key] = cached_text

    image_cu = module._lens_to_cu(lengths, device)
    text_flat, vector, text_lengths = cached_text
    text, text_cu, text_mask, vector = module._slice_packed(
        text_flat,
        vector,
        text_lengths,
        0,
        batch,
        device,
    )
    context = module._build_pack_ctx(
        image_ids,
        image_cu,
        image_shapes,
        lengths,
        text,
        text_cu,
        text_mask,
        vector,
        None,
        None,
        None,
        None,
        1.0,
        False,
        False,
        device,
    )
    with torch.no_grad():
        velocity = module._velocity(
            model.transformer,
            image_tokens,
            context,
            sigmas[0],
        )
    if target_index is not None:
        velocity = velocity[:, target_index, :]
    return velocity.reshape(
        batch, latent_height, latent_width, channels
    ).permute(0, 3, 1, 2).contiguous()


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
        self.module = None
        if args.mode == "vae_encode":
            self.pipe = _instantiate_mage_vae(source_dir, args.model_id)
        else:
            self.module = _load_pipeline_module(source_dir)
            self.pipe = _instantiate_pipeline(
                self.module,
                args.model_id,
                args.model_revision,
            )
        self.pipeline_load_count = 1
        self.native_cache: dict[Any, Any] = {}

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        _validate_request(request)
        request_id = str(request["request_id"])
        start = time.monotonic()
        emit_progress(request_id, "inference", 0, 1, message="Running MageFlow inference")
        latent_output_modes = {
            "vae_encode",
            "native_denoise",
            "native_vae_encode",
        }
        is_latent_output = request["mode"] in latent_output_modes
        if is_latent_output:
            output_key = "output_latent"
        elif request["mode"] == "native_vae_decode":
            output_key = "output_tensor"
        else:
            output_key = "output_image"
        output = Path(str(request[output_key]))
        metadata = Path(str(request["metadata_output"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        if request["mode"] == "vae_encode":
            import torch

            latent = _encode_mage_vae(
                self.pipe,
                Path(str(request["input_image"])),
                request,
            )
            torch.save({"samples": latent.detach().cpu()}, output)
        elif request["mode"] == "native_vae_encode":
            import torch

            payload = torch.load(
                Path(str(request["input_latent"])),
                map_location="cpu",
                weights_only=True,
            )
            pixels = payload.get("pixels") if isinstance(payload, dict) else None
            if not isinstance(pixels, torch.Tensor):
                raise RuntimeError(
                    "Mage-Flow native VAE encode received an invalid image batch"
                )
            latent = _encode_mage_vae_tensor(
                self.pipe.model.vae,
                pixels,
                request,
            )
            torch.save({"samples": latent.detach().cpu()}, output)
        elif request["mode"] == "native_vae_decode":
            import torch

            images = _decode_mage_vae_tensor(
                self.pipe.model.vae,
                Path(str(request["input_latent"])),
                request,
            )
            torch.save({"images": images}, output)
        elif request["mode"] == "native_denoise":
            import torch

            if self.module is None:
                raise RuntimeError("Mage-Flow pipeline module is unavailable")
            velocity = _native_velocity(
                self.pipe,
                self.module,
                request,
                self.native_cache,
            )
            torch.save({"samples": velocity.detach().cpu()}, output)
        else:
            image = _call_pipeline(self.pipe, request)
            image.save(output)
        if output.stat().st_size <= 0:
            raise RuntimeError("MageFlow wrote an empty output artifact")
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
        if request["mode"] in {"vae_encode", "native_vae_encode"}:
            meta.update(
                {
                    "schema": "comfycolab-mage-vae-latent-v1",
                    "latent_channels": int(latent.shape[1]),
                    "latent_height": int(latent.shape[2]),
                    "latent_width": int(latent.shape[3]),
                    "posterior": "mean",
                    "tile_size": int(request["tile_size"]),
                    "tile_overlap": int(request["tile_overlap"]),
                }
            )
        metadata.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        emit_progress(request_id, "complete", 1, 1, message="MageFlow inference complete")
        result = {
            "request_id": request_id,
            "status": "ok",
            "metadata_output": str(metadata),
            "worker_pid": os.getpid(),
            "pipeline_load_count": self.pipeline_load_count,
            "runtime_seconds": elapsed,
        }
        result[output_key] = str(output)
        return result


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
    parser.add_argument(
        "--mode",
        choices=("text", "edit", "vae_encode", "native"),
        required=True,
    )
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
    parser.add_argument("--input-latent", default="")
    parser.add_argument("--output-latent", default="")
    parser.add_argument("--output-tensor", default="")
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--tile-size", type=int, default=1536)
    parser.add_argument("--tile-overlap", type=int, default=384)
    parser.add_argument("--prompts-json", default="[]")
    parser.add_argument("--sigmas-json", default="[]")
    return parser.parse_args(argv)


def request_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VERSION,
        "request_id": args.request_id,
        "model_id": args.model_id,
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
        "input_latent": args.input_latent,
        "output_latent": args.output_latent,
        "output_tensor": args.output_tensor,
        "prompts": json.loads(args.prompts_json),
        "sigmas": json.loads(args.sigmas_json),
        "strength": args.strength,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
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
