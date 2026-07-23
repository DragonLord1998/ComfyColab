from __future__ import annotations

import hashlib
import json
import uuid
import zlib
from pathlib import Path
from typing import Any

from .mage_flow_worker import MageFlowWorkerCommand, global_mage_flow_worker_pool


_PROMPT_BYTES = 4096
_PROMPT_HEADER_BYTES = 6


def _encode_prompt(text: str):
    import torch

    payload = text.encode("utf-8")
    if len(payload) > _PROMPT_BYTES:
        raise ValueError(
            f"Mage-Flow prompt is {len(payload)} bytes; maximum is {_PROMPT_BYTES}"
        )
    values = torch.zeros((_PROMPT_BYTES + _PROMPT_HEADER_BYTES,), dtype=torch.float32)
    values[0] = len(payload) & 0xFF
    values[1] = (len(payload) >> 8) & 0xFF
    checksum = zlib.crc32(payload)
    for index in range(4):
        values[2 + index] = (checksum >> (8 * index)) & 0xFF
    if payload:
        start = _PROMPT_HEADER_BYTES
        values[start : start + len(payload)] = torch.tensor(
            list(payload),
            dtype=torch.float32,
        )
    return values.reshape(1, _PROMPT_BYTES + _PROMPT_HEADER_BYTES, 1)


def _decode_prompt(values) -> str:
    raw = values.detach().cpu().float().reshape(-1)
    length = int(round(float(raw[0]))) + (int(round(float(raw[1]))) << 8)
    if length < 0 or length > _PROMPT_BYTES:
        raise ValueError(
            "Mage-Flow conditioning was modified and no longer contains a valid prompt"
        )
    checksum = sum(
        int(round(float(raw[2 + index]))) << (8 * index)
        for index in range(4)
    )
    start = _PROMPT_HEADER_BYTES
    integers = [int(round(float(value))) for value in raw[start : start + length]]
    if any(value < 0 or value > 255 for value in integers):
        raise ValueError(
            "Mage-Flow conditioning was modified and no longer contains a valid prompt"
        )
    try:
        payload = bytes(integers)
        if zlib.crc32(payload) != checksum:
            raise ValueError(
                "Mage-Flow conditioning was modified by an unsupported conditioning transform"
            )
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            "Mage-Flow conditioning was modified by an unsupported conditioning transform"
        ) from error


class MageFlowTextEncoder:
    """Small CLIP-compatible prompt carrier.

    Text embeddings stay inside the isolated Mage worker. ComfyUI only carries a
    stable prompt identifier through its normal CONDITIONING graph.
    """

    def clone(self):
        return self

    def tokenize(self, text: str, **_kwargs: Any) -> dict[str, Any]:
        return {"mage_flow_prompt": str(text)}

    def encode_from_tokens_scheduled(
        self,
        tokens: dict[str, Any],
        *_args: Any,
        **_kwargs: Any,
    ):
        import torch

        prompt = str(tokens["mage_flow_prompt"])
        conditioning = _encode_prompt(prompt)
        return [[conditioning, {}]]


class MageFlowLatentFormat:
    scale_factor = 1.0
    latent_channels = 128
    latent_dimensions = 2
    latent_rgb_factors = None
    latent_rgb_factors_bias = None
    latent_rgb_factors_reshape = None
    taesd_decoder_name = None
    spacial_downscale_ratio = 16
    temporal_downscale_ratio = 1

    @staticmethod
    def process_in(latent):
        return latent

    @staticmethod
    def process_out(latent):
        return latent


class _MageFlowModelConfig:
    unet_config = {"disable_unet_model_creation": True}
    latent_format = MageFlowLatentFormat()
    manual_cast_dtype = None
    custom_operations = None
    optimizations: dict[str, Any] = {}
    sampling_settings = {"shift": 6.0, "multiplier": 1000}
    memory_usage_factor = 0.0


class MageFlowModel:
    def __new__(cls, component: "MageFlowComponents"):
        import comfy.model_base
        import comfy.model_patcher
        import comfy.model_management
        import torch

        class WorkerBackedModel(comfy.model_base.BaseModel):
            def __init__(self):
                super().__init__(
                    _MageFlowModelConfig(),
                    model_type=comfy.model_base.ModelType.FLOW,
                    device=torch.device("cpu"),
                )
                self.component = component
                self.worker_anchor = torch.nn.Parameter(
                    torch.zeros((), dtype=torch.float32),
                    requires_grad=False,
                )

            def get_dtype(self):
                return torch.bfloat16

            def memory_required(self, input_shape, cond_shapes={}):
                del input_shape, cond_shapes
                return 0

            def _apply_model(
                self,
                x,
                t,
                c_concat=None,
                c_crossattn=None,
                control=None,
                transformer_options={},
                **kwargs,
            ):
                del c_concat, control, transformer_options, kwargs
                if c_crossattn is None:
                    raise ValueError(
                        "Mage-Flow MODEL requires conditioning from its exposed "
                        "text encoder through CLIP Text Encode."
                    )
                prompts = tuple(
                    _decode_prompt(c_crossattn[index])
                    for index in range(c_crossattn.shape[0])
                )
                velocity = self.component.denoise(x, t, prompts)
                return self.model_sampling.calculate_denoised(t, velocity, x)

        model = WorkerBackedModel()
        patcher = comfy.model_patcher.ModelPatcher(
            model,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=torch.device("cpu"),
            size=0,
        )
        patcher.disable_model_cfg1_optimization()
        return patcher


class MageFlowVAE:
    latent_channels = 128
    downscale_ratio = 16
    upscale_ratio = 16
    latent_dim = 2
    output_channels = 3

    def __init__(self, component: "MageFlowComponents") -> None:
        self.component = component

    def encode(self, pixels):
        return self.component.encode(pixels)

    def encode_tiled(self, pixels, tile_x=1536, tile_y=1536, overlap=384, **_kwargs):
        return self.component.encode(
            pixels,
            tile_size=min(int(tile_x), int(tile_y)),
            tile_overlap=int(overlap),
        )

    def decode(self, samples, **_kwargs):
        return self.component.decode(samples)

    def decode_tiled(
        self,
        samples,
        tile_x=32,
        tile_y=32,
        overlap=4,
        **_kwargs,
    ):
        return self.component.decode(
            samples,
            tile_size=min(int(tile_x), int(tile_y)) * 16,
            tile_overlap=int(overlap) * 16,
        )

    @staticmethod
    def spacial_compression_encode():
        return 16

    @staticmethod
    def spacial_compression_decode():
        return 16

    @staticmethod
    def temporal_compression_decode():
        return None

    @staticmethod
    def temporal_compression_encode():
        return None


class MageFlowComponents:
    def __init__(
        self,
        *,
        variant: str,
        model_id: str,
        model_revision: str,
        source_ref: str,
        source_dir: Path,
        worker_script: Path,
        python: str,
        site_packages: str,
        runtime_root: Path,
        keep_worker_loaded: bool,
        reference_image: Any = None,
    ) -> None:
        self.variant = variant
        self.model_id = model_id
        self.model_revision = model_revision
        self.source_ref = source_ref
        self.source_dir = source_dir
        self.worker_script = worker_script
        self.python = python
        self.site_packages = site_packages
        self.runtime_root = runtime_root
        self.keep_worker_loaded = keep_worker_loaded
        self.reference_image = self._persist_reference_image(reference_image)

    def _persist_reference_image(self, image: Any) -> str:
        if image is None:
            return ""
        import numpy as np

        from .nodes import _image_tensor_to_png

        tensor = image.detach().cpu() if hasattr(image, "detach") else image
        array = tensor.contiguous().numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
        if array.ndim == 4 and array.shape[0] != 1:
            raise ValueError(
                "Mage-Flow Edit sampler components require exactly one reference image"
            )
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        path = self.runtime_root / "native" / "references" / f"{digest}.png"
        if not path.is_file():
            _image_tensor_to_png(image, path)
        return str(path)

    def objects(self):
        return MageFlowModel(self), MageFlowTextEncoder(), MageFlowVAE(self)

    def _command(self, mode: str, **values: Any) -> MageFlowWorkerCommand:
        return MageFlowWorkerCommand(
            python=self.python,
            worker_script=str(self.worker_script),
            source_dir=str(self.source_dir),
            model_id=self.model_id,
            model_revision=self.model_revision,
            mode=mode,
            prompt=str(values.get("prompt", "")),
            negative_prompt="",
            output_image=str(values.get("output_image", "")),
            output_latent=str(values.get("output_latent", "")),
            output_tensor=str(values.get("output_tensor", "")),
            input_latent=str(values.get("input_latent", "")),
            metadata_output=str(values["metadata_output"]),
            request_id=uuid.uuid4().hex,
            seed=0,
            width=int(values["width"]),
            height=int(values["height"]),
            steps=1,
            guidance_scale=1.0,
            input_image=str(values.get("input_image", "")),
            tile_size=int(values.get("tile_size", 1536)),
            tile_overlap=int(values.get("tile_overlap", 384)),
            prompts=tuple(values.get("prompts", ())),
            sigmas=tuple(float(value) for value in values.get("sigmas", ())),
            source_ref=self.source_ref,
            keep_worker_loaded=self.keep_worker_loaded,
            site_packages=self.site_packages,
        )

    def denoise(self, samples, sigmas, prompts: tuple[str, ...]):
        import torch

        height = int(samples.shape[-2]) * 16
        width = int(samples.shape[-1]) * 16
        key = hashlib.sha256(
            json.dumps(
                {
                    "mode": "native_denoise",
                    "variant": self.variant,
                    "shape": list(samples.shape),
                    "sigmas": [float(value) for value in sigmas.detach().cpu().flatten()],
                    "prompts": list(prompts),
                },
                sort_keys=True,
            ).encode("utf-8")
            + samples.detach().cpu().float().contiguous().numpy().tobytes()
        ).hexdigest()
        case_dir = self.runtime_root / "native" / key
        input_latent = case_dir / "input.pt"
        output_latent = case_dir / "output.pt"
        metadata = case_dir / "metadata.json"
        case_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"samples": samples.detach().cpu()}, input_latent)
        command = self._command(
            "native_denoise",
            input_latent=input_latent,
            input_image=self.reference_image,
            output_latent=output_latent,
            metadata_output=metadata,
            width=width,
            height=height,
            prompts=prompts,
            sigmas=tuple(float(value) for value in sigmas.detach().cpu().flatten()),
        )
        try:
            global_mage_flow_worker_pool().run(command)
            payload = torch.load(output_latent, map_location="cpu", weights_only=True)
            return payload["samples"].to(device=samples.device, dtype=samples.dtype)
        finally:
            for path in (input_latent, output_latent, metadata):
                path.unlink(missing_ok=True)
            try:
                case_dir.rmdir()
            except OSError:
                pass

    def encode(self, pixels, *, tile_size: int = 1536, tile_overlap: int = 384):
        from .nodes import _load_latent, _request_key

        width, height = int(pixels.shape[2]), int(pixels.shape[1])
        width = max(16, ((width + 15) // 16) * 16)
        height = max(16, ((height + 15) // 16) * 16)
        key = _request_key(
            "native_vae_encode",
            model_variant=self.variant,
            image=pixels,
            width=width,
            height=height,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            model_id=self.model_id,
            model_revision=self.model_revision,
            source_ref=self.source_ref,
        )
        case_dir = self.runtime_root / "native" / key
        input_tensor = case_dir / "input.pt"
        output_latent = case_dir / "latent.pt"
        metadata = case_dir / "metadata.json"
        if output_latent.is_file() and metadata.is_file():
            return _load_latent(output_latent)["samples"]
        case_dir.mkdir(parents=True, exist_ok=True)
        import torch

        torch.save({"pixels": pixels.detach().cpu()}, input_tensor)
        command = self._command(
            "native_vae_encode",
            input_latent=input_tensor,
            output_latent=output_latent,
            metadata_output=metadata,
            width=width,
            height=height,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        global_mage_flow_worker_pool().run(command)
        return _load_latent(output_latent)["samples"]

    def decode(
        self,
        samples,
        *,
        tile_size: int = 1536,
        tile_overlap: int = 384,
    ):
        import torch

        from .nodes import _request_key

        width = int(samples.shape[-1]) * 16
        height = int(samples.shape[-2]) * 16
        key = _request_key(
            "native_vae_decode",
            model_variant=self.variant,
            latent=samples,
            width=width,
            height=height,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            model_id=self.model_id,
            model_revision=self.model_revision,
            source_ref=self.source_ref,
        )
        case_dir = self.runtime_root / "native" / key
        input_latent = case_dir / "input.pt"
        output_tensor = case_dir / "image.pt"
        metadata = case_dir / "metadata.json"
        if output_tensor.is_file() and metadata.is_file():
            payload = torch.load(output_tensor, map_location="cpu", weights_only=True)
            return payload["images"]
        case_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"samples": samples.detach().cpu()}, input_latent)
        command = self._command(
            "native_vae_decode",
            input_latent=input_latent,
            output_tensor=output_tensor,
            metadata_output=metadata,
            width=width,
            height=height,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        global_mage_flow_worker_pool().run(command)
        payload = torch.load(output_tensor, map_location="cpu", weights_only=True)
        return payload["images"]


def build_components(**kwargs: Any) -> tuple[Any, Any, Any]:
    return MageFlowComponents(**kwargs).objects()
