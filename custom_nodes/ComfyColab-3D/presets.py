from __future__ import annotations

from dataclasses import dataclass


CACHE_MODES = ("Use cache", "Refresh this node", "Disable cache")
RESOLUTION_OVERRIDES = (
    "Auto",
    "512",
    "1024",
    "1024_cascade",
    "1536_cascade",
)


@dataclass(frozen=True)
class TrellisSettings:
    resolution: str
    sampling_steps: int
    target_face_count: int
    texture_size: int
    max_tokens: int


@dataclass(frozen=True)
class UltraShapeSettings:
    steps: int
    num_latents: int
    decode_chunk_size: int
    octree_resolution: int


@dataclass(frozen=True)
class Pixal3DSettings:
    pipeline_type: str
    sampling_steps: int
    target_face_count: int
    texture_size: int
    max_tokens: int
    low_vram: bool


TRELLIS_PRESETS = {
    "512 — Fast": TrellisSettings("512", 10, 200_000, 1024, 49_152),
    "1024 — Quality": TrellisSettings("1024_cascade", 12, 500_000, 2048, 49_152),
    "1536 — Maximum": TrellisSettings("1536_cascade", 12, 750_000, 4096, 49_152),
}

ULTRASHAPE_PRESETS = {
    "Fast": UltraShapeSettings(12, 8192, 2048, 512),
    "Conservative": UltraShapeSettings(24, 16_384, 4096, 512),
    "Detailed": UltraShapeSettings(24, 16_384, 4096, 1024),
    "Ultra": UltraShapeSettings(50, 32_768, 4096, 1024),
}

ULTRASHAPE_EXPERIMENTAL_PRESETS = frozenset({"Detailed", "Ultra"})

PIXAL3D_PRESETS = {
    "1024 — Stable": Pixal3DSettings("1024_cascade", 12, 200_000, 2048, 49_152, True),
    "1536 — Experimental": Pixal3DSettings("1536_cascade", 12, 1_000_000, 4096, 49_152, True),
}

PIXAL3D_EXPERIMENTAL_PRESETS = frozenset({"1536 — Experimental"})


def resolve_trellis_settings(
    quality: str,
    *,
    resolution: str = "Auto",
    sampling_steps: int = 0,
    target_face_count: int = 0,
    texture_size: int = 0,
    max_tokens: int = 0,
) -> TrellisSettings:
    try:
        preset = TRELLIS_PRESETS[quality]
    except KeyError as exc:
        raise ValueError(f"Unknown TRELLIS.2 quality preset: {quality}") from exc
    if resolution not in RESOLUTION_OVERRIDES:
        raise ValueError(f"Unsupported TRELLIS.2 resolution: {resolution}")
    if sampling_steps > 50:
        raise ValueError("sampling_steps must be 0 for the preset or between 1 and 50")
    if 0 < target_face_count < 1000:
        raise ValueError("target_face_count must be 0 for the preset or at least 1000")
    if 0 < texture_size < 512:
        raise ValueError("texture_size must be 0 for the preset or at least 512")
    values = TrellisSettings(
        preset.resolution if resolution == "Auto" else resolution,
        sampling_steps or preset.sampling_steps,
        target_face_count or preset.target_face_count,
        texture_size or preset.texture_size,
        max_tokens or preset.max_tokens,
    )
    if min(values.sampling_steps, values.target_face_count, values.texture_size, values.max_tokens) <= 0:
        raise ValueError("TRELLIS.2 numeric settings must be positive or zero for preset defaults")
    return values


def resolve_ultrashape_settings(
    detail: str,
    *,
    steps: int = 0,
    num_latents: int = 0,
    decode_chunk_size: int = 0,
    octree_resolution: int = 0,
) -> UltraShapeSettings:
    try:
        preset = ULTRASHAPE_PRESETS[detail]
    except KeyError as exc:
        raise ValueError(f"Unknown UltraShape detail preset: {detail}") from exc
    values = UltraShapeSettings(
        steps or preset.steps,
        num_latents or preset.num_latents,
        decode_chunk_size or preset.decode_chunk_size,
        octree_resolution or preset.octree_resolution,
    )
    if min(values.steps, values.num_latents, values.decode_chunk_size, values.octree_resolution) <= 0:
        raise ValueError("UltraShape numeric settings must be positive or zero for preset defaults")
    return values


def resolve_pixal3d_settings(
    quality: str,
    *,
    sampling_steps: int = 0,
    target_face_count: int = 0,
    texture_size: int = 0,
    max_tokens: int = 0,
) -> Pixal3DSettings:
    try:
        preset = PIXAL3D_PRESETS[quality]
    except KeyError as exc:
        raise ValueError(f"Unknown Pixal3D quality preset: {quality}") from exc
    if sampling_steps < 0 or sampling_steps > 100:
        raise ValueError("sampling_steps must be 0 for the preset or between 1 and 100")
    if 0 < target_face_count < 1000:
        raise ValueError("target_face_count must be 0 for the preset or at least 1000")
    if 0 < texture_size < 512:
        raise ValueError("texture_size must be 0 for the preset or at least 512")
    if max_tokens < 0 or 0 < max_tokens < 16_384:
        raise ValueError("max_tokens must be 0 for the preset or at least 16384")
    values = Pixal3DSettings(
        pipeline_type=preset.pipeline_type,
        sampling_steps=sampling_steps or preset.sampling_steps,
        target_face_count=target_face_count or preset.target_face_count,
        texture_size=texture_size or preset.texture_size,
        max_tokens=max_tokens or preset.max_tokens,
        low_vram=True,
    )
    if min(values.sampling_steps, values.target_face_count, values.texture_size, values.max_tokens) <= 0:
        raise ValueError("Pixal3D numeric settings must be positive or zero for preset defaults")
    if values.pipeline_type == "1536_cascade" and values.max_tokens < preset.max_tokens:
        raise ValueError(
            "1536 — Experimental requires max_tokens >= 49152. "
            "Use 1024 — Stable or set a larger explicit token cap."
        )
    return values
