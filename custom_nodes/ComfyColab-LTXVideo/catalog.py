from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


CATALOG_PATH = Path(__file__).with_name("catalog") / "ltx_2_3.json"
H3_CATALOG_PATH = Path(__file__).with_name("catalog") / "minimax_h3.json"


class CatalogError(RuntimeError):
    pass


def _validate_file(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogError(f"Catalog field '{name}' must be an object.")
    required = {"filename", "folder_key", "url", "sha256", "size_bytes"}
    missing = sorted(required - value.keys())
    if missing:
        raise CatalogError(f"Catalog field '{name}' is missing: {', '.join(missing)}")
    url = str(value["url"])
    if not url.startswith("https://huggingface.co/") or "/resolve/main/" in url:
        raise CatalogError(
            f"Catalog field '{name}' must use a revision-pinned Hugging Face URL."
        )
    sha256 = str(value["sha256"])
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise CatalogError(f"Catalog field '{name}' has an invalid SHA-256 value.")
    if not isinstance(value["size_bytes"], int) or value["size_bytes"] <= 0:
        raise CatalogError(f"Catalog field '{name}' has an invalid byte size.")
    if not str(value["filename"]) or not str(value["folder_key"]):
        raise CatalogError(f"Catalog field '{name}' has an empty filename or folder.")
    return value


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    try:
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogError(f"Unable to read the LTX-2.3 catalog: {error}") from error
    if catalog.get("schema_version") != 1:
        raise CatalogError("Unsupported LTX-2.3 catalog schema version.")
    ggufs = catalog.get("ggufs")
    shared = catalog.get("shared")
    spatial = catalog.get("spatial_upscalers")
    if not isinstance(ggufs, dict) or not ggufs:
        raise CatalogError("The LTX-2.3 catalog has no GGUF choices.")
    if not isinstance(shared, dict) or not shared:
        raise CatalogError("The LTX-2.3 catalog has no shared assets.")
    if not isinstance(spatial, dict) or set(spatial) != {"None", "1.5x", "2x"}:
        raise CatalogError("The LTX-2.3 spatial choices must be None, 1.5x, and 2x.")
    if catalog.get("default_gguf") not in ggufs:
        raise CatalogError("The LTX-2.3 catalog has an invalid default GGUF.")
    for name, value in ggufs.items():
        _validate_file(f"ggufs.{name}", value)
    for name, value in shared.items():
        _validate_file(f"shared.{name}", value)
    for name, value in spatial.items():
        if name == "None":
            if value is not None:
                raise CatalogError("The None spatial choice must not contain a file.")
        else:
            _validate_file(f"spatial_upscalers.{name}", value)
    return catalog


def gguf_names() -> list[str]:
    return list(load_catalog()["ggufs"])


def spatial_upscaler_names() -> list[str]:
    return list(load_catalog()["spatial_upscalers"])


def selected_assets(
    gguf_model: str,
    spatial_upscaler: str,
    fps: int | str,
) -> dict[str, dict[str, Any]]:
    catalog = load_catalog()
    try:
        model = catalog["ggufs"][gguf_model]
    except KeyError as error:
        raise CatalogError(f"Unknown LTX-2.3 GGUF selection: {gguf_model}") from error
    try:
        spatial = catalog["spatial_upscalers"][spatial_upscaler]
    except KeyError as error:
        raise CatalogError(
            f"Unknown LTX-2.3 spatial upscaler: {spatial_upscaler}"
        ) from error
    fps_value = int(fps)
    if fps_value not in {24, 48}:
        raise CatalogError("LTX-2.3 FPS must be 24 or 48.")
    assets = {"model": model, **catalog["shared"]}
    if spatial is not None:
        assets["spatial_upscaler"] = spatial
    if fps_value != 48:
        assets.pop("temporal_2x")
    return assets


@lru_cache(maxsize=1)
def load_h3_catalog() -> dict[str, Any]:
    try:
        catalog = json.loads(H3_CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogError(f"Unable to read the MiniMax H3 catalog: {error}") from error
    if catalog.get("schema_version") != 1:
        raise CatalogError("Unsupported MiniMax H3 catalog schema version.")
    if catalog.get("repo_id") != "Comfy-Org/MiniMax-H3":
        raise CatalogError("The MiniMax H3 catalog must use the official Comfy-Org repo.")
    revision = str(catalog.get("revision", ""))
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise CatalogError("The MiniMax H3 catalog must pin a 40-character revision.")
    variants = catalog.get("variants")
    shared = catalog.get("shared")
    if not isinstance(variants, dict) or set(variants) != {"FL2VA", "Ref2VA"}:
        raise CatalogError("The MiniMax H3 catalog must contain FL2VA and Ref2VA.")
    if not isinstance(shared, dict) or set(shared) != {
        "text_encoder",
        "video_vae",
        "audio_vae",
    }:
        raise CatalogError("The MiniMax H3 catalog has invalid shared assets.")
    for name, value in variants.items():
        _validate_file(f"h3.variants.{name}", value)
    for name, value in shared.items():
        _validate_file(f"h3.shared.{name}", value)
    return catalog


def h3_variant_names() -> list[str]:
    return list(load_h3_catalog()["variants"])


def h3_assets_for(model_variant: str) -> dict[str, dict[str, Any]]:
    catalog = load_h3_catalog()
    variant_key = normalize_h3_variant(model_variant)
    return {"model": catalog["variants"][variant_key], **catalog["shared"]}


def normalize_h3_variant(model_variant: str) -> str:
    value = str(model_variant)
    if value.startswith("FL2VA"):
        return "FL2VA"
    if value.startswith("Ref2VA"):
        return "Ref2VA"
    raise CatalogError(f"Unknown MiniMax H3 variant: {model_variant}")
