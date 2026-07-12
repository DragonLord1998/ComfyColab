from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


CATALOG_PATH = Path(__file__).with_name("catalog") / "z_image_turbo.json"


class CatalogError(RuntimeError):
    pass


def _validate_file(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogError(f"Catalog field '{name}' must be an object.")
    required = {"filename", "url", "sha256"}
    missing = sorted(required - value.keys())
    if missing:
        raise CatalogError(f"Catalog field '{name}' is missing: {', '.join(missing)}")
    if not str(value["url"]).startswith("https://huggingface.co/"):
        raise CatalogError(f"Catalog field '{name}' must use a Hugging Face HTTPS URL.")
    sha256 = str(value["sha256"])
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise CatalogError(f"Catalog field '{name}' has an invalid SHA-256 value.")
    return value


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    try:
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogError(f"Unable to read Z-Image catalog: {error}") from error

    if catalog.get("schema_version") != 1:
        raise CatalogError("Unsupported Z-Image catalog schema version.")
    quantizations = catalog.get("quantizations")
    if not isinstance(quantizations, dict) or not quantizations:
        raise CatalogError("The Z-Image catalog has no quantizations.")
    for quantization, value in quantizations.items():
        _validate_file(f"quantizations.{quantization}", value)
    _validate_file("text_encoder", catalog.get("text_encoder"))
    _validate_file("vae", catalog.get("vae"))
    return catalog


def quantization_names() -> list[str]:
    return list(load_catalog()["quantizations"])


def bundle_for(quantization: str) -> dict[str, dict[str, Any]]:
    catalog = load_catalog()
    try:
        model = catalog["quantizations"][quantization]
    except KeyError as error:
        raise CatalogError(f"Unknown Z-Image Turbo quantization: {quantization}") from error
    return {
        "model": model,
        "text_encoder": catalog["text_encoder"],
        "vae": catalog["vae"],
    }
