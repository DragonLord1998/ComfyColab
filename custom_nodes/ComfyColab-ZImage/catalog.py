from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


CATALOG_DIR = Path(__file__).with_name("catalog")
CATALOG_PATHS = {
    "z_image_turbo": CATALOG_DIR / "z_image_turbo.json",
    "qwen_image_edit_2511": CATALOG_DIR / "qwen_image_edit_2511.json",
    "krea_2": CATALOG_DIR / "krea_2.json",
}


class CatalogError(RuntimeError):
    pass


def _validate_file(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogError(f"Catalog field '{name}' must be an object.")
    required = {"filename", "url", "sha256", "size_bytes"}
    missing = sorted(required - value.keys())
    if missing:
        raise CatalogError(f"Catalog field '{name}' is missing: {', '.join(missing)}")
    if not str(value["url"]).startswith("https://huggingface.co/"):
        raise CatalogError(f"Catalog field '{name}' must use a Hugging Face HTTPS URL.")
    sha256 = str(value["sha256"])
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise CatalogError(f"Catalog field '{name}' has an invalid SHA-256 value.")
    if not isinstance(value["size_bytes"], int) or value["size_bytes"] <= 0:
        raise CatalogError(f"Catalog field '{name}' has an invalid byte size.")
    return value


@lru_cache(maxsize=len(CATALOG_PATHS))
def load_catalog(name: str = "z_image_turbo") -> dict[str, Any]:
    try:
        path = CATALOG_PATHS[name]
    except KeyError as error:
        raise CatalogError(f"Unknown catalog: {name}") from error
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogError(f"Unable to read '{name}' catalog: {error}") from error

    if catalog.get("schema_version") != 1:
        raise CatalogError(f"Unsupported '{name}' catalog schema version.")
    selections = catalog.get("selections")
    if not isinstance(selections, dict) or not selections:
        raise CatalogError(f"The '{name}' catalog has no selections.")
    default_selection = catalog.get("default_selection") or catalog.get(
        "default_quantization"
    )
    if default_selection not in selections:
        raise CatalogError(f"The '{name}' catalog has an invalid default selection.")
    for selection, value in selections.items():
        _validate_file(f"selections.{selection}", value)
    _validate_file("text_encoder", catalog.get("text_encoder"))
    _validate_file("vae", catalog.get("vae"))
    return catalog


def quantization_names() -> list[str]:
    return selection_names("z_image_turbo")


def qwen_edit_quantization_names() -> list[str]:
    return selection_names("qwen_image_edit_2511")


def krea_2_variants() -> list[str]:
    return selection_names("krea_2")


def selection_names(catalog_name: str) -> list[str]:
    return list(load_catalog(catalog_name)["selections"])


def bundle_for(quantization: str) -> dict[str, dict[str, Any]]:
    return selected_bundle("z_image_turbo", quantization)


def qwen_edit_bundle_for(quantization: str) -> dict[str, dict[str, Any]]:
    return selected_bundle("qwen_image_edit_2511", quantization)


def krea_2_bundle_for(variant: str) -> dict[str, dict[str, Any]]:
    return selected_bundle("krea_2", variant)


def selected_bundle(catalog_name: str, selection: str) -> dict[str, dict[str, Any]]:
    catalog = load_catalog(catalog_name)
    try:
        model = catalog["selections"][selection]
    except KeyError as error:
        raise CatalogError(
            f"Unknown selection '{selection}' for {catalog['family']}."
        ) from error
    return {
        "model": model,
        "text_encoder": catalog["text_encoder"],
        "vae": catalog["vae"],
    }
