#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ZIMAGE_CATALOG_DIR = ROOT / "custom_nodes" / "ComfyColab-ZImage" / "catalog"
LTX_CATALOG_PATH = (
    ROOT
    / "custom_nodes"
    / "ComfyColab-LTXVideo"
    / "catalog"
    / "ltx_2_3.json"
)
RESOLVE_PATTERN = re.compile(
    r"^/(?P<repo>[^/]+/[^/]+)/resolve/(?P<revision>[0-9a-f]{40})/(?P<path>.+)$"
)
NEXT_LINK_PATTERN = re.compile(r'<(?P<url>[^>]+)>;\s*rel="next"')


def specifications(catalog: dict[str, object]):
    selections = catalog["selections"]
    assert isinstance(selections, dict)
    yield from selections.items()
    yield "text_encoder", catalog["text_encoder"]
    yield "vae", catalog["vae"]


def catalog_entries():
    for catalog_path in sorted(ZIMAGE_CATALOG_DIR.glob("*.json")):
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        selections = catalog["selections"]
        assert isinstance(selections, dict)
        component_by_name = {
            **{name: "model" for name in selections},
            "text_encoder": "text_encoder",
            "vae": "vae",
        }
        for name, specification in specifications(catalog):
            assert isinstance(specification, dict)
            yield catalog_path, name, component_by_name[name], specification

    ltx_catalog = json.loads(LTX_CATALOG_PATH.read_text(encoding="utf-8"))
    for group in ("ggufs", "shared", "spatial_upscalers"):
        assets = ltx_catalog[group]
        assert isinstance(assets, dict)
        for name, specification in assets.items():
            if specification is None:
                continue
            assert isinstance(specification, dict)
            yield (
                LTX_CATALOG_PATH,
                f"{group}.{name}",
                str(specification["folder_key"]),
                specification,
            )


def model_tree(repo: str, revision: str) -> dict[str, dict[str, object]]:
    url: str | None = (
        f"https://huggingface.co/api/models/{repo}/tree/{revision}"
        "?recursive=true&expand=true"
    )
    entries: dict[str, dict[str, object]] = {}
    while url is not None:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "ComfyColab catalog verifier"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            page = json.load(response)
            link = response.headers.get("Link", "")
        entries.update({entry["path"]: entry for entry in page})
        match = NEXT_LINK_PATTERN.search(link)
        url = match.group("url") if match is not None else None
    return entries


def main() -> None:
    trees: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    destinations: dict[tuple[str, str], tuple[str, str]] = {}
    checked = 0
    for catalog_path, name, component, specification in catalog_entries():
        catalog_name = str(catalog_path.relative_to(ROOT))
        destination_key = (component, str(specification["filename"]))
        destination_value = (str(specification["sha256"]), catalog_name)
        previous = destinations.get(destination_key)
        if previous is not None and previous[0] != destination_value[0]:
            raise RuntimeError(
                f"{catalog_name}:{name} collides with {previous[1]} using "
                f"different weights at {destination_key[1]}"
            )
        destinations[destination_key] = destination_value
        parsed = urllib.parse.urlsplit(str(specification["url"]))
        match = RESOLVE_PATTERN.match(parsed.path)
        if match is None:
            raise RuntimeError(
                f"{catalog_name}:{name} does not use a pinned Hugging Face URL"
            )

        repo = match.group("repo")
        revision = match.group("revision")
        file_path = urllib.parse.unquote(match.group("path"))
        key = (repo, revision)
        if key not in trees:
            trees[key] = model_tree(repo, revision)

        try:
            remote = trees[key][file_path]
            remote_lfs = remote["lfs"]
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                f"{catalog_name}:{name} is missing from {repo}@{revision}"
            ) from error

        if remote_lfs["oid"] != specification["sha256"]:
            raise RuntimeError(f"{catalog_name}:{name} has the wrong SHA-256")
        if remote_lfs["size"] != specification["size_bytes"]:
            raise RuntimeError(f"{catalog_name}:{name} has the wrong byte size")
        checked += 1

    print(f"Verified {checked} pinned catalog files against Hugging Face metadata.")


if __name__ == "__main__":
    main()
