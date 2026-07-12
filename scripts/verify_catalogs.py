#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = ROOT / "custom_nodes" / "ComfyColab-ZImage" / "catalog"
RESOLVE_PATTERN = re.compile(
    r"^/(?P<repo>[^/]+/[^/]+)/resolve/(?P<revision>[0-9a-f]{40})/(?P<path>.+)$"
)


def specifications(catalog: dict[str, object]):
    selections = catalog["selections"]
    assert isinstance(selections, dict)
    yield from selections.items()
    yield "text_encoder", catalog["text_encoder"]
    yield "vae", catalog["vae"]


def main() -> None:
    trees: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    checked = 0
    for catalog_path in sorted(CATALOG_DIR.glob("*.json")):
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        for name, raw_specification in specifications(catalog):
            assert isinstance(raw_specification, dict)
            specification = raw_specification
            parsed = urllib.parse.urlsplit(str(specification["url"]))
            match = RESOLVE_PATTERN.match(parsed.path)
            if match is None:
                raise RuntimeError(
                    f"{catalog_path.name}:{name} does not use a pinned Hugging Face URL"
                )

            repo = match.group("repo")
            revision = match.group("revision")
            file_path = urllib.parse.unquote(match.group("path"))
            key = (repo, revision)
            if key not in trees:
                api_url = (
                    f"https://huggingface.co/api/models/{repo}/tree/{revision}"
                    "?recursive=true&expand=true"
                )
                request = urllib.request.Request(
                    api_url,
                    headers={"User-Agent": "ComfyColab catalog verifier"},
                )
                with urllib.request.urlopen(request, timeout=30) as response:
                    entries = json.load(response)
                trees[key] = {entry["path"]: entry for entry in entries}

            try:
                remote = trees[key][file_path]
                remote_lfs = remote["lfs"]
            except (KeyError, TypeError) as error:
                raise RuntimeError(
                    f"{catalog_path.name}:{name} is missing from {repo}@{revision}"
                ) from error

            if remote_lfs["oid"] != specification["sha256"]:
                raise RuntimeError(f"{catalog_path.name}:{name} has the wrong SHA-256")
            if remote_lfs["size"] != specification["size_bytes"]:
                raise RuntimeError(f"{catalog_path.name}:{name} has the wrong byte size")
            checked += 1

    print(f"Verified {checked} pinned catalog files against Hugging Face metadata.")


if __name__ == "__main__":
    main()
