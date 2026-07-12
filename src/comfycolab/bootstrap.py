from __future__ import annotations

import base64
import json
from pathlib import Path


CONFIG_MARKER = "__COMFYCOLAB_CONFIG_B64__"


def render_bootstrap(
    *,
    repository_url: str,
    repository_ref: str,
    port: int,
    refresh: bool = False,
) -> str:
    template_path = Path(__file__).with_name("remote_bootstrap.py")
    template = template_path.read_text(encoding="utf-8")
    if template.count(CONFIG_MARKER) != 1:
        raise RuntimeError("Remote bootstrap template marker is missing or duplicated.")
    config = {
        "repository_url": repository_url,
        "repository_ref": repository_ref,
        "port": port,
        "refresh": refresh,
    }
    encoded = base64.b64encode(
        json.dumps(config, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    return template.replace(CONFIG_MARKER, encoded)
