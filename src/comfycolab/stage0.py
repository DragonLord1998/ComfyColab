from __future__ import annotations

import base64
from pathlib import Path

from .config import CoreStage0ConfigV1


CONFIG_MARKER = "__COMFYCOLAB_STAGE0_CONFIG_B64__"


def render_stage0(config: CoreStage0ConfigV1) -> str:
    template = Path(__file__).with_name("stage0_runtime.py").read_text(encoding="utf-8")
    if template.count(CONFIG_MARKER) != 1:
        raise RuntimeError("Stage-0 template marker is missing or duplicated.")
    encoded = base64.b64encode(config.validated().canonical_bytes()).decode("ascii")
    return template.replace(CONFIG_MARKER, encoded)
