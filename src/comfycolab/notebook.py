from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from .config import CoreStage0ConfigV1
from .stage0 import render_stage0


def render_notebook(config: CoreStage0ConfigV1) -> dict[str, Any]:
    config.validated()
    proxy_source = (
        "from google.colab import output\n"
        f"proxy_url = output.eval_js(\"google.colab.kernel.proxyPort({config.port})\")\n"
        "print(f\"Reserved Colab proxy: {proxy_url}\")\n"
    )
    bootstrap_source = render_stage0(config)
    notebook: dict[str, Any] = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "ComfyColab.ipynb", "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "reserve-comfyui-proxy",
                "metadata": {},
                "outputs": [],
                "source": proxy_source.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "authenticated-comfycolab-bootstrap",
                "metadata": {},
                "outputs": [],
                "source": bootstrap_source.splitlines(keepends=True),
            },
        ],
    }
    for cell in notebook["cells"]:
        ast.parse("".join(cell["source"]))
    return notebook


def notebook_bytes(config: CoreStage0ConfigV1) -> bytes:
    return (
        json.dumps(
            render_notebook(config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def write_notebook(path: Path, config: CoreStage0ConfigV1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(notebook_bytes(config))
    temporary.replace(path)
