"""Lazy ComfyUI V3 extension entrypoint for ComfyColab TripoSplat."""

from __future__ import annotations

import importlib


async def comfy_entrypoint():
    latest = importlib.import_module("comfy_api.latest")
    definitions = importlib.import_module(f"{__package__}.nodes")
    node_classes = [
        type(core.__name__, (core, latest.io.ComfyNode), {"__module__": __name__})
        for core in definitions.NODE_CLASS_MAPPINGS.values()
    ]

    class ComfyColabTripoSplatExtension(latest.ComfyExtension):
        async def get_node_list(self):
            return node_classes

    return ComfyColabTripoSplatExtension()


__all__ = ["comfy_entrypoint"]
