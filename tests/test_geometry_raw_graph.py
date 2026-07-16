from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "custom_nodes" / "ComfyColab-3D"
PACKAGE_NAME = "comfycolab_geometry_raw_graph_test"


def load_graph_module():
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(PACKAGE_DIR)]
    sys.modules[PACKAGE_NAME] = package
    for name in ("cache", "presets", "graph"):
        qualified_name = f"{PACKAGE_NAME}.{name}"
        spec = importlib.util.spec_from_file_location(
            qualified_name,
            PACKAGE_DIR / f"{name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        assert spec.loader
        spec.loader.exec_module(module)
    return sys.modules[f"{PACKAGE_NAME}.graph"]


class FakeNode:
    def __init__(self, index, class_type, inputs):
        self.index = index
        self.class_type = class_type
        self.inputs = inputs

    def out(self, slot):
        return [str(self.index), slot]

    def set_override_display_id(self, value):
        self.override_display_id = value


class FakeGraph:
    def __init__(self):
        self.nodes = []

    def node(self, class_type, **inputs):
        node = FakeNode(len(self.nodes), class_type, inputs)
        self.nodes.append(node)
        return node

    def finalize(self):
        return [
            {"class_type": node.class_type, "inputs": node.inputs}
            for node in self.nodes
        ]


class RawGeometryGraphTests(unittest.TestCase):
    def setUp(self):
        self.saved_modules = {
            name: sys.modules.get(name)
            for name in ("comfy_api", "comfy_api.latest")
        }
        comfy_api = types.ModuleType("comfy_api")
        latest = types.ModuleType("comfy_api.latest")
        latest.io = types.SimpleNamespace(
            NodeOutput=lambda value, expand=None: types.SimpleNamespace(
                result=value,
                expand=expand,
            )
        )
        comfy_api.latest = latest
        sys.modules["comfy_api"] = comfy_api
        sys.modules["comfy_api.latest"] = latest

    def tearDown(self):
        for module_name in list(sys.modules):
            if module_name == PACKAGE_NAME or module_name.startswith(PACKAGE_NAME + "."):
                del sys.modules[module_name]
        for name, module in self.saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_only_raw_trellis_shape_uses_bounded_memory_analysis(self):
        graph_module = load_graph_module()
        fake_graph = FakeGraph()
        settings = types.SimpleNamespace(
            resolution="512",
            sampling_steps=12,
            max_tokens=49_152,
            target_face_count=25_000,
            texture_size=1024,
        )
        with mock.patch.object(graph_module, "_builder", return_value=fake_graph):
            result = graph_module.build_trellis_graph(
                object(),
                settings,
                seed=0,
                remove_background="Off",
                cache_mode="Refresh",
                cache_key="test-cache-key",
            )

        validators = [
            node
            for node in result.expand
            if node["class_type"] == "ComfyColab3DValidateMesh"
        ]
        self.assertEqual(len(validators), 3)
        self.assertEqual(validators[0]["inputs"]["stage"], "TRELLIS raw shape")
        self.assertEqual(validators[0]["inputs"]["analysis_mode"], "raw")
        self.assertNotIn("analysis_mode", validators[1]["inputs"])
        self.assertNotIn("analysis_mode", validators[2]["inputs"])


if __name__ == "__main__":
    unittest.main()
