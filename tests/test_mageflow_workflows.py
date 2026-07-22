from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"

EXPECTED_WORKFLOWS = {
    "comfycolab_mageflow.json": "ComfyColabMageFlow",
    "comfycolab_mageflow_turbo.json": "ComfyColabMageFlowTurbo",
    "comfycolab_mageflow_edit.json": "ComfyColabMageFlowEdit",
    "comfycolab_mageflow_edit_turbo.json": "ComfyColabMageFlowEditTurbo",
}
EDIT_NODE_IDS = {"ComfyColabMageFlowEdit", "ComfyColabMageFlowEditTurbo"}
FORBIDDEN_TEXT = (
    "content_screen",
    "content-screen",
    "prompt_screen",
    "prompt-screen",
    "safety_checker",
    "moderation",
    "gaussian_shading",
    "gaussian-shading",
    "watermark",
    "watermark_key",
)


class MageFlowWorkflowTests(unittest.TestCase):
    def _workflow(self, filename: str):
        path = WORKFLOWS / filename
        raw = path.read_text(encoding="utf-8")
        lowered = raw.lower()
        for token in FORBIDDEN_TEXT:
            self.assertNotIn(token, lowered)
        return json.loads(raw)

    def test_exactly_four_public_mageflow_workflows_are_present(self):
        actual = sorted(path.name for path in WORKFLOWS.glob("comfycolab_mageflow*.json"))
        self.assertEqual(actual, sorted(EXPECTED_WORKFLOWS))

    def test_workflows_use_one_public_facade_and_standard_image_outputs(self):
        for filename, facade_id in EXPECTED_WORKFLOWS.items():
            with self.subTest(filename=filename):
                workflow = self._workflow(filename)
                nodes = {node["id"]: node for node in workflow["nodes"]}
                types = [node["type"] for node in nodes.values()]

                self.assertEqual(types.count(facade_id), 1)
                self.assertIn("PreviewImage", types)
                self.assertIn("SaveImage", types)
                self.assertFalse(any("Base" in node_type for node_type in types))
                self.assertFalse(any("Screen" in node_type for node_type in types))
                self.assertFalse(any("Watermark" in node_type for node_type in types))

                for link_id, source_id, source_slot, target_id, target_slot, socket_type in workflow["links"]:
                    self.assertGreater(link_id, 0)
                    self.assertIn(source_id, nodes)
                    self.assertIn(target_id, nodes)
                    self.assertGreaterEqual(source_slot, 0)
                    self.assertGreaterEqual(target_slot, 0)
                    self.assertIsInstance(socket_type, str)

                facade = next(node for node in nodes.values() if node["type"] == facade_id)
                preview = next(node for node in nodes.values() if node["type"] == "PreviewImage")
                save = next(node for node in nodes.values() if node["type"] == "SaveImage")
                facade_outputs = facade["outputs"]
                linked_preview_inputs = {
                    item["name"] for item in preview["inputs"] if item.get("link") is not None
                }
                linked_save_inputs = {
                    item["name"] for item in save["inputs"] if item.get("link") is not None
                }

                self.assertEqual(facade_outputs[0]["type"], "IMAGE")
                self.assertIn("images", linked_preview_inputs)
                self.assertIn("images", linked_save_inputs)

                input_names = {item["name"] for item in facade["inputs"]}
                self.assertIn("prompt", input_names)
                self.assertIn("seed", input_names)
                self.assertNotIn("content_screening", input_names)
                self.assertNotIn("gaussian_shading", input_names)
                self.assertNotIn("watermark", input_names)
                if facade_id in EDIT_NODE_IDS:
                    self.assertEqual(types.count("LoadImage"), 1)
                    self.assertIn("image", input_names)
                    image_input = next(item for item in facade["inputs"] if item["name"] == "image")
                    self.assertIsNotNone(image_input.get("link"))
                else:
                    self.assertNotIn("LoadImage", types)
                    self.assertNotIn("image", input_names)

    def test_workflow_defaults_match_seeded_gaussian_noise_contract(self):
        for filename, facade_id in EXPECTED_WORKFLOWS.items():
            with self.subTest(filename=filename):
                workflow = self._workflow(filename)
                facade = next(node for node in workflow["nodes"] if node["type"] == facade_id)
                widget_values = facade["widgets_values"]
                widget_names = [
                    item["widget"]["name"] for item in facade["inputs"] if "widget" in item
                ]
                widgets = dict(zip(widget_names, widget_values))

                self.assertEqual(widgets["seed"], 0)
                self.assertGreaterEqual(widgets["steps"], 1)
                self.assertIn("guidance", widgets)
                self.assertTrue(str(widgets["prompt"]).strip())
                self.assertNotIn("watermark", widgets)
                self.assertNotIn("content_screening", widgets)


if __name__ == "__main__":
    unittest.main()
