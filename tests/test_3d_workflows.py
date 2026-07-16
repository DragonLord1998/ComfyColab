from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"


class ThreeDWorkflowTests(unittest.TestCase):
    def test_simple_workflows_are_valid_and_connect_to_native_3d_outputs(self) -> None:
        paths = [
            WORKFLOWS / "comfycolab_trellis_image_to_3d.json",
            WORKFLOWS / "comfycolab_ultrashape_refine.json",
            WORKFLOWS / "comfycolab_pixal3d_image_to_3d.json",
        ]
        for path in paths:
            workflow = json.loads(path.read_text(encoding="utf-8"))
            nodes = {node["id"]: node for node in workflow["nodes"]}
            self.assertIn("Preview3D", {node["type"] for node in nodes.values()})
            self.assertIn("SaveGLB", {node["type"] for node in nodes.values()})
            for link_id, source_id, source_slot, target_id, target_slot, socket_type in workflow["links"]:
                self.assertGreater(link_id, 0)
                self.assertIn(source_id, nodes)
                self.assertIn(target_id, nodes)
                self.assertGreaterEqual(source_slot, 0)
                self.assertGreaterEqual(target_slot, 0)
                self.assertIsInstance(socket_type, str)

    def test_refinement_workflow_chains_the_two_public_facades(self) -> None:
        workflow = json.loads(
            (WORKFLOWS / "comfycolab_ultrashape_refine.json").read_text(encoding="utf-8")
        )
        types = [node["type"] for node in workflow["nodes"]]
        self.assertEqual(types.count("ComfyColabTrellisImageTo3D"), 1)
        self.assertEqual(types.count("ComfyColabUltraShapeRefine"), 1)
        ultra = next(node for node in workflow["nodes"] if node["type"] == "ComfyColabUltraShapeRefine")
        linked_inputs = {item["name"] for item in ultra["inputs"] if item.get("link") is not None}
        self.assertEqual(linked_inputs, {"model_3d", "reference_image"})
        self.assertEqual(ultra["widgets_values"][0], "Conservative")
        self.assertEqual(
            ultra["widgets_values"][5],
            0,
            "the bundled workflow must resolve the conservative 512 preset instead of forcing 1024",
        )

    def test_pixal3d_workflow_uses_public_facade_preview3d_and_saveglb(self) -> None:
        workflow = json.loads(
            (WORKFLOWS / "comfycolab_pixal3d_image_to_3d.json").read_text(encoding="utf-8")
        )
        types = [node["type"] for node in workflow["nodes"]]

        self.assertEqual(types.count("ComfyColabPixal3DImageTo3D"), 1)
        self.assertIn("Preview3D", types)
        self.assertIn("SaveGLB", types)

        pixal = next(node for node in workflow["nodes"] if node["type"] == "ComfyColabPixal3DImageTo3D")
        preview = next(node for node in workflow["nodes"] if node["type"] == "Preview3D")
        save = next(node for node in workflow["nodes"] if node["type"] == "SaveGLB")
        linked_preview_inputs = {item["name"] for item in preview["inputs"] if item.get("link") is not None}
        linked_save_inputs = {item["name"] for item in save["inputs"] if item.get("link") is not None}

        self.assertEqual(pixal["widgets_values"][0], "1024 — Stable")
        self.assertNotIn("mode", {item["name"] for item in pixal["inputs"]})
        self.assertNotIn("num_views", {item["name"] for item in pixal["inputs"]})
        self.assertIn("model_file", linked_preview_inputs)
        self.assertIn("mesh", linked_save_inputs)


if __name__ == "__main__":
    unittest.main()
