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


if __name__ == "__main__":
    unittest.main()
