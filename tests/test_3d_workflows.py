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
            WORKFLOWS / "comfycolab_trellis2mv_to_3d.json",
            WORKFLOWS / "comfycolab_pixal3dmv_to_3d.json",
            WORKFLOWS / "comfycolab_pixal3dmv_advanced_to_3d.json",
            WORKFLOWS / "comfycolab_skintokens_auto_rig.json",
            WORKFLOWS / "comfycolab_cubepart_segment.json",
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

    def test_new_multiview_rigging_and_segmentation_workflows_use_public_facades(self) -> None:
        expected = {
            "comfycolab_trellis2mv_to_3d.json": "ComfyColabTrellis2MV",
            "comfycolab_pixal3dmv_to_3d.json": "ComfyColabPixal3DMV",
            "comfycolab_pixal3dmv_advanced_to_3d.json": "ComfyColabPixal3DMVAdvanced",
            "comfycolab_skintokens_auto_rig.json": "ComfyColabSkinTokensAutoRig",
            "comfycolab_cubepart_segment.json": "ComfyColabCubePartSegment",
        }
        for filename, facade in expected.items():
            workflow = json.loads((WORKFLOWS / filename).read_text(encoding="utf-8"))
            types = [node["type"] for node in workflow["nodes"]]
            self.assertEqual(types.count(facade), 1)
            self.assertIn("Preview3D", types)
            self.assertIn("SaveGLB", types)

        cube = json.loads(
            (WORKFLOWS / "comfycolab_cubepart_segment.json").read_text(encoding="utf-8")
        )
        node = next(item for item in cube["nodes"] if item["type"] == "ComfyColabCubePartSegment")
        self.assertFalse(node["widgets_values"][1], "example must not pre-accept research terms")

        advanced = json.loads(
            (WORKFLOWS / "comfycolab_pixal3dmv_advanced_to_3d.json").read_text(
                encoding="utf-8"
            )
        )
        node = next(
            item
            for item in advanced["nodes"]
            if item["type"] == "ComfyColabPixal3DMVAdvanced"
        )
        input_names = [item["name"] for item in node["inputs"]]
        self.assertIn("geometry_fallback", input_names)
        self.assertIn("geometry_strength", input_names)
        self.assertIn("max_normalized_alignment_error", input_names)
        widget_names = [
            item["widget"]["name"] for item in node["inputs"] if "widget" in item
        ]
        widget_values = dict(zip(widget_names, node["widgets_values"]))
        self.assertEqual(
            widget_values["geometry_fallback"],
            "Strict — require VGGT-Ω",
        )


if __name__ == "__main__":
    unittest.main()
