from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows" / "comfycolab_triposplat_image_to_gaussian_splat.json"


class TripoSplatWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        self.nodes = {node["id"]: node for node in self.workflow["nodes"]}
        self.links = {link[0]: link for link in self.workflow["links"]}

    def test_workflow_has_expected_node_ids_and_types(self) -> None:
        self.assertEqual(self.workflow["last_node_id"], 6)
        self.assertEqual(self.workflow["last_link_id"], 5)
        self.assertEqual(
            {node_id: node["type"] for node_id, node in self.nodes.items()},
            {
                1: "LoadImage",
                2: "ComfyColabTripoSplatImageToGaussianSplat",
                3: "RenderSplat",
                4: "CreateVideo",
                5: "SaveVideo",
                6: "Preview3D",
            },
        )

    def test_links_form_image_to_splat_preview_video_and_file_3d_preview(self) -> None:
        self.assertEqual(self.workflow["links"], [
            [1, 1, 0, 2, 0, "IMAGE"],
            [2, 2, 0, 3, 0, "SPLAT"],
            [3, 3, 0, 4, 0, "IMAGE"],
            [4, 4, 0, 5, 0, "VIDEO"],
            [5, 2, 1, 6, 0, "FILE_3D_SPLAT_ANY"],
        ])

        for link_id, source_id, source_slot, target_id, target_slot, socket_type in self.workflow["links"]:
            self.assertIn(source_id, self.nodes)
            self.assertIn(target_id, self.nodes)
            self.assertGreaterEqual(source_slot, 0)
            self.assertGreaterEqual(target_slot, 0)
            self.assertIsInstance(socket_type, str)
            self.assertEqual(self.links[link_id][5], socket_type)

    def test_triposplat_defaults_and_output_types_are_bundled_for_preview(self) -> None:
        triposplat = self.nodes[2]
        self.assertEqual(
            [output["type"] for output in triposplat["outputs"]],
            ["SPLAT", "FILE_3D_SPLAT_ANY"],
        )
        self.assertEqual(triposplat["outputs"][0]["links"], [2])
        self.assertEqual(triposplat["outputs"][1]["links"], [5])
        self.assertEqual(
            triposplat["widgets_values"],
            ["Quality — 262K", 0, True, 20, 3.0, True, "ply"],
        )
        self.assertEqual(
            [item["name"] for item in triposplat["inputs"]],
            [
                "image",
                "quality",
                "seed",
                "remove_background",
                "sampling_steps",
                "guidance_scale",
                "enable_sampling_preview",
                "output_format",
            ],
        )

        linked_inputs = {
            node["type"]: {item["name"] for item in node["inputs"] if item.get("link") is not None}
            for node in self.nodes.values()
        }
        self.assertEqual(linked_inputs["ComfyColabTripoSplatImageToGaussianSplat"], {"image"})
        self.assertEqual(linked_inputs["RenderSplat"], {"splat"})
        self.assertEqual(linked_inputs["CreateVideo"], {"images"})
        self.assertEqual(linked_inputs["SaveVideo"], {"video"})
        self.assertEqual(linked_inputs["Preview3D"], {"model_file"})


if __name__ == "__main__":
    unittest.main()
