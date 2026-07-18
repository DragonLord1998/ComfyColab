from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows" / "comfycolab_ltx23_text_image_to_video.json"
PUBLIC_NODE_ID = "ComfyColabLTX23Video"


class LTXWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        self.nodes = {node["id"]: node for node in self.workflow["nodes"]}
        self.links = {link[0]: link for link in self.workflow["links"]}

    def test_workflow_uses_one_public_facade_and_valid_native_video_links(self):
        types = [node["type"] for node in self.nodes.values()]
        self.assertEqual(types.count(PUBLIC_NODE_ID), 1)
        self.assertEqual(types.count("LoadImage"), 1)
        self.assertEqual(types.count("SaveVideo"), 1)

        for link in self.workflow["links"]:
            link_id, source_id, source_slot, target_id, target_slot, socket_type = link
            self.assertGreater(link_id, 0)
            self.assertIn(source_id, self.nodes)
            self.assertIn(target_id, self.nodes)
            self.assertGreaterEqual(source_slot, 0)
            self.assertGreaterEqual(target_slot, 0)
            self.assertIsInstance(socket_type, str)
            self.assertEqual(self.links[link_id], link)

        facade = next(
            node for node in self.nodes.values() if node["type"] == PUBLIC_NODE_ID
        )
        load_image = next(
            node for node in self.nodes.values() if node["type"] == "LoadImage"
        )
        save_video = next(
            node for node in self.nodes.values() if node["type"] == "SaveVideo"
        )
        self.assertEqual(
            [output["type"] for output in facade["outputs"]],
            ["VIDEO", "IMAGE", "AUDIO"],
        )

        image_input = next(
            item for item in facade["inputs"] if item["name"] == "image"
        )
        video_input = next(
            item for item in save_video["inputs"] if item["name"] == "video"
        )
        self.assertTrue(
            {"video", "filename_prefix", "format", "codec"}
            <= {item["name"] for item in save_video["inputs"]}
        )
        self.assertEqual(
            save_video["widgets_values"],
            ["video/ComfyColab_LTX23", "auto", "auto"],
        )
        self.assertIsNone(image_input.get("link"))
        self.assertIsNotNone(video_input.get("link"))
        video_link = self.links[video_input["link"]]
        self.assertIsNone(load_image["outputs"][0]["links"])
        self.assertEqual(video_link[1:4], [facade["id"], 0, save_video["id"]])
        self.assertEqual(video_link[5], "VIDEO")

    def test_workflow_defaults_match_the_public_ltx23_contract(self):
        facade = next(
            node for node in self.nodes.values() if node["type"] == PUBLIC_NODE_ID
        )
        input_names = {item["name"] for item in facade["inputs"]}
        self.assertTrue(
            {
                "prompt",
                "gguf_model",
                "fps",
                "spatial_upscaler",
                "width",
                "height",
                "frame_count",
                "seed",
                "image_strength",
                "image",
            }
            <= input_names
        )

        values = facade["widgets_values"]
        normalized = {str(value) for value in values}
        self.assertIn("Q3_K_S", normalized)
        self.assertIn("24", normalized)
        self.assertIn("2x", normalized)
        self.assertIn("960", normalized)
        self.assertIn("544", normalized)
        self.assertIn("121", normalized)
        self.assertIn("0", normalized)
        self.assertTrue(
            any(abs(float(value) - 0.7) < 1e-9 for value in values if isinstance(value, float))
        )
        self.assertTrue(
            any(
                isinstance(value, str)
                and len(value.strip()) >= 20
                and value not in {"Q3_K_S", "24", "2x"}
                for value in values
            ),
            "The bundled workflow should include a useful example prompt.",
        )


if __name__ == "__main__":
    unittest.main()
