from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows" / "comfycolab_pid_upscale.json"
PUBLIC_NODE_ID = "ComfyColabPiDUpscale"


class PiDWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        self.nodes = {node["id"]: node for node in self.workflow["nodes"]}
        self.links = {link[0]: link for link in self.workflow["links"]}

    def test_workflow_uses_one_public_facade_and_valid_image_links(self):
        types = [node["type"] for node in self.nodes.values()]
        self.assertEqual(types.count(PUBLIC_NODE_ID), 1)
        self.assertEqual(types.count("LoadImage"), 1)
        self.assertEqual(types.count("SaveImage"), 1)

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
        save_image = next(
            node for node in self.nodes.values() if node["type"] == "SaveImage"
        )
        self.assertEqual([output["type"] for output in facade["outputs"]], ["IMAGE"])

        image_input = next(item for item in facade["inputs"] if item["name"] == "image")
        save_input = next(item for item in save_image["inputs"] if item["name"] == "images")
        self.assertIsNotNone(image_input.get("link"))
        self.assertIsNotNone(save_input.get("link"))
        input_link = self.links[image_input["link"]]
        output_link = self.links[save_input["link"]]
        self.assertEqual(input_link[1:4], [load_image["id"], 0, facade["id"]])
        self.assertEqual(input_link[5], "IMAGE")
        self.assertEqual(output_link[1:4], [facade["id"], 0, save_image["id"]])
        self.assertEqual(output_link[5], "IMAGE")

    def test_workflow_defaults_match_the_public_pid_contract(self):
        facade = next(
            node for node in self.nodes.values() if node["type"] == PUBLIC_NODE_ID
        )
        input_names = {item["name"] for item in facade["inputs"]}
        self.assertTrue(
            {
                "image",
                "vae_family",
                "prompt",
                "scale",
                "seed",
                "degrade_sigma",
                "tile_size",
                "tile_overlap",
                "accept_nvidia_noncommercial_license",
            }
            <= input_names
        )

        values = facade["widgets_values"]
        normalized = {str(value) for value in values}
        self.assertIn("FLUX.1", normalized)
        self.assertIn("4x", normalized)
        self.assertIn("12345", normalized)
        self.assertIn("1536", normalized)
        self.assertIn("384", normalized)
        self.assertIn(0.0, values)
        self.assertIn(False, values)
        self.assertTrue(
            any(
                isinstance(value, str)
                and len(value.strip()) >= 20
                and value not in {"FLUX.1", "4x"}
                for value in values
            ),
            "The bundled workflow should include a useful example prompt.",
        )


if __name__ == "__main__":
    unittest.main()
