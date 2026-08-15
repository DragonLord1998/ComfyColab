from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"


class H3WorkflowTests(unittest.TestCase):
    def _load(self, filename: str) -> dict:
        return json.loads((WORKFLOWS / filename).read_text(encoding="utf-8"))

    def test_fl2va_workflow_has_one_loader_one_generator_and_save_video(self):
        workflow = self._load("comfycolab_minimax_h3_text_image_to_video.json")
        types = [node["type"] for node in workflow["nodes"]]
        self.assertEqual(types.count("ComfyColabMiniMaxH3BundleLoader"), 1)
        self.assertEqual(types.count("ComfyColabMiniMaxH3PromptEnhancer"), 1)
        self.assertEqual(types.count("ComfyColabMiniMaxH3Video"), 1)
        self.assertEqual(types.count("SaveVideo"), 1)
        self.assertEqual(types.count("LoadImage"), 2)
        loader = next(node for node in workflow["nodes"] if node["type"] == "ComfyColabMiniMaxH3BundleLoader")
        self.assertEqual(loader["widgets_values"][0], "FL2VA — Text / First / Last Frame")
        self.assertFalse(loader["widgets_values"][1])
        enhancer = next(
            node
            for node in workflow["nodes"]
            if node["type"] == "ComfyColabMiniMaxH3PromptEnhancer"
        )
        generator = next(
            node for node in workflow["nodes"] if node["type"] == "ComfyColabMiniMaxH3Video"
        )
        self.assertEqual(enhancer["widgets_values"][1], "T2VA — Text only")
        self.assertEqual(generator["inputs"][1]["link"], 3)
        links = workflow["links"]
        self.assertIn([1, 1, 0, 2, 0, "MINIMAX_H3_BUNDLE"], links)
        self.assertIn([2, 2, 0, 3, 0, "VIDEO"], links)
        self.assertIn([3, enhancer["id"], 0, generator["id"], 1, "STRING"], links)

    def test_ref2va_workflow_has_reference_inputs_and_exact_prompt_tags(self):
        workflow = self._load("comfycolab_minimax_h3_reference_to_video.json")
        types = [node["type"] for node in workflow["nodes"]]
        self.assertEqual(types.count("ComfyColabMiniMaxH3BundleLoader"), 1)
        self.assertEqual(types.count("ComfyColabMiniMaxH3PromptEnhancer"), 1)
        self.assertEqual(types.count("ComfyColabMiniMaxH3ReferenceVideo"), 1)
        self.assertEqual(types.count("SaveVideo"), 1)
        self.assertEqual(types.count("LoadVideo"), 1)
        self.assertEqual(types.count("GetVideoComponents"), 1)
        loader = next(node for node in workflow["nodes"] if node["type"] == "ComfyColabMiniMaxH3BundleLoader")
        self.assertEqual(loader["widgets_values"][0], "Ref2VA — Reference Images / Video / Audio")
        enhancer = next(
            node
            for node in workflow["nodes"]
            if node["type"] == "ComfyColabMiniMaxH3PromptEnhancer"
        )
        generator = next(node for node in workflow["nodes"] if node["type"] == "ComfyColabMiniMaxH3ReferenceVideo")
        load_video = next(node for node in workflow["nodes"] if node["type"] == "LoadVideo")
        components = next(node for node in workflow["nodes"] if node["type"] == "GetVideoComponents")
        self.assertEqual(load_video["inputs"][0]["name"], "file")
        self.assertEqual([output["type"] for output in load_video["outputs"]], ["VIDEO"])
        self.assertEqual(
            [output["type"] for output in components["outputs"]],
            ["IMAGE", "AUDIO", "FLOAT", "INT"],
        )
        input_names = [input_["name"] for input_ in generator["inputs"]]
        self.assertIn("ref_images.ref_image_0", input_names)
        self.assertIn("ref_videos.ref_video_0", input_names)
        self.assertIn("ref_video_audios.ref_video_audio_0", input_names)
        self.assertIn("ref_audios.ref_audio_0", input_names)
        self.assertEqual(enhancer["widgets_values"][1], "Ref2VA — Full references")
        prompt = enhancer["widgets_values"][0]
        self.assertIn("<Picture 1>", prompt)
        self.assertIn("<Video 1>", prompt)
        self.assertIn("<Audio 1>", prompt)
        self.assertIn([1, 1, 0, 2, 0, "MINIMAX_H3_BUNDLE"], workflow["links"])
        self.assertIn([2, 2, 0, 3, 0, "VIDEO"], workflow["links"])
        self.assertIn([3, 4, 0, 2, 8, "IMAGE"], workflow["links"])
        self.assertIn([7, load_video["id"], 0, components["id"], 0, "VIDEO"], workflow["links"])
        self.assertIn([5, components["id"], 0, 2, 9, "IMAGE"], workflow["links"])
        self.assertIn([4, components["id"], 1, 2, 10, "AUDIO"], workflow["links"])
        self.assertIn([6, 6, 0, 2, 11, "AUDIO"], workflow["links"])
        self.assertIn([8, enhancer["id"], 0, generator["id"], 1, "STRING"], workflow["links"])

    def test_fl2va_to_ref2va_chain_workflow_links_frames_and_audio(self):
        workflow = self._load("comfycolab_minimax_h3_fl2va_to_ref2va_chain.json")
        types = [node["type"] for node in workflow["nodes"]]
        self.assertEqual(types.count("ComfyColabMiniMaxH3BundleLoader"), 2)
        self.assertEqual(types.count("ComfyColabMiniMaxH3PromptEnhancer"), 2)
        self.assertEqual(types.count("ComfyColabMiniMaxH3Video"), 1)
        self.assertEqual(types.count("ComfyColabMiniMaxH3ReferenceVideo"), 1)
        self.assertEqual(types.count("SaveVideo"), 1)
        loaders = [
            node
            for node in workflow["nodes"]
            if node["type"] == "ComfyColabMiniMaxH3BundleLoader"
        ]
        self.assertEqual(
            [loader["widgets_values"][0] for loader in loaders],
            [
                "FL2VA — Text / First / Last Frame",
                "Ref2VA — Reference Images / Video / Audio",
            ],
        )
        enhancers = [
            node
            for node in workflow["nodes"]
            if node["type"] == "ComfyColabMiniMaxH3PromptEnhancer"
        ]
        self.assertEqual(
            [enhancer["widgets_values"][1] for enhancer in enhancers],
            ["T2VA — Text only", "Ref2VA — Full references"],
        )
        reference = next(
            node
            for node in workflow["nodes"]
            if node["type"] == "ComfyColabMiniMaxH3ReferenceVideo"
        )
        input_names = [input_["name"] for input_ in reference["inputs"]]
        self.assertIn("ref_videos.ref_video_0", input_names)
        self.assertIn("ref_video_audios.ref_video_audio_0", input_names)
        self.assertIn([1, 1, 0, 2, 0, "MINIMAX_H3_BUNDLE"], workflow["links"])
        self.assertIn([2, 2, 1, 4, 9, "IMAGE"], workflow["links"])
        self.assertIn([3, 2, 2, 4, 10, "AUDIO"], workflow["links"])
        self.assertIn([4, 3, 0, 4, 0, "MINIMAX_H3_BUNDLE"], workflow["links"])
        self.assertIn([5, 4, 0, 5, 0, "VIDEO"], workflow["links"])
        self.assertIn([6, enhancers[0]["id"], 0, 2, 1, "STRING"], workflow["links"])
        self.assertIn([7, enhancers[1]["id"], 0, 4, 1, "STRING"], workflow["links"])


if __name__ == "__main__":
    unittest.main()
