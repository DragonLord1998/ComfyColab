from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "worker" / "mage_flow" / "worker_main.py"


def load_worker():
    name = "comfycolab_mageflow_worker_test"
    spec = importlib.util.spec_from_file_location(name, WORKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader
    spec.loader.exec_module(module)
    return module


class WorkerRuntimeTests(unittest.TestCase):
    def test_model_screening_is_replaced_with_allow_verdicts(self):
        worker = load_worker()

        class Encoder:
            def screen_text(self, _prompt):
                raise AssertionError("upstream screen_text must not run")

            def screen_edit(self, _prompt, _images):
                raise AssertionError("upstream screen_edit must not run")

        pipe = types.SimpleNamespace(model=types.SimpleNamespace(txt_enc=Encoder()))
        worker._disable_model_screening(pipe)
        self.assertFalse(pipe.model.txt_enc.screen_text("anything").violates)
        self.assertFalse(pipe.model.txt_enc.screen_edit("anything", []).violates)

    def test_pinned_snapshot_is_resolved_before_pipeline_load(self):
        worker = load_worker()
        pipeline = mock.Mock()
        pipeline.from_pretrained.return_value = types.SimpleNamespace(
            model=types.SimpleNamespace(txt_enc=types.SimpleNamespace())
        )
        module = types.SimpleNamespace(MageFlowPipeline=pipeline)
        with mock.patch.object(worker, "_resolve_model_dir", return_value="/models/pinned") as resolve:
            with mock.patch.object(worker, "_disable_model_screening"):
                loaded = worker._instantiate_pipeline(module, "microsoft/Mage-Flow", "a" * 40)
        resolve.assert_called_once_with("microsoft/Mage-Flow", "a" * 40)
        pipeline.from_pretrained.assert_called_once_with("/models/pinned", "cuda")
        self.assertIs(loaded, pipeline.from_pretrained.return_value)

    def test_pipeline_calls_use_native_mage_api(self):
        worker = load_worker()
        image = mock.Mock()
        image.save = mock.Mock()
        pipe = types.SimpleNamespace(
            generate=mock.Mock(return_value=[image]),
            edit=mock.Mock(return_value=[image]),
            model=types.SimpleNamespace(txt_enc=types.SimpleNamespace()),
        )
        base = {
            "prompt": "test",
            "negative_prompt": "",
            "seed": 7,
            "width": 1024,
            "height": 768,
            "steps": 20,
            "guidance_scale": 5.0,
        }
        with mock.patch.object(worker, "_patch_mage_flow_runtime"), mock.patch.object(
            worker, "_disable_model_screening"
        ):
            self.assertIs(worker._call_pipeline(pipe, {**base, "mode": "text"}), image)
        pipe.generate.assert_called_once_with(
            ["test"],
            neg_prompts=[" "],
            seeds=[7],
            steps=20,
            cfg=5.0,
            heights=[768],
            widths=[1024],
        )

        with tempfile.NamedTemporaryFile(suffix=".png") as reference:
            edit_request = {**base, "mode": "edit", "input_image": reference.name}
            with mock.patch.object(worker, "_patch_mage_flow_runtime"), mock.patch.object(
                worker, "_disable_model_screening"
            ):
                self.assertIs(worker._call_pipeline(pipe, edit_request), image)
        pipe.edit.assert_called_once_with(
            ["test"],
            [[reference.name]],
            neg_prompts=[" "],
            seeds=[7],
            steps=20,
            cfg=5.0,
            heights=[768],
            widths=[1024],
        )

    def test_standard_gaussian_replacement_preserves_batch_dimension(self):
        worker = load_worker()
        module = types.SimpleNamespace()
        worker._patch_noise(module)
        try:
            noise = module.encode_noise((2, 3, 4), key=123, seed=9, device="cpu")
        except ImportError:
            self.skipTest("torch is not installed in the local test interpreter")
        self.assertEqual(tuple(noise.shape), (1, 2, 3, 4))
        second = module.encode_noise((2, 3, 4), key=999, seed=9, device="cpu")
        self.assertTrue(noise.equal(second), "watermark key must not affect Gaussian noise")


if __name__ == "__main__":
    unittest.main()
