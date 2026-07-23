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
    def test_native_sampler_request_requires_prompt_and_sigma_per_batch(self):
        worker_module_path = ROOT / "custom_nodes" / "ComfyColab-MageFlow" / "mage_flow_worker.py"
        name = "comfycolab_mageflow_protocol_test"
        spec = importlib.util.spec_from_file_location(name, worker_module_path)
        protocol = importlib.util.module_from_spec(spec)
        sys.modules[name] = protocol
        assert spec.loader
        spec.loader.exec_module(protocol)
        with tempfile.NamedTemporaryFile(suffix=".pt") as latent:
            command = protocol.MageFlowWorkerCommand(
                python="python",
                worker_script="worker.py",
                source_dir="/source",
                model_id="microsoft/Mage-Flow",
                model_revision="a" * 40,
                mode="native_denoise",
                prompt="",
                negative_prompt="",
                output_image="",
                metadata_output=latent.name + ".json",
                request_id="native",
                seed=0,
                width=1024,
                height=1024,
                steps=1,
                guidance_scale=1.0,
                input_latent=latent.name,
                output_latent=latent.name + ".out",
                prompts=("positive", "negative"),
                sigmas=(0.5, 0.5),
            )
            request = protocol.build_mage_flow_request(command)
        self.assertEqual(request["prompts"], ["positive", "negative"])
        self.assertEqual(request["sigmas"], [0.5, 0.5])
        self.assertEqual(command.server_argv()[-1], "native")
        with self.assertRaisesRegex(ValueError, "one prompt and sigma"):
            protocol.build_mage_flow_request(
                protocol.MageFlowWorkerCommand(
                    **{**command.__dict__, "sigmas": (0.5,)}
                )
            )
        decode = protocol.MageFlowWorkerCommand(
            **{
                **command.__dict__,
                "mode": "native_vae_decode",
                "prompts": (),
                "sigmas": (),
                "output_latent": "",
                "output_tensor": latent.name + ".images.pt",
            }
        )
        decode_request = protocol.build_mage_flow_request(decode)
        self.assertEqual(
            decode_request["output_tensor"],
            latent.name + ".images.pt",
        )
        with self.assertRaisesRegex(ValueError, "output_tensor"):
            protocol.build_mage_flow_request(
                protocol.MageFlowWorkerCommand(
                    **{**decode.__dict__, "output_tensor": ""}
                )
            )

    def test_vae_encode_request_validation_requires_tiled_latent_contract(self):
        worker = load_worker()
        with tempfile.NamedTemporaryFile(suffix=".png") as reference:
            request = {
                "protocol": worker.PROTOCOL_VERSION,
                "request_id": "encode",
                "mode": "vae_encode",
                "prompt": "",
                "output_image": "",
                "output_latent": reference.name + ".pt",
                "metadata_output": reference.name + ".json",
                "input_image": reference.name,
                "seed": 0,
                "width": 1024,
                "height": 768,
                "steps": 1,
                "guidance_scale": 0.0,
                "strength": 0.75,
                "tile_size": 1536,
                "tile_overlap": 384,
            }
            worker._validate_request(request)
            worker._validate_request({**request, "width": 128, "height": 128})
            with self.assertRaisesRegex(ValueError, "tile_overlap"):
                worker._validate_request({**request, "tile_overlap": 1536})
            with self.assertRaisesRegex(ValueError, "between 256 and 2048"):
                worker._validate_request(
                    {
                        **request,
                        "mode": "edit",
                        "width": 128,
                        "height": 128,
                    }
                )

    def test_tiled_mage_vae_encode_preserves_expected_latent_shape(self):
        worker = load_worker()
        try:
            import torch
            from PIL import Image
        except ImportError:
            self.skipTest("torch and Pillow are required")

        class FakeVAE:
            latent_channels = 128
            downsample_factor = 16
            device = torch.device("cpu")
            dtype = torch.float32

            @staticmethod
            def encode(image):
                shape = (
                    image.shape[0],
                    128,
                    image.shape[-2] // 16,
                    image.shape[-1] // 16,
                )
                return torch.ones(shape, dtype=image.dtype, device=image.device)

        with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
            Image.new("RGB", (768, 1024), color=(64, 128, 192)).save(image_file.name)
            latent = worker._encode_mage_vae(
                FakeVAE(),
                Path(image_file.name),
                {
                    "width": 768,
                    "height": 1024,
                    "tile_size": 512,
                    "tile_overlap": 128,
                },
            )
        self.assertEqual(tuple(latent.shape), (1, 128, 64, 48))
        self.assertTrue(torch.allclose(latent, torch.ones_like(latent)))

    def test_native_mage_vae_preserves_image_and_latent_batches(self):
        worker = load_worker()
        try:
            import torch
        except ImportError:
            self.skipTest("torch is required")

        class FakeVAE:
            latent_channels = 128
            downsample_factor = 16
            device = torch.device("cpu")
            dtype = torch.float32

            @staticmethod
            def encode(image):
                return torch.ones(
                    (
                        image.shape[0],
                        128,
                        image.shape[-2] // 16,
                        image.shape[-1] // 16,
                    ),
                    dtype=image.dtype,
                    device=image.device,
                )

            @staticmethod
            def decode(samples):
                return torch.nn.functional.interpolate(
                    samples[:, :3],
                    scale_factor=16,
                    mode="nearest",
                )

        request = {
            "width": 128,
            "height": 128,
            "tile_size": 64,
            "tile_overlap": 16,
        }
        pixels = torch.rand((2, 128, 128, 3), dtype=torch.float32)
        latent = worker._encode_mage_vae_tensor(FakeVAE(), pixels, request)
        self.assertEqual(tuple(latent.shape), (2, 128, 8, 8))

        with tempfile.NamedTemporaryFile(suffix=".pt") as latent_file:
            torch.save({"samples": latent}, latent_file.name)
            images = worker._decode_mage_vae_tensor(
                FakeVAE(),
                Path(latent_file.name),
                request,
            )
        self.assertEqual(tuple(images.shape), (2, 128, 128, 3))

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
