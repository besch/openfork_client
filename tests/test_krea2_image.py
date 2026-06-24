import unittest

from dgn_client import DGNClient
from services.processors.image.krea2 import (
    Krea2TextToImageProcessor,
    clamp_krea2_cfg,
    clamp_krea2_steps,
)


class Krea2ImageProcessorTests(unittest.TestCase):
    def test_defaults_are_turbo_friendly(self):
        self.assertEqual(clamp_krea2_steps(None), 8)
        self.assertEqual(clamp_krea2_steps(99), 24)
        self.assertEqual(clamp_krea2_cfg(None), 1.0)
        self.assertEqual(clamp_krea2_cfg(-2), 0.0)

    def test_dimensions_are_24gb_safe_and_multiple_of_16(self):
        self.assertEqual(
            Krea2TextToImageProcessor._get_dimensions("16:9"),
            (1024, 576),
        )
        self.assertEqual(
            Krea2TextToImageProcessor._get_dimensions("9:16", requested_long_edge=2048),
            (864, 1536),
        )

    def test_dimensions_can_be_capped_for_lower_vram_tiers(self):
        self.assertEqual(
            Krea2TextToImageProcessor._get_dimensions(
                "16:9",
                requested_long_edge=2048,
                max_long_edge=1024,
            ),
            (1024, 576),
        )
        self.assertEqual(
            Krea2TextToImageProcessor._get_dimensions(
                "1:1",
                default_long_edge=768,
                max_long_edge=768,
            ),
            (768, 768),
        )

    def test_workflow_injection_sets_prompt_sampler_and_dimensions(self):
        processor = Krea2TextToImageProcessor.__new__(Krea2TextToImageProcessor)
        workflow = {
            "1": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "old prompt"},
            },
            "2": {
                "class_type": "EmptySD3LatentImage",
                "inputs": {"width": 1, "height": 1, "batch_size": 1},
            },
            "3": {
                "class_type": "ModelSamplingAuraFlow",
                "inputs": {"shift": 0.0},
            },
            "4": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 0,
                    "steps": 1,
                    "cfg": 1,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1,
                },
            },
        }

        processor._apply_inputs_to_workflow(
            workflow,
            prompt="new prompt",
            width=1344,
            height=768,
            inputs={
                "seed": 42,
                "steps": 12,
                "cfg_scale": 0.5,
                "sampler": "dpmpp_2m",
                "scheduler": "beta",
                "shift": 1.2,
            },
        )

        self.assertEqual(workflow["1"]["inputs"]["text"], "new prompt")
        self.assertEqual(workflow["2"]["inputs"]["width"], 1344)
        self.assertEqual(workflow["2"]["inputs"]["height"], 768)
        self.assertEqual(workflow["3"]["inputs"]["shift"], 1.2)
        self.assertEqual(workflow["4"]["inputs"]["seed"], 42)
        self.assertEqual(workflow["4"]["inputs"]["steps"], 12)
        self.assertEqual(workflow["4"]["inputs"]["cfg"], 0.5)
        self.assertEqual(workflow["4"]["inputs"]["sampler_name"], "dpmpp_2m")
        self.assertEqual(workflow["4"]["inputs"]["scheduler"], "beta")

    def test_gguf_workflow_keeps_clip_on_cpu_and_sets_default_model(self):
        processor = Krea2TextToImageProcessor.__new__(Krea2TextToImageProcessor)
        workflow = {
            "1": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "qwen3vl_4b_fp8_scaled.safetensors",
                    "type": "krea2",
                    "device": "default",
                },
            },
            "2": {"class_type": "UnetLoaderGGUF", "inputs": {}},
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 0,
                    "steps": 1,
                    "cfg": 1,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1,
                },
            },
        }

        processor._apply_inputs_to_workflow(
            workflow,
            prompt="new prompt",
            width=768,
            height=768,
            inputs={"seed": 7},
        )

        self.assertEqual(workflow["1"]["inputs"]["device"], "cpu")
        self.assertEqual(
            workflow["2"]["inputs"]["unet_name"],
            "krea2_turbo-Q3_K_M.gguf",
        )

    def test_krea2_workflow_is_registered_in_processor_map(self):
        client = DGNClient.__new__(DGNClient)
        client.services_config = {}
        client.config = {
            "krea2-turbo-text-to-image-8gb": {
                "service_name": "krea2-turbo-8gb",
                "processor": "Krea2TextToImageProcessor",
            },
            "krea2-turbo-text-to-image-16gb": {
                "service_name": "krea2-turbo-16gb",
                "processor": "Krea2TextToImageProcessor",
            },
            "krea2-turbo-text-to-image-24gb": {
                "service_name": "krea2-turbo-24gb",
                "processor": "Krea2TextToImageProcessor",
            },
        }

        processor_map = DGNClient._build_processor_map(client)

        self.assertIs(
            processor_map["krea2-turbo-text-to-image-8gb"],
            Krea2TextToImageProcessor,
        )
        self.assertIs(
            processor_map["krea2-turbo-text-to-image-16gb"],
            Krea2TextToImageProcessor,
        )
        self.assertIs(
            processor_map["krea2-turbo-text-to-image-24gb"],
            Krea2TextToImageProcessor,
        )


if __name__ == "__main__":
    unittest.main()
