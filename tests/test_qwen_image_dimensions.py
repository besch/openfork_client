import unittest

from services.processors.image.qwen import (
    QwenImage2512LoraT2IProcessor,
    QwenImageEditProcessor,
    QwenImageInpaintProcessor,
    QwenImageT2IProcessor,
)


class QwenImageT2IDimensionTests(unittest.TestCase):
    def setUp(self):
        self.processor = QwenImageT2IProcessor.__new__(QwenImageT2IProcessor)

    def test_t2i_uses_8gb_safe_landscape_dimensions(self):
        self.assertEqual(self.processor._get_dimensions("16:9"), (512, 288))

    def test_t2i_uses_8gb_safe_square_dimensions(self):
        self.assertEqual(self.processor._get_dimensions("1:1"), (512, 512))

    def test_t2i_injects_safe_dimensions_into_empty_latent(self):
        workflow = {
            "prompt": {
                "4": {
                    "class_type": "EmptySD3LatentImage",
                    "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
                },
                "5": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": "placeholder"},
                },
                "6": {
                    "class_type": "ConditioningZeroOut",
                    "inputs": {"conditioning": ["5", 0]},
                },
                "8": {
                    "class_type": "KSampler",
                    "inputs": {"seed": 0, "steps": 10},
                },
            }
        }

        graph = self.processor._inject_t2i_workflow(
            workflow,
            "patched prompt",
            512,
            288,
            seed=123,
            steps=4,
        )

        self.assertEqual((graph["4"]["inputs"]["width"], graph["4"]["inputs"]["height"]), (512, 288))
        self.assertEqual(graph["5"]["inputs"]["text"], "patched prompt")
        self.assertEqual(graph["8"]["inputs"]["seed"], 123)
        self.assertEqual(graph["8"]["inputs"]["steps"], 4)

    def test_t2i_clamps_injected_steps_to_workflow_limit(self):
        workflow = {
            "prompt": {
                "8": {
                    "class_type": "KSampler",
                    "inputs": {"seed": 0, "steps": 10},
                },
            }
        }

        graph = self.processor._inject_t2i_workflow(
            workflow,
            "patched prompt",
            512,
            512,
            seed=123,
            steps=99,
        )

        self.assertEqual(graph["8"]["inputs"]["steps"], 10)


class QwenImageEditStepTests(unittest.TestCase):
    def setUp(self):
        self.processor = QwenImageEditProcessor.__new__(QwenImageEditProcessor)

    def test_edit_injects_requested_steps(self):
        workflow = {
            "prompt": {
                "1": {"class_type": "LoadImage", "inputs": {"image": "old.png"}},
                "2": {
                    "class_type": "KSampler",
                    "inputs": {"seed": 0, "steps": 20, "denoise": 1.0},
                },
            }
        }

        graph = self.processor._inject_edit_workflow(
            workflow,
            "patched prompt",
            "source.png",
            0.55,
            seed=123,
            steps=4,
        )

        self.assertEqual(graph["2"]["inputs"]["steps"], 4)
        self.assertEqual(graph["2"]["inputs"]["seed"], 123)

    def test_edit_clamps_requested_steps_to_workflow_limit(self):
        workflow = {
            "prompt": {
                "1": {"class_type": "LoadImage", "inputs": {"image": "old.png"}},
                "2": {
                    "class_type": "KSampler",
                    "inputs": {"seed": 0, "steps": 20, "denoise": 1.0},
                },
            }
        }

        graph = self.processor._inject_edit_workflow(
            workflow,
            "patched prompt",
            "source.png",
            0.55,
            seed=123,
            steps=99,
        )

        self.assertEqual(graph["2"]["inputs"]["steps"], 10)


class QwenImageInpaintStepTests(unittest.TestCase):
    def setUp(self):
        self.processor = QwenImageInpaintProcessor.__new__(QwenImageInpaintProcessor)

    def test_inpaint_injects_requested_steps(self):
        workflow = {
            "prompt": {
                "1": {"class_type": "LoadImage", "inputs": {"image": "old.png"}},
                "2": {"class_type": "LoadImage", "inputs": {"image": "mask.png"}},
                "3": {
                    "class_type": "KSampler",
                    "inputs": {"seed": 0, "steps": 10, "denoise": 0.8},
                },
            }
        }

        graph = self.processor._inject_inpaint_workflow(
            workflow,
            "patched prompt",
            "source.png",
            "mask.png",
            0.65,
            seed=123,
            steps=4,
        )

        self.assertEqual(graph["3"]["inputs"]["steps"], 4)
        self.assertEqual(graph["3"]["inputs"]["seed"], 123)


class QwenImage2512LoraT2ITests(unittest.TestCase):
    def setUp(self):
        self.processor = QwenImage2512LoraT2IProcessor.__new__(
            QwenImage2512LoraT2IProcessor
        )

    def test_lora_24gb_workflow_injects_gguf_model_lora_dimensions_and_steps(self):
        workflow = {
            "prompt": {
                "3": {
                    "class_type": "UnetLoaderGGUF",
                    "inputs": {"unet_name": "qwen-image-2512-Q4_K_M.gguf"},
                },
                "4": {
                    "class_type": "EmptySD3LatentImage",
                    "inputs": {"width": 1328, "height": 1328, "batch_size": 1},
                },
                "5": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": "placeholder"},
                },
                "6": {
                    "class_type": "ConditioningZeroOut",
                    "inputs": {"conditioning": ["5", 0]},
                },
                "7": {
                    "class_type": "LoraLoaderModelOnly",
                    "inputs": {
                        "lora_name": "old_lora.safetensors",
                        "strength_model": 0.8,
                        "model": ["3", 0],
                    },
                },
                "9": {
                    "class_type": "KSampler",
                    "inputs": {"seed": 0, "steps": 50},
                },
            }
        }

        width, height = self.processor._get_dimensions("16:9", requested_long_edge=512)
        graph = self.processor._inject_lora_t2i_workflow(
            workflow,
            "patched prompt",
            width,
            height,
            "character.safetensors",
            1.1,
            seed=123,
            steps=99,
        )

        self.assertIsNotNone(graph)
        self.assertEqual((width, height), (512, 288))
        self.assertEqual((graph["4"]["inputs"]["width"], graph["4"]["inputs"]["height"]), (512, 288))
        self.assertEqual(graph["5"]["inputs"]["text"], "patched prompt")
        self.assertEqual(graph["7"]["inputs"]["lora_name"], "character.safetensors")
        self.assertEqual(graph["7"]["inputs"]["strength_model"], 1.1)
        self.assertEqual(graph["9"]["inputs"]["seed"], 123)
        self.assertEqual(graph["9"]["inputs"]["steps"], 60)
        self.assertEqual(
            self.processor._get_base_model_name(graph),
            "qwen-image-2512-Q4_K_M.gguf",
        )


if __name__ == "__main__":
    unittest.main()
