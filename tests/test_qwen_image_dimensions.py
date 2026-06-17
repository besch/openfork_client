import unittest

from services.processors.image.qwen import QwenImageT2IProcessor


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


if __name__ == "__main__":
    unittest.main()
