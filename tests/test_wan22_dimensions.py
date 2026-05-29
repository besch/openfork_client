import unittest

from utils.comfyui_workflow_utils import (
    get_dimensions,
    inject_prompt_and_image_into_workflow,
    inject_prompt_into_text_to_video_workflow,
)


class WAN22DimensionTests(unittest.TestCase):
    def test_8gb_tier_uses_smaller_16_by_9_resolution(self):
        self.assertEqual(
            get_dimensions("16:9", vram_tier="wan22-image-to-video-8gb"),
            (512, 288),
        )

    def test_default_tier_keeps_existing_16_by_9_resolution(self):
        self.assertEqual(get_dimensions("16:9"), (768, 432))

    def test_image_to_video_injects_8gb_dimensions(self):
        workflow = {
            "prompt": {
                "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
                "7": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
                "8": {"class_type": "LoadImage", "inputs": {"image": ""}},
                "9": {
                    "class_type": "WanImageToVideo",
                    "inputs": {"width": 768, "height": 432},
                },
                "10": {
                    "class_type": "ImageResizeKJv2",
                    "inputs": {"width": 768, "height": 432},
                },
            }
        }

        graph = inject_prompt_and_image_into_workflow(
            workflow,
            "prompt",
            "negative",
            "start.png",
            "16:9",
            vram_tier="wan22-image-to-video-8gb",
        )

        self.assertEqual((graph["9"]["inputs"]["width"], graph["9"]["inputs"]["height"]), (512, 288))
        self.assertEqual((graph["10"]["inputs"]["width"], graph["10"]["inputs"]["height"]), (512, 288))

    def test_text_to_video_injects_8gb_dimensions(self):
        workflow = {
            "prompt": {
                "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
                "7": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
                "9": {
                    "class_type": "WanImageToVideo",
                    "inputs": {"width": 768, "height": 432},
                },
            }
        }

        graph = inject_prompt_into_text_to_video_workflow(
            workflow,
            "prompt",
            "negative",
            "16:9",
            seed=1,
            vram_tier="wan22-text-to-video-8gb",
        )

        self.assertEqual((graph["9"]["inputs"]["width"], graph["9"]["inputs"]["height"]), (512, 288))


if __name__ == "__main__":
    unittest.main()
