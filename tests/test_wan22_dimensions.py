import unittest

from utils.comfyui_workflow_utils import (
    get_dimensions,
    inject_prompt_and_image_into_workflow,
    inject_prompt_into_text_to_video_workflow,
    resolve_wan22_dimensions,
)


class WAN22DimensionTests(unittest.TestCase):
    def test_8gb_tier_uses_smaller_16_by_9_resolution(self):
        self.assertEqual(
            get_dimensions("16:9", vram_tier="wan22-image-to-video-8gb"),
            (320, 176),
        )

    def test_default_tier_keeps_existing_16_by_9_resolution(self):
        self.assertEqual(get_dimensions("16:9"), (768, 432))

    def test_16gb_tier_uses_probe_practical_16_by_9_resolution(self):
        self.assertEqual(
            get_dimensions("16:9", vram_tier="wan22-image-to-video-16gb"),
            (896, 512),
        )
        self.assertEqual(
            get_dimensions("16:9", vram_tier="wan22-comfyui-16"),
            (896, 512),
        )

    def test_24gb_tier_uses_classic_workflow_safe_480p_resolution(self):
        self.assertEqual(
            get_dimensions("16:9", vram_tier="wan22-image-to-video-24gb"),
            (832, 480),
        )
        self.assertEqual(
            get_dimensions("16:9", vram_tier="wan22-comfyui-24"),
            (832, 480),
        )

    def test_16gb_tier_accepts_explicit_probe_dimensions(self):
        self.assertEqual(
            resolve_wan22_dimensions(
                "16:9",
                vram_tier="wan22-text-to-video-16gb",
                target_width=1024,
                target_height=576,
            ),
            (1024, 576),
        )

    def test_24gb_tier_caps_explicit_probe_dimensions_to_safe_480p(self):
        self.assertEqual(
            resolve_wan22_dimensions(
                "16:9",
                vram_tier="wan22-image-to-video-24gb",
                target_width=1025,
                target_height=577,
            ),
            (832, 480),
        )

    def test_24gb_tier_accepts_explicit_dimensions_below_safe_480p(self):
        self.assertEqual(
            resolve_wan22_dimensions(
                "16:9",
                vram_tier="wan22-image-to-video-24gb",
                target_width=640,
                target_height=368,
            ),
            (640, 368),
        )

    def test_8gb_tier_ignores_explicit_probe_dimensions(self):
        self.assertEqual(
            resolve_wan22_dimensions(
                "16:9",
                vram_tier="wan22-image-to-video-8gb",
                target_width=1024,
                target_height=576,
            ),
            (320, 176),
        )

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

        self.assertEqual((graph["9"]["inputs"]["width"], graph["9"]["inputs"]["height"]), (320, 176))
        self.assertEqual((graph["10"]["inputs"]["width"], graph["10"]["inputs"]["height"]), (320, 176))

    def test_image_to_video_injects_explicit_16gb_dimensions(self):
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
            vram_tier="wan22-image-to-video-16gb",
            target_width=1024,
            target_height=576,
        )

        self.assertEqual((graph["9"]["inputs"]["width"], graph["9"]["inputs"]["height"]), (1024, 576))
        self.assertEqual((graph["10"]["inputs"]["width"], graph["10"]["inputs"]["height"]), (1024, 576))

    def test_image_to_video_caps_explicit_24gb_dimensions(self):
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
            vram_tier="wan22-image-to-video-24gb",
            target_width=960,
            target_height=544,
        )

        self.assertEqual((graph["9"]["inputs"]["width"], graph["9"]["inputs"]["height"]), (832, 480))
        self.assertEqual((graph["10"]["inputs"]["width"], graph["10"]["inputs"]["height"]), (832, 480))

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

        self.assertEqual((graph["9"]["inputs"]["width"], graph["9"]["inputs"]["height"]), (320, 176))

    def test_text_to_video_caps_explicit_24gb_dimensions(self):
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
            vram_tier="wan22-text-to-video-24gb",
            target_width=1280,
            target_height=720,
        )

        self.assertEqual((graph["9"]["inputs"]["width"], graph["9"]["inputs"]["height"]), (832, 480))


if __name__ == "__main__":
    unittest.main()
