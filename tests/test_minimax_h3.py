import json
import unittest
from pathlib import Path

from dgn_client import DGNClient
from services.processors.video.minimax_h3 import (
    MiniMaxH3ImageToVideoProcessor,
    MiniMaxH3TextToVideoProcessor,
    build_minimax_h3_workflow,
    resolve_minimax_h3_dimensions,
    resolve_minimax_h3_frames,
    resolve_minimax_h3_tier,
)


ROOT = Path(__file__).resolve().parents[1]


class MiniMaxH3WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = json.loads(
            (ROOT / "workflows" / "minimax-h3.api.json").read_text(encoding="utf-8")
        )

    def test_resolves_all_registered_vram_tiers(self):
        for tier in (6, 8, 12, 16, 24, 32, 48, 80):
            self.assertEqual(
                resolve_minimax_h3_tier(f"minimax-h3-video-{tier}gb"), tier
            )

    def test_all_tiers_are_registered_for_client_and_mcp_discovery(self):
        registry = json.loads(
            (ROOT.parent / "website" / "services.json").read_text(encoding="utf-8")
        )
        client = DGNClient.__new__(DGNClient)
        client.services_config = registry["services"]
        client.config = registry["workflows"]
        processor_map = DGNClient._build_processor_map(client)

        for tier in (6, 8, 12, 16, 24, 32, 48, 80):
            service_id = f"minimax-h3-video-{tier}gb"
            text_id = f"minimax-h3-text-to-video-{tier}gb"
            image_id = f"minimax-h3-image-to-video-{tier}gb"
            with self.subTest(tier=tier):
                self.assertEqual(
                    registry["services"][service_id]["video_capabilities"],
                    {"text": text_id, "image": image_id},
                )
                self.assertIs(processor_map[text_id], MiniMaxH3TextToVideoProcessor)
                self.assertIs(processor_map[image_id], MiniMaxH3ImageToVideoProcessor)

    def test_dimensions_are_tier_capped_and_multiple_of_32(self):
        self.assertEqual(resolve_minimax_h3_dimensions("16:9", 6), (640, 352))
        self.assertEqual(resolve_minimax_h3_dimensions("9:16", 48), (768, 1344))
        self.assertEqual(resolve_minimax_h3_dimensions("1:1", 80), (768, 768))
        width, height = resolve_minimax_h3_dimensions(
            "16:9", 6, target_width=1344, target_height=768
        )
        self.assertLessEqual(width * height, 640 * 352)
        self.assertEqual(width % 32, 0)
        self.assertEqual(height % 32, 0)

    def test_duration_uses_h3_frame_grid_and_tier_cap(self):
        requested, frames = resolve_minimax_h3_frames(5, 6)
        self.assertEqual(requested, 5)
        self.assertEqual(frames, 124)
        self.assertEqual((frames - 5) % 17, 0)

        requested, frames = resolve_minimax_h3_frames(15, 48)
        self.assertEqual(requested, 15)
        self.assertEqual(frames, 362)
        self.assertEqual((frames - 5) % 17, 0)

        requested, _ = resolve_minimax_h3_frames(15, 8)
        self.assertEqual(requested, 5)

    def test_native_graph_injects_joint_audio_video_settings(self):
        graph = build_minimax_h3_workflow(
            self.workflow,
            prompt="A train passes. Audio: steel wheels and rain.",
            width=1088,
            height=608,
            frames=243,
            steps=20,
            seed=123,
            filename_prefix="video/minimax-h3/test",
        )
        self.assertEqual(graph["6"]["inputs"]["length"], 243)
        self.assertEqual(graph["7"]["inputs"]["noise_seed"], 123)
        self.assertEqual(graph["14"]["inputs"]["audio"], ["13", 0])
        self.assertNotIn("16", graph)
        self.assertNotIn("17", graph)

    def test_image_graph_and_spectrum_are_explicit_opt_ins(self):
        graph = build_minimax_h3_workflow(
            self.workflow,
            prompt="A portrait speaks.",
            width=608,
            height=352,
            frames=124,
            steps=20,
            seed=456,
            filename_prefix="video/minimax-h3/test",
            start_image_filename="start.png",
            acceleration_mode="spectrum",
        )
        self.assertEqual(graph["6"]["inputs"]["first_frame"], ["16", 0])
        self.assertEqual(graph["17"]["class_type"], "SpectrumApplyMiniMaxH3")
        self.assertEqual(graph["17"]["inputs"]["history_storage"], "system_ram")
        self.assertEqual(graph["9"]["inputs"]["model"], ["17", 0])
        self.assertEqual(graph["10"]["inputs"]["model"], ["17", 0])


if __name__ == "__main__":
    unittest.main()
