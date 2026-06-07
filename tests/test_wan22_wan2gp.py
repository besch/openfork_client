import json
import unittest
from pathlib import Path

from dgn_client import DGNClient
from services.wan2gp_runtime import build_wan2gp_environment
from services.processors.video.wan22_wan2gp import (
    ImageToVideoFromLastFrameWan2GPProcessor,
    WAN22ImageToVideoWan2GPProcessor,
    WAN22TextToVideoWan2GPProcessor,
    clamp_wan22_wan2gp_duration,
    clamp_wan22_wan2gp_steps,
    duration_to_wan22_frames,
    get_wan22_wan2gp_runtime_limits,
    get_wan22_wan2gp_tier,
    wan22_wan2gp_resolution,
)


class WAN22Wan2GPTests(unittest.TestCase):
    def test_service_type_selects_expected_tiers(self):
        self.assertEqual(get_wan22_wan2gp_tier("wan22-wan2gp-8gb"), "8gb")
        self.assertEqual(get_wan22_wan2gp_tier("wan22-wan2gp-10gb"), "10gb")
        self.assertEqual(get_wan22_wan2gp_tier("wan22-wan2gp-12gb"), "12gb")
        self.assertEqual(get_wan22_wan2gp_tier("wan22-wan2gp-16gb"), "16gb")
        self.assertEqual(get_wan22_wan2gp_tier("wan22-wan2gp-24gb"), "24gb")

    def test_low_vram_runtime_limits_are_conservative(self):
        limits = get_wan22_wan2gp_runtime_limits("wan22-wan2gp-8gb")

        self.assertEqual(limits["duration_default"], 3.0)
        self.assertEqual(limits["duration_max"], 4.0)
        self.assertEqual(limits["steps_default"], 8)
        self.assertEqual(limits["steps_max"], 12)
        self.assertEqual(clamp_wan22_wan2gp_duration(8, "wan22-wan2gp-8gb"), 4.0)
        self.assertEqual(clamp_wan22_wan2gp_steps(40, "wan22-wan2gp-8gb"), 12)

    def test_10gb_and_12gb_tiers_are_available(self):
        self.assertEqual(clamp_wan22_wan2gp_duration(5, "wan22-wan2gp-10gb"), 5.0)
        self.assertEqual(clamp_wan22_wan2gp_steps(20, "wan22-wan2gp-10gb"), 16)
        self.assertEqual(clamp_wan22_wan2gp_duration(8, "wan22-wan2gp-12gb"), 5.0)
        self.assertEqual(clamp_wan22_wan2gp_steps(20, "wan22-wan2gp-12gb"), 18)

    def test_duration_to_frames_aligns_with_wan_frame_count(self):
        self.assertEqual(duration_to_wan22_frames(1), 17)
        self.assertEqual(duration_to_wan22_frames(3), 49)
        self.assertEqual(duration_to_wan22_frames(5), 81)
        self.assertEqual((duration_to_wan22_frames(4) - 1) % 4, 0)

    def test_resolution_uses_tier_specific_presets(self):
        self.assertEqual(wan22_wan2gp_resolution("16:9", "wan22-wan2gp-8gb"), "320x176")
        self.assertEqual(wan22_wan2gp_resolution("9:16", "wan22-wan2gp-10gb"), "256x448")
        self.assertEqual(wan22_wan2gp_resolution("1:1", "wan22-wan2gp-12gb"), "480x480")
        self.assertEqual(wan22_wan2gp_resolution("unknown", "wan22-wan2gp-24gb"), "832x480")

    def test_wan22_wan2gp_workflows_are_registered_in_processor_map(self):
        client = DGNClient.__new__(DGNClient)
        client.services_config = {}
        client.config = {
            "wan22-wan2gp-text-to-video-8gb": {
                "processor": "WAN22TextToVideoWan2GPProcessor",
            },
            "wan22-wan2gp-image-to-video-10gb": {
                "processor": "WAN22ImageToVideoWan2GPProcessor",
            },
            "wan22-wan2gp-image-to-video-from-last-frame-12gb": {
                "processor": "ImageToVideoFromLastFrameWan2GPProcessor",
            },
        }

        processor_map = DGNClient._build_processor_map(client)

        self.assertIs(
            processor_map["wan22-wan2gp-text-to-video-8gb"],
            WAN22TextToVideoWan2GPProcessor,
        )
        self.assertIs(
            processor_map["wan22-wan2gp-image-to-video-10gb"],
            WAN22ImageToVideoWan2GPProcessor,
        )
        self.assertIs(
            processor_map["wan22-wan2gp-image-to-video-from-last-frame-12gb"],
            ImageToVideoFromLastFrameWan2GPProcessor,
        )

    def test_registry_entries_keep_8gb_16gb_and_24gb_wan22_wan2gp_enabled(self):
        registry_path = Path(__file__).resolve().parents[2] / "website" / "services.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))

        disabled_service_names = {
            "wan22-wan2gp-10gb",
            "wan22-wan2gp-12gb",
        }
        enabled_service_names = {
            "wan22-wan2gp-8gb",
            "wan22-wan2gp-16gb",
            "wan22-wan2gp-24gb",
        }
        workflow_names = {
            workflow_id
            for workflow_id in registry["workflows"]
            if workflow_id.startswith("wan22-wan2gp-")
        }

        self.assertTrue((disabled_service_names | enabled_service_names).issubset(registry["services"]))
        self.assertEqual(
            {
                "wan22-wan2gp-text-to-video-8gb",
                "wan22-wan2gp-image-to-video-8gb",
                "wan22-wan2gp-image-to-video-from-last-frame-8gb",
                "wan22-wan2gp-text-to-video-10gb",
                "wan22-wan2gp-image-to-video-10gb",
                "wan22-wan2gp-image-to-video-from-last-frame-10gb",
                "wan22-wan2gp-text-to-video-12gb",
                "wan22-wan2gp-image-to-video-12gb",
                "wan22-wan2gp-image-to-video-from-last-frame-12gb",
                "wan22-wan2gp-text-to-video-16gb",
                "wan22-wan2gp-image-to-video-16gb",
                "wan22-wan2gp-text-to-video-24gb",
                "wan22-wan2gp-image-to-video-24gb",
            },
            workflow_names,
        )

        for service_name in disabled_service_names:
            with self.subTest(service_name=service_name):
                self.assertEqual(
                    bool(registry["services"][service_name].get("disabled")),
                    True,
                )
                self.assertEqual(registry["services"][service_name]["backend"], "wan2gp")

        for service_name in enabled_service_names:
            with self.subTest(service_name=service_name):
                self.assertEqual(
                    bool(registry["services"][service_name].get("disabled")),
                    False,
                )
                self.assertEqual(registry["services"][service_name]["backend"], "wan2gp")

        for workflow_id, workflow in registry["workflows"].items():
            if workflow_id.startswith("wan22-wan2gp-"):
                with self.subTest(workflow_id=workflow_id):
                    expected_disabled = not any(
                        tier in workflow_id for tier in ("8gb", "16gb", "24gb")
                    )
                    self.assertEqual(
                        bool(workflow.get("disabled")),
                        expected_disabled,
                    )

    def test_runtime_args_match_wan22_model_variants(self):
        low_vram_args = build_wan2gp_environment("wan22-wan2gp-8gb")["WAN2GP_CLI_ARGS"]
        high_vram_args = build_wan2gp_environment("wan22-wan2gp-24gb")["WAN2GP_CLI_ARGS"]

        self.assertIn("--preload 0", low_vram_args)
        self.assertNotIn("--bf16", low_vram_args)
        self.assertIn("--profile 4", high_vram_args)
        self.assertNotIn("--bf16", high_vram_args)


if __name__ == "__main__":
    unittest.main()
