from pathlib import Path
import unittest

from services.processors.video.ltx23_common import (
    MODEL_TYPE_Q4,
    MODEL_TYPE_Q6,
    MODEL_TYPE_Q8,
    clamp_ltx23_duration,
    clamp_ltx23_steps,
    get_ltx23_runtime_limits,
    get_ltx23_model_type,
)
from services.processors.wan2gp_processor import Wan2GPProcessor


class LTX23ModelSelectionTests(unittest.TestCase):
    def test_8gb_tier_uses_q4_distilled_preset(self):
        self.assertEqual(get_ltx23_model_type("ltx23-video-8gb"), MODEL_TYPE_Q4)

    def test_stability_constrained_tiers_use_q6_distilled_preset(self):
        self.assertEqual(get_ltx23_model_type("ltx23-video-12gb"), MODEL_TYPE_Q6)
        self.assertEqual(get_ltx23_model_type("ltx23-video-24gb"), MODEL_TYPE_Q6)

    def test_standard_tiers_use_q8_distilled_preset(self):
        for service_type in (
            "ltx23-video-16gb",
            "ltx23-video-32gb",
            "",
        ):
            with self.subTest(service_type=service_type):
                self.assertEqual(get_ltx23_model_type(service_type), MODEL_TYPE_Q8)

    def test_8gb_runtime_limits_are_conservative(self):
        limits = get_ltx23_runtime_limits("ltx23-video-8gb")

        self.assertEqual(limits["duration_default"], 2.0)
        self.assertEqual(limits["duration_max"], 2.0)
        self.assertEqual(limits["steps_default"], 6)
        self.assertEqual(limits["steps_max"], 8)
        self.assertEqual(clamp_ltx23_duration(3, "ltx23-video-8gb"), 2.0)
        self.assertEqual(clamp_ltx23_steps(20, "ltx23-video-8gb"), 8)

    def test_non_8gb_runtime_limits_preserve_longer_jobs(self):
        self.assertEqual(clamp_ltx23_duration(5, "ltx23-video-16gb"), 5.0)
        self.assertEqual(clamp_ltx23_duration(10, "ltx23-video-24gb"), 7.0)
        self.assertEqual(clamp_ltx23_steps(12, "ltx23-video-16gb"), 12)

    def test_wan2gp_cuda_oom_detail_is_infrastructure_error(self):
        detail = {
            "detail": {
                "errors": [
                    "generation: CUDA driver error: out of memory",
                ],
            }
        }

        self.assertTrue(Wan2GPProcessor._is_wan2gp_cuda_oom(detail))

    def test_wan2gp_server_accepts_moved_hdr_lora_path(self):
        server_path = (
            Path(__file__).resolve().parents[1]
            / "comfyui-storage"
            / "wan2gp_server.py"
        )
        source = server_path.read_text(encoding="utf-8")

        self.assertIn("_HDR_LORA_CANDIDATE_PATHS", source)
        self.assertIn('"loras", "ltx2"', source)
        self.assertIn("_resolve_existing_path", source)
        self.assertIn('settings["lora_filename"] = hdr_lora_path', source)


if __name__ == "__main__":
    unittest.main()
