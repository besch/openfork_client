from pathlib import Path
import unittest

from services.processors.video.ltx23_common import (
    MODEL_TYPE_DISTILLED_11,
    MODEL_TYPE_Q4,
    MODEL_TYPE_Q6,
    MODEL_TYPE_Q8,
    build_ltx23_prompt,
    clamp_ltx23_duration,
    clamp_ltx23_steps,
    get_ltx23_runtime_limits,
    get_ltx23_model_type,
    should_use_ltx23_hdr,
)
from services.processors.video.ltx23_image import (
    LTX23ImageToVideoWan2GPProcessor,
    _resolution_to_dimensions,
)
from services.processors.wan2gp_processor import Wan2GPProcessor
from services.disk_space_utils import estimate_image_size_bytes
from services.wan2gp_runtime import build_wan2gp_environment
from unittest.mock import Mock, patch


class LTX23ModelSelectionTests(unittest.TestCase):
    def test_8gb_tier_uses_q4_distilled_preset(self):
        self.assertEqual(get_ltx23_model_type("ltx23-video-8gb"), MODEL_TYPE_Q4)

    def test_stability_constrained_tier_uses_q6_distilled_preset(self):
        self.assertEqual(get_ltx23_model_type("ltx23-video-12gb"), MODEL_TYPE_Q6)

    def test_high_quality_tiers_use_distilled_1_1_preset(self):
        self.assertEqual(
            get_ltx23_model_type("ltx23-video-24gb"), MODEL_TYPE_DISTILLED_11
        )
        self.assertEqual(
            get_ltx23_model_type("ltx23-video-32gb"), MODEL_TYPE_DISTILLED_11
        )

    def test_standard_tiers_use_q8_distilled_preset(self):
        for service_type in (
            "ltx23-video-16gb",
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

    def test_ltx23_last_frame_extraction_uses_generation_dimensions(self):
        processor = Mock()
        processor.job = {
            "id": "ltx-job",
            "workflow_type": "ltx23-image-to-video-from-last-frame-24gb",
        }

        with patch(
            "services.processors.video.ltx23_image.materialize_last_frame_start_image",
            return_value="/tmp/last-frame.jpg",
        ) as materialize:
            path = LTX23ImageToVideoWan2GPProcessor._resolve_start_image(
                processor,
                {"input_video_url": "https://assets.example/source.mp4"},
                target_dimensions=(960, 544),
            )

        self.assertEqual(path, "/tmp/last-frame.jpg")
        materialize.assert_called_once_with(
            processor,
            {"input_video_url": "https://assets.example/source.mp4"},
            target_dimensions=(960, 544),
        )

    def test_ltx23_resolution_string_parses_to_dimensions(self):
        self.assertEqual(_resolution_to_dimensions("960x544"), (960, 544))
        self.assertIsNone(_resolution_to_dimensions("bad"))

    def test_ltx23_hdr_defaults_follow_hdr_capable_tiers(self):
        self.assertFalse(should_use_ltx23_hdr({}, "ltx23-video-8gb"))
        for service_type in (
            "ltx23-video-12gb",
            "ltx23-video-16gb",
            "ltx23-video-24gb",
            "ltx23-video-32gb",
        ):
            with self.subTest(service_type=service_type):
                self.assertTrue(should_use_ltx23_hdr({}, service_type))

        self.assertFalse(should_use_ltx23_hdr({"hdr": "false"}, "ltx23-video-24gb"))

    def test_ltx23_prompt_appends_explicit_audio_direction(self):
        prompt, audio_prompt = build_ltx23_prompt(
            "A quiet room with a blinking console.",
            {"audio_prompt": "low electrical hum, no melody, no vocals"},
        )

        self.assertEqual(audio_prompt, "low electrical hum, no melody, no vocals")
        self.assertIn("Audio: low electrical hum, no melody, no vocals", prompt)

    def test_ltx23_prompt_can_request_silence(self):
        prompt, audio_prompt = build_ltx23_prompt(
            "A candle flickers beside a window.",
            {"no_audio": True},
        )

        self.assertEqual(audio_prompt, "Silent video, no audio track.")
        self.assertIn("Audio: Silent video, no audio track.", prompt)

    def test_missing_ltx23_seed_can_be_randomized_without_changing_explicit_zero(self):
        with patch(
            "services.processors.wan2gp_processor.secrets.randbelow",
            return_value=12345,
        ):
            self.assertEqual(
                Wan2GPProcessor.resolve_seed(None, randomize_missing=True),
                12345,
            )
            self.assertEqual(
                Wan2GPProcessor.resolve_seed("", randomize_missing=True),
                12345,
            )

        self.assertEqual(
            Wan2GPProcessor.resolve_seed("0", randomize_missing=True),
            0,
        )

    def test_ltx23_runtime_args_are_tier_specific(self):
        twelve_args = build_wan2gp_environment("ltx23-video-12gb")["WAN2GP_CLI_ARGS"]
        twenty_four_args = build_wan2gp_environment("ltx23-video-24gb")[
            "WAN2GP_CLI_ARGS"
        ]
        thirty_two_args = build_wan2gp_environment("ltx23-video-32gb")[
            "WAN2GP_CLI_ARGS"
        ]

        self.assertIn("--profile 4.5", twelve_args)
        self.assertIn("--perc-reserved-mem-max 0.55", twelve_args)
        self.assertIn("--vram-safety-coefficient 0.70", twelve_args)
        self.assertIn("--profile 4.5", twenty_four_args)
        self.assertIn("--perc-reserved-mem-max 0.45", twenty_four_args)
        self.assertIn("--profile 4 ", thirty_two_args)
        self.assertIn("--vram-safety-coefficient 0.80", thirty_two_args)

    def test_ltx23_hdr_image_disk_estimate_matches_quantized_v1_1_payload(self):
        size_gb = estimate_image_size_bytes(
            "beschiak/openfork-ltx23-wan2gp-hdr:latest"
        ) // (1024**3)

        self.assertEqual(size_gb, 200)


if __name__ == "__main__":
    unittest.main()
