import unittest

from services.processors.video.davinci_magihuman import (
    MODEL_TYPE_BASE_SR1080,
    MODEL_TYPE_DISTILL_SR1080,
    clamp_davinci_magihuman_duration,
    clamp_davinci_magihuman_steps,
    davinci_magihuman_resolution,
    duration_to_wangp_frames,
    get_davinci_magihuman_model_type,
)


class DaVinciMagiHumanWan2GPTests(unittest.TestCase):
    def test_16gb_tier_uses_distill_sr1080(self):
        self.assertEqual(
            get_davinci_magihuman_model_type("davinci-magihuman-16gb"),
            MODEL_TYPE_DISTILL_SR1080,
        )

    def test_24gb_and_32gb_tiers_use_base_sr1080(self):
        for service_type in (
            "davinci-magihuman-24gb",
            "davinci-magihuman-32gb",
            "",
        ):
            with self.subTest(service_type=service_type):
                self.assertEqual(
                    get_davinci_magihuman_model_type(service_type),
                    MODEL_TYPE_BASE_SR1080,
                )

    def test_runtime_limits_match_wangp_101_frame_target(self):
        self.assertEqual(clamp_davinci_magihuman_duration(8, "davinci-magihuman-16gb"), 4.0)
        self.assertEqual(clamp_davinci_magihuman_steps(16, "davinci-magihuman-16gb"), 8)
        self.assertEqual(clamp_davinci_magihuman_duration(8, "davinci-magihuman-32gb"), 5.0)
        self.assertEqual(duration_to_wangp_frames(4.0), 101)

    def test_resolution_uses_1080p_sr_presets(self):
        self.assertEqual(davinci_magihuman_resolution("16:9"), "1920x1088")
        self.assertEqual(davinci_magihuman_resolution("9:16"), "1088x1920")
        self.assertEqual(davinci_magihuman_resolution("unknown"), "1920x1088")


if __name__ == "__main__":
    unittest.main()
