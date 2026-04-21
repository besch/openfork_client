import unittest

from services.processors.video.ltx23_common import (
    MODEL_TYPE_Q4,
    MODEL_TYPE_Q8,
    get_ltx23_model_type,
)


class LTX23ModelSelectionTests(unittest.TestCase):
    def test_8gb_tier_uses_q4_distilled_preset(self):
        self.assertEqual(get_ltx23_model_type("ltx23-video-8gb"), MODEL_TYPE_Q4)

    def test_standard_tiers_use_q8_distilled_preset(self):
        for service_type in (
            "ltx23-video-16gb",
            "ltx23-video-24gb",
            "ltx23-video-32gb",
            "",
        ):
            with self.subTest(service_type=service_type):
                self.assertEqual(get_ltx23_model_type(service_type), MODEL_TYPE_Q8)


if __name__ == "__main__":
    unittest.main()
