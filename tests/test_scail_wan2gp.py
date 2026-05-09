import unittest

from services.processors.video.scail import (
    clamp_scail_duration,
    clamp_scail_steps,
    duration_to_wangp_frames,
    scail_resolution,
)


class SCAILWan2GPTests(unittest.TestCase):
    def test_resolution_defaults_to_scail_512p_landscape(self):
        self.assertEqual(scail_resolution("16:9"), "896x512")
        self.assertEqual(scail_resolution("9:16"), "512x896")
        self.assertEqual(scail_resolution("unknown"), "896x512")

    def test_duration_clamped_to_short_scail_window(self):
        self.assertEqual(clamp_scail_duration(None), 5.0)
        self.assertEqual(clamp_scail_duration(0.5), 1.0)
        self.assertEqual(clamp_scail_duration(8), 5.0)

    def test_steps_clamped_to_supported_ui_range(self):
        self.assertEqual(clamp_scail_steps(None), 8)
        self.assertEqual(clamp_scail_steps(2), 6)
        self.assertEqual(clamp_scail_steps(30), 20)

    def test_duration_to_frames_aligns_with_wangp_frame_count(self):
        self.assertEqual(duration_to_wangp_frames(5), 81)
        self.assertEqual((duration_to_wangp_frames(4) - 1) % 4, 0)


if __name__ == "__main__":
    unittest.main()
