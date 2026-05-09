import unittest

from services.processors.video.vista4d import (
    clamp_float,
    clamp_vista4d_steps,
    normalize_seed,
    normalize_vista4d_camera_mode,
    vista4d_resolution,
)


class Vista4DWan2GPTests(unittest.TestCase):
    def test_resolution_defaults_to_vista4d_384p_landscape(self):
        self.assertEqual(vista4d_resolution("16:9"), "672x384")
        self.assertEqual(vista4d_resolution("9:16"), "384x672")
        self.assertEqual(vista4d_resolution("unknown"), "672x384")

    def test_steps_clamped_to_wan2gp_default_range(self):
        self.assertEqual(clamp_vista4d_steps(None), 50)
        self.assertEqual(clamp_vista4d_steps(4), 10)
        self.assertEqual(clamp_vista4d_steps(80), 50)

    def test_camera_mode_accepts_wan2gp_modes_and_aliases(self):
        self.assertEqual(normalize_vista4d_camera_mode("truck-left"), "truck_left")
        self.assertEqual(normalize_vista4d_camera_mode("orbit_left"), "arc_left_45")
        self.assertEqual(normalize_vista4d_camera_mode("none"), "dolly_zoom")

    def test_float_clamp_uses_default_and_bounds(self):
        self.assertEqual(clamp_float(None, 1.0, 0.1, 10.0), 1.0)
        self.assertEqual(clamp_float(0, 1.0, 0.1, 10.0), 0.1)
        self.assertEqual(clamp_float(99, 1.0, 0.1, 10.0), 10.0)

    def test_seed_defaults_to_random_on_missing_or_invalid_value(self):
        self.assertEqual(normalize_seed(None), -1)
        self.assertEqual(normalize_seed("12"), 12)
        self.assertEqual(normalize_seed("bad"), -1)


if __name__ == "__main__":
    unittest.main()
