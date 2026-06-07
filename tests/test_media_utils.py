import unittest
from unittest import mock

from utils.media_utils import extract_last_frame


class LastFrameExtractionTests(unittest.TestCase):
    def test_target_dimensions_add_resize_filter_to_ffmpeg(self):
        with mock.patch("utils.media_utils.get_video_duration", return_value=5.0):
            with mock.patch("utils.media_utils.subprocess.run") as run_mock:
                self.assertTrue(
                    extract_last_frame(
                        "input.mp4",
                        "last.jpg",
                        target_dimensions=(320, 176),
                    )
                )

        command = run_mock.call_args.args[0]
        self.assertIn("-vf", command)
        filter_arg = command[command.index("-vf") + 1]
        self.assertEqual(
            filter_arg,
            "scale=320:176:force_original_aspect_ratio=increase,crop=320:176,setsar=1",
        )

    def test_missing_target_dimensions_keep_original_frame_size(self):
        with mock.patch("utils.media_utils.get_video_duration", return_value=5.0):
            with mock.patch("utils.media_utils.subprocess.run") as run_mock:
                self.assertTrue(extract_last_frame("input.mp4", "last.jpg"))

        self.assertNotIn("-vf", run_mock.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
