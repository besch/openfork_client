import unittest
from unittest import mock

from utils.media_utils import extract_last_frame, generate_thumbnail


class LastFrameExtractionTests(unittest.TestCase):
    def test_thumbnail_generation_uses_jpeg_compatible_pixel_format(self):
        with mock.patch("utils.media_utils.subprocess.run") as run_mock:
            with mock.patch("utils.media_utils.os.path.isfile", return_value=True):
                with mock.patch("utils.media_utils.os.path.getsize", return_value=123):
                    self.assertTrue(generate_thumbnail("input.mp4", "thumb.jpg", width=384))

        command = run_mock.call_args.args[0]
        self.assertIn("-pix_fmt", command)
        self.assertEqual(command[command.index("-pix_fmt") + 1], "yuvj420p")
        self.assertIn("-q:v", command)

    def test_thumbnail_generation_retries_first_frame_when_seek_outputs_nothing(self):
        with mock.patch("utils.media_utils.subprocess.run") as run_mock:
            with mock.patch("utils.media_utils.os.path.exists", return_value=False):
                with mock.patch("utils.media_utils.os.path.isfile", side_effect=[False, True]):
                    with mock.patch("utils.media_utils.os.path.getsize", return_value=123):
                        self.assertTrue(generate_thumbnail("input.mp4", "thumb.jpg", width=384))

        self.assertEqual(run_mock.call_count, 2)
        first_command = run_mock.call_args_list[0].args[0]
        retry_command = run_mock.call_args_list[1].args[0]
        self.assertEqual(first_command[first_command.index("-ss") + 1], "00:00:01.000")
        self.assertEqual(retry_command[retry_command.index("-ss") + 1], "00:00:00.000")

    def test_thumbnail_generation_requires_output_file(self):
        with mock.patch("utils.media_utils.subprocess.run"):
            with mock.patch("utils.media_utils.os.path.exists", return_value=False):
                with mock.patch("utils.media_utils.os.path.isfile", return_value=False):
                    self.assertFalse(generate_thumbnail("input.mp4", "thumb.jpg", width=384))

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
