import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from services.processors.video.last_frame import (
    materialize_last_frame_start_image,
    resolve_input_video_url,
)


class LastFrameHelperTests(unittest.TestCase):
    def _processor(self, tmpdir, job=None):
        processor = Mock()
        processor.job = job or {}
        processor.job_id = processor.job.get("id", "job-1")
        processor.input_dir = tmpdir
        processor.client.config = {"SUPABASE_URL": "https://example.supabase.co"}
        processor.orchestrator_service.download_asset_by_url.return_value = (
            os.path.join(tmpdir, "source.mp4")
        )
        return processor

    def test_materializes_last_frame_from_input_video_url(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._processor(
                tmpdir,
                {
                    "id": "ltx-job",
                    "workflow_type": "ltx23-image-to-video-from-last-frame-16gb",
                },
            )
            inputs = {"input_video_url": "https://assets.example/source.mp4"}

            with patch(
                "services.processors.video.last_frame.extract_last_frame",
                return_value=True,
            ) as extract_last_frame:
                output_path = materialize_last_frame_start_image(
                    processor,
                    inputs,
                    target_dimensions=(768, 432),
                )

            self.assertEqual(output_path, os.path.join(tmpdir, "ltx-job_last_frame.jpg"))
            processor.orchestrator_service.download_asset_by_url.assert_called_once_with(
                "https://assets.example/source.mp4",
                tmpdir,
            )
            extract_last_frame.assert_called_once_with(
                os.path.join(tmpdir, "source.mp4"),
                os.path.join(tmpdir, "ltx-job_last_frame.jpg"),
                target_dimensions=(768, 432),
            )

    def test_from_last_frame_can_use_legacy_start_image_url_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._processor(
                tmpdir,
                {
                    "id": "dreamid-job",
                    "workflow_type": "dreamid-omni-image-to-video-from-last-frame-24gb",
                },
            )
            inputs = {"start_image_url": "https://assets.example/source.mp4"}

            self.assertEqual(
                resolve_input_video_url(processor, inputs),
                "https://assets.example/source.mp4",
            )

    def test_ignores_plain_image_storage_path_for_normal_i2v(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._processor(
                tmpdir,
                {
                    "id": "scail2-job",
                    "workflow_type": "scail2-image-to-video-16gb",
                    "input_storage_path": "images/reference.png",
                },
            )

            self.assertIsNone(materialize_last_frame_start_image(processor, {}))
            processor.orchestrator_service.download_asset_by_url.assert_not_called()

    def test_ignores_plain_image_input_video_url_for_normal_i2v(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._processor(
                tmpdir,
                {
                    "id": "scail2-job",
                    "workflow_type": "scail2-image-to-video-16gb",
                },
            )
            inputs = {
                "input_video_url": (
                    "https://example.supabase.co/storage/v1/object/sign/"
                    "projects_private/reference.jpg?token=signed"
                )
            }

            self.assertIsNone(materialize_last_frame_start_image(processor, inputs))
            processor.orchestrator_service.download_asset_by_url.assert_not_called()

    def test_accepts_video_input_url_for_normal_i2v(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._processor(
                tmpdir,
                {
                    "id": "video-job",
                    "workflow_type": "scail2-image-to-video-16gb",
                },
            )
            inputs = {
                "input_video_url": (
                    "https://example.supabase.co/storage/v1/object/sign/"
                    "projects_private/source.mp4?token=signed"
                )
            }

            with patch(
                "services.processors.video.last_frame.extract_last_frame",
                return_value=True,
            ):
                self.assertEqual(
                    materialize_last_frame_start_image(processor, inputs),
                    os.path.join(tmpdir, "video-job_last_frame.jpg"),
                )

    def test_resolves_video_storage_path_to_public_url(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = self._processor(
                tmpdir,
                {
                    "id": "davinci-job",
                    "workflow_type": "davinci-magihuman-image-to-video-from-last-frame-16gb",
                    "bucket": "projects_public",
                    "input_storage_path": "videos/source.mp4",
                },
            )

            with patch.dict(
                os.environ,
                {"SUPABASE_URL": "https://example.supabase.co"},
            ):
                self.assertEqual(
                    resolve_input_video_url(processor, {}),
                    "https://example.supabase.co/storage/v1/object/public/projects_public/videos/source.mp4",
                )


if __name__ == "__main__":
    unittest.main()
