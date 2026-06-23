import tempfile
import unittest
from unittest.mock import Mock

from dgn_client import DGNClient
from services.processors.video.scail2 import (
    SCAIL2ImageToVideoProcessor,
    clamp_scail2_duration,
    clamp_scail2_steps,
    duration_to_wangp_frames,
    get_scail_vram_tier,
    parse_scail_bool,
    scail2_resolution,
)


class SCAILWan2GPTests(unittest.TestCase):
    def test_duration_to_frames_aligns_with_wangp_frame_count(self):
        self.assertEqual(duration_to_wangp_frames(5), 81)
        self.assertEqual((duration_to_wangp_frames(4) - 1) % 4, 0)

    def test_service_type_selects_vram_tier(self):
        self.assertEqual(get_scail_vram_tier("scail2-wan2gp-16gb"), "16gb")
        self.assertEqual(get_scail_vram_tier("scail2-wan2gp-24gb"), "24gb")

    def test_scail2_resolution_profiles_are_divisible_by_32(self):
        self.assertEqual(scail2_resolution("16:9", "24gb"), "832x480")
        self.assertEqual(scail2_resolution("9:16", "24gb"), "480x832")
        self.assertEqual(scail2_resolution("unknown", "24gb"), "832x480")
        width, height = [int(part) for part in scail2_resolution("16:9", "24gb").split("x")]
        self.assertEqual(width % 32, 0)
        self.assertEqual(height % 32, 0)

    def test_scail2_duration_and_steps_use_wangp_defaults(self):
        self.assertEqual(clamp_scail2_duration(None, "24gb"), 5.0)
        self.assertEqual(clamp_scail2_duration(99, "24gb"), 5.0)
        self.assertEqual(clamp_scail2_duration(0.5, "24gb"), 1.0)
        self.assertEqual(clamp_scail2_steps(None), 40)
        self.assertEqual(clamp_scail2_steps(2), 8)
        self.assertEqual(clamp_scail2_steps(80), 50)

    def test_scail_bool_parser_handles_string_flags(self):
        self.assertFalse(parse_scail_bool("false"))
        self.assertFalse(parse_scail_bool("0"))
        self.assertTrue(parse_scail_bool("true"))
        self.assertTrue(parse_scail_bool(1))

    def test_scail_pose_video_ignores_still_input_video_url(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = SCAIL2ImageToVideoProcessor.__new__(
                SCAIL2ImageToVideoProcessor
            )
            processor.job = {
                "id": "job-1",
                "input_video_url": "https://example.test/storage/reference.jpg?token=x",
            }
            processor.input_dir = tmpdir
            processor.orchestrator_service = Mock()

            self.assertIsNone(processor._resolve_pose_video({}))
            processor.orchestrator_service.download_asset_by_url.assert_not_called()
            processor.orchestrator_service.download_storage_asset.assert_not_called()

    def test_scail_pose_video_uses_signed_pose_video_url(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = SCAIL2ImageToVideoProcessor.__new__(
                SCAIL2ImageToVideoProcessor
            )
            processor.job = {"id": "job-1"}
            processor.input_dir = tmpdir
            processor.orchestrator_service = Mock()
            processor.orchestrator_service.download_asset_by_url.return_value = (
                f"{tmpdir}/pose.mp4"
            )

            result = processor._resolve_pose_video(
                {"pose_video_url": "https://example.test/storage/pose.mp4?token=x"}
            )

            self.assertEqual(result, f"{tmpdir}/pose.mp4")
            processor.orchestrator_service.download_asset_by_url.assert_called_once_with(
                "https://example.test/storage/pose.mp4?token=x",
                tmpdir,
            )

    def test_scail_pose_video_downloads_video_storage_path_with_bucket_hint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = SCAIL2ImageToVideoProcessor.__new__(
                SCAIL2ImageToVideoProcessor
            )
            processor.job = {"id": "job-1", "bucket": "projects_private"}
            processor.input_dir = tmpdir
            processor.orchestrator_service = Mock()
            processor.orchestrator_service.download_storage_asset.return_value = (
                f"{tmpdir}/pose.mp4"
            )

            result = processor._resolve_pose_video(
                {
                    "pose_video": "project/video/pose.mp4",
                    "pose_video_bucket": "projects_private",
                }
            )

            self.assertEqual(result, f"{tmpdir}/pose.mp4")
            processor.orchestrator_service.download_storage_asset.assert_called_once_with(
                "projects_private",
                "project/video/pose.mp4",
                tmpdir,
            )

    def test_scail_pose_video_rejects_non_video_storage_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = SCAIL2ImageToVideoProcessor.__new__(
                SCAIL2ImageToVideoProcessor
            )
            processor.job = {"id": "job-1", "bucket": "projects_private"}
            processor.input_dir = tmpdir
            processor.orchestrator_service = Mock()

            self.assertIsNone(
                processor._resolve_pose_video({"pose_video": "project/reference.jpg"})
            )
            processor.orchestrator_service.download_asset_by_url.assert_not_called()
            processor.orchestrator_service.download_storage_asset.assert_not_called()

    def test_scail2_workflows_are_registered_in_processor_map(self):
        client = DGNClient.__new__(DGNClient)
        client.services_config = {}
        client.config = {
            "scail2-image-to-video-24gb": {
                "processor": "SCAIL2ImageToVideoProcessor",
            },
        }

        processor_map = DGNClient._build_processor_map(client)

        self.assertIs(
            processor_map["scail2-image-to-video-24gb"],
            SCAIL2ImageToVideoProcessor,
        )

    def test_processable_services_excludes_workflows_without_processors(self):
        client = DGNClient.__new__(DGNClient)
        client.services_config = {}
        client.config = {
            "future-workflow": {
                "service_name": "future-service",
                "processor": "FutureProcessor",
            },
            "scail2-image-to-video-24gb": {
                "service_name": "scail2-wan2gp-24gb",
                "processor": "SCAIL2ImageToVideoProcessor",
            },
        }

        client.processor_map = DGNClient._build_processor_map(client)

        self.assertEqual(
            DGNClient._build_processable_services(client),
            {"scail2-wan2gp-24gb"},
        )

    def test_missing_processor_map_entry_refreshes_config_once(self):
        class DummyProcessor:
            def __init__(self, client, job, shutdown_event):
                self.client = client
                self.job = job
                self.shutdown_event = shutdown_event

        client = DGNClient.__new__(DGNClient)
        client.processor_map = {}
        client.config = {}
        refreshes = []

        def load_config():
            refreshes.append(True)
            client.processor_map = {
                "scail2-image-to-video-16gb": DummyProcessor,
            }

        client.load_config = load_config

        processor = DGNClient._get_job_processor(
            client,
            {"id": "job-1", "workflow_type": "scail2-image-to-video-16gb"},
            None,
        )

        self.assertIsInstance(processor, DummyProcessor)
        self.assertEqual(refreshes, [True])


if __name__ == "__main__":
    unittest.main()
