import unittest

from dgn_client import DGNClient
from services.processors.video.scail import (
    SCAILImageToVideoProcessor,
    clamp_scail_duration,
    clamp_scail_steps,
    duration_to_wangp_frames,
    get_scail_vram_tier,
    scail_resolution,
)


class SCAILWan2GPTests(unittest.TestCase):
    def test_resolution_defaults_to_scail_512p_landscape(self):
        self.assertEqual(scail_resolution("16:9"), "896x512")
        self.assertEqual(scail_resolution("9:16"), "512x896")
        self.assertEqual(scail_resolution("unknown"), "896x512")

    def test_16gb_resolution_uses_lower_vram_profile(self):
        self.assertEqual(scail_resolution("16:9", "16gb"), "768x432")
        self.assertEqual(scail_resolution("9:16", "16gb"), "432x768")
        self.assertEqual(scail_resolution("unknown", "16gb"), "768x432")

    def test_duration_clamped_to_short_scail_window(self):
        self.assertEqual(clamp_scail_duration(None), 5.0)
        self.assertEqual(clamp_scail_duration(0.5), 1.0)
        self.assertEqual(clamp_scail_duration(8), 5.0)

    def test_16gb_duration_clamped_to_shorter_window(self):
        self.assertEqual(clamp_scail_duration(None, "16gb"), 4.0)
        self.assertEqual(clamp_scail_duration(0.5, "16gb"), 1.0)
        self.assertEqual(clamp_scail_duration(8, "16gb"), 4.0)

    def test_steps_clamped_to_supported_ui_range(self):
        self.assertEqual(clamp_scail_steps(None), 8)
        self.assertEqual(clamp_scail_steps(2), 6)
        self.assertEqual(clamp_scail_steps(30), 20)

    def test_duration_to_frames_aligns_with_wangp_frame_count(self):
        self.assertEqual(duration_to_wangp_frames(5), 81)
        self.assertEqual((duration_to_wangp_frames(4) - 1) % 4, 0)

    def test_service_type_selects_vram_tier(self):
        self.assertEqual(get_scail_vram_tier("scail-wan2gp-16gb"), "16gb")
        self.assertEqual(get_scail_vram_tier("scail-wan2gp-24gb"), "24gb")

    def test_scail_workflows_are_registered_in_processor_map(self):
        client = DGNClient.__new__(DGNClient)
        client.config = {
            "scail-image-to-video-16gb": {
                "processor": "SCAILImageToVideoProcessor",
            },
            "scail-image-to-video-24gb": {
                "processor": "SCAILImageToVideoProcessor",
            },
        }

        processor_map = DGNClient._build_processor_map(client)

        self.assertIs(
            processor_map["scail-image-to-video-16gb"],
            SCAILImageToVideoProcessor,
        )
        self.assertIs(
            processor_map["scail-image-to-video-24gb"],
            SCAILImageToVideoProcessor,
        )

    def test_processable_services_excludes_workflows_without_processors(self):
        client = DGNClient.__new__(DGNClient)
        client.config = {
            "scail-image-to-video-24gb": {
                "service_name": "scail-wan2gp-24gb",
                "processor": "SCAILImageToVideoProcessor",
            },
            "future-workflow": {
                "service_name": "future-service",
                "processor": "FutureProcessor",
            },
        }

        client.processor_map = DGNClient._build_processor_map(client)

        self.assertEqual(
            DGNClient._build_processable_services(client),
            {"scail-wan2gp-24gb"},
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
                "scail-image-to-video-16gb": DummyProcessor,
            }

        client.load_config = load_config

        processor = DGNClient._get_job_processor(
            client,
            {"id": "job-1", "workflow_type": "scail-image-to-video-16gb"},
            None,
        )

        self.assertIsInstance(processor, DummyProcessor)
        self.assertEqual(refreshes, [True])


if __name__ == "__main__":
    unittest.main()
