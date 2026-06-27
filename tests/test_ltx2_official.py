import json
import unittest
from pathlib import Path

from dgn_client import DGNClient
from services.processors.video.ltx2_official import (
    LTX2PipelinesI2VLoraProcessor,
    LTX2PipelinesTextToVideoProcessor,
)


class LTX2OfficialProcessorTests(unittest.TestCase):
    def test_ltx2_pipelines_workflows_are_registered_in_processor_map(self):
        client = DGNClient.__new__(DGNClient)
        client.services_config = {}
        client.config = {
            "ltx2-pipelines-text-to-video-24gb": {
                "service_name": "ltx2-pipelines-lora-24gb",
                "processor": "LTX2PipelinesTextToVideoProcessor",
            },
            "ltx2-pipelines-i2v-lora-24gb": {
                "service_name": "ltx2-pipelines-lora-24gb",
                "processor": "LTX2PipelinesI2VLoraProcessor",
            },
        }

        processor_map = DGNClient._build_processor_map(client)

        self.assertIs(
            processor_map["ltx2-pipelines-text-to-video-24gb"],
            LTX2PipelinesTextToVideoProcessor,
        )
        self.assertIs(
            processor_map["ltx2-pipelines-i2v-lora-24gb"],
            LTX2PipelinesI2VLoraProcessor,
        )

    def test_ltx2_services_advertise_text_and_image_capabilities(self):
        registry_path = Path(__file__).resolve().parents[2] / "website" / "services.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))

        for tier in ("16gb", "24gb", "32gb"):
            service_name = f"ltx2-pipelines-lora-{tier}"
            text_workflow = f"ltx2-pipelines-text-to-video-{tier}"
            i2v_workflow = f"ltx2-pipelines-i2v-lora-{tier}"

            with self.subTest(tier=tier):
                capabilities = registry["services"][service_name]["video_capabilities"]
                self.assertEqual(capabilities["text"], text_workflow)
                self.assertEqual(capabilities["image"], i2v_workflow)
                self.assertEqual(
                    registry["workflows"][text_workflow]["processor"],
                    "LTX2PipelinesTextToVideoProcessor",
                )
                self.assertEqual(
                    registry["workflows"][text_workflow]["agent_operation"],
                    "text-to-video",
                )


if __name__ == "__main__":
    unittest.main()
