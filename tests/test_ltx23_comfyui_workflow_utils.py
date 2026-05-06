import json
import unittest
from pathlib import Path

from utils.comfyui_workflow_utils import inject_prompt_into_ltx23_video_workflow


WORKFLOW_DIR = Path(__file__).resolve().parents[1] / "workflows"
LTX23_COMFYUI_WORKFLOWS = (
    "ltx23-comfyui-text-to-video-8gb.api.json",
    "ltx23-comfyui-text-to-video-12gb.api.json",
    "ltx23-comfyui-text-to-video-16gb.api.json",
    "ltx23-comfyui-text-to-video-24gb.api.json",
    "ltx23-comfyui-image-to-video-8gb.api.json",
    "ltx23-comfyui-image-to-video-12gb.api.json",
    "ltx23-comfyui-image-to-video-16gb.api.json",
    "ltx23-comfyui-image-to-video-24gb.api.json",
)


def load_workflow(filename: str) -> dict:
    with (WORKFLOW_DIR / filename).open("r", encoding="utf-8") as handle:
        return json.load(handle)


class LTX23ComfyUIWorkflowUtilsTests(unittest.TestCase):
    def test_shipped_ltx23_comfyui_workflows_wire_audio_vae_into_node_8(self):
        for filename in LTX23_COMFYUI_WORKFLOWS:
            with self.subTest(filename=filename):
                workflow = load_workflow(filename)
                self.assertEqual(
                    workflow["prompt"]["8"]["inputs"].get("audio_vae"),
                    ["7", 0],
                )

    def test_injector_repairs_stale_ltx23_workflows_missing_audio_vae(self):
        workflow = load_workflow("ltx23-comfyui-text-to-video-8gb.api.json")
        workflow["prompt"]["8"]["inputs"].pop("audio_vae", None)

        injected = inject_prompt_into_ltx23_video_workflow(
            workflow,
            prompt="A fox running through snow",
            negative_prompt="blurry",
            aspect_ratio="16:9",
            tier="8gb",
        )

        self.assertEqual(injected["8"]["inputs"]["audio_vae"], ["7", 0])


if __name__ == "__main__":
    unittest.main()
