"""
LTX-2.3 Text-to-Video Processor (24GB VRAM tier)

Generates synchronized audio+video from a text prompt.
24GB tier runs at 768x432 @ 121 frames (~5s).
Uses the distilled model for fast 8-step generation.
Requires 64GB+ system RAM for CPU offloading the 46GB BF16 model.
"""

from config import DEV_MODE
from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import VideoOutputHandler
from utils.comfyui_workflow_utils import inject_prompt_into_ltx23_video_workflow


class LTX23TextToVideo24GBJobProcessor(ComfyUIProcessor, VideoOutputHandler):
    """Processor for LTX-2.3 text-to-video + audio generation (24GB tier)."""

    def process(self):
        if DEV_MODE:
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        aspect_ratio = inputs.get("aspect_ratio", "16:9")
        steps = inputs.get("steps")
        cfg_scale = inputs.get("cfg_scale")

        wf_ready = inject_prompt_into_ltx23_video_workflow(
            workflow_data, self.positive_prompt, self.negative_prompt, aspect_ratio,
            steps=steps, cfg_scale=cfg_scale, tier="24gb"
        )
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        result = self.handle_video_output(outputs)
        if not result:
            return

        video_storage_path, thumbnail_storage_path, duration = result
        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=video_storage_path,
            thumbnail_storage_path=thumbnail_storage_path,
            duration_seconds=duration,
            prompt=self.positive_prompt,
        )