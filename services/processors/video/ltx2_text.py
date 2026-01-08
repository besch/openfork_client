"""
LTX-2 Text-to-Video Processor

Uses Gemma-3 text encoder and LTX-2 19B model architecture.
Handles CreateVideo/SaveVideo output nodes.
"""

import logging

from config import DEV_MODE
from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import VideoOutputHandler
from utils.comfyui_workflow_utils import inject_prompt_into_ltx2_video_workflow


class LTX2TextToVideoJobProcessor(ComfyUIProcessor, VideoOutputHandler):
    """Processor for LTX-2 text-to-video generation."""

    def process(self):
        if DEV_MODE:
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        aspect_ratio = inputs.get("aspect_ratio", "16:9")

        wf_ready = inject_prompt_into_ltx2_video_workflow(
            workflow_data, self.positive_prompt, self.negative_prompt, aspect_ratio
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
