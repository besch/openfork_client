"""
WAN 2.2 Text-to-Video Processor
"""

import os
import logging

from config import DEV_MODE
from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import VideoOutputHandler
from utils.comfyui_workflow_utils import inject_prompt_into_text_to_video_workflow


class WAN22TextToVideoJobProcessor(ComfyUIProcessor, VideoOutputHandler):
    """Processor for WAN 2.2 text-to-video generation."""

    def process(self):
        if DEV_MODE:
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        aspect_ratio = inputs.get("aspect_ratio", "16:9")

        wf_ready = inject_prompt_into_text_to_video_workflow(
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
        )

        self._maybe_submit_upscale_job(video_storage_path, inputs)

    def _maybe_submit_upscale_job(self, video_storage_path: str, workflow_config: dict):
        """Submit upscale job if enabled in config."""
        if workflow_config.get("upscale_enabled"):
            logging.info(f"Upscale enabled for job {self.job_id}. Submitting upscale job.")
            upscale_params = workflow_config.get("upscale_params", {})

            submit_body = {
                "sceneId": self.job.get("scene_id"),
                "branchId": self.job.get("branch_id"),
                "model": "esrgan-upscaler",
                "prompt": "Upscaling video",
                "input_storage_path": video_storage_path,
                "upscale_params": upscale_params,
                "originalJobId": self.job.get("id"),
            }

            new_job_id = self.orchestrator_service.submit_job(submit_body)
            if new_job_id:
                logging.info(f"Successfully submitted upscale job: {new_job_id}")
            else:
                logging.error("Failed to submit upscale job.")
