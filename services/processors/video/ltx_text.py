"""
LTX-Video Text-to-Video Processor
"""

import logging

from config import DEV_MODE
from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import VideoOutputHandler
from utils.comfyui_workflow_utils import inject_prompt_into_ltx_video_workflow


class LTXTextToVideoJobProcessor(ComfyUIProcessor, VideoOutputHandler):
    """Processor for LTX-Video text-to-video generation."""

    def process(self):
        if DEV_MODE:
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        aspect_ratio = inputs.get("aspect_ratio", "16:9")
        cfg_scale = inputs.get("cfg_scale")
        steps = inputs.get("steps")
        flow_shift = inputs.get("flow_shift")
        sampler = inputs.get("sampler")
        scheduler = inputs.get("scheduler")

        wf_ready = inject_prompt_into_ltx_video_workflow(
            workflow_data, self.positive_prompt, self.negative_prompt, aspect_ratio,
            cfg_scale=cfg_scale, steps=steps,
            flow_shift=flow_shift, sampler=sampler, scheduler=scheduler
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
