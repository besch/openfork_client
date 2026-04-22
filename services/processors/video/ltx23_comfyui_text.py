"""LTX-2.3 Text-to-Video Processor (ComfyUI backend, 24GB VRAM)."""

import logging

from config import DEV_MODE
from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import VideoOutputHandler
from utils.comfyui_workflow_utils import (
    inject_prompt_into_ltx23_video_workflow,
    get_ltx23_dimensions,
)


class LTX23ComfyUITextToVideoProcessor(ComfyUIProcessor, VideoOutputHandler):
    """ComfyUI-based LTX-2.3 T2V processor. Applies the distilled LoRA via
    LoraLoaderModelOnly (node 2 in the workflow) to fix the grey-colour issue."""

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
            workflow_data, self.positive_prompt, self.negative_prompt,
            aspect_ratio, steps=steps, cfg_scale=cfg_scale, tier="24gb",
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
        logging.info(
            f"LTX-2.3 ComfyUI T2V job {self.job_id} completed: {video_storage_path}"
        )
