"""
Text-to-Image Processor
"""

import logging

from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import ImageOutputHandler
from utils.comfyui_workflow_utils import inject_prompt_into_qwen_workflow


class TextToImageJobProcessor(ComfyUIProcessor, ImageOutputHandler):
    """Processor for text-to-image generation."""

    def process(self):
        if not self.job:
            self._fail_job(f"Job object is None for TextToImageJobProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        aspect_ratio = inputs.get("aspect_ratio", "1:1")

        wf_ready = inject_prompt_into_qwen_workflow(
            workflow_data, self.positive_prompt, self.negative_prompt, aspect_ratio
        )
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        image_storage_path = self.handle_image_output(outputs)
        if not image_storage_path:
            return

        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=image_storage_path,
            thumbnail_storage_path=image_storage_path,
            prompt=self.positive_prompt,
        )
