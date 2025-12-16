"""
Foley Audio Processor
"""

import os
import logging

from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import AudioOutputHandler
from utils.comfyui_workflow_utils import inject_video_and_prompt_into_foley_workflow


class FoleyJobProcessor(ComfyUIProcessor, AudioOutputHandler):
    """Processor for Foley sound effect generation from video."""

    def process(self):
        if not self.job:
            self._fail_job(f"Job object is None for FoleyJobProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        input_video_url = self.job.get("input_video_url")
        if not input_video_url:
            self._fail_job(f"Foley job {self.job_id} missing 'input_video_url'.")
            return

        video_path = self.orchestrator_service.download_asset_by_url(input_video_url, self.input_dir)
        if not video_path:
            self._fail_job(f"Failed to download input video for foley job {self.job_id}.")
            return

        video_filename = os.path.basename(video_path)
        wf_ready = inject_video_and_prompt_into_foley_workflow(
            workflow_data, video_filename, self.positive_prompt, self.negative_prompt
        )

        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        result = self.handle_audio_output(outputs)
        if not result:
            return

        audio_storage_path, duration = result
        completion_metadata = self.job.get("completion_metadata") or {}

        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=audio_storage_path,
            duration_seconds=duration,
            completion_metadata=completion_metadata,
        )
