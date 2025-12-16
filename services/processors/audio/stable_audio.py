"""
Stable Audio Processor
"""

import logging

from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import AudioOutputHandler
from utils.comfyui_workflow_utils import inject_prompt_into_stable_audio_workflow


class StableAudioJobProcessor(ComfyUIProcessor, AudioOutputHandler):
    """Processor for Stable Audio sound effect generation."""

    def process(self):
        if not self.job:
            self._fail_job(f"Job object is None for StableAudioJobProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        duration = inputs.get("duration_seconds", 5)

        wf_ready = inject_prompt_into_stable_audio_workflow(workflow_data, self.positive_prompt, duration)
        payload = {"prompt": wf_ready}

        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        result = self.handle_audio_output(outputs, scan_directory=True)
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
