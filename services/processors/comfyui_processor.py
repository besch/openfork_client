"""
ComfyUI Processor Base

Base class for all ComfyUI-based job processors.
"""

import logging
from .base import BaseJobProcessor


class ComfyUIProcessor(BaseJobProcessor):
    """Base class for ComfyUI-based workflow processors."""

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.comfyui_client = client.comfyui_client

    def _trigger_and_get_output(self, payload):
        """Trigger ComfyUI workflow and wait for output."""
        prompt_id = self.comfyui_client.trigger_workflow(payload)
        if not prompt_id:
            logging.error(f"Failed to trigger workflow for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, "failed")
            return None

        outputs = self.comfyui_client.get_workflow_output(
            prompt_id,
            job_id=self.job_id,
            orchestrator_service=self.orchestrator_service,
            timeout_sec=7200,
            shutdown_event=self.shutdown_event,
        )

        if self._check_interruption(outputs):
            return None

        if not outputs:
            logging.error(f"Workflow for job {self.job_id} failed to produce outputs.")
            self.orchestrator_service.update_job_status(self.job_id, "failed")
            return None

        return outputs
