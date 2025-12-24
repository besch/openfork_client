"""
Chatterbox TTS Processors
"""

import os
import logging

from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import AudioOutputHandler
from utils.comfyui_workflow_utils import (
    inject_prompt_into_chatterbox_workflow,
    inject_prompt_and_clone_into_chatterbox_workflow,
)


class ChatterboxTTSJobProcessor(ComfyUIProcessor, AudioOutputHandler):
    """Processor for Chatterbox text-to-speech."""

    def process(self):
        if not self.job:
            self._fail_job(f"Job object is None for ChatterboxTTSJobProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        exaggeration = inputs.get("exaggeration", 0.5)
        cfg_weight = inputs.get("cfg_weight", 0.5)
        temperature = inputs.get("temperature", 0.8)
        seed = inputs.get("seed")

        wf_ready = inject_prompt_into_chatterbox_workflow(
            workflow_data,
            self.positive_prompt,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            seed=seed,
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


class ChatterboxVoiceCloneJobProcessor(ComfyUIProcessor, AudioOutputHandler):
    """Processor for Chatterbox voice cloning TTS."""

    def process(self):
        if not self.job:
            self._fail_job(f"Job object is None for ChatterboxVoiceCloneJobProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        voice_clone_urls = inputs.get("voice_clone_urls", [])
        if not voice_clone_urls:
            self._fail_job(f"Chatterbox voice clone job {self.job_id} missing 'voice_clone_urls'.")
            return

        # Download the voice reference audio
        clone_path = self.orchestrator_service.download_asset_by_url(
            voice_clone_urls[0], self.input_dir
        )
        if not clone_path:
            self._fail_job(f"Failed to download voice clone from {voice_clone_urls[0]} for job {self.job_id}.")
            return

        exaggeration = inputs.get("exaggeration", 0.5)
        cfg_weight = inputs.get("cfg_weight", 0.5)
        temperature = inputs.get("temperature", 0.8)
        seed = inputs.get("seed")

        wf_ready = inject_prompt_and_clone_into_chatterbox_workflow(
            workflow_data,
            self.positive_prompt,
            os.path.basename(clone_path),
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            seed=seed,
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
