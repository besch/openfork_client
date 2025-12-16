"""
VibeVoice TTS Processors
"""

import os
import logging

from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import AudioOutputHandler
from utils.comfyui_workflow_utils import (
    inject_prompt_into_vibevoice_workflow,
    inject_script_and_clones_into_vibevoice_workflow,
)


class VibeVoiceJobProcessor(ComfyUIProcessor, AudioOutputHandler):
    """Processor for VibeVoice text-to-speech with single speaker."""

    def process(self):
        if not self.job:
            self._fail_job(f"Job object is None for VibeVoiceJobProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        cfg_scale = inputs.get("cfg_scale", 3.5)
        diffusion_steps = inputs.get("diffusion_steps", 10)
        temperature = inputs.get("temperature", 0.8)
        top_p = inputs.get("top_p", 0.95)
        seed = inputs.get("seed")
        voice_id = inputs.get("voice_id", "Alice")

        wf_ready = inject_prompt_into_vibevoice_workflow(
            workflow_data,
            self.positive_prompt,
            cfg_scale=cfg_scale,
            diffusion_steps=diffusion_steps,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            voice_id=voice_id,
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


class VibeVoiceMultiCloneJobProcessor(ComfyUIProcessor, AudioOutputHandler):
    """Processor for VibeVoice multi-speaker voice cloning."""

    def process(self):
        if not self.job:
            self._fail_job(f"Job object is None for VibeVoiceMultiCloneJobProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        voice_clone_urls = inputs.get("voice_clone_urls", [])
        if not voice_clone_urls:
            self._fail_job(f"VibeVoice multi-clone job {self.job_id} missing 'voice_clone_urls'.")
            return

        clone_paths = []
        for url in voice_clone_urls:
            clone_path = self.orchestrator_service.download_asset_by_url(url, self.input_dir)
            if not clone_path:
                self._fail_job(f"Failed to download voice clone from {url} for job {self.job_id}.")
                return
            clone_paths.append(os.path.basename(clone_path))

        cfg_scale = inputs.get("cfg_scale", 3.5)
        diffusion_steps = inputs.get("diffusion_steps", 10)
        temperature = inputs.get("temperature", 0.8)
        top_p = inputs.get("top_p", 0.95)
        seed = inputs.get("seed")

        wf_ready = inject_script_and_clones_into_vibevoice_workflow(
            workflow_data,
            self.positive_prompt,
            clone_paths,
            cfg_scale=cfg_scale,
            diffusion_steps=diffusion_steps,
            temperature=temperature,
            top_p=top_p,
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
