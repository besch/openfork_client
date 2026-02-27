"""
Chatterbox TTS Processors
"""

import os
import logging
import json

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

        # Robust input retrieval
        inputs = self.job.get("inputs")
        if isinstance(inputs, str):
            try:
                inputs = json.loads(inputs)
            except:
                logging.warning(f"Failed to parse inputs string for job {self.job_id}")
                inputs = {}
        elif not isinstance(inputs, dict):
            inputs = {}
            
        logging.info(f"Chatterbox TTS job {self.job_id} inputs: {inputs}")
        
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

        logging.info(f"DEBUG: Full job data for {self.job_id}: {json.dumps(self.job, default=str)}")

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        # Robust input retrieval
        inputs = self.job.get("inputs")
        if isinstance(inputs, str):
            try:
                inputs = json.loads(inputs)
            except:
                logging.warning(f"Failed to parse inputs string for job {self.job_id}")
                inputs = {}
        elif not isinstance(inputs, dict):
            inputs = {}

        logging.info(f"Chatterbox voice clone job {self.job_id} inputs: {inputs}")
        
        # Try multiple potential keys for the voice clone URL
        voice_clone_urls = (
            inputs.get("voice_clone_urls") or 
            inputs.get("voice_clone_url") or 
            self.job.get("voice_clone_urls") or
            []
        )
        
        # If it's a single string, wrap it in a list
        if isinstance(voice_clone_urls, str):
            voice_clone_urls = [voice_clone_urls]
            
        if not voice_clone_urls:
            self._fail_job(f"Chatterbox voice clone job {self.job_id} missing 'voice_clone_urls'. Inputs keys: {list(inputs.keys())}")
            return

        # Download the voice reference audio
        clone_path = self.orchestrator_service.download_asset_by_url(
            voice_clone_urls[0], self.input_dir
        )
        if not clone_path:
            self._fail_job(f"Failed to download voice clone from {voice_clone_urls[0]} for job {self.job_id}.")
            return

        clone_filename = os.path.basename(clone_path)

        # Copy the audio file into the ComfyUI Docker container's input directory
        # so the LoadAudio node can find it
        try:
            from services.docker_manager import docker_manager
            container_input_path = f"/opt/ComfyUI/input/{clone_filename}"
            if docker_manager:
                docker_manager.copy_file_to_container(
                    service_type=self.client.active_service_type,
                    source_on_host=clone_path,
                    dest_in_container=container_input_path,
                    shutdown_event=self.shutdown_event,
                )
            else:
                # Headless Mode: already inside the container, copy locally
                import shutil
                os.makedirs(os.path.dirname(container_input_path), exist_ok=True)
                shutil.copy2(clone_path, container_input_path)
                logging.info(f"Copied voice clone to {container_input_path} (Headless)")
        except Exception as e:
            self._fail_job(f"Failed to copy voice clone audio to container for job {self.job_id}: {e}")
            return

        exaggeration = inputs.get("exaggeration", 0.5)
        cfg_weight = inputs.get("cfg_weight", 0.5)
        temperature = inputs.get("temperature", 0.8)
        seed = inputs.get("seed")

        wf_ready = inject_prompt_and_clone_into_chatterbox_workflow(
            workflow_data,
            self.positive_prompt,
            clone_filename,
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
