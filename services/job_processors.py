import os
import json
import logging
from typing import Union, Dict
from abc import ABC, abstractmethod
from services.orchestrator_service import TokenExpiredError
from config import DEV_MODE
from services.docker_manager import docker_manager


class CriticalWorkflowError(Exception):
    """Exception raised when a workflow fails critically and the client should terminate."""
    pass
from utils.media_utils import get_audio_duration, find_audio_in_output, find_audio_file_in_directory, find_image_in_output, find_video_in_output, generate_thumbnail, get_video_duration, get_video_dimensions, get_video_framerate, extract_last_frame
from utils.comfyui_workflow_utils import (
    inject_prompt_and_image_into_workflow,
    inject_video_and_prompt_into_foley_workflow,
    inject_prompt_into_qwen_workflow,
    inject_prompt_into_text_to_video_workflow,
    inject_prompt_into_vibevoice_workflow,
    inject_script_and_clones_into_vibevoice_workflow,
    inject_prompt_into_diffrhythm_workflow,
    inject_prompt_into_stable_audio_workflow,
    materialize_start_image,
    inject_video_into_upscaler_workflow,
    inject_prompt_into_ltx_video_workflow,
    inject_prompt_and_image_into_ltx_video_workflow,
    inject_prompt_into_hunyuan_workflow,
    inject_prompt_and_image_into_hunyuan_workflow
)
import random


class BaseJobProcessor(ABC):
    def __init__(self, client, job, shutdown_event):
        self.client = client
        self.orchestrator_service = client.orchestrator_service
        self.comfyui_client = client.comfyui_client
        self.job = job
        self.job_id = job['id']
        self.shutdown_event = shutdown_event
        self.root_dir = client.root_dir
        self.input_dir = client.input_dir
        self.cache_dir = client.cache_dir
        self.positive_prompt = job.get('prompt') or ""
        self.negative_prompt = job.get('negative_prompt') or ""
        self.workflow_type = job.get('workflow_type')

    @property
    def workflow_file(self) -> str:
        """The filename of the workflow to be used for this processor."""
        if self.workflow_type in self.client.config:
            return self.client.config[self.workflow_type]["workflow_file"]
        raise ValueError(f"Workflow file not found for type {self.workflow_type}")

    def _get_workflow_payload(self) -> Union[Dict, None]:
        """Loads the workflow from the local filesystem."""
        try:
            local_filename = self.workflow_file
        except ValueError as e:
            logging.error(f"Cannot get workflow payload for job {self.job_id}: {e}")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Cannot get workflow payload for job {self.job_id}: {e}")

        # Construct the full path to the workflow file.
        workflow_path = os.path.join(self.root_dir, 'workflows', local_filename)
        
        logging.info(f"Loading local workflow for job {self.job_id} from: {workflow_path}")

        try:
            with open(workflow_path, 'r') as f:
                workflow_data = json.load(f)
            return workflow_data
        except FileNotFoundError:
            logging.error(f"Local workflow file not found: {workflow_path}")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Local workflow file not found: {workflow_path}")
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from workflow file: {workflow_path}")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Failed to decode JSON from workflow file: {workflow_path}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading local workflow {local_filename}: {e}")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"An unexpected error occurred while loading local workflow {local_filename}: {e}")

    def _check_interruption(self, outputs):
        if outputs == "interrupted":
            logging.warning(f"Processing of job {self.job_id} was interrupted.")
            return True
        return False

    def _copy_file_from_container(self, filename: str, subfolder: str) -> Union[str, None]:
        """Copies a file from the active container to a temporary location on the host."""
        if not self.client.active_service_type:
            logging.error("Cannot copy from container: no active service type is set.")
            return None

        # Sanitize filename to prevent path traversal
        safe_filename = os.path.basename(filename)

        source_in_container = os.path.join("/opt/ComfyUI/output", subfolder, safe_filename).replace('\\', '/')
        
        os.makedirs(self.cache_dir, exist_ok=True)

        temp_filename = f"{self.job_id}_{safe_filename}"
        dest_on_host = os.path.join(self.cache_dir, temp_filename)

        try:
            docker_manager.copy_file_from_container(
                service_type=self.client.active_service_type,
                source_in_container=source_in_container,
                dest_on_host=dest_on_host,
                shutdown_event=self.shutdown_event
            )
            if os.path.exists(dest_on_host):
                logging.info(f"Successfully copied file to temporary host path: {dest_on_host}")
                return dest_on_host
            else:
                raise RuntimeError("docker cp command finished but destination file does not exist.")
        except Exception as e:
            logging.error(f"Failed to copy file from container: {e}", exc_info=True)
            raise CriticalWorkflowError(f"Failed to copy file from container: {e}")

    def _trigger_and_get_output(self, payload):
        prompt_id = self.comfyui_client.trigger_workflow(payload)
        if not prompt_id:
            logging.error(f"Failed to trigger workflow for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Failed to trigger workflow for job {self.job_id}.")

        outputs = self.comfyui_client.get_workflow_output(
            prompt_id,
            job_id=self.job_id,
            orchestrator_service=self.orchestrator_service,
            timeout_sec=7200,
            shutdown_event=self.shutdown_event
        )
        if self._check_interruption(outputs):
            return None
        
        if not outputs:
            logging.error(f"Workflow for job {self.job_id} failed to produce outputs.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Workflow for job {self.job_id} failed to produce outputs.")
            
        return outputs

    @abstractmethod
    def process(self):
        pass

class StableAudioJobProcessor(BaseJobProcessor):
    def process(self):
        if not self.job:
            logging.error(f"Job object is None for StableAudioJobProcessor. Cannot proceed.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError("Job object is None for StableAudioJobProcessor.")

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get('inputs', {})
        duration = inputs.get('duration_seconds', 5)

        wf_ready = inject_prompt_into_stable_audio_workflow(workflow_data, self.positive_prompt, duration)
        payload = {"prompt": wf_ready}

        # StableAudio_ node pipes output to SaveAudio node for standard ComfyUI outputs
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        # First try standard output format (in case the custom node is updated to support it)
        audio_info = find_audio_in_output(outputs)

        # If not found in standard outputs, scan the output directory for audio files
        if not audio_info:
            logging.info(f"Audio not found in workflow outputs for job {self.job_id}, scanning output directory...")
            # The output directory path on the host (Docker volume mount)
            if self.client.active_service_type:
                output_dir = docker_manager.get_output_dir_path(self.client.active_service_type)
                audio_info = find_audio_file_in_directory(output_dir, self.job_id)
            
        if not audio_info:
            logging.error(f"StableAudio workflow for job {self.job_id} completed, but no audio file found. (Block 1)")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"StableAudio workflow for job {self.job_id} completed, but no audio file found. (Block 1)")

        audio_filename, subfolder = audio_info
        temp_host_path = self._copy_file_from_container(audio_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}. (Block 2)")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        try:
            audio_storage_path = self.orchestrator_service.upload_audio_output(temp_host_path, self.job_id)
            if audio_storage_path:
                duration = get_audio_duration(temp_host_path)
                completion_metadata = self.job.get('completion_metadata') or {}
                self.orchestrator_service.update_job_status(self.job_id, 'completed', storage_path=audio_storage_path, duration_seconds=duration, completion_metadata=completion_metadata)
            else:
                logging.error(f"StableAudio job {self.job_id} completed, but audio upload failed. (Block 3)")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                raise CriticalWorkflowError(f"StableAudio job {self.job_id} completed, but audio upload failed. (Block 3)")
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)


class FoleyJobProcessor(BaseJobProcessor):
    def process(self):
        if not self.job:
            logging.error(f"Job object is None for FoleyJobProcessor. Cannot proceed.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError("Job object is None for FoleyJobProcessor.")

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        input_video_url = self.job.get('input_video_url')
        if not input_video_url:
            logging.error(f"Foley job {self.job_id} missing 'input_video_url'. (Block 4)")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Foley job {self.job_id} missing 'input_video_url'. (Block 4)")

        video_path = self.orchestrator_service.download_asset_by_url(input_video_url, self.input_dir)
        if not video_path:
            logging.error(f"Failed to download input video for foley job {self.job_id}. (Block 5)")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Failed to download input video for foley job {self.job_id}. (Block 5)")
        
        video_filename = os.path.basename(video_path)
        wf_ready = inject_video_and_prompt_into_foley_workflow(workflow_data, video_filename, self.positive_prompt, self.negative_prompt)
        
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        audio_info = find_audio_in_output(outputs)
        if not audio_info:
            logging.error(f"Foley workflow for job {self.job_id} completed, but no audio file found. (Block 6)")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Foley workflow for job {self.job_id} completed, but no audio file found. (Block 6)")

        audio_filename, subfolder = audio_info
        temp_host_path = self._copy_file_from_container(audio_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}. (Block 7)")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        try:
            audio_storage_path = self.orchestrator_service.upload_audio_output(temp_host_path, self.job_id)
            if audio_storage_path:
                duration = get_audio_duration(temp_host_path)
                completion_metadata = self.job.get('completion_metadata') or {}
                self.orchestrator_service.update_job_status(self.job_id, 'completed', storage_path=audio_storage_path, duration_seconds=duration, completion_metadata=completion_metadata)
            else:
                logging.error(f"Foley job {self.job_id} completed, but audio upload failed. (Block 8)")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                raise CriticalWorkflowError(f"Foley job {self.job_id} completed, but audio upload failed. (Block 8)")
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class TextToImageJobProcessor(BaseJobProcessor):
    def process(self):
        if not self.job:
            logging.error(f"Job object is None for TextToImageJobProcessor. Cannot proceed.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError("Job object is None for TextToImageJobProcessor.")

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get('inputs', {})
        aspect_ratio = inputs.get('aspect_ratio', '1:1')
        wf_ready = inject_prompt_into_qwen_workflow(workflow_data, self.positive_prompt, self.negative_prompt, aspect_ratio)
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        image_info = find_image_in_output(outputs)
        if not image_info:
            logging.error(f"Workflow for job {self.job_id} completed, but no image file found. (Block 9)")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Workflow for job {self.job_id} completed, but no image file found. (Block 9)")

        image_filename, subfolder = image_info
        temp_host_path = self._copy_file_from_container(image_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}. (Block 10)")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        try:
            image_storage_path = self.orchestrator_service.upload_image_output(temp_host_path, self.job_id)
            if image_storage_path:
                self.orchestrator_service.update_job_status(self.job_id, 'completed', storage_path=image_storage_path, thumbnail_storage_path=image_storage_path, prompt=self.positive_prompt)
            else:
                logging.error(f"Image upload failed for job {self.job_id}. (Block 11)")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                raise CriticalWorkflowError(f"Image upload failed for job {self.job_id}. (Block 11)")
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class VibeVoiceJobProcessor(BaseJobProcessor):
    def process(self):
        if not self.job:
            logging.error(f"Job object is None for VibeVoiceJobProcessor. Cannot proceed.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError("Job object is None for VibeVoiceJobProcessor.")

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get('inputs', {})
        cfg_scale = inputs.get('cfg_scale', 3.5)
        diffusion_steps = inputs.get('diffusion_steps', 10)
        audio_info = find_audio_in_output(outputs)
        if not audio_info:
            logging.error(f"VibeVoice workflow for job {self.job_id} completed, but no audio file found. (Block 12)")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"VibeVoice workflow for job {self.job_id} completed, but no audio file found. (Block 12)")

        audio_filename, subfolder = audio_info
        temp_host_path = self._copy_file_from_container(audio_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}. (Block 13)")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        try:
            audio_storage_path = self.orchestrator_service.upload_audio_output(temp_host_path, self.job_id)
            if audio_storage_path:
                duration = get_audio_duration(temp_host_path)
                completion_metadata = self.job.get('completion_metadata') or {}
                self.orchestrator_service.update_job_status(self.job_id, 'completed', storage_path=audio_storage_path, duration_seconds=duration, completion_metadata=completion_metadata)
            else:
                logging.error(f"VibeVoice job {self.job_id} completed, but audio upload failed. (Block 14)")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                raise CriticalWorkflowError(f"VibeVoice job {self.job_id} completed, but audio upload failed. (Block 14)")
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class VibeVoiceMultiCloneJobProcessor(BaseJobProcessor):
    def process(self):
        if not self.job:
            logging.error(f"Job object is None for VibeVoiceMultiCloneJobProcessor. Cannot proceed.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError("Job object is None for VibeVoiceMultiCloneJobProcessor.")

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get('inputs', {})
        voice_clone_urls = inputs.get('voice_clone_urls', [])
        if not voice_clone_urls:
            logging.error(f"VibeVoice multi-clone job {self.job_id} missing 'voice_clone_urls'. (Block 15)")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"VibeVoice multi-clone job {self.job_id} missing 'voice_clone_urls'. (Block 15)")

        clone_paths = []
        for url in voice_clone_urls:
            clone_path = self.orchestrator_service.download_asset_by_url(url, self.input_dir)
            if not clone_path:
                logging.error(f"Failed to download voice clone from {url} for job {self.job_id}. (Block 16)")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                raise CriticalWorkflowError(f"Failed to download voice clone from {url} for job {self.job_id}. (Block 16)")
            clone_paths.append(os.path.basename(clone_path))

        inputs = self.job.get('inputs', {})
        cfg_scale = inputs.get('cfg_scale', 3.5)
        diffusion_steps = inputs.get('diffusion_steps', 10)
        temperature = inputs.get('temperature', 0.8)
        top_p = inputs.get('top_p', 0.95)
        seed = inputs.get('seed')

        wf_ready = inject_script_and_clones_into_vibevoice_workflow(
            workflow_data, 
            self.positive_prompt, 
            clone_paths,
            cfg_scale=cfg_scale,
            diffusion_steps=diffusion_steps,
            temperature=temperature,
            top_p=top_p,
            seed=seed
        )
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        audio_info = find_audio_in_output(outputs)
        if not audio_info:
            logging.error(f"VibeVoice multi-clone workflow for job {self.job_id} completed, but no audio file found. (Block 17)")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"VibeVoice multi-clone workflow for job {self.job_id} completed, but no audio file found. (Block 17)")

        audio_filename, subfolder = audio_info
        temp_host_path = self._copy_file_from_container(audio_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}. (Block 18)")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        try:
            audio_storage_path = self.orchestrator_service.upload_audio_output(temp_host_path, self.job_id)
            if audio_storage_path:
                duration = get_audio_duration(temp_host_path)
                completion_metadata = self.job.get('completion_metadata') or {}
                self.orchestrator_service.update_job_status(self.job_id, 'completed', storage_path=audio_storage_path, duration_seconds=duration, completion_metadata=completion_metadata)
            else:
                logging.error(f"VibeVoice multi-clone job {self.job_id} completed, but audio upload failed. (Block 19)")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                raise CriticalWorkflowError(f"VibeVoice multi-clone job {self.job_id} completed, but audio upload failed. (Block 19)")
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class DiffRhythmJobProcessor(BaseJobProcessor):
    def process(self):
        if not self.job:
            logging.error(f"Job object is None for DiffRhythmJobProcessor. Cannot proceed.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError("Job object is None for DiffRhythmJobProcessor.")

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        # Parse enhanced prompt structure
        lyrics_or_edit_lyrics = ""
        style_prompt = self.positive_prompt
        
        try:
            if self.positive_prompt:
                parsed_prompt = json.loads(self.positive_prompt)
                lyrics_or_edit_lyrics = parsed_prompt.get('lyrics_or_edit_lyrics', '')
                style_prompt = parsed_prompt.get('style_prompt', self.positive_prompt)
                logging.info(f"Parsed enhanced prompt - Lyrics: {len(lyrics_or_edit_lyrics)} chars, Style: {len(style_prompt)} chars")
        except json.JSONDecodeError:
            # Fallback to old behavior for backward compatibility
            logging.warning(f"Failed to parse enhanced prompt structure for job {self.job_id}, using fallback")
            lyrics_or_edit_lyrics = ""
            style_prompt = self.positive_prompt

        wf_ready = inject_prompt_into_diffrhythm_workflow(workflow_data, lyrics_or_edit_lyrics, style_prompt)
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        audio_info = find_audio_in_output(outputs)
        if not audio_info:
            logging.error(f"DiffRhythm workflow for job {self.job_id} completed, but no audio file found. (Block 20)")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"DiffRhythm workflow for job {self.job_id} completed, but no audio file found. (Block 20)")

        audio_filename, subfolder = audio_info
        temp_host_path = self._copy_file_from_container(audio_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}. (Block 21)")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        try:
            audio_storage_path = self.orchestrator_service.upload_audio_output(temp_host_path, self.job_id)
            if audio_storage_path:
                duration = get_audio_duration(temp_host_path)
                completion_metadata = self.job.get('completion_metadata') or {}
                completion_metadata.update({
                    'lyrics_length': len(lyrics_or_edit_lyrics),
                    'style_prompt_length': len(style_prompt),
                    'has_lyrics': bool(lyrics_or_edit_lyrics.strip())
                })
                self.orchestrator_service.update_job_status(self.job_id, 'completed', storage_path=audio_storage_path, duration_seconds=duration, completion_metadata=completion_metadata)
            else:
                logging.error(f"DiffRhythm job {self.job_id} completed, but audio upload failed. (Block 22)")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                raise CriticalWorkflowError(f"DiffRhythm job {self.job_id} completed, but audio upload failed. (Block 22)")
        except TokenExpiredError:
            raise  # Re-raise to be handled by the main loop
        except Exception as e:
            logging.error(f"Error processing DiffRhythm job {self.job_id}: {e}", exc_info=True)
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Error processing DiffRhythm job {self.job_id}: {e}")
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class WAN22TextToVideoJobProcessor(BaseJobProcessor):
    def process(self):
        if DEV_MODE:
            # Dev mode logic remains unchanged
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return
            
        inputs = self.job.get('inputs', {})
        aspect_ratio = inputs.get('aspect_ratio', '16:9')
        wf_ready = inject_prompt_into_text_to_video_workflow(workflow_data, self.positive_prompt, self.negative_prompt, aspect_ratio)
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        video_info = find_video_in_output(outputs)
        if not video_info:
            logging.error(f"Workflow for job {self.job_id} completed, but no video file found. (Block 23)")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        video_filename, subfolder = video_info
        temp_host_path = self._copy_file_from_container(video_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}. (Block 24)")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        try:
            video_storage_path = self.orchestrator_service.upload_output(temp_host_path, self.job_id, 'video/mp4')
            if video_storage_path:
                thumbnail_local_path = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
                thumbnail_storage_path = None
                
                if generate_thumbnail(temp_host_path, thumbnail_local_path, width=100):
                    thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(thumbnail_local_path, self.job_id)
                    os.remove(thumbnail_local_path)
                
                duration = get_video_duration(temp_host_path)
                self.orchestrator_service.update_job_status(self.job_id, 'completed', storage_path=video_storage_path, thumbnail_storage_path=thumbnail_storage_path, duration_seconds=duration)

                # Check for upscale chaining
                workflow_config = self.job.get('inputs', {})
                if workflow_config.get('upscale_enabled'):
                    logging.info(f"Upscale enabled for job {self.job_id}. Submitting upscale job.")
                    upscale_params = workflow_config.get('upscale_params', {})
                    
                    # Construct submit body
                    submit_body = {
                        "sceneId": self.job.get('scene_id'),
                        "branchId": self.job.get('branch_id'),
                        "model": "esrgan-upscaler", # WORKFLOW_TYPES.ESRGAN_UPSCALER
                        "prompt": "Upscaling video",
                        "input_storage_path": video_storage_path,
                        "upscale_params": upscale_params,
                        "originalJobId": self.job.get('id')
                    }
                    
                    new_job_id = self.orchestrator_service.submit_job(submit_body)
                    if new_job_id:
                        logging.info(f"Successfully submitted upscale job: {new_job_id}")
                    else:
                        logging.error("Failed to submit upscale job.")

            else:
                logging.error(f"Video upload failed for job {self.job_id}. (Block 25)")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class WAN22ImageToVideoJobProcessor(BaseJobProcessor):
    def process(self):
        if DEV_MODE:
            # Dev mode logic remains unchanged
            return

        if not self.job:
            logging.error(f"Job object is None for WAN22ImageToVideoJobProcessor. Cannot proceed.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError("Job object is None for WAN22ImageToVideoJobProcessor.")

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        start_image_filename = materialize_start_image(self.job, self.input_dir)
        if not start_image_filename:
            logging.error(f"Failed to materialize start image for job {self.job_id}. (Block 26)")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        start_image_full_path = os.path.join(self.input_dir, start_image_filename)

        try:
            container_input_path = f"/opt/ComfyUI/input/{start_image_filename}"
            docker_manager.copy_file_to_container(
                service_type=self.client.active_service_type,
                source_on_host=start_image_full_path,
                dest_in_container=container_input_path,
                shutdown_event=self.shutdown_event
            )
        except Exception as e:
            logging.error(f"Failed to copy start image to container for job {self.job_id}: {e}", exc_info=True)
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        inputs = self.job.get('inputs', {})
        aspect_ratio = inputs.get('aspect_ratio', '16:9')
        wf_ready = inject_prompt_and_image_into_workflow(workflow_data, self.positive_prompt, self.negative_prompt, start_image_filename, aspect_ratio)
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        video_info = find_video_in_output(outputs)
        if not video_info:
            logging.error(f"Workflow for job {self.job_id} completed, but no video file found. (Block 27)")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        video_filename, subfolder = video_info
        temp_host_path = self._copy_file_from_container(video_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}. (Block 28)")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        try:
            video_storage_path = self.orchestrator_service.upload_output(temp_host_path, self.job_id, 'video/mp4')
            if video_storage_path:
                thumbnail_local_path = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
                thumbnail_storage_path = None
                
                if generate_thumbnail(temp_host_path, thumbnail_local_path, width=100):
                    thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(thumbnail_local_path, self.job_id)
                    os.remove(thumbnail_local_path)
                
                duration = get_video_duration(temp_host_path)
                self.orchestrator_service.update_job_status(self.job_id, 'completed', storage_path=video_storage_path, thumbnail_storage_path=thumbnail_storage_path, duration_seconds=duration)

                # Check for upscale chaining
                workflow_config = self.job.get('workflow', {})
                if workflow_config.get('upscale_enabled'):
                    logging.info(f"Upscale enabled for job {self.job_id}. Submitting upscale job.")
                    upscale_params = workflow_config.get('upscale_params', {})
                    
                    # Construct submit body
                    submit_body = {
                        "sceneId": self.job.get('scene_id'),
                        "branchId": self.job.get('branch_id'),
                        "model": "esrgan-upscaler", # WORKFLOW_TYPES.ESRGAN_UPSCALER
                        "prompt": "Upscaling video",
                        "input_storage_path": video_storage_path,
                        "upscale_params": upscale_params
                    }
                    
                    new_job_id = self.orchestrator_service.submit_job(submit_body)
                    if new_job_id:
                        logging.info(f"Successfully submitted upscale job: {new_job_id}")
                    else:
                        logging.error("Failed to submit upscale job.")

            else:
                logging.error(f"Video upload failed for job {self.job_id}. (Block 29)")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                raise CriticalWorkflowError(f"Video upload failed for job {self.job_id}. (Block 29)")
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)
            
class ImageToVideoFromLastFrameJobProcessor(BaseJobProcessor):
    def process(self):
        if DEV_MODE:
            # Dev mode logic remains unchanged
            return

        if not self.job:
            logging.error(f"Job object is None for ImageToVideoFromLastFrameJobProcessor. Cannot proceed.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError("Job object is None for ImageToVideoFromLastFrameJobProcessor.")

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get('inputs', {})
        input_video_url = inputs.get('input_video_url')
        if not input_video_url:
            logging.error(f"Job {self.job_id} missing 'input_video_url' in inputs.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Job {self.job_id} missing 'input_video_url' in inputs.")

        # Extract last frame
        start_image_filename = f"{self.job_id}_last_frame.jpg"
        start_image_full_path = os.path.join(self.input_dir, start_image_filename)
        
        if not extract_last_frame(video_path, start_image_full_path):
             logging.error(f"Failed to extract last frame for job {self.job_id}.")
             self.orchestrator_service.update_job_status(self.job_id, 'failed')
             raise CriticalWorkflowError(f"Failed to extract last frame for job {self.job_id}.")

        try:
            container_input_path = f"/opt/ComfyUI/input/{start_image_filename}"
            docker_manager.copy_file_to_container(
                service_type=self.client.active_service_type,
                source_on_host=start_image_full_path,
                dest_in_container=container_input_path,
                shutdown_event=self.shutdown_event
            )
        except Exception as e:
            logging.error(f"Failed to copy start image to container for job {self.job_id}: {e}", exc_info=True)
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Failed to copy start image to container for job {self.job_id}: {e}")

        inputs = self.job.get('inputs', {})
        aspect_ratio = inputs.get('aspect_ratio', '16:9')
        wf_ready = inject_prompt_and_image_into_workflow(workflow_data, self.positive_prompt, self.negative_prompt, start_image_filename, aspect_ratio)
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        video_info = find_video_in_output(outputs)
        if not video_info:
            logging.error(f"Workflow for job {self.job_id} completed, but no video file found.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Workflow for job {self.job_id} completed, but no video file found.")

        video_filename, subfolder = video_info
        temp_host_path = self._copy_file_from_container(video_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Failed to copy output file from container for job {self.job_id}.")

        try:
            video_storage_path = self.orchestrator_service.upload_output(temp_host_path, self.job_id, 'video/mp4')
            if video_storage_path:
                thumbnail_local_path = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
                thumbnail_storage_path = None
                
                if generate_thumbnail(temp_host_path, thumbnail_local_path, width=100):
                    thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(thumbnail_local_path, self.job_id)
                    os.remove(thumbnail_local_path)
                
                duration = get_video_duration(temp_host_path)
                self.orchestrator_service.update_job_status(self.job_id, 'completed', storage_path=video_storage_path, thumbnail_storage_path=thumbnail_storage_path, duration_seconds=duration)

                # Check for upscale chaining
                workflow_config = self.job.get('workflow', {})
                if workflow_config.get('upscale_enabled'):
                    logging.info(f"Upscale enabled for job {self.job_id}. Submitting upscale job.")
                    upscale_params = workflow_config.get('upscale_params', {})
                    
                    # Construct submit body
                    submit_body = {
                        "sceneId": self.job.get('scene_id'),
                        "branchId": self.job.get('branch_id'),
                        "model": "esrgan-upscaler", # WORKFLOW_TYPES.ESRGAN_UPSCALER
                        "prompt": "Upscaling video",
                        "input_storage_path": video_storage_path,
                        "upscale_params": upscale_params
                    }
                    
                    new_job_id = self.orchestrator_service.submit_job(submit_body)
                    if new_job_id:
                        logging.info(f"Successfully submitted upscale job: {new_job_id}")
                    else:
                        logging.error("Failed to submit upscale job.")

            else:
                logging.error(f"Video upload failed for job {self.job_id}.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)
            # Also clean up source video and extracted frame
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(start_image_full_path):
                os.remove(start_image_full_path)

class LTXVideoTextToVideoJobProcessor(BaseJobProcessor):
    def process(self):
        if DEV_MODE:
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get('inputs', {})
        aspect_ratio = inputs.get('aspect_ratio', '16:9')
        quality_preset = inputs.get('quality_preset', 'standard')
        wf_ready = inject_prompt_into_ltx_video_workflow(workflow_data, self.positive_prompt, self.negative_prompt, aspect_ratio, quality_preset)
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        video_info = find_video_in_output(outputs)
        if not video_info:
            logging.error(f"Workflow for job {self.job_id} completed, but no video file found.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Workflow for job {self.job_id} completed, but no video file found.")

        video_filename, subfolder = video_info
        temp_host_path = self._copy_file_from_container(video_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Failed to copy output file from container for job {self.job_id}.")

        try:
            video_storage_path = self.orchestrator_service.upload_output(temp_host_path, self.job_id, 'video/mp4')
            if video_storage_path:
                thumbnail_local_path = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
                thumbnail_storage_path = None
                
                if generate_thumbnail(temp_host_path, thumbnail_local_path, width=100):
                    thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(thumbnail_local_path, self.job_id)
                    os.remove(thumbnail_local_path)
                
                duration = get_video_duration(temp_host_path)
                self.orchestrator_service.update_job_status(self.job_id, 'completed', storage_path=video_storage_path, thumbnail_storage_path=thumbnail_storage_path, duration_seconds=duration)
            else:
                logging.error(f"Video upload failed for job {self.job_id}.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                raise CriticalWorkflowError(f"Video upload failed for job {self.job_id}.")
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class LTXVideoImageToVideoJobProcessor(BaseJobProcessor):
    def process(self):
        if DEV_MODE:
            return

        if not self.job:
            logging.error(f"Job object is None for LTXVideoImageToVideoJobProcessor. Cannot proceed.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError("Job object is None for LTXVideoImageToVideoJobProcessor.")

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        start_image_filename = materialize_start_image(self.job, self.input_dir)
        if not start_image_filename:
            logging.error(f"Failed to materialize start image for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Failed to materialize start image for job {self.job_id}.")

        start_image_full_path = os.path.join(self.input_dir, start_image_filename)

        try:
            container_input_path = f"/opt/ComfyUI/input/{start_image_filename}"
            docker_manager.copy_file_to_container(
                service_type=self.client.active_service_type,
                source_on_host=start_image_full_path,
                dest_in_container=container_input_path,
                shutdown_event=self.shutdown_event
            )
        except Exception as e:
            logging.error(f"Failed to copy start image to container for job {self.job_id}: {e}", exc_info=True)
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Failed to copy start image to container for job {self.job_id}: {e}")

        inputs = self.job.get('inputs', {})
        aspect_ratio = inputs.get('aspect_ratio', '16:9')
        quality_preset = inputs.get('quality_preset', 'standard')
        wf_ready = inject_prompt_and_image_into_ltx_video_workflow(workflow_data, self.positive_prompt, self.negative_prompt, start_image_filename, aspect_ratio, quality_preset)
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        video_info = find_video_in_output(outputs)
        if not video_info:
            logging.error(f"Workflow for job {self.job_id} completed, but no video file found.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Workflow for job {self.job_id} completed, but no video file found.")

        video_filename, subfolder = video_info
        temp_host_path = self._copy_file_from_container(video_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Failed to copy output file from container for job {self.job_id}.")

        try:
            video_storage_path = self.orchestrator_service.upload_output(temp_host_path, self.job_id, 'video/mp4')
            if video_storage_path:
                thumbnail_local_path = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
                thumbnail_storage_path = None
                
                if generate_thumbnail(temp_host_path, thumbnail_local_path, width=100):
                    thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(thumbnail_local_path, self.job_id)
                    os.remove(thumbnail_local_path)
                
                duration = get_video_duration(temp_host_path)
                self.orchestrator_service.update_job_status(self.job_id, 'completed', storage_path=video_storage_path, thumbnail_storage_path=thumbnail_storage_path, duration_seconds=duration)
            else:
                logging.error(f"Video upload failed for job {self.job_id}.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                raise CriticalWorkflowError(f"Video upload failed for job {self.job_id}.")
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class VideoUpscalerJobProcessor(BaseJobProcessor):
    def process(self):
        if not self.job:
            logging.error(f"Job object is None for VideoUpscalerJobProcessor. Cannot proceed.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError("Job object is None for VideoUpscalerJobProcessor.")

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        input_video_url = self.job.get('input_video_url')
        if not input_video_url:
            logging.error(f"Video upscaler job {self.job_id} missing 'input_video_url'.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Video upscaler job {self.job_id} missing 'input_video_url'.")

        video_path = self.orchestrator_service.download_asset_by_url(input_video_url, self.input_dir)
        if not video_path:
            logging.error(f"Failed to download input video for upscaler job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Failed to download input video for upscaler job {self.job_id}.")
        
        video_filename = os.path.basename(video_path)

        # Copy video to container input directory
        try:
            container_input_path = f"/opt/ComfyUI/input/{video_filename}"
            docker_manager.copy_file_to_container(
                service_type=self.client.active_service_type,
                source_on_host=video_path,
                dest_in_container=container_input_path,
                shutdown_event=self.shutdown_event
            )
            logging.info(f"Copied video to container: {container_input_path}")
        except Exception as e:
            logging.error(f"Failed to copy video to container for job {self.job_id}: {e}", exc_info=True)
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Failed to copy video to container for job {self.job_id}: {e}")
        
        # --- New Upscale Logic ---
        original_width = None
        original_height = None
        try:
            original_width, original_height = get_video_dimensions(video_path)
            logging.info(f"Original video dimensions: {original_width}x{original_height}")
        except Exception as e:
            logging.error(f"Could not get video dimensions for {video_path}: {e}")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Could not get video dimensions for {video_path}: {e}")

        job_inputs = self.job.get('inputs', {})
        upscale_model = job_inputs.get('upscale_model', 'RealESRGAN_x4plus.pth')
        
        # Determine frame rate: prefer explicit input, otherwise detect from source, fallback to 30
        input_frame_rate = job_inputs.get('frame_rate')
        if input_frame_rate:
            frame_rate = int(input_frame_rate)
        else:
            frame_rate = int(get_video_framerate(video_path))
            logging.info(f"Detected source frame rate: {frame_rate}")

        upscale_factor = job_inputs.get('upscale_factor')
        target_width = job_inputs.get('target_width')
        target_height = job_inputs.get('target_height')
        control_method = job_inputs.get('control_method', 'factor') # Default to factor if not specified

        final_width = None
        final_height = None
        scale_by = None

        # If control method is 'dimensions', prioritize target dimensions
        if control_method == 'dimensions':
            if target_width and target_height:
                final_width = int(target_width)
                final_height = int(target_height)
                logging.info(f"Using explicit target dimensions: {final_width}x{final_height}")
            elif target_width:
                ratio = float(target_width) / float(original_width)
                final_width = int(target_width)
                final_height = int(original_height * ratio)
                logging.info(f"Using target width {target_width} and calculated height {final_height} to preserve aspect ratio.")
            elif target_height:
                ratio = float(target_height) / float(original_height)
                final_height = int(target_height)
                final_width = int(original_width * ratio)
                logging.info(f"Using target height {target_height} and calculated width {final_width} to preserve aspect ratio.")
            else:
                 # Fallback if dimensions selected but none provided
                 logging.warning("Control method is dimensions but no dimensions provided. Falling back to 2x upscale.")
                 final_width = original_width * 2
                 final_height = original_height * 2
        
        # If control method is 'factor' (or fallback), use upscale factor
        else:
            if upscale_factor:
                scale_by = float(upscale_factor)
                logging.info(f"Using upscale factor {scale_by} with ImageScaleBy.")
            else:
                # Default to 2x if nothing specified
                scale_by = 2.0
                logging.info(f"No factor or dimensions provided. Defaulting to 2x upscale with ImageScaleBy.")
        
        # Ensure dimensions are even (required for some codecs) - ONLY if using explicit dimensions
        if scale_by is None:
            if final_width % 2 != 0: final_width += 1
            if final_height % 2 != 0: final_height += 1
        
        wf_ready = inject_video_into_upscaler_workflow(
            workflow_data, 
            video_filename, 
            upscale_model,
            frame_rate,
            target_width=final_width,
            target_height=final_height,
            scale_by=scale_by
        )
        
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        video_info = find_video_in_output(outputs)
        if not video_info:
            logging.error(f"Upscaler workflow for job {self.job_id} completed, but no video file found.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Upscaler workflow for job {self.job_id} completed, but no video file found.")

        video_filename, subfolder = video_info
        temp_host_path = self._copy_file_from_container(video_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Failed to copy output file from container for job {self.job_id}.")

        try:
            video_storage_path = self.orchestrator_service.upload_output(temp_host_path, self.job_id, 'video/mp4')
            if video_storage_path:
                thumbnail_local_path = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
                thumbnail_storage_path = None
                
                if generate_thumbnail(temp_host_path, thumbnail_local_path, width=100):
                    thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(thumbnail_local_path, self.job_id)
                    os.remove(thumbnail_local_path)
                
                duration = get_video_duration(temp_host_path)
                completion_metadata = self.job.get('completion_metadata') or {}
                completion_metadata['upscale_model'] = upscale_model
                
                self.orchestrator_service.update_job_status(
                    self.job_id, 
                    'completed', 
                    storage_path=video_storage_path, 
                    thumbnail_storage_path=thumbnail_storage_path, 
                    duration_seconds=duration,
                    completion_metadata=completion_metadata
                )
            else:
                logging.error(f"Video upscaler job {self.job_id} completed, but video upload failed.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                raise CriticalWorkflowError(f"Video upscaler job {self.job_id} completed, but video upload failed.")
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class TextToVideoLightningJobProcessor(WAN22TextToVideoJobProcessor):
    workflow_name = 'WAN22_LIGHTNING_TEXT_TO_VIDEO.json'

class ImageToVideoLightningJobProcessor(WAN22ImageToVideoJobProcessor):
    workflow_name = 'WAN22_LIGHTNING_IMAGE_TO_VIDEO.json'



class TextGenerationJobProcessor(BaseJobProcessor):
    def process(self):
        if not self.job:
            logging.error(f"Job object is None for TextGenerationJobProcessor. Cannot proceed.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError("Job object is None for TextGenerationJobProcessor.")

        # Container is already running (started by job listener)
        # Wait for Ollama to be ready
        api_base = os.getenv("LLM_API_BASE", "http://localhost:11434")
        
        import time
        import requests
        
        logging.info(f"Waiting for Ollama service to be ready at {api_base}...")
        
        # Give Ollama a few seconds to initialize before first check
        time.sleep(3)
        
        ready = False
        for attempt in range(60):  # Wait up to 60 seconds for Ollama to start
            if self.shutdown_event.is_set():
                logging.info("Shutdown event received during Ollama readiness check.")
                return
            try:
                # Use Ollama's native API endpoint to check readiness
                # /api/tags lists available models and confirms the server is responding
                response = requests.get(f"{api_base}/api/tags", timeout=2)
                response.raise_for_status()
                ready = True
                logging.info(f"Ollama service is ready! (attempt {attempt + 1}/60)")
                break
            except requests.exceptions.RequestException as e:
                if attempt % 10 == 0:  # Log every 10 attempts to avoid spam
                    logging.debug(f"Ollama not ready yet (attempt {attempt + 1}/60): {e}")
                time.sleep(1)
        
        if not ready:
            logging.error(f"Ollama service failed to become ready within 60 seconds at {api_base}")
            logging.error("Please check: 1) Container is running, 2) Port 11434 is mapped correctly, 3) No firewall blocking")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Ollama service failed to become ready within 60 seconds at {api_base}")

        # Get generation parameters
        inputs = self.job.get('inputs', {})
        model_name = inputs.get('model', 'llama3.1:8b')
        system_prompt = inputs.get('system_prompt', "You are a helpful assistant.")
        temperature = inputs.get('temperature', 0.7)
        max_tokens = inputs.get('max_tokens', 2000)
        
        # Generate random seed if not provided
        seed = inputs.get('seed')
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
            logging.info(f"No seed provided, using random seed: {seed}")

        logging.info(f"Generating text with model {model_name}...")
        
        # First, check if the model exists and pull if needed
        try:
            logging.info(f"Checking if model {model_name} is available...")
            tags_response = requests.get(f"{api_base}/api/tags", timeout=5)
            tags_response.raise_for_status()
            tags_data = tags_response.json()
            
            available_models = [m.get('name', '') for m in tags_data.get('models', [])]
            logging.info(f"Available models: {available_models}")
            
            if model_name not in available_models:
                logging.info(f"Model {model_name} not found. Pulling it now...")
                pull_payload = {"name": model_name, "stream": False}
                pull_response = requests.post(f"{api_base}/api/pull", json=pull_payload, timeout=600)
                pull_response.raise_for_status()
                logging.info(f"Successfully pulled model {model_name}")
            else:
                logging.info(f"Model {model_name} is already available")
        except Exception as e:
            logging.error(f"Failed to verify/pull model {model_name}: {e}")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Failed to verify/pull model {model_name}: {e}")

        try:
            # Use Ollama's /api/generate endpoint (available in all versions including 0.13.0)
            # Combine system and user prompts into a single string
            full_prompt = f"{system_prompt}\n\nUser: {self.positive_prompt}\n\nAssistant:"
            
            payload = {
                "model": model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "seed": seed
                }
            }
            
            # Call Ollama's /api/generate endpoint
            logging.info(f"Calling Ollama API at {api_base}/api/generate with model {model_name}")
            response = requests.post(f"{api_base}/api/generate", json=payload, timeout=1200)
            response.raise_for_status()
            
            result = response.json()
            # Response format: {"model": "...", "response": "..."}
            generated_text = result.get('response', '')
            
            if not generated_text:
                logging.error(f"Ollama returned empty response. Full result: {result}")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                raise CriticalWorkflowError(f"Ollama returned empty response. Full result: {result}")
            
            logging.info(f"Generation complete. Length: {len(generated_text)} chars.")

            # Save and upload output
            output_filename = f"{self.job_id}_script.txt"
            output_path = os.path.join(self.cache_dir, output_filename)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(generated_text)

            storage_path = self.orchestrator_service.upload_output(output_path, self.job_id, "text/plain")
            
            if storage_path:
                self.orchestrator_service.update_job_status(
                    self.job_id, 
                    'completed', 
                    storage_path=storage_path,
                    completion_metadata={"model": model_name}
                )
            else:
                logging.error("Failed to upload generated script.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                raise CriticalWorkflowError("Failed to upload generated script.")

            # Cleanup
            if os.path.exists(output_path):
                os.remove(output_path)

        except Exception as e:
            logging.error(f"Text generation failed: {e}", exc_info=True)
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Text generation failed: {e}")


class HunyuanVideoJobProcessor(BaseJobProcessor):
    def process(self):
        if DEV_MODE:
             return

        if not self.job:
            logging.error(f"Job object is None for HunyuanVideoJobProcessor. Cannot proceed.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError("Job object is None for HunyuanVideoJobProcessor.")

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get('inputs', {})
        steps = inputs.get('steps', 30)
        guidance = inputs.get('guidance', 6.0)
        strength = inputs.get('strength', 1.0)
        # Hunyuan 1.5 standard res (could be configurable)
        width = 854 
        height = 480
        frame_count = 49

        wf_ready = None

        if self.workflow_type == 'hunyuan-video-image-to-video':
            start_image_filename = materialize_start_image(self.job, self.input_dir)
            if not start_image_filename:
                logging.error(f"Failed to materialize start image for job {self.job_id}.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                raise CriticalWorkflowError(f"Failed to materialize start image for job {self.job_id}.")

            start_image_full_path = os.path.join(self.input_dir, start_image_filename)

            try:
                container_input_path = f"/opt/ComfyUI/input/{start_image_filename}"
                docker_manager.copy_file_to_container(
                    service_type=self.client.active_service_type,
                    source_on_host=start_image_full_path,
                    dest_in_container=container_input_path,
                    shutdown_event=self.shutdown_event
                )
            except Exception as e:
                logging.error(f"Failed to copy start image to container for job {self.job_id}: {e}", exc_info=True)
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                raise CriticalWorkflowError(f"Failed to copy start image to container for job {self.job_id}: {e}")

            wf_ready = inject_prompt_and_image_into_hunyuan_workflow(
                workflow_data, 
                self.positive_prompt, 
                start_image_filename,
                steps=steps, 
                guidance=guidance, 
                strength=strength,
                width=width,
                height=height,
                frame_count=frame_count
            )
        else:
            wf_ready = inject_prompt_into_hunyuan_workflow(
                workflow_data, 
                self.positive_prompt, 
                steps=steps, 
                guidance=guidance, 
                strength=strength,
                width=width,
                height=height,
                frame_count=frame_count
            )

        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        video_info = find_video_in_output(outputs)
        if not video_info:
            logging.error(f"Workflow for job {self.job_id} completed, but no video file found.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Workflow for job {self.job_id} completed, but no video file found.")

        video_filename, subfolder = video_info
        temp_host_path = self._copy_file_from_container(video_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            raise CriticalWorkflowError(f"Failed to copy output file from container for job {self.job_id}.")

        try:
            video_storage_path = self.orchestrator_service.upload_output(temp_host_path, self.job_id, 'video/mp4')
            if video_storage_path:
                thumbnail_local_path = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
                thumbnail_storage_path = None
                
                if generate_thumbnail(temp_host_path, thumbnail_local_path, width=100):
                    thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(thumbnail_local_path, self.job_id)
                    os.remove(thumbnail_local_path)
                
                duration = get_video_duration(temp_host_path)
                self.orchestrator_service.update_job_status(self.job_id, 'completed', storage_path=video_storage_path, thumbnail_storage_path=thumbnail_storage_path, duration_seconds=duration)

                # Upscale chaining
                workflow_config = self.job.get('inputs', {})
                if workflow_config.get('upscale_enabled'):
                    logging.info(f"Upscale enabled for job {self.job_id}. Submitting upscale job.")
                    upscale_params = workflow_config.get('upscale_params', {})
                    
                    submit_body = {
                        "sceneId": self.job.get('scene_id'),
                        "branchId": self.job.get('branch_id'),
                        "model": "esrgan-upscaler",
                        "prompt": "Upscaling video",
                        "input_storage_path": video_storage_path,
                        "upscale_params": upscale_params,
                        "originalJobId": self.job.get('id')
                    }
                    
                    new_job_id = self.orchestrator_service.submit_job(submit_body)
                    if new_job_id:
                        logging.info(f"Successfully submitted upscale job: {new_job_id}")
                    else:
                        logging.error("Failed to submit upscale job.")

            else:
                logging.error(f"Video upload failed for job {self.job_id}.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                raise CriticalWorkflowError(f"Video upload failed for job {self.job_id}.")
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)
