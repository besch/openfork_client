import os
import json
import logging
from typing import Union, Dict
from abc import ABC, abstractmethod
from services.orchestrator_service import TokenExpiredError
from config import DEV_MODE
from services.docker_manager import docker_manager
from utils.media_utils import get_audio_duration, find_audio_in_output, find_image_in_output, find_video_in_output, generate_thumbnail, get_video_duration, get_video_dimensions, get_video_framerate, extract_last_frame
from utils.comfyui_workflow_utils import (
    inject_prompt_and_image_into_workflow,
    inject_video_and_prompt_into_foley_workflow,
    inject_prompt_into_qwen_workflow,
    inject_prompt_into_text_to_video_workflow,
    inject_prompt_into_vibevoice_workflow,
    inject_script_and_clones_into_vibevoice_workflow,
    inject_prompt_into_diffrhythm_workflow,
    materialize_start_image,
    inject_video_into_upscaler_workflow
)

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
            return None

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
            return None
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from workflow file: {workflow_path}")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading local workflow {local_filename}: {e}")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return None

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
                dest_on_host=dest_on_host
            )
            if os.path.exists(dest_on_host):
                logging.info(f"Successfully copied file to temporary host path: {dest_on_host}")
                return dest_on_host
            else:
                raise RuntimeError("docker cp command finished but destination file does not exist.")
        except Exception as e:
            logging.error(f"Failed to copy file from container: {e}", exc_info=True)
            return None

    def _trigger_and_get_output(self, payload):
        prompt_id = self.comfyui_client.trigger_workflow(payload)
        if not prompt_id:
            logging.error(f"Failed to trigger workflow for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return None

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
            return None
            
        return outputs

    @abstractmethod
    def process(self):
        pass

class FoleyJobProcessor(BaseJobProcessor):
    def process(self):
        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        input_video_url = self.job.get('input_video_url')
        if not input_video_url:
            logging.error(f"Foley job {self.job_id} missing 'input_video_url'.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        video_path = self.orchestrator_service.download_asset_by_url(input_video_url, self.input_dir)
        if not video_path:
            logging.error(f"Failed to download input video for foley job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return
        
        video_filename = os.path.basename(video_path)
        wf_ready = inject_video_and_prompt_into_foley_workflow(workflow_data, video_filename, self.positive_prompt, self.negative_prompt)
        
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        audio_info = find_audio_in_output(outputs)
        if not audio_info:
            logging.error(f"Foley workflow for job {self.job_id} completed, but no audio file found.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        audio_filename, subfolder = audio_info
        temp_host_path = self._copy_file_from_container(audio_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        try:
            audio_storage_path = self.orchestrator_service.upload_audio_output(temp_host_path, self.job_id)
            if audio_storage_path:
                duration = get_audio_duration(temp_host_path)
                completion_metadata = self.job.get('completion_metadata') or {}
                self.orchestrator_service.update_job_status(self.job_id, 'completed', storage_path=audio_storage_path, duration_seconds=duration, completion_metadata=completion_metadata)
            else:
                logging.error(f"Foley job {self.job_id} completed, but audio upload failed.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class TextToImageJobProcessor(BaseJobProcessor):
    def process(self):
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
            logging.error(f"Workflow for job {self.job_id} completed, but no image file found.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        image_filename, subfolder = image_info
        temp_host_path = self._copy_file_from_container(image_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        try:
            image_storage_path = self.orchestrator_service.upload_image_output(temp_host_path, self.job_id)
            if image_storage_path:
                self.orchestrator_service.update_job_status(self.job_id, 'completed', storage_path=image_storage_path, thumbnail_storage_path=image_storage_path, prompt=self.positive_prompt)
            else:
                logging.error(f"Image upload failed for job {self.job_id}.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class VibeVoiceJobProcessor(BaseJobProcessor):
    def process(self):
        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        wf_ready = inject_prompt_into_vibevoice_workflow(workflow_data, self.positive_prompt)
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        audio_info = find_audio_in_output(outputs)
        if not audio_info:
            logging.error(f"VibeVoice workflow for job {self.job_id} completed, but no audio file found.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        audio_filename, subfolder = audio_info
        temp_host_path = self._copy_file_from_container(audio_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        try:
            audio_storage_path = self.orchestrator_service.upload_audio_output(temp_host_path, self.job_id)
            if audio_storage_path:
                duration = get_audio_duration(temp_host_path)
                completion_metadata = self.job.get('completion_metadata') or {}
                self.orchestrator_service.update_job_status(self.job_id, 'completed', storage_path=audio_storage_path, duration_seconds=duration, completion_metadata=completion_metadata)
            else:
                logging.error(f"VibeVoice job {self.job_id} completed, but audio upload failed.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class VibeVoiceMultiCloneJobProcessor(BaseJobProcessor):
    def process(self):
        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        voice_clone_urls = self.job.get('voice_clone_urls', [])
        if not voice_clone_urls:
            logging.error(f"VibeVoice multi-clone job {self.job_id} missing 'voice_clone_urls'.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        clone_paths = []
        for url in voice_clone_urls:
            clone_path = self.orchestrator_service.download_asset_by_url(url, self.input_dir)
            if not clone_path:
                logging.error(f"Failed to download voice clone from {url} for job {self.job_id}.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                return
            clone_paths.append(os.path.basename(clone_path))

        wf_ready = inject_script_and_clones_into_vibevoice_workflow(workflow_data, self.positive_prompt, clone_paths)
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        audio_info = find_audio_in_output(outputs)
        if not audio_info:
            logging.error(f"VibeVoice multi-clone workflow for job {self.job_id} completed, but no audio file found.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        audio_filename, subfolder = audio_info
        temp_host_path = self._copy_file_from_container(audio_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        try:
            audio_storage_path = self.orchestrator_service.upload_audio_output(temp_host_path, self.job_id)
            if audio_storage_path:
                duration = get_audio_duration(temp_host_path)
                completion_metadata = self.job.get('completion_metadata') or {}
                self.orchestrator_service.update_job_status(self.job_id, 'completed', storage_path=audio_storage_path, duration_seconds=duration, completion_metadata=completion_metadata)
            else:
                logging.error(f"VibeVoice multi-clone job {self.job_id} completed, but audio upload failed.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class DiffRhythmJobProcessor(BaseJobProcessor):
    def process(self):
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
            logging.error(f"DiffRhythm workflow for job {self.job_id} completed, but no audio file found.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        audio_filename, subfolder = audio_info
        temp_host_path = self._copy_file_from_container(audio_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}.")
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
                logging.error(f"DiffRhythm job {self.job_id} completed, but audio upload failed.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        except TokenExpiredError:
            raise  # Re-raise to be handled by the main loop
        except Exception as e:
            logging.error(f"Error processing DiffRhythm job {self.job_id}: {e}", exc_info=True)
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
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
            logging.error(f"Workflow for job {self.job_id} completed, but no video file found.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        video_filename, subfolder = video_info
        temp_host_path = self._copy_file_from_container(video_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        try:
            video_storage_path = self.orchestrator_service.upload_output(temp_host_path, self.job_id, 'video/mp4')
            if video_storage_path:
                thumbnail_local_path = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
                thumbnail_storage_path = None
                
                if generate_thumbnail(temp_host_path, thumbnail_local_path):
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
                logging.error(f"Video upload failed for job {self.job_id}.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class WAN22ImageToVideoJobProcessor(BaseJobProcessor):
    def process(self):
        if DEV_MODE:
            # Dev mode logic remains unchanged
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        start_image_filename = materialize_start_image(self.job, self.input_dir)
        if not start_image_filename:
            logging.error(f"Failed to materialize start image for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        start_image_full_path = os.path.join(self.input_dir, start_image_filename)

        try:
            container_input_path = f"/opt/ComfyUI/input/{start_image_filename}"
            docker_manager.copy_file_to_container(
                service_type=self.client.active_service_type,
                source_on_host=start_image_full_path,
                dest_in_container=container_input_path
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
            logging.error(f"Workflow for job {self.job_id} completed, but no video file found.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        video_filename, subfolder = video_info
        temp_host_path = self._copy_file_from_container(video_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        try:
            video_storage_path = self.orchestrator_service.upload_output(temp_host_path, self.job_id, 'video/mp4')
            if video_storage_path:
                thumbnail_local_path = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
                thumbnail_storage_path = None
                
                if generate_thumbnail(temp_host_path, thumbnail_local_path):
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
            
class ImageToVideoFromLastFrameJobProcessor(BaseJobProcessor):
    def process(self):
        if DEV_MODE:
            # Dev mode logic remains unchanged
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get('inputs', {})
        input_video_url = inputs.get('input_video_url')
        if not input_video_url:
            logging.error(f"Job {self.job_id} missing 'input_video_url' in inputs.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        video_path = self.orchestrator_service.download_asset_by_url(input_video_url, self.input_dir)
        if not video_path:
            logging.error(f"Failed to download input video for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        # Extract last frame
        start_image_filename = f"{self.job_id}_last_frame.jpg"
        start_image_full_path = os.path.join(self.input_dir, start_image_filename)
        
        if not extract_last_frame(video_path, start_image_full_path):
             logging.error(f"Failed to extract last frame for job {self.job_id}.")
             self.orchestrator_service.update_job_status(self.job_id, 'failed')
             return

        try:
            container_input_path = f"/opt/ComfyUI/input/{start_image_filename}"
            docker_manager.copy_file_to_container(
                service_type=self.client.active_service_type,
                source_on_host=start_image_full_path,
                dest_in_container=container_input_path
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
            logging.error(f"Workflow for job {self.job_id} completed, but no video file found.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        video_filename, subfolder = video_info
        temp_host_path = self._copy_file_from_container(video_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        try:
            video_storage_path = self.orchestrator_service.upload_output(temp_host_path, self.job_id, 'video/mp4')
            if video_storage_path:
                thumbnail_local_path = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
                thumbnail_storage_path = None
                
                if generate_thumbnail(temp_host_path, thumbnail_local_path):
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

class VideoUpscalerJobProcessor(BaseJobProcessor):
    def process(self):
        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        input_video_url = self.job.get('input_video_url')
        if not input_video_url:
            logging.error(f"Video upscaler job {self.job_id} missing 'input_video_url'.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        video_path = self.orchestrator_service.download_asset_by_url(input_video_url, self.input_dir)
        if not video_path:
            logging.error(f"Failed to download input video for upscaler job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return
        
        video_filename = os.path.basename(video_path)

        # Copy video to container input directory
        try:
            container_input_path = f"/opt/ComfyUI/input/{video_filename}"
            docker_manager.copy_file_to_container(
                service_type=self.client.active_service_type,
                source_on_host=video_path,
                dest_in_container=container_input_path
            )
            logging.info(f"Copied video to container: {container_input_path}")
        except Exception as e:
            logging.error(f"Failed to copy video to container for job {self.job_id}: {e}", exc_info=True)
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return
        
        # --- New Upscale Logic ---
        original_width = None
        original_height = None
        try:
            original_width, original_height = get_video_dimensions(video_path)
            logging.info(f"Original video dimensions: {original_width}x{original_height}")
        except Exception as e:
            logging.error(f"Could not get video dimensions for {video_path}: {e}")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

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
            return

        video_filename, subfolder = video_info
        temp_host_path = self._copy_file_from_container(video_filename, subfolder)
        if not temp_host_path:
            logging.error(f"Failed to copy output file from container for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        try:
            video_storage_path = self.orchestrator_service.upload_output(temp_host_path, self.job_id, 'video/mp4')
            if video_storage_path:
                thumbnail_local_path = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
                thumbnail_storage_path = None
                
                if generate_thumbnail(temp_host_path, thumbnail_local_path):
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
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class TextToVideoLightningJobProcessor(WAN22TextToVideoJobProcessor):
    workflow_name = 'WAN22_LIGHTNING_TEXT_TO_VIDEO.json'

class ImageToVideoLightningJobProcessor(WAN22ImageToVideoJobProcessor):
    workflow_name = 'WAN22_LIGHTNING_IMAGE_TO_VIDEO.json'
