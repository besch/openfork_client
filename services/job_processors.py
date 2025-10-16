import os
import logging
import subprocess
from typing import Union, Dict
from abc import ABC, abstractmethod
from config import DEV_MODE
from services.docker_manager import docker_manager
from utils.media_utils import get_audio_duration, find_audio_in_output, find_image_in_output, find_video_in_output, generate_thumbnail, get_video_duration
from utils.comfyui_workflow_utils import (
    inject_prompt_and_image_into_workflow,
    inject_video_and_prompt_into_foley_workflow,
    inject_prompt_into_qwen_workflow,
    inject_prompt_into_text_to_video_workflow,
    inject_prompt_into_vibevoice_workflow,
    inject_script_and_clones_into_vibevoice_workflow,
    inject_prompt_into_diffrhythm_workflow,
    materialize_start_image
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

    @property
    @abstractmethod
    def workflow_name(self) -> str:
        """The filename of the workflow to be used for this processor."""
        pass

    def _get_workflow_payload(self) -> Union[Dict, None]:
        """Fetches the workflow from the orchestrator and returns it."""
        workflow_data = self.orchestrator_service.get_workflow(self.workflow_name, self.cache_dir)
        if not workflow_data:
            logging.error(f"Failed to get workflow {self.workflow_name} for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return None
        return workflow_data

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

        source_in_container = os.path.join("/opt/ComfyUI/output", subfolder, filename).replace('\\', '/')
        
        os.makedirs(self.cache_dir, exist_ok=True)

        temp_filename = f"{self.job_id}_{filename}"
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
    workflow_name = 'HUNYUAN_VIDEO_FOLEY.json'

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
                self.orchestrator_service.update_job_status(self.job_id, 'completed', output_path=audio_storage_path, duration_seconds=duration, completion_metadata=self.job.get('completion_metadata'))
            else:
                logging.error(f"Foley job {self.job_id} completed, but audio upload failed.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class TextToImageJobProcessor(BaseJobProcessor):
    workflow_name = 'QWEN_TEXT_TO_IMAGE.json'

    def process(self):
        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        wf_ready = inject_prompt_into_qwen_workflow(workflow_data, self.positive_prompt, self.negative_prompt)
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
                self.orchestrator_service.update_job_status(self.job_id, 'completed', output_path=image_storage_path, thumbnail_path=image_storage_path, prompt=self.positive_prompt)
            else:
                logging.error(f"Image upload failed for job {self.job_id}.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class VibeVoiceJobProcessor(BaseJobProcessor):
    workflow_name = 'VIBEVOICE_TTS.json'

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
                self.orchestrator_service.update_job_status(self.job_id, 'completed', output_path=audio_storage_path, duration_seconds=duration, completion_metadata=self.job.get('completion_metadata'))
            else:
                logging.error(f"VibeVoice job {self.job_id} completed, but audio upload failed.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class VibeVoiceMultiCloneJobProcessor(BaseJobProcessor):
    workflow_name = 'VIBEVOICE_TTS_MULTI_CLONE.json'

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
                self.orchestrator_service.update_job_status(self.job_id, 'completed', output_path=audio_storage_path, duration_seconds=duration, completion_metadata=self.job.get('completion_metadata'))
            else:
                logging.error(f"VibeVoice multi-clone job {self.job_id} completed, but audio upload failed.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class DiffRhythmJobProcessor(BaseJobProcessor):
    workflow_name = 'DIFFRHYTHM_MUSIC_GENERATION.json'

    def process(self):
        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        wf_ready = inject_prompt_into_diffrhythm_workflow(workflow_data, self.positive_prompt)
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
                self.orchestrator_service.update_job_status(self.job_id, 'completed', output_path=audio_storage_path, duration_seconds=duration, completion_metadata=self.job.get('completion_metadata'))
            else:
                logging.error(f"DiffRhythm job {self.job_id} completed, but audio upload failed.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class TextToVideoJobProcessor(BaseJobProcessor):
    workflow_name = 'WAN22_TEXT_TO_VIDEO.json'

    def process(self):
        if DEV_MODE:
            # Dev mode logic remains unchanged
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return
            
        wf_ready = inject_prompt_into_text_to_video_workflow(workflow_data, self.positive_prompt, self.negative_prompt)
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
                self.orchestrator_service.update_job_status(self.job_id, 'completed', output_path=video_storage_path, thumbnail_path=thumbnail_storage_path, duration_seconds=duration)
            else:
                logging.error(f"Video upload failed for job {self.job_id}.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class ImageToVideoJobProcessor(BaseJobProcessor):
    workflow_name = 'WAN22_IMAGE_TO_VIDEO.json'

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

        wf_ready = inject_prompt_and_image_into_workflow(workflow_data, self.positive_prompt, self.negative_prompt, start_image_filename)
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
                self.orchestrator_service.update_job_status(self.job_id, 'completed', output_path=video_storage_path, thumbnail_path=thumbnail_storage_path, duration_seconds=duration)
            else:
                logging.error(f"Video upload failed for job {self.job_id}.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        finally:
            logging.info(f"Cleaning up temporary file: {temp_host_path}")
            os.remove(temp_host_path)

class TextToVideoLightningJobProcessor(TextToVideoJobProcessor):
    workflow_name = 'WAN22_LIGHTNING_TEXT_TO_VIDEO.json'

class ImageToVideoLightningJobProcessor(ImageToVideoJobProcessor):
    workflow_name = 'WAN22_LIGHTNING_IMAGE_TO_VIDEO.json'
