import os
import logging
from abc import ABC, abstractmethod
from config import DEV_MODE
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
        self.output_dir = client.output_dir
        self.positive_prompt = job.get('prompt') or ""
        self.negative_prompt = job.get('negative_prompt') or ""

    def _check_interruption(self, outputs):
        if outputs == "interrupted":
            logging.warning(f"Processing of job {self.job_id} was interrupted by shutdown.")
            return True
        return False

    def _trigger_and_get_output(self, payload):
        prompt_id = self.comfyui_client.trigger_workflow(payload)
        if not prompt_id:
            logging.error(f"Failed to trigger workflow for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return None

        outputs = self.comfyui_client.get_workflow_output(prompt_id, timeout_sec=7200, shutdown_event=self.shutdown_event)
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
        workflow_api_path = os.path.join(self.root_dir, 'workflows', 'hunyuan-video-foley.api.json')
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
        wf_ready = inject_video_and_prompt_into_foley_workflow(workflow_api_path, video_filename, self.positive_prompt, self.negative_prompt)
        
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        audio_info = find_audio_in_output(outputs)
        if audio_info:
            audio_filename, subfolder = audio_info
            local_audio_path = os.path.join(self.output_dir, subfolder, audio_filename)
            audio_storage_path = self.orchestrator_service.upload_audio_output(local_audio_path, self.job_id)
            if audio_storage_path:
                duration = get_audio_duration(local_audio_path)
                self.orchestrator_service.update_job_status(self.job_id, 'completed', output_path=audio_storage_path, duration_seconds=duration, completion_metadata=self.job.get('completion_metadata'))
            else:
                logging.error(f"Foley job {self.job_id} completed, but audio upload failed.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        else:
            logging.error(f"Foley workflow for job {self.job_id} completed, but no audio file found.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')

class TextToImageJobProcessor(BaseJobProcessor):
    def process(self):
        workflow_api_path = os.path.join(self.root_dir, 'workflows', 'qwen.api.json')
        if not os.path.exists(workflow_api_path):
            logging.error(f"Workflow API file not found at {workflow_api_path}")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        wf_ready = inject_prompt_into_qwen_workflow(workflow_api_path, self.positive_prompt, self.negative_prompt)
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        image_info = find_image_in_output(outputs)
        if image_info:
            image_filename, subfolder = image_info
            local_image_path = os.path.join(self.output_dir, subfolder, image_filename)
            image_storage_path = self.orchestrator_service.upload_image_output(local_image_path, self.job_id)
            if image_storage_path:
                self.orchestrator_service.update_job_status(self.job_id, 'completed', output_path=image_storage_path, thumbnail_path=image_storage_path, prompt=self.positive_prompt)
            else:
                logging.error(f"Image upload failed for job {self.job_id}.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        else:
            logging.error(f"Workflow for job {self.job_id} completed, but no image file found.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')

class VibeVoiceJobProcessor(BaseJobProcessor):
    def process(self):
        workflow_api_path = os.path.join(self.root_dir, 'workflows', 'vibevoice.api.json')
        if not os.path.exists(workflow_api_path):
            logging.error(f"Workflow API file not found at {workflow_api_path}")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        wf_ready = inject_prompt_into_vibevoice_workflow(workflow_api_path, self.positive_prompt)
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        audio_info = find_audio_in_output(outputs)
        if not audio_info:
            logging.warning("find_audio_in_output failed. Looking for {'audio': [...]} pattern based on logs.")
            for node_id, node_output in outputs.items():
                if 'audio' in node_output and isinstance(node_output.get('audio'), list):
                    for item in node_output['audio']:
                        if isinstance(item, dict):
                            filename = item.get('filename')
                            if filename:
                                audio_info = (filename, item.get('subfolder', ''))
                                break
                if audio_info:
                    break
        
        if audio_info:
            audio_filename, subfolder = audio_info
            local_audio_path = os.path.join(self.output_dir, subfolder, audio_filename)
            audio_storage_path = self.orchestrator_service.upload_audio_output(local_audio_path, self.job_id)
            if audio_storage_path:
                duration = get_audio_duration(local_audio_path)
                self.orchestrator_service.update_job_status(self.job_id, 'completed', output_path=audio_storage_path, duration_seconds=duration, completion_metadata=self.job.get('completion_metadata'))
            else:
                logging.error(f"VibeVoice job {self.job_id} completed, but audio upload failed.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        else:
            logging.error(f"VibeVoice workflow for job {self.job_id} completed, but no audio file found.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')

class VibeVoiceMultiCloneJobProcessor(BaseJobProcessor):
    def process(self):
        workflow_api_path = os.path.join(self.root_dir, 'workflows', 'vibevoice-multi-speaker-clone.api.json')
        if not os.path.exists(workflow_api_path):
            logging.error(f"Workflow API file not found at {workflow_api_path}")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
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

        wf_ready = inject_script_and_clones_into_vibevoice_workflow(workflow_api_path, self.positive_prompt, clone_paths)
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        audio_info = find_audio_in_output(outputs)
        if audio_info:
            audio_filename, subfolder = audio_info
            local_audio_path = os.path.join(self.output_dir, subfolder, audio_filename)
            audio_storage_path = self.orchestrator_service.upload_audio_output(local_audio_path, self.job_id)
            if audio_storage_path:
                duration = get_audio_duration(local_audio_path)
                self.orchestrator_service.update_job_status(self.job_id, 'completed', output_path=audio_storage_path, duration_seconds=duration, completion_metadata=self.job.get('completion_metadata'))
            else:
                logging.error(f"VibeVoice multi-clone job {self.job_id} completed, but audio upload failed.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        else:
            logging.error(f"VibeVoice multi-clone workflow for job {self.job_id} completed, but no audio file found.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')

class DiffRhythmJobProcessor(BaseJobProcessor):
    def process(self):
        workflow_api_path = os.path.join(self.root_dir, 'workflows', 'diffrhythm.api.json')
        if not os.path.exists(workflow_api_path):
            logging.error(f"Workflow API file not found at {workflow_api_path}")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        wf_ready = inject_prompt_into_diffrhythm_workflow(workflow_api_path, self.positive_prompt)
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        audio_info = find_audio_in_output(outputs)
        if audio_info:
            audio_filename, subfolder = audio_info
            local_audio_path = os.path.join(self.output_dir, subfolder, audio_filename)
            audio_storage_path = self.orchestrator_service.upload_audio_output(local_audio_path, self.job_id)
            if audio_storage_path:
                duration = get_audio_duration(local_audio_path)
                self.orchestrator_service.update_job_status(self.job_id, 'completed', output_path=audio_storage_path, duration_seconds=duration, completion_metadata=self.job.get('completion_metadata'))
            else:
                logging.error(f"DiffRhythm job {self.job_id} completed, but audio upload failed.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        else:
            logging.error(f"DiffRhythm workflow for job {self.job_id} completed, but no audio file found.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')

class TextToVideoJobProcessor(BaseJobProcessor):
    def process(self):
        if DEV_MODE:
            logging.info(f"DEV_MODE is True. Using sample video for job {self.job_id}.")
            
            sample_dir_path = os.path.join(self.output_dir, "2025-08-15")
            local_video_path = os.path.join(sample_dir_path, "wan22__00001.mp4")
            thumbnail_local_path = os.path.join(sample_dir_path, "wan22__00001.png")

            if not os.path.exists(local_video_path) or not os.path.exists(thumbnail_local_path):
                logging.error(f"Sample files not found for DEV_MODE in {sample_dir_path}. Looked for wan22__00001.mp4 and wan22__00001.png.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                return

            video_storage_path = self.orchestrator_service.upload_output(local_video_path, self.job_id)
            
            if video_storage_path:
                thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(thumbnail_local_path, self.job_id)
                
                duration = get_video_duration(local_video_path)
                self.orchestrator_service.update_job_status(self.job_id, 'completed', output_path=video_storage_path, thumbnail_path=thumbnail_storage_path, duration_seconds=duration)
            else:
                logging.error(f"DEV_MODE: Video upload failed for job {self.job_id}.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return
            
        workflow_api_path = os.path.join(self.root_dir, 'workflows', 'wan2.2-text-to-video.api.json')
        if not os.path.exists(workflow_api_path):
            logging.error(f"Workflow API file not found at {workflow_api_path}")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        wf_ready = inject_prompt_into_text_to_video_workflow(workflow_api_path, self.positive_prompt, self.negative_prompt)
        
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        video_info = find_video_in_output(outputs)
        if video_info:
            video_filename, subfolder = video_info
            local_video_path = os.path.join(self.output_dir, subfolder, video_filename)
            video_storage_path = self.orchestrator_service.upload_output(local_video_path, self.job_id)
            
            if video_storage_path:
                thumbnail_filename = os.path.splitext(video_filename)[0] + ".jpg"
                thumbnail_local_path = os.path.join(self.output_dir, subfolder, thumbnail_filename)
                thumbnail_storage_path = None
                
                if generate_thumbnail(local_video_path, thumbnail_local_path):
                    thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(thumbnail_local_path, self.job_id)
                
                duration = get_video_duration(local_video_path)
                self.orchestrator_service.update_job_status(self.job_id, 'completed', output_path=video_storage_path, thumbnail_path=thumbnail_storage_path, duration_seconds=duration)
            else:
                logging.error(f"Video upload failed for job {self.job_id}.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        else:
            logging.error(f"Workflow for job {self.job_id} completed, but no video file found.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')

class ImageToVideoJobProcessor(BaseJobProcessor):
    def process(self):
        if DEV_MODE:
            logging.info(f"DEV_MODE is True. Using sample video for job {self.job_id}.")
            
            sample_dir_path = os.path.join(self.output_dir, "2025-08-15")
            local_video_path = os.path.join(sample_dir_path, "wan22__00001.mp4")
            thumbnail_local_path = os.path.join(sample_dir_path, "wan22__00001.png")

            if not os.path.exists(local_video_path) or not os.path.exists(thumbnail_local_path):
                logging.error(f"Sample files not found for DEV_MODE in {sample_dir_path}. Looked for wan22__00001.mp4 and wan22__00001.png.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                return

            video_storage_path = self.orchestrator_service.upload_output(local_video_path, self.job_id)
            
            if video_storage_path:
                thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(thumbnail_local_path, self.job_id)
                
                duration = get_video_duration(local_video_path)
                self.orchestrator_service.update_job_status(self.job_id, 'completed', output_path=video_storage_path, thumbnail_path=thumbnail_storage_path, duration_seconds=duration)
            else:
                logging.error(f"DEV_MODE: Video upload failed for job {self.job_id}.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        workflow_api_path = os.path.join(self.root_dir, 'workflows', 'wan2.2-image-to-video.api.json')
        if not os.path.exists(workflow_api_path):
            logging.error(f"Workflow API file not found at {workflow_api_path}")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        start_image_filename = materialize_start_image(self.job, self.input_dir)
        wf_ready = inject_prompt_and_image_into_workflow(workflow_api_path, self.positive_prompt, self.negative_prompt, start_image_filename)
        
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        video_info = find_video_in_output(outputs)
        if video_info:
            video_filename, subfolder = video_info
            local_video_path = os.path.join(self.output_dir, subfolder, video_filename)
            video_storage_path = self.orchestrator_service.upload_output(local_video_path, self.job_id)
            
            if video_storage_path:
                thumbnail_filename = os.path.splitext(video_filename)[0] + ".jpg"
                thumbnail_local_path = os.path.join(self.output_dir, subfolder, thumbnail_filename)
                thumbnail_storage_path = None
                
                if generate_thumbnail(local_video_path, thumbnail_local_path):
                    thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(thumbnail_local_path, self.job_id)
                
                duration = get_video_duration(local_video_path)
                self.orchestrator_service.update_job_status(self.job_id, 'completed', output_path=video_storage_path, thumbnail_path=thumbnail_storage_path, duration_seconds=duration)
            else:
                logging.error(f"Video upload failed for job {self.job_id}.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        else:
            logging.error(f"Workflow for job {self.job_id} completed, but no video file found.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
