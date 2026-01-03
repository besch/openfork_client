"""
Hunyuan 1.5 Image-to-Video Processor
"""

import os
import logging

from config import DEV_MODE
from services.docker_manager import docker_manager
from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import VideoOutputHandler
from utils.comfyui_workflow_utils import (
    inject_prompt_and_image_into_hunyuan_video_workflow,
    materialize_start_image,
)


class HunyuanImageToVideoJobProcessor(ComfyUIProcessor, VideoOutputHandler):
    """Processor for Hunyuan 1.5 image-to-video generation."""

    def process(self):
        if DEV_MODE:
            return

        if not self.job:
            self._fail_job(f"Job object is None for HunyuanImageToVideoJobProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        start_image_filename = materialize_start_image(self.job, self.input_dir)
        
        # If not materialized from base64, try downloading from storage path
        if not start_image_filename:
            input_storage_path = self.job.get("input_storage_path")
            
            # Fallback: sometimes the path is passed in start_image_base64 (legacy/action behavior)
            if not input_storage_path:
                possible_path = self.job.get('start_image_base64')
                # If it's short and doesn't look like a data URL, treat it as a path
                if possible_path and isinstance(possible_path, str) and not possible_path.startswith('data:') and len(possible_path) < 2048:
                    input_storage_path = possible_path

            if input_storage_path:
                bucket = self.job.get("bucket", "projects_public")
                supabase_url = os.environ.get('SUPABASE_URL', self.client.config.get('SUPABASE_URL', ''))
                if supabase_url:
                    source_url = f"{supabase_url}/storage/v1/object/public/{bucket}/{input_storage_path}"
                    
                    logging.info(f"Downloading start image from: {source_url}")
                    downloaded_path = self.orchestrator_service.download_asset_by_url(source_url, self.input_dir)
                    if downloaded_path:
                        start_image_filename = os.path.basename(downloaded_path)
                else:
                    logging.warning("SUPABASE_URL not found in environment or config. Cannot download input image.")

        if not start_image_filename:
            self._fail_job(f"Failed to materialize start image for job {self.job_id}.")
            return

        start_image_full_path = os.path.join(self.input_dir, start_image_filename)

        try:
            container_input_path = f"/opt/ComfyUI/input/{start_image_filename}"
            docker_manager.copy_file_to_container(
                service_type=self.client.active_service_type,
                source_on_host=start_image_full_path,
                dest_in_container=container_input_path,
                shutdown_event=self.shutdown_event,
            )
        except Exception as e:
            self._fail_job(f"Failed to copy start image to container for job {self.job_id}: {e}")
            return

        inputs = self.job.get("inputs", {})
        aspect_ratio = inputs.get("aspect_ratio", "16:9")
        cfg_scale = inputs.get("cfg_scale")
        steps = inputs.get("steps")
        flow_shift = inputs.get("flow_shift")
        sampler = inputs.get("sampler")
        scheduler = inputs.get("scheduler")

        wf_ready = inject_prompt_and_image_into_hunyuan_video_workflow(
            workflow_data, self.positive_prompt, self.negative_prompt, start_image_filename, aspect_ratio,
            cfg_scale=cfg_scale, steps=steps,
            flow_shift=flow_shift, sampler=sampler, scheduler=scheduler
        )
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        result = self.handle_video_output(outputs)
        if not result:
            return

        video_storage_path, thumbnail_storage_path, duration = result
        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=video_storage_path,
            thumbnail_storage_path=thumbnail_storage_path,
            duration_seconds=duration,
            prompt=self.positive_prompt,
        )
