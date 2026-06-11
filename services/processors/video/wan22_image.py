"""
WAN 2.2 Image-to-Video Processors
"""

import os
import logging

from config import DEV_MODE, SUPABASE_URL
from services.docker_manager import docker_manager
from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import VideoOutputHandler
from utils.comfyui_workflow_utils import (
    inject_prompt_and_image_into_workflow,
    materialize_start_image,
    resolve_wan22_dimensions,
)
from utils.media_utils import extract_last_frame


class WAN22ImageToVideoJobProcessor(ComfyUIProcessor, VideoOutputHandler):
    """Processor for WAN 2.2 image-to-video generation."""

    def process(self):
        if DEV_MODE:
            return

        if not self.job:
            self._fail_job(f"Job object is None for WAN22ImageToVideoJobProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        start_image_url = inputs.get("start_image_url")
        start_image_filename = None
        start_image_owned = False

        if start_image_url:
            logging.info("Downloading start image from signed URL")
            downloaded_path = self.orchestrator_service.download_asset_by_url(start_image_url, self.input_dir)
            if downloaded_path:
                start_image_filename = os.path.basename(downloaded_path)
                start_image_owned = True

        if not start_image_filename:
            start_image_owned = bool(
                self.job.get("start_image_base64")
                or inputs.get("start_image_base64")
            )
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
                # Construct URL assuming Supabase storage structure
                # Note: This relies on SUPABASE_URL being set in the environment or client config
                supabase_url = os.environ.get('SUPABASE_URL', self.client.config.get('SUPABASE_URL', SUPABASE_URL))
                if supabase_url:
                    source_url = f"{supabase_url}/storage/v1/object/public/{bucket}/{input_storage_path}"
                    
                    logging.info(f"Downloading start image from: {source_url}")
                    downloaded_path = self.orchestrator_service.download_asset_by_url(source_url, self.input_dir)
                    if downloaded_path:
                        start_image_filename = os.path.basename(downloaded_path)
                        start_image_owned = True
                else:
                    logging.warning("SUPABASE_URL not found in environment or config. Cannot download input image.")

        if not start_image_filename:
            self._fail_job(f"Failed to materialize start image for job {self.job_id}.")
            return

        start_image_full_path = os.path.join(self.input_dir, start_image_filename)
        container_input_path = f"/opt/ComfyUI/input/{start_image_filename}"

        try:
            if docker_manager:
                docker_manager.copy_file_to_container(
                    service_type=self.client.active_service_type,
                    source_on_host=start_image_full_path,
                    dest_in_container=container_input_path,
                    shutdown_event=self.shutdown_event,
                )
            else:
                # Headless Mode: Copy locally
                import shutil
                os.makedirs(os.path.dirname(container_input_path), exist_ok=True)
                shutil.copy2(start_image_full_path, container_input_path)
                logging.info(f"Copied start image to {container_input_path} (Headless)")
        except Exception as e:
            self._fail_job(f"Failed to copy start image to container for job {self.job_id}: {e}")
            return

        try:
            inputs = self.job.get("inputs", {})
            aspect_ratio = inputs.get("aspect_ratio", "16:9")
            cfg_scale = inputs.get("cfg_scale")
            steps = inputs.get("steps")
            flow_shift = inputs.get("flow_shift")
            sampler = inputs.get("sampler")
            scheduler = inputs.get("scheduler")
            seed = inputs.get("seed")
            target_width = inputs.get("target_width") or inputs.get("width")
            target_height = inputs.get("target_height") or inputs.get("height")
            vram_tier = str(inputs.get("model") or self.job.get("workflow_type") or "")

            wf_ready = inject_prompt_and_image_into_workflow(
                workflow_data, self.positive_prompt, self.negative_prompt, start_image_filename, aspect_ratio,
                cfg_scale=cfg_scale, steps=steps,
                flow_shift=flow_shift, sampler=sampler, scheduler=scheduler, seed=seed,
                vram_tier=vram_tier, target_width=target_width, target_height=target_height
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
        finally:
            if start_image_owned:
                self._cleanup_local_file(start_image_full_path, "WAN22 start image")
            self._cleanup_container_file(container_input_path, "WAN22 container start image")

class ImageToVideoFromLastFrameJobProcessor(ComfyUIProcessor, VideoOutputHandler):
    """Processor for creating video from the last frame of an input video."""

    def process(self):
        if DEV_MODE:
            return

        if not self.job:
            self._fail_job(f"Job object is None for ImageToVideoFromLastFrameJobProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        input_video_url = inputs.get("input_video_url") or self.job.get("input_video_url")
        if not input_video_url:
            input_video_url = inputs.get("start_image_url")
            if input_video_url:
                logging.info("Using start_image_url as source video URL for last-frame continuation.")
        if not input_video_url:
            input_storage_path = (
                inputs.get("input_storage_path")
                or inputs.get("start_image")
                or self.job.get("input_storage_path")
            )
            if input_storage_path:
                bucket = self.job.get("bucket", "projects_public")
                supabase_url = os.environ.get(
                    "SUPABASE_URL",
                    self.client.config.get("SUPABASE_URL", SUPABASE_URL),
                )
                if supabase_url:
                    input_video_url = f"{supabase_url}/storage/v1/object/public/{bucket}/{input_storage_path}"
                    logging.info("Using input storage path as source video URL for last-frame continuation.")
        if not input_video_url:
            self._fail_job(f"Job {self.job_id} missing 'input_video_url' in inputs.")
            return

        video_path = self.orchestrator_service.download_asset_by_url(input_video_url, self.input_dir)
        if not video_path:
            self._fail_job(f"Failed to download input video for job {self.job_id}.")
            return

        start_image_filename = f"{self.job_id}_last_frame.jpg"
        start_image_full_path = os.path.join(self.input_dir, start_image_filename)
        aspect_ratio = inputs.get("aspect_ratio", "16:9")
        cfg_scale = inputs.get("cfg_scale")
        steps = inputs.get("steps")
        flow_shift = inputs.get("flow_shift")
        sampler = inputs.get("sampler")
        scheduler = inputs.get("scheduler")
        seed = inputs.get("seed")
        target_width = inputs.get("target_width") or inputs.get("width")
        target_height = inputs.get("target_height") or inputs.get("height")
        vram_tier = str(inputs.get("model") or self.job.get("workflow_type") or "")
        target_dimensions = resolve_wan22_dimensions(
            aspect_ratio,
            vram_tier=vram_tier,
            target_width=target_width,
            target_height=target_height,
        )

        if not extract_last_frame(
            video_path,
            start_image_full_path,
            target_dimensions=target_dimensions,
        ):
            self._fail_job(f"Failed to extract last frame for job {self.job_id}.")
            return

        try:
            container_input_path = f"/opt/ComfyUI/input/{start_image_filename}"
            if docker_manager:
                docker_manager.copy_file_to_container(
                    service_type=self.client.active_service_type,
                    source_on_host=start_image_full_path,
                    dest_in_container=container_input_path,
                    shutdown_event=self.shutdown_event,
                )
            else:
                # Headless Mode: Copy locally
                import shutil
                os.makedirs(os.path.dirname(container_input_path), exist_ok=True)
                shutil.copy2(start_image_full_path, container_input_path)
                logging.info(f"Copied start image to {container_input_path} (Headless)")
        except Exception as e:
            self._fail_job(f"Failed to copy start image to container for job {self.job_id}: {e}")
            return

        wf_ready = inject_prompt_and_image_into_workflow(
            workflow_data, self.positive_prompt, self.negative_prompt, start_image_filename, aspect_ratio,
            cfg_scale=cfg_scale, steps=steps,
            flow_shift=flow_shift, sampler=sampler, scheduler=scheduler,
            seed=seed, vram_tier=vram_tier, target_width=target_width, target_height=target_height
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

        # Cleanup source video and extracted frame
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(start_image_full_path):
            os.remove(start_image_full_path)
