"""LTX-2.3 Image-to-Video Processor (ComfyUI backend, 24GB VRAM)."""

import os
import logging

from config import DEV_MODE, SUPABASE_URL
from services.docker_manager import docker_manager
from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import VideoOutputHandler
from utils.comfyui_workflow_utils import (
    inject_prompt_and_image_into_ltx23_video_workflow,
    materialize_start_image,
)


class LTX23ComfyUIImageToVideoProcessor(ComfyUIProcessor, VideoOutputHandler):
    """ComfyUI-based LTX-2.3 I2V processor. Applies the distilled LoRA via
    LoraLoaderModelOnly (node 2 in the workflow) to fix the grey-colour issue."""

    def process(self):
        if DEV_MODE:
            return

        if not self.job:
            self._fail_job(f"Job is None for LTX23ComfyUIImageToVideoProcessor.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        start_image_filename = None

        start_image_url = inputs.get("start_image_url")
        if start_image_url:
            logging.info(f"Downloading start image from signed URL: {start_image_url}")
            downloaded_path = self.orchestrator_service.download_asset_by_url(
                start_image_url, self.input_dir
            )
            if downloaded_path:
                start_image_filename = os.path.basename(downloaded_path)

        if not start_image_filename:
            start_image_filename = materialize_start_image(self.job, self.input_dir)

        if not start_image_filename:
            input_storage_path = self.job.get("input_storage_path")
            if not input_storage_path:
                possible_path = self.job.get("start_image_base64")
                if (
                    possible_path
                    and isinstance(possible_path, str)
                    and not possible_path.startswith("data:")
                    and len(possible_path) < 2048
                ):
                    input_storage_path = possible_path

            if input_storage_path:
                bucket = self.job.get("bucket", "projects_public")
                supabase_url = os.environ.get(
                    "SUPABASE_URL", self.client.config.get("SUPABASE_URL", SUPABASE_URL)
                )
                if supabase_url:
                    source_url = f"{supabase_url}/storage/v1/object/public/{bucket}/{input_storage_path}"
                    logging.info(f"Downloading start image from: {source_url}")
                    downloaded_path = self.orchestrator_service.download_asset_by_url(
                        source_url, self.input_dir
                    )
                    if downloaded_path:
                        start_image_filename = os.path.basename(downloaded_path)
                else:
                    logging.warning("SUPABASE_URL not found; cannot download start image.")

        if not start_image_filename:
            self._fail_job(f"Failed to materialise start image for job {self.job_id}.")
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
                import shutil
                os.makedirs(os.path.dirname(container_input_path), exist_ok=True)
                shutil.copy2(start_image_full_path, container_input_path)
                logging.info(f"Copied start image to {container_input_path} (Headless)")
        except Exception as e:
            self._fail_job(f"Failed to copy start image to container: {e}")
            return

        aspect_ratio = inputs.get("aspect_ratio", "16:9")
        steps = inputs.get("steps")
        cfg_scale = inputs.get("cfg_scale")
        strength = inputs.get("strength")
        if strength is None:
            strength = 0.65

        wf_ready = inject_prompt_and_image_into_ltx23_video_workflow(
            workflow_data,
            self.positive_prompt,
            self.negative_prompt,
            start_image_filename,
            aspect_ratio,
            steps=steps,
            cfg_scale=cfg_scale,
            strength=strength,
            tier="24gb",
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
        logging.info(
            f"LTX-2.3 ComfyUI I2V job {self.job_id} completed: {video_storage_path}"
        )
