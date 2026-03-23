"""
LTX-2.3 Multi-Frame Image-to-Video Processor (32GB VRAM tier)

Generates audio+video interpolating between a start frame and an end frame.
Uses two LTXVImgToVideoConditionOnly nodes with LatentFlip to anchor:
  - Start image at frame 0
  - End image at the last frame
The model then generates the interpolating motion and audio between them.

Both start and end images are required. The workflow uses:
  - Node 21/22/23: start frame conditioning (normal direction)
  - Node 26/27/28: end frame conditioning (on reversed latent via LatentFlip)
  - Node 29/30: temporal flip and unflip of the latent

Requires 64GB+ system RAM for CPU offloading the 46GB BF16 model.
"""

import os
import logging

from config import DEV_MODE, SUPABASE_URL
from services.docker_manager import docker_manager
from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import VideoOutputHandler
from utils.comfyui_workflow_utils import (
    inject_prompt_and_images_into_ltx23_multiframe_workflow,
    materialize_start_image,
)


class LTX23MultiFrameImageToVideo32GBJobProcessor(ComfyUIProcessor, VideoOutputHandler):
    """Processor for LTX-2.3 multi-frame (start+end) image-to-video + audio generation (32GB tier)."""

    def process(self):
        if DEV_MODE:
            return

        if not self.job:
            self._fail_job(f"Job object is None for LTX23MultiFrameImageToVideo32GBJobProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})

        # --- Materialize start image ---
        start_image_filename = self._get_image_filename(
            inputs.get("start_image_url"),
            self.job.get("input_storage_path"),
            self.job.get("start_image_base64"),
            "start"
        )
        if not start_image_filename:
            self._fail_job(f"Failed to materialize start image for job {self.job_id}.")
            return

        # --- Materialize end image ---
        end_image_filename = self._get_image_filename(
            inputs.get("end_image_url"),
            inputs.get("end_image_storage_path"),
            inputs.get("end_image_base64"),
            "end"
        )
        if not end_image_filename:
            self._fail_job(f"Failed to materialize end image for job {self.job_id}. end_image_url or end_image_base64 required.")
            return

        # Copy both images into container
        for filename in [start_image_filename, end_image_filename]:
            full_path = os.path.join(self.input_dir, filename)
            container_path = f"/opt/ComfyUI/input/{filename}"
            try:
                if docker_manager:
                    docker_manager.copy_file_to_container(
                        service_type=self.client.active_service_type,
                        source_on_host=full_path,
                        dest_in_container=container_path,
                        shutdown_event=self.shutdown_event,
                    )
                else:
                    import shutil
                    os.makedirs(os.path.dirname(container_path), exist_ok=True)
                    shutil.copy2(full_path, container_path)
                    logging.info(f"Copied image to {container_path} (Headless)")
            except Exception as e:
                self._fail_job(f"Failed to copy {filename} to container: {e}")
                return

        aspect_ratio = inputs.get("aspect_ratio", "16:9")
        steps = inputs.get("steps")
        cfg_scale = inputs.get("cfg_scale")
        start_strength = inputs.get("start_strength", inputs.get("strength"))
        end_strength = inputs.get("end_strength")

        wf_ready = inject_prompt_and_images_into_ltx23_multiframe_workflow(
            workflow_data,
            self.positive_prompt,
            self.negative_prompt,
            start_image_filename,
            end_image_filename,
            aspect_ratio,
            steps=steps,
            cfg_scale=cfg_scale,
            start_strength=start_strength,
            end_strength=end_strength,
            tier="32gb",
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

    def _get_image_filename(self, url, storage_path, base64_data, label):
        """Download or materialize an image, return filename or None."""
        filename = None

        if url:
            logging.info(f"Downloading {label} image from signed URL: {url}")
            downloaded = self.orchestrator_service.download_asset_by_url(url, self.input_dir)
            if downloaded:
                filename = os.path.basename(downloaded)

        if not filename and label == "start":
            filename = materialize_start_image(self.job, self.input_dir)

        if not filename and base64_data:
            import base64
            import uuid
            try:
                if base64_data.startswith("data:"):
                    base64_data = base64_data.split(",", 1)[1]
                img_bytes = base64.b64decode(base64_data)
                filename = f"{label}_image_{uuid.uuid4().hex[:8]}.png"
                full_path = os.path.join(self.input_dir, filename)
                with open(full_path, "wb") as f:
                    f.write(img_bytes)
                logging.info(f"Saved {label} image from base64: {filename}")
            except Exception as e:
                logging.warning(f"Failed to decode {label} image from base64: {e}")

        if not filename and storage_path:
            bucket = self.job.get("bucket", "projects_public")
            supabase_url = os.environ.get('SUPABASE_URL', self.client.config.get('SUPABASE_URL', SUPABASE_URL))
            if supabase_url:
                source_url = f"{supabase_url}/storage/v1/object/public/{bucket}/{storage_path}"
                logging.info(f"Downloading {label} image from storage: {source_url}")
                downloaded = self.orchestrator_service.download_asset_by_url(source_url, self.input_dir)
                if downloaded:
                    filename = os.path.basename(downloaded)

        return filename
