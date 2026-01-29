"""
Video Upscaler Processor
"""

import os
import logging

from services.docker_manager import docker_manager
from config import THUMBNAIL_WIDTH
from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import VideoOutputHandler
from utils.comfyui_workflow_utils import inject_video_into_upscaler_workflow
from utils.media_utils import (
    find_video_in_output,
    generate_thumbnail,
    get_video_duration,
    get_video_dimensions,
    get_video_framerate,
)


class VideoUpscalerJobProcessor(ComfyUIProcessor, VideoOutputHandler):
    """Processor for video upscaling with Stream-DiffVSR."""

    def process(self):
        if not self.job:
            self._fail_job(f"Job object is None for VideoUpscalerJobProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        input_video_url = self.job.get("input_video_url")
        if not input_video_url:
            self._fail_job(f"Video upscaler job {self.job_id} missing 'input_video_url'.")
            return

        video_path = self.orchestrator_service.download_asset_by_url(input_video_url, self.input_dir)
        if not video_path:
            self._fail_job(f"Failed to download input video for upscaler job {self.job_id}.")
            return

        video_filename = os.path.basename(video_path)

        try:
            container_input_path = f"/opt/ComfyUI/input/{video_filename}"
            if docker_manager:
                docker_manager.copy_file_to_container(
                    service_type=self.client.active_service_type,
                    source_on_host=video_path,
                    dest_in_container=container_input_path,
                    shutdown_event=self.shutdown_event,
                )
            else:
                # Headless Mode: Copy locally
                import shutil
                os.makedirs(os.path.dirname(container_input_path), exist_ok=True)
                shutil.copy2(video_path, container_input_path)
                logging.info(f"Copied video to {container_input_path} (Headless)")
            logging.info(f"Copied video to container: {container_input_path}")
        except Exception as e:
            self._fail_job(f"Failed to copy video to container for job {self.job_id}: {e}")
            return

        try:
            original_width, original_height = get_video_dimensions(video_path)
            logging.info(f"Original video dimensions: {original_width}x{original_height}")
        except Exception as e:
            self._fail_job(f"Could not get video dimensions for {video_path}: {e}")
            return

        job_inputs = self.job.get("inputs", {})
        upscale_model = job_inputs.get("upscale_model", "Stream-DiffVSR")

        input_frame_rate = job_inputs.get("frame_rate")
        if input_frame_rate:
            frame_rate = int(input_frame_rate)
        else:
            frame_rate = int(get_video_framerate(video_path))
            logging.info(f"Detected source frame rate: {frame_rate}")

        # Stream-DiffVSR usually does 4x. Params for generic resizing might be ignored by the wrapper
        # unless we explicitly add resizing nodes.
        # For now, let's keep the params but they might not be used by inject function
        upscale_factor = job_inputs.get("upscale_factor")
        target_width = job_inputs.get("target_width")
        target_height = job_inputs.get("target_height")
        control_method = job_inputs.get("control_method", "factor")

        final_width, final_height, scale_by = self._calculate_upscale_params(
            control_method, upscale_factor, target_width, target_height, original_width, original_height
        )

        wf_ready = inject_video_into_upscaler_workflow(
            workflow_data,
            video_filename,
            upscale_model,
            frame_rate,
            target_width=final_width,
            target_height=final_height,
            scale_by=scale_by,
        )

        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        video_info = find_video_in_output(outputs)
        if not video_info:
            self._fail_job(f"Upscaler workflow for job {self.job_id} completed, but no video file found.")
            return

        output_video_filename, subfolder = video_info
        temp_host_path = self._copy_file_from_container(output_video_filename, subfolder)
        if not temp_host_path:
            self._fail_job(f"Failed to copy output file from container for job {self.job_id}.")
            return

        try:
            video_storage_path = self.orchestrator_service.upload_output(temp_host_path, self.job_id, "video/mp4")
            if video_storage_path:
                thumbnail_local_path = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
                thumbnail_storage_path = None

                if generate_thumbnail(temp_host_path, thumbnail_local_path, width=THUMBNAIL_WIDTH):
                    thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(
                        thumbnail_local_path, self.job_id
                    )
                    if os.path.exists(thumbnail_local_path):
                        os.remove(thumbnail_local_path)

                duration = get_video_duration(temp_host_path)
                completion_metadata = self.job.get("completion_metadata") or {}
                completion_metadata["upscale_model"] = upscale_model

                self.orchestrator_service.update_job_status(
                    self.job_id,
                    "completed",
                    storage_path=video_storage_path,
                    thumbnail_storage_path=thumbnail_storage_path,
                    duration_seconds=duration,
                    completion_metadata=completion_metadata,
                )
            else:
                self._fail_job(f"Video upscaler job {self.job_id} completed, but video upload failed.")
        finally:
            if os.path.exists(temp_host_path):
                logging.info(f"Cleaning up temporary file: {temp_host_path}")
                os.remove(temp_host_path)

    def _calculate_upscale_params(
        self, control_method, upscale_factor, target_width, target_height, original_width, original_height
    ):
        """Calculate final upscale parameters based on control method."""
        final_width = None
        final_height = None
        scale_by = None

        if control_method == "dimensions":
            if target_width and target_height:
                final_width = int(target_width)
                final_height = int(target_height)
                logging.info(f"Using explicit target dimensions: {final_width}x{final_height}")
            elif target_width:
                ratio = float(target_width) / float(original_width)
                final_width = int(target_width)
                final_height = int(original_height * ratio)
                logging.info(f"Using target width {target_width} and calculated height {final_height}")
            elif target_height:
                ratio = float(target_height) / float(original_height)
                final_height = int(target_height)
                final_width = int(original_width * ratio)
                logging.info(f"Using target height {target_height} and calculated width {final_width}")
            else:
                logging.warning("Control method is dimensions but none provided. Falling back to 2x upscale.")
                final_width = original_width * 2
                final_height = original_height * 2
        else:
            if upscale_factor:
                scale_by = float(upscale_factor)
                logging.info(f"Using upscale factor {scale_by} with ImageScaleBy.")
            else:
                scale_by = 2.0
                logging.info(f"No factor or dimensions provided. Defaulting to 2x upscale.")

        if scale_by is None:
            if final_width % 2 != 0:
                final_width += 1
            if final_height % 2 != 0:
                final_height += 1

        return final_width, final_height, scale_by
