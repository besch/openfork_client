"""
YUME 1.5 Video Processors

Processors for YUME 1.5 text-to-video and image-to-video generation.
"""

import os
import logging

from config import DEV_MODE
from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import VideoOutputHandler
from utils.comfyui_workflow_utils import (
    inject_prompt_into_yume_workflow,
    inject_image_into_yume_workflow,
    materialize_start_image,
    get_dimensions,
)


class YumeTextToVideoJobProcessor(ComfyUIProcessor, VideoOutputHandler):
    """Processor for YUME 1.5 text-to-video generation."""

    def process(self):
        if DEV_MODE:
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        
        # Get aspect ratio and calculate dimensions
        aspect_ratio = inputs.get("aspect_ratio", "16:9")
        width, height = get_dimensions(aspect_ratio, default_width=1280, default_height=720)
        
        # Get advanced settings
        cfg_scale = inputs.get("cfg_scale", 7.0)
        steps = inputs.get("steps", 30)
        num_frames = inputs.get("num_frames", 49)
        frame_rate = inputs.get("frame_rate", 24)
        seed = inputs.get("seed", None)

        wf_ready = inject_prompt_into_yume_workflow(
            workflow_data,
            prompt=self.positive_prompt,
            negative_prompt=self.negative_prompt,
            frame_rate=frame_rate,
            steps=steps,
            cfg=cfg_scale,
            seed=seed,
            width=width,
            height=height,
            num_frames=num_frames,
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


class YumeImageToVideoJobProcessor(ComfyUIProcessor, VideoOutputHandler):
    """Processor for YUME 1.5 image-to-video generation."""

    def process(self):
        if DEV_MODE:
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        
        # Materialize start image if provided
        start_image_filename = materialize_start_image(self.job, self.input_dir)
        if not start_image_filename:
            logging.error(f"Failed to materialize start image for job {self.job_id}")
            self.orchestrator_service.update_job_status(self.job_id, "failed")
            return
        
        # Get aspect ratio and calculate dimensions
        aspect_ratio = inputs.get("aspect_ratio", "16:9")
        width, height = get_dimensions(aspect_ratio, default_width=1280, default_height=720)
        
        # Get advanced settings
        cfg_scale = inputs.get("cfg_scale", 7.0)
        steps = inputs.get("steps", 30)
        num_frames = inputs.get("num_frames", 49)
        frame_rate = inputs.get("frame_rate", 24)
        seed = inputs.get("seed", None)

        wf_ready = inject_image_into_yume_workflow(
            workflow_data,
            image_path=start_image_filename,
            prompt=self.positive_prompt,
            negative_prompt=self.negative_prompt,
            frame_rate=frame_rate,
            steps=steps,
            cfg=cfg_scale,
            seed=seed,
            width=width,
            height=height,
            num_frames=num_frames,
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
