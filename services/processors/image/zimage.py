"""
Z-Image Processors

Processors for Z-Image-Turbo text-to-image, ControlNet, and inpainting.
"""

import logging
import os
import base64
import uuid

from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import ImageOutputHandler
from utils.comfyui_workflow_utils import (
    inject_prompt_into_zimage_workflow,
    inject_image_into_zimage_controlnet_workflow,
    inject_image_into_zimage_inpaint_workflow,
    materialize_start_image,
)


class ZImageTextToImageProcessor(ComfyUIProcessor, ImageOutputHandler):
    """Processor for Z-Image text-to-image generation."""

    def process(self):
        if not self.job:
            self._fail_job("Job object is None for ZImageTextToImageProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        aspect_ratio = inputs.get("aspect_ratio", "1:1")
        
        # Build advanced settings dict from inputs
        advanced_settings = {}
        if "steps" in inputs:
            advanced_settings["steps"] = inputs["steps"]
        if "cfg" in inputs:
            advanced_settings["cfg"] = inputs["cfg"]
        if "sampler_name" in inputs:
            advanced_settings["sampler_name"] = inputs["sampler_name"]
        if "scheduler" in inputs:
            advanced_settings["scheduler"] = inputs["scheduler"]
        if "shift" in inputs:
            advanced_settings["shift"] = inputs["shift"]

        wf_ready = inject_prompt_into_zimage_workflow(
            workflow_data, 
            self.positive_prompt, 
            aspect_ratio,
            advanced_settings=advanced_settings if advanced_settings else None
        )
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        image_storage_path = self.handle_image_output(outputs)
        if not image_storage_path:
            return

        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=image_storage_path,
            thumbnail_storage_path=image_storage_path,
            prompt=self.positive_prompt,
        )


class ZImageControlNetProcessor(ComfyUIProcessor, ImageOutputHandler):
    """Processor for Z-Image ControlNet-based image editing."""

    def process(self):
        if not self.job:
            self._fail_job("Job object is None for ZImageControlNetProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        aspect_ratio = inputs.get("aspect_ratio", "1:1")
        control_mode = inputs.get("control_mode", "pose")
        strength = inputs.get("control_strength", 1.0)

        # Materialize the reference image to host
        reference_image_filename = materialize_start_image(self.job, self.client.input_dir)
        if not reference_image_filename:
            self._fail_job("No reference image provided for ControlNet workflow.")
            return

        # Copy the image to the Docker container
        reference_image_full_path = os.path.join(self.client.input_dir, reference_image_filename)
        try:
            from services.docker_manager import docker_manager
            container_input_path = f"/opt/ComfyUI/input/{reference_image_filename}"
            docker_manager.copy_file_to_container(
                service_type=self.client.active_service_type,
                source_on_host=reference_image_full_path,
                dest_in_container=container_input_path,
                shutdown_event=self.shutdown_event,
            )
        except Exception as e:
            self._fail_job(f"Failed to copy reference image to container: {e}")
            return

        wf_ready = inject_image_into_zimage_controlnet_workflow(
            workflow_data,
            self.positive_prompt,
            reference_image_filename,
            control_mode=control_mode,
            strength=strength,
            aspect_ratio=aspect_ratio,
        )
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        image_storage_path = self.handle_image_output(outputs)
        if not image_storage_path:
            return

        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=image_storage_path,
            thumbnail_storage_path=image_storage_path,
            prompt=self.positive_prompt,
        )


class ZImageInpaintProcessor(ComfyUIProcessor, ImageOutputHandler):
    """Processor for Z-Image mask-based inpainting."""

    def process(self):
        if not self.job:
            self._fail_job("Job object is None for ZImageInpaintProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        denoise_strength = inputs.get("denoise_strength", 0.8)

        # Materialize the source image
        source_image_filename = materialize_start_image(self.job, self.client.input_dir)
        if not source_image_filename:
            self._fail_job("No source image provided for inpaint workflow.")
            return

        # Materialize the mask from base64
        mask_base64 = inputs.get("mask_image_base64")
        if not mask_base64:
            self._fail_job("No mask image provided for inpaint workflow.")
            return

        try:
            # Remove data URL prefix if present
            if "," in mask_base64:
                mask_base64 = mask_base64.split(",")[1]
            
            mask_data = base64.b64decode(mask_base64)
            mask_filename = f"mask_{uuid.uuid4().hex[:8]}.png"
            mask_path = os.path.join(self.client.input_dir, mask_filename)
            
            with open(mask_path, "wb") as f:
                f.write(mask_data)
            
            logging.info(f"Saved mask to: {mask_path}")
        except Exception as e:
            self._fail_job(f"Failed to process mask image: {str(e)}")
            return

        wf_ready = inject_image_into_zimage_inpaint_workflow(
            workflow_data,
            self.positive_prompt,
            source_image_filename,
            mask_filename,
            denoise_strength=denoise_strength,
        )
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        image_storage_path = self.handle_image_output(outputs)
        if not image_storage_path:
            return

        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=image_storage_path,
            thumbnail_storage_path=image_storage_path,
            prompt=self.positive_prompt,
        )

