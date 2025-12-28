"""
Qwen-Image-Edit-2511 Processors

Processors for Qwen instruction-based image editing and inpainting.
"""

import logging
import os
import base64
import uuid

from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import ImageOutputHandler


class QwenImageEditProcessor(ComfyUIProcessor, ImageOutputHandler):
    """Processor for Qwen instruction-based image editing."""

    def process(self):
        if not self.job:
            self._fail_job("Job object is None for QwenImageEditProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        denoise_strength = inputs.get("denoise_strength", 0.7)

        # Get source image - either from base64 or storage path
        source_image_filename = self._get_source_image()
        if not source_image_filename:
            return

        # Copy the image to the Docker container
        self._copy_image_to_container(source_image_filename)

        # Inject prompt and image into workflow
        wf_ready = self._inject_edit_workflow(
            workflow_data,
            self.positive_prompt,
            source_image_filename,
            denoise_strength,
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

    def _get_source_image(self):
        """Get source image from base64 or storage path."""
        from utils.comfyui_workflow_utils import materialize_start_image
        
        # Try base64 first
        source_image_filename = materialize_start_image(self.job, self.client.input_dir)
        
        # If no base64, try downloading from storage path
        if not source_image_filename:
            input_storage_path = self.job.get("input_storage_path")
            if input_storage_path:
                bucket = self.job.get("bucket", "projects_public")
                source_url = f"{os.environ.get('SUPABASE_URL', '')}/storage/v1/object/public/{bucket}/{input_storage_path}"
                
                logging.info(f"Downloading source image from: {source_url}")
                downloaded_path = self.orchestrator_service.download_asset_by_url(source_url, self.client.input_dir)
                if downloaded_path:
                    source_image_filename = os.path.basename(downloaded_path)
                    logging.info(f"Downloaded source image: {source_image_filename}")
        
        if not source_image_filename:
            self._fail_job("No source image provided for Qwen edit workflow.")
            return None
        
        return source_image_filename

    def _copy_image_to_container(self, image_filename):
        """Copy image to Docker container."""
        image_full_path = os.path.join(self.client.input_dir, image_filename)
        try:
            from services.docker_manager import docker_manager
            from config import HEADLESS_MODE
            
            if not HEADLESS_MODE:
                container_input_path = f"/opt/ComfyUI/input/{image_filename}"
                docker_manager.copy_file_to_container(
                    service_type=self.client.active_service_type,
                    source_on_host=image_full_path,
                    dest_in_container=container_input_path,
                    shutdown_event=self.shutdown_event,
                )
        except Exception as e:
            self._fail_job(f"Failed to copy image to container: {e}")

    def _inject_edit_workflow(self, workflow_data, prompt, image_filename, denoise_strength):
        """Inject prompt and image into the editing workflow."""
        wf = workflow_data.copy()
        
        # Update prompt in CLIPTextEncode node
        for node_id, node in wf.items():
            if node.get("class_type") == "CLIPTextEncode":
                node["inputs"]["text"] = prompt
            elif node.get("class_type") == "LoadImage":
                node["inputs"]["image"] = image_filename
            elif node.get("class_type") == "KSampler":
                node["inputs"]["denoise"] = denoise_strength
        
        return wf


class QwenImageInpaintProcessor(ComfyUIProcessor, ImageOutputHandler):
    """Processor for Qwen mask-based inpainting."""

    def process(self):
        if not self.job:
            self._fail_job("Job object is None for QwenImageInpaintProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        denoise_strength = inputs.get("denoise_strength", 0.8)

        # Get source image
        source_image_filename = self._get_source_image()
        if not source_image_filename:
            return

        # Materialize the mask from base64
        mask_filename = self._get_mask_image(inputs)
        if not mask_filename:
            return

        # Copy images to Docker container
        self._copy_images_to_container(source_image_filename, mask_filename)

        # Inject into workflow
        wf_ready = self._inject_inpaint_workflow(
            workflow_data,
            self.positive_prompt,
            source_image_filename,
            mask_filename,
            denoise_strength,
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

    def _get_source_image(self):
        """Get source image from base64 or storage path."""
        from utils.comfyui_workflow_utils import materialize_start_image
        
        source_image_filename = materialize_start_image(self.job, self.client.input_dir)
        
        if not source_image_filename:
            input_storage_path = self.job.get("input_storage_path")
            if input_storage_path:
                bucket = self.job.get("bucket", "projects_public")
                source_url = f"{os.environ.get('SUPABASE_URL', '')}/storage/v1/object/public/{bucket}/{input_storage_path}"
                
                logging.info(f"Downloading source image from: {source_url}")
                downloaded_path = self.orchestrator_service.download_asset_by_url(source_url, self.client.input_dir)
                if downloaded_path:
                    source_image_filename = os.path.basename(downloaded_path)
                    logging.info(f"Downloaded source image: {source_image_filename}")
        
        if not source_image_filename:
            self._fail_job("No source image provided for inpaint workflow.")
            return None
        
        return source_image_filename

    def _get_mask_image(self, inputs):
        """Materialize mask from base64."""
        mask_base64 = inputs.get("mask_image_base64")
        if not mask_base64:
            self._fail_job("No mask image provided for inpaint workflow.")
            return None

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
            return mask_filename
        except Exception as e:
            self._fail_job(f"Failed to process mask image: {str(e)}")
            return None

    def _copy_images_to_container(self, source_filename, mask_filename):
        """Copy source and mask images to Docker container."""
        try:
            from services.docker_manager import docker_manager
            from config import HEADLESS_MODE
            
            if not HEADLESS_MODE:
                # Copy source image
                docker_manager.copy_file_to_container(
                    service_type=self.client.active_service_type,
                    source_on_host=os.path.join(self.client.input_dir, source_filename),
                    dest_in_container=f"/opt/ComfyUI/input/{source_filename}",
                    shutdown_event=self.shutdown_event,
                )
                # Copy mask
                docker_manager.copy_file_to_container(
                    service_type=self.client.active_service_type,
                    source_on_host=os.path.join(self.client.input_dir, mask_filename),
                    dest_in_container=f"/opt/ComfyUI/input/{mask_filename}",
                    shutdown_event=self.shutdown_event,
                )
        except Exception as e:
            self._fail_job(f"Failed to copy images to container: {e}")

    def _inject_inpaint_workflow(self, workflow_data, prompt, image_filename, mask_filename, denoise_strength):
        """Inject prompt and images into the inpainting workflow."""
        wf = workflow_data.copy()
        
        image_node_count = 0
        for node_id, node in wf.items():
            if node.get("class_type") == "CLIPTextEncode":
                node["inputs"]["text"] = prompt
            elif node.get("class_type") == "LoadImage":
                # First LoadImage is the source, second is the mask
                if image_node_count == 0:
                    node["inputs"]["image"] = image_filename
                else:
                    node["inputs"]["image"] = mask_filename
                image_node_count += 1
            elif node.get("class_type") == "KSampler":
                node["inputs"]["denoise"] = denoise_strength
        
        return wf
