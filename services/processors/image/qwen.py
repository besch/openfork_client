"""
Qwen-Image-Edit-2511 Processors

Processors for Qwen instruction-based image editing and inpainting.
"""

import logging
import os
import base64
import uuid
import copy
import random
from config import SUPABASE_URL

from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import ImageOutputHandler


def _clamp_qwen_steps(steps):
    try:
        requested_steps = int(steps) if steps is not None else None
    except (TypeError, ValueError):
        requested_steps = None
    if requested_steps is None:
        return None
    return max(1, min(requested_steps, 10))


def _clamp_qwen_2512_steps(steps):
    try:
        requested_steps = int(steps) if steps is not None else None
    except (TypeError, ValueError):
        requested_steps = None
    if requested_steps is None:
        return None
    return max(1, min(requested_steps, 60))


def _safe_model_filename(value):
    if not value:
        return None
    filename = os.path.basename(str(value).replace("\\", "/")).strip()
    if not filename or filename in (".", ".."):
        return None
    if not filename.lower().endswith(".safetensors"):
        return None
    return filename


def _host_path_for_download(input_dir, downloaded_path):
    if not downloaded_path:
        return None
    if os.path.isabs(downloaded_path):
        return downloaded_path
    return os.path.join(input_dir, downloaded_path)


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
        aspect_ratio = inputs.get("aspect_ratio", "1:1")
        steps = inputs.get("steps", inputs.get("num_steps"))

        # Get source image - either from base64 or storage path
        source_image_filename = self._get_source_image()
        if not source_image_filename:
            return
        second_image_requested = self._has_second_source_image_input()
        second_image_filename = self._get_second_source_image()
        if second_image_requested and not second_image_filename:
            return

        # Copy the image to the Docker container
        self._copy_image_to_container(source_image_filename)
        if second_image_filename:
            self._copy_image_to_container(second_image_filename)

        target_dimensions = self._get_dimensions(
            aspect_ratio,
            has_second_image=bool(second_image_filename),
            requested_long_edge=inputs.get("max_long_edge")
            or inputs.get("long_edge")
            or inputs.get("target_long_edge"),
        )

        # Inject prompt and image into workflow
        seed = inputs.get("seed")
        wf_ready = self._inject_edit_workflow(
            workflow_data,
            self.positive_prompt,
            source_image_filename,
            denoise_strength,
            second_image_filename=second_image_filename,
            aspect_ratio=aspect_ratio,
            seed=seed,
            steps=steps,
            requested_long_edge=inputs.get("max_long_edge")
            or inputs.get("long_edge")
            or inputs.get("target_long_edge"),
        )
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload, timeout_sec=1200)
        if not outputs:
            return

        image_storage_path = self.handle_image_output(
            outputs,
            target_dimensions=target_dimensions,
        )
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
        
        # If no base64, try downloading from signed URL or constructed URL
        if not source_image_filename:
            inputs = self.job.get("inputs", {})
            start_image_url = inputs.get("start_image_url")
            
            if start_image_url:
                logging.info(f"Downloading source image from signed URL: {start_image_url}")
                downloaded_path = self.orchestrator_service.download_asset_by_url(start_image_url, self.client.input_dir)
                if downloaded_path:
                    source_image_filename = os.path.basename(downloaded_path)

            if not source_image_filename:
                input_storage_path = self.job.get("input_storage_path")
                if input_storage_path:
                    bucket = self.job.get("bucket", "projects_public")
                    downloaded_path = self.orchestrator_service.download_storage_asset(
                        bucket,
                        input_storage_path,
                        self.client.input_dir,
                    )
                    if not downloaded_path:
                        source_url = f"{os.environ.get('SUPABASE_URL', self.client.config.get('SUPABASE_URL', SUPABASE_URL))}/storage/v1/object/public/{bucket}/{input_storage_path}"
                        logging.info(f"Downloading source image from: {source_url}")
                        downloaded_path = self.orchestrator_service.download_asset_by_url(source_url, self.client.input_dir)
                    if downloaded_path:
                        source_image_filename = os.path.basename(downloaded_path)
                        logging.info(f"Downloaded source image: {source_image_filename}")
        
        if not source_image_filename:
            self._fail_job("No source image provided for Qwen edit workflow.")
            return None
        
        return source_image_filename

    def _has_second_source_image_input(self):
        inputs = self.job.get("inputs", {})
        return any(
            inputs.get(key)
            for key in (
                "reference_image_2_base64",
                "reference_image2_base64",
                "second_reference_image_base64",
                "reference_image_2_url",
                "reference_image2_url",
                "second_reference_image_url",
                "reference_image_2_storage_path",
                "reference_image2_storage_path",
                "second_reference_image_storage_path",
                "reference_image_2",
                "reference_image2",
            )
        )

    def _materialize_base64_image(self, data_url, prefix):
        """Save a base64 image input into the ComfyUI input directory."""
        if not data_url:
            return None

        try:
            image_base64 = data_url
            extension = "png"
            if "," in image_base64:
                header, image_base64 = image_base64.split(",", 1)
                if "image/jpeg" in header or "image/jpg" in header:
                    extension = "jpg"
                elif "image/webp" in header:
                    extension = "webp"

            image_data = base64.b64decode(image_base64)
            filename = f"{prefix}_{uuid.uuid4().hex[:8]}.{extension}"
            image_path = os.path.join(self.client.input_dir, filename)
            with open(image_path, "wb") as f:
                f.write(image_data)
            logging.info(f"Saved {prefix} image to: {image_path}")
            return filename
        except Exception as e:
            self._fail_job(f"Failed to process {prefix} image: {str(e)}")
            return None

    def _get_second_source_image(self):
        """Get optional second edit image from base64, signed URL, or storage."""
        inputs = self.job.get("inputs", {})

        second_image_filename = self._materialize_base64_image(
            inputs.get("reference_image_2_base64")
            or inputs.get("reference_image2_base64")
            or inputs.get("second_reference_image_base64"),
            "reference_image_2",
        )
        if second_image_filename:
            return second_image_filename

        second_image_url = (
            inputs.get("reference_image_2_url")
            or inputs.get("reference_image2_url")
            or inputs.get("second_reference_image_url")
        )
        if second_image_url:
            logging.info(f"Downloading second edit image from signed URL: {second_image_url}")
            downloaded_path = self.orchestrator_service.download_asset_by_url(
                second_image_url,
                self.client.input_dir,
            )
            if downloaded_path:
                return os.path.basename(downloaded_path)

        second_storage_path = (
            inputs.get("reference_image_2_storage_path")
            or inputs.get("reference_image2_storage_path")
            or inputs.get("second_reference_image_storage_path")
            or inputs.get("reference_image_2")
            or inputs.get("reference_image2")
        )
        if second_storage_path:
            bucket = self.job.get("bucket", "projects_public")
            downloaded_path = self.orchestrator_service.download_storage_asset(
                bucket,
                second_storage_path,
                self.client.input_dir,
            )
            if not downloaded_path:
                source_url = f"{os.environ.get('SUPABASE_URL', self.client.config.get('SUPABASE_URL', SUPABASE_URL))}/storage/v1/object/public/{bucket}/{second_storage_path}"
                logging.info(f"Downloading second edit image from: {source_url}")
                downloaded_path = self.orchestrator_service.download_asset_by_url(
                    source_url,
                    self.client.input_dir,
                )
            if downloaded_path:
                second_image_filename = os.path.basename(downloaded_path)
                logging.info(f"Downloaded second edit image: {second_image_filename}")
                return second_image_filename

        if self._has_second_source_image_input():
            self._fail_job("Failed to load second edit image for Qwen edit workflow.")
        return None

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
            else:
                # Headless Mode: Copy locally
                import shutil
                container_input_path = f"/opt/ComfyUI/input/{image_filename}"
                os.makedirs(os.path.dirname(container_input_path), exist_ok=True)
                shutil.copy2(image_full_path, container_input_path)
                logging.info(f"Copied {image_filename} to {container_input_path} (Headless)")
        except Exception as e:
            self._fail_job(f"Failed to copy image to container: {e}")

    def _find_scaled_image_ref(self, wf, image_node_id):
        """Return the scaled image node attached to a LoadImage node when present."""
        source_ref = [image_node_id, 0]
        for node_id, node in wf.items():
            if node.get("class_type") not in (
                "FluxKontextImageScale",
                "ImageScaleToMaxDimension",
            ):
                continue
            if node.get("inputs", {}).get("image") == source_ref:
                return [node_id, 0]
        return source_ref

    def _inject_edit_workflow(self, workflow_data, prompt, image_filename, denoise_strength, second_image_filename=None, aspect_ratio="1:1", seed=None, steps=None, requested_long_edge=None):
        """Inject prompt and image into the editing workflow."""
        wf = copy.deepcopy(workflow_data.get("prompt", workflow_data))
        
        # Calculate dimensions
        width, height = self._get_dimensions(
            aspect_ratio,
            has_second_image=bool(second_image_filename),
            requested_long_edge=requested_long_edge,
        )
        max_dim = max(width, height)
        uses_empty_latent = any(
            node.get("class_type") in ("EmptySD3LatentImage", "EmptyQwenImageLayeredLatentImage")
            for node in wf.values()
        )
        requested_steps = _clamp_qwen_steps(steps)
        load_image_ids = [
            node_id
            for node_id, node in wf.items()
            if node.get("class_type") == "LoadImage"
        ]
        source_image_ref = (
            self._find_scaled_image_ref(wf, load_image_ids[0])
            if load_image_ids
            else None
        )
        second_image_ref = None
        if second_image_filename and len(load_image_ids) > 1:
            second_image_ref = self._find_scaled_image_ref(wf, load_image_ids[1])
        if not second_image_filename:
            for unused_node_id in load_image_ids[1:]:
                wf.pop(unused_node_id, None)

        # Update prompt in CLIPTextEncode node
        image_node_count = 0
        for node_id, node in wf.items():
            class_type = node.get("class_type")
            inputs = node.get("inputs", {})

            if class_type == "CLIPTextEncode":
                node["inputs"]["text"] = prompt
            elif class_type == "TextEncodeQwenImageEditPlus":
                has_prompt = bool(str(inputs.get("prompt") or "").strip())
                if has_prompt:
                    inputs["prompt"] = prompt
                    template_has_image2 = "image2" in inputs
                    if second_image_ref and not template_has_image2:
                        inputs["image1"] = second_image_ref
                    elif source_image_ref:
                        inputs["image1"] = source_image_ref
                    if second_image_ref and template_has_image2:
                        inputs["image2"] = second_image_ref
                    else:
                        inputs.pop("image2", None)
                else:
                    inputs.pop("vae", None)
                    inputs.pop("image1", None)
                    inputs.pop("image2", None)
                inputs.pop("image3", None)
            elif class_type == "LoadImage":
                if image_node_count == 0:
                    node["inputs"]["image"] = image_filename
                elif second_image_filename:
                    node["inputs"]["image"] = second_image_filename
                image_node_count += 1
            elif class_type == "ImageScaleToMaxDimension":
                node["inputs"]["largest_size"] = max_dim
            elif class_type in ("EmptySD3LatentImage", "EmptyQwenImageLayeredLatentImage"):
                node["inputs"]["width"] = width
                node["inputs"]["height"] = height
            elif class_type == "KSampler":
                node["inputs"]["denoise"] = 1.0 if uses_empty_latent else denoise_strength
                if seed is not None:
                    node["inputs"]["seed"] = seed
                else:
                    node["inputs"]["seed"] = random.randint(0, 2**63 - 1)
                if requested_steps is not None:
                    node["inputs"]["steps"] = requested_steps
        
        return wf

    def _get_dimensions(self, aspect_ratio, has_second_image=False, requested_long_edge=None):
        """Calculate edit dimensions for 8GB GPUs.

        Qwen edit can technically run larger frames, but on 8GB cards
        two-reference edits need a smaller latent to avoid CUDA fallback stalls.
        Keep the long edge conservative and let the video model handle motion.
        """
        default_long_edge = 384 if has_second_image else 512
        try:
            long_edge = int(requested_long_edge or default_long_edge)
        except (TypeError, ValueError):
            long_edge = default_long_edge

        if has_second_image:
            long_edge = max(256, min(long_edge, 512))
        else:
            long_edge = max(320, min(long_edge, 768))
        ratio_map = {
            "1:1": (long_edge, long_edge),
            "16:9": (long_edge, round(long_edge * 9 / 16)),
            "9:16": (round(long_edge * 9 / 16), long_edge),
            "4:3": (long_edge, round(long_edge * 3 / 4)),
            "3:4": (round(long_edge * 3 / 4), long_edge),
            "3:2": (long_edge, round(long_edge * 2 / 3)),
            "2:3": (round(long_edge * 2 / 3), long_edge),
            "21:9": (long_edge, round(long_edge * 9 / 21)),
        }
        return ratio_map.get(aspect_ratio, (long_edge, long_edge))


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
        aspect_ratio = inputs.get("aspect_ratio", "1:1")
        steps = inputs.get("steps", inputs.get("num_steps"))
        requested_long_edge = (
            inputs.get("max_long_edge")
            or inputs.get("long_edge")
            or inputs.get("target_long_edge")
        )
        target_dimensions = self._get_dimensions(
            aspect_ratio,
            requested_long_edge=requested_long_edge,
        )

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
        seed = inputs.get("seed")
        wf_ready = self._inject_inpaint_workflow(
            workflow_data,
            self.positive_prompt,
            source_image_filename,
            mask_filename,
            denoise_strength,
            aspect_ratio=aspect_ratio,
            seed=seed,
            steps=steps,
            requested_long_edge=requested_long_edge,
        )
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        image_storage_path = self.handle_image_output(
            outputs,
            target_dimensions=target_dimensions,
        )
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
            inputs = self.job.get("inputs", {})
            start_image_url = inputs.get("start_image_url")
            
            if start_image_url:
                logging.info(f"Downloading source image from signed URL: {start_image_url}")
                downloaded_path = self.orchestrator_service.download_asset_by_url(start_image_url, self.client.input_dir)
                if downloaded_path:
                    source_image_filename = os.path.basename(downloaded_path)

            if not source_image_filename:
                input_storage_path = self.job.get("input_storage_path")
                if input_storage_path:
                    bucket = self.job.get("bucket", "projects_public")
                    downloaded_path = self.orchestrator_service.download_storage_asset(
                        bucket,
                        input_storage_path,
                        self.client.input_dir,
                    )
                    if not downloaded_path:
                        source_url = f"{os.environ.get('SUPABASE_URL', self.client.config.get('SUPABASE_URL', SUPABASE_URL))}/storage/v1/object/public/{bucket}/{input_storage_path}"
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
            else:
                # Headless Mode: Copy locally
                import shutil
                # Copy source image
                dest_src = f"/opt/ComfyUI/input/{source_filename}"
                os.makedirs(os.path.dirname(dest_src), exist_ok=True)
                shutil.copy2(os.path.join(self.client.input_dir, source_filename), dest_src)
                
                # Copy mask
                dest_mask = f"/opt/ComfyUI/input/{mask_filename}"
                shutil.copy2(os.path.join(self.client.input_dir, mask_filename), dest_mask)
                logging.info(f"Copied images to ComfyUI input (Headless)")
        except Exception as e:
            self._fail_job(f"Failed to copy images to container: {e}")

    def _inject_inpaint_workflow(self, workflow_data, prompt, image_filename, mask_filename, denoise_strength, aspect_ratio="1:1", seed=None, steps=None, requested_long_edge=None):
        """Inject prompt and images into the inpainting workflow."""
        wf = copy.deepcopy(workflow_data.get("prompt", workflow_data))
        
        # Calculate dimensions
        width, height = self._get_dimensions(
            aspect_ratio,
            requested_long_edge=requested_long_edge,
        )
        max_dim = max(width, height)
        requested_steps = _clamp_qwen_steps(steps)

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
            elif node.get("class_type") == "ImageScaleToMaxDimension":
                node["inputs"]["largest_size"] = max_dim
            elif node.get("class_type") == "KSampler":
                node["inputs"]["denoise"] = denoise_strength
                if seed is not None:
                    node["inputs"]["seed"] = seed
                else:
                    node["inputs"]["seed"] = random.randint(0, 2**63 - 1)
                if requested_steps is not None:
                    node["inputs"]["steps"] = requested_steps
        
        return wf

    def _get_dimensions(self, aspect_ratio, requested_long_edge=None):
        """Calculate width and height based on aspect ratio."""
        default_long_edge = 512
        try:
            long_edge = int(requested_long_edge or default_long_edge)
        except (TypeError, ValueError):
            long_edge = default_long_edge

        long_edge = max(320, min(long_edge, 768))
        ratio_map = {
            "1:1": (long_edge, long_edge),
            "16:9": (long_edge, round(long_edge * 9 / 16)),
            "9:16": (round(long_edge * 9 / 16), long_edge),
            "4:3": (long_edge, round(long_edge * 3 / 4)),
            "3:4": (round(long_edge * 3 / 4), long_edge),
            "3:2": (long_edge, round(long_edge * 2 / 3)),
            "2:3": (round(long_edge * 2 / 3), long_edge),
            "21:9": (long_edge, round(long_edge * 9 / 21)),
        }
        return ratio_map.get(aspect_ratio, (long_edge, long_edge))


class QwenImageT2IProcessor(ComfyUIProcessor, ImageOutputHandler):
    """Processor for Qwen text-to-image generation."""

    def process(self):
        if not self.job:
            self._fail_job("Job object is None for QwenImageT2IProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        aspect_ratio = inputs.get("aspect_ratio", "1:1")
        
        # Calculate dimensions based on aspect ratio
        width, height = self._get_dimensions(aspect_ratio)
        steps = inputs.get("steps", inputs.get("num_steps"))

        # Inject prompt and dimensions into workflow
        seed = inputs.get("seed")
        wf_ready = self._inject_t2i_workflow(
            workflow_data,
            self.positive_prompt,
            width,
            height,
            seed=seed,
            steps=steps,
        )
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        image_storage_path = self.handle_image_output(
            outputs,
            target_dimensions=(width, height),
        )
        if not image_storage_path:
            return

        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=image_storage_path,
            thumbnail_storage_path=image_storage_path,
            prompt=self.positive_prompt,
        )

    def _get_dimensions(self, aspect_ratio):
        """Calculate width and height based on aspect ratio."""
        ratio_map = {
            "1:1": (512, 512),
            "16:9": (512, 288),
            "9:16": (288, 512),
            "4:3": (512, 384),
            "3:4": (384, 512),
            "3:2": (512, 336),
            "2:3": (336, 512),
            "21:9": (512, 224),
        }
        return ratio_map.get(aspect_ratio, (512, 512))

    def _inject_t2i_workflow(self, workflow_data, prompt, width, height, seed=None, steps=None):
        """Inject prompt and dimensions into the text-to-image workflow."""
        wf = copy.deepcopy(workflow_data.get("prompt", workflow_data))
        requested_steps = _clamp_qwen_steps(steps)
        
        for node_id, node in wf.items():
            if node.get("class_type") == "CLIPTextEncode":
                # Only update positive prompt (not negative/empty one)
                if node["inputs"].get("text") not in ["", None]:
                    node["inputs"]["text"] = prompt
            elif node.get("class_type") == "EmptySD3LatentImage":
                node["inputs"]["width"] = width
                node["inputs"]["height"] = height
            elif node.get("class_type") == "KSampler":
                if seed is not None:
                    node["inputs"]["seed"] = seed
                else:
                    node["inputs"]["seed"] = random.randint(0, 2**63 - 1)
                if requested_steps is not None:
                    node["inputs"]["steps"] = requested_steps
        
        return wf


class QwenImage2512LoraT2IProcessor(QwenImageT2IProcessor):
    """Processor for Qwen-Image-2512 text-to-image with a trained character LoRA."""

    def process(self):
        if not self.job:
            self._fail_job("Job object is None for QwenImage2512LoraT2IProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        aspect_ratio = inputs.get("aspect_ratio", "1:1")
        requested_long_edge = (
            inputs.get("max_long_edge")
            or inputs.get("long_edge")
            or inputs.get("target_long_edge")
        )
        width, height = self._get_dimensions(
            aspect_ratio,
            requested_long_edge=requested_long_edge,
        )

        lora_filename = self._get_lora_filename(inputs)
        if not lora_filename:
            return

        seed = inputs.get("seed")
        steps = inputs.get("steps", inputs.get("num_steps"))
        lora_strength = self._get_lora_strength(inputs)
        wf_ready = self._inject_lora_t2i_workflow(
            workflow_data,
            self.positive_prompt,
            width,
            height,
            lora_filename,
            lora_strength,
            seed=seed,
            steps=steps,
        )
        if not wf_ready:
            return

        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload, timeout_sec=1800)
        if not outputs:
            return

        image_storage_path = self.handle_image_output(
            outputs,
            target_dimensions=(width, height),
        )
        if not image_storage_path:
            return

        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=image_storage_path,
            thumbnail_storage_path=image_storage_path,
            prompt=self.positive_prompt,
            completion_metadata={
                "lora_name": lora_filename,
                "lora_strength": lora_strength,
                "base_model": self._get_base_model_name(wf_ready),
            },
        )

    def _get_dimensions(self, aspect_ratio, requested_long_edge=None):
        """Use the official Qwen-Image-2512 aspect buckets, with an optional cap."""
        ratio_map = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1104),
            "3:4": (1104, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        }
        width, height = ratio_map.get(aspect_ratio, (1328, 1328))

        if requested_long_edge is None:
            return width, height

        try:
            long_edge = int(requested_long_edge)
        except (TypeError, ValueError):
            return width, height

        long_edge = max(512, min(long_edge, 1664))
        scale = long_edge / max(width, height)
        if scale >= 1:
            return width, height

        width = max(16, round((width * scale) / 16) * 16)
        height = max(16, round((height * scale) / 16) * 16)
        return width, height

    def _get_lora_strength(self, inputs):
        value = 0.8
        for key in ("lora_strength", "loraStrength", "lora_scale", "loraScale"):
            if inputs.get(key) is not None:
                value = inputs.get(key)
                break
        try:
            return max(0.0, min(float(value), 1.5))
        except (TypeError, ValueError):
            return 0.8

    def _get_base_model_name(self, workflow):
        for node in workflow.values():
            class_type = node.get("class_type")
            if class_type not in (
                "UNETLoader",
                "UnetLoaderGGUF",
                "UnetLoaderGGUFAdvanced",
                "LoadDiffusionModel",
            ):
                continue

            inputs = node.get("inputs", {})
            for key in ("unet_name", "model_name"):
                value = inputs.get(key)
                if value:
                    return value

        return "qwen_image_2512_fp8_e4m3fn.safetensors"

    def _get_lora_filename(self, inputs):
        local_lora_path = self._download_lora(inputs)
        if local_lora_path:
            filename = _safe_model_filename(local_lora_path)
            if not filename:
                self._fail_job("Qwen 2512 LoRA file must use a .safetensors filename.")
                return None
            if not os.path.exists(local_lora_path):
                self._fail_job(f"Downloaded LoRA file not found on host: {local_lora_path}")
                return None
            if not self._copy_lora_to_container(local_lora_path, filename):
                return None
            return filename

        lora_name = (
            inputs.get("lora_name")
            or inputs.get("loraName")
            or inputs.get("character_lora_name")
            or inputs.get("characterLoraName")
        )
        filename = _safe_model_filename(lora_name)
        if filename:
            return filename

        self._fail_job(
            "No character LoRA provided. Pass lora_storage_path, lora_url, or preinstalled lora_name."
        )
        return None

    def _download_lora(self, inputs):
        storage_path = (
            inputs.get("lora_storage_path")
            or inputs.get("loraStoragePath")
            or inputs.get("character_lora_storage_path")
            or inputs.get("characterLoraStoragePath")
            or inputs.get("adapter_storage_path")
            or inputs.get("adapterStoragePath")
        )
        if storage_path:
            bucket = (
                inputs.get("lora_bucket")
                or inputs.get("loraBucket")
                or self.job.get("bucket")
                or "projects_public"
            )
            downloaded_path = self.orchestrator_service.download_storage_asset(
                bucket,
                storage_path,
                self.client.input_dir,
            )
            if not downloaded_path:
                source_url = f"{os.environ.get('SUPABASE_URL', self.client.config.get('SUPABASE_URL', SUPABASE_URL))}/storage/v1/object/public/{bucket}/{storage_path}"
                logging.info("Downloading Qwen 2512 LoRA from storage URL.")
                downloaded_path = self.orchestrator_service.download_asset_by_url(
                    source_url,
                    self.client.input_dir,
                )
            return _host_path_for_download(self.client.input_dir, downloaded_path)

        lora_url = (
            inputs.get("lora_url")
            or inputs.get("loraUrl")
            or inputs.get("character_lora_url")
            or inputs.get("characterLoraUrl")
            or inputs.get("adapter_url")
            or inputs.get("adapterUrl")
        )
        if lora_url:
            logging.info("Downloading Qwen 2512 LoRA from signed URL.")
            downloaded_path = self.orchestrator_service.download_asset_by_url(
                lora_url,
                self.client.input_dir,
            )
            return _host_path_for_download(self.client.input_dir, downloaded_path)

        return None

    def _copy_lora_to_container(self, local_lora_path, lora_filename):
        try:
            from services.docker_manager import docker_manager
            from config import HEADLESS_MODE

            container_lora_path = f"/opt/ComfyUI/models/loras/{lora_filename}"
            if not HEADLESS_MODE:
                docker_manager.copy_file_to_container(
                    service_type=self.client.active_service_type,
                    source_on_host=local_lora_path,
                    dest_in_container=container_lora_path,
                    shutdown_event=self.shutdown_event,
                )
            else:
                import shutil

                os.makedirs(os.path.dirname(container_lora_path), exist_ok=True)
                shutil.copy2(local_lora_path, container_lora_path)
                logging.info(f"Copied {lora_filename} to {container_lora_path} (Headless)")
            return True
        except Exception as e:
            self._fail_job(f"Failed to copy LoRA to container: {e}")
            return False

    def _inject_lora_t2i_workflow(
        self,
        workflow_data,
        prompt,
        width,
        height,
        lora_filename,
        lora_strength,
        seed=None,
        steps=None,
    ):
        wf = copy.deepcopy(workflow_data.get("prompt", workflow_data))
        requested_steps = _clamp_qwen_2512_steps(steps)
        found_lora_node = False

        for node in wf.values():
            class_type = node.get("class_type")
            inputs = node.get("inputs", {})
            if class_type == "CLIPTextEncode":
                if inputs.get("text") not in ["", None]:
                    inputs["text"] = prompt
            elif class_type == "EmptySD3LatentImage":
                inputs["width"] = width
                inputs["height"] = height
            elif class_type == "LoraLoaderModelOnly":
                inputs["lora_name"] = lora_filename
                inputs["strength_model"] = lora_strength
                found_lora_node = True
            elif class_type == "KSampler":
                if seed is not None:
                    inputs["seed"] = seed
                else:
                    inputs["seed"] = random.randint(0, 2**63 - 1)
                if requested_steps is not None:
                    inputs["steps"] = requested_steps

        if not found_lora_node:
            self._fail_job("Qwen 2512 LoRA workflow is missing a LoraLoaderModelOnly node.")
            return None

        return wf

