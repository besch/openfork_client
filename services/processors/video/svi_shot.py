"""
SVI 2.0 Shot Image-to-Video Processor

Generates continuous single-scene shots with infinite-length capability.
Uses WAN 2.2 base model + SVI LoRA with VACE-based padding for reference image.

Key characteristics:
- Uses 1 motion frame (last frame) for clip-to-clip coherence
- VACE-based padding with reference image to prevent drift
- Different seeds per clip for variety
- Designed for talking heads, consistent action, single-scene videos
"""

import os
import logging
import random

from config import DEV_MODE, SUPABASE_URL
from services.docker_manager import docker_manager
from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import VideoOutputHandler
from utils.comfyui_workflow_utils import (
    materialize_start_image,
)


class SVIShotImageToVideoJobProcessor(ComfyUIProcessor, VideoOutputHandler):
    """Processor for SVI 2.0 Shot image-to-video generation.
    
    SVI-Shot is designed for single continuous shots where the subject
    stays in frame. It uses VACE-based padding with the reference image
    to ensure no drift or forgetting issues over long generations.
    """

    def process(self):
        if DEV_MODE:
            return

        if not self.job:
            self._fail_job(f"Job object is None for SVIShotImageToVideoJobProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs", {})
        start_image_url = inputs.get("start_image_url")
        start_image_filename = None

        if start_image_url:
            logging.info(f"Downloading start image from signed URL: {start_image_url}")
            downloaded_path = self.orchestrator_service.download_asset_by_url(start_image_url, self.input_dir)
            if downloaded_path:
                start_image_filename = os.path.basename(downloaded_path)

        if not start_image_filename:
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
                supabase_url = os.environ.get('SUPABASE_URL', self.client.config.get('SUPABASE_URL', SUPABASE_URL))
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
        num_clips = inputs.get("num_clips", 1)  # Number of video clips to generate

        wf_ready = self._inject_svi_shot_params(
            workflow_data, 
            self.positive_prompt, 
            self.negative_prompt, 
            start_image_filename, 
            aspect_ratio,
            cfg_scale=cfg_scale, 
            steps=steps,
            flow_shift=flow_shift, 
            sampler=sampler, 
            scheduler=scheduler,
            num_clips=num_clips
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
        logging.info(f"SVI-Shot job {self.job_id} completed successfully")

    def _inject_svi_shot_params(
        self,
        workflow_api_data: dict,
        prompt: str,
        negative_prompt: str,
        start_image_filename: str,
        aspect_ratio: str = "16:9",
        cfg_scale: float = None,
        steps: int = None,
        flow_shift: float = None,
        sampler: str = None,
        scheduler: str = None,
        num_clips: int = 1
    ):
        """Inject SVI-Shot specific parameters into workflow.
        
        SVI-Shot uses:
        - 1 motion frame (last frame only)
        - VACE-based padding with reference image (ref_pad_num=-1)
        - Different seeds for each clip (critical for quality)
        """
        import copy
        from datetime import datetime
        
        api_graph = copy.deepcopy(workflow_api_data["prompt"])

        # Get dimensions based on aspect ratio
        width, height = self._get_dimensions(aspect_ratio)

        # Inject prompts and image filename
        for node in api_graph.values():
            class_type = node.get("class_type", "")
            
            if class_type == "CLIPTextEncode":
                # Check for positive/negative by title or position
                title = node.get("title", "").lower()
                if "positive" in title or "6" in str(node):
                    node["inputs"]["text"] = prompt
                elif "negative" in title or "7" in str(node):
                    node["inputs"]["text"] = negative_prompt
                    
            elif class_type == "LoadImage":
                node["inputs"]["image"] = start_image_filename
                
            elif class_type == "WanImageToVideo":
                node["inputs"]["width"] = width
                node["inputs"]["height"] = height
                
            elif class_type == "ImageResizeKJv2":
                node["inputs"]["width"] = width
                node["inputs"]["height"] = height
                
            elif class_type == "VHS_VideoCombine":
                prefix = node["inputs"].get("filename_prefix", "")
                if "%date:yyyy-MM-dd%" in prefix:
                    datestr = datetime.now().strftime("%Y-%m-%d")
                    node["inputs"]["filename_prefix"] = prefix.replace("%date:yyyy-MM-dd%", datestr)
                    
            elif "KSampler" in class_type and "inputs" in node:
                # Inject cfg and steps
                if cfg_scale is not None and "cfg" in node["inputs"]:
                    node["inputs"]["cfg"] = cfg_scale
                if steps is not None and "steps" in node["inputs"]:
                    node["inputs"]["steps"] = steps
                # Critical: Use different seed for SVI
                if "noise_seed" in node["inputs"] or "seed" in node["inputs"]:
                    seed_key = "noise_seed" if "noise_seed" in node["inputs"] else "seed"
                    node["inputs"][seed_key] = random.randint(0, 2**63 - 1)
                    
            elif class_type == "ModelSamplingSD3" and flow_shift is not None:
                node["inputs"]["shift"] = flow_shift
                
            elif class_type == "KSamplerSelect" and sampler is not None:
                node["inputs"]["sampler_name"] = sampler
                
            elif class_type == "BasicScheduler" and scheduler is not None:
                node["inputs"]["scheduler"] = scheduler
                
            elif class_type == "CFGGuider" and cfg_scale is not None:
                node["inputs"]["cfg"] = cfg_scale

        return api_graph

    def _get_dimensions(self, aspect_ratio: str) -> tuple:
        """Get dimensions for SVI generation (480p base)."""
        if aspect_ratio == "16:9":
            return 832, 480  # 480p widescreen
        elif aspect_ratio == "9:16":
            return 480, 832  # Portrait
        elif aspect_ratio == "1:1":
            return 512, 512  # Square
        elif aspect_ratio == "4:3":
            return 640, 480  # 4:3
        elif aspect_ratio == "3:4":
            return 480, 640  # Portrait 4:3
        elif aspect_ratio == "21:9":
            return 896, 384  # Ultrawide
        else:
            return 832, 480  # Default to 16:9
