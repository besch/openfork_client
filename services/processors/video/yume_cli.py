"""
YUME CLI Job Processor

This processor communicates with the official YUME webapp (webapp_single_gpu.py)
which provides a complete video generation pipeline including diffusion sampling
and VAE decoding.

Reference: https://github.com/stdstu12/YUME
"""

import os
import json
import time
import logging
import requests
from typing import Optional, Dict
from pathlib import Path

from services.processors.base import BaseJobProcessor
from services.orchestrator_service import TokenExpiredError
from utils.media_utils import get_video_duration, generate_thumbnail
from utils.comfyui_workflow_utils import get_dimensions, materialize_start_image


class YumeCLIJobProcessor(BaseJobProcessor):
    """
    Job processor that communicates with YUME's official webapp.
    The webapp handles the full generation pipeline including model loading,
    diffusion sampling, and VAE decoding.
    """

    API_PORT = 8000
    POLL_INTERVAL = 10  # Video generation takes longer than audio
    MAX_WAIT_TIME = 1800  # 30 minutes

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.api_base_url = f"http://localhost:{self.API_PORT}"
        # For I2V, we need access to the materializer
        self.input_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "temp_input"
        )

    def process(self):
        """Main processing method."""
        if not self.job:
            self._fail_job("Job object is None for YumeCLIJobProcessor. Cannot proceed.")
            return

        workflow_type = self.job.get("workflow_type", "yume-text-to-video")
        is_i2v = "image" in workflow_type.lower()
        
        logging.info(f"Processing YUME CLI job {self.job_id} (I2V: {is_i2v})")

        # Parse inputs
        inputs = self.job.get("inputs") or {}
        aspect_ratio = inputs.get("aspect_ratio", "16:9")
        width, height = get_dimensions(aspect_ratio, default_width=1280, default_height=720)
        
        cfg_scale = inputs.get("cfg_scale", 5.0)  # YUME default is 5.0
        steps = inputs.get("steps", 50)  # YUME default is 50
        num_frames = inputs.get("num_frames", 81)  # YUME default is 81
        seed = inputs.get("seed", 0)
        shift = inputs.get("shift", 5.0)  # YUME default shift

        if not self._wait_for_api():
            self._fail_job(f"YUME API did not become available for job {self.job_id}")
            return

        try:
            # Generate video synchronously via webapp
            if is_i2v:
                result = self._generate_i2v(
                    width, height, num_frames, steps, cfg_scale, seed, shift
                )
            else:
                result = self._generate_t2v(
                    width, height, num_frames, steps, cfg_scale, seed, shift
                )
            
            if not result.get("success"):
                error_msg = result.get("error", "Unknown error")
                self._fail_job(f"YUME generation failed: {error_msg}")
                return

            # Download the output video
            video_path = result.get("video_abs")
            if not video_path or not os.path.exists(video_path):
                # Try relative path via outputs endpoint
                video_rel = result.get("video_rel")
                if video_rel:
                    local_path = self._download_output(video_rel)
                else:
                    self._fail_job(f"No output path in result for job {self.job_id}")
                    return
            else:
                local_path = video_path

            if not local_path or not os.path.exists(local_path):
                self._fail_job(f"Failed to get output for job {self.job_id}")
                return

            # Upload video
            video_storage_path = self.orchestrator_service.upload_video_output(local_path, self.job_id)

            if video_storage_path:
                # Generate and upload thumbnail
                thumbnail_path = local_path.replace(".mp4", "_thumb.jpg")
                generate_thumbnail(local_path, thumbnail_path)
                
                thumbnail_storage_path = None
                if os.path.exists(thumbnail_path):
                    thumbnail_storage_path = self.orchestrator_service.upload_thumbnail_output(
                        thumbnail_path, self.job_id
                    )
                    os.remove(thumbnail_path)
                
                duration = get_video_duration(local_path)
                
                self.orchestrator_service.update_job_status(
                    self.job_id,
                    "completed",
                    storage_path=video_storage_path,
                    thumbnail_storage_path=thumbnail_storage_path,
                    duration_seconds=duration,
                    prompt=self.positive_prompt,
                )
                logging.info(f"YUME job {self.job_id} completed successfully")
            else:
                self._fail_job(f"YUME job {self.job_id} completed, but upload failed")

        except TokenExpiredError:
            raise
        except Exception as e:
            logging.error(f"Error processing YUME job {self.job_id}: {e}", exc_info=True)
            self._fail_job(f"Error processing job: {e}")

    def _wait_for_api(self, timeout: int = 300) -> bool:
        """Wait for the YUME webapp to become available and model loaded."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.shutdown_event.is_set():
                return False
            try:
                response = requests.get(f"{self.api_base_url}/api/status", timeout=5)
                if response.status_code == 200:
                    status = response.json()
                    models_loaded = status.get("models_loaded", False)
                    if models_loaded:
                        logging.info("YUME webapp is ready and models are loaded")
                        return True
                    else:
                        logging.info("YUME webapp available but models still loading...")
            except requests.exceptions.RequestException:
                pass
            time.sleep(3)

        logging.error(f"YUME webapp did not become available within {timeout}s")
        return False

    def _generate_t2v(
        self, width: int, height: int, num_frames: int, 
        steps: int, cfg: float, seed: int, shift: float
    ) -> Dict:
        """Generate text-to-video using the webapp's /api/generate_long endpoint."""
        try:
            # Calculate resolution for webapp (uses single number for max dimension)
            resolution = max(width, height)
            
            payload = {
                "prompt": self.positive_prompt,
                "negative_prompt": self.negative_prompt or "",
                "mode": "T2V",
                "sample_num": num_frames,
                "sample_steps": steps,
                "guide_scale": cfg,
                "shift": shift,
                "resolution": resolution,
                "seed": seed if seed != 0 else None,
                "frame_zero": 8,  # Default latent conditioning frames
                "memory_optimization": True,
                "vae_memory_optimization": True,
            }

            logging.info(f"Submitting T2V generation to webapp: prompt={self.positive_prompt[:50]}...")
            response = requests.post(
                f"{self.api_base_url}/api/generate_long",
                json=payload,
                timeout=self.MAX_WAIT_TIME
            )

            if response.status_code == 200:
                data = response.json()
                logging.info(f"YUME webapp generation completed: {data.get('info', '')}")
                return data
            else:
                error_text = response.text[:500] if response.text else "No error message"
                logging.error(f"YUME webapp returned {response.status_code}: {error_text}")
                return {"success": False, "error": f"HTTP {response.status_code}: {error_text}"}

        except requests.exceptions.Timeout:
            return {"success": False, "error": "Generation timed out"}
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to call webapp generate: {e}")
            return {"success": False, "error": str(e)}

    def _generate_i2v(
        self, width: int, height: int, num_frames: int,
        steps: int, cfg: float, seed: int, shift: float
    ) -> Dict:
        """Generate image-to-video using the webapp's /api/generate_long endpoint."""
        try:
            # Materialize start image
            os.makedirs(self.input_dir, exist_ok=True)
            start_image_filename = materialize_start_image(self.job, self.input_dir)
            
            if not start_image_filename:
                logging.error(f"Failed to materialize start image for job {self.job_id}")
                return {"success": False, "error": "Failed to get start image"}
            
            image_path = os.path.join(self.input_dir, start_image_filename)
            
            # Calculate resolution for webapp (uses single number for max dimension)
            resolution = max(width, height)
            
            payload = {
                "prompt": self.positive_prompt,
                "negative_prompt": self.negative_prompt or "",
                "mode": "I2V",
                "jpg_path": image_path,  # Local path to image
                "sample_num": num_frames,
                "sample_steps": steps,
                "guide_scale": cfg,
                "shift": shift,
                "resolution": resolution,
                "seed": seed if seed != 0 else None,
                "frame_zero": 8,
                "memory_optimization": True,
                "vae_memory_optimization": True,
            }

            logging.info(f"Submitting I2V generation to webapp with image: {image_path}")
            response = requests.post(
                f"{self.api_base_url}/api/generate_long",
                json=payload,
                timeout=self.MAX_WAIT_TIME
            )

            # Cleanup local image
            if os.path.exists(image_path):
                os.remove(image_path)

            if response.status_code == 200:
                data = response.json()
                logging.info(f"YUME webapp I2V generation completed: {data.get('info', '')}")
                return data
            else:
                error_text = response.text[:500] if response.text else "No error message"
                logging.error(f"YUME webapp returned {response.status_code}: {error_text}")
                return {"success": False, "error": f"HTTP {response.status_code}: {error_text}"}

        except requests.exceptions.Timeout:
            return {"success": False, "error": "I2V generation timed out"}
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to call webapp generate for I2V: {e}")
            return {"success": False, "error": str(e)}

    def _download_output(self, video_rel: str) -> Optional[str]:
        """Download the generated video file from the webapp's outputs endpoint."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            local_path = os.path.join(self.cache_dir, f"{self.job_id}.mp4")

            # The webapp serves files at /outputs/<path>
            response = requests.get(
                f"{self.api_base_url}/outputs/{video_rel}",
                timeout=120,
                stream=True
            )

            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logging.info(f"Downloaded output to {local_path}")
                return local_path
            else:
                logging.error(f"Failed to download output: {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download output: {e}")
            return None


# Create aliases for both T2V and I2V - same processor handles both
class YumeTextToVideoCLIJobProcessor(YumeCLIJobProcessor):
    """Alias for YUME Text-to-Video via CLI."""
    pass


class YumeImageToVideoCLIJobProcessor(YumeCLIJobProcessor):
    """Alias for YUME Image-to-Video via CLI."""
    pass
