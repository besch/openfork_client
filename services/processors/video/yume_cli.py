"""
YUME CLI Job Processor

This processor communicates with the custom YUME REST API (yume_api.py)
which provides a memory-efficient generation pipeline.
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
    Job processor that communicates with the local YUME REST API.
    """

    API_PORT = 8000
    POLL_INTERVAL = 5
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
        
        cfg_scale = inputs.get("cfg_scale", 7.0)
        steps = inputs.get("steps", 30)
        num_frames = inputs.get("num_frames", 49)
        seed = inputs.get("seed", 0)
        
        # Yume API specific params
        shift = inputs.get("shift", 7.0)

        if not self._wait_for_api():
            self._fail_job(f"YUME API did not become available for job {self.job_id}")
            return

        try:
            # Submit job to API
            if is_i2v:
                submission = self._generate_i2v(
                    width, height, num_frames, steps, cfg_scale, seed
                )
            else:
                submission = self._generate_t2v(
                    width, height, num_frames, steps, cfg_scale, seed
                )
            
            if not submission.get("success"):
                error_msg = submission.get("error", "Unknown error")
                self._fail_job(f"YUME submission failed: {error_msg}")
                return

            api_job_id = submission.get("job_id")
            logging.info(f"YUME API accepted job as {api_job_id}. Polling for completion...")

            # Poll for completion
            result = self._poll_job(api_job_id)
            if not result.get("success"):
                self._fail_job(f"YUME generation failed: {result.get('error')}")
                return

            # Download the output video
            local_path = self._download_output(api_job_id)
            
            if not local_path or not os.path.exists(local_path):
                self._fail_job(f"Failed to get output for job {self.job_id}")
                return

            # Clear job from API to free space
            self._cleanup_api_job(api_job_id)

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
        """Wait for the YUME API to become available and model loaded."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.shutdown_event.is_set():
                return False
            try:
                response = requests.get(f"{self.api_base_url}/health", timeout=5)
                if response.status_code == 200:
                    status = response.json()
                    model_loaded = status.get("model_loaded", False)
                    if model_loaded:
                        logging.info("YUME API is ready and model is loaded")
                        return True
                    else:
                        logging.info("YUME API available but model loading...")
                elif response.status_code == 503:
                    logging.info("YUME API loading model...")
            except requests.exceptions.RequestException:
                pass
            time.sleep(3)

        logging.error(f"YUME API did not become available within {timeout}s")
        return False

    def _generate_t2v(
        self, width: int, height: int, num_frames: int, 
        steps: int, cfg: float, seed: int
    ) -> Dict:
        """Submit text-to-video generation job."""
        try:
            payload = {
                "prompt": self.positive_prompt,
                "negative_prompt": self.negative_prompt or "",
                "num_frames": num_frames,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg,
                "seed": seed if seed != 0 else 0
            }

            logging.info(f"Submitting T2V job: {self.positive_prompt[:50]}...")
            response = requests.post(
                f"{self.api_base_url}/generate",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                return {"success": True, "job_id": data.get("job_id")}
            else:
                logging.error(f"YUME API returned {response.status_code}: {response.text}")
                return {"success": False, "error": f"HTTP {response.status_code}"}

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to submit T2V job: {e}")
            return {"success": False, "error": str(e)}

    def _generate_i2v(
        self, width: int, height: int, num_frames: int,
        steps: int, cfg: float, seed: int
    ) -> Dict:
        """Submit image-to-video generation job."""
        try:
            # Materialize start image
            os.makedirs(self.input_dir, exist_ok=True)
            start_image_filename = materialize_start_image(self.job, self.input_dir)
            
            if not start_image_filename:
                return {"success": False, "error": "Failed to get start image"}
            
            image_path = os.path.join(self.input_dir, start_image_filename)
            
            files = {
                'image': open(image_path, 'rb')
            }
            
            data = {
                "prompt": self.positive_prompt,
                "negative_prompt": self.negative_prompt or "",
                "num_frames": num_frames,
                "steps": steps,
                "cfg": cfg,
                "seed": seed if seed != 0 else 0
            }

            logging.info(f"Submitting I2V job with image: {image_path}")
            response = requests.post(
                f"{self.api_base_url}/generate-i2v",
                files=files,
                data=data,
                timeout=120
            )
            
            # Close file
            files['image'].close()
            
            # Cleanup local image
            if os.path.exists(image_path):
                os.remove(image_path)

            if response.status_code == 200:
                data = response.json()
                return {"success": True, "job_id": data.get("job_id")}
            else:
                logging.error(f"YUME API returned {response.status_code}: {response.text}")
                return {"success": False, "error": f"HTTP {response.status_code}"}

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to submit I2V job: {e}")
            return {"success": False, "error": str(e)}

    def _poll_job(self, api_job_id: str) -> Dict:
        """Poll job status until complete or failed."""
        start_time = time.time()
        
        while time.time() - start_time < self.MAX_WAIT_TIME:
            if self.shutdown_event.is_set():
                return {"success": False, "error": "Client shutdown"}
                
            try:
                response = requests.get(f"{self.api_base_url}/status/{api_job_id}", timeout=5)
                if response.status_code == 200:
                    status_data = response.json()
                    state = status_data.get("status")
                    
                    if state == "completed":
                        return {"success": True}
                    elif state == "failed":
                        return {"success": False, "error": status_data.get("error", "Unknown API error")}
                    
                    # Log wait occasionally
                    if time.time() % 30 < self.POLL_INTERVAL:
                        logging.info(f"Job {api_job_id} is {state}...")
                        
                elif response.status_code == 404:
                     return {"success": False, "error": "Job disappeared from API"}
            
            except requests.exceptions.RequestException:
                pass
                
            time.sleep(self.POLL_INTERVAL)
            
        return {"success": False, "error": "Timeout waiting for generation"}

    def _download_output(self, api_job_id: str) -> Optional[str]:
        """Download the generated video file."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            local_path = os.path.join(self.cache_dir, f"{self.job_id}.mp4")

            response = requests.get(
                f"{self.api_base_url}/download/{api_job_id}",
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

    def _cleanup_api_job(self, api_job_id: str):
        """Tell API to delete job data."""
        try:
            requests.delete(f"{self.api_base_url}/job/{api_job_id}", timeout=5)
        except:
            pass


class YumeTextToVideoCLIJobProcessor(YumeCLIJobProcessor):
    """Alias for YUME Text-to-Video via CLI."""
    pass


class YumeImageToVideoCLIJobProcessor(YumeCLIJobProcessor):
    """Alias for YUME Image-to-Video via CLI."""
    pass
