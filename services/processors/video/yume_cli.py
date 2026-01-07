"""
YUME CLI Job Processor

This processor uses a REST API running inside the YUME Docker container
instead of the ComfyUI workflow approach. This provides better compatibility
with the official YUME codebase which uses a custom diffusion pipeline.

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
    Job processor that communicates with YUME via REST API.
    The YUME container runs a FastAPI server that handles generation.
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
        
        cfg_scale = inputs.get("cfg_scale", 7.0)
        steps = inputs.get("steps", 30)
        num_frames = inputs.get("num_frames", 49)
        seed = inputs.get("seed", 0)

        if not self._wait_for_api():
            self._fail_job(f"YUME API did not become available for job {self.job_id}")
            return

        if is_i2v:
            remote_job_id = self._submit_i2v_generation(
                width, height, num_frames, steps, cfg_scale, seed
            )
        else:
            remote_job_id = self._submit_t2v_generation(
                width, height, num_frames, steps, cfg_scale, seed
            )
        
        if not remote_job_id:
            self._fail_job(f"Failed to submit generation for job {self.job_id}")
            return

        try:
            result = self._poll_for_completion(remote_job_id)

            if result.get("status") != "completed":
                error_msg = result.get("error", "Unknown error")
                self._fail_job(f"YUME generation failed: {error_msg}")
                return

            local_path = self._download_output(remote_job_id)
            if not local_path:
                self._fail_job(f"Failed to download output for job {self.job_id}")
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
        finally:
            self._cleanup_remote_job(remote_job_id)
            # Cleanup local file
            local_path = os.path.join(self.cache_dir, f"{self.job_id}.mp4")
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                    logging.info(f"Cleaned up temporary file: {local_path}")
                except OSError:
                    pass

    def _wait_for_api(self, timeout: int = 120) -> bool:
        """Wait for the YUME API to become available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.shutdown_event.is_set():
                return False
            try:
                response = requests.get(f"{self.api_base_url}/health", timeout=5)
                if response.status_code == 200:
                    health = response.json()
                    if health.get("yume_available") and health.get("model_available"):
                        logging.info("YUME API is ready and model is loaded")
                        return True
                    else:
                        logging.warning(f"YUME API health check: {health}")
            except requests.exceptions.RequestException:
                pass
            time.sleep(3)

        logging.error(f"YUME API did not become available within {timeout}s")
        return False

    def _submit_t2v_generation(
        self, width: int, height: int, num_frames: int, 
        steps: int, cfg: float, seed: int
    ) -> Optional[str]:
        """Submit a text-to-video generation request."""
        try:
            payload = {
                "prompt": self.positive_prompt,
                "negative_prompt": self.negative_prompt,
                "num_frames": num_frames,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg,
                "seed": seed
            }

            response = requests.post(f"{self.api_base_url}/generate", json=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()
                remote_job_id = data.get("job_id")
                logging.info(f"YUME T2V job submitted: {remote_job_id}")
                return remote_job_id
            else:
                logging.error(f"Failed to submit T2V generation: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to submit T2V generation request: {e}")
            return None

    def _submit_i2v_generation(
        self, width: int, height: int, num_frames: int,
        steps: int, cfg: float, seed: int
    ) -> Optional[str]:
        """Submit an image-to-video generation request."""
        try:
            # Materialize start image
            os.makedirs(self.input_dir, exist_ok=True)
            start_image_filename = materialize_start_image(self.job, self.input_dir)
            
            if not start_image_filename:
                logging.error(f"Failed to materialize start image for job {self.job_id}")
                return None
            
            image_path = os.path.join(self.input_dir, start_image_filename)
            
            with open(image_path, "rb") as f:
                files = {"image": (start_image_filename, f, "image/jpeg")}
                data = {
                    "prompt": self.positive_prompt,
                    "negative_prompt": self.negative_prompt,
                    "num_frames": num_frames,
                    "steps": steps,
                    "cfg": cfg,
                    "seed": seed
                }
                
                response = requests.post(
                    f"{self.api_base_url}/generate-i2v",
                    files=files,
                    data=data,
                    timeout=60
                )

            # Cleanup local image
            if os.path.exists(image_path):
                os.remove(image_path)

            if response.status_code == 200:
                data = response.json()
                remote_job_id = data.get("job_id")
                logging.info(f"YUME I2V job submitted: {remote_job_id}")
                return remote_job_id
            else:
                logging.error(f"Failed to submit I2V generation: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to submit I2V generation request: {e}")
            return None

    def _poll_for_completion(self, remote_job_id: str) -> Dict:
        """Poll the API for job completion."""
        start_time = time.time()

        while time.time() - start_time < self.MAX_WAIT_TIME:
            if self.shutdown_event.is_set():
                return {"status": "cancelled", "error": "Shutdown requested"}

            try:
                response = requests.get(f"{self.api_base_url}/status/{remote_job_id}", timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status")

                    if status == "completed":
                        logging.info(f"Remote job {remote_job_id} completed")
                        return data
                    elif status == "failed":
                        logging.error(f"Remote job {remote_job_id} failed: {data.get('error')}")
                        return data
                    else:
                        logging.debug(f"Remote job {remote_job_id} status: {status}")
                else:
                    logging.warning(f"Status check returned {response.status_code}")

            except requests.exceptions.RequestException as e:
                logging.warning(f"Status check failed: {e}")

            time.sleep(self.POLL_INTERVAL)

        return {"status": "failed", "error": "Timeout waiting for generation"}

    def _download_output(self, remote_job_id: str) -> Optional[str]:
        """Download the generated video file from the API."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            local_path = os.path.join(self.cache_dir, f"{self.job_id}.mp4")

            response = requests.get(f"{self.api_base_url}/download/{remote_job_id}", timeout=120, stream=True)

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

    def _cleanup_remote_job(self, remote_job_id: str):
        """Clean up the remote job and its files."""
        try:
            requests.delete(f"{self.api_base_url}/job/{remote_job_id}", timeout=10)
        except requests.exceptions.RequestException:
            pass


# Create aliases for both T2V and I2V - same processor handles both
class YumeTextToVideoCLIJobProcessor(YumeCLIJobProcessor):
    """Alias for YUME Text-to-Video via CLI."""
    pass


class YumeImageToVideoCLIJobProcessor(YumeCLIJobProcessor):
    """Alias for YUME Image-to-Video via CLI."""
    pass
