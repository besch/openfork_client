"""
TurboDiffusion Standalone Processors

These processors communicate with the TurboDiffusion REST API server
instead of ComfyUI for 100-200x faster video generation.

Supports:
- Text-to-Video (T2V) using TurboWan2.1-T2V-1.3B
- Image-to-Video (I2V) using TurboWan2.2-I2V-A14B
"""

import os
import time
import logging
import requests
from typing import Optional, Dict

from services.processors.base import BaseJobProcessor
from services.processors.output_handlers import VideoOutputHandler
from services.orchestrator_service import TokenExpiredError
from utils.comfyui_workflow_utils import materialize_start_image
from utils.media_utils import get_video_duration


class TurboDiffusionBaseProcessor(BaseJobProcessor, VideoOutputHandler):
    """Base class for TurboDiffusion processors."""
    
    API_PORT = 8000
    POLL_INTERVAL = 5
    MAX_WAIT_TIME = 600

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.api_base_url = f"http://localhost:{self.API_PORT}"

    def _wait_for_api(self, timeout: int = 120) -> bool:
        """Wait for the TurboDiffusion API to become available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.shutdown_event.is_set():
                return False
            try:
                response = requests.get(f"{self.api_base_url}/health", timeout=5)
                if response.status_code == 200:
                    logging.info("TurboDiffusion API is ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

        logging.error(f"TurboDiffusion API did not become available within {timeout}s")
        return False

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
                        logging.info(f"TurboDiffusion job {remote_job_id} completed")
                        return data
                    elif status == "failed":
                        logging.error(f"TurboDiffusion job {remote_job_id} failed: {data.get('error')}")
                        return data
                    else:
                        logging.debug(f"TurboDiffusion job {remote_job_id} status: {status}")
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

    def _finalize_job(self, local_path: str, remote_job_id: str):
        """Upload output and update job status."""
        try:
            video_storage_path = self.orchestrator_service.upload_video_output(local_path, self.job_id)
            
            # Generate thumbnail
            thumbnail_storage_path = None
            from utils.media_utils import extract_first_frame
            thumbnail_path = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
            if extract_first_frame(local_path, thumbnail_path):
                thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(thumbnail_path, self.job_id)
                if os.path.exists(thumbnail_path):
                    os.remove(thumbnail_path)

            if video_storage_path:
                duration = get_video_duration(local_path)
                self.orchestrator_service.update_job_status(
                    self.job_id,
                    "completed",
                    storage_path=video_storage_path,
                    thumbnail_storage_path=thumbnail_storage_path,
                    duration_seconds=duration,
                )
                logging.info(f"TurboDiffusion job {self.job_id} completed successfully")
            else:
                self._fail_job(f"TurboDiffusion job {self.job_id} completed, but upload failed")
                
        except TokenExpiredError:
            raise
        except Exception as e:
            logging.error(f"Error finalizing job {self.job_id}: {e}", exc_info=True)
            self._fail_job(f"Error finalizing job: {e}")
        finally:
            self._cleanup_remote_job(remote_job_id)
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except OSError:
                    pass


class TurboDiffusionT2VJobProcessor(TurboDiffusionBaseProcessor):
    """Processor for TurboDiffusion text-to-video generation."""

    def process(self):
        """Main processing method."""
        if not self.job:
            self._fail_job("Job object is None for TurboDiffusionT2VJobProcessor. Cannot proceed.")
            return

        logging.info(f"Processing TurboDiffusion T2V job {self.job_id}")

        if not self._wait_for_api():
            self._fail_job(f"TurboDiffusion API did not become available for job {self.job_id}")
            return

        inputs = self.job.get("inputs") or {}
        resolution = inputs.get("resolution", "480p")
        num_steps = inputs.get("num_steps", 4)
        seed = inputs.get("seed", 0)

        remote_job_id = self._submit_t2v_generation(resolution, num_steps, seed)
        if not remote_job_id:
            self._fail_job(f"Failed to submit T2V generation for job {self.job_id}")
            return

        result = self._poll_for_completion(remote_job_id)
        if result.get("status") != "completed":
            error_msg = result.get("error", "Unknown error")
            self._fail_job(f"TurboDiffusion T2V generation failed: {error_msg}")
            return

        local_path = self._download_output(remote_job_id)
        if not local_path:
            self._fail_job(f"Failed to download output for job {self.job_id}")
            return

        self._finalize_job(local_path, remote_job_id)

    def _submit_t2v_generation(self, resolution: str, num_steps: int, seed: int) -> Optional[str]:
        """Submit a T2V generation request."""
        try:
            payload = {
                "prompt": self.positive_prompt,
                "resolution": resolution,
                "num_steps": num_steps,
                "seed": seed,
            }

            response = requests.post(f"{self.api_base_url}/generate/t2v", json=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()
                remote_job_id = data.get("job_id")
                logging.info(f"T2V job submitted: {remote_job_id}")
                return remote_job_id
            else:
                logging.error(f"Failed to submit T2V: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to submit T2V request: {e}")
            return None


class TurboDiffusionI2VJobProcessor(TurboDiffusionBaseProcessor):
    """Processor for TurboDiffusion image-to-video generation."""

    def process(self):
        """Main processing method."""
        if not self.job:
            self._fail_job("Job object is None for TurboDiffusionI2VJobProcessor. Cannot proceed.")
            return

        logging.info(f"Processing TurboDiffusion I2V job {self.job_id}")

        # Materialize start image
        start_image_filename = materialize_start_image(self.job, self.input_dir)
        
        if not start_image_filename:
            input_storage_path = self.job.get("input_storage_path")
            if not input_storage_path:
                possible_path = self.job.get('start_image_base64')
                if possible_path and isinstance(possible_path, str) and not possible_path.startswith('data:') and len(possible_path) < 2048:
                    input_storage_path = possible_path

            if input_storage_path:
                bucket = self.job.get("bucket", "projects_public")
                supabase_url = os.environ.get('SUPABASE_URL', self.client.config.get('SUPABASE_URL', ''))
                if supabase_url:
                    source_url = f"{supabase_url}/storage/v1/object/public/{bucket}/{input_storage_path}"
                    logging.info(f"Downloading start image from: {source_url}")
                    downloaded_path = self.orchestrator_service.download_asset_by_url(source_url, self.input_dir)
                    if downloaded_path:
                        start_image_filename = os.path.basename(downloaded_path)

        if not start_image_filename:
            self._fail_job(f"Failed to materialize start image for job {self.job_id}.")
            return

        start_image_full_path = os.path.join(self.input_dir, start_image_filename)

        if not self._wait_for_api():
            self._fail_job(f"TurboDiffusion API did not become available for job {self.job_id}")
            return

        inputs = self.job.get("inputs") or {}
        resolution = inputs.get("resolution", "480p")
        num_steps = inputs.get("num_steps", 4)
        num_frames = inputs.get("num_frames", 49)
        seed = inputs.get("seed", 0)

        remote_job_id = self._submit_i2v_generation(start_image_full_path, resolution, num_steps, num_frames, seed)
        if not remote_job_id:
            self._fail_job(f"Failed to submit I2V generation for job {self.job_id}")
            return

        result = self._poll_for_completion(remote_job_id)
        if result.get("status") != "completed":
            error_msg = result.get("error", "Unknown error")
            self._fail_job(f"TurboDiffusion I2V generation failed: {error_msg}")
            return

        local_path = self._download_output(remote_job_id)
        if not local_path:
            self._fail_job(f"Failed to download output for job {self.job_id}")
            return

        self._finalize_job(local_path, remote_job_id)

    def _submit_i2v_generation(self, image_path: str, resolution: str, num_steps: int, num_frames: int, seed: int) -> Optional[str]:
        """Submit an I2V generation request with image upload."""
        try:
            with open(image_path, "rb") as f:
                files = {"image": (os.path.basename(image_path), f, "image/png")}
                data = {
                    "prompt": self.positive_prompt,
                    "resolution": resolution,
                    "num_steps": num_steps,
                    "num_frames": num_frames,
                    "seed": seed,
                }

                response = requests.post(f"{self.api_base_url}/generate/i2v", data=data, files=files, timeout=60)

            if response.status_code == 200:
                resp_data = response.json()
                remote_job_id = resp_data.get("job_id")
                logging.info(f"I2V job submitted: {remote_job_id}")
                return remote_job_id
            else:
                logging.error(f"Failed to submit I2V: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to submit I2V request: {e}")
            return None
