"""
Prism Audio Video-to-Audio Processor

REST-based processor for Prism Audio video-to-audio synthesis.
Downloads the input video, sends it to the Prism Audio REST API,
and uploads the generated audio.
"""

import os
import time
import logging
import requests
from typing import Optional, Dict

from services.processors.base import BaseJobProcessor
from services.orchestrator_service import TokenExpiredError
from utils.media_utils import get_audio_duration


class PrismAudioJobProcessor(BaseJobProcessor):
    """
    Job processor for Prism Audio video-to-audio synthesis.
    Communicates with the Prism container via REST API on port 8000.
    """

    API_PORT = 8000
    POLL_INTERVAL = 5
    MAX_WAIT_TIME = 600  # 10 minutes

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.api_base_url = f"http://localhost:{self.API_PORT}"

    def process(self):
        """Main processing method."""
        if not self.job:
            self._fail_job("Job object is None for PrismAudioJobProcessor. Cannot proceed.")
            return

        logging.info(f"Processing Prism Audio job {self.job_id}")

        # Get the input video URL
        input_video_url = self.job.get("input_video_url")
        if not input_video_url:
            self._fail_job(f"Prism Audio job {self.job_id} missing 'input_video_url'.")
            return

        # Download the video locally
        video_path = self.orchestrator_service.download_asset_by_url(input_video_url, self.input_dir)
        if not video_path:
            self._fail_job(f"Failed to download input video for Prism Audio job {self.job_id}.")
            return

        inputs = self.job.get("inputs") or {}
        prompt = self.positive_prompt or inputs.get("prompt", "")
        negative_prompt = self.negative_prompt or inputs.get("negative_prompt", "")
        duration = inputs.get("duration", 8.0)
        seed = inputs.get("seed", 42)
        cfg_strength = inputs.get("cfg_strength", 4.5)
        num_steps = inputs.get("num_steps", 25)

        # Wait for the Prism API
        if not self._wait_for_api():
            self._fail_job(f"Prism Audio API did not become available for job {self.job_id}")
            return

        # Submit generation
        remote_job_id = self._submit_generation(
            video_path=video_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            duration=duration,
            seed=seed,
            cfg_strength=cfg_strength,
            num_steps=num_steps,
        )
        if not remote_job_id:
            self._fail_job(f"Failed to submit Prism Audio generation for job {self.job_id}")
            return

        try:
            result = self._poll_for_completion(remote_job_id)

            if result.get("status") != "completed":
                error_msg = result.get("error", "Unknown error")
                self._fail_job(f"Prism Audio generation failed: {error_msg}")
                return

            local_path = self._download_output(remote_job_id)
            if not local_path:
                self._fail_job(f"Failed to download Prism Audio output for job {self.job_id}")
                return

            audio_storage_path = self.orchestrator_service.upload_audio_output(local_path, self.job_id)

            if audio_storage_path:
                duration_seconds = get_audio_duration(local_path)
                completion_metadata = self.job.get("completion_metadata") or {}
                completion_metadata.update({
                    "prompt": prompt,
                    "seed": seed,
                    "processor": "PrismAudioJobProcessor",
                })

                self.orchestrator_service.update_job_status(
                    self.job_id,
                    "completed",
                    storage_path=audio_storage_path,
                    duration_seconds=duration_seconds,
                    completion_metadata=completion_metadata,
                )
                logging.info(f"Prism Audio job {self.job_id} completed successfully")
            else:
                self._fail_job(f"Prism Audio job {self.job_id} completed, but upload failed")

        except TokenExpiredError:
            raise
        except Exception as e:
            logging.error(f"Error processing Prism Audio job {self.job_id}: {e}", exc_info=True)
            self._fail_job(f"Error processing job: {e}")
        finally:
            self._cleanup_remote_job(remote_job_id)
            # Clean up local cache
            cache_path = os.path.join(self.cache_dir, f"{self.job_id}.wav")
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

    def _wait_for_api(self, timeout: int = 120) -> bool:
        """Wait for the Prism Audio API to become available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.shutdown_event.is_set():
                return False
            try:
                response = requests.get(f"{self.api_base_url}/health", timeout=5)
                if response.status_code == 200:
                    logging.info("Prism Audio API is ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

        logging.error(f"Prism Audio API did not become available within {timeout}s")
        return False

    def _submit_generation(
        self,
        video_path: str,
        prompt: str,
        negative_prompt: str = "",
        duration: float = 8.0,
        seed: int = 42,
        cfg_strength: float = 4.5,
        num_steps: int = 25,
    ) -> Optional[str]:
        """Submit a video-to-audio generation request."""
        try:
            with open(video_path, "rb") as video_file:
                files = {"video": (os.path.basename(video_path), video_file, "video/mp4")}
                data = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "duration": str(duration),
                    "seed": str(seed),
                    "cfg_strength": str(cfg_strength),
                    "num_steps": str(num_steps),
                }

                response = requests.post(
                    f"{self.api_base_url}/generate",
                    files=files,
                    data=data,
                    timeout=60,
                )

            if response.status_code == 200:
                result = response.json()
                remote_job_id = result.get("job_id")
                logging.info(f"Prism Audio generation submitted: {remote_job_id}")
                return remote_job_id
            else:
                logging.error(f"Failed to submit Prism Audio generation: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to submit Prism Audio generation request: {e}")
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
                        logging.info(f"Prism Audio remote job {remote_job_id} completed")
                        return data
                    elif status == "failed":
                        logging.error(f"Prism Audio remote job {remote_job_id} failed: {data.get('error')}")
                        return data
                    else:
                        logging.debug(f"Prism Audio remote job {remote_job_id} status: {status}")
                else:
                    logging.warning(f"Prism Audio status check returned {response.status_code}")

            except requests.exceptions.RequestException as e:
                logging.warning(f"Prism Audio status check failed: {e}")

            time.sleep(self.POLL_INTERVAL)

        return {"status": "failed", "error": "Timeout waiting for Prism Audio generation"}

    def _download_output(self, remote_job_id: str) -> Optional[str]:
        """Download the generated audio file from the API."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            local_path = os.path.join(self.cache_dir, f"{self.job_id}.wav")

            response = requests.get(
                f"{self.api_base_url}/download/{remote_job_id}",
                timeout=60,
                stream=True,
            )

            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logging.info(f"Downloaded Prism Audio output to {local_path}")
                return local_path
            else:
                logging.error(f"Failed to download Prism Audio output: {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download Prism Audio output: {e}")
            return None

    def _cleanup_remote_job(self, remote_job_id: str):
        """Clean up the remote job and its files."""
        try:
            requests.delete(f"{self.api_base_url}/job/{remote_job_id}", timeout=10)
        except requests.exceptions.RequestException:
            pass
