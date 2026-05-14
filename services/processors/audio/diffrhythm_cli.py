"""
DiffRhythm CLI Job Processor

This processor uses a REST API running inside the DiffRhythm Docker container
instead of the ComfyUI workflow approach. This provides better compatibility
with the official DiffRhythm codebase and 8GB VRAM support with --chunked flag.
"""

import os
import json
import time
import logging
import requests
from typing import Optional, Dict

from services.processors.base import BaseJobProcessor
from services.orchestrator_service import TokenExpiredError
from utils.media_utils import get_audio_duration


class DiffRhythmCLIJobProcessor(BaseJobProcessor):
    """
    Job processor that communicates with DiffRhythm via REST API.
    The DiffRhythm container runs a FastAPI server that handles generation.
    """

    API_PORT = 8000
    POLL_INTERVAL = 5
    MAX_WAIT_TIME = 600

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.api_base_url = f"http://localhost:{self.API_PORT}"

    def process(self):
        """Main processing method."""
        if not self.job:
            self._fail_job(f"Job object is None for DiffRhythmCLIJobProcessor. Cannot proceed.")
            return

        logging.info(f"Processing DiffRhythm CLI job {self.job_id}")

        lyrics, style_prompt = self._parse_prompt()
        inputs = self.job.get("inputs") or {}
        seed = inputs.get("seed", 0)

        if not self._wait_for_api():
            self._fail_job(f"DiffRhythm API did not become available for job {self.job_id}")
            return

        remote_job_id = self._submit_generation(lyrics, style_prompt, seed)
        if not remote_job_id:
            self._fail_job(f"Failed to submit generation for job {self.job_id}")
            return

        try:
            result = self._poll_for_completion(remote_job_id)

            if result.get("status") != "completed":
                error_msg = result.get("error", "Unknown error")
                self._fail_job(f"DiffRhythm generation failed: {error_msg}")
                return

            local_path = self._download_output(remote_job_id)
            if not local_path:
                self._fail_job(f"Failed to download output for job {self.job_id}")
                return

            audio_storage_path = self.orchestrator_service.upload_audio_output(local_path, self.job_id)

            if audio_storage_path:
                duration = get_audio_duration(local_path)
                completion_metadata = self.job.get("completion_metadata") or {}
                completion_metadata.update(
                    {
                        "lyrics_length": len(lyrics),
                        "style_prompt": style_prompt,
                        "has_lyrics": bool(lyrics.strip()),
                        "processor": "DiffRhythmCLIJobProcessor",
                    }
                )

                self.orchestrator_service.update_job_status(
                    self.job_id,
                    "completed",
                    storage_path=audio_storage_path,
                    duration_seconds=duration,
                    completion_metadata=completion_metadata,
                )
                logging.info(f"DiffRhythm job {self.job_id} completed successfully")
            else:
                self._fail_job(f"DiffRhythm job {self.job_id} completed, but upload failed")

        except TokenExpiredError:
            raise
        except Exception as e:
            logging.error(f"Error processing DiffRhythm job {self.job_id}: {e}", exc_info=True)
            self._fail_job(f"Error processing job: {e}")
        finally:
            self._cleanup_remote_job(remote_job_id)
            local_path = os.path.join(self.cache_dir, f"{self.job_id}.wav")
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                    logging.info(f"Cleaned up temporary file: {local_path}")
                except OSError:
                    pass

    def _parse_prompt(self) -> tuple:
        """Parse the prompt JSON to extract lyrics and style."""
        lyrics = ""
        style_prompt = self.positive_prompt

        try:
            if self.positive_prompt:
                parsed = json.loads(self.positive_prompt)
                lyrics = parsed.get("lyrics_or_edit_lyrics", "")
                style_prompt = parsed.get("style_prompt", "Pop")
                logging.info(f"Parsed prompt - Lyrics: {len(lyrics)} chars, Style: {style_prompt}")
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse prompt JSON for job {self.job_id}, using as style prompt")
            lyrics = ""
            style_prompt = self.positive_prompt or "Pop"

        return lyrics, style_prompt

    def _wait_for_api(self, timeout: int = 60) -> bool:
        """Wait for the DiffRhythm API to become available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_cancelled():
                return False
            try:
                response = requests.get(f"{self.api_base_url}/health", timeout=5)
                if response.status_code == 200:
                    logging.info("DiffRhythm API is ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

        logging.error(f"DiffRhythm API did not become available within {timeout}s")
        return False

    def _submit_generation(self, lyrics: str, style_prompt: str, seed: int = 0) -> Optional[str]:
        """Submit a generation request to the DiffRhythm API."""
        try:
            payload = {"lyrics": lyrics, "style_prompt": style_prompt, "seed": seed, "chunked": True}

            response = requests.post(f"{self.api_base_url}/generate", json=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()
                remote_job_id = data.get("job_id")
                logging.info(f"Generation job submitted: {remote_job_id}")
                return remote_job_id
            else:
                logging.error(f"Failed to submit generation: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to submit generation request: {e}")
            return None

    def _poll_for_completion(self, remote_job_id: str) -> Dict:
        """Poll the API for job completion."""
        start_time = time.time()

        while time.time() - start_time < self.MAX_WAIT_TIME:
            if self.is_cancelled():
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
        """Download the generated audio file from the API."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            local_path = os.path.join(self.cache_dir, f"{self.job_id}.wav")

            response = requests.get(f"{self.api_base_url}/download/{remote_job_id}", timeout=60, stream=True)

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
