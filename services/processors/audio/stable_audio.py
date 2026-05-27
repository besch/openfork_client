"""
Stable Audio 3 sound effect processor.

The Docker service exposes the Stability AI Stable Audio 3 small-sfx checkpoint
through a lightweight REST API. This processor keeps the client side aligned
with the REST-based audio services instead of the old ComfyUI workflow.
"""

import logging
import os
import time
from typing import Dict, Optional

import requests

from services.orchestrator_service import TokenExpiredError
from services.processors.base import BaseJobProcessor
from services.processors.rest_recovery import (
    poll_rest_job_with_clean_exit,
    recover_output_from_clean_container_exit,
)
from utils.media_utils import get_audio_duration


class StableAudioJobProcessor(BaseJobProcessor):
    """Processor for Stable Audio 3 Small-SFX generation."""

    API_PORT = 8000
    POLL_INTERVAL = 5
    MAX_WAIT_TIME = 900
    DEFAULT_DURATION_SECONDS = 8.0
    DEFAULT_STEPS = 8
    DEFAULT_CFG_SCALE = 1.0

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.api_base_url = f"http://localhost:{self.API_PORT}"

    def process(self):
        if not self.job:
            self._fail_job(
                "Job object is None for StableAudioJobProcessor. Cannot proceed."
            )
            return

        logging.info("Processing Stable Audio 3 SFX job %s", self.job_id)

        inputs = self.job.get("inputs") or {}
        prompt = self.positive_prompt or inputs.get("prompt", "")
        negative_prompt = self.negative_prompt or inputs.get("negative_prompt", "")
        duration = inputs.get(
            "duration_seconds",
            inputs.get("duration", self.DEFAULT_DURATION_SECONDS),
        )
        seed = inputs.get("seed", -1)
        steps = inputs.get(
            "diffusion_steps",
            inputs.get("steps", self.DEFAULT_STEPS),
        )
        cfg_scale = inputs.get("cfg_scale", self.DEFAULT_CFG_SCALE)
        chunked_decode = inputs.get("chunked_decode")

        if not self._wait_for_api():
            self._fail_job(
                f"Stable Audio 3 API did not become available for job {self.job_id}"
            )
            return

        remote_job_id = self._submit_generation(
            prompt=prompt,
            negative_prompt=negative_prompt,
            duration=duration,
            seed=seed,
            steps=steps,
            cfg_scale=cfg_scale,
            chunked_decode=chunked_decode,
        )
        if not remote_job_id:
            self._fail_job(
                f"Failed to submit Stable Audio 3 generation for job {self.job_id}"
            )
            return

        try:
            result = self._poll_for_completion(remote_job_id)
            if result.get("status") != "completed":
                self._fail_job(
                    f"Stable Audio 3 generation failed: {result.get('error', 'Unknown error')}"
                )
                return

            local_path = self._download_output(remote_job_id)
            if not local_path:
                self._fail_job(
                    f"Failed to download Stable Audio 3 output for job {self.job_id}"
                )
                return

            audio_storage_path = self.orchestrator_service.upload_audio_output(
                local_path,
                self.job_id,
            )
            if not audio_storage_path:
                self._fail_job(
                    f"Stable Audio 3 job {self.job_id} completed, but upload failed"
                )
                return

            duration_seconds = get_audio_duration(local_path)
            completion_metadata = self.job.get("completion_metadata") or {}
            completion_metadata.update(
                {
                    "prompt": prompt,
                    "seed": result.get("seed", seed),
                    "requested_duration_seconds": result.get("duration", duration),
                    "steps": result.get("steps", steps),
                    "cfg_scale": result.get("cfg_scale", cfg_scale),
                    "model_version": result.get("model", "stable-audio-3-small-sfx"),
                    "processor": "StableAudioJobProcessor",
                }
            )

            self.orchestrator_service.update_job_status(
                self.job_id,
                "completed",
                storage_path=audio_storage_path,
                duration_seconds=duration_seconds,
                completion_metadata=completion_metadata,
            )
            logging.info("Stable Audio 3 job %s completed successfully", self.job_id)
        except TokenExpiredError:
            raise
        except Exception as exc:
            logging.error(
                "Error processing Stable Audio 3 job %s: %s",
                self.job_id,
                exc,
                exc_info=True,
            )
            self._fail_job(f"Error processing Stable Audio 3 job: {exc}")
        finally:
            self._cleanup_remote_job(remote_job_id)
            cache_path = os.path.join(self.cache_dir, f"{self.job_id}.wav")
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

    def _wait_for_api(self, timeout: int = 300) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_cancelled():
                return False
            try:
                response = requests.get(f"{self.api_base_url}/health", timeout=5)
                if response.status_code == 200:
                    logging.info("Stable Audio 3 API is ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            self.shutdown_event.wait(2)
        return False

    def _submit_generation(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        duration,
        seed,
        steps,
        cfg_scale,
        chunked_decode,
    ) -> Optional[str]:
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "duration": duration,
            "seed": seed,
            "steps": steps,
            "cfg_scale": cfg_scale,
        }
        if chunked_decode is not None:
            payload["chunked_decode"] = bool(chunked_decode)

        try:
            response = requests.post(
                f"{self.api_base_url}/generate",
                json=payload,
                timeout=60,
            )
            if response.status_code == 200:
                return response.json().get("job_id")

            logging.error(
                "Stable Audio 3 submit failed: %s - %s",
                response.status_code,
                response.text,
            )
            return None
        except requests.exceptions.RequestException as exc:
            logging.error("Failed to submit Stable Audio 3 request: %s", exc)
            return None

    def _poll_for_completion(self, remote_job_id: str) -> Dict:
        return poll_rest_job_with_clean_exit(
            self,
            self.api_base_url,
            remote_job_id,
            poll_interval=self.POLL_INTERVAL,
            max_wait_time=self.MAX_WAIT_TIME,
            service_label="Stable Audio 3",
        )

    def _download_output(self, remote_job_id: str) -> Optional[str]:
        os.makedirs(self.cache_dir, exist_ok=True)
        local_path = os.path.join(self.cache_dir, f"{self.job_id}.wav")
        try:
            response = requests.get(
                f"{self.api_base_url}/download/{remote_job_id}",
                timeout=60,
                stream=True,
            )
            if response.status_code != 200:
                logging.error(
                    "Failed to download Stable Audio 3 output: %s", response.status_code
                )
                return recover_output_from_clean_container_exit(
                    self,
                    local_path,
                    container_output_path=f"/app/output/{remote_job_id}.wav",
                    extensions=(".wav",),
                    prefer_name=remote_job_id,
                )

            with open(local_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)
            return local_path
        except requests.exceptions.RequestException as exc:
            logging.error("Failed to download Stable Audio 3 output: %s", exc)
            return recover_output_from_clean_container_exit(
                self,
                local_path,
                container_output_path=f"/app/output/{remote_job_id}.wav",
                extensions=(".wav",),
                prefer_name=remote_job_id,
            )

    def _cleanup_remote_job(self, remote_job_id: str) -> None:
        if not remote_job_id:
            return
        try:
            requests.delete(f"{self.api_base_url}/job/{remote_job_id}", timeout=10)
        except requests.exceptions.RequestException:
            pass
