"""
AudioX sound effect processor.

Supports text-to-audio jobs today and can optionally pass a video input through
to the AudioX REST API for video-conditioned audio.
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


class AudioXJobProcessor(BaseJobProcessor):
    """Job processor for AudioX sound effect generation."""

    API_PORT = 8000
    POLL_INTERVAL = 5
    MAX_WAIT_TIME = 900
    FIXED_TEXT_AUDIO_DURATION = 10.0

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.api_base_url = f"http://localhost:{self.API_PORT}"

    def process(self):
        if not self.job:
            self._fail_job("Job object is None for AudioXJobProcessor. Cannot proceed.")
            return

        logging.info("Processing AudioX job %s", self.job_id)

        inputs = self.job.get("inputs") or {}
        prompt = self.positive_prompt or inputs.get("prompt", "")
        negative_prompt = self.negative_prompt or inputs.get("negative_prompt", "")
        duration = inputs.get("duration_seconds", inputs.get("duration", 10.0))
        seed = inputs.get("seed", -1)
        cfg_scale = inputs.get("cfg_scale", 7.0)
        num_steps = inputs.get("diffusion_steps", inputs.get("num_steps", 100))
        sampler_type = inputs.get("sampler_type", "dpmpp-3m-sde")
        sigma_min = inputs.get("sigma_min", 0.03)
        sigma_max = inputs.get("sigma_max", 500)

        if not self._wait_for_api():
            self._fail_job(f"AudioX API did not become available for job {self.job_id}")
            return

        video_path = None
        input_video_url = self.job.get("input_video_url") or inputs.get("input_video_url")
        if input_video_url:
            video_path = self.orchestrator_service.download_asset_by_url(
                input_video_url,
                self.input_dir,
            )
            if not video_path:
                self._fail_job(f"Failed to download input video for AudioX job {self.job_id}.")
                return
        else:
            duration = self._normalize_text_duration(duration)

        remote_job_id = self._submit_generation(
            prompt=prompt,
            negative_prompt=negative_prompt,
            video_path=video_path,
            duration=duration,
            seed=seed,
            cfg_scale=cfg_scale,
            num_steps=num_steps,
            sampler_type=sampler_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        if not remote_job_id:
            self._fail_job(f"Failed to submit AudioX generation for job {self.job_id}")
            return

        try:
            result = self._poll_for_completion(remote_job_id)
            if result.get("status") != "completed":
                self._fail_job(f"AudioX generation failed: {result.get('error', 'Unknown error')}")
                return

            local_path = self._download_output(remote_job_id)
            if not local_path:
                self._fail_job(f"Failed to download AudioX output for job {self.job_id}")
                return

            audio_storage_path = self.orchestrator_service.upload_audio_output(
                local_path,
                self.job_id,
            )
            if not audio_storage_path:
                self._fail_job(f"AudioX job {self.job_id} completed, but upload failed")
                return

            duration_seconds = get_audio_duration(local_path)
            completion_metadata = self.job.get("completion_metadata") or {}
            completion_metadata.update(
                {
                    "prompt": prompt,
                    "seed": seed,
                    "processor": "AudioXJobProcessor",
                }
            )

            self.orchestrator_service.update_job_status(
                self.job_id,
                "completed",
                storage_path=audio_storage_path,
                duration_seconds=duration_seconds,
                completion_metadata=completion_metadata,
            )
            logging.info("AudioX job %s completed successfully", self.job_id)
        except TokenExpiredError:
            raise
        except Exception as exc:
            logging.error("Error processing AudioX job %s: %s", self.job_id, exc, exc_info=True)
            self._fail_job(f"Error processing AudioX job: {exc}")
        finally:
            self._cleanup_remote_job(remote_job_id)
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except OSError:
                    pass
            cache_path = os.path.join(self.cache_dir, f"{self.job_id}.wav")
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

    def _wait_for_api(self, timeout: int = 180) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_cancelled():
                return False
            try:
                response = requests.get(f"{self.api_base_url}/health", timeout=5)
                if response.status_code == 200:
                    logging.info("AudioX API is ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        return False

    def _submit_generation(
        self,
        prompt: str,
        negative_prompt: str,
        video_path: Optional[str],
        duration: float,
        seed: int,
        cfg_scale: float,
        num_steps: int,
        sampler_type: str,
        sigma_min: float,
        sigma_max: float,
    ) -> Optional[str]:
        try:
            if video_path:
                with open(video_path, "rb") as video_file:
                    files = {"video": (os.path.basename(video_path), video_file, "video/mp4")}
                    data = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "duration": str(duration),
                        "seed": str(seed),
                        "cfg_scale": str(cfg_scale),
                        "num_steps": str(num_steps),
                        "sampler_type": sampler_type,
                        "sigma_min": str(sigma_min),
                        "sigma_max": str(sigma_max),
                    }
                    response = requests.post(
                        f"{self.api_base_url}/generate",
                        files=files,
                        data=data,
                        timeout=60,
                    )
            else:
                payload = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "duration": duration,
                    "seed": seed,
                    "cfg_scale": cfg_scale,
                    "steps": num_steps,
                    "sampler_type": sampler_type,
                    "sigma_min": sigma_min,
                    "sigma_max": sigma_max,
                }
                response = requests.post(
                    f"{self.api_base_url}/generate-text",
                    json=payload,
                    timeout=60,
                )

            if response.status_code == 200:
                return response.json().get("job_id")

            logging.error("AudioX submit failed: %s - %s", response.status_code, response.text)
            return None
        except requests.exceptions.RequestException as exc:
            logging.error("Failed to submit AudioX request: %s", exc)
            return None

    def _normalize_text_duration(self, duration) -> float:
        try:
            requested = float(duration)
        except (TypeError, ValueError):
            requested = self.FIXED_TEXT_AUDIO_DURATION

        if requested != self.FIXED_TEXT_AUDIO_DURATION:
            logging.warning(
                "AudioX text-to-audio currently supports a fixed %.0fs "
                "conditioning window. Requested %.2fs will be run as %.0fs "
                "to avoid known tensor-shape failures.",
                self.FIXED_TEXT_AUDIO_DURATION,
                requested,
                self.FIXED_TEXT_AUDIO_DURATION,
            )

        return self.FIXED_TEXT_AUDIO_DURATION

    def _poll_for_completion(self, remote_job_id: str) -> Dict:
        return poll_rest_job_with_clean_exit(
            self,
            self.api_base_url,
            remote_job_id,
            poll_interval=self.POLL_INTERVAL,
            max_wait_time=self.MAX_WAIT_TIME,
            service_label="AudioX",
        )

    def _download_output(self, remote_job_id: str) -> Optional[str]:
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            local_path = os.path.join(self.cache_dir, f"{self.job_id}.wav")
            response = requests.get(
                f"{self.api_base_url}/download/{remote_job_id}",
                timeout=60,
                stream=True,
            )
            if response.status_code != 200:
                logging.error("Failed to download AudioX output: %s", response.status_code)
                return recover_output_from_clean_container_exit(
                    self,
                    local_path,
                    container_output_path=f"/app/output/{remote_job_id}.wav",
                    extensions=(".wav",),
                    prefer_name=remote_job_id,
                )

            with open(local_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    handle.write(chunk)
            return local_path
        except requests.exceptions.RequestException as exc:
            logging.error("Failed to download AudioX output: %s", exc)
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
