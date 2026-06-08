"""
Ideogram 4 image processor.

Communicates with the Ideogram 4 REST API (FastAPI server on port 8000) for
text-to-image generation.
"""

import logging
import os
import time
from typing import Optional

import requests

from config import TimeoutConfig
from services.orchestrator_service import TokenExpiredError
from services.processors.base import BaseJobProcessor
from services.processors.output_handlers import ImageOutputHandler
from services.processors.rest_recovery import (
    poll_rest_job_with_clean_exit,
    recover_output_from_clean_container_exit,
)


class Ideogram4ImageProcessor(BaseJobProcessor, ImageOutputHandler):
    API_HOST = "127.0.0.1"
    API_PORT = 8000
    POLL_INTERVAL = 3
    API_WAIT_TIMEOUT = int(os.environ.get("IDEOGRAM4_API_WAIT_TIMEOUT", "420"))
    MAX_WAIT_TIME = int(
        os.environ.get(
            "IDEOGRAM4_GENERATION_TIMEOUT",
            str(max(TimeoutConfig.WORKFLOW_TIMEOUT, 7200)),
        )
    )

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        api_host = os.environ.get("IDEOGRAM4_API_HOST", self.API_HOST)
        api_port = int(os.environ.get("IDEOGRAM4_API_PORT", self.API_PORT))
        self.api_base_url = f"http://{api_host}:{api_port}"
        self.session = requests.Session()
        self.session.trust_env = False
        self._api_startup_error: Optional[str] = None

    def process(self):
        if not self.job:
            self._fail_job("Job object is None. Cannot proceed.")
            return

        logging.info("Processing Ideogram 4 job %s", self.job_id)
        inputs = self.job.get("inputs") or {}

        if not self._wait_for_api(timeout=self.API_WAIT_TIMEOUT):
            self._fail_job(
                self._api_startup_error
                or "Ideogram 4 API / model did not become ready in time"
            )
            return

        job_id = self._submit_generation(inputs)
        if not job_id:
            self._fail_job("Failed to submit Ideogram 4 generation")
            return

        local_path = None
        try:
            result = self._poll_for_completion(job_id)

            if result.get("status") != "completed":
                error_msg = result.get("error", "Unknown error")
                self._fail_job(f"Ideogram 4 generation failed: {error_msg}")
                return

            local_path = self._download_output(job_id)
            if not local_path:
                self._fail_job("Failed to download Ideogram 4 output")
                return

            self._apply_transparent_background_if_requested(local_path)
            image_storage_path = self.orchestrator_service.upload_image_output(
                local_path, self.job_id
            )

            if image_storage_path:
                completion_metadata = self.job.get("completion_metadata") or {}
                completion_metadata.update(
                    {
                        "processor": "Ideogram4ImageProcessor",
                        "model": "ideogram4",
                        "seed": result.get("seed"),
                    }
                )

                self.orchestrator_service.update_job_status(
                    self.job_id,
                    "completed",
                    storage_path=image_storage_path,
                    thumbnail_storage_path=image_storage_path,
                    prompt=self.positive_prompt,
                    completion_metadata=completion_metadata,
                )
                logging.info("Ideogram 4 job %s completed successfully", self.job_id)
            else:
                self._fail_job("Job completed, but image upload failed")

        except TokenExpiredError:
            raise
        except Exception as exc:
            logging.error(
                "Error processing Ideogram 4 job %s: %s",
                self.job_id,
                exc,
                exc_info=True,
            )
            self._fail_job(f"Error processing job: {exc}")
        finally:
            if local_path and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except OSError:
                    pass

    def _wait_for_api(self, timeout: int = 420) -> bool:
        start_time = time.monotonic()
        last_log = -31
        logging.info(
            "Waiting for Ideogram 4 API at %s (timeout: %ss)...",
            self.api_base_url,
            timeout,
        )
        self._api_startup_error = None

        while time.monotonic() - start_time < timeout:
            if self.is_cancelled():
                logging.warning("Shutdown requested while waiting for API.")
                return False

            elapsed = int(time.monotonic() - start_time)

            try:
                response = self.session.get(f"{self.api_base_url}/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    api_status = data.get("status", "unknown")
                    api_error = data.get("error")

                    if api_status == "error":
                        self._api_startup_error = (
                            "Ideogram 4 model failed to load"
                            if not api_error
                            else f"Ideogram 4 model failed to load: {api_error}"
                        )
                        logging.error("%s (after %ss)", self._api_startup_error, elapsed)
                        return False

                    if data.get("model_loaded"):
                        logging.info(
                            "Ideogram 4 API ready after %ss (model: %s)",
                            elapsed,
                            data.get("model_id", "unknown"),
                        )
                        return True

                    if elapsed - last_log >= 30:
                        msg = (
                            "Ideogram 4 API reachable, model loading "
                            f"(status={api_status}, {elapsed}/{timeout}s)"
                        )
                        if api_error:
                            msg += f" | error: {api_error}"
                        logging.info(msg)
                        last_log = elapsed

            except requests.exceptions.RequestException:
                if elapsed - last_log >= 30:
                    logging.info(
                        "Ideogram 4 API not reachable yet (%s/%ss)",
                        elapsed,
                        timeout,
                    )
                    last_log = elapsed

            time.sleep(5)

        logging.error("Ideogram 4 API did not become ready within %ss", timeout)
        return False

    def _submit_generation(self, inputs: dict) -> Optional[str]:
        try:
            seed = inputs.get("seed")
            width, height = self._resolve_dimensions(inputs.get("aspect_ratio"))
            payload = {
                "prompt": self.positive_prompt,
                "width": width,
                "height": height,
                "sampler_preset": inputs.get("sampler_preset") or "V4_QUALITY_48",
                "warn_on_caption_issues": True,
            }
            if seed is not None:
                payload["seed"] = int(seed)
            if inputs.get("use_magic_prompt") is not None:
                payload["use_magic_prompt"] = bool(inputs.get("use_magic_prompt"))

            response = self.session.post(
                f"{self.api_base_url}/generate", json=payload, timeout=30
            )
            if response.status_code == 200:
                return response.json().get("job_id")
            logging.error(
                "Ideogram 4 generate failed: %s %s",
                response.status_code,
                response.text,
            )
            return None
        except requests.exceptions.RequestException as exc:
            logging.error("Ideogram 4 generate request error: %s", exc)
            return None

    def _poll_for_completion(self, api_job_id: str) -> dict:
        return poll_rest_job_with_clean_exit(
            self,
            self.api_base_url,
            api_job_id,
            poll_interval=self.POLL_INTERVAL,
            max_wait_time=self.MAX_WAIT_TIME,
            service_label="Ideogram 4",
            session=self.session,
        )

    def _download_output(self, api_job_id: str) -> Optional[str]:
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            local_path = os.path.join(self.cache_dir, f"{self.job_id}.png")

            response = self.session.get(
                f"{self.api_base_url}/output/{api_job_id}",
                timeout=60,
                stream=True,
            )
            if response.status_code == 200:
                with open(local_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=8192):
                        handle.write(chunk)
                return local_path
            logging.error(
                "Ideogram 4 download failed: %s %s",
                response.status_code,
                response.text,
            )
            return recover_output_from_clean_container_exit(
                self,
                local_path,
                container_output_path=f"/app/output/{api_job_id}.png",
                extensions=(".png", ".jpg", ".jpeg", ".webp"),
                prefer_name=api_job_id,
            )
        except requests.exceptions.RequestException as exc:
            logging.error("Ideogram 4 download error: %s", exc)
            return recover_output_from_clean_container_exit(
                self,
                local_path,
                container_output_path=f"/app/output/{api_job_id}.png",
                extensions=(".png", ".jpg", ".jpeg", ".webp"),
                prefer_name=api_job_id,
            )

    def _resolve_dimensions(self, aspect_ratio: Optional[str]) -> tuple[int, int]:
        ratios = {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "9:16": (768, 1344),
            "4:3": (1152, 864),
            "3:4": (864, 1152),
            "3:2": (1216, 816),
            "2:3": (816, 1216),
        }
        return ratios.get(aspect_ratio, (1024, 1024))
