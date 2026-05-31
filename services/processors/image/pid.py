"""
NVIDIA PiD image upscaler processor.

PiD runs as a lightweight REST service inside its own Docker image. The image
bakes the PiD code and Flux/Z-Image-compatible decoder weights, then exposes a
single image-to-image upscale API on port 8000.
"""

import logging
import os
from typing import Optional

import requests

from config import SUPABASE_URL, TimeoutConfig
from services.orchestrator_service import TokenExpiredError
from services.processors.base import BaseJobProcessor
from services.processors.output_handlers import ImageOutputHandler
from services.processors.rest_recovery import (
    poll_rest_job_with_clean_exit,
    recover_output_from_clean_container_exit,
)


class PiDImageUpscaleProcessor(BaseJobProcessor, ImageOutputHandler):
    """Processor for NVIDIA PiD still-image super-resolution via REST API."""

    API_HOST = "127.0.0.1"
    API_PORT = 8000
    POLL_INTERVAL = 3
    API_WAIT_TIMEOUT = int(os.environ.get("PID_IMAGE_API_WAIT_TIMEOUT", "300"))
    MAX_WAIT_TIME = int(
        os.environ.get(
            "PID_IMAGE_GENERATION_TIMEOUT",
            str(max(TimeoutConfig.WORKFLOW_TIMEOUT, 7200)),
        )
    )

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        api_host = os.environ.get("PID_IMAGE_API_HOST", self.API_HOST)
        api_port = int(os.environ.get("PID_IMAGE_API_PORT", self.API_PORT))
        self.api_base_url = f"http://{api_host}:{api_port}"
        self.session = requests.Session()
        self.session.trust_env = False
        self._api_startup_error: Optional[str] = None

    def process(self):
        if not self.job:
            self._fail_job("Job object is None. Cannot process PiD upscale.")
            return

        logging.info("Processing PiD image upscale job %s", self.job_id)

        if not self._wait_for_api(timeout=self.API_WAIT_TIMEOUT):
            self._fail_job(
                self._api_startup_error
                or "PiD image API / model did not become ready in time"
            )
            return

        source_image_path = self._get_source_image_path()
        if not source_image_path:
            return

        api_job_id = self._submit_upscale(source_image_path)
        if not api_job_id:
            self._fail_job("Failed to submit PiD image upscale")
            return

        local_path = None
        try:
            result = self._poll_for_completion(api_job_id)
            if result.get("status") != "completed":
                error_msg = result.get("error", "Unknown error")
                self._fail_job(f"PiD image upscale failed: {error_msg}")
                return

            local_path = self._download_output(api_job_id)
            if not local_path:
                self._fail_job("Failed to download PiD image upscale output")
                return

            image_storage_path = self.orchestrator_service.upload_image_output(
                local_path,
                self.job_id,
            )
            if not image_storage_path:
                self._fail_job("Job completed, but PiD image upload failed")
                return

            inputs = self.job.get("inputs") or {}
            try:
                upscale_factor = int(inputs.get("scale") or 4)
            except (TypeError, ValueError):
                upscale_factor = 4

            completion_metadata = self.job.get("completion_metadata") or {}
            completion_metadata.update(
                {
                    "processor": "PiDImageUpscaleProcessor",
                    "upscale_model": "NVIDIA PiD",
                    "pid_backbone": "flux-zimage-compatible",
                    "pid_ckpt_type": inputs.get("pid_ckpt_type") or "2k",
                    "upscale_factor": upscale_factor,
                    "image_upscale": True,
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
            logging.info("PiD image upscale job %s completed", self.job_id)
        except TokenExpiredError:
            raise
        except Exception as exc:
            logging.error(
                "Error processing PiD image upscale job %s: %s",
                self.job_id,
                exc,
                exc_info=True,
            )
            self._fail_job(f"Error processing PiD image upscale job: {exc}")
        finally:
            if local_path and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except OSError:
                    pass

    def _wait_for_api(self, timeout: int = 300) -> bool:
        import time

        start_time = time.monotonic()
        last_log = -31
        self._api_startup_error = None

        logging.info(
            "Waiting for PiD image API at %s (timeout: %ss)...",
            self.api_base_url,
            timeout,
        )

        while time.monotonic() - start_time < timeout:
            if self.is_cancelled():
                logging.warning("Shutdown requested while waiting for PiD API.")
                return False

            elapsed = int(time.monotonic() - start_time)
            try:
                response = self.session.get(f"{self.api_base_url}/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "unknown")
                    api_error = data.get("error")

                    if status == "error":
                        self._api_startup_error = (
                            "PiD model failed to load"
                            if not api_error
                            else f"PiD model failed to load: {api_error}"
                        )
                        logging.error("%s (after %ss)", self._api_startup_error, elapsed)
                        return False

                    if data.get("model_loaded"):
                        logging.info(
                            "PiD image API ready after %ss (model: %s)",
                            elapsed,
                            data.get("model_id", "unknown"),
                        )
                        return True

                    if elapsed - last_log >= 30:
                        logging.info(
                            "PiD image API reachable, model loading "
                            "(status=%s, %s/%ss)",
                            status,
                            elapsed,
                            timeout,
                        )
                        last_log = elapsed
            except requests.exceptions.RequestException:
                if elapsed - last_log >= 30:
                    logging.info("PiD image API not reachable yet (%s/%ss)", elapsed, timeout)
                    last_log = elapsed

            self.shutdown_event.wait(5)

        logging.error("PiD image API did not become ready within %ss", timeout)
        return False

    def _submit_upscale(self, source_image_path: str) -> Optional[str]:
        inputs = self.job.get("inputs") or {}

        def as_int(value, fallback):
            try:
                return int(value)
            except (TypeError, ValueError):
                return fallback

        def as_float(value, fallback):
            try:
                return float(value)
            except (TypeError, ValueError):
                return fallback

        input_resolution = as_int(
            inputs.get("input_resolution")
            or inputs.get("max_long_edge")
            or inputs.get("target_long_edge"),
            512,
        )
        input_resolution = max(256, min(input_resolution, 512))

        data = {
            "prompt": self.positive_prompt or "high quality detailed image",
            "input_resolution": str(input_resolution),
            "scale": "4",
            "cfg_scale": str(as_float(inputs.get("cfg", inputs.get("cfg_scale")), 1.0)),
            "pid_inference_steps": str(as_int(inputs.get("steps"), 4)),
            "degrade_sigma": str(as_float(inputs.get("degrade_sigma"), 0.0)),
            "seed": str(as_int(inputs.get("seed"), 5)),
            "preserve_aspect": "true",
        }

        try:
            with open(source_image_path, "rb") as image_file:
                files = {
                    "image": (
                        os.path.basename(source_image_path),
                        image_file,
                        "application/octet-stream",
                    )
                }
                response = self.session.post(
                    f"{self.api_base_url}/upscale",
                    data=data,
                    files=files,
                    timeout=60,
                )

            if response.status_code == 200:
                return response.json().get("job_id")

            logging.error(
                "PiD upscale submit failed: %s %s",
                response.status_code,
                response.text,
            )
            return None
        except (OSError, requests.exceptions.RequestException) as exc:
            logging.error("PiD upscale submit request error: %s", exc)
            return None

    def _poll_for_completion(self, api_job_id: str) -> dict:
        return poll_rest_job_with_clean_exit(
            self,
            self.api_base_url,
            api_job_id,
            poll_interval=self.POLL_INTERVAL,
            max_wait_time=self.MAX_WAIT_TIME,
            service_label="PiD image upscale",
            session=self.session,
        )

    def _download_output(self, api_job_id: str) -> Optional[str]:
        os.makedirs(self.cache_dir, exist_ok=True)
        local_path = os.path.join(self.cache_dir, f"{self.job_id}.png")

        try:
            response = self.session.get(
                f"{self.api_base_url}/output/{api_job_id}",
                timeout=60,
                stream=True,
            )
            if response.status_code == 200:
                with open(local_path, "wb") as output_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            output_file.write(chunk)
                return local_path

            logging.error(
                "PiD output download failed: %s %s",
                response.status_code,
                response.text,
            )
        except requests.exceptions.RequestException as exc:
            logging.error("PiD output download error: %s", exc)

        return recover_output_from_clean_container_exit(
            self,
            local_path,
            container_output_path=f"/app/output/{api_job_id}.png",
            extensions=(".png", ".jpg", ".jpeg", ".webp"),
            prefer_name=api_job_id,
        )

    def _get_source_image_path(self) -> Optional[str]:
        from utils.comfyui_workflow_utils import materialize_start_image

        filename = materialize_start_image(self.job, self.client.input_dir)
        if filename:
            path = os.path.join(self.client.input_dir, filename)
            if os.path.exists(path):
                return path

        inputs = self.job.get("inputs") or {}
        for key in ("start_image_url", "reference_image_url"):
            source_url = inputs.get(key)
            if not source_url:
                continue
            logging.info("Downloading PiD source image from signed URL")
            downloaded_path = self.orchestrator_service.download_asset_by_url(
                source_url,
                self.client.input_dir,
            )
            if downloaded_path:
                return downloaded_path

        input_storage_path = (
            self.job.get("input_storage_path")
            or inputs.get("input_storage_path")
            or inputs.get("reference_image_storage_path")
        )
        if input_storage_path:
            bucket = self.job.get("bucket", "projects_public")
            downloaded_path = self.orchestrator_service.download_storage_asset(
                bucket,
                input_storage_path,
                self.client.input_dir,
            )
            if downloaded_path:
                return downloaded_path

            source_url = (
                f"{os.environ.get('SUPABASE_URL', self.client.config.get('SUPABASE_URL', SUPABASE_URL))}"
                f"/storage/v1/object/public/{bucket}/{input_storage_path}"
            )
            logging.info("Downloading PiD source image from public URL fallback")
            downloaded_path = self.orchestrator_service.download_asset_by_url(
                source_url,
                self.client.input_dir,
            )
            if downloaded_path:
                return downloaded_path

        self._fail_job("No source image provided for PiD image upscale.")
        return None
