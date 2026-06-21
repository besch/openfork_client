"""
TeleStyleV2 image processor.

Submits content/style reference images to the TeleStyleV2 REST API and uploads
the generated style-transfer result back to the orchestrator.
"""

import base64
import logging
import os
import uuid
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


class TeleStyleV2Processor(BaseJobProcessor, ImageOutputHandler):
    """Processor for TeleStyleV2 content/style reference transfer."""

    API_HOST = "127.0.0.1"
    API_PORT = 8000
    POLL_INTERVAL = 5
    API_WAIT_TIMEOUT = int(os.environ.get("TELESTYLE_API_WAIT_TIMEOUT", "3600"))
    MAX_WAIT_TIME = int(
        os.environ.get(
            "TELESTYLE_GENERATION_TIMEOUT",
            str(max(TimeoutConfig.WORKFLOW_TIMEOUT, 7200)),
        )
    )

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        api_host = os.environ.get("TELESTYLE_API_HOST", self.API_HOST)
        api_port = int(os.environ.get("TELESTYLE_API_PORT", self.API_PORT))
        self.api_base_url = f"http://{api_host}:{api_port}"
        self.session = requests.Session()
        self.session.trust_env = False
        self._api_startup_error: Optional[str] = None

    def process(self):
        if not self.job:
            self._fail_job("Job object is None. Cannot process TeleStyleV2 job.")
            return

        logging.info("Processing TeleStyleV2 job %s", self.job_id)
        inputs = self.job.get("inputs") or {}

        if not self._wait_for_api(timeout=self.API_WAIT_TIMEOUT):
            self._fail_job(
                self._api_startup_error
                or "TeleStyleV2 API / model did not become ready in time"
            )
            return

        content_path = self._get_content_image_path()
        style_path = self._get_style_image_path()
        if not content_path and not style_path:
            self._fail_job("No content or style reference image provided for TeleStyleV2.")
            return

        api_job_id = self._submit_generation(content_path, style_path, inputs)
        if not api_job_id:
            self._fail_job("Failed to submit TeleStyleV2 generation")
            return

        local_path = None
        try:
            result = self._poll_for_completion(api_job_id)
            if result.get("status") != "completed":
                error_msg = result.get("error", "Unknown error")
                self._fail_job(f"TeleStyleV2 generation failed: {error_msg}")
                return

            local_path = self._download_output(api_job_id)
            if not local_path:
                self._fail_job("Failed to download TeleStyleV2 output")
                return

            image_storage_path = self.orchestrator_service.upload_image_output(
                local_path,
                self.job_id,
            )
            if not image_storage_path:
                self._fail_job("Job completed, but TeleStyleV2 image upload failed")
                return

            completion_metadata = self.job.get("completion_metadata") or {}
            completion_metadata.update(
                {
                    "processor": "TeleStyleV2Processor",
                    "model": "TeleStyleV2",
                    "seed": result.get("seed"),
                    "content_prompt": result.get("content_prompt"),
                    "style_prompt": result.get("style_prompt"),
                    "final_prompt": result.get("final_prompt"),
                    "style_transfer": True,
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
            logging.info("TeleStyleV2 job %s completed", self.job_id)
        except TokenExpiredError:
            raise
        except Exception as exc:
            logging.error(
                "Error processing TeleStyleV2 job %s: %s",
                self.job_id,
                exc,
                exc_info=True,
            )
            self._fail_job(f"Error processing TeleStyleV2 job: {exc}")
        finally:
            for path in (local_path, content_path, style_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def _wait_for_api(self, timeout: int = 3600) -> bool:
        import time

        start_time = time.monotonic()
        last_log = -31
        self._api_startup_error = None
        logging.info(
            "Waiting for TeleStyleV2 API at %s (timeout: %ss)...",
            self.api_base_url,
            timeout,
        )

        while time.monotonic() - start_time < timeout:
            if self.is_cancelled():
                logging.warning("Shutdown requested while waiting for TeleStyleV2 API.")
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
                            "TeleStyleV2 model failed to load"
                            if not api_error
                            else f"TeleStyleV2 model failed to load: {api_error}"
                        )
                        logging.error("%s (after %ss)", self._api_startup_error, elapsed)
                        return False
                    if data.get("model_loaded"):
                        logging.info(
                            "TeleStyleV2 API ready after %ss (model: %s)",
                            elapsed,
                            data.get("model_id", "unknown"),
                        )
                        return True
                    if elapsed - last_log >= 30:
                        logging.info(
                            "TeleStyleV2 API reachable, model loading "
                            "(status=%s, %s/%ss)",
                            status,
                            elapsed,
                            timeout,
                        )
                        last_log = elapsed
            except requests.exceptions.RequestException:
                if elapsed - last_log >= 30:
                    logging.info(
                        "TeleStyleV2 API not reachable yet (%s/%ss)",
                        elapsed,
                        timeout,
                    )
                    last_log = elapsed

            self.shutdown_event.wait(5)

        logging.error("TeleStyleV2 API did not become ready within %ss", timeout)
        return False

    def _submit_generation(
        self,
        content_path: Optional[str],
        style_path: Optional[str],
        inputs: dict,
    ) -> Optional[str]:
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

        prompt = (
            self.positive_prompt
            or inputs.get("prompt")
            or "Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1."
        )
        data = {
            "prompt": prompt,
            "seed": str(as_int(inputs.get("seed"), 123)),
            "randomize_seed": str(bool(inputs.get("randomize_seed", False))).lower(),
            "true_guidance_scale": str(
                as_float(
                    inputs.get("true_guidance_scale")
                    or inputs.get("cfg")
                    or inputs.get("cfg_scale"),
                    1.0,
                )
            ),
            "num_inference_steps": str(as_int(inputs.get("steps"), 4)),
            "min_edge": str(
                as_int(
                    inputs.get("min_edge")
                    or inputs.get("minedge")
                    or inputs.get("long_edge")
                    or inputs.get("target_long_edge"),
                    1024,
                )
            ),
            "use_content_prompt": str(
                bool(inputs.get("use_content_prompt", False))
            ).lower(),
            "use_style_prompt": str(bool(inputs.get("use_style_prompt", False))).lower(),
        }

        files = {}
        handles = []
        try:
            if content_path:
                content_handle = open(content_path, "rb")
                handles.append(content_handle)
                files["content_image"] = (
                    os.path.basename(content_path),
                    content_handle,
                    "application/octet-stream",
                )
            if style_path:
                style_handle = open(style_path, "rb")
                handles.append(style_handle)
                files["style_image"] = (
                    os.path.basename(style_path),
                    style_handle,
                    "application/octet-stream",
                )

            response = self.session.post(
                f"{self.api_base_url}/generate",
                data=data,
                files=files,
                timeout=120,
            )
            if response.status_code == 200:
                return response.json().get("job_id")
            logging.error(
                "TeleStyleV2 generate submit failed: %s %s",
                response.status_code,
                response.text,
            )
            return None
        except (OSError, requests.exceptions.RequestException) as exc:
            logging.error("TeleStyleV2 generate submit request error: %s", exc)
            return None
        finally:
            for handle in handles:
                try:
                    handle.close()
                except OSError:
                    pass

    def _poll_for_completion(self, api_job_id: str) -> dict:
        return poll_rest_job_with_clean_exit(
            self,
            self.api_base_url,
            api_job_id,
            poll_interval=self.POLL_INTERVAL,
            max_wait_time=self.MAX_WAIT_TIME,
            service_label="TeleStyleV2",
            session=self.session,
        )

    def _download_output(self, api_job_id: str) -> Optional[str]:
        os.makedirs(self.cache_dir, exist_ok=True)
        local_path = os.path.join(self.cache_dir, f"{self.job_id}.png")
        try:
            response = self.session.get(
                f"{self.api_base_url}/output/{api_job_id}",
                timeout=120,
                stream=True,
            )
            if response.status_code == 200:
                with open(local_path, "wb") as output_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            output_file.write(chunk)
                return local_path
            logging.error(
                "TeleStyleV2 output download failed: %s %s",
                response.status_code,
                response.text,
            )
        except requests.exceptions.RequestException as exc:
            logging.error("TeleStyleV2 output download error: %s", exc)

        return recover_output_from_clean_container_exit(
            self,
            local_path,
            container_output_path=f"/app/output/{api_job_id}.png",
            extensions=(".png", ".jpg", ".jpeg", ".webp"),
            prefer_name=api_job_id,
        )

    def _get_content_image_path(self) -> Optional[str]:
        from utils.comfyui_workflow_utils import materialize_start_image

        filename = materialize_start_image(self.job, self.client.input_dir)
        if filename:
            path = os.path.join(self.client.input_dir, filename)
            if os.path.exists(path):
                return path

        inputs = self.job.get("inputs") or {}
        return self._resolve_image_from_inputs(
            url_keys=(
                "content_image_url",
                "content_reference_image_url",
                "start_image_url",
                "reference_image_url",
            ),
            storage_keys=(
                "content_image_storage_path",
                "content_reference_image_storage_path",
                "start_image_storage_path",
                "reference_image_storage_path",
                "input_storage_path",
            ),
            base64_keys=("content_image_base64", "content_reference_image_base64"),
            bucket_keys=(
                "content_image_bucket",
                "content_reference_image_bucket",
                "reference_image_bucket",
            ),
            inputs=inputs,
            label="content",
        ) or self._resolve_job_input_storage(label="content")

    def _get_style_image_path(self) -> Optional[str]:
        inputs = self.job.get("inputs") or {}
        return self._resolve_image_from_inputs(
            url_keys=(
                "style_image_url",
                "style_reference_image_url",
                "reference_image_2_url",
                "reference_image2_url",
                "second_reference_image_url",
            ),
            storage_keys=(
                "style_image_storage_path",
                "style_reference_image_storage_path",
                "reference_image_2_storage_path",
                "reference_image2_storage_path",
                "second_reference_image_storage_path",
                "reference_image_2",
                "reference_image2",
            ),
            base64_keys=(
                "style_image_base64",
                "style_reference_image_base64",
                "reference_image_2_base64",
                "reference_image2_base64",
                "second_reference_image_base64",
            ),
            bucket_keys=(
                "style_image_bucket",
                "style_reference_image_bucket",
                "reference_image_2_bucket",
                "reference_image2_bucket",
            ),
            inputs=inputs,
            label="style",
        )

    def _resolve_image_from_inputs(
        self,
        *,
        url_keys: tuple[str, ...],
        storage_keys: tuple[str, ...],
        base64_keys: tuple[str, ...],
        bucket_keys: tuple[str, ...],
        inputs: dict,
        label: str,
    ) -> Optional[str]:
        for key in base64_keys:
            path = self._materialize_base64_image(inputs.get(key), label)
            if path:
                return path

        for key in url_keys:
            source_url = inputs.get(key)
            if source_url:
                logging.info("Downloading TeleStyleV2 %s image from signed URL", label)
                path = self.orchestrator_service.download_asset_by_url(
                    source_url,
                    self.client.input_dir,
                )
                if path:
                    return path

        storage_path = next((inputs.get(key) for key in storage_keys if inputs.get(key)), None)
        if storage_path:
            bucket = (
                next((inputs.get(key) for key in bucket_keys if inputs.get(key)), None)
                or self.job.get("bucket", "projects_public")
            )
            path = self.orchestrator_service.download_storage_asset(
                bucket,
                storage_path,
                self.client.input_dir,
            )
            if path:
                return path

            source_url = (
                f"{os.environ.get('SUPABASE_URL', self.client.config.get('SUPABASE_URL', SUPABASE_URL))}"
                f"/storage/v1/object/public/{bucket}/{storage_path}"
            )
            logging.info("Downloading TeleStyleV2 %s image from public URL fallback", label)
            return self.orchestrator_service.download_asset_by_url(
                source_url,
                self.client.input_dir,
            )

        return None

    def _resolve_job_input_storage(self, label: str) -> Optional[str]:
        input_storage_path = self.job.get("input_storage_path")
        if not input_storage_path:
            return None
        bucket = self.job.get("bucket", "projects_public")
        path = self.orchestrator_service.download_storage_asset(
            bucket,
            input_storage_path,
            self.client.input_dir,
        )
        if path:
            return path
        source_url = (
            f"{os.environ.get('SUPABASE_URL', self.client.config.get('SUPABASE_URL', SUPABASE_URL))}"
            f"/storage/v1/object/public/{bucket}/{input_storage_path}"
        )
        logging.info("Downloading TeleStyleV2 %s image from public URL fallback", label)
        return self.orchestrator_service.download_asset_by_url(
            source_url,
            self.client.input_dir,
        )

    def _materialize_base64_image(self, data_url, label: str) -> Optional[str]:
        if not data_url:
            return None
        try:
            image_base64 = data_url
            extension = "png"
            if "," in image_base64:
                header, image_base64 = image_base64.split(",", 1)
                if "image/jpeg" in header or "image/jpg" in header:
                    extension = "jpg"
                elif "image/webp" in header:
                    extension = "webp"
            image_data = base64.b64decode(image_base64)
            filename = f"telestylev2_{label}_{uuid.uuid4().hex[:8]}.{extension}"
            path = os.path.join(self.client.input_dir, filename)
            with open(path, "wb") as output:
                output.write(image_data)
            logging.info("Saved TeleStyleV2 %s image to: %s", label, path)
            return path
        except Exception as exc:
            self._fail_job(f"Failed to process TeleStyleV2 {label} image: {exc}")
            return None
