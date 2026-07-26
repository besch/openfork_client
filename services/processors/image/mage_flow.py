"""DGN processor for Microsoft Mage-Flow generation and multi-image editing."""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

from config import TimeoutConfig
from services.processors.base import BaseJobProcessor
from services.processors.output_handlers import ImageOutputHandler
from services.processors.rest_recovery import poll_rest_job_with_clean_exit

log = logging.getLogger(__name__)


class MageFlowImageProcessor(BaseJobProcessor, ImageOutputHandler):
    API_HOST = os.getenv("MAGE_FLOW_API_HOST", "127.0.0.1")
    API_PORT = int(os.getenv("MAGE_FLOW_API_PORT", "8000"))
    API_WAIT_TIMEOUT = int(os.getenv("MAGE_FLOW_API_WAIT_TIMEOUT", "600"))
    MAX_WAIT_TIME = int(
        os.getenv(
            "MAGE_FLOW_GENERATION_TIMEOUT",
            str(max(TimeoutConfig.WORKFLOW_TIMEOUT, 7200)),
        )
    )

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.api_base_url = f"http://{self.API_HOST}:{self.API_PORT}"
        self.session = requests.Session()
        self.session.trust_env = False

    def process(self):
        inputs = self.job.get("inputs") or {}
        references: list[str] = []
        output_path = None
        try:
            if not self._wait_for_api():
                self._fail_job("Mage-Flow API did not become ready")
                return
            references = self._materialize_references(inputs)
            is_edit = "image-edit" in (self.workflow_type or "")
            if is_edit and not references:
                self._fail_job("Mage-Flow edit requires at least one reference image")
                return
            remote_job_id = self._submit(inputs, references, is_edit)
            if not remote_job_id:
                self._fail_job("Failed to submit Mage-Flow generation")
                return
            result = poll_rest_job_with_clean_exit(
                self,
                self.api_base_url,
                remote_job_id,
                poll_interval=3,
                max_wait_time=self.MAX_WAIT_TIME,
                service_label="Mage-Flow",
                session=self.session,
            )
            if result.get("status") != "completed":
                self._fail_job(
                    f"Mage-Flow generation failed: {result.get('error', 'Unknown error')}"
                )
                return
            output_path = os.path.join(self.cache_dir, f"{self.job_id}_mage.png")
            os.makedirs(self.cache_dir, exist_ok=True)
            response = self.session.get(
                f"{self.api_base_url}/output/{remote_job_id}",
                timeout=180,
                stream=True,
            )
            response.raise_for_status()
            with open(output_path, "wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            self._apply_transparent_background_if_requested(output_path)
            storage_path = self.orchestrator_service.upload_image_output(
                output_path, self.job_id
            )
            if not storage_path:
                self._fail_job("Mage-Flow output upload failed")
                return
            metadata = self.job.get("completion_metadata") or {}
            metadata.update(
                {
                    "processor": "MageFlowImageProcessor",
                    "model": "Mage-Flow",
                    "variant": "turbo" if "turbo" in (self.workflow_type or "") else "quality",
                    "operation": "edit" if is_edit else "generate",
                    "reference_image_count": len(references),
                }
            )
            self.orchestrator_service.update_job_status(
                self.job_id,
                "completed",
                storage_path=storage_path,
                thumbnail_storage_path=storage_path,
                prompt=self.positive_prompt,
                completion_metadata=metadata,
            )
        except Exception as exc:
            if "TokenExpiredError" in type(exc).__name__:
                raise
            log.exception("Mage-Flow job %s failed", self.job_id)
            self._fail_job(f"Mage-Flow processing error: {exc}")
        finally:
            for path in [*references, output_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def _wait_for_api(self) -> bool:
        started = time.monotonic()
        while time.monotonic() - started < self.API_WAIT_TIMEOUT:
            if self.is_cancelled():
                return False
            try:
                response = self.session.get(f"{self.api_base_url}/health", timeout=5)
                if response.status_code == 200 and response.json().get("model_loaded"):
                    return True
                if response.status_code == 200 and response.json().get("status") == "error":
                    log.error("Mage-Flow API error: %s", response.json().get("error"))
            except requests.RequestException:
                pass
            self.shutdown_event.wait(5)
        return False

    def _materialize_references(self, inputs: dict) -> list[str]:
        values = inputs.get("reference_image_storage_paths") or []
        if isinstance(values, str):
            values = [values]
        primary = (
            inputs.get("input_storage_path")
            or self.job.get("input_storage_path")
            or inputs.get("start_image")
        )
        ordered = [primary, *values]
        unique = []
        for value in ordered:
            if not isinstance(value, str) or not value.strip() or value in unique:
                continue
            unique.append(value)

        paths = []
        bucket = inputs.get("bucket") or self.job.get("bucket") or "projects_public"
        for value in unique[:8]:
            if value.startswith(("http://", "https://")):
                path = self.orchestrator_service.download_asset_by_url(
                    value, self.input_dir
                )
            else:
                path = self.orchestrator_service.download_storage_asset(
                    bucket, value.split("|", 1)[0], self.input_dir
                )
            if path:
                paths.append(path)
        return paths

    def _submit(self, inputs: dict, references: list[str], is_edit: bool) -> Optional[str]:
        ratio_dimensions = {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "9:16": (768, 1344),
            "4:3": (1152, 864),
            "3:4": (864, 1152),
            "2:1": (1440, 720),
            "1:2": (720, 1440),
            "4:1": (2048, 512),
            "1:4": (512, 2048),
        }
        width, height = ratio_dimensions.get(inputs.get("aspect_ratio"), (1024, 1024))
        is_turbo = "turbo" in (self.workflow_type or "")
        payload = {
            "prompt": self.positive_prompt,
            "operation": "edit" if is_edit else "generate",
            "variant": "turbo" if is_turbo else "quality",
            "negative_prompt": inputs.get("negative_prompt") or self.negative_prompt or "",
            "width": inputs.get("width") or width,
            "height": inputs.get("height") or height,
            "steps": inputs.get("steps") or (4 if is_turbo else 30 if is_edit else 20),
            "cfg": inputs.get("cfg_scale") or inputs.get("cfg") or (1 if is_turbo else 5),
            "seed": self.normalize_seed(inputs.get("seed", 42)),
        }
        handles = []
        files = []
        try:
            for path in references:
                handle = open(path, "rb")
                handles.append(handle)
                files.append(("reference_images", (os.path.basename(path), handle, "application/octet-stream")))
            response = self.session.post(
                f"{self.api_base_url}/generate",
                data={key: str(value) for key, value in payload.items()},
                files=files,
                timeout=120,
            )
            if response.status_code == 200:
                return response.json().get("job_id")
            log.error("Mage-Flow submit failed: %s %s", response.status_code, response.text)
            return None
        finally:
            for handle in handles:
                handle.close()
