"""
ERNIE-Image Processor

Processor for Baidu ERNIE-Image text-to-image generation.
Communicates with the ERNIE-Image REST API (FastAPI server on port 8000).
"""

import os
import time
import logging
import requests
from typing import Optional

from services.processors.base import BaseJobProcessor
from services.orchestrator_service import TokenExpiredError


class ErnieImageProcessor(BaseJobProcessor):
    """
    Job processor for ERNIE-Image text-to-image generation via REST API.

    Endpoints:
      POST /generate   - submit generation, returns job_id
      GET  /status/{id} - poll status
      GET  /output/{id} - download output image
      GET  /health      - health check
    """

    API_HOST = "127.0.0.1"
    API_PORT = 8000
    POLL_INTERVAL = 3
    MAX_WAIT_TIME = 600

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        api_host = os.environ.get("ERNIE_IMAGE_API_HOST", self.API_HOST)
        api_port = int(os.environ.get("ERNIE_IMAGE_API_PORT", self.API_PORT))
        self.api_base_url = f"http://{api_host}:{api_port}"
        self.session = requests.Session()
        self.session.trust_env = False

    def process(self):
        if not self.job:
            self._fail_job("Job object is None. Cannot proceed.")
            return

        logging.info(f"Processing ERNIE-Image job {self.job_id}")

        inputs = self.job.get("inputs") or {}

        if not self._wait_for_api():
            self._fail_job("ERNIE-Image API did not become available")
            return

        job_id = self._submit_generation(inputs)
        if not job_id:
            self._fail_job("Failed to submit ERNIE-Image generation")
            return

        local_path = None
        try:
            result = self._poll_for_completion(job_id)

            if result.get("status") != "completed":
                error_msg = result.get("error", "Unknown error")
                self._fail_job(f"ERNIE-Image generation failed: {error_msg}")
                return

            local_path = self._download_output(job_id)
            if not local_path:
                self._fail_job("Failed to download ERNIE-Image output")
                return

            image_storage_path = self.orchestrator_service.upload_image_output(
                local_path, self.job_id
            )

            if image_storage_path:
                completion_metadata = self.job.get("completion_metadata") or {}
                completion_metadata.update(
                    {
                        "processor": "ErnieImageProcessor",
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
                logging.info(f"ERNIE-Image job {self.job_id} completed successfully")
            else:
                self._fail_job("Job completed, but image upload failed")

        except TokenExpiredError:
            raise
        except Exception as e:
            logging.error(
                f"Error processing ERNIE-Image job {self.job_id}: {e}", exc_info=True
            )
            self._fail_job(f"Error processing job: {e}")
        finally:
            if local_path and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except OSError:
                    pass

    def _wait_for_api(self, timeout: int = 600) -> bool:
        start_time = time.monotonic()
        last_log = -30

        logging.info(
            f"Waiting for ERNIE-Image API at {self.api_base_url} (timeout: {timeout}s)..."
        )

        while time.monotonic() - start_time < timeout:
            if self.shutdown_event.is_set():
                return False

            elapsed = int(time.monotonic() - start_time)
            try:
                response = self.session.get(f"{self.api_base_url}/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("model_loaded"):
                        logging.info(
                            f"ERNIE-Image API is ready after {elapsed}s (model: {data.get('model_id', 'unknown')})"
                        )
                        return True
                    status = data.get("status", "unknown")
                    if elapsed - last_log >= 30:
                        logging.info(
                            f"ERNIE-Image API reachable but model not loaded (status={status}, {elapsed}/{timeout}s)"
                        )
                        last_log = elapsed
            except requests.exceptions.RequestException:
                if elapsed - last_log >= 30:
                    logging.info(
                        f"ERNIE-Image API not reachable yet ({elapsed}/{timeout}s)"
                    )
                    last_log = elapsed

            time.sleep(5)
        return False

    def _submit_generation(self, inputs: dict) -> Optional[str]:
        try:
            seed = inputs.get("seed")
            width, height = self._resolve_dimensions(inputs.get("aspect_ratio"))
            steps = inputs.get("steps")
            cfg = inputs.get("cfg", 5.0)

            payload = {
                "prompt": self.positive_prompt,
                "negative_prompt": inputs.get("negative_prompt", ""),
                "width": width,
                "height": height,
                "guidance_scale": float(cfg),
            }
            if steps is not None:
                payload["num_inference_steps"] = int(steps)
            if seed is not None:
                payload["seed"] = int(seed)

            response = self.session.post(
                f"{self.api_base_url}/generate", json=payload, timeout=30
            )
            if response.status_code == 200:
                return response.json().get("job_id")
            logging.error(
                f"ERNIE-Image generate failed: {response.status_code} {response.text}"
            )
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"ERNIE-Image generate request error: {e}")
            return None

    def _poll_for_completion(self, api_job_id: str) -> dict:
        start_time = time.time()
        while time.time() - start_time < self.MAX_WAIT_TIME:
            if self.shutdown_event.is_set():
                return {"status": "cancelled", "error": "Shutdown requested"}
            try:
                response = self.session.get(
                    f"{self.api_base_url}/status/{api_job_id}", timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status")
                    if status == "completed":
                        return {"status": "completed"}
                    elif status == "failed":
                        return {
                            "status": "failed",
                            "error": data.get("error", "Generation failed"),
                        }
            except requests.exceptions.RequestException:
                pass
            time.sleep(self.POLL_INTERVAL)
        return {"status": "failed", "error": "Timeout"}

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
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return local_path
            logging.error(
                f"ERNIE-Image download failed: {response.status_code} {response.text}"
            )
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"ERNIE-Image download error: {e}")
            return None

    def _resolve_dimensions(self, aspect_ratio: Optional[str]) -> tuple:
        ratios = {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "9:16": (768, 1344),
            "4:3": (1152, 896),
            "3:4": (896, 1152),
            "3:2": (1216, 832),
            "2:3": (832, 1216),
        }
        return ratios.get(aspect_ratio, (1024, 1024))
