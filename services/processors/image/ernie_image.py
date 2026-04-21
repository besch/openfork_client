"""
ERNIE-Image Processor

Processor for Baidu ERNIE-Image text-to-image generation.
Communicates with the ERNIE-Image REST API (FastAPI server on port 8000).

The Docker image bakes model weights at build time, so the container
starts the API server immediately and loads the model in a background
thread. The _wait_for_api method polls /health until model_loaded=true.
"""

import os
import time
import logging
import requests
from typing import Optional

from config import TimeoutConfig
from services.processors.base import BaseJobProcessor
from services.orchestrator_service import TokenExpiredError


class ErnieImageProcessor(BaseJobProcessor):
    """
    Job processor for ERNIE-Image text-to-image generation via REST API.

    Endpoints:
      POST /generate    - submit generation, returns job_id
      GET  /status/{id} - poll status
      GET  /output/{id} - download output image
      GET  /health      - health check (model_loaded bool)
    """

    API_HOST = "127.0.0.1"
    API_PORT = 8000
    POLL_INTERVAL = 3       # seconds between status polls after generation starts
    # How long to wait for the model to finish loading after the container starts.
    # With weights baked into the image this is typically 30-120s (just model load,
    # no download). Set conservatively in case the host GPU is slow to initialise.
    API_WAIT_TIMEOUT = int(os.environ.get("ERNIE_IMAGE_API_WAIT_TIMEOUT", "300"))
    # Generation on lower-VRAM GPUs can take much longer than 10-30 minutes even
    # when the API is healthy and the GPU is fully utilized, so allow a
    # processor-specific override and otherwise fall back to a 2-hour ceiling,
    # matching the more permissive long-running ComfyUI path.
    MAX_WAIT_TIME = int(
        os.environ.get(
            "ERNIE_IMAGE_GENERATION_TIMEOUT",
            str(max(TimeoutConfig.WORKFLOW_TIMEOUT, 7200)),
        )
    )

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

        if not self._wait_for_api(timeout=self.API_WAIT_TIMEOUT):
            self._fail_job("ERNIE-Image API / model did not become ready in time")
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

    def _wait_for_api(self, timeout: int = 300) -> bool:
        """
        Poll /health until the model reports model_loaded=true.

        Because the model weights are baked into the image, the server starts
        almost immediately and only needs time to load weights into GPU memory
        (typically 30-120s). We distinguish between three phases and log each:
          1. Server not yet reachable (container still starting)
          2. Server reachable but model still loading
          3. Model loaded and ready
        """
        start_time = time.monotonic()
        last_log = -31  # force an immediate first log

        logging.info(
            f"Waiting for ERNIE-Image API at {self.api_base_url} (timeout: {timeout}s)..."
        )

        while time.monotonic() - start_time < timeout:
            if self.shutdown_event.is_set():
                logging.warning("Shutdown requested while waiting for API.")
                return False

            elapsed = int(time.monotonic() - start_time)

            try:
                response = self.session.get(
                    f"{self.api_base_url}/health", timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("model_loaded"):
                        logging.info(
                            f"ERNIE-Image API ready after {elapsed}s "
                            f"(model: {data.get('model_id', 'unknown')})"
                        )
                        return True

                    # Server is up but model is still loading — log every 30s
                    if elapsed - last_log >= 30:
                        api_status = data.get("status", "unknown")
                        api_error = data.get("error")
                        msg = (
                            f"ERNIE-Image API reachable, model loading "
                            f"(status={api_status}, {elapsed}/{timeout}s)"
                        )
                        if api_error:
                            msg += f" | error: {api_error}"
                        logging.info(msg)
                        last_log = elapsed

            except requests.exceptions.RequestException:
                # Container may still be starting — log every 30s
                if elapsed - last_log >= 30:
                    logging.info(
                        f"ERNIE-Image API not reachable yet ({elapsed}/{timeout}s)"
                    )
                    last_log = elapsed

            time.sleep(5)

        logging.error(
            f"ERNIE-Image API did not become ready within {timeout}s"
        )
        return False

    def _submit_generation(self, inputs: dict) -> Optional[str]:
        try:
            seed = inputs.get("seed")
            width, height = self._resolve_dimensions(inputs.get("aspect_ratio"))
            steps = inputs.get("steps")
            cfg = inputs.get("cfg")
            if cfg is None:
                cfg = inputs.get("cfg_scale")
            # Turbo is distilled for CFG=1.0. Higher CFG doubles denoising work in
            # the Diffusers pipeline and is especially punishing on the 8GB tier.
            if cfg is not None and self.workflow_type and "8gb" in self.workflow_type:
                try:
                    if float(cfg) != 1.0:
                        logging.warning(
                            "Ignoring ERNIE-Image Turbo CFG=%s for 8GB tier; using CFG=1.0",
                            cfg,
                        )
                    cfg = 1.0
                except (TypeError, ValueError):
                    cfg = 1.0

            payload = {
                "prompt": self.positive_prompt,
                "negative_prompt": inputs.get("negative_prompt", ""),
                "width": width,
                "height": height,
                "guidance_scale": float(cfg) if cfg is not None else -1.0,
            }
            if steps is not None:
                payload["num_inference_steps"] = int(steps)
            if seed is not None:
                payload["seed"] = int(seed)

            # 8GB tier: always disable PE regardless of job input. The Mistral3
            # prompt enhancer consumes ~3GB of VRAM during text encoding and
            # significantly extends that phase — not worth it on this tier.
            if self.workflow_type and "8gb" in self.workflow_type:
                if inputs.get("use_pe"):
                    logging.warning(
                        "8GB tier: overriding use_pe=True to False (insufficient VRAM)"
                    )
                payload["use_pe"] = False
            elif inputs.get("use_pe") is not None:
                payload["use_pe"] = bool(inputs.get("use_pe"))

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
        last_log = -30
        while time.time() - start_time < self.MAX_WAIT_TIME:
            if self.shutdown_event.is_set():
                return {"status": "cancelled", "error": "Shutdown requested"}
            elapsed = int(time.time() - start_time)
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
                    elif elapsed - last_log >= 30:
                        logging.info(
                            "ERNIE-Image job %s still %s (%ss/%ss)",
                            api_job_id,
                            status or "processing",
                            elapsed,
                            self.MAX_WAIT_TIME,
                        )
                        last_log = elapsed
            except requests.exceptions.RequestException:
                pass
            time.sleep(self.POLL_INTERVAL)
        return {
            "status": "failed",
            "error": f"Timeout waiting for generation after {self.MAX_WAIT_TIME}s",
        }

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
        # Standard resolutions — suitable for 16GB+ VRAM.
        ratios = {
            "1:1":  (1024, 1024),
            "16:9": (1344, 768),
            "9:16": (768, 1344),
            "4:3":  (1152, 896),
            "3:4":  (896, 1152),
            "3:2":  (1216, 832),
            "2:3":  (832, 1216),
        }
        # 8GB tier: use the smallest ERNIE-supported non-square resolutions so
        # the denoising loop stays well within 8GB even in fp8 with cpu_offload.
        ratios_8gb = {
            "1:1":  (1024, 1024),
            "16:9": (1264, 848),
            "9:16": (848, 1264),
            "4:3":  (1024, 768),
            "3:4":  (768, 1024),
            "3:2":  (1024, 768),
            "2:3":  (768, 1024),
        }
        table = ratios_8gb if (self.workflow_type and "8gb" in self.workflow_type) else ratios
        return table.get(aspect_ratio, (1024, 1024))