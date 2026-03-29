"""
daVinci-MagiHuman Processors

Two processors that communicate with the daVinci-MagiHuman REST API server:

  DaVinciMagiHumanT2VProcessor — text-to-video (prompt → avatar video + audio)
  DaVinciMagiHumanI2VProcessor — image-to-video (reference image + prompt → avatar)

Both inherit from TurboDiffusionBaseProcessor which provides:
  - _wait_for_api()          : polls /health until ready
  - _poll_for_completion()   : polls /status/{job_id}
  - _download_output()       : streams video from /download/{job_id}
  - _finalize_job()          : uploads to storage + updates job status
  - _cleanup_remote_job()    : calls DELETE /job/{job_id}

daVinci-MagiHuman specifics:
  - Backend: REST API on port 8000 (started by start_cloud.sh)
  - Distilled model: 8 fixed steps, no CFG
  - Both T2V (JSON body) and I2V (multipart upload + form fields)
  - Output: MP4 with embedded audio track
"""

import logging
import os
from typing import Optional

import requests

from services.processors.video.turbodiffusion import TurboDiffusionBaseProcessor
from utils.comfyui_workflow_utils import materialize_start_image
from config import SUPABASE_URL, DEV_MODE

logger = logging.getLogger(__name__)

# Resolution lookup: aspect_ratio → (height, width) for 256p distilled baseline
_ASPECT_RESOLUTIONS = {
    "16:9": (256, 448),
    "9:16": (448, 256),
    "1:1":  (256, 256),
    "4:3":  (256, 320),
    "3:4":  (320, 256),
    "21:9": (176, 448),
    "2:1":  (224, 448),
}

_DEFAULT_STEPS = 8        # distilled — do not change
_DEFAULT_DURATION = 5.0   # seconds


def _aspect_to_hw(aspect_ratio: str) -> tuple[int, int]:
    return _ASPECT_RESOLUTIONS.get(aspect_ratio, (256, 448))


class DaVinciMagiHumanBaseProcessor(TurboDiffusionBaseProcessor):
    """Shared base for daVinci-MagiHuman T2V & I2V processors."""

    SERVICE_NAME = "daVinci-MagiHuman"
    # Allow up to 20 minutes per job (model load + compilation + generation)
    MAX_WAIT_TIME = 1200
    POLL_INTERVAL = 10

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.api_base_url = "http://localhost:8000"

    def _wait_for_api(self, timeout: int = 900) -> bool:
        """Wait up to 15 minutes for the API to become available (model load is slow)."""
        import time
        start = time.time()
        logger.info(
            "Waiting for %s API at %s (timeout=%ds)…",
            self.SERVICE_NAME, self.api_base_url, timeout,
        )
        while time.time() - start < timeout:
            if self.shutdown_event.is_set():
                return False
            try:
                resp = requests.get(f"{self.api_base_url}/health", timeout=10)
                if resp.status_code == 200:
                    logger.info("%s API is ready", self.SERVICE_NAME)
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(15)
        logger.error(
            "%s API not available after %ds at %s",
            self.SERVICE_NAME, timeout, self.api_base_url,
        )
        return False


class DaVinciMagiHumanT2VProcessor(DaVinciMagiHumanBaseProcessor):
    """daVinci-MagiHuman text-to-video processor."""

    def process(self):
        if DEV_MODE:
            return

        if not self.job:
            self._fail_job("Job object is None for DaVinciMagiHumanT2VProcessor.")
            return

        logger.info("Processing daVinci-MagiHuman T2V job %s", self.job_id)

        if not self._wait_for_api():
            self._fail_job(
                f"daVinci-MagiHuman API not available for job {self.job_id}"
            )
            return

        inputs = self.job.get("inputs") or {}
        duration_seconds = float(inputs.get("duration", _DEFAULT_DURATION))
        duration_seconds = max(3.0, min(duration_seconds, 10.0))
        aspect_ratio = inputs.get("aspect_ratio", "16:9")
        height, width = _aspect_to_hw(aspect_ratio)
        seed = int(inputs.get("seed", 0))

        remote_job_id = self._submit_t2v(duration_seconds, height, width, seed)
        if not remote_job_id:
            self._fail_job(f"Failed to submit T2V to daVinci API for job {self.job_id}")
            return

        result = self._poll_for_completion(remote_job_id)
        if result.get("status") != "completed":
            self._fail_job(
                f"daVinci T2V failed: {result.get('error', 'Unknown error')}"
            )
            return

        local_path = self._download_output(remote_job_id)
        if not local_path:
            self._fail_job(f"Failed to download daVinci output for job {self.job_id}")
            return

        self._finalize_job(local_path, remote_job_id)

    def _submit_t2v(self, duration_seconds: float, height: int, width: int, seed: int) -> Optional[str]:
        try:
            payload = {
                "prompt": self.positive_prompt,
                "negative_prompt": self.negative_prompt or "",
                "duration_seconds": duration_seconds,
                "height": height,
                "width": width,
                "num_steps": _DEFAULT_STEPS,
                "seed": seed,
                "use_distilled": True,
            }
            resp = requests.post(
                f"{self.api_base_url}/generate/t2v", json=payload, timeout=60
            )
            if resp.status_code == 200:
                remote_job_id = resp.json().get("job_id")
                logger.info("daVinci T2V job submitted: %s", remote_job_id)
                return remote_job_id
            logger.error("daVinci T2V submission failed: %s — %s", resp.status_code, resp.text)
            return None
        except requests.exceptions.RequestException as exc:
            logger.error("daVinci T2V request error: %s", exc)
            return None


class DaVinciMagiHumanI2VProcessor(DaVinciMagiHumanBaseProcessor):
    """daVinci-MagiHuman image-to-video (avatar) processor."""

    def process(self):
        if DEV_MODE:
            return

        if not self.job:
            self._fail_job("Job object is None for DaVinciMagiHumanI2VProcessor.")
            return

        logger.info("Processing daVinci-MagiHuman I2V job %s", self.job_id)

        inputs = self.job.get("inputs") or {}

        # ── Resolve reference image ──────────────────────────────────────────
        image_path = self._resolve_reference_image(inputs)
        if not image_path:
            self._fail_job(
                f"Could not resolve reference image for job {self.job_id}"
            )
            return

        # ── Wait for API ─────────────────────────────────────────────────────
        if not self._wait_for_api():
            self._fail_job(
                f"daVinci-MagiHuman API not available for job {self.job_id}"
            )
            return

        duration_seconds = float(inputs.get("duration", _DEFAULT_DURATION))
        duration_seconds = max(3.0, min(duration_seconds, 10.0))
        aspect_ratio = inputs.get("aspect_ratio", "16:9")
        height, width = _aspect_to_hw(aspect_ratio)
        seed = int(inputs.get("seed", 0))

        remote_job_id = self._submit_i2v(
            image_path, duration_seconds, height, width, seed
        )
        if not remote_job_id:
            self._fail_job(
                f"Failed to submit I2V to daVinci API for job {self.job_id}"
            )
            return

        result = self._poll_for_completion(remote_job_id)
        if result.get("status") != "completed":
            self._fail_job(
                f"daVinci I2V failed: {result.get('error', 'Unknown error')}"
            )
            return

        local_path = self._download_output(remote_job_id)
        if not local_path:
            self._fail_job(
                f"Failed to download daVinci output for job {self.job_id}"
            )
            return

        self._finalize_job(local_path, remote_job_id)

    def _submit_i2v(
        self,
        image_path: str,
        duration_seconds: float,
        height: int,
        width: int,
        seed: int,
    ) -> Optional[str]:
        try:
            with open(image_path, "rb") as f:
                files = {
                    "image": (os.path.basename(image_path), f, "image/jpeg")
                }
                data = {
                    "prompt": self.positive_prompt,
                    "negative_prompt": self.negative_prompt or "",
                    "duration_seconds": str(duration_seconds),
                    "height": str(height),
                    "width": str(width),
                    "num_steps": str(_DEFAULT_STEPS),
                    "seed": str(seed),
                    "use_distilled": "true",
                }
                resp = requests.post(
                    f"{self.api_base_url}/generate/i2v",
                    data=data,
                    files=files,
                    timeout=90,
                )
            if resp.status_code == 200:
                remote_job_id = resp.json().get("job_id")
                logger.info("daVinci I2V job submitted: %s", remote_job_id)
                return remote_job_id
            logger.error(
                "daVinci I2V submission failed: %s — %s", resp.status_code, resp.text
            )
            return None
        except requests.exceptions.RequestException as exc:
            logger.error("daVinci I2V request error: %s", exc)
            return None

    def _resolve_reference_image(self, inputs: dict) -> Optional[str]:
        """Return a local file path for the reference image, trying multiple sources."""
        # 1. Signed URL from inputs
        url = inputs.get("start_image_url")
        if url:
            path = self.orchestrator_service.download_asset_by_url(url, self.input_dir)
            if path:
                return path

        # 2. materialize_start_image handles base64 / embedded payloads
        filename = materialize_start_image(self.job, self.input_dir)
        if filename:
            return os.path.join(self.input_dir, filename)

        # 3. Supabase storage path
        storage_path = self.job.get("input_storage_path")
        if not storage_path:
            maybe = self.job.get("start_image_base64")
            if (
                maybe
                and isinstance(maybe, str)
                and not maybe.startswith("data:")
                and len(maybe) < 2048
            ):
                storage_path = maybe

        if storage_path:
            supabase_url = os.environ.get("SUPABASE_URL", SUPABASE_URL)
            if supabase_url:
                bucket = self.job.get("bucket", "projects_public")
                src = f"{supabase_url}/storage/v1/object/public/{bucket}/{storage_path}"
                path = self.orchestrator_service.download_asset_by_url(
                    src, self.input_dir
                )
                if path:
                    return path

        return None
