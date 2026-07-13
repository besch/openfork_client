"""DGN adapter for action-conditioned world-model REST containers."""

import logging
import os
import time
from typing import Optional

import requests

from services.processors.base import BaseJobProcessor
from services.processors.output_handlers import VideoOutputHandler
from utils.media_utils import generate_thumbnail, get_video_duration
from config import THUMBNAIL_WIDTH


class InteractiveWorldJobProcessor(BaseJobProcessor, VideoOutputHandler):
    API_PORT = 8000
    POLL_INTERVAL = 2
    MAX_WAIT_TIME = int(os.environ.get("WORLD_MODEL_GENERATION_TIMEOUT", "3600"))

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.api_base_url = "http://127.0.0.1:%s" % os.environ.get(
            "WORLD_MODEL_API_PORT", self.API_PORT
        )
        self.session = requests.Session()
        self.session.trust_env = False

    def process(self):
        inputs = self.job.get("inputs") or {}
        source_url = self.job.get("input_video_url") or inputs.get("input_video_url")
        if not source_url:
            self._fail_job("Interactive world rollout requires a source scene video")
            return
        actions = inputs.get("action_sequence") or []
        if not actions:
            self._fail_job("Interactive world rollout requires at least one action")
            return

        output_path = None
        try:
            if not self._wait_for_api():
                self._fail_job("World-model API did not become ready")
                return
            response = self.session.post(
                f"{self.api_base_url}/generate",
                json={
                    "source_video_url": source_url,
                    "prompt": self.positive_prompt,
                    "action_sequence": actions,
                    "world_event": inputs.get("world_event"),
                    "seed": inputs.get("seed"),
                },
                timeout=30,
            )
            response.raise_for_status()
            api_job_id = response.json().get("job_id")
            if not api_job_id:
                raise RuntimeError("World-model API returned no job id")
            result = self._poll(api_job_id)
            if result.get("status") != "completed":
                self._fail_job(result.get("error") or "World rollout failed")
                return
            output_path = os.path.join(self.cache_dir, f"{self.job_id}_world.mp4")
            os.makedirs(self.cache_dir, exist_ok=True)
            with self.session.get(
                f"{self.api_base_url}/output/{api_job_id}", stream=True, timeout=180
            ) as output:
                output.raise_for_status()
                with open(output_path, "wb") as handle:
                    for chunk in output.iter_content(1024 * 1024):
                        handle.write(chunk)

            storage_path = self.orchestrator_service.upload_output(
                output_path, self.job_id, "video/mp4"
            )
            thumb_path = os.path.join(self.cache_dir, f"{self.job_id}_world.jpg")
            thumbnail_path = None
            if generate_thumbnail(output_path, thumb_path, width=THUMBNAIL_WIDTH):
                thumbnail_path = self.orchestrator_service.upload_thumbnail(
                    thumb_path, self.job_id
                )
                os.remove(thumb_path)
            metadata = self.job.get("completion_metadata") or {}
            metadata.update(
                {
                    "processor": "InteractiveWorldJobProcessor",
                    "action_sequence": actions,
                    "world_event": inputs.get("world_event"),
                }
            )
            self.orchestrator_service.update_job_status(
                self.job_id,
                "completed",
                storage_path=storage_path,
                thumbnail_storage_path=thumbnail_path,
                duration_seconds=get_video_duration(output_path),
                completion_metadata=metadata,
            )
        except Exception as exc:
            logging.exception("Interactive world job failed")
            self._fail_job(f"Interactive world job failed: {exc}")
        finally:
            if output_path and os.path.exists(output_path):
                os.remove(output_path)

    def _wait_for_api(self, timeout=600):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.is_cancelled():
            try:
                response = self.session.get(f"{self.api_base_url}/health", timeout=5)
                if response.ok and response.json().get("model_loaded"):
                    return True
            except requests.RequestException:
                pass
            time.sleep(5)
        return False

    def _poll(self, api_job_id: str) -> dict:
        deadline = time.monotonic() + self.MAX_WAIT_TIME
        while time.monotonic() < deadline and not self.is_cancelled():
            try:
                response = self.session.get(
                    f"{self.api_base_url}/status/{api_job_id}", timeout=10
                )
                if response.ok:
                    result = response.json()
                    if result.get("status") in ("completed", "failed"):
                        return result
            except requests.RequestException:
                pass
            time.sleep(self.POLL_INTERVAL)
        return {"status": "failed", "error": "World rollout timed out"}
