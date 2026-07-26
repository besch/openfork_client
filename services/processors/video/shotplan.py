"""DGN processor for ShotPlan frame-accurate multi-shot generation."""

from __future__ import annotations

import logging
import os
import time

import requests

from config import THUMBNAIL_WIDTH, TimeoutConfig
from services.processors.base import BaseJobProcessor
from services.processors.output_handlers import VideoOutputHandler
from services.processors.rest_recovery import poll_rest_job_with_clean_exit
from utils.media_utils import generate_thumbnail, get_video_duration, get_video_probe_metadata

log = logging.getLogger(__name__)


class ShotPlanVideoProcessor(BaseJobProcessor, VideoOutputHandler):
    API_HOST = os.getenv("SHOTPLAN_API_HOST", "127.0.0.1")
    API_PORT = int(os.getenv("SHOTPLAN_API_PORT", "8000"))
    API_WAIT_TIMEOUT = int(os.getenv("SHOTPLAN_API_WAIT_TIMEOUT", "600"))
    MAX_WAIT_TIME = int(
        os.getenv(
            "SHOTPLAN_GENERATION_TIMEOUT",
            str(max(TimeoutConfig.WORKFLOW_TIMEOUT, 14400)),
        )
    )

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.api_base_url = f"http://{self.API_HOST}:{self.API_PORT}"
        self.session = requests.Session()
        self.session.trust_env = False

    def process(self):
        inputs = self.job.get("inputs") or {}
        output_path = None
        thumbnail_path = None
        try:
            if not self._wait_for_api():
                self._fail_job("ShotPlan API did not become ready")
                return
            cut_frames = self._cut_frames(inputs)
            if not cut_frames:
                self._fail_job("ShotPlan requires at least one hard-cut frame")
                return
            base_prompt = (
                self.job.get("prompt") or inputs.get("prompt") or self.positive_prompt
            )
            shot_plan = inputs.get("shot_plan") or []
            shot_prompts = [
                str(shot.get("prompt") or "").strip()
                for shot in shot_plan
                if isinstance(shot, dict) and str(shot.get("prompt") or "").strip()
            ]
            runtime_prompt = " ".join(
                [
                    base_prompt.strip(),
                    *[
                        f"Shot {index + 1}: {shot_prompt}"
                        for index, shot_prompt in enumerate(shot_prompts)
                    ],
                ]
            )
            payload = {
                "prompt": runtime_prompt,
                "cut_frames": cut_frames,
                "negative_prompt": inputs.get("negative_prompt") or self.negative_prompt or "",
                "width": 832,
                "height": 480,
                "num_frames": 81,
                "steps": max(1, min(100, int(inputs.get("steps") or 50))),
                "seed": self.normalize_seed(inputs.get("seed", 42)),
            }
            response = self.session.post(
                f"{self.api_base_url}/generate", json=payload, timeout=60
            )
            if response.status_code != 200:
                self._fail_job(f"ShotPlan submit failed: {response.text[:1000]}")
                return
            remote_job_id = response.json().get("job_id")
            result = poll_rest_job_with_clean_exit(
                self,
                self.api_base_url,
                remote_job_id,
                poll_interval=10,
                max_wait_time=self.MAX_WAIT_TIME,
                service_label="ShotPlan",
                session=self.session,
            )
            if result.get("status") != "completed":
                self._fail_job(
                    f"ShotPlan generation failed: {result.get('error', 'Unknown error')}"
                )
                return
            output_path = os.path.join(self.cache_dir, f"{self.job_id}_shotplan.mp4")
            os.makedirs(self.cache_dir, exist_ok=True)
            download = self.session.get(
                f"{self.api_base_url}/output/{remote_job_id}",
                timeout=300,
                stream=True,
            )
            download.raise_for_status()
            with open(output_path, "wb") as handle:
                for chunk in download.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            storage_path = self.orchestrator_service.upload_output(
                output_path, self.job_id, "video/mp4"
            )
            if not storage_path:
                self._fail_job("ShotPlan video upload failed")
                return
            thumbnail_path = os.path.join(self.cache_dir, f"{self.job_id}_shotplan.jpg")
            thumbnail_storage_path = None
            if generate_thumbnail(output_path, thumbnail_path, width=THUMBNAIL_WIDTH):
                thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(
                    thumbnail_path, self.job_id
                )
            duration = get_video_duration(output_path)
            metadata = self.job.get("completion_metadata") or {}
            metadata.update(
                {
                    "processor": "ShotPlanVideoProcessor",
                    "model": "ShotPlan-Wan2.2-T2V-A14B-HighNoise",
                    "cut_frames": cut_frames,
                    "shot_count": len(cut_frames) + 1,
                    "frame_count": 81,
                    "fps": 16,
                    "transition_type": "hard_cut",
                    **get_video_probe_metadata(output_path, duration),
                }
            )
            self.orchestrator_service.update_job_status(
                self.job_id,
                "completed",
                storage_path=storage_path,
                thumbnail_storage_path=thumbnail_storage_path,
                duration_seconds=duration,
                prompt=payload["prompt"],
                completion_metadata=metadata,
            )
        except Exception as exc:
            if "TokenExpiredError" in type(exc).__name__:
                raise
            log.exception("ShotPlan job %s failed", self.job_id)
            self._fail_job(f"ShotPlan processing error: {exc}")
        finally:
            for path in (output_path, thumbnail_path):
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
            except requests.RequestException:
                pass
            self.shutdown_event.wait(5)
        return False

    @staticmethod
    def _cut_frames(inputs: dict) -> list[int]:
        value = inputs.get("cut_frames") or inputs.get("cut_at") or []
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",")]
        frames = []
        for item in value:
            try:
                frame = int(item)
            except (TypeError, ValueError):
                continue
            if 0 < frame < 80:
                frames.append(frame)
        return sorted(set(frames))[:5]
