"""DGN processor for HOMIE multi-subject and multi-view reference video."""

from __future__ import annotations

import json
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


class HomieVideoProcessor(BaseJobProcessor, VideoOutputHandler):
    API_HOST = os.getenv("HOMIE_API_HOST", "127.0.0.1")
    API_PORT = int(os.getenv("HOMIE_API_PORT", "8000"))
    API_WAIT_TIMEOUT = int(os.getenv("HOMIE_API_WAIT_TIMEOUT", "600"))
    MAX_WAIT_TIME = int(
        os.getenv(
            "HOMIE_GENERATION_TIMEOUT",
            str(max(TimeoutConfig.WORKFLOW_TIMEOUT, 18000)),
        )
    )

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.api_base_url = f"http://{self.API_HOST}:{self.API_PORT}"
        self.session = requests.Session()
        self.session.trust_env = False

    def process(self):
        inputs = self.job.get("inputs") or {}
        paths: list[str] = []
        output_path = None
        thumbnail_path = None
        try:
            if not self._wait_for_api():
                self._fail_job("HOMIE API did not become ready")
                return
            groups, paths = self._materialize_groups(inputs)
            if not groups:
                self._fail_job("HOMIE requires at least one non-empty reference group")
                return
            payload = {
                "prompt": self.job.get("prompt") or inputs.get("prompt") or self.positive_prompt,
                "group_sizes": [len(group) for group in groups],
                "size": self._size(inputs),
                "fps": 24,
                "steps": int(inputs.get("steps") or 50),
                "flow_shift": float(inputs.get("flow_shift") or 3),
                "cfg": float(inputs.get("cfg_scale") or 5),
                "seed": self.normalize_seed(inputs.get("seed", 6666)),
            }
            handles = []
            files = []
            try:
                for path in paths:
                    handle = open(path, "rb")
                    handles.append(handle)
                    files.append(
                        (
                            "reference_images",
                            (os.path.basename(path), handle, "application/octet-stream"),
                        )
                    )
                response = self.session.post(
                    f"{self.api_base_url}/generate",
                    data={"payload": json.dumps(payload)},
                    files=files,
                    timeout=180,
                )
            finally:
                for handle in handles:
                    handle.close()
            if response.status_code != 200:
                self._fail_job(f"HOMIE submit failed: {response.text[:1000]}")
                return
            remote_job_id = response.json().get("job_id")
            result = poll_rest_job_with_clean_exit(
                self,
                self.api_base_url,
                remote_job_id,
                poll_interval=10,
                max_wait_time=self.MAX_WAIT_TIME,
                service_label="HOMIE",
                session=self.session,
            )
            if result.get("status") != "completed":
                self._fail_job(
                    f"HOMIE generation failed: {result.get('error', 'Unknown error')}"
                )
                return
            output_path = os.path.join(self.cache_dir, f"{self.job_id}_homie.mp4")
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
                self._fail_job("HOMIE video upload failed")
                return
            thumbnail_path = os.path.join(self.cache_dir, f"{self.job_id}_homie.jpg")
            thumbnail_storage_path = None
            if generate_thumbnail(output_path, thumbnail_path, width=THUMBNAIL_WIDTH):
                thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(
                    thumbnail_path, self.job_id
                )
            duration = get_video_duration(output_path)
            metadata = self.job.get("completion_metadata") or {}
            metadata.update(
                {
                    "processor": "HomieVideoProcessor",
                    "model": "HOMIE-Wan2.1-14B",
                    "subject_count": len(groups),
                    "reference_image_count": len(paths),
                    "reference_group_sizes": [len(group) for group in groups],
                    "frame_count": 97,
                    "fps": 24,
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
            log.exception("HOMIE job %s failed", self.job_id)
            self._fail_job(f"HOMIE processing error: {exc}")
        finally:
            for path in [*paths, output_path, thumbnail_path]:
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

    def _materialize_groups(self, inputs: dict) -> tuple[list[list[str]], list[str]]:
        raw_groups = inputs.get("homie_reference_groups") or []
        if not raw_groups:
            primary = (
                inputs.get("input_storage_path")
                or self.job.get("input_storage_path")
                or inputs.get("start_image")
            )
            raw_groups = [{"imageStoragePaths": [primary]}] if primary else []
        bucket = inputs.get("bucket") or self.job.get("bucket") or "projects_public"
        groups: list[list[str]] = []
        all_paths: list[str] = []
        for group in raw_groups[:8]:
            if not isinstance(group, dict):
                continue
            values = group.get("imageStoragePaths") or group.get("image_storage_paths") or []
            if isinstance(values, str):
                values = [values]
            ocr = group.get("ocrMapStoragePath") or group.get("ocr_map_storage_path")
            if ocr:
                values = [*values, ocr]
            paths = []
            for value in values[:9]:
                if not isinstance(value, str) or not value:
                    continue
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
                    all_paths.append(path)
            if paths:
                groups.append(paths)
        return groups, all_paths

    @staticmethod
    def _size(inputs: dict) -> str:
        resolution = str(inputs.get("resolution") or "").lower().replace("×", "x")
        width = inputs.get("target_width") or inputs.get("width")
        height = inputs.get("target_height") or inputs.get("height")
        if resolution in {"1280x720", "720p"} or (width == 1280 and height == 720):
            return "1280*720"
        return "832*480"
