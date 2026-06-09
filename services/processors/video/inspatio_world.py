"""
InSpatio-World Novel-View Video Synthesis Processor (24GB)

Handles video-to-video novel-view synthesis via the InSpatio-World REST API.
Takes an input video + camera trajectory and generates a new video from a
different camera angle using spatiotemporal autoregressive modeling.

Pipeline (runs server-side inside the Docker container):
  Step 1 — Video captioning (Florence-2)
  Step 2 — Depth estimation (DA3) + point cloud rendering
  Step 3 — V2V inference (InSpatio-World 1.3B)
"""

import os
import time
import logging
import requests
from typing import Optional


from services.processors.base import BaseJobProcessor
from services.processors.output_handlers import VideoOutputHandler
from services.processors.rest_recovery import (
    clean_container_exit_detected,
    recover_output_from_clean_container_exit,
)
from config import THUMBNAIL_WIDTH
from utils.media_utils import (
    generate_thumbnail,
    get_video_duration,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("InSpatioWorld")


class InSpatioWorldJobProcessor(BaseJobProcessor, VideoOutputHandler):
    """Processor for novel-view video synthesis with InSpatio-World (24GB, REST API)."""

    API_HOST = os.environ.get("INSPATIO_API_HOST", "127.0.0.1")
    API_PORT = int(os.environ.get("INSPATIO_API_PORT", "8000"))

    POLL_INTERVAL = 5
    MAX_WAIT_TIME = 1800

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.api_base_url = f"http://{self.API_HOST}:{self.API_PORT}"
        self.session = requests.Session()
        self.session.trust_env = False

    def process(self):
        if not self.job:
            self._fail_job("Job object is None. Cannot proceed.")
            return

        log.info(f"Processing InSpatio-World job {self.job_id}")

        input_video_url = self.job.get("input_video_url")
        if not input_video_url:
            self._fail_job(f"Job {self.job_id} missing 'input_video_url'.")
            return

        job_inputs = self.job.get("inputs", {})
        trajectory = job_inputs.get("trajectory", "x_y_circle_cycle")
        rotation_only = bool(job_inputs.get("rotation_only", False))
        relative_to_source = bool(job_inputs.get("relative_to_source", False))
        disable_adaptive_frame = bool(job_inputs.get("disable_adaptive_frame", False))
        freeze_repeat = int(job_inputs.get("freeze_repeat", 0))
        freeze_frame = job_inputs.get("freeze_frame")
        use_tae = bool(job_inputs.get("use_tae", False))

        output_path = None
        task_id = None
        cleanup_task_dir = False

        try:
            if not self._wait_for_api():
                self._fail_job("InSpatio-World API did not become available")
                return

            task_id = self._submit_generation(
                video_url=input_video_url,
                trajectory=trajectory,
                rotation_only=rotation_only,
                relative_to_source=relative_to_source,
                disable_adaptive_frame=disable_adaptive_frame,
                freeze_repeat=freeze_repeat,
                freeze_frame=freeze_frame,
                use_tae=use_tae,
            )
            if not task_id:
                self._fail_job("Failed to submit generation")
                return

            result = self._poll_for_completion(task_id)
            if result.get("status") != "completed":
                cleanup_task_dir = result.get("status") == "failed"
                self._fail_job(
                    f"Generation failed: {result.get('error', 'Unknown error')}"
                )
                return
            cleanup_task_dir = True

            output_path = self._download_output(task_id)
            if not output_path:
                self._fail_job("Failed to download output video")
                return

            video_storage_path = self.orchestrator_service.upload_output(
                output_path, self.job_id, "video/mp4"
            )
            if not video_storage_path:
                self._fail_job("Video upload failed")
                return

            thumbnail_local_path = os.path.join(
                self.cache_dir, f"{self.job_id}_thumb.jpg"
            )
            thumbnail_storage_path = None
            if generate_thumbnail(
                output_path, thumbnail_local_path, width=THUMBNAIL_WIDTH
            ):
                thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(
                    thumbnail_local_path, self.job_id
                )
                if os.path.exists(thumbnail_local_path):
                    os.remove(thumbnail_local_path)

            duration = get_video_duration(output_path)
            completion_metadata = self.job.get("completion_metadata") or {}
            completion_metadata.update(
                {
                    "trajectory": trajectory,
                    "rotation_only": rotation_only,
                    "relative_to_source": relative_to_source,
                    "processor": "InSpatioWorldJobProcessor",
                }
            )

            self.orchestrator_service.update_job_status(
                self.job_id,
                "completed",
                storage_path=video_storage_path,
                thumbnail_storage_path=thumbnail_storage_path,
                duration_seconds=duration,
                completion_metadata=completion_metadata,
            )
            log.info(f"Job {self.job_id} completed successfully")

        except Exception as e:
            if "TokenExpiredError" in type(e).__name__:
                raise
            log.error(f"Error processing job {self.job_id}: {e}", exc_info=True)
            self._fail_job(f"Error processing job: {e}")
        finally:
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            if cleanup_task_dir and task_id:
                self._cleanup_container_file(
                    f"/data/inspatio_tasks/{task_id}",
                    "InSpatio-World task output",
                    recursive=True,
                )

    def _wait_for_api(self, timeout: int = 600) -> bool:
        start = time.monotonic()
        last_log = -30
        log.info(
            f"Waiting for InSpatio-World API at {self.api_base_url} (timeout: {timeout}s)..."
        )

        while time.monotonic() - start < timeout:
            if self.is_cancelled():
                return False
            elapsed = int(time.monotonic() - start)
            try:
                resp = self.session.get(f"{self.api_base_url}/health", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "ok" or data.get("models_initialized"):
                        log.info(f"InSpatio-World API ready after {elapsed}s")
                        return True
            except requests.exceptions.RequestException:
                pass

            if elapsed - last_log >= 30:
                log.info(f"InSpatio-World API not ready ({elapsed}/{timeout}s)")
                last_log = elapsed
            time.sleep(self.POLL_INTERVAL)

        return False

    def _submit_generation(self, **kwargs) -> Optional[str]:

        try:
            resp = self.session.post(
                f"{self.api_base_url}/generate",
                json=kwargs,
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json().get("data", {}).get("task_id")
            log.error(f"Generate failed: {resp.status_code} {resp.text}")
            return None
        except requests.exceptions.RequestException as e:
            log.error(f"Generate request error: {e}")
            return None

    def _poll_for_completion(self, task_id: str) -> dict:
        start = time.time()
        saw_task = False
        consecutive_connection_errors = 0
        while time.time() - start < self.MAX_WAIT_TIME:
            if self.is_cancelled():
                return {"status": "cancelled", "error": "Shutdown requested"}
            try:
                resp = self.session.get(
                    f"{self.api_base_url}/status/{task_id}",
                    timeout=10,
                )
                if resp.status_code == 200:
                    saw_task = True
                    consecutive_connection_errors = 0
                    items = resp.json().get("data", [])
                    if items:
                        item = items[0]
                        status_code = item.get("status")
                        if status_code == 1:
                            return {"status": "completed"}
                        elif status_code == 2:
                            return {
                                "status": "failed",
                                "error": item.get("error", "Generation failed"),
                            }
            except requests.exceptions.RequestException as exc:
                consecutive_connection_errors += 1
                log.warning(f"InSpatio-World status check failed: {exc}")
                if (
                    saw_task
                    and consecutive_connection_errors >= 3
                    and clean_container_exit_detected(self)
                ):
                    log.warning(
                        "InSpatio-World API exited cleanly before final status for "
                        "task %s; attempting output recovery from container.",
                        task_id,
                    )
                    return {"status": "completed", "recovered_from_clean_container_exit": True}
            time.sleep(self.POLL_INTERVAL)
        return {"status": "failed", "error": "Timeout"}

    def _download_output(self, task_id: str) -> Optional[str]:

        try:
            resp = self.session.get(
                f"{self.api_base_url}/download/{task_id}",
                timeout=120,
                stream=True,
            )
            if resp.status_code == 200:
                os.makedirs(self.cache_dir, exist_ok=True)
                output_path = os.path.join(
                    self.cache_dir, f"{self.job_id}_inspatio.mp4"
                )
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return output_path
            log.error(f"Download failed: {resp.status_code} {resp.text}")
            output_path = os.path.join(self.cache_dir, f"{self.job_id}_inspatio.mp4")
            return recover_output_from_clean_container_exit(
                self,
                output_path,
                container_output_path=f"/data/inspatio_tasks/{task_id}/output",
                extensions=(".mp4", ".avi", ".mov", ".mkv"),
                prefer_name=task_id,
            )
        except requests.exceptions.RequestException as e:
            log.error(f"Download error: {e}")
            output_path = os.path.join(self.cache_dir, f"{self.job_id}_inspatio.mp4")
            return recover_output_from_clean_container_exit(
                self,
                output_path,
                container_output_path=f"/data/inspatio_tasks/{task_id}/output",
                extensions=(".mp4", ".avi", ".mov", ".mkv"),
                prefer_name=task_id,
            )
