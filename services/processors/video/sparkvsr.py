"""
SparkVSR Video Super-Resolution Processor (24GB)

Handles video upscaling via the SparkVSR REST API.
SparkVSR uses a Diffusion Transformer architecture built on CogVideoX1.5-5B-I2V
for keyframe-conditioned video super-resolution.

Inference modes:
  - no_ref: Blind SR without reference keyframes (default, simplest)
  - api: Uses external ISR API for keyframe generation  
  - pisasr: Uses PiSA-SR for reference generation

This processor uses no_ref mode for simplicity in the DGN pipeline.
"""

import os
import json
import time
import logging
import requests

from services.processors.base import BaseJobProcessor
from services.processors.output_handlers import VideoOutputHandler
from services.processors.rest_recovery import (
    clean_container_exit_detected,
    recover_output_from_clean_container_exit,
)
from config import THUMBNAIL_WIDTH
from utils.media_utils import (
    find_video_in_output,
    generate_thumbnail,
    get_video_duration,
    get_video_dimensions,
    get_video_framerate,
)


class SparkVSRUpscalerJobProcessor(BaseJobProcessor, VideoOutputHandler):
    """Processor for video upscaling with SparkVSR (24GB, REST API)."""

    HEALTH_ENDPOINT = "http://127.0.0.1:8000/health"
    UPSCALE_ENDPOINT = "http://127.0.0.1:8000/upscale"
    STATUS_ENDPOINT = "http://127.0.0.1:8000/status"
    RESULT_ENDPOINT = "http://127.0.0.1:8000/result"

    def process(self):
        if not self.job:
            self._fail_job("Job object is None for SparkVSRUpscalerJobProcessor. Cannot proceed.")
            return

        # ── 1. Download input video ──
        input_video_url = self.job.get("input_video_url")
        if not input_video_url:
            self._fail_job(f"SparkVSR job {self.job_id} missing 'input_video_url'.")
            return

        video_path = self.orchestrator_service.download_asset_by_url(input_video_url, self.input_dir)
        if not video_path:
            self._fail_job(f"Failed to download input video for SparkVSR job {self.job_id}.")
            return

        logging.info(f"[SparkVSR] Downloaded input video: {video_path}")

        # ── 2. Gather parameters from job inputs ──
        job_inputs = self.job.get("inputs", {})
        upscale_factor = int(job_inputs.get("upscale_factor", 4))
        ref_mode = job_inputs.get("ref_mode", "no_ref")
        ref_guidance_scale = float(job_inputs.get("ref_guidance_scale", 1.0))
        cpu_offload = job_inputs.get("cpu_offload", True)
        chunk_len = int(job_inputs.get("chunk_len", 49))
        tile_size = int(job_inputs.get("tile_size", 0))

        try:
            original_width, original_height = get_video_dimensions(video_path)
            frame_rate = int(job_inputs.get("frame_rate") or get_video_framerate(video_path))
            logging.info(
                f"[SparkVSR] Input: {original_width}x{original_height} @ {frame_rate}fps, "
                f"upscale={upscale_factor}x, ref_mode={ref_mode}"
            )
        except Exception as e:
            self._fail_job(f"Could not analyse input video: {e}")
            return

        # ── 3. Wait for model to finish downloading ──
        model_wait_max = 3600  # 60 minutes (first-run model download can be slow)
        model_wait_elapsed = 0
        model_poll_interval = 15
        while model_wait_elapsed < model_wait_max:
            if self.is_cancelled():
                logging.info("[SparkVSR] Shutdown requested while waiting for model. Aborting.")
                return
            try:
                h = requests.get(self.HEALTH_ENDPOINT, timeout=10)
                if h.status_code == 200 and h.json().get("model_ready"):
                    break
            except Exception:
                pass
            if model_wait_elapsed % 60 == 0:
                logging.info(f"[SparkVSR] Waiting for model to be ready... ({model_wait_elapsed}s elapsed)")
            self.shutdown_event.wait(model_poll_interval)
            model_wait_elapsed += model_poll_interval
        else:
            self._fail_job(f"SparkVSR model was not ready after {model_wait_max}s")
            return

        # ── 4. Submit to SparkVSR REST API ──
        try:
            with open(video_path, "rb") as f:
                files = {"video": (os.path.basename(video_path), f, "video/mp4")}
                data = {
                    "upscale": str(upscale_factor),
                    "ref_mode": ref_mode,
                    "ref_guidance_scale": str(ref_guidance_scale),
                    "cpu_offload": str(cpu_offload).lower(),
                    "chunk_len": str(chunk_len),
                    "tile_size": str(tile_size),
                }
                logging.info(f"[SparkVSR] Submitting to {self.UPSCALE_ENDPOINT}")
                resp = requests.post(self.UPSCALE_ENDPOINT, files=files, data=data, timeout=30)

            if resp.status_code != 200:
                self._fail_job(f"SparkVSR API returned {resp.status_code}: {resp.text}")
                return

            result = resp.json()
            task_id = result.get("task_id")
            if not task_id:
                self._fail_job(f"SparkVSR API did not return a task_id: {result}")
                return

            logging.info(f"[SparkVSR] Task submitted: {task_id}")
        except requests.exceptions.ConnectionError:
            self._fail_job("SparkVSR API is not reachable at port 8000. Is the service running?")
            return
        except Exception as e:
            self._fail_job(f"Failed to submit to SparkVSR API: {e}")
            return

        # ── 5. Poll for completion ──
        max_wait = 1800  # 30 minutes max
        poll_interval = 10
        elapsed = 0
        saw_task = False
        consecutive_connection_errors = 0

        while elapsed < max_wait:
            if self.is_cancelled():
                logging.warning(f"[SparkVSR] Shutdown requested, aborting job {self.job_id}")
                return

            time.sleep(poll_interval)
            elapsed += poll_interval

            try:
                status_resp = requests.get(f"{self.STATUS_ENDPOINT}/{task_id}", timeout=15)
                if status_resp.status_code != 200:
                    logging.warning(f"[SparkVSR] Status check failed: {status_resp.status_code}")
                    continue

                saw_task = True
                consecutive_connection_errors = 0
                status_data = status_resp.json()
                status = status_data.get("status", "unknown")

                if status == "completed":
                    logging.info(f"[SparkVSR] Task {task_id} completed after {elapsed}s")
                    break
                elif status == "failed":
                    error_msg = status_data.get("error", "Unknown error")
                    self._fail_job(f"SparkVSR processing failed: {error_msg}")
                    return
                elif status in ("processing", "pending", "queued"):
                    progress = status_data.get("progress", 0)
                    if elapsed % 60 == 0:
                        logging.info(f"[SparkVSR] Still processing... {progress}% ({elapsed}s elapsed)")
                else:
                    logging.warning(f"[SparkVSR] Unknown status: {status}")

            except Exception as e:
                consecutive_connection_errors += 1
                logging.warning(f"[SparkVSR] Status poll error: {e}")
                if (
                    saw_task
                    and consecutive_connection_errors >= 3
                    and clean_container_exit_detected(self)
                ):
                    logging.warning(
                        "[SparkVSR] API exited cleanly before final status for task %s; "
                        "attempting output recovery from container.",
                        task_id,
                    )
                    break

        else:
            self._fail_job(f"SparkVSR job {self.job_id} timed out after {max_wait}s")
            return

        # ── 6. Download result ──
        try:
            result_resp = requests.get(f"{self.RESULT_ENDPOINT}/{task_id}", timeout=60, stream=True)
            if result_resp.status_code != 200:
                output_path = os.path.join(self.cache_dir, f"{self.job_id}_sparkvsr_output.mp4")
                recovered_path = recover_output_from_clean_container_exit(
                    self,
                    output_path,
                    container_output_path=f"/app/output/{task_id}_out.mp4",
                    extensions=(".mp4",),
                    prefer_name=task_id,
                )
                if not recovered_path:
                    self._fail_job(f"Failed to download SparkVSR result: {result_resp.status_code}")
                    return

                output_path = recovered_path
            else:
                output_path = os.path.join(self.cache_dir, f"{self.job_id}_sparkvsr_output.mp4")
                with open(output_path, "wb") as f:
                    for chunk in result_resp.iter_content(chunk_size=8192):
                        f.write(chunk)

            logging.info(f"[SparkVSR] Downloaded result to {output_path}")

        except Exception as e:
            output_path = os.path.join(self.cache_dir, f"{self.job_id}_sparkvsr_output.mp4")
            recovered_path = recover_output_from_clean_container_exit(
                self,
                output_path,
                container_output_path=f"/app/output/{task_id}_out.mp4",
                extensions=(".mp4",),
                prefer_name=task_id,
            )
            if not recovered_path:
                self._fail_job(f"Failed to download SparkVSR result: {e}")
                return
            output_path = recovered_path

        # ── 7. Upload result and complete job ──
        try:
            video_storage_path = self.orchestrator_service.upload_output(
                output_path, self.job_id, "video/mp4"
            )
            if not video_storage_path:
                self._fail_job(f"SparkVSR job {self.job_id}: video upload failed.")
                return

            # Generate thumbnail
            thumbnail_local_path = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
            thumbnail_storage_path = None

            if generate_thumbnail(output_path, thumbnail_local_path, width=THUMBNAIL_WIDTH):
                thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(
                    thumbnail_local_path, self.job_id
                )
                if os.path.exists(thumbnail_local_path):
                    os.remove(thumbnail_local_path)

            duration = get_video_duration(output_path)
            completion_metadata = self.job.get("completion_metadata") or {}
            completion_metadata["upscale_model"] = "SparkVSR"
            completion_metadata["upscale_factor"] = upscale_factor
            completion_metadata["ref_mode"] = ref_mode

            self.orchestrator_service.update_job_status(
                self.job_id,
                "completed",
                storage_path=video_storage_path,
                thumbnail_storage_path=thumbnail_storage_path,
                duration_seconds=duration,
                completion_metadata=completion_metadata,
            )
            logging.info(f"[SparkVSR] Job {self.job_id} completed successfully")

        finally:
            if os.path.exists(output_path):
                logging.info(f"Cleaning up temporary file: {output_path}")
                os.remove(output_path)
