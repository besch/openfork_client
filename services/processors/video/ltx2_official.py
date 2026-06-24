"""Official LTX-2 Trainer and pipelines processors.

These processors intentionally avoid the existing LTX-2.3 Wan2GP stack. They
talk to small FastAPI wrappers around Lightricks' official `ltx-trainer` and
`ltx-pipelines` packages.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests

from config import SUPABASE_URL, THUMBNAIL_WIDTH, TimeoutConfig
from services.orchestrator_service import TokenExpiredError
from services.processors.base import BaseJobProcessor
from services.processors.output_handlers import VideoOutputHandler
from services.processors.rest_recovery import (
    poll_rest_job_with_clean_exit,
    recover_output_from_clean_container_exit,
)
from services.processors.video.last_frame import materialize_last_frame_start_image
from utils.comfyui_workflow_utils import materialize_start_image
from utils.media_utils import generate_thumbnail, get_video_duration

DEFAULT_TRAINER_BUCKETS = os.environ.get("LTX2_TRAINER_RESOLUTION_BUCKETS", "512x288x49")
DEFAULT_TRAINER_STEPS = int(os.environ.get("LTX2_TRAINER_STEPS", "2000"))
DEFAULT_TRAINER_RANK = int(os.environ.get("LTX2_TRAINER_RANK", "16"))
DEFAULT_TRAINER_ALPHA = int(os.environ.get("LTX2_TRAINER_ALPHA", str(DEFAULT_TRAINER_RANK)))
DEFAULT_TRAINER_QUANTIZATION = os.environ.get("LTX2_TRAINER_QUANTIZATION", "int8-quanto")
DEFAULT_TARGET_VRAM_GB = int(os.environ.get("LTX2_TARGET_VRAM_GB", "32"))

DEFAULT_PIPELINES_WIDTH = int(os.environ.get("LTX2_PIPELINES_DEFAULT_WIDTH", "1024"))
DEFAULT_PIPELINES_HEIGHT = int(os.environ.get("LTX2_PIPELINES_DEFAULT_HEIGHT", "576"))
DEFAULT_PIPELINES_FRAMES = int(os.environ.get("LTX2_PIPELINES_DEFAULT_NUM_FRAMES", "49"))
DEFAULT_PIPELINES_STEPS = int(os.environ.get("LTX2_PIPELINES_DEFAULT_STEPS", "30"))
DEFAULT_PIPELINES_OFFLOAD = os.environ.get("LTX2_PIPELINES_DEFAULT_OFFLOAD", "cpu")
DEFAULT_PIPELINES_QUANTIZATION = os.environ.get("LTX2_PIPELINES_DEFAULT_QUANTIZATION", "fp8-cast")


def _as_int(value: Any, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _as_float(
    value: Any,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _looks_like_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _safe_name(value: Any, fallback: str) -> str:
    raw = str(value or fallback).strip() or fallback
    name = os.path.basename(raw.replace("\\", "/"))
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    safe = safe.strip("._") or fallback
    return safe[:120]


class LTX2OfficialBaseProcessor(BaseJobProcessor):
    API_HOST = "127.0.0.1"
    API_PORT = 8000
    API_WAIT_TIMEOUT = int(os.environ.get("LTX2_API_WAIT_TIMEOUT", "1800"))
    POLL_INTERVAL = 10

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        api_host = os.environ.get("LTX2_API_HOST", self.API_HOST)
        api_port = int(os.environ.get("LTX2_API_PORT", self.API_PORT))
        self.api_base_url = f"http://{api_host}:{api_port}"
        self.session = requests.Session()
        self.session.trust_env = False
        self._api_startup_error: Optional[str] = None

    def _wait_for_api(self, timeout: int | None = None) -> bool:
        timeout = timeout or self.API_WAIT_TIMEOUT
        start = time.monotonic()
        last_log = -31
        self._api_startup_error = None

        while time.monotonic() - start < timeout:
            if self.is_cancelled():
                return False
            elapsed = int(time.monotonic() - start)
            try:
                response = self.session.get(f"{self.api_base_url}/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "error":
                        self._api_startup_error = data.get("error") or "LTX-2 API reported an error"
                        logging.error("LTX-2 API error: %s", data)
                        return False
                    if data.get("model_loaded", True):
                        logging.info("LTX-2 API ready after %ss", elapsed)
                        return True
                    if elapsed - last_log >= 30:
                        logging.info("LTX-2 API reachable, model loading (%s/%ss): %s", elapsed, timeout, data)
                        last_log = elapsed
            except requests.exceptions.RequestException:
                if elapsed - last_log >= 30:
                    logging.info("LTX-2 API not reachable yet (%s/%ss)", elapsed, timeout)
                    last_log = elapsed
            time.sleep(5)
        return False

    def _poll_for_completion(self, api_job_id: str, *, max_wait_time: int, label: str) -> dict:
        return poll_rest_job_with_clean_exit(
            self,
            self.api_base_url,
            api_job_id,
            poll_interval=self.POLL_INTERVAL,
            max_wait_time=max_wait_time,
            service_label=label,
            session=self.session,
        )

    def _cleanup_remote_job(self, api_job_id: str) -> None:
        try:
            self.session.delete(f"{self.api_base_url}/job/{api_job_id}", timeout=10)
        except requests.exceptions.RequestException:
            pass

    def _download_storage_or_url(
        self,
        value: str,
        dest_dir: str,
        *,
        bucket: Optional[str] = None,
    ) -> Optional[str]:
        if not value:
            return None
        if _looks_like_url(value):
            return self.orchestrator_service.download_asset_by_url(value, dest_dir)
        return self.orchestrator_service.download_storage_asset(
            bucket or self.job.get("bucket", "projects_public"),
            value,
            dest_dir,
        )


class LTX2TrainerI2VLoraProcessor(LTX2OfficialBaseProcessor):
    """Train an official LTX-2 I2V LoRA and upload the trainer output package."""

    MAX_WAIT_TIME = int(os.environ.get("LTX2_TRAINER_TIMEOUT", str(18 * 60 * 60)))

    def process(self):
        if not self.job:
            self._fail_job("Job object is None for LTX2TrainerI2VLoraProcessor.")
            return

        logging.info("Processing official LTX-2 trainer job %s", self.job_id)
        if not self._wait_for_api(int(os.environ.get("LTX2_TRAINER_API_WAIT_TIMEOUT", "1800"))):
            self._fail_job(self._api_startup_error or "LTX-2 trainer API did not become ready")
            return

        inputs = self.job.get("inputs") or {}
        archive_path = self._resolve_dataset_archive(inputs)
        if not archive_path:
            self._fail_job(
                "No dataset archive or training clips were provided. Supply "
                "dataset_archive_storage_path/dataset_archive_url or training_clips."
            )
            return

        api_job_id = self._submit_training(archive_path, inputs)
        if not api_job_id:
            self._fail_job("Failed to submit official LTX-2 trainer job")
            return

        local_path: Optional[str] = None
        try:
            result = self._poll_for_completion(
                api_job_id,
                max_wait_time=self.MAX_WAIT_TIME,
                label="LTX-2 trainer",
            )
            if result.get("status") != "completed":
                self._fail_job(f"LTX-2 trainer failed: {result.get('error', 'Unknown error')}")
                return

            local_path = self._download_output(api_job_id)
            if not local_path:
                self._fail_job("Failed to download LTX-2 trainer output")
                return

            storage_path = self.orchestrator_service.upload_output(
                local_path,
                self.job_id,
                "application/zip",
            )
            if not storage_path:
                self._fail_job("LTX-2 trainer completed, but LoRA package upload failed")
                return

            metadata = self.job.get("completion_metadata") or {}
            metadata.update(
                {
                    "processor": "LTX2TrainerI2VLoraProcessor",
                    "model": "official-ltx-2.3-22b-dev",
                    "artifact_kind": "ltx2_lora_package",
                    "ltx_lora_trigger": inputs.get("lora_trigger") or inputs.get("trigger"),
                    "training_steps": _as_int(inputs.get("steps"), DEFAULT_TRAINER_STEPS, 1),
                    "resolution_buckets": inputs.get("resolution_buckets") or DEFAULT_TRAINER_BUCKETS,
                    "quantization": inputs.get("quantization") or DEFAULT_TRAINER_QUANTIZATION,
                    "vram_requirement": {
                        "standard_gb": 80,
                        "low_vram_gb": 32,
                        "target_gb": DEFAULT_TARGET_VRAM_GB,
                        "experimental": DEFAULT_TARGET_VRAM_GB < 32,
                        "note": (
                            "24GB trainer profiles are experimental; upstream documents 32GB "
                            "as the low-VRAM trainer target."
                        ),
                    },
                    "api_result": {
                        key: result.get(key)
                        for key in ("settings", "lora_files", "output_size_bytes")
                        if result.get(key) is not None
                    },
                }
            )
            self.orchestrator_service.update_job_status(
                self.job_id,
                "completed",
                storage_path=storage_path,
                completion_metadata=metadata,
            )
        except TokenExpiredError:
            raise
        except Exception as exc:
            logging.error("Official LTX-2 trainer job failed: %s", exc, exc_info=True)
            self._fail_job(f"Error processing LTX-2 trainer job: {exc}")
        finally:
            self._cleanup_remote_job(api_job_id)
            self._cleanup_container_file(
                f"/app/output/{api_job_id}.zip",
                "LTX-2 trainer API output",
            )
            if local_path and os.path.exists(local_path):
                self._cleanup_local_file(local_path, "LTX-2 trainer output")
            if archive_path and os.path.exists(archive_path):
                self._cleanup_local_file(archive_path, "LTX-2 trainer dataset archive")

    def _resolve_dataset_archive(self, inputs: dict) -> Optional[str]:
        archive_ref = (
            inputs.get("dataset_archive_storage_path")
            or inputs.get("dataset_zip_storage_path")
            or inputs.get("dataset_archive_url")
            or inputs.get("dataset_url")
        )
        if archive_ref:
            return self._download_storage_or_url(
                str(archive_ref),
                self.input_dir,
                bucket=inputs.get("dataset_archive_bucket") or inputs.get("bucket"),
            )
        return self._build_dataset_archive_from_clips(inputs)

    def _build_dataset_archive_from_clips(self, inputs: dict) -> Optional[str]:
        clips = (
            inputs.get("training_clips")
            or inputs.get("clips")
            or inputs.get("dataset")
            or inputs.get("items")
        )
        if not isinstance(clips, list):
            single_ref = (
                inputs.get("video_storage_path")
                or inputs.get("media_storage_path")
                or inputs.get("video_url")
                or inputs.get("media_url")
            )
            clips = [{"video_storage_path": single_ref, "caption": self.positive_prompt}] if single_ref else []
        if not clips:
            return None

        staging_dir = Path(self.cache_dir) / f"{self.job_id}_ltx2_dataset"
        media_dir = staging_dir / "videos"
        media_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, str]] = []

        try:
            for index, clip in enumerate(clips):
                if not isinstance(clip, dict):
                    continue
                source = (
                    clip.get("video_storage_path")
                    or clip.get("media_storage_path")
                    or clip.get("storage_path")
                    or clip.get("video_url")
                    or clip.get("media_url")
                    or clip.get("url")
                )
                if not source:
                    continue
                downloaded = self._download_storage_or_url(
                    str(source),
                    self.input_dir,
                    bucket=clip.get("bucket") or inputs.get("bucket"),
                )
                if not downloaded:
                    raise RuntimeError(f"Could not download training clip {index + 1}: {source}")
                safe = _safe_name(downloaded, f"clip_{index:04d}.mp4")
                target = media_dir / f"{index:04d}_{safe}"
                shutil.copy2(downloaded, target)
                caption = (
                    clip.get("caption")
                    or clip.get("prompt")
                    or clip.get("text")
                    or inputs.get("caption")
                    or self.positive_prompt
                    or "subject in consistent character animation"
                )
                records.append({"video": f"videos/{target.name}", "caption": str(caption)})

            if not records:
                return None
            dataset_path = staging_dir / "dataset.json"
            with dataset_path.open("w", encoding="utf-8") as handle:
                json.dump(records, handle, ensure_ascii=False, indent=2)

            archive_path = Path(self.cache_dir) / f"{self.job_id}_ltx2_dataset.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in staging_dir.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(staging_dir))
            return str(archive_path)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def _submit_training(self, archive_path: str, inputs: dict) -> Optional[str]:
        data = {
            "lora_trigger": str(inputs.get("lora_trigger") or inputs.get("trigger") or "").strip(),
            "resolution_buckets": str(inputs.get("resolution_buckets") or DEFAULT_TRAINER_BUCKETS),
            "steps": str(_as_int(inputs.get("steps"), DEFAULT_TRAINER_STEPS, 1)),
            "rank": str(_as_int(inputs.get("rank"), DEFAULT_TRAINER_RANK, 1, 128)),
            "alpha": str(
                _as_int(
                    inputs.get("alpha"),
                    _as_int(inputs.get("rank"), DEFAULT_TRAINER_ALPHA, 1, 128),
                    1,
                    128,
                )
            ),
            "learning_rate": str(_as_float(inputs.get("learning_rate"), 1e-4, 1e-7, 1e-2)),
            "first_frame_probability": str(_as_float(inputs.get("first_frame_probability"), 0.5, 0.0, 1.0)),
            "gradient_accumulation_steps": str(_as_int(inputs.get("gradient_accumulation_steps"), 1, 1)),
            "quantization": str(inputs.get("quantization") or DEFAULT_TRAINER_QUANTIZATION),
            "skip_audio": str(bool(inputs.get("skip_audio", True))).lower(),
            "validation_prompt": str(inputs.get("validation_prompt") or self.positive_prompt or ""),
            "seed": str(_as_int(inputs.get("seed"), 42)),
        }
        mime = mimetypes.guess_type(archive_path)[0] or "application/zip"
        try:
            with open(archive_path, "rb") as handle:
                files = {
                    "dataset_archive": (
                        os.path.basename(archive_path),
                        handle,
                        mime,
                    )
                }
                response = self.session.post(
                    f"{self.api_base_url}/train/i2v-lora",
                    data=data,
                    files=files,
                    timeout=180,
                )
            if response.status_code == 200:
                api_job_id = response.json().get("job_id")
                logging.info("Submitted official LTX-2 trainer job %s", api_job_id)
                return api_job_id
            logging.error("LTX-2 trainer submit failed: %s %s", response.status_code, response.text)
            return None
        except requests.exceptions.RequestException as exc:
            logging.error("LTX-2 trainer submit error: %s", exc)
            return None

    def _download_output(self, api_job_id: str) -> Optional[str]:
        local_path = os.path.join(self.cache_dir, f"{self.job_id}_ltx2_lora_package.zip")
        try:
            response = self.session.get(
                f"{self.api_base_url}/output/{api_job_id}",
                timeout=300,
                stream=True,
            )
            if response.status_code == 200:
                with open(local_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                return local_path
            logging.error("LTX-2 trainer output download failed: %s %s", response.status_code, response.text)
        except requests.exceptions.RequestException as exc:
            logging.error("LTX-2 trainer output download error: %s", exc)

        return recover_output_from_clean_container_exit(
            self,
            local_path,
            container_output_path=f"/app/output/{api_job_id}.zip",
            extensions=(".zip",),
            prefer_name=api_job_id,
        )


class LTX2PipelinesI2VLoraProcessor(LTX2OfficialBaseProcessor, VideoOutputHandler):
    """Generate I2V with official ltx-pipelines and a trained LoRA."""

    MAX_WAIT_TIME = int(
        os.environ.get(
            "LTX2_PIPELINES_TIMEOUT",
            str(max(TimeoutConfig.WORKFLOW_TIMEOUT, 2 * 60 * 60)),
        )
    )

    def process(self):
        if not self.job:
            self._fail_job("Job object is None for LTX2PipelinesI2VLoraProcessor.")
            return

        logging.info("Processing official LTX-2 pipelines I2V LoRA job %s", self.job_id)
        if not self._wait_for_api(int(os.environ.get("LTX2_PIPELINES_API_WAIT_TIMEOUT", "1800"))):
            self._fail_job(self._api_startup_error or "LTX-2 pipelines API did not become ready")
            return

        inputs = self.job.get("inputs") or {}
        start_image_path = self._resolve_start_image(inputs)
        if not start_image_path:
            self._fail_job("Could not resolve a start image for official LTX-2 I2V LoRA generation")
            return

        lora_path = self._resolve_lora(inputs)
        if not lora_path:
            self._fail_job(
                "No LoRA artifact was provided. Supply ltx_lora_storage_path, "
                "lora_storage_path, lora_package_storage_path, or lora_url."
            )
            return

        api_job_id = self._submit_generation(start_image_path, lora_path, inputs)
        if not api_job_id:
            self._fail_job("Failed to submit LTX-2 pipelines I2V LoRA job")
            return

        local_path: Optional[str] = None
        try:
            result = self._poll_for_completion(
                api_job_id,
                max_wait_time=self.MAX_WAIT_TIME,
                label="LTX-2 pipelines",
            )
            if result.get("status") != "completed":
                self._fail_job(f"LTX-2 pipelines generation failed: {result.get('error', 'Unknown error')}")
                return
            local_path = self._download_video(api_job_id)
            if not local_path:
                self._fail_job("Failed to download LTX-2 pipelines video output")
                return
            self._finalize_video(local_path, result, inputs)
        except TokenExpiredError:
            raise
        except Exception as exc:
            logging.error("Official LTX-2 pipelines job failed: %s", exc, exc_info=True)
            self._fail_job(f"Error processing LTX-2 pipelines job: {exc}")
        finally:
            self._cleanup_remote_job(api_job_id)
            self._cleanup_container_file(
                f"/app/output/{api_job_id}.mp4",
                "LTX-2 pipelines API output",
            )
            if local_path and os.path.exists(local_path):
                self._cleanup_local_file(local_path, "LTX-2 pipelines video")

    def _resolve_start_image(self, inputs: dict) -> Optional[str]:
        last_frame = materialize_last_frame_start_image(self, inputs)
        if last_frame:
            return last_frame

        start_image_url = inputs.get("start_image_url")
        if start_image_url:
            downloaded = self.orchestrator_service.download_asset_by_url(start_image_url, self.input_dir)
            if downloaded:
                return downloaded

        filename = materialize_start_image(self.job, self.input_dir)
        if filename:
            return os.path.join(self.input_dir, filename)

        storage_path = (
            inputs.get("start_image_storage_path")
            or inputs.get("input_storage_path")
            or self.job.get("input_storage_path")
        )
        if not storage_path:
            maybe = inputs.get("start_image_base64")
            if isinstance(maybe, str) and not maybe.startswith("data:") and len(maybe) < 2048:
                storage_path = maybe
        if storage_path:
            bucket = inputs.get("start_image_bucket") or self.job.get("bucket", "projects_public")
            downloaded = self.orchestrator_service.download_storage_asset(
                bucket,
                str(storage_path),
                self.input_dir,
            )
            if downloaded:
                return downloaded
            supabase_url = os.environ.get("SUPABASE_URL", SUPABASE_URL)
            if supabase_url:
                source_url = f"{supabase_url}/storage/v1/object/public/{bucket}/{storage_path}"
                return self.orchestrator_service.download_asset_by_url(source_url, self.input_dir)
        return None

    def _resolve_lora(self, inputs: dict) -> Optional[str]:
        lora_ref = (
            inputs.get("ltx_lora_storage_path")
            or inputs.get("lora_storage_path")
            or inputs.get("lora_package_storage_path")
            or inputs.get("lora_url")
            or inputs.get("ltx_lora_url")
        )
        if not lora_ref:
            return None
        return self._download_storage_or_url(
            str(lora_ref),
            self.input_dir,
            bucket=inputs.get("lora_bucket") or inputs.get("bucket"),
        )

    def _submit_generation(self, image_path: str, lora_path: str, inputs: dict) -> Optional[str]:
        width, height = self._resolve_dimensions(inputs)
        data = {
            "prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt
            or "worst quality, inconsistent motion, blurry, jittery, distorted",
            "trigger": str(inputs.get("lora_trigger") or inputs.get("trigger") or ""),
            "lora_strength": str(_as_float(inputs.get("lora_strength"), 0.8, -4.0, 4.0)),
            "width": str(width),
            "height": str(height),
            "num_frames": str(_as_int(inputs.get("num_frames") or inputs.get("frames"), DEFAULT_PIPELINES_FRAMES, 1, 257)),
            "frame_rate": str(_as_float(inputs.get("frame_rate"), 25.0, 1.0, 60.0)),
            "steps": str(_as_int(inputs.get("steps") or inputs.get("num_inference_steps"), DEFAULT_PIPELINES_STEPS, 1, 80)),
            "seed": str(_as_int(inputs.get("seed"), 42)),
            "image_strength": str(_as_float(inputs.get("image_strength"), 1.0, 0.0, 1.5)),
            "offload_mode": str(inputs.get("offload_mode") or inputs.get("offload") or DEFAULT_PIPELINES_OFFLOAD),
            "quantization": str(inputs.get("quantization") or DEFAULT_PIPELINES_QUANTIZATION),
            "distilled_lora_strength": str(_as_float(inputs.get("distilled_lora_strength"), 0.8, 0.0, 2.0)),
        }
        try:
            image_mime = mimetypes.guess_type(image_path)[0] or "image/png"
            lora_mime = "application/zip" if lora_path.lower().endswith(".zip") else "application/octet-stream"
            with open(image_path, "rb") as image_handle, open(lora_path, "rb") as lora_handle:
                files = {
                    "image": (os.path.basename(image_path), image_handle, image_mime),
                    "lora": (os.path.basename(lora_path), lora_handle, lora_mime),
                }
                response = self.session.post(
                    f"{self.api_base_url}/generate/i2v-lora",
                    data=data,
                    files=files,
                    timeout=180,
                )
            if response.status_code == 200:
                api_job_id = response.json().get("job_id")
                logging.info("Submitted official LTX-2 pipelines job %s", api_job_id)
                return api_job_id
            logging.error("LTX-2 pipelines submit failed: %s %s", response.status_code, response.text)
            return None
        except requests.exceptions.RequestException as exc:
            logging.error("LTX-2 pipelines submit error: %s", exc)
            return None

    def _download_video(self, api_job_id: str) -> Optional[str]:
        local_path = os.path.join(self.cache_dir, f"{self.job_id}.mp4")
        try:
            response = self.session.get(
                f"{self.api_base_url}/output/{api_job_id}",
                timeout=300,
                stream=True,
            )
            if response.status_code == 200:
                with open(local_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                return local_path
            logging.error("LTX-2 pipelines output download failed: %s %s", response.status_code, response.text)
        except requests.exceptions.RequestException as exc:
            logging.error("LTX-2 pipelines output download error: %s", exc)

        return recover_output_from_clean_container_exit(
            self,
            local_path,
            container_output_path=f"/app/output/{api_job_id}.mp4",
            extensions=(".mp4",),
            prefer_name=api_job_id,
        )

    def _finalize_video(self, local_path: str, result: dict, inputs: dict) -> None:
        video_storage_path = self.orchestrator_service.upload_output(
            local_path,
            self.job_id,
            "video/mp4",
        )
        if not video_storage_path:
            self._fail_job("LTX-2 pipelines completed, but video upload failed")
            return

        thumbnail_storage_path = None
        thumbnail_path = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
        if generate_thumbnail(local_path, thumbnail_path, width=THUMBNAIL_WIDTH):
            thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(
                thumbnail_path,
                self.job_id,
            )
            self._cleanup_local_file(thumbnail_path, "LTX-2 pipelines thumbnail")

        metadata = self._video_completion_metadata()
        metadata.update(
            {
                "processor": "LTX2PipelinesI2VLoraProcessor",
                "model": "official-ltx-2.3-22b-dev",
                "inference": "ltx_pipelines.ti2vid_two_stages",
                "lora_strength": _as_float(inputs.get("lora_strength"), 0.8),
                "ltx_lora_trigger": inputs.get("lora_trigger") or inputs.get("trigger"),
                "target_vram_gb": DEFAULT_TARGET_VRAM_GB,
                "experimental_vram_profile": DEFAULT_TARGET_VRAM_GB < 32,
                "api_result": {
                    key: result.get(key)
                    for key in ("settings", "output_size_bytes")
                    if result.get(key) is not None
                },
            }
        )
        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=video_storage_path,
            thumbnail_storage_path=thumbnail_storage_path,
            duration_seconds=get_video_duration(local_path),
            completion_metadata=metadata,
        )

    def _resolve_dimensions(self, inputs: dict) -> tuple[int, int]:
        width = _as_int(inputs.get("width") or inputs.get("target_width"), 0)
        height = _as_int(inputs.get("height") or inputs.get("target_height"), 0)
        if width > 0 and height > 0:
            return width, height
        aspect_ratio = str(inputs.get("aspect_ratio") or "16:9")
        if aspect_ratio == "9:16":
            return DEFAULT_PIPELINES_HEIGHT, DEFAULT_PIPELINES_WIDTH
        if aspect_ratio == "1:1":
            side = min(DEFAULT_PIPELINES_WIDTH, DEFAULT_PIPELINES_HEIGHT)
            return side, side
        if aspect_ratio == "4:3":
            return DEFAULT_PIPELINES_WIDTH, max(288, int(round(DEFAULT_PIPELINES_WIDTH * 3 / 4)))
        if aspect_ratio == "3:4":
            return max(288, int(round(DEFAULT_PIPELINES_HEIGHT * 3 / 4))), DEFAULT_PIPELINES_HEIGHT
        return DEFAULT_PIPELINES_WIDTH, DEFAULT_PIPELINES_HEIGHT
