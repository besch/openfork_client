"""DomainShuttle image-to-video processor.

Runs the DomainShuttle REST wrapper inside the 80GB service image. The upstream
runner is Wan2.2 A14B based and expects one or more reference images plus a
prompt.
"""

from __future__ import annotations

import base64
import logging
import os
import uuid
from typing import Iterable, Optional

import requests

from config import SUPABASE_URL, THUMBNAIL_WIDTH, TimeoutConfig
from services.processors.base import BaseJobProcessor
from services.processors.output_handlers import VideoOutputHandler
from services.processors.rest_recovery import (
    poll_rest_job_with_clean_exit,
    recover_output_from_clean_container_exit,
)
from services.processors.video.last_frame import materialize_last_frame_start_image
from utils.comfyui_workflow_utils import materialize_start_image
from utils.media_utils import (
    generate_thumbnail,
    get_video_duration,
    get_video_probe_metadata,
)

log = logging.getLogger(__name__)


_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
_DEFAULT_PROMPT = (
    "A cinematic video of the reference subject moving naturally, detailed, "
    "stable identity, coherent motion."
)


def _storage_value_path(value: str) -> str:
    return value.split("|", 1)[0].strip()


def _looks_like_http_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _looks_like_image_reference(value: str) -> bool:
    clean = _storage_value_path(value).split("?", 1)[0].lower()
    return clean.endswith(_IMAGE_EXTENSIONS)


def _safe_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _safe_float(value, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


class DomainShuttleImageToVideoProcessor(BaseJobProcessor, VideoOutputHandler):
    """Processor for DomainShuttle Wan2.2 A14B reference image-to-video jobs."""

    API_HOST = os.environ.get("DOMAINSHUTTLE_API_HOST", "127.0.0.1")
    API_PORT = int(os.environ.get("DOMAINSHUTTLE_API_PORT", "8000"))
    POLL_INTERVAL = int(os.environ.get("DOMAINSHUTTLE_POLL_INTERVAL", "10"))
    API_WAIT_TIMEOUT = int(os.environ.get("DOMAINSHUTTLE_API_WAIT_TIMEOUT", "3600"))
    MAX_WAIT_TIME = int(
        os.environ.get(
            "DOMAINSHUTTLE_GENERATION_TIMEOUT",
            str(max(TimeoutConfig.WORKFLOW_TIMEOUT, 10800)),
        )
    )

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.api_base_url = f"http://{self.API_HOST}:{self.API_PORT}"
        self.session = requests.Session()
        self.session.trust_env = False

    def process(self):
        if not self.job:
            self._fail_job("Job object is None. Cannot proceed.")
            return

        inputs = self.job.get("inputs") or {}
        prompt = (self.job.get("prompt") or inputs.get("prompt") or _DEFAULT_PROMPT).strip()

        output_path = None
        thumbnail_local_path = None
        remote_job_id = None
        cleanup_task_dir = False

        try:
            if not self._wait_for_api():
                self._fail_job("DomainShuttle API did not become available")
                return

            reference_images = self._resolve_reference_images(inputs)
            if not reference_images:
                self._fail_job(
                    "DomainShuttle requires at least one reference image. "
                    "Use the image-to-video workflow with a start image."
                )
                return

            payload = self._build_payload(inputs, prompt, len(reference_images))
            remote_job_id = self._submit_generation(reference_images, payload)
            if not remote_job_id:
                self._fail_job("Failed to submit DomainShuttle generation")
                return

            result = poll_rest_job_with_clean_exit(
                self,
                self.api_base_url,
                remote_job_id,
                poll_interval=self.POLL_INTERVAL,
                max_wait_time=self.MAX_WAIT_TIME,
                service_label="DomainShuttle",
                session=self.session,
            )
            if result.get("status") != "completed":
                cleanup_task_dir = result.get("status") == "failed"
                self._fail_job(
                    f"DomainShuttle generation failed: {result.get('error', 'Unknown error')}"
                )
                return
            cleanup_task_dir = True

            output_path = self._download_output(remote_job_id)
            if not output_path:
                self._fail_job("Failed to download DomainShuttle output video")
                return

            video_storage_path = self.orchestrator_service.upload_output(
                output_path,
                self.job_id,
                "video/mp4",
            )
            if not video_storage_path:
                self._fail_job("DomainShuttle video upload failed")
                return

            thumbnail_local_path = os.path.join(
                self.cache_dir,
                f"{self.job_id}_domainshuttle_thumb.jpg",
            )
            thumbnail_storage_path = None
            if generate_thumbnail(output_path, thumbnail_local_path, width=THUMBNAIL_WIDTH):
                thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(
                    thumbnail_local_path,
                    self.job_id,
                )

            duration = get_video_duration(output_path)
            video_metadata = get_video_probe_metadata(output_path, duration)
            completion_metadata = self.job.get("completion_metadata") or {}
            completion_metadata.update(
                {
                    "processor": "DomainShuttleImageToVideoProcessor",
                    "model": "Wan2.2-DomainShuttle-A14B",
                    "reference_image_count": len(reference_images),
                    "domain_code": payload.get("domain_code"),
                    "resolution": f"{payload.get('width')}x{payload.get('height')}",
                    "video_length": payload.get("video_length"),
                    "fps": payload.get("fps"),
                    "num_inference_steps": payload.get("num_inference_steps"),
                    "shift": payload.get("shift"),
                    "seed": payload.get("seed"),
                    **video_metadata,
                }
            )

            self.orchestrator_service.update_job_status(
                self.job_id,
                "completed",
                storage_path=video_storage_path,
                thumbnail_storage_path=thumbnail_storage_path,
                duration_seconds=duration,
                prompt=prompt,
                completion_metadata=completion_metadata,
            )
            log.info("DomainShuttle job %s completed", self.job_id)

        except Exception as exc:
            if "TokenExpiredError" in type(exc).__name__:
                raise
            log.error("DomainShuttle job %s failed: %s", self.job_id, exc, exc_info=True)
            self._fail_job(f"DomainShuttle processing error: {exc}")
        finally:
            for path in (output_path, thumbnail_local_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if cleanup_task_dir and remote_job_id:
                self._cleanup_container_file(
                    f"/app/output/{remote_job_id}",
                    "DomainShuttle task output",
                    recursive=True,
                )
                self._cleanup_container_file(
                    f"/app/input/{remote_job_id}",
                    "DomainShuttle task input",
                    recursive=True,
                )

    def _wait_for_api(self) -> bool:
        log.info(
            "Waiting for DomainShuttle API at %s (timeout: %ss)",
            self.api_base_url,
            self.API_WAIT_TIMEOUT,
        )
        start = self._monotonic()
        last_log = -30

        while self._monotonic() - start < self.API_WAIT_TIMEOUT:
            if self.is_cancelled():
                return False
            elapsed = int(self._monotonic() - start)
            try:
                response = self.session.get(f"{self.api_base_url}/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") in {"healthy", "ok"} or data.get("model_loaded"):
                        log.info("DomainShuttle API ready after %ss", elapsed)
                        return True
                    if data.get("status") == "error":
                        log.error("DomainShuttle API reported error: %s", data.get("error"))
            except requests.exceptions.RequestException:
                pass

            if elapsed - last_log >= 30:
                log.info("DomainShuttle API not ready (%s/%ss)", elapsed, self.API_WAIT_TIMEOUT)
                last_log = elapsed
            self.shutdown_event.wait(self.POLL_INTERVAL)
        return False

    @staticmethod
    def _monotonic() -> float:
        import time

        return time.monotonic()

    def _build_payload(self, inputs: dict, prompt: str, reference_count: int) -> dict:
        aspect_ratio = inputs.get("aspect_ratio", "16:9")
        default_width, default_height = (832, 480)
        if aspect_ratio == "9:16":
            default_width, default_height = (480, 832)
        elif aspect_ratio == "1:1":
            default_width, default_height = (640, 640)

        duration = _safe_float(inputs.get("duration"), 81 / 24, 1.0, 6.0)
        fps = _safe_int(inputs.get("fps"), 24, 8, 30)
        requested_video_length = inputs.get("video_length") or inputs.get("frames")
        if requested_video_length is None:
            video_length = max(17, int(round(duration * fps)))
        else:
            video_length = _safe_int(requested_video_length, 81, 17, 145)
        if video_length % 2 == 0:
            video_length += 1

        domain_code = self._domain_code(inputs, reference_count)
        seed = self.normalize_seed(inputs.get("seed", -1))
        if seed < 0:
            seed = None

        return {
            "prompt": prompt,
            "negative_prompt": inputs.get("negative_prompt") or self.negative_prompt or "",
            "domain_code": ",".join(domain_code),
            "height": _safe_int(inputs.get("height"), default_height, 256, 1024),
            "width": _safe_int(inputs.get("width"), default_width, 256, 1024),
            "video_length": video_length,
            "fps": fps,
            "num_inference_steps": _safe_int(inputs.get("steps"), 40, 8, 50),
            "guidance_scale_high": _safe_float(
                inputs.get("guidance_scale_high", inputs.get("cfg_scale")),
                4.0,
                1.0,
                12.0,
            ),
            "guidance_scale_low": _safe_float(
                inputs.get("guidance_scale_low", inputs.get("cfg_scale_low")),
                3.0,
                1.0,
                12.0,
            ),
            "shift": _safe_float(inputs.get("flow_shift", inputs.get("shift")), 5.0, 0.1, 20.0),
            "seed": seed,
            "nproc_per_node": _safe_int(
                inputs.get("nproc_per_node"),
                int(os.environ.get("DOMAINSHUTTLE_NPROC_PER_NODE", "1")),
                1,
                8,
            ),
            "ring_degree": _safe_int(inputs.get("ring_degree"), 1, 1, 8),
        }

    def _domain_code(self, inputs: dict, reference_count: int) -> list[str]:
        value = inputs.get("domain_code") or inputs.get("subject_domain") or inputs.get("domain")
        if isinstance(value, list):
            codes = [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(value, str) and value.strip().startswith("["):
            try:
                import json

                parsed = json.loads(value)
                codes = [str(item).strip() for item in parsed if str(item).strip()]
            except Exception:
                codes = [value.strip()]
        elif isinstance(value, str) and "," in value:
            codes = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, str) and value.strip():
            codes = [value.strip()]
        else:
            codes = ["Human"]

        if len(codes) < reference_count:
            codes.extend([codes[-1]] * (reference_count - len(codes)))
        return codes[:reference_count]

    def _submit_generation(self, image_paths: Iterable[str], payload: dict) -> Optional[str]:
        files = []
        handles = []
        try:
            for path in image_paths:
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
                data={
                    key: str(value)
                    for key, value in payload.items()
                    if value is not None
                },
                files=files,
                timeout=60,
            )
            if response.status_code == 200:
                return response.json().get("job_id")
            log.error("DomainShuttle generate failed: %s %s", response.status_code, response.text)
            return None
        except requests.exceptions.RequestException as exc:
            log.error("DomainShuttle generate request failed: %s", exc)
            return None
        finally:
            for handle in handles:
                handle.close()

    def _download_output(self, remote_job_id: str) -> Optional[str]:
        output_path = os.path.join(self.cache_dir, f"{self.job_id}_domainshuttle.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            response = self.session.get(
                f"{self.api_base_url}/output/{remote_job_id}",
                timeout=180,
                stream=True,
            )
            if response.status_code == 200:
                with open(output_path, "wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    return output_path
            else:
                log.error(
                    "DomainShuttle output download failed: %s %s",
                    response.status_code,
                    response.text[:500],
                )
        except requests.exceptions.RequestException as exc:
            log.error("DomainShuttle output download error: %s", exc)

        return recover_output_from_clean_container_exit(
            self,
            output_path,
            container_output_path=f"/app/output/{remote_job_id}",
            extensions=(".mp4", ".mov", ".mkv", ".avi"),
            prefer_name=remote_job_id,
        )

    def _resolve_reference_images(self, inputs: dict) -> list[str]:
        paths: list[str] = []

        def add(path: Optional[str]) -> None:
            if not path:
                return
            absolute = os.path.abspath(path)
            if absolute not in {os.path.abspath(existing) for existing in paths}:
                paths.append(path)

        add(materialize_last_frame_start_image(self, inputs))

        for key in (
            "start_image_url",
            "reference_image_url",
            "content_image_url",
            "image_url",
        ):
            add(self._resolve_url(inputs.get(key)))

        filename = materialize_start_image(self.job, self.input_dir)
        if filename:
            add(os.path.join(self.input_dir, filename))

        for key in (
            "input_storage_path",
            "start_image_storage_path",
            "reference_image_storage_path",
            "content_image_storage_path",
            "image_storage_path",
            "start_image",
            "reference_image",
            "content_image",
            "image",
        ):
            add(self._resolve_storage_or_reference(inputs.get(key), inputs, key))

        job_storage_path = self.job.get("input_storage_path")
        if job_storage_path:
            add(self._resolve_storage_or_reference(job_storage_path, inputs, "input_storage_path"))

        for index in range(2, 6):
            for suffix in ("url", "storage_path", "image"):
                add(self._resolve_storage_or_reference(
                    inputs.get(f"reference_image_{index}_{suffix}"),
                    inputs,
                    f"reference_image_{index}_{suffix}",
                ))

        add(self._decode_base64_image(inputs.get("start_image_base64"), "start"))
        add(self._decode_base64_image(inputs.get("reference_image_base64"), "reference"))

        return paths[:5]

    def _resolve_url(self, value) -> Optional[str]:
        if not value or not isinstance(value, str) or not _looks_like_http_url(value):
            return None
        return self.orchestrator_service.download_asset_by_url(value, self.input_dir)

    def _resolve_storage_or_reference(self, value, inputs: dict, key: str) -> Optional[str]:
        if not value or not isinstance(value, str):
            return None

        value = value.strip()
        if _looks_like_http_url(value):
            return self._resolve_url(value)

        local_path = _storage_value_path(value)
        if os.path.exists(local_path):
            return local_path

        if not _looks_like_image_reference(value) and len(value) > 2048:
            return None

        bucket_hint = (
            inputs.get(f"{key}_bucket")
            or inputs.get("bucket")
            or self.job.get("bucket")
            or "projects_public"
        )
        downloaded = self.orchestrator_service.download_storage_asset(
            bucket_hint,
            _storage_value_path(value),
            self.input_dir,
        )
        if downloaded:
            return downloaded

        if SUPABASE_URL and "/" in _storage_value_path(value):
            public_url = (
                f"{SUPABASE_URL}/storage/v1/object/public/"
                f"{bucket_hint}/{_storage_value_path(value)}"
            )
            return self.orchestrator_service.download_asset_by_url(
                public_url,
                self.input_dir,
            )
        return None

    def _decode_base64_image(self, value, label: str) -> Optional[str]:
        if not value or not isinstance(value, str) or len(value) < 128:
            return None
        if value.startswith("data:"):
            value = value.split(",", 1)[-1]
        try:
            data = base64.b64decode(value, validate=True)
        except Exception:
            return None
        if not data:
            return None

        os.makedirs(self.input_dir, exist_ok=True)
        path = os.path.join(
            self.input_dir,
            f"domainshuttle_{label}_{uuid.uuid4().hex[:8]}.png",
        )
        with open(path, "wb") as image_file:
            image_file.write(data)
        return path
