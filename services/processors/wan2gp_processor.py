"""
Wan2GP Processor Base

Wan2GP runs as an HTTP server (wan2gp_server.py) inside the Docker container.
The host client communicates with it via a simple REST API on port 8188.

On first call, _run_task() waits up to WAN2GP_READY_TIMEOUT seconds for the
server to become available — the model can take 5-15 minutes to load.
"""

import base64
import io
import logging
import os
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import requests

from config import THUMBNAIL_WIDTH
from exceptions import InfrastructureError
from services.processors.base import BaseJobProcessor
from services.processors.rest_recovery import (
    clean_container_exit_detected,
    recover_output_from_clean_container_exit,
)
from utils.media_utils import generate_thumbnail, get_video_duration, get_video_probe_metadata

WAN2GP_HTTP_URL = os.environ.get("WAN2GP_HTTP_URL", "http://127.0.0.1:8188")
WAN2GP_READY_TIMEOUT = int(os.environ.get("WAN2GP_READY_TIMEOUT", "1800"))  # 30 min
WAN2GP_LOG_PATH = os.environ.get("WAN2GP_LOG_PATH", "/tmp/wan2gp_server.log")
WAN2GP_MAX_SEED = 2**32 - 1

# Aspect ratio → "WIDTHxHEIGHT" strings required by Wan2GP
_ASPECT_RESOLUTIONS = {
    "16:9": "1280x720",
    "9:16": "720x1280",
    "1:1": "1024x1024",
    "4:3": "1024x768",
    "3:4": "768x1024",
    "21:9": "1280x544",
    "2:1": "1280x640",
}

_ASPECT_RESOLUTIONS_16GB = {
    "16:9": "768x432",
    "9:16": "432x768",
    "1:1": "640x640",
    "4:3": "640x480",
    "3:4": "480x640",
    "21:9": "768x320",
    "2:1": "768x384",
}

_ASPECT_RESOLUTIONS_24GB = {
    "16:9": "960x544",
    "9:16": "544x960",
    "1:1": "832x832",
    "4:3": "896x672",
    "3:4": "672x896",
    "21:9": "960x416",
    "2:1": "960x480",
}

_ASPECT_RESOLUTIONS_12GB = {
    "16:9": "544x304",
    "9:16": "304x544",
    "1:1": "480x480",
    "4:3": "480x368",
    "3:4": "368x480",
    "21:9": "544x240",
    "2:1": "544x272",
}

_ASPECT_RESOLUTIONS_8GB = {
    "16:9": "512x288",
    "9:16": "288x512",
    "1:1": "448x448",
    "4:3": "448x336",
    "3:4": "336x448",
    "21:9": "512x224",
    "2:1": "512x256",
}


def _pil_to_data_uri(image) -> str:
    """Encode a PIL Image as a JPEG data-URI for JSON transport."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class Wan2GPProcessor(BaseJobProcessor):
    """Base class for processors that use the Wan2GP HTTP server."""

    MAX_GENERATION_SECONDS = 7200  # 2 hours
    CONTAINER_OUTPUT_DIR = "/opt/wan2gp/outputs"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infrastructure_interrupted = False
        self.last_wan2gp_error: Optional[str] = None

    @staticmethod
    def aspect_to_resolution(aspect_ratio: str, vram_tier: str = "") -> str:
        # Empty / unrecognized vram_tier silently selects 1280x720, which OOMs
        # any 8 GB card. Log the selection so misrouted jobs surface in logs.
        if "8gb" in vram_tier:
            tier, resolution = "8gb", _ASPECT_RESOLUTIONS_8GB.get(aspect_ratio, "512x288")
        elif "12gb" in vram_tier:
            tier, resolution = "12gb", _ASPECT_RESOLUTIONS_12GB.get(aspect_ratio, "544x304")
        elif "16gb" in vram_tier:
            tier, resolution = "16gb", _ASPECT_RESOLUTIONS_16GB.get(aspect_ratio, "768x432")
        elif "24gb" in vram_tier:
            tier, resolution = "24gb", _ASPECT_RESOLUTIONS_24GB.get(aspect_ratio, "960x544")
        else:
            tier, resolution = "default(24+gb)", _ASPECT_RESOLUTIONS.get(aspect_ratio, "1280x720")
        logging.info(
            "Wan2GP resolution: vram_tier=%r aspect=%s tier=%s -> %s",
            vram_tier, aspect_ratio, tier, resolution,
        )
        return resolution

    @staticmethod
    def normalize_seed(value, default: int = 0) -> int:
        """Wan2GP accepts unsigned 32-bit seeds only."""
        try:
            seed = int(default if value is None else value)
        except (TypeError, ValueError):
            seed = int(default)
        return seed % (WAN2GP_MAX_SEED + 1)

    def _wait_for_server(self) -> bool:
        """Poll /health until the Wan2GP server responds or timeout is reached."""
        url = f"{WAN2GP_HTTP_URL}/health"
        deadline = time.monotonic() + WAN2GP_READY_TIMEOUT
        last_log = time.monotonic()
        while time.monotonic() < deadline:
            if self.is_cancelled():
                return False
            if self._job_was_interrupted_by_infrastructure():
                logging.warning(
                    "Wan2GP server wait stopped because job %s was interrupted "
                    "by a container-level failure.",
                    self.job_id,
                )
                return False
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200 and resp.json().get("status") == "ok":
                    logging.info("Wan2GP server is ready.")
                    return True
            except requests.exceptions.RequestException:
                pass
            now = time.monotonic()
            if now - last_log >= 60:
                elapsed = int(now - (deadline - WAN2GP_READY_TIMEOUT))
                logging.info(f"Waiting for Wan2GP server... ({elapsed}s elapsed)")
                last_log = now
            self.shutdown_event.wait(5)
        logging.error(
            f"Wan2GP server did not become ready within {WAN2GP_READY_TIMEOUT}s."
        )
        self._log_backend_tail()
        return False

    @staticmethod
    def _log_backend_tail(lines: int = 60) -> None:
        if not os.path.isfile(WAN2GP_LOG_PATH):
            logging.error("Wan2GP server log not found at %s.", WAN2GP_LOG_PATH)
            return

        try:
            with open(
                WAN2GP_LOG_PATH,
                "r",
                encoding="utf-8",
                errors="replace",
            ) as handle:
                tail = deque(handle, maxlen=max(1, lines))
        except OSError as exc:
            logging.error(
                "Could not read Wan2GP server log at %s: %s",
                WAN2GP_LOG_PATH,
                exc,
            )
            return

        rendered = "".join(tail).strip()
        if rendered:
            logging.error(
                "Recent Wan2GP server log (%s):\n%s",
                WAN2GP_LOG_PATH,
                rendered,
            )
        else:
            logging.error("Wan2GP server log at %s is empty.", WAN2GP_LOG_PATH)

    def _job_was_interrupted_by_infrastructure(self) -> bool:
        interrupted_token = getattr(self.client, "interrupted_job_execution_token", None)
        execution_token = self.job.get("execution_token")
        if (
            getattr(self.client, "interrupted_job_id", None) == self.job_id
            and (
                not interrupted_token
                or not execution_token
                or interrupted_token == execution_token
            )
        ):
            self.infrastructure_interrupted = True
            return True
        return False

    @staticmethod
    def _is_wan2gp_cuda_oom(detail) -> bool:
        text = str(detail).lower()
        oom_markers = (
            "cuda driver error: out of memory",
            "cuda out of memory",
            "torch.cuda.outofmemoryerror",
            "cublas_status_alloc_failed",
        )
        return any(marker in text for marker in oom_markers)

    @staticmethod
    def _format_wan2gp_error(detail) -> str:
        if isinstance(detail, dict):
            nested = detail.get("detail", detail)
            if isinstance(nested, dict):
                errors = nested.get("errors")
                if isinstance(errors, list) and errors:
                    return "; ".join(str(error) for error in errors[:3])
                for key in ("error", "message"):
                    value = nested.get(key)
                    if value:
                        return str(value)
            return str(detail)

        return str(detail).strip()

    def _wan2gp_no_output_message(self, service_name: str = "Wan2GP") -> str:
        if self.last_wan2gp_error:
            if "fps_mode" in self.last_wan2gp_error and "ffmpeg" in self.last_wan2gp_error.lower():
                return (
                    f"{service_name} failed for job {self.job_id}: FFmpeg is "
                    "too old for Wan2GP video-guide decode (`-fps_mode` is "
                    "unsupported). Update or restart the worker with FFmpeg 7.1+ "
                    "before retrying."
                )
            if "FieldsBuilder finalized" in self.last_wan2gp_error:
                return (
                    f"{service_name} failed for job {self.job_id}: Taichi "
                    "FieldsBuilder finalized. Restart or update the worker so "
                    "the stable main-thread recycle Wan2GP wrapper is active, "
                    "then retry."
                )
            return (
                f"{service_name} produced no output for job {self.job_id}: "
                f"{self.last_wan2gp_error}"
            )

        return f"Wan2GP produced no output for job {self.job_id}"

    def _run_task(self, settings: dict) -> List[str]:
        """Submit a generation task and return local paths to the output files."""
        self.last_wan2gp_error = None

        if not self._wait_for_server():
            self.last_wan2gp_error = "Wan2GP server did not become ready."
            return []

        # Serialize any PIL Images to data-URIs so they survive JSON transport
        serialized = {}
        for key, val in settings.items():
            try:
                from PIL import Image as _PIL_Image

                if isinstance(val, _PIL_Image.Image):
                    serialized[key] = _pil_to_data_uri(val)
                    continue
            except ImportError:
                pass
            serialized[key] = val

        generate_url = f"{WAN2GP_HTTP_URL}/generate"
        resp_holder: list = [None]
        error_holder: list = [None]

        def _do_generate():
            try:
                resp_holder[0] = requests.post(
                    generate_url,
                    json={"settings": serialized},
                    timeout=self.MAX_GENERATION_SECONDS + 60,
                )
            except requests.exceptions.RequestException as e:
                error_holder[0] = e

        gen_thread = threading.Thread(target=_do_generate, daemon=True)
        gen_thread.start()
        while gen_thread.is_alive():
            gen_thread.join(timeout=1.0)
            if self.is_cancelled():
                logging.info(
                    "Shutdown event received during Wan2GP generation. Aborting job %s.",
                    self.job_id,
                )
                return []

        if error_holder[0]:
            if self._job_was_interrupted_by_infrastructure():
                logging.warning(
                    "Wan2GP generate request stopped after container-level "
                    "failure for job %s.",
                    self.job_id,
                )
                return []
            recovered = self._recover_output_after_clean_exit()
            if recovered:
                return recovered
            self.last_wan2gp_error = str(error_holder[0])
            logging.error(
                f"Wan2GP generate request failed for job {self.job_id}: {error_holder[0]}"
            )
            return []

        resp = resp_holder[0]

        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text

            if self._is_wan2gp_cuda_oom(detail):
                self.infrastructure_interrupted = True
                raise InfrastructureError(
                    f"Wan2GP CUDA out of memory for job {self.job_id}: {detail}"
                )

            self.last_wan2gp_error = self._format_wan2gp_error(detail)
            logging.error(
                f"Wan2GP server returned error for job {self.job_id}: {detail}"
            )
            return self._recover_output_after_clean_exit()

        basenames = resp.json().get("files", [])
        if not basenames:
            self.last_wan2gp_error = "Wan2GP returned no output files."
            logging.error(f"Wan2GP returned no output files for job {self.job_id}")
            return self._recover_output_after_clean_exit()

        # Download each output file to a local temp path
        local_paths = []
        for name in basenames:
            download_url = f"{WAN2GP_HTTP_URL}/output/{name}"
            logging.info(
                "Downloading generated Wan2GP output from local container: %s",
                name,
            )
            try:
                dl = requests.get(download_url, timeout=300, stream=True)
                dl.raise_for_status()
            except requests.exceptions.RequestException as e:
                recovered_path = self._recover_named_output_after_clean_exit(name)
                if recovered_path:
                    local_paths.append(recovered_path)
                    continue
                logging.error(
                    f"Failed to download output '{name}' for job {self.job_id}: {e}"
                )
                continue

            suffix = Path(name).suffix or ".mp4"
            os.makedirs(self.cache_dir, exist_ok=True)
            tmp = tempfile.NamedTemporaryFile(
                delete=False,
                prefix=f"{self.job_id}_",
                suffix=suffix,
                dir=self.cache_dir,
            )
            try:
                bytes_written = 0
                started_at = time.monotonic()
                for chunk in dl.iter_content(chunk_size=1 << 20):
                    bytes_written += len(chunk)
                    tmp.write(chunk)
                tmp.close()
                local_paths.append(tmp.name)
                elapsed = max(time.monotonic() - started_at, 0.001)
                logging.info(
                    "Saved Wan2GP output locally: %.1f MB in %.1fs (%.1f MB/s)",
                    bytes_written / (1024 * 1024),
                    elapsed,
                    (bytes_written / (1024 * 1024)) / elapsed,
                )
                self._cleanup_container_file(
                    f"{self.CONTAINER_OUTPUT_DIR}/{Path(name).name}",
                    "Wan2GP video output",
                )
            except Exception as e:
                tmp.close()
                os.unlink(tmp.name)
                logging.error(
                    f"Failed to write output file for job {self.job_id}: {e}"
                )

        if basenames and not local_paths:
            self.last_wan2gp_error = (
                "Wan2GP generated files, but the client could not download or "
                "write any output."
            )

        return local_paths

    def _recover_output_after_clean_exit(self) -> List[str]:
        if not clean_container_exit_detected(self):
            return []

        os.makedirs(self.cache_dir, exist_ok=True)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                prefix=f"{self.job_id}_recover_",
                suffix=".mp4",
                dir=self.cache_dir,
            ) as tmp:
                tmp_path = tmp.name

            recovered = recover_output_from_clean_container_exit(
                self,
                tmp_path,
                container_output_path=self.CONTAINER_OUTPUT_DIR,
                extensions=(".mp4", ".mov", ".webm", ".mkv"),
                prefer_name=self.job_id,
            )
            return [recovered] if recovered else []
        finally:
            if tmp_path and os.path.exists(tmp_path) and os.path.getsize(tmp_path) == 0:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _recover_named_output_after_clean_exit(self, name: str) -> Optional[str]:
        if not clean_container_exit_detected(self):
            return None

        suffix = Path(name).suffix or ".mp4"
        os.makedirs(self.cache_dir, exist_ok=True)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                prefix=f"{self.job_id}_recover_",
                suffix=suffix,
                dir=self.cache_dir,
            ) as tmp:
                tmp_path = tmp.name

            recovered = recover_output_from_clean_container_exit(
                self,
                tmp_path,
                container_output_path=f"{self.CONTAINER_OUTPUT_DIR}/{Path(name).name}",
                extensions=(".mp4", ".mov", ".webm", ".mkv"),
                prefer_name=Path(name).stem,
            )
            if recovered:
                self._cleanup_container_file(
                    f"{self.CONTAINER_OUTPUT_DIR}/{Path(name).name}",
                    "Wan2GP recovered video output",
                )
            return recovered
        finally:
            if tmp_path and os.path.exists(tmp_path) and os.path.getsize(tmp_path) == 0:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _handle_video_output(
        self, file_path: str
    ) -> Optional[Tuple[str, Optional[str], float]]:
        """Upload a locally generated video file.

        Returns (storage_path, thumbnail_storage_path, duration) or None.
        The source file is deleted after a successful upload.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            video_storage_path = self.orchestrator_service.upload_output(
                file_path, self.job_id, "video/mp4"
            )
            if not video_storage_path:
                logging.error(f"Video upload failed for job {self.job_id}")
                return None

            thumbnail_storage_path = None
            thumbnail_generated = False
            thumb_local = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
            if generate_thumbnail(file_path, thumb_local, width=THUMBNAIL_WIDTH):
                thumbnail_generated = True
                thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(
                    thumb_local, self.job_id
                )
                if os.path.exists(thumb_local):
                    os.remove(thumb_local)

            duration = get_video_duration(file_path)
            video_metadata = get_video_probe_metadata(file_path, duration)
            if thumbnail_generated and not thumbnail_storage_path:
                video_metadata["thumbnail_upload_failed"] = True
            self.last_video_output_metadata = video_metadata
            return video_storage_path, thumbnail_storage_path, duration

        except Exception as e:
            logging.error(
                f"Failed to handle video output for job {self.job_id}: {e}",
                exc_info=True,
            )
            return None
        finally:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass

    def _video_completion_metadata(self) -> dict:
        metadata = dict(self.job.get("completion_metadata") or {})
        video_metadata = getattr(self, "last_video_output_metadata", None)
        if isinstance(video_metadata, dict):
            metadata.update(video_metadata)
        return metadata
