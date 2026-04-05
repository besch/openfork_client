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
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests

from config import THUMBNAIL_WIDTH
from services.processors.base import BaseJobProcessor
from utils.media_utils import generate_thumbnail, get_video_duration

WAN2GP_HTTP_URL = os.environ.get("WAN2GP_HTTP_URL", "http://127.0.0.1:8188")
WAN2GP_READY_TIMEOUT = int(os.environ.get("WAN2GP_READY_TIMEOUT", "1800"))  # 30 min

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

    @staticmethod
    def aspect_to_resolution(aspect_ratio: str, vram_tier: str = "") -> str:
        if "8gb" in vram_tier:
            return _ASPECT_RESOLUTIONS_8GB.get(aspect_ratio, "512x288")
        if "16gb" in vram_tier:
            return _ASPECT_RESOLUTIONS_16GB.get(aspect_ratio, "768x432")
        return _ASPECT_RESOLUTIONS.get(aspect_ratio, "1280x720")

    def _wait_for_server(self) -> bool:
        """Poll /health until the Wan2GP server responds or timeout is reached."""
        url = f"{WAN2GP_HTTP_URL}/health"
        deadline = time.monotonic() + WAN2GP_READY_TIMEOUT
        last_log = time.monotonic()
        while time.monotonic() < deadline:
            if self.shutdown_event.is_set():
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
        return False

    def _run_task(self, settings: dict) -> List[str]:
        """Submit a generation task and return local paths to the output files."""
        if not self._wait_for_server():
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
        try:
            resp = requests.post(
                generate_url,
                json={"settings": serialized},
                timeout=self.MAX_GENERATION_SECONDS + 60,
            )
        except requests.exceptions.RequestException as e:
            logging.error(
                f"Wan2GP generate request failed for job {self.job_id}: {e}"
            )
            return []

        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            logging.error(
                f"Wan2GP server returned error for job {self.job_id}: {detail}"
            )
            return []

        basenames = resp.json().get("files", [])
        if not basenames:
            logging.error(f"Wan2GP returned no output files for job {self.job_id}")
            return []

        # Download each output file to a local temp path
        local_paths = []
        for name in basenames:
            download_url = f"{WAN2GP_HTTP_URL}/output/{name}"
            try:
                dl = requests.get(download_url, timeout=300, stream=True)
                dl.raise_for_status()
            except requests.exceptions.RequestException as e:
                logging.error(
                    f"Failed to download output '{name}' for job {self.job_id}: {e}"
                )
                continue

            suffix = Path(name).suffix or ".mp4"
            os.makedirs(self.cache_dir, exist_ok=True)
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix, dir=self.cache_dir
            )
            try:
                for chunk in dl.iter_content(chunk_size=1 << 20):
                    tmp.write(chunk)
                tmp.close()
                local_paths.append(tmp.name)
            except Exception as e:
                tmp.close()
                os.unlink(tmp.name)
                logging.error(
                    f"Failed to write output file for job {self.job_id}: {e}"
                )

        return local_paths

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
            thumb_local = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
            if generate_thumbnail(file_path, thumb_local, width=THUMBNAIL_WIDTH):
                thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(
                    thumb_local, self.job_id
                )
                if os.path.exists(thumb_local):
                    os.remove(thumb_local)

            duration = get_video_duration(file_path)
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
