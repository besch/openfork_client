"""Local image-motion video processor for low-VRAM workflow validation."""

import logging
import os
import shutil
import subprocess
from typing import Optional, Tuple

from config import SUPABASE_URL, THUMBNAIL_WIDTH
from services.docker_utils import get_subprocess_hidden_kwargs
from services.processors.base import BaseJobProcessor
from services.processors.output_handlers import VideoOutputHandler
from services.orchestrator_service import TokenExpiredError
from utils.comfyui_workflow_utils import materialize_start_image
from utils.media_utils import generate_thumbnail, get_video_duration


class ImageMotionVideoProcessor(BaseJobProcessor, VideoOutputHandler):
    """Render a short MP4 from a generated image without requiring a model container."""

    FPS = 24
    MAX_DURATION_SECONDS = 2.0

    def process(self):
        if not self.job:
            self._fail_job("Job object is None for ImageMotionVideoProcessor.")
            return

        start_image_path = self._resolve_start_image()
        if not start_image_path:
            self._fail_job(f"Failed to materialise start image for job {self.job_id}.")
            return

        inputs = self.job.get("inputs") or {}
        aspect_ratio = inputs.get("aspect_ratio", "16:9")
        width, height = self._resolution_for_aspect(aspect_ratio)
        duration = self._clamp_duration(
            inputs.get("duration_seconds", inputs.get("duration"))
        )
        frames = max(1, int(round(duration * self.FPS)))

        os.makedirs(self.cache_dir, exist_ok=True)
        output_path = os.path.join(self.cache_dir, f"{self.job_id}_image_motion.mp4")

        if not self._render_motion_video(
            start_image_path,
            output_path,
            width,
            height,
            frames,
            inputs.get("camera_movement", ""),
        ):
            return

        self._finalize_video(output_path, duration, frames, width, height)

    def _resolve_start_image(self) -> Optional[str]:
        inputs = self.job.get("inputs") or {}
        start_image_url = inputs.get("start_image_url")

        if start_image_url:
            downloaded_path = self.orchestrator_service.download_asset_by_url(
                start_image_url,
                self.input_dir,
            )
            if downloaded_path:
                return downloaded_path

        start_image_filename = materialize_start_image(self.job, self.input_dir)
        if start_image_filename:
            return os.path.join(self.input_dir, start_image_filename)

        input_storage_path = self.job.get("input_storage_path")
        if not input_storage_path:
            possible_path = self.job.get("start_image_base64")
            if (
                possible_path
                and isinstance(possible_path, str)
                and not possible_path.startswith("data:")
                and len(possible_path) < 2048
            ):
                input_storage_path = possible_path

        if not input_storage_path:
            return None

        bucket = self.job.get("bucket", "projects_public")
        downloaded_path = self.orchestrator_service.download_storage_asset(
            bucket,
            input_storage_path,
            self.input_dir,
        )
        if downloaded_path:
            return downloaded_path

        supabase_url = os.environ.get(
            "SUPABASE_URL",
            self.client.config.get("SUPABASE_URL", SUPABASE_URL),
        )
        if not supabase_url:
            return None

        source_url = (
            f"{supabase_url}/storage/v1/object/public/{bucket}/{input_storage_path}"
        )
        return self.orchestrator_service.download_asset_by_url(
            source_url,
            self.input_dir,
        )

    def _resolution_for_aspect(self, aspect_ratio: str) -> Tuple[int, int]:
        if aspect_ratio == "9:16":
            return 432, 768
        if aspect_ratio == "1:1":
            return 640, 640
        if aspect_ratio == "4:3":
            return 640, 480
        if aspect_ratio == "3:4":
            return 480, 640
        if aspect_ratio == "21:9":
            return 896, 384
        return 768, 432

    def _clamp_duration(self, value) -> float:
        try:
            duration = float(value)
        except (TypeError, ValueError):
            duration = self.MAX_DURATION_SECONDS
        return max(1.0, min(duration, self.MAX_DURATION_SECONDS))

    def _motion_filter(
        self,
        width: int,
        height: int,
        frames: int,
        camera_movement: str,
    ) -> str:
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},"
            f"fps={self.FPS},"
            "format=yuv420p"
        )

    def _render_motion_video(
        self,
        image_path: str,
        output_path: str,
        width: int,
        height: int,
        frames: int,
        camera_movement: str,
    ) -> bool:
        vf = self._motion_filter(width, height, frames, camera_movement)
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        command = [
            ffmpeg_bin,
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-i",
            image_path,
            "-vf",
            vf,
            "-frames:v",
            str(frames),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            output_path,
        ]

        try:
            result = subprocess.run(
                command,
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
                **get_subprocess_hidden_kwargs(),
            )
            if result.stderr:
                logging.debug("ffmpeg image-motion render stderr: %s", result.stderr)
            if not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
                self._fail_job(f"Image motion renderer produced an empty video for job {self.job_id}.")
                return False
            logging.info(
                "Image motion render completed for job %s using %s (%s bytes).",
                self.job_id,
                ffmpeg_bin,
                os.path.getsize(output_path),
            )
            return True
        except subprocess.TimeoutExpired as exc:
            stderr = exc.stderr or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            logging.error("ffmpeg image-motion render timed out: %s", stderr)
            self._fail_job(f"Timed out rendering image motion video for job {self.job_id}.")
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or ""
            logging.error("ffmpeg image-motion render failed: %s", stderr)
            self._fail_job(f"Image motion render failed for job {self.job_id}.")
        except FileNotFoundError:
            self._fail_job("ffmpeg is not available for local image motion rendering.")
        return False

    def _finalize_video(
        self,
        output_path: str,
        requested_duration: float,
        frames: int,
        width: int,
        height: int,
    ) -> None:
        try:
            video_storage_path = self.orchestrator_service.upload_output(
                output_path,
                self.job_id,
                "video/mp4",
            )
            if not video_storage_path:
                self._fail_job(f"Video upload failed for job {self.job_id}.")
                return

            thumbnail_storage_path = None
            thumbnail_path = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
            if generate_thumbnail(output_path, thumbnail_path, width=THUMBNAIL_WIDTH):
                thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(
                    thumbnail_path,
                    self.job_id,
                )
                if os.path.exists(thumbnail_path):
                    os.remove(thumbnail_path)

            duration = get_video_duration(output_path) or requested_duration
            self.orchestrator_service.update_job_status(
                self.job_id,
                "completed",
                storage_path=video_storage_path,
                thumbnail_storage_path=thumbnail_storage_path,
                duration_seconds=duration,
                prompt=self.positive_prompt,
                completion_metadata={
                    "renderer": "image_motion",
                    "frames": frames,
                    "width": width,
                    "height": height,
                    "vram_required_mb": 0,
                },
            )
            logging.info(
                "Image motion video job %s completed: %s",
                self.job_id,
                video_storage_path,
            )
        except TokenExpiredError:
            raise
        except Exception as exc:
            logging.error("Error finalizing image motion video: %s", exc, exc_info=True)
            self._fail_job(f"Error finalizing image motion video: {exc}")
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
