"""
Output Handlers

Mixin classes for handling different output types (video, audio, image).
These eliminate code duplication across processors.
"""

import os
import logging
from collections import deque
from typing import Union, Tuple

from services.docker_manager import docker_manager
from config import THUMBNAIL_WIDTH
from utils.media_utils import (
    find_video_in_output,
    find_audio_in_output,
    find_audio_file_in_directory,
    find_image_in_output,
    generate_thumbnail,
    get_video_duration,
    get_audio_duration,
)


class OutputHandlerMixin:
    """Base mixin providing common output handling utilities."""

    def _copy_file_from_container(self, filename: str, subfolder: str) -> Union[str, None]:
        """
        Copies a file from the ComfyUI output directory to a temporary location.
        
        In headless mode (cloud deployment), files are on the local filesystem.
        In Docker mode, files need to be copied from the container.
        """
        from config import HEADLESS_MODE
        import shutil
        
        safe_filename = os.path.basename(filename)
        os.makedirs(self.cache_dir, exist_ok=True)
        temp_filename = f"{self.job_id}_{safe_filename}"
        dest_on_host = os.path.join(self.cache_dir, temp_filename)
        
        # Build source path - same structure whether local or in container
        source_path = os.path.join("/opt/ComfyUI/output", subfolder, safe_filename).replace("\\", "/")

        if HEADLESS_MODE:
            # Headless mode: files are directly on the local filesystem
            try:
                if os.path.exists(source_path):
                    shutil.copy2(source_path, dest_on_host)
                    logging.info(f"Headless mode: copied file from {source_path} to {dest_on_host}")
                    return dest_on_host
                else:
                    logging.error(f"Headless mode: source file not found at {source_path}")
                    return None
            except Exception as e:
                logging.error(f"Headless mode: failed to copy file: {e}", exc_info=True)
                return None
        else:
            # Docker mode: copy file from container
            if not self.client.active_service_type:
                logging.error("Cannot copy from container: no active service type is set.")
                return None

            import time

            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    if os.path.exists(dest_on_host):
                        os.remove(dest_on_host)
                    docker_manager.copy_file_from_container(
                        service_type=self.client.active_service_type,
                        source_in_container=source_path,
                        dest_on_host=dest_on_host,
                        shutdown_event=self.shutdown_event,
                    )
                    if os.path.exists(dest_on_host):
                        logging.info(f"Successfully copied file to temporary host path: {dest_on_host}")
                        return dest_on_host
                    raise RuntimeError("docker cp command finished but destination file does not exist.")
                except Exception as e:
                    if attempt < max_attempts and not self.shutdown_event.is_set():
                        logging.warning(
                            "Copy from container failed on attempt %s/%s; retrying shortly: %s",
                            attempt,
                            max_attempts,
                            e,
                        )
                        time.sleep(min(2 * attempt, 5))
                        continue
                    logging.error(f"Failed to copy file from container: {e}", exc_info=True)
                    return None


class VideoOutputHandler(OutputHandlerMixin):
    """Mixin for handling video output: find, copy, thumbnail, upload, duration."""

    def handle_video_output(self, outputs) -> Union[Tuple[str, str, float], None]:
        """
        Process video output from ComfyUI workflow.

        Returns:
            Tuple of (video_storage_path, thumbnail_storage_path, duration) or None on failure.
        """
        video_info = find_video_in_output(outputs)
        if not video_info:
            self._fail_job(f"Workflow for job {self.job_id} completed, but no video file found.")
            return None

        video_filename, subfolder = video_info
        temp_host_path = self._copy_file_from_container(video_filename, subfolder)
        if not temp_host_path:
            self._fail_job(f"Failed to copy output file from container for job {self.job_id}.")
            return None

        try:
            video_storage_path = self.orchestrator_service.upload_output(temp_host_path, self.job_id, "video/mp4")
            if not video_storage_path:
                self._fail_job(f"Video upload failed for job {self.job_id}.")
                return None

            thumbnail_storage_path = None
            thumbnail_local_path = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")

            if generate_thumbnail(temp_host_path, thumbnail_local_path, width=THUMBNAIL_WIDTH):
                thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(thumbnail_local_path, self.job_id)
                if os.path.exists(thumbnail_local_path):
                    os.remove(thumbnail_local_path)

            duration = get_video_duration(temp_host_path)

            return video_storage_path, thumbnail_storage_path, duration
        finally:
            if os.path.exists(temp_host_path):
                logging.info(f"Cleaning up temporary file: {temp_host_path}")
                os.remove(temp_host_path)


class AudioOutputHandler(OutputHandlerMixin):
    """Mixin for handling audio output: find, copy, upload, duration."""

    def handle_audio_output(self, outputs, scan_directory: bool = False) -> Union[Tuple[str, float], None]:
        """
        Process audio output from ComfyUI workflow.

        Args:
            outputs: The workflow outputs dict
            scan_directory: If True, scan output directory for audio files if not found in outputs

        Returns:
            Tuple of (audio_storage_path, duration) or None on failure.
        """
        audio_info = find_audio_in_output(outputs)

        if not audio_info and scan_directory:
            logging.info(f"Audio not found in workflow outputs for job {self.job_id}, scanning output directory...")
            from config import HEADLESS_MODE
            if HEADLESS_MODE:
                output_dir = "/opt/ComfyUI/output"
                audio_info = find_audio_file_in_directory(output_dir, self.job_id)
            elif self.client.active_service_type:
                output_dir = docker_manager.get_output_dir_path(self.client.active_service_type)
                audio_info = find_audio_file_in_directory(output_dir, self.job_id)

        if not audio_info:
            self._fail_job(f"Workflow for job {self.job_id} completed, but no audio file found.")
            return None

        audio_filename, subfolder = audio_info
        temp_host_path = self._copy_file_from_container(audio_filename, subfolder)
        if not temp_host_path:
            self._fail_job(f"Failed to copy output file from container for job {self.job_id}.")
            return None

        try:
            audio_storage_path = self.orchestrator_service.upload_audio_output(temp_host_path, self.job_id)
            if not audio_storage_path:
                self._fail_job(f"Audio upload failed for job {self.job_id}.")
                return None

            duration = get_audio_duration(temp_host_path)
            return audio_storage_path, duration
        finally:
            if os.path.exists(temp_host_path):
                logging.info(f"Cleaning up temporary file: {temp_host_path}")
                os.remove(temp_host_path)


class ImageOutputHandler(OutputHandlerMixin):
    """Mixin for handling image output: find, copy, optionally normalize, upload."""

    def _normalize_image_to_dimensions(self, file_path: str, target_dimensions) -> None:
        if not target_dimensions:
            return

        try:
            target_width, target_height = target_dimensions
            target_width = int(target_width)
            target_height = int(target_height)
        except (TypeError, ValueError):
            return

        if target_width <= 0 or target_height <= 0:
            return

        try:
            from PIL import Image

            with Image.open(file_path) as img:
                source_width, source_height = img.size
                if source_width <= 0 or source_height <= 0:
                    return

                target_ratio = target_width / target_height
                source_ratio = source_width / source_height

                if (
                    (source_width, source_height) == (target_width, target_height)
                    and abs(source_ratio - target_ratio) < 0.001
                ):
                    return

                if source_ratio > target_ratio:
                    crop_width = max(1, int(round(source_height * target_ratio)))
                    left = max(0, (source_width - crop_width) // 2)
                    box = (left, 0, left + crop_width, source_height)
                else:
                    crop_height = max(1, int(round(source_width / target_ratio)))
                    top = max(0, (source_height - crop_height) // 2)
                    box = (0, top, source_width, top + crop_height)

                resampling = getattr(
                    getattr(Image, "Resampling", Image),
                    "LANCZOS",
                    Image.LANCZOS,
                )
                normalized = img.crop(box)
                if normalized.size != (target_width, target_height):
                    normalized = normalized.resize(
                        (target_width, target_height),
                        resampling,
                    )

                if normalized.mode not in ("RGB", "RGBA"):
                    normalized = normalized.convert("RGB")
                normalized.save(file_path, format="PNG")
                logging.info(
                    "Normalized image output for job %s from %sx%s to %sx%s.",
                    self.job_id,
                    source_width,
                    source_height,
                    target_width,
                    target_height,
                )
        except Exception as exc:
            logging.warning(
                "Could not normalize image output for job %s: %s",
                self.job_id,
                exc,
            )

    def _should_make_transparent_background(self) -> bool:
        job = getattr(self, "job", None) or {}
        inputs = job.get("inputs") or {}
        metadata = job.get("completion_metadata") or {}

        return any(
            value is True
            for value in (
                inputs.get("transparent_background"),
                inputs.get("transparent_ready"),
                inputs.get("remove_background"),
                metadata.get("transparent_ready"),
            )
        )

    def _apply_transparent_background_if_requested(self, file_path: str) -> None:
        """Turn clean plain-background reference renders into alpha cutouts."""
        if not self._should_make_transparent_background():
            return

        try:
            from PIL import Image, ImageFilter

            with Image.open(file_path) as img:
                rgba = img.convert("RGBA")
                alpha = rgba.getchannel("A")
                if alpha.getextrema()[0] < 255:
                    logging.info(
                        "Image output for job %s already contains transparency.",
                        self.job_id,
                    )
                    return

                rgb = rgba.convert("RGB")
                width, height = rgb.size
                if width < 4 or height < 4:
                    return

                pixels = rgb.load()
                edge_pixels = []
                for x in range(width):
                    edge_pixels.append(pixels[x, 0])
                    edge_pixels.append(pixels[x, height - 1])
                for y in range(height):
                    edge_pixels.append(pixels[0, y])
                    edge_pixels.append(pixels[width - 1, y])

                def median_channel(index: int) -> int:
                    values = sorted(pixel[index] for pixel in edge_pixels)
                    return values[len(values) // 2]

                background = tuple(median_channel(i) for i in range(3))
                brightness = sum(background) / 3
                threshold = 52 if brightness < 40 or brightness > 215 else 42
                threshold_sq = threshold * threshold

                def near_background(pixel) -> bool:
                    return (
                        (pixel[0] - background[0]) ** 2
                        + (pixel[1] - background[1]) ** 2
                        + (pixel[2] - background[2]) ** 2
                    ) <= threshold_sq

                visited = bytearray(width * height)
                queue = deque()

                def enqueue(x: int, y: int) -> None:
                    idx = y * width + x
                    if visited[idx] or not near_background(pixels[x, y]):
                        return
                    visited[idx] = 1
                    queue.append((x, y))

                for x in range(width):
                    enqueue(x, 0)
                    enqueue(x, height - 1)
                for y in range(height):
                    enqueue(0, y)
                    enqueue(width - 1, y)

                while queue:
                    x, y = queue.popleft()
                    if x > 0:
                        enqueue(x - 1, y)
                    if x < width - 1:
                        enqueue(x + 1, y)
                    if y > 0:
                        enqueue(x, y - 1)
                    if y < height - 1:
                        enqueue(x, y + 1)

                mask = Image.new("L", (width, height), 255)
                mask_pixels = mask.load()
                removed_pixels = 0
                for y in range(height):
                    offset = y * width
                    for x in range(width):
                        if visited[offset + x]:
                            mask_pixels[x, y] = 0
                            removed_pixels += 1

                removed_ratio = removed_pixels / float(width * height)
                if removed_ratio < 0.02:
                    logging.info(
                        "Transparent-background pass for job %s found no removable plain background.",
                        self.job_id,
                    )
                    return

                mask = mask.filter(ImageFilter.MedianFilter(3)).filter(
                    ImageFilter.GaussianBlur(0.7)
                )
                rgba.putalpha(mask)
                rgba.save(file_path, format="PNG")
                logging.info(
                    "Applied transparent-background alpha cutout for job %s; removed %.1f%% of pixels.",
                    self.job_id,
                    removed_ratio * 100,
                )
        except Exception as exc:
            logging.warning(
                "Could not apply transparent background for job %s: %s",
                self.job_id,
                exc,
            )

    def handle_image_output(self, outputs, target_dimensions=None) -> Union[str, None]:
        """
        Process image output from ComfyUI workflow.

        Returns:
            image_storage_path or None on failure.
        """
        image_info = find_image_in_output(outputs)
        if not image_info:
            self._fail_job(f"Workflow for job {self.job_id} completed, but no image file found.")
            return None

        image_filename, subfolder = image_info
        temp_host_path = self._copy_file_from_container(image_filename, subfolder)
        if not temp_host_path:
            self._fail_job(f"Failed to copy output file from container for job {self.job_id}.")
            return None

        try:
            self._normalize_image_to_dimensions(temp_host_path, target_dimensions)
            self._apply_transparent_background_if_requested(temp_host_path)
            image_storage_path = self.orchestrator_service.upload_image_output(temp_host_path, self.job_id)
            if not image_storage_path:
                self._fail_job(f"Image upload failed for job {self.job_id}.")
                return None

            return image_storage_path
        finally:
            if os.path.exists(temp_host_path):
                logging.info(f"Cleaning up temporary file: {temp_host_path}")
                os.remove(temp_host_path)
