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

        explicit_alpha_cutout = any(
            value is True
            for value in (
                inputs.get("remove_background"),
                inputs.get("alpha_cutout"),
                metadata.get("remove_background"),
                metadata.get("alpha_cutout"),
            )
        )
        if explicit_alpha_cutout:
            return True

        transparent_hint = any(
            value is True
            for value in (
                inputs.get("transparent_background"),
                inputs.get("transparent_ready"),
                metadata.get("transparent_ready"),
            )
        )
        if transparent_hint:
            logging.info(
                "Skipping automatic alpha cutout for job %s; transparent-ready now means clean matte reference unless remove_background/alpha_cutout is explicit.",
                self.job_id,
            )

        return False

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
                threshold = 36 if brightness < 40 or brightness > 215 else 42
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

                mask = mask.filter(ImageFilter.MaxFilter(5)).filter(
                    ImageFilter.MinFilter(5)
                )

                # Bright matte backgrounds can leak through tiny anti-aliased
                # gaps in white clothing or hats. Fill transparent islands that
                # are enclosed by foreground before feathering the final cutout.
                closed_pixels = mask.load()
                exterior = bytearray(width * height)
                exterior_queue = deque()

                def enqueue_exterior(x: int, y: int) -> None:
                    idx = y * width + x
                    if exterior[idx] or closed_pixels[x, y] >= 128:
                        return
                    exterior[idx] = 1
                    exterior_queue.append((x, y))

                for x in range(width):
                    enqueue_exterior(x, 0)
                    enqueue_exterior(x, height - 1)
                for y in range(height):
                    enqueue_exterior(0, y)
                    enqueue_exterior(width - 1, y)

                while exterior_queue:
                    x, y = exterior_queue.popleft()
                    if x > 0:
                        enqueue_exterior(x - 1, y)
                    if x < width - 1:
                        enqueue_exterior(x + 1, y)
                    if y > 0:
                        enqueue_exterior(x, y - 1)
                    if y < height - 1:
                        enqueue_exterior(x, y + 1)

                filled_pixels = 0
                for y in range(height):
                    offset = y * width
                    for x in range(width):
                        if closed_pixels[x, y] < 128 and not exterior[offset + x]:
                            closed_pixels[x, y] = 255
                            filled_pixels += 1

                if filled_pixels:
                    logging.info(
                        "Filled %s enclosed alpha holes for job %s.",
                        filled_pixels,
                        self.job_id,
                    )

                foreground_pixels = mask.load()
                seen_foreground = bytearray(width * height)
                components = []

                for y in range(height):
                    offset = y * width
                    for x in range(width):
                        idx = offset + x
                        if seen_foreground[idx] or foreground_pixels[x, y] < 128:
                            continue
                        seen_foreground[idx] = 1
                        component_queue = deque([(x, y)])
                        points = []
                        min_x = max_x = x
                        min_y = max_y = y

                        while component_queue:
                            cx, cy = component_queue.popleft()
                            points.append((cx, cy))
                            if cx < min_x:
                                min_x = cx
                            if cx > max_x:
                                max_x = cx
                            if cy < min_y:
                                min_y = cy
                            if cy > max_y:
                                max_y = cy

                            for nx, ny in (
                                (cx - 1, cy),
                                (cx + 1, cy),
                                (cx, cy - 1),
                                (cx, cy + 1),
                            ):
                                if 0 <= nx < width and 0 <= ny < height:
                                    nidx = ny * width + nx
                                    if (
                                        not seen_foreground[nidx]
                                        and foreground_pixels[nx, ny] >= 128
                                    ):
                                        seen_foreground[nidx] = 1
                                        component_queue.append((nx, ny))

                        components.append(
                            {
                                "points": points,
                                "bbox": (min_x, min_y, max_x + 1, max_y + 1),
                            }
                        )

                if len(components) > 1:
                    components.sort(key=lambda item: len(item["points"]), reverse=True)
                    main = components[0]
                    main_area = len(main["points"])
                    main_left, main_top, main_right, main_bottom = main["bbox"]
                    expansion = 64
                    min_keep_area = max(3000, int(main_area * 0.012))
                    cleaned_mask = Image.new("L", (width, height), 0)
                    cleaned_pixels = cleaned_mask.load()
                    removed_components = 0
                    prompt_text = str(getattr(self, "positive_prompt", "") or "").lower()
                    allow_disjoint_identity = any(
                        marker in prompt_text
                        for marker in (
                            "floating gloved hands",
                            "floating hands",
                            "detached hands",
                            "orbiting hands",
                            "separate hands",
                        )
                    )

                    for index, component in enumerate(components):
                        left, top, right, bottom = component["bbox"]
                        is_near_main = not (
                            right < main_left - expansion
                            or left > main_right + expansion
                            or bottom < main_top - expansion
                            or top > main_bottom + expansion
                        )
                        keep_component = (
                            index == 0
                            or len(component["points"]) >= min_keep_area
                            or (allow_disjoint_identity and is_near_main)
                        )

                        if keep_component:
                            for px, py in component["points"]:
                                cleaned_pixels[px, py] = foreground_pixels[px, py]
                        else:
                            removed_components += 1

                    if removed_components:
                        logging.info(
                            "Removed %s detached alpha islands for job %s.",
                            removed_components,
                            self.job_id,
                        )
                        mask = cleaned_mask

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

    def _should_clean_reference_matte(self) -> bool:
        job = getattr(self, "job", None) or {}
        metadata = job.get("completion_metadata") or {}
        inputs = job.get("inputs") or {}
        return (
            metadata.get("target") == "character"
            and (metadata.get("transparent_ready") is True or inputs.get("transparent_ready") is True)
        )

    def _clean_reference_matte_artifacts(self, file_path: str) -> None:
        """Remove small detached decorations from flat-matte character references."""
        if not self._should_clean_reference_matte():
            return

        try:
            from PIL import Image

            with Image.open(file_path) as img:
                rgba = img.convert("RGBA")
                rgb = rgba.convert("RGB")
                width, height = rgb.size
                if width < 32 or height < 32:
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
                threshold = 34 if brightness < 40 or brightness > 215 else 42
                threshold_sq = threshold * threshold

                def near_background(pixel) -> bool:
                    return (
                        (pixel[0] - background[0]) ** 2
                        + (pixel[1] - background[1]) ** 2
                        + (pixel[2] - background[2]) ** 2
                    ) <= threshold_sq

                visited_background = bytearray(width * height)
                background_queue = deque()

                def enqueue_background(x: int, y: int) -> None:
                    idx = y * width + x
                    if visited_background[idx] or not near_background(pixels[x, y]):
                        return
                    visited_background[idx] = 1
                    background_queue.append((x, y))

                for x in range(width):
                    enqueue_background(x, 0)
                    enqueue_background(x, height - 1)
                for y in range(height):
                    enqueue_background(0, y)
                    enqueue_background(width - 1, y)

                while background_queue:
                    x, y = background_queue.popleft()
                    if x > 0:
                        enqueue_background(x - 1, y)
                    if x < width - 1:
                        enqueue_background(x + 1, y)
                    if y > 0:
                        enqueue_background(x, y - 1)
                    if y < height - 1:
                        enqueue_background(x, y + 1)

                seen_foreground = bytearray(width * height)
                components = []

                for y in range(height):
                    offset = y * width
                    for x in range(width):
                        idx = offset + x
                        if seen_foreground[idx] or visited_background[idx]:
                            continue
                        seen_foreground[idx] = 1
                        component_queue = deque([(x, y)])
                        points = []
                        min_x = max_x = x
                        min_y = max_y = y

                        while component_queue:
                            cx, cy = component_queue.popleft()
                            points.append((cx, cy))
                            min_x = min(min_x, cx)
                            max_x = max(max_x, cx)
                            min_y = min(min_y, cy)
                            max_y = max(max_y, cy)

                            for nx, ny in (
                                (cx - 1, cy),
                                (cx + 1, cy),
                                (cx, cy - 1),
                                (cx, cy + 1),
                            ):
                                if 0 <= nx < width and 0 <= ny < height:
                                    nidx = ny * width + nx
                                    if (
                                        not seen_foreground[nidx]
                                        and not visited_background[nidx]
                                    ):
                                        seen_foreground[nidx] = 1
                                        component_queue.append((nx, ny))

                        components.append(
                            {
                                "points": points,
                                "bbox": (min_x, min_y, max_x + 1, max_y + 1),
                            }
                        )

                if len(components) <= 1:
                    return

                components.sort(key=lambda item: len(item["points"]), reverse=True)
                main = components[0]
                main_area = len(main["points"])
                main_left, main_top, main_right, main_bottom = main["bbox"]
                expansion = max(36, min(width, height) // 12)
                min_keep_area = max(2600, int(main_area * 0.014))
                protected_left = int(width * 0.25)
                protected_right = int(width * 0.75)
                protected_top = int(height * 0.08)
                protected_bottom = int(height * 0.94)
                prompt_text = str(getattr(self, "positive_prompt", "") or "").lower()
                allow_disjoint_identity = any(
                    marker in prompt_text
                    for marker in (
                        "floating gloved hands",
                        "floating hands",
                        "detached hands",
                        "orbiting hands",
                        "separate hands",
                    )
                )

                removed_points = []
                for index, component in enumerate(components):
                    if index == 0:
                        continue
                    left, top, right, bottom = component["bbox"]
                    component_area = len(component["points"])
                    dark_pixels = 0
                    saturated_pixels = 0
                    bright_pixels = 0
                    for px, py in component["points"]:
                        red, green, blue = pixels[px, py]
                        component_brightness = (red + green + blue) / 3
                        component_chroma = max(red, green, blue) - min(red, green, blue)
                        if component_brightness < 115 or min(red, green, blue) < 75:
                            dark_pixels += 1
                        if component_chroma > 42 and component_brightness < 245:
                            saturated_pixels += 1
                        if component_brightness > 210:
                            bright_pixels += 1

                    dark_fraction = dark_pixels / component_area
                    saturated_fraction = saturated_pixels / component_area
                    bright_fraction = bright_pixels / component_area
                    low_ink_artifact = (
                        component_area < max(9000, int(main_area * 0.018))
                        and dark_fraction < 0.008
                        and saturated_fraction < 0.08
                        and bright_fraction > 0.45
                    )
                    tiny_detached_decoration = (
                        component_area < max(5200, int(main_area * 0.012))
                        and not allow_disjoint_identity
                        and (
                            bright_fraction > 0.32
                            or saturated_fraction > 0.18
                            or dark_fraction < 0.012
                        )
                    )
                    is_near_main = not (
                        right < main_left - expansion
                        or left > main_right + expansion
                        or bottom < main_top - expansion
                        or top > main_bottom + expansion
                    )
                    center_x = (left + right) // 2
                    center_y = (top + bottom) // 2
                    is_in_protected_center = (
                        protected_left <= center_x <= protected_right
                        and protected_top <= center_y <= protected_bottom
                    )
                    keep_component = not (
                        low_ink_artifact or tiny_detached_decoration
                    ) and (
                        component_area >= min_keep_area
                        or is_in_protected_center
                        or (
                            allow_disjoint_identity
                            and is_near_main
                            and (dark_fraction > 0.015 or saturated_fraction > 0.12)
                        )
                    )
                    if not keep_component:
                        removed_points.extend(component["points"])

                if not removed_points:
                    return

                cleaned = rgb.copy()
                cleaned_pixels = cleaned.load()
                for px, py in removed_points:
                    cleaned_pixels[px, py] = background
                cleaned.save(file_path, format="PNG")
                logging.info(
                    "Cleaned %s detached matte artifact pixels across %s components for character reference job %s.",
                    len(removed_points),
                    max(0, len(components) - 1),
                    self.job_id,
                )
        except Exception as exc:
            logging.warning(
                "Could not clean reference matte artifacts for job %s: %s",
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
            self._clean_reference_matte_artifacts(temp_host_path)
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
