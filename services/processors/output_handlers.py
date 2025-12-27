"""
Output Handlers

Mixin classes for handling different output types (video, audio, image).
These eliminate code duplication across processors.
"""

import os
import logging
from typing import Union, Tuple

from services.docker_manager import docker_manager
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

            try:
                docker_manager.copy_file_from_container(
                    service_type=self.client.active_service_type,
                    source_in_container=source_path,
                    dest_on_host=dest_on_host,
                    shutdown_event=self.shutdown_event,
                )
                if os.path.exists(dest_on_host):
                    logging.info(f"Successfully copied file to temporary host path: {dest_on_host}")
                    return dest_on_host
                else:
                    raise RuntimeError("docker cp command finished but destination file does not exist.")
            except Exception as e:
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

            if generate_thumbnail(temp_host_path, thumbnail_local_path, width=100):
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
    """Mixin for handling image output: find, copy, upload."""

    def handle_image_output(self, outputs) -> Union[str, None]:
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
            image_storage_path = self.orchestrator_service.upload_image_output(temp_host_path, self.job_id)
            if not image_storage_path:
                self._fail_job(f"Image upload failed for job {self.job_id}.")
                return None

            return image_storage_path
        finally:
            if os.path.exists(temp_host_path):
                logging.info(f"Cleaning up temporary file: {temp_host_path}")
                os.remove(temp_host_path)
