"""
DreamID-Omni image+voice-to-video processor.

The ComfyUI node requires one reference image and one reference audio clip,
with an optional second image/audio pair for two-person dialogue. This
processor materializes those inputs into ComfyUI's input directory, injects the
24GB FP8 workflow, and returns the generated video.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from config import DEV_MODE, SUPABASE_URL
from services.docker_manager import docker_manager
from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import VideoOutputHandler
from utils.comfyui_workflow_utils import (
    inject_dreamid_omni_workflow,
    materialize_start_image,
)

logger = logging.getLogger(__name__)


class DreamIDOmniImageToVideoProcessor(ComfyUIProcessor, VideoOutputHandler):
    """DreamID-Omni talking-head video generation for one or two characters."""

    def _inputs(self) -> dict:
        inputs = self.job.get("inputs") or {}
        if isinstance(inputs, str):
            try:
                parsed = json.loads(inputs)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                logger.warning("Failed to parse DreamID-Omni inputs JSON")
                return {}
        return inputs if isinstance(inputs, dict) else {}

    def _download_storage_path(self, storage_path: Optional[str]) -> Optional[str]:
        if not storage_path:
            return None

        supabase_url = os.environ.get(
            "SUPABASE_URL", self.client.config.get("SUPABASE_URL", SUPABASE_URL)
        )
        if not supabase_url:
            return None

        bucket = self.job.get("bucket", "projects_public")
        source_url = f"{supabase_url}/storage/v1/object/public/{bucket}/{storage_path}"
        return self.orchestrator_service.download_asset_by_url(
            source_url, self.input_dir
        )

    def _resolve_reference_image(self, inputs: dict) -> Optional[str]:
        start_image_url = inputs.get("start_image_url")
        if start_image_url:
            path = self.orchestrator_service.download_asset_by_url(
                start_image_url, self.input_dir
            )
            if path:
                return path

        filename = materialize_start_image(self.job, self.input_dir)
        if filename:
            return os.path.join(self.input_dir, filename)

        storage_path = self.job.get("input_storage_path")
        if not storage_path:
            maybe_path = self.job.get("start_image_base64") or inputs.get(
                "start_image_base64"
            )
            if (
                isinstance(maybe_path, str)
                and not maybe_path.startswith("data:")
                and len(maybe_path) < 2048
            ):
                storage_path = maybe_path

        return self._download_storage_path(storage_path)

    def _resolve_second_reference_image(self, inputs: dict) -> Optional[str]:
        image_url = (
            inputs.get("reference_image_2_url")
            or inputs.get("reference_image2_url")
            or inputs.get("ref_image2_url")
            or inputs.get("second_reference_image_url")
            or inputs.get("second_image_url")
        )
        if image_url:
            path = self.orchestrator_service.download_asset_by_url(
                image_url, self.input_dir
            )
            if path:
                return path

        storage_path = (
            inputs.get("reference_image_2")
            or inputs.get("reference_image2")
            or inputs.get("reference_image_2_storage_path")
            or inputs.get("reference_image2_storage_path")
            or inputs.get("ref_image2_storage_path")
            or inputs.get("second_reference_image")
            or inputs.get("second_reference_image_storage_path")
            or inputs.get("second_image")
            or inputs.get("second_image_storage_path")
        )
        if isinstance(storage_path, str):
            return self._download_storage_path(storage_path)

        return None

    def _resolve_reference_audio(
        self, inputs: dict, *, second: bool = False
    ) -> Optional[str]:
        if second:
            audio_url = (
                inputs.get("reference_audio_2_url")
                or inputs.get("reference_audio2_url")
                or inputs.get("ref_audio2_url")
                or inputs.get("second_reference_audio_url")
                or inputs.get("second_audio_url")
            )
        else:
            audio_url = (
                inputs.get("reference_audio_url")
                or inputs.get("voice_audio_url")
                or inputs.get("ref_audio_url")
            )
        if audio_url:
            path = self.orchestrator_service.download_asset_by_url(
                audio_url, self.input_dir
            )
            if path:
                return path

        if second:
            storage_path = (
                inputs.get("reference_audio_2")
                or inputs.get("reference_audio2")
                or inputs.get("reference_audio_2_storage_path")
                or inputs.get("reference_audio2_storage_path")
                or inputs.get("ref_audio2_storage_path")
                or inputs.get("second_reference_audio")
                or inputs.get("second_reference_audio_storage_path")
                or inputs.get("second_audio")
                or inputs.get("second_audio_storage_path")
            )
        else:
            storage_path = (
                inputs.get("reference_audio")
                or inputs.get("reference_audio_storage_path")
                or inputs.get("voice_audio")
                or inputs.get("voice_clone_storage_path")
                or inputs.get("ref_audio_storage_path")
            )
        if isinstance(storage_path, str):
            return self._download_storage_path(storage_path)

        return None

    def _copy_to_comfy_input(self, local_path: str) -> Optional[str]:
        filename = os.path.basename(local_path)
        container_input_path = f"/opt/ComfyUI/input/{filename}"
        try:
            if docker_manager:
                docker_manager.copy_file_to_container(
                    service_type=self.client.active_service_type,
                    source_on_host=local_path,
                    dest_in_container=container_input_path,
                    shutdown_event=self.shutdown_event,
                )
            else:
                import shutil

                os.makedirs(os.path.dirname(container_input_path), exist_ok=True)
                shutil.copy2(local_path, container_input_path)
                logger.info("Copied DreamID-Omni input to %s", container_input_path)
            return filename
        except Exception as exc:
            self._fail_job(
                f"Failed to copy DreamID-Omni input to ComfyUI for job {self.job_id}: {exc}"
            )
            return None

    def _normalize_prompt(self, prompt: str, *, two_person: bool = False) -> str:
        clean_prompt = (prompt or "").strip()
        if not clean_prompt:
            clean_prompt = (
                "Hello, this is a short natural introduction."
                if not two_person
                else "Hello, it is good to see you here."
            )

        if not two_person and "<img1>" in clean_prompt and "<sub1>" in clean_prompt:
            return clean_prompt

        if (
            two_person
            and "<img1>" in clean_prompt
            and "<img2>" in clean_prompt
            and "<sub1>" in clean_prompt
            and "<sub2>" in clean_prompt
        ):
            return clean_prompt

        if "<S>" in clean_prompt and "<E>" in clean_prompt:
            speech_line = clean_prompt
        elif two_person:
            speech_line = (
                f"<sub1> says, <S>{clean_prompt}<E> "
                "<sub2> listens, then replies with a natural reaction."
            )
        else:
            speech_line = f"<sub1> says, <S>{clean_prompt}<E>"

        if two_person:
            return (
                "<img1>: The person in the first reference image is identified as <sub1>.\n"
                "<img2>: The person in the second reference image is identified as <sub2>.\n"
                "**Overall Environment/Scene**: A clean two-person conversation shot "
                "with natural lighting and subtle background motion.\n"
                "**Main Characters/Subjects Actions**: <sub1> and <sub2> face each "
                "other, exchange dialogue naturally, and maintain believable eye lines.\n"
                f"{speech_line}"
            )

        return (
            "<img1>: The person in the reference image is identified as <sub1>.\n"
            "**Overall Environment/Scene**: A clean upper-body talking-head shot "
            "with natural lighting and subtle background motion.\n"
            "**Main Characters/Subjects Actions**: <sub1> faces the camera, "
            "speaks naturally, and maintains believable eye contact.\n"
            f"{speech_line}"
        )

    def process(self):
        if DEV_MODE:
            return

        if not self.job:
            self._fail_job("Job object is None for DreamIDOmniImageToVideoProcessor.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self._inputs()
        image_path = self._resolve_reference_image(inputs)
        if not image_path:
            self._fail_job(
                "DreamID-Omni requires a reference face image. Use the image-to-video workflow."
            )
            return

        audio_path = self._resolve_reference_audio(inputs)
        if not audio_path:
            self._fail_job(
                "DreamID-Omni requires a reference voice audio clip. Upload or drop a voice reference."
            )
            return

        image2_path = self._resolve_second_reference_image(inputs)
        audio2_path = self._resolve_reference_audio(inputs, second=True)
        if bool(image2_path) != bool(audio2_path):
            self._fail_job(
                "DreamID-Omni second character requires both a second face image and a second voice clip."
            )
            return

        image_filename = self._copy_to_comfy_input(image_path)
        audio_filename = self._copy_to_comfy_input(audio_path)
        if not image_filename or not audio_filename:
            return
        image2_filename = (
            self._copy_to_comfy_input(image2_path) if image2_path else None
        )
        audio2_filename = (
            self._copy_to_comfy_input(audio2_path) if audio2_path else None
        )
        if bool(image2_path or audio2_path) and (
            not image2_filename or not audio2_filename
        ):
            return

        try:
            workflow = inject_dreamid_omni_workflow(
                workflow_data,
                self._normalize_prompt(
                    self.positive_prompt, two_person=bool(image2_filename)
                ),
                image_filename,
                audio_filename,
                reference_image2_filename=image2_filename,
                reference_audio2_filename=audio2_filename,
                aspect_ratio=inputs.get("aspect_ratio", "16:9"),
                steps=inputs.get("steps"),
                seed=inputs.get("seed"),
                solver_name=inputs.get("solver_name")
                or inputs.get("sampler")
                or "unipc",
            )
        except Exception as exc:
            self._fail_job(
                f"Failed to prepare DreamID-Omni workflow for job {self.job_id}: {exc}"
            )
            return

        outputs = self._trigger_and_get_output({"prompt": workflow})
        if not outputs:
            return

        result = self.handle_video_output(outputs)
        if not result:
            return

        video_storage_path, thumbnail_storage_path, duration = result
        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=video_storage_path,
            thumbnail_storage_path=thumbnail_storage_path,
            duration_seconds=duration,
            prompt=self.positive_prompt,
        )
