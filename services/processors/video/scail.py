"""
SCAIL image-to-video processor (Wan2GP backend).

SCAIL animates a reference image from a driving video. Wan2GP handles the
NLF pose extraction/preprocess internally when it receives video_guide.
SCAIL-2 is also routed through WanGP, using its native scail2_14B model support
and lower-VRAM int8 weights.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Optional

from PIL import Image

from config import DEV_MODE
from services.docker_manager import docker_manager
from services.processors.wan2gp_processor import Wan2GPProcessor
from services.processors.video.last_frame import materialize_last_frame_start_image
from utils.comfyui_workflow_utils import materialize_start_image
from utils.media_utils import VIDEO_EXTENSIONS

logger = logging.getLogger(__name__)

_FPS = 16
_DEFAULT_DURATION_SECONDS = 5.0
_DEFAULT_DURATION_SECONDS_16GB = 4.0
_DEFAULT_SEED = -1

_SCAIL_RESOLUTIONS = {
    "16:9": "896x512",
    "9:16": "512x896",
    "1:1": "640x640",
    "4:3": "768x576",
    "3:4": "576x768",
    "21:9": "1024x448",
    "2:1": "896x448",
}

_SCAIL_RESOLUTIONS_16GB = {
    "16:9": "768x432",
    "9:16": "432x768",
    "1:1": "544x544",
    "4:3": "640x480",
    "3:4": "480x640",
    "21:9": "768x336",
    "2:1": "768x384",
}

_SCAIL2_RESOLUTIONS = {
    "16:9": "832x480",
    "9:16": "480x832",
    "1:1": "640x640",
    "4:3": "768x576",
    "3:4": "576x768",
    "21:9": "960x416",
    "2:1": "896x448",
}

_SCAIL2_RESOLUTIONS_16GB = {
    "16:9": "832x480",
    "9:16": "480x832",
    "1:1": "640x640",
    "4:3": "768x576",
    "3:4": "576x768",
    "21:9": "896x384",
    "2:1": "832x416",
}


def get_scail_vram_tier(service_type: str) -> str:
    return "16gb" if "16gb" in (service_type or "").lower() else "24gb"


def scail_resolution(aspect_ratio: str, vram_tier: str = "24gb") -> str:
    if vram_tier == "16gb":
        return _SCAIL_RESOLUTIONS_16GB.get(aspect_ratio, "768x432")
    return _SCAIL_RESOLUTIONS.get(aspect_ratio, "896x512")


def clamp_scail_duration(requested_duration, vram_tier: str = "24gb") -> float:
    default_duration = (
        _DEFAULT_DURATION_SECONDS_16GB
        if vram_tier == "16gb"
        else _DEFAULT_DURATION_SECONDS
    )
    max_duration = 4.0 if vram_tier == "16gb" else 5.0
    try:
        duration = float(requested_duration)
    except (TypeError, ValueError):
        duration = default_duration
    return max(1.0, min(duration, max_duration))


def clamp_scail_steps(requested_steps) -> int:
    try:
        steps = int(requested_steps)
    except (TypeError, ValueError):
        steps = 8
    return max(6, min(steps, 20))


def scail2_resolution(aspect_ratio: str, vram_tier: str = "24gb") -> str:
    if vram_tier == "16gb":
        return _SCAIL2_RESOLUTIONS_16GB.get(aspect_ratio, "832x480")
    return _SCAIL2_RESOLUTIONS.get(aspect_ratio, "832x480")


def clamp_scail2_duration(requested_duration, vram_tier: str = "24gb") -> float:
    default_duration = 5.0
    max_duration = 5.0
    try:
        duration = float(requested_duration)
    except (TypeError, ValueError):
        duration = default_duration
    return max(1.0, min(duration, max_duration))


def clamp_scail2_steps(requested_steps) -> int:
    try:
        steps = int(requested_steps)
    except (TypeError, ValueError):
        steps = 40
    return max(8, min(steps, 50))


def clamp_scail2_float(value, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def clamp_scail2_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def parse_scail_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def duration_to_wangp_frames(duration_seconds: float) -> int:
    frame_count = int(duration_seconds * _FPS) + 1
    return max(17, ((frame_count - 1) // 4) * 4 + 1)


def _storage_value_path(value: str) -> str:
    return value.split("|", 1)[0].strip()


def _looks_like_video_reference(value: str) -> bool:
    path = _storage_value_path(value).split("?", 1)[0].lower()
    return path.endswith(VIDEO_EXTENSIONS)


class SCAILImageToVideoProcessor(Wan2GPProcessor):
    """SCAIL reference-image + driving-video animation via Wan2GP."""

    SERVICE_NAME = "SCAIL"
    MAX_GENERATION_SECONDS = 7200

    def _download_storage_path(
        self,
        storage_path: str,
        bucket_hint: Optional[str] = None,
    ) -> Optional[str]:
        bucket = bucket_hint or self.job.get("bucket", "projects_public")
        return self.orchestrator_service.download_storage_asset(
            bucket,
            _storage_value_path(storage_path),
            self.input_dir,
        )

    def _resolve_reference_image(self, inputs: dict) -> Optional[str]:
        last_frame_path = materialize_last_frame_start_image(self, inputs)
        if last_frame_path:
            return last_frame_path

        url = inputs.get("start_image_url")
        if url:
            path = self.orchestrator_service.download_asset_by_url(url, self.input_dir)
            if path:
                return path

        filename = materialize_start_image(self.job, self.input_dir)
        if filename:
            return os.path.join(self.input_dir, filename)

        storage_path = self.job.get("input_storage_path")
        if not storage_path:
            maybe = self.job.get("inputs", {}).get("start_image_base64")
            if (
                maybe
                and isinstance(maybe, str)
                and not maybe.startswith("data:")
                and len(maybe) < 2048
            ):
                storage_path = maybe

        if storage_path:
            return self._download_storage_path(storage_path)

        return None

    def _resolve_pose_video(self, inputs: dict) -> Optional[str]:
        for key in ("pose_video_url", "driving_video_url", "video_guide_url"):
            url = inputs.get(key)
            if url:
                path = self.orchestrator_service.download_asset_by_url(
                    url, self.input_dir
                )
                if path:
                    return path

        for key in ("pose_video", "driving_video", "video_guide"):
            value = inputs.get(key)
            if not value or not isinstance(value, str):
                continue

            if value.startswith(("http://", "https://")):
                path = self.orchestrator_service.download_asset_by_url(
                    value, self.input_dir
                )
                if path:
                    return path
                continue

            local_path = _storage_value_path(value)
            if os.path.exists(local_path):
                return local_path

            if len(value) < 2048 and _looks_like_video_reference(value):
                bucket_hint = (
                    inputs.get(f"{key}_bucket")
                    or inputs.get("pose_video_bucket")
                    or inputs.get("video_guide_bucket")
                )
                path = self._download_storage_path(value, bucket_hint)
                if path:
                    return path
            elif len(value) < 2048:
                logger.warning(
                    "Ignoring non-video SCAIL driving input %s=%s",
                    key,
                    value,
                )

        input_video_url = self.job.get("input_video_url") or inputs.get(
            "input_video_url"
        )
        if input_video_url and _looks_like_video_reference(str(input_video_url)):
            return self.orchestrator_service.download_asset_by_url(
                input_video_url, self.input_dir
            )
        if input_video_url:
            logger.debug("Ignoring non-video input_video_url for SCAIL driving video.")

        return None

    def _prepare_pose_video_for_wan2gp(self, pose_video_path: str) -> Optional[str]:
        filename = f"{self.job_id}_{os.path.basename(pose_video_path)}"
        container_path = f"/opt/wan2gp/input/{filename}"

        try:
            if docker_manager:
                service_type = (
                    self.client.active_service_type or self.job.get("service_type")
                )
                if not service_type:
                    raise RuntimeError("No active Wan2GP service type is available")
                docker_manager.copy_file_to_container(
                    service_type=service_type,
                    source_on_host=pose_video_path,
                    dest_in_container=container_path,
                    shutdown_event=self.shutdown_event,
                )
                return container_path

            os.makedirs(os.path.dirname(container_path), exist_ok=True)
            if os.path.abspath(pose_video_path) != os.path.abspath(container_path):
                shutil.copy2(pose_video_path, container_path)
            return container_path
        except Exception as exc:
            self._fail_job(
                f"Failed to prepare SCAIL driving video for job {self.job_id}: {exc}"
            )
            return None

    def _build_settings(
        self,
        inputs: dict,
        start_image: Image.Image,
        pose_video_path: str,
    ) -> dict:
        service_type = self.client.active_service_type or self.job.get("service_type")
        vram_tier = get_scail_vram_tier(service_type or "")
        duration = clamp_scail_duration(inputs.get("duration"), vram_tier)
        return {
            "model_type": "scail",
            "prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "image_start": start_image,
            "video_guide": pose_video_path,
            "video_prompt_type": inputs.get("video_prompt_type", "V#1#"),
            "image_prompt_type": "S",
            "audio_prompt_type": "R",
            "resolution": scail_resolution(
                inputs.get("aspect_ratio", "16:9"), vram_tier
            ),
            "video_length": duration_to_wangp_frames(duration),
            "force_fps": "control",
            "num_inference_steps": clamp_scail_steps(inputs.get("steps")),
            "guidance_scale": float(inputs.get("cfg_scale", 1.0)),
            "flow_shift": float(inputs.get("flow_shift", 8.0)),
            "sample_solver": inputs.get("sampler", "unipc"),
            "sliding_window_size": 81,
            "sliding_window_overlap": 1,
            "sliding_window_color_correction_strength": 1,
            "seed": self.normalize_seed(inputs.get("seed", _DEFAULT_SEED)),
        }

    def process(self):
        if DEV_MODE:
            return

        if not self.job:
            self._fail_job("Job object is None.")
            return

        inputs = self.job.get("inputs") or {}
        image_path = self._resolve_reference_image(inputs)
        if not image_path:
            self._fail_job(
                "SCAIL requires a reference image. Use the image-to-video workflow."
            )
            return

        pose_video_path = self._resolve_pose_video(inputs)
        if not pose_video_path:
            self._fail_job(
                "SCAIL requires a driving video in pose_video/video_guide inputs."
            )
            return

        wan2gp_pose_video_path = self._prepare_pose_video_for_wan2gp(
            pose_video_path
        )
        if not wan2gp_pose_video_path:
            return

        try:
            start_image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            self._fail_job(
                f"Failed to open SCAIL reference image for job {self.job_id}: {exc}"
            )
            return

        settings = self._build_settings(inputs, start_image, wan2gp_pose_video_path)
        service_type = self.client.active_service_type or self.job.get("service_type")
        vram_tier = get_scail_vram_tier(service_type or "")
        logger.info(
            "Processing SCAIL job %s with tier=%s resolution=%s frames=%s pose_video=%s",
            self.job_id,
            vram_tier,
            settings["resolution"],
            settings["video_length"],
            os.path.basename(pose_video_path),
        )

        files = self._run_task(settings)
        if not files:
            if (
                not self.is_cancelled()
                and not self.infrastructure_interrupted
            ):
                self._fail_job(self._wan2gp_no_output_message("SCAIL/Wan2GP"))
            return

        result = self._handle_video_output(files[0])
        if not result:
            if not self.is_cancelled():
                self._fail_job(f"Failed to process video output for job {self.job_id}")
            return

        video_storage_path, thumbnail_storage_path, actual_duration = result
        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=video_storage_path,
            thumbnail_storage_path=thumbnail_storage_path,
            duration_seconds=actual_duration,
            prompt=self.positive_prompt,
        )


class SCAIL2ImageToVideoProcessor(Wan2GPProcessor):
    """SCAIL-2 reference-image + driving-video animation via WanGP."""

    SERVICE_NAME = "SCAIL-2"
    MAX_GENERATION_SECONDS = 10800

    def _download_storage_path(
        self,
        storage_path: str,
        bucket_hint: Optional[str] = None,
    ) -> Optional[str]:
        return SCAILImageToVideoProcessor._download_storage_path(
            self,
            storage_path,
            bucket_hint,
        )

    def _resolve_reference_image(self, inputs: dict) -> Optional[str]:
        return SCAILImageToVideoProcessor._resolve_reference_image(self, inputs)

    def _resolve_pose_video(self, inputs: dict) -> Optional[str]:
        return SCAILImageToVideoProcessor._resolve_pose_video(self, inputs)

    def _prepare_pose_video_for_scail2(self, pose_video_path: str) -> Optional[str]:
        filename = f"{self.job_id}_{os.path.basename(pose_video_path)}"
        container_path = f"/opt/wan2gp/input/{filename}"

        try:
            if docker_manager:
                service_type = (
                    self.client.active_service_type or self.job.get("service_type")
                )
                if not service_type:
                    raise RuntimeError("No active SCAIL-2 service type is available")
                docker_manager.copy_file_to_container(
                    service_type=service_type,
                    source_on_host=pose_video_path,
                    dest_in_container=container_path,
                    shutdown_event=self.shutdown_event,
                )
                return container_path

            os.makedirs(os.path.dirname(container_path), exist_ok=True)
            if os.path.abspath(pose_video_path) != os.path.abspath(container_path):
                shutil.copy2(pose_video_path, container_path)
            return container_path
        except Exception as exc:
            self._fail_job(
                f"Failed to prepare SCAIL-2 driving video for job {self.job_id}: {exc}"
            )
            return None

    def _build_settings(
        self,
        inputs: dict,
        start_image: Image.Image,
        pose_video_path: str,
    ) -> dict:
        service_type = self.client.active_service_type or self.job.get("service_type")
        vram_tier = get_scail_vram_tier(service_type or "")
        duration = clamp_scail2_duration(inputs.get("duration"), vram_tier)
        return {
            "model_type": "scail2_14B",
            "job_id": str(self.job_id),
            "prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "image_start": start_image,
            "video_guide": pose_video_path,
            "video_prompt_type": inputs.get("video_prompt_type", "V1"),
            "image_prompt_type": "S",
            "audio_prompt_type": "R",
            "resolution": scail2_resolution(
                inputs.get("aspect_ratio", "16:9"), vram_tier
            ),
            "duration_seconds": duration,
            "video_length": duration_to_wangp_frames(duration),
            "force_fps": "control",
            "num_inference_steps": clamp_scail2_steps(inputs.get("steps")),
            "guidance_scale": clamp_scail2_float(
                inputs.get("cfg_scale"), 5.0, 1.0, 12.0
            ),
            "flow_shift": clamp_scail2_float(inputs.get("flow_shift"), 3.0, 0.1, 20.0),
            "sample_solver": inputs.get("sampler", "unipc"),
            "sliding_window_size": clamp_scail2_int(
                inputs.get("sliding_window_size"), 81, 17, 241
            ),
            "sliding_window_overlap": clamp_scail2_int(
                inputs.get("sliding_window_overlap"), 5, 1, 80
            ),
            "sliding_window_color_correction_strength": 0,
            "remove_background_images_ref": 0,
            "max_persons": clamp_scail2_int(inputs.get("max_persons"), 2, 1, 6),
            "sam_text": inputs.get("sam_text") or inputs.get("segmentation_text"),
            "replace_flag": parse_scail_bool(inputs.get("replace_flag"), False),
            "custom_settings": {
                "scail2_animate_preprocessing": inputs.get(
                    "scail2_animate_preprocessing", "raw"
                ),
                "image_ref_keyword_content": inputs.get(
                    "image_ref_keyword_content", "human character"
                ),
            },
            "seed": self.normalize_seed(inputs.get("seed", _DEFAULT_SEED)),
        }

    def process(self):
        if DEV_MODE:
            return

        if not self.job:
            self._fail_job("Job object is None.")
            return

        inputs = self.job.get("inputs") or {}
        image_path = self._resolve_reference_image(inputs)
        if not image_path:
            self._fail_job(
                "SCAIL-2 requires a reference image. Use the image-to-video workflow."
            )
            return

        pose_video_path = self._resolve_pose_video(inputs)
        if not pose_video_path:
            self._fail_job(
                "SCAIL-2 requires a driving video in pose_video/video_guide inputs."
            )
            return

        scail2_pose_video_path = self._prepare_pose_video_for_scail2(pose_video_path)
        if not scail2_pose_video_path:
            return

        try:
            start_image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            self._fail_job(
                f"Failed to open SCAIL-2 reference image for job {self.job_id}: {exc}"
            )
            return

        settings = self._build_settings(inputs, start_image, scail2_pose_video_path)
        service_type = self.client.active_service_type or self.job.get("service_type")
        vram_tier = get_scail_vram_tier(service_type or "")
        logger.info(
            "Processing SCAIL-2 job %s with tier=%s resolution=%s frames=%s pose_video=%s",
            self.job_id,
            vram_tier,
            settings["resolution"],
            settings["video_length"],
            os.path.basename(pose_video_path),
        )

        files = self._run_task(settings)
        if not files:
            if not self.is_cancelled() and not self.infrastructure_interrupted:
                self._fail_job(self._wan2gp_no_output_message("SCAIL-2"))
            return

        result = self._handle_video_output(files[0])
        if not result:
            if not self.is_cancelled():
                self._fail_job(f"Failed to process video output for job {self.job_id}")
            return

        video_storage_path, thumbnail_storage_path, actual_duration = result
        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=video_storage_path,
            thumbnail_storage_path=thumbnail_storage_path,
            duration_seconds=actual_duration,
            prompt=self.positive_prompt,
            completion_metadata={
                "processor": "SCAIL2ImageToVideoProcessor",
                "resolution": settings["resolution"],
                "duration_requested_seconds": settings["duration_seconds"],
                "num_inference_steps": settings["num_inference_steps"],
                "guidance_scale": settings["guidance_scale"],
                "flow_shift": settings["flow_shift"],
            },
        )
