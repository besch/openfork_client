"""
SCAIL image-to-video processor (Wan2GP backend).

SCAIL animates a reference image from a driving video. Wan2GP handles the
NLF pose extraction/preprocess internally when it receives video_guide.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Optional

from PIL import Image

from config import DEV_MODE, SUPABASE_URL
from services.docker_manager import docker_manager
from services.processors.wan2gp_processor import Wan2GPProcessor
from utils.comfyui_workflow_utils import materialize_start_image

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


def duration_to_wangp_frames(duration_seconds: float) -> int:
    frame_count = int(duration_seconds * _FPS) + 1
    return max(17, ((frame_count - 1) // 4) * 4 + 1)


class SCAILImageToVideoProcessor(Wan2GPProcessor):
    """SCAIL reference-image + driving-video animation via Wan2GP."""

    SERVICE_NAME = "SCAIL"
    MAX_GENERATION_SECONDS = 7200

    def _download_storage_path(self, storage_path: str) -> Optional[str]:
        supabase_url = os.environ.get("SUPABASE_URL", SUPABASE_URL)
        if not supabase_url:
            return None

        bucket = self.job.get("bucket", "projects_public")
        source_url = f"{supabase_url}/storage/v1/object/public/{bucket}/{storage_path}"
        return self.orchestrator_service.download_asset_by_url(
            source_url, self.input_dir
        )

    def _resolve_reference_image(self, inputs: dict) -> Optional[str]:
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

            local_path = value.split("|", 1)[0]
            if os.path.exists(local_path):
                return local_path

            if len(value) < 2048:
                path = self._download_storage_path(value)
                if path:
                    return path

        input_video_url = self.job.get("input_video_url") or inputs.get(
            "input_video_url"
        )
        if input_video_url:
            return self.orchestrator_service.download_asset_by_url(
                input_video_url, self.input_dir
            )

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
