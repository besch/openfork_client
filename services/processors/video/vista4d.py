"""
Vista4D video-to-video processor (Wan2GP backend).

Vista4D reshoots an input source video from a new camera trajectory. Wan2GP
performs the depth/SAM preprocessing internally when it receives video_guide.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Optional

from config import DEV_MODE, SUPABASE_URL
from services.docker_manager import docker_manager
from services.processors.wan2gp_processor import Wan2GPProcessor

logger = logging.getLogger(__name__)

_DEFAULT_SEED = -1
_DEFAULT_PROMPT = (
    "a cinematic video of the scene with realistic motion and consistent geometry"
)
_VISTA4D_FRAME_COUNT = 49

_VISTA4D_RESOLUTIONS = {
    "16:9": "672x384",
    "9:16": "384x672",
    "1:1": "512x512",
    "4:3": "512x384",
    "3:4": "384x512",
    "21:9": "672x288",
    "2:1": "640x320",
}

_VISTA4D_CAMERA_MODES = {
    "dolly_zoom",
    "left_front_zoom",
    "right_front_zoom",
    "close_crane_above",
    "close_crane_below",
    "arc_right_45",
    "arc_left_45",
    "push_in",
    "pull_back",
    "truck_right",
    "truck_left",
    "pedestal_up",
    "pedestal_down",
    "pan_right_45",
    "pan_left_45",
    "tilt_up_45",
    "tilt_down_45",
    "zoom_in",
    "zoom_out",
    "bird_view",
    "crane_above_right",
    "crane_above_left",
    "crane_below_right",
    "crane_below_left",
}

_CAMERA_MODE_ALIASES = {
    "orbit": "arc_right_45",
    "orbit_right": "arc_right_45",
    "orbit_left": "arc_left_45",
    "pan_right": "pan_right_45",
    "pan_left": "pan_left_45",
    "tilt_up": "tilt_up_45",
    "tilt_down": "tilt_down_45",
    "dolly_in": "push_in",
    "push": "push_in",
    "dolly_out": "pull_back",
    "pull_backwards": "pull_back",
    "crane": "close_crane_above",
}


def vista4d_resolution(aspect_ratio: str) -> str:
    return _VISTA4D_RESOLUTIONS.get(aspect_ratio, "672x384")


def clamp_vista4d_steps(requested_steps) -> int:
    try:
        steps = int(requested_steps)
    except (TypeError, ValueError):
        steps = 50
    return max(10, min(steps, 50))


def clamp_float(value, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def normalize_vista4d_camera_mode(value) -> str:
    if not value:
        return "dolly_zoom"

    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return (
        normalized
        if normalized in _VISTA4D_CAMERA_MODES
        else _CAMERA_MODE_ALIASES.get(normalized, "dolly_zoom")
    )


def normalize_seed(value) -> int:
    return Wan2GPProcessor.normalize_seed(value, _DEFAULT_SEED)


class Vista4DVideoToVideoProcessor(Wan2GPProcessor):
    """Vista4D source-video + camera-trajectory reshooting via Wan2GP."""

    SERVICE_NAME = "Vista4D"
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

    def _resolve_source_video(self, inputs: dict) -> Optional[str]:
        for key in (
            "input_video_url",
            "source_video_url",
            "video_guide_url",
            "video_source_url",
        ):
            url = self.job.get(key) or inputs.get(key)
            if url:
                path = self.orchestrator_service.download_asset_by_url(
                    url, self.input_dir
                )
                if path:
                    return path

        for key in ("video_guide", "video_source", "source_video", "input_video"):
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

        storage_path = self.job.get("input_storage_path") or inputs.get(
            "input_storage_path"
        )
        if storage_path and isinstance(storage_path, str):
            return self._download_storage_path(storage_path)

        return None

    def _prepare_source_video_for_wan2gp(
        self, source_video_path: str
    ) -> Optional[str]:
        filename = f"{self.job_id}_{os.path.basename(source_video_path)}"
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
                    source_on_host=source_video_path,
                    dest_in_container=container_path,
                    shutdown_event=self.shutdown_event,
                )
                return container_path

            os.makedirs(os.path.dirname(container_path), exist_ok=True)
            if os.path.abspath(source_video_path) != os.path.abspath(container_path):
                shutil.copy2(source_video_path, container_path)
            return container_path
        except Exception as exc:
            self._fail_job(
                f"Failed to prepare Vista4D source video for job {self.job_id}: {exc}"
            )
            return None

    def _build_settings(self, inputs: dict, source_video_path: str) -> dict:
        camera_mode = normalize_vista4d_camera_mode(
            inputs.get("model_mode")
            or inputs.get("camera_mode")
            or inputs.get("trajectory")
        )

        prompt = self.positive_prompt or str(inputs.get("prompt") or _DEFAULT_PROMPT)
        seg_keywords = str(inputs.get("vista4d_seg_keywords") or "_all_").strip()

        return {
            "model_type": "vista4d",
            "prompt": prompt,
            "negative_prompt": self.negative_prompt,
            "video_guide": source_video_path,
            "video_source": source_video_path,
            "video_prompt_type": "UV",
            "resolution": vista4d_resolution(inputs.get("aspect_ratio", "16:9")),
            "video_length": _VISTA4D_FRAME_COUNT,
            "fps": "control",
            "force_fps": "control",
            "model_mode": camera_mode,
            "num_inference_steps": clamp_vista4d_steps(inputs.get("steps")),
            "guidance_scale": clamp_float(inputs.get("cfg_scale"), 5.0, 1.0, 10.0),
            "flow_shift": clamp_float(inputs.get("flow_shift"), 5.0, 1.0, 20.0),
            "sample_solver": inputs.get("sampler", "unipc"),
            "vista4d_scene_scale": clamp_float(
                inputs.get("vista4d_scene_scale"), 1.0, 0.1, 10.0
            ),
            "vista4d_camera_strength": clamp_float(
                inputs.get("vista4d_camera_strength"), 100.0, 0.0, 300.0
            ),
            "vista4d_seg_keywords": seg_keywords or "_all_",
            "seed": normalize_seed(inputs.get("seed")),
        }

    def process(self):
        if DEV_MODE:
            return

        if not self.job:
            self._fail_job("Job object is None.")
            return

        inputs = self.job.get("inputs") or {}
        source_video_path = self._resolve_source_video(inputs)
        if not source_video_path:
            self._fail_job(
                "Vista4D requires a source video in input_video_url/video_guide inputs."
            )
            return

        wan2gp_source_video_path = self._prepare_source_video_for_wan2gp(
            source_video_path
        )
        if not wan2gp_source_video_path:
            return

        settings = self._build_settings(inputs, wan2gp_source_video_path)
        logger.info(
            "Processing Vista4D job %s with resolution=%s mode=%s source_video=%s",
            self.job_id,
            settings["resolution"],
            settings["model_mode"],
            os.path.basename(source_video_path),
        )

        files = self._run_task(settings)
        if not files:
            if (
                not self.is_cancelled()
                and not self.infrastructure_interrupted
            ):
                self._fail_job(f"Wan2GP produced no output for job {self.job_id}")
            return

        result = self._handle_video_output(files[0])
        if not result:
            if not self.is_cancelled():
                self._fail_job(f"Failed to process video output for job {self.job_id}")
            return

        video_storage_path, thumbnail_storage_path, actual_duration = result
        completion_metadata = self.job.get("completion_metadata") or {}
        completion_metadata.update(
            {
                "processor": "Vista4DVideoToVideoProcessor",
                "model_mode": settings["model_mode"],
                "vista4d_scene_scale": settings["vista4d_scene_scale"],
                "vista4d_camera_strength": settings["vista4d_camera_strength"],
                "vista4d_seg_keywords": settings["vista4d_seg_keywords"],
            }
        )

        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=video_storage_path,
            thumbnail_storage_path=thumbnail_storage_path,
            duration_seconds=actual_duration,
            prompt=settings["prompt"],
            completion_metadata=completion_metadata,
        )
