"""
WAN 2.2 processors for the Wan2GP backend.

These are experimental canary processors for disabled WAN 2.2 Wan2GP service
tiers. The existing ComfyUI WAN 2.2 processors remain the production path.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from PIL import Image

from config import DEV_MODE, SUPABASE_URL
from services.processors.wan2gp_processor import Wan2GPProcessor
from utils.comfyui_workflow_utils import materialize_start_image
from utils.media_utils import extract_last_frame

logger = logging.getLogger(__name__)

WAN22_T2V_MODEL_TYPE = "t2v_2_2"
WAN22_I2V_MODEL_TYPE = "i2v_2_2"

_FPS = 16
_DEFAULT_SEED = -1

_WAN22_RUNTIME_LIMITS = {
    "8gb": {
        "duration_default": 3.0,
        "duration_max": 4.0,
        "steps_default": 8,
        "steps_max": 12,
    },
    "10gb": {
        "duration_default": 4.0,
        "duration_max": 5.0,
        "steps_default": 10,
        "steps_max": 16,
    },
    "12gb": {
        "duration_default": 5.0,
        "duration_max": 5.0,
        "steps_default": 12,
        "steps_max": 18,
    },
    "16gb": {
        "duration_default": 5.0,
        "duration_max": 6.0,
        "steps_default": 14,
        "steps_max": 24,
    },
    "24gb": {
        "duration_default": 5.0,
        "duration_max": 8.0,
        "steps_default": 16,
        "steps_max": 30,
    },
    "default": {
        "duration_default": 5.0,
        "duration_max": 5.0,
        "steps_default": 12,
        "steps_max": 18,
    },
}

_WAN22_RESOLUTIONS = {
    "8gb": {
        "16:9": "320x176",
        "9:16": "176x320",
        "1:1": "256x256",
        "4:3": "320x240",
        "3:4": "240x320",
        "21:9": "320x144",
        "2:1": "320x160",
    },
    "10gb": {
        "16:9": "448x256",
        "9:16": "256x448",
        "1:1": "384x384",
        "4:3": "448x336",
        "3:4": "336x448",
        "21:9": "448x192",
        "2:1": "448x224",
    },
    "12gb": {
        "16:9": "544x304",
        "9:16": "304x544",
        "1:1": "480x480",
        "4:3": "544x416",
        "3:4": "416x544",
        "21:9": "544x240",
        "2:1": "544x272",
    },
    "16gb": {
        "16:9": "768x432",
        "9:16": "432x768",
        "1:1": "640x640",
        "4:3": "640x480",
        "3:4": "480x640",
        "21:9": "768x320",
        "2:1": "768x384",
    },
    "24gb": {
        "16:9": "832x480",
        "9:16": "480x832",
        "1:1": "768x768",
        "4:3": "768x576",
        "3:4": "576x768",
        "21:9": "896x384",
        "2:1": "832x416",
    },
}


def get_wan22_wan2gp_tier(service_type: str) -> str:
    service_type = (service_type or "").lower()
    for tier in ("8gb", "10gb", "12gb", "16gb", "24gb"):
        if tier in service_type:
            return tier
    return "default"


def get_wan22_wan2gp_runtime_limits(service_type: str) -> dict:
    tier = get_wan22_wan2gp_tier(service_type)
    return dict(_WAN22_RUNTIME_LIMITS.get(tier, _WAN22_RUNTIME_LIMITS["default"]))


def wan22_wan2gp_resolution(aspect_ratio: str, service_type: str = "") -> str:
    tier = get_wan22_wan2gp_tier(service_type)
    tier_map = _WAN22_RESOLUTIONS.get(tier) or _WAN22_RESOLUTIONS["12gb"]
    return tier_map.get(aspect_ratio, tier_map["16:9"])


def clamp_wan22_wan2gp_duration(requested_duration, service_type: str) -> float:
    limits = get_wan22_wan2gp_runtime_limits(service_type)
    try:
        duration = float(requested_duration)
    except (TypeError, ValueError):
        duration = float(limits["duration_default"])
    return max(1.0, min(duration, float(limits["duration_max"])))


def clamp_wan22_wan2gp_steps(requested_steps, service_type: str) -> int:
    limits = get_wan22_wan2gp_runtime_limits(service_type)
    try:
        steps = int(requested_steps)
    except (TypeError, ValueError):
        steps = int(limits["steps_default"])
    return max(1, min(steps, int(limits["steps_max"])))


def duration_to_wan22_frames(duration_seconds: float) -> int:
    frame_count = int(duration_seconds * _FPS) + 1
    return max(17, ((frame_count - 1) // 4) * 4 + 1)


def _float_input(inputs: dict, key: str, default: float) -> float:
    try:
        return float(inputs.get(key, default))
    except (TypeError, ValueError):
        return default


def _int_input(inputs: dict, key: str, default: int) -> int:
    try:
        return int(inputs.get(key, default))
    except (TypeError, ValueError):
        return default


class WAN22Wan2GPBaseProcessor(Wan2GPProcessor):
    SERVICE_NAME = "WAN 2.2/Wan2GP"
    MAX_GENERATION_SECONDS = 7200

    def _service_type(self) -> str:
        return getattr(self.client, "active_service_type", None) or self.job.get(
            "service_type",
            "",
        )

    def _common_settings(self, inputs: dict, model_type: str) -> dict:
        service_type = self._service_type()
        duration = clamp_wan22_wan2gp_duration(inputs.get("duration"), service_type)
        settings = {
            "model_type": model_type,
            "prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "resolution": wan22_wan2gp_resolution(
                inputs.get("aspect_ratio", "16:9"),
                service_type,
            ),
            "video_length": duration_to_wan22_frames(duration),
            "force_fps": _FPS,
            "num_inference_steps": clamp_wan22_wan2gp_steps(
                inputs.get("steps"),
                service_type,
            ),
            "seed": self.normalize_seed(inputs.get("seed"), _DEFAULT_SEED),
            "sample_solver": inputs.get("sampler") or "unipc",
            "guidance_phases": 2,
        }
        return settings

    def _complete_from_first_output(self, files: list[str]) -> None:
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

    def _download_storage_path(self, storage_path: str) -> Optional[str]:
        bucket = self.job.get("bucket", "projects_public")
        path = self.orchestrator_service.download_storage_asset(
            bucket,
            storage_path,
            self.input_dir,
        )
        if path:
            return path

        supabase_url = os.environ.get("SUPABASE_URL", SUPABASE_URL)
        if not supabase_url:
            return None

        source_url = f"{supabase_url}/storage/v1/object/public/{bucket}/{storage_path}"
        return self.orchestrator_service.download_asset_by_url(
            source_url,
            self.input_dir,
        )


class WAN22TextToVideoWan2GPProcessor(WAN22Wan2GPBaseProcessor):
    """WAN 2.2 text-to-video through Wan2GP."""

    def process(self):
        if DEV_MODE:
            return

        inputs = self.job.get("inputs") or {}
        settings = self._common_settings(inputs, WAN22_T2V_MODEL_TYPE)
        settings.update(
            {
                "guidance_scale": _float_input(inputs, "cfg_scale", 4.0),
                "guidance2_scale": _float_input(inputs, "cfg2_scale", 3.0),
                "flow_shift": _float_input(inputs, "flow_shift", 12.0),
                "switch_threshold": _int_input(inputs, "switch_threshold", 875),
            }
        )

        logger.info(
            "Processing WAN 2.2 T2V Wan2GP job %s service=%s resolution=%s frames=%s steps=%s",
            self.job_id,
            self._service_type(),
            settings["resolution"],
            settings["video_length"],
            settings["num_inference_steps"],
        )

        files = self._run_task(settings)
        if not files:
            if not self.is_cancelled() and not self.infrastructure_interrupted:
                self._fail_job(self._wan2gp_no_output_message(self.SERVICE_NAME))
            return

        self._complete_from_first_output(files)


class WAN22ImageToVideoWan2GPProcessor(WAN22Wan2GPBaseProcessor):
    """WAN 2.2 image-to-video through Wan2GP."""

    def process(self):
        if DEV_MODE:
            return

        if not self.job:
            self._fail_job("Job object is None.")
            return

        inputs = self.job.get("inputs") or {}
        image_path = self._resolve_start_image(inputs)
        if not image_path:
            self._fail_job(f"Could not resolve start image for job {self.job_id}")
            return

        try:
            start_image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            self._fail_job(f"Failed to open start image for job {self.job_id}: {exc}")
            return

        settings = self._common_settings(inputs, WAN22_I2V_MODEL_TYPE)
        settings.update(
            {
                "image_start": start_image,
                "image_prompt_type": inputs.get("image_prompt_type", "S"),
                "guidance_scale": _float_input(inputs, "cfg_scale", 3.5),
                "guidance2_scale": _float_input(inputs, "cfg2_scale", 3.5),
                "flow_shift": _float_input(inputs, "flow_shift", 5.0),
                "switch_threshold": _int_input(inputs, "switch_threshold", 900),
                "masking_strength": _float_input(inputs, "masking_strength", 0.1),
                "denoising_strength": _float_input(
                    inputs,
                    "denoising_strength",
                    0.9,
                ),
            }
        )

        logger.info(
            "Processing WAN 2.2 I2V Wan2GP job %s service=%s resolution=%s frames=%s steps=%s",
            self.job_id,
            self._service_type(),
            settings["resolution"],
            settings["video_length"],
            settings["num_inference_steps"],
        )

        files = self._run_task(settings)
        if not files:
            if not self.is_cancelled() and not self.infrastructure_interrupted:
                self._fail_job(self._wan2gp_no_output_message(self.SERVICE_NAME))
            return

        self._complete_from_first_output(files)

    def _resolve_start_image(self, inputs: dict) -> Optional[str]:
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


class ImageToVideoFromLastFrameWan2GPProcessor(WAN22ImageToVideoWan2GPProcessor):
    """WAN 2.2 Wan2GP continuation from the last frame of an input video."""

    def _resolve_start_image(self, inputs: dict) -> Optional[str]:
        input_video_url = inputs.get("input_video_url")
        if not input_video_url:
            self._fail_job(f"Job {self.job_id} missing 'input_video_url' in inputs.")
            return None

        video_path = self.orchestrator_service.download_asset_by_url(
            input_video_url,
            self.input_dir,
        )
        if not video_path:
            self._fail_job(f"Failed to download input video for job {self.job_id}.")
            return None

        start_image_path = os.path.join(self.input_dir, f"{self.job_id}_last_frame.jpg")
        if not extract_last_frame(video_path, start_image_path):
            self._fail_job(f"Failed to extract last frame for job {self.job_id}.")
            return None

        return start_image_path
