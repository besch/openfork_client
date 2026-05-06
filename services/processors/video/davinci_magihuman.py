"""
daVinci-MagiHuman processors (Wan2GP backend).

WanGP's Magi Human implementation is a talking-head model: it needs a start
image and can then generate synchronized speech/audio from the text prompt.
The supported production tiers use DeepBeepMeep/MagiHuman quanto int8
checkpoints through the Wan2GP HTTP server on port 8188.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from PIL import Image

from config import DEV_MODE, SUPABASE_URL
from services.processors.wan2gp_processor import Wan2GPProcessor
from utils.comfyui_workflow_utils import materialize_start_image

logger = logging.getLogger(__name__)

MODEL_TYPE_DISTILL_SR1080 = "magi_human_distill_sr1080"
MODEL_TYPE_BASE_SR1080 = "magi_human_sr1080"

_FPS = 25
_DEFAULT_DURATION_SECONDS = 4.0
_DEFAULT_SEED = -1

_MAGIHUMAN_RESOLUTIONS_1080 = {
    "16:9": "1920x1088",
    "9:16": "1088x1920",
    "1:1": "1088x1088",
    "4:3": "1440x1088",
    "3:4": "1088x1440",
    "21:9": "1920x816",
    "2:1": "1920x960",
}

_RUNTIME_LIMITS = {
    "16gb": {
        "model_type": MODEL_TYPE_DISTILL_SR1080,
        "duration_default": 4.0,
        "duration_max": 4.0,
        "steps_default": 8,
        "steps_max": 8,
        "guidance_default": 1.0,
        "audio_guidance_default": 1.0,
    },
    "24gb": {
        "model_type": MODEL_TYPE_BASE_SR1080,
        "duration_default": 4.0,
        "duration_max": 4.0,
        "steps_default": 32,
        "steps_max": 32,
        "guidance_default": 5.0,
        "audio_guidance_default": 5.0,
    },
    "32gb": {
        "model_type": MODEL_TYPE_BASE_SR1080,
        "duration_default": 4.0,
        "duration_max": 5.0,
        "steps_default": 32,
        "steps_max": 32,
        "guidance_default": 5.0,
        "audio_guidance_default": 5.0,
    },
    "default": {
        "model_type": MODEL_TYPE_BASE_SR1080,
        "duration_default": 4.0,
        "duration_max": 4.0,
        "steps_default": 32,
        "steps_max": 32,
        "guidance_default": 5.0,
        "audio_guidance_default": 5.0,
    },
}


def get_davinci_magihuman_runtime_limits(service_type: str) -> dict:
    service_type = (service_type or "").lower()
    for tier in ("16gb", "24gb", "32gb"):
        if tier in service_type:
            return dict(_RUNTIME_LIMITS[tier])
    return dict(_RUNTIME_LIMITS["default"])


def get_davinci_magihuman_model_type(service_type: str) -> str:
    return str(get_davinci_magihuman_runtime_limits(service_type)["model_type"])


def clamp_davinci_magihuman_duration(requested_duration, service_type: str) -> float:
    limits = get_davinci_magihuman_runtime_limits(service_type)
    try:
        duration = float(requested_duration)
    except (TypeError, ValueError):
        duration = float(limits["duration_default"])
    return max(1.0, min(duration, float(limits["duration_max"])))


def clamp_davinci_magihuman_steps(requested_steps, service_type: str) -> int:
    limits = get_davinci_magihuman_runtime_limits(service_type)
    try:
        steps = int(requested_steps)
    except (TypeError, ValueError):
        steps = int(limits["steps_default"])
    return max(1, min(steps, int(limits["steps_max"])))


def davinci_magihuman_resolution(aspect_ratio: str) -> str:
    return _MAGIHUMAN_RESOLUTIONS_1080.get(aspect_ratio, "1920x1088")


def duration_to_wangp_frames(duration_seconds: float) -> int:
    frame_count = int(duration_seconds * _FPS) + 1
    return max(26, ((frame_count - 1) // 4) * 4 + 1)


class DaVinciMagiHumanBaseProcessor(Wan2GPProcessor):
    """Shared Wan2GP settings and image resolution for MagiHuman."""

    SERVICE_NAME = "daVinci-MagiHuman"
    MAX_GENERATION_SECONDS = 7200

    def _resolve_reference_image(self, inputs: dict) -> Optional[str]:
        """Return a local path for the required start/reference image."""
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
            supabase_url = os.environ.get("SUPABASE_URL", SUPABASE_URL)
            if supabase_url:
                bucket = self.job.get("bucket", "projects_public")
                src = f"{supabase_url}/storage/v1/object/public/{bucket}/{storage_path}"
                path = self.orchestrator_service.download_asset_by_url(
                    src, self.input_dir
                )
                if path:
                    return path

        return None

    def _build_settings(self, inputs: dict, start_image: Image.Image) -> dict:
        service_type = self.job.get("service_type", "")
        limits = get_davinci_magihuman_runtime_limits(service_type)
        duration = clamp_davinci_magihuman_duration(
            inputs.get("duration", _DEFAULT_DURATION_SECONDS), service_type
        )

        return {
            "model_type": str(limits["model_type"]),
            "prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "image_start": start_image,
            "image_prompt_type": "S",
            "audio_prompt_type": "",
            "resolution": davinci_magihuman_resolution(
                inputs.get("aspect_ratio", "16:9")
            ),
            "video_length": duration_to_wangp_frames(duration),
            "force_fps": _FPS,
            "num_inference_steps": clamp_davinci_magihuman_steps(
                inputs.get("steps"), service_type
            ),
            "guidance_scale": float(
                inputs.get("cfg_scale", limits["guidance_default"])
            ),
            "audio_guidance_scale": float(
                inputs.get("audio_cfg_scale", limits["audio_guidance_default"])
            ),
            "flow_shift": float(inputs.get("flow_shift", 5.0)),
            "sample_solver": "unipc",
            "guidance_phases": 2,
            "sliding_window_size": 101,
            "sliding_window_overlap": 1,
            "sliding_window_discard_last_frames": 0,
            "seed": int(inputs.get("seed", _DEFAULT_SEED)),
        }

    def _run_magihuman_i2v(self) -> None:
        if DEV_MODE:
            return

        if not self.job:
            self._fail_job("Job object is None.")
            return

        inputs = self.job.get("inputs") or {}
        image_path = self._resolve_reference_image(inputs)
        if not image_path:
            self._fail_job(
                "daVinci-MagiHuman via Wan2GP requires a start image. "
                "Use the image-to-video workflow with a portrait reference."
            )
            return

        try:
            start_image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            self._fail_job(
                f"Failed to open daVinci-MagiHuman start image for job {self.job_id}: {exc}"
            )
            return

        settings = self._build_settings(inputs, start_image)
        logger.info(
            "Processing daVinci-MagiHuman job %s with Wan2GP model=%s resolution=%s frames=%s",
            self.job_id,
            settings["model_type"],
            settings["resolution"],
            settings["video_length"],
        )

        files = self._run_task(settings)
        if not files:
            if (
                not self.shutdown_event.is_set()
                and not self.infrastructure_interrupted
            ):
                self._fail_job(f"Wan2GP produced no output for job {self.job_id}")
            return

        result = self._handle_video_output(files[0])
        if not result:
            if not self.shutdown_event.is_set():
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


class DaVinciMagiHumanT2VProcessor(DaVinciMagiHumanBaseProcessor):
    """Compatibility processor; MagiHuman Wan2GP still requires a start image."""

    def process(self):
        self._run_magihuman_i2v()


class DaVinciMagiHumanI2VProcessor(DaVinciMagiHumanBaseProcessor):
    """daVinci-MagiHuman start-image-to-video processor via Wan2GP."""

    def process(self):
        self._run_magihuman_i2v()
