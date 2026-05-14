"""
LTX-2.3 Text-to-Video Processor (Wan2GP backend)

LTX-2.3 is a 22B DiT audio-video model that generates synchronised video
and audio in a single pass. Wan2GP handles VRAM management and audio
generation natively — no ComfyUI workflow is required.

Model type note: if generation fails with an unknown-model error, verify the
exact string by running `python wgp.py --list-models` inside the Wan2GP
installation.
"""

import logging

from config import DEV_MODE
from services.processors.wan2gp_processor import Wan2GPProcessor
from services.processors.video.ltx23_common import (
    clamp_ltx23_duration,
    clamp_ltx23_steps,
    get_ltx23_model_type,
)

_DEFAULT_CFG = 3.0
_FPS = 24


class LTX23TextToVideoWan2GPProcessor(Wan2GPProcessor):
    """LTX-2.3 text-to-video via Wan2GP (Supports multiple tiers dynamically)."""

    def process(self):
        if DEV_MODE:
            return

        inputs = self.job.get("inputs", {})
        service_type = self.job.get("service_type", "")

        duration = clamp_ltx23_duration(inputs.get("duration"), service_type)
        video_length = int(duration * _FPS) + 1

        settings = {
            "model_type": get_ltx23_model_type(service_type),
            "prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "resolution": self.aspect_to_resolution(
                inputs.get("aspect_ratio", "16:9"), service_type
            ),
            "num_inference_steps": clamp_ltx23_steps(
                inputs.get("steps"), service_type
            ),
            "guidance_scale": inputs.get("cfg_scale", _DEFAULT_CFG),
            "video_length": video_length,
            "force_fps": _FPS,
        }

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
        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=video_storage_path,
            thumbnail_storage_path=thumbnail_storage_path,
            duration_seconds=actual_duration,
            prompt=self.positive_prompt,
        )
