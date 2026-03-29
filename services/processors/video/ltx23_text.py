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

_MODEL_TYPE = "ltx2_22B"
_DEFAULT_STEPS = 8
_DEFAULT_CFG = 3.0
_FPS = 24


class LTX23TextToVideoWan2GPProcessor(Wan2GPProcessor):
    """LTX-2.3 text-to-video via Wan2GP (Supports multiple tiers dynamically)."""

    def process(self):
        if DEV_MODE:
            return

        inputs = self.job.get("inputs", {})
        service_type = self.job.get("service_type", "")
        
        # Determine dynamic constraints based on the service tier
        if "16gb" in service_type:
            duration_max = 5
            duration_default = 4
        elif "32gb" in service_type:
            duration_max = 10
            duration_default = 7
        else:
            # 24GB fallback
            duration_max = 7
            duration_default = 5

        # Calculate exact frame count based on requested duration
        requested_duration = float(inputs.get("duration", duration_default))
        duration = max(1.0, min(requested_duration, float(duration_max)))
        video_length = int(duration * _FPS) + 1

        settings = {
            "model_type": _MODEL_TYPE,
            "prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "resolution": self.aspect_to_resolution(inputs.get("aspect_ratio", "16:9"), service_type),
            "num_inference_steps": inputs.get("steps", _DEFAULT_STEPS),
            "guidance_scale": inputs.get("cfg_scale", _DEFAULT_CFG),
            "video_length": video_length,
            "force_fps": _FPS,
        }

        files = self._run_task(settings)
        if not files:
            self._fail_job(f"Wan2GP produced no output for job {self.job_id}")
            return

        result = self._handle_video_output(files[0])
        if not result:
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
