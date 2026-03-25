"""
LTX-2.3 Text-to-Video Processor (Wan2GP backend - 24GB VRAM tier)

LTX-2.3 is a 22B DiT audio-video model that generates synchronised video
and audio in a single pass. Wan2GP handles VRAM management and audio
generation natively — no ComfyUI workflow is required.

Model type note: if generation fails with an unknown-model error, verify the
exact string by running `python wgp.py --list-models` inside the Wan2GP
installation.

24GB tier: Uses Q8_0 GGUF model (~20.6 GB) which fits entirely in 24GB VRAM.
"""

import logging

from config import DEV_MODE
from services.processors.wan2gp_processor import Wan2GPProcessor

_MODEL_TYPE = "ltx2_22B_distilled"
_DEFAULT_STEPS = 8
_DEFAULT_CFG = 3.0
_VIDEO_LENGTH = 121   # ~5 s @ 24 fps
_FPS = 24


class LTX23TextToVideo24GBWan2GPProcessor(Wan2GPProcessor):
    """LTX-2.3 text-to-video via Wan2GP (24GB VRAM tier)."""

    def process(self):
        if DEV_MODE:
            return

        inputs = self.job.get("inputs", {})
        settings = {
            "model_type": _MODEL_TYPE,
            "prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "resolution": self.aspect_to_resolution(inputs.get("aspect_ratio", "16:9")),
            "num_inference_steps": inputs.get("steps", _DEFAULT_STEPS),
            "guidance_scale": inputs.get("cfg_scale", _DEFAULT_CFG),
            "video_length": _VIDEO_LENGTH,
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

        video_storage_path, thumbnail_storage_path, duration = result
        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=video_storage_path,
            thumbnail_storage_path=thumbnail_storage_path,
            duration_seconds=duration,
            prompt=self.positive_prompt,
        )