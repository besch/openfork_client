"""
LTX-2.3 Image-to-Video Processor (Wan2GP backend - 24GB VRAM tier)

Wan2GP requires BOTH "image_start" (file path) AND "image_prompt_type": "S"
for I2V. Without "image_prompt_type" containing "S", validate_settings()
explicitly sets image_start = None regardless of what was provided.

24GB tier: Uses Q8_0 GGUF model (~20.6 GB) which fits entirely in 24GB VRAM.
"""

import os
import logging
from typing import Optional

from config import DEV_MODE, SUPABASE_URL
from services.processors.wan2gp_processor import Wan2GPProcessor
from utils.comfyui_workflow_utils import materialize_start_image

_MODEL_TYPE = "ltx2_22B_distilled"
_DEFAULT_STEPS = 8
_DEFAULT_CFG = 3.0
_VIDEO_LENGTH = 121   # ~5 s @ 24 fps
_FPS = 24


class LTX23ImageToVideo24GBWan2GPProcessor(Wan2GPProcessor):
    """LTX-2.3 image-to-video via Wan2GP (24GB VRAM tier)."""

    def process(self):
        if DEV_MODE:
            return

        if not self.job:
            self._fail_job("Job object is None.")
            return

        inputs = self.job.get("inputs", {})
        image_path = self._resolve_start_image(inputs)
        if not image_path:
            self._fail_job(f"Could not resolve start image for job {self.job_id}")
            return

        settings = {
            "model_type": _MODEL_TYPE,
            "prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "image_start": image_path,
            "image_prompt_type": "S",  # Required: tells Wan2GP to use image_start for I2V
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

    def _resolve_start_image(self, inputs: dict) -> Optional[str]:
        """Return a local file path for the start image, trying multiple sources."""
        # 1. Signed URL from inputs
        url = inputs.get("start_image_url")
        if url:
            path = self.orchestrator_service.download_asset_by_url(url, self.input_dir)
            if path:
                return path

        # 2. materialize_start_image handles base64 / embedded payloads
        filename = materialize_start_image(self.job, self.input_dir)
        if filename:
            return os.path.join(self.input_dir, filename)

        # 3. Supabase storage path
        storage_path = self.job.get("input_storage_path")
        if not storage_path:
            maybe = self.job.get("start_image_base64")
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
                path = self.orchestrator_service.download_asset_by_url(src, self.input_dir)
                if path:
                    return path

        return None