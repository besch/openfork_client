"""
LTX-2.3 Image-to-Video Processor (Wan2GP backend)

Wan2GP accepts a PIL Image via the "image_start" settings key for I2V.
No ComfyUI workflow is required; the image is loaded locally and passed
directly to the Wan2GP session.
"""

import os
import logging
from typing import Optional
from PIL import Image

from config import DEV_MODE, SUPABASE_URL
from services.processors.wan2gp_processor import Wan2GPProcessor
from utils.comfyui_workflow_utils import materialize_start_image

_MODEL_TYPE = "ltx2_22B"
_DEFAULT_STEPS = 8
_DEFAULT_CFG = 3.0
_FPS = 24


class LTX23ImageToVideoWan2GPProcessor(Wan2GPProcessor):
    """LTX-2.3 image-to-video via Wan2GP (Supports multiple tiers dynamically)."""

    def process(self):
        if DEV_MODE:
            return

        if not self.job:
            self._fail_job("Job object is None.")
            return

        inputs = self.job.get("inputs", {})
        service_type = self.job.get("service_type", "")

        # Determine dynamic constraints based on the service tier
        if "8gb" in service_type:
            duration_max = 3
            duration_default = 2
        elif "16gb" in service_type:
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

        image_path = self._resolve_start_image(inputs)
        if not image_path:
            self._fail_job(f"Could not resolve start image for job {self.job_id}")
            return

        try:
            start_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            self._fail_job(f"Failed to open start image for job {self.job_id}: {e}")
            return

        # Handle image_prompt_type dynamically based on settings
        image_prompt_type = "S"
        if "image_prompt_type" in inputs:
            image_prompt_type = inputs["image_prompt_type"]

        settings = {
            "model_type": _MODEL_TYPE,
            "prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "image_start": start_image,
            "image_prompt_type": image_prompt_type,
            "resolution": self.aspect_to_resolution(
                inputs.get("aspect_ratio", "16:9"), service_type
            ),
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
