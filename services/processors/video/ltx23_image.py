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
from services.processors.video.ltx23_common import (
    build_ltx23_prompt,
    clamp_ltx23_duration,
    clamp_ltx23_steps,
    get_ltx23_model_type,
    ltx23_lora_weight,
    should_use_ltx23_hdr,
)
from services.processors.video.last_frame import materialize_last_frame_start_image
from utils.comfyui_workflow_utils import materialize_start_image

_DEFAULT_CFG = 1.0
_FPS = 24


def _resolution_to_dimensions(resolution: str) -> Optional[tuple[int, int]]:
    try:
        width, height = str(resolution).lower().split("x", 1)
        return int(width), int(height)
    except (AttributeError, TypeError, ValueError):
        return None


class LTX23ImageToVideoWan2GPProcessor(Wan2GPProcessor):
    """LTX-2.3 image-to-video via Wan2GP (Supports multiple tiers dynamically)."""

    def process(self):
        if DEV_MODE:
            return

        if not self.job:
            self._fail_job("Job object is None.")
            return

        inputs = self.job.get("inputs") or {}
        service_type = self.job.get("service_type", "")

        duration = clamp_ltx23_duration(inputs.get("duration"), service_type)
        video_length = int(duration * _FPS) + 1
        resolution = self.aspect_to_resolution(
            inputs.get("aspect_ratio", "16:9"), service_type
        )
        prompt, audio_prompt = build_ltx23_prompt(self.positive_prompt, inputs)
        seed_was_randomized = not self.has_explicit_seed(inputs.get("seed"))
        seed = self.resolve_seed(inputs.get("seed"), randomize_missing=True)
        model_type = get_ltx23_model_type(service_type)
        steps = clamp_ltx23_steps(inputs.get("steps"), service_type)
        guidance_scale = inputs.get("cfg_scale", _DEFAULT_CFG)
        hdr = should_use_ltx23_hdr(inputs, service_type)
        lora_weight = ltx23_lora_weight(inputs)

        image_path = self._resolve_start_image(
            inputs,
            target_dimensions=_resolution_to_dimensions(resolution),
        )
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
            "model_type": model_type,
            "prompt": prompt,
            "negative_prompt": self.negative_prompt,
            "image_start": start_image,
            "image_prompt_type": image_prompt_type,
            "resolution": resolution,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "video_length": video_length,
            "force_fps": _FPS,
            "seed": seed,
            "hdr": hdr,
            "lora_weight": lora_weight,
        }
        audio_cfg_scale = inputs.get("audio_cfg_scale", inputs.get("audio_guidance_scale"))
        if audio_cfg_scale is not None:
            settings["audio_cfg_scale"] = audio_cfg_scale

        files = self._run_task(settings)
        if not files:
            if (
                not self.is_cancelled()
                and not self.infrastructure_interrupted
            ):
                self._fail_job(self._wan2gp_no_output_message("LTX-2.3/Wan2GP"))
            return

        result = self._handle_video_output(files[0])
        if not result:
            if not self.is_cancelled():
                self._fail_job(f"Failed to process video output for job {self.job_id}")
            return

        video_storage_path, thumbnail_storage_path, duration = result
        completion_metadata = self._video_completion_metadata()
        completion_metadata.update(
            {
                "seed": seed,
                "seed_source": "randomized" if seed_was_randomized else "input",
                "model_type": model_type,
                "requested_resolution": resolution,
                "requested_duration_seconds": clamp_ltx23_duration(
                    inputs.get("duration"), service_type
                ),
                "requested_video_length_frames": video_length,
                "requested_fps": _FPS,
                "steps": steps,
                "cfg_scale": guidance_scale,
                "hdr": hdr,
                "lora_weight": lora_weight,
                "image_prompt_type": image_prompt_type,
            }
        )
        if audio_prompt:
            completion_metadata["audio_prompt"] = audio_prompt
        if audio_cfg_scale is not None:
            completion_metadata["audio_cfg_scale"] = audio_cfg_scale

        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=video_storage_path,
            thumbnail_storage_path=thumbnail_storage_path,
            duration_seconds=duration,
            completion_metadata=completion_metadata,
            prompt=prompt,
        )

    def _resolve_start_image(
        self,
        inputs: dict,
        *,
        target_dimensions: Optional[tuple[int, int]] = None,
    ) -> Optional[str]:
        """Return a local file path for the start image, trying multiple sources."""
        last_frame_path = materialize_last_frame_start_image(
            self,
            inputs,
            target_dimensions=target_dimensions,
        )
        if last_frame_path:
            return last_frame_path

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
            bucket = self.job.get("bucket", "projects_public")
            path = self.orchestrator_service.download_storage_asset(
                bucket,
                storage_path,
                self.input_dir,
            )
            if path:
                return path

            supabase_url = os.environ.get("SUPABASE_URL", SUPABASE_URL)
            if supabase_url:
                src = f"{supabase_url}/storage/v1/object/public/{bucket}/{storage_path}"
                path = self.orchestrator_service.download_asset_by_url(
                    src, self.input_dir
                )
                if path:
                    return path

        return None
