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
    build_ltx23_prompt,
    clamp_ltx23_duration,
    clamp_ltx23_steps,
    get_ltx23_model_type,
    ltx23_lora_weight,
    should_use_ltx23_hdr,
)

_DEFAULT_CFG = 1.0
_FPS = 24


class LTX23TextToVideoWan2GPProcessor(Wan2GPProcessor):
    """LTX-2.3 text-to-video via Wan2GP (Supports multiple tiers dynamically)."""

    def process(self):
        if DEV_MODE:
            return

        inputs = self.job.get("inputs") or {}
        service_type = self.job.get("service_type", "")

        duration = clamp_ltx23_duration(inputs.get("duration"), service_type)
        video_length = int(duration * _FPS) + 1
        prompt, audio_prompt = build_ltx23_prompt(self.positive_prompt, inputs)
        seed_was_randomized = not self.has_explicit_seed(inputs.get("seed"))
        seed = self.resolve_seed(inputs.get("seed"), randomize_missing=True)
        model_type = get_ltx23_model_type(service_type)
        resolution = self.aspect_to_resolution(
            inputs.get("aspect_ratio", "16:9"), service_type
        )
        steps = clamp_ltx23_steps(inputs.get("steps"), service_type)
        guidance_scale = inputs.get("cfg_scale", _DEFAULT_CFG)
        hdr = should_use_ltx23_hdr(inputs, service_type)
        lora_weight = ltx23_lora_weight(inputs)

        settings = {
            "model_type": model_type,
            "prompt": prompt,
            "negative_prompt": self.negative_prompt,
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

        video_storage_path, thumbnail_storage_path, actual_duration = result
        completion_metadata = self._video_completion_metadata()
        completion_metadata.update(
            {
                "seed": seed,
                "seed_source": "randomized" if seed_was_randomized else "input",
                "model_type": model_type,
                "requested_resolution": resolution,
                "requested_duration_seconds": duration,
                "requested_video_length_frames": video_length,
                "requested_fps": _FPS,
                "steps": steps,
                "cfg_scale": guidance_scale,
                "hdr": hdr,
                "lora_weight": lora_weight,
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
            duration_seconds=actual_duration,
            completion_metadata=completion_metadata,
            prompt=prompt,
        )
