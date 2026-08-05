"""MiniMax-H3 native ComfyUI text/image-to-video processors.

MiniMax-H3 generates video and synchronized stereo audio in one diffusion
pass.  The official pruned INT8/NVFP4 assets are much larger than the low
VRAM tiers, so those tiers rely on ComfyUI dynamic CPU offload.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import random
import re
import shutil
from typing import Any, Optional

from config import DEV_MODE, HEADLESS_MODE, SUPABASE_URL
from services.docker_manager import docker_manager
from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import VideoOutputHandler
from services.processors.video.last_frame import materialize_last_frame_start_image
from services.processors.video.ltx23_common import build_ltx23_prompt
from utils.comfyui_workflow_utils import materialize_start_image


FPS = 24
MIN_TRAINED_FRAMES = 124
MAX_TRAINED_FRAMES = 362
DEFAULT_STEPS = 20
DEFAULT_TIER_GB = 6

# Native H3 uses a 768px short edge, a 1344px long-edge cap, and dimensions
# divisible by 32. Lower tiers reduce the short edge to limit activation VRAM.
TIER_SHORT_EDGE = {
    6: 352,
    8: 384,
    12: 480,
    16: 544,
    24: 608,
    32: 672,
    48: 768,
    80: 768,
}
TIER_MAX_DURATION_SECONDS = {
    6: 5,
    8: 5,
    12: 5,
    16: 8,
    24: 10,
    32: 12,
    48: 15,
    80: 15,
}


def _as_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _as_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def resolve_minimax_h3_tier(value: Any) -> int:
    """Resolve a registered H3 tier from a workflow/service/model string."""
    match = re.search(r"(?:^|[-_])(6|8|12|16|24|32|48|80)gb(?:$|[-_])", str(value or "").lower())
    if match:
        return int(match.group(1))
    return DEFAULT_TIER_GB


def _multiple_of_32(value: float) -> int:
    return max(32, int(round(value / 32.0)) * 32)


def resolve_minimax_h3_dimensions(
    aspect_ratio: Any,
    tier_gb: int,
    target_width: Any = None,
    target_height: Any = None,
) -> tuple[int, int]:
    """Return a tier-safe resolution within H3's native canvas constraints."""
    tier_gb = tier_gb if tier_gb in TIER_SHORT_EDGE else DEFAULT_TIER_GB
    short_edge = TIER_SHORT_EDGE[tier_gb]
    ratio_map = {
        "16:9": (16.0, 9.0),
        "9:16": (9.0, 16.0),
        "1:1": (1.0, 1.0),
        "4:3": (4.0, 3.0),
        "3:4": (3.0, 4.0),
        "21:9": (21.0, 9.0),
    }
    ratio_width, ratio_height = ratio_map.get(str(aspect_ratio), ratio_map["16:9"])

    if ratio_width >= ratio_height:
        height = short_edge
        width = _multiple_of_32(short_edge * ratio_width / ratio_height)
    else:
        width = short_edge
        height = _multiple_of_32(short_edge * ratio_height / ratio_width)
    width, height = min(width, 1344), min(height, 1344)

    # Advanced callers may request a smaller canvas, never a larger one than
    # the selected tier profile. This prevents a 6GB job from bypassing limits.
    try:
        requested_width = _multiple_of_32(float(target_width))
        requested_height = _multiple_of_32(float(target_height))
    except (TypeError, ValueError):
        requested_width = requested_height = 0
    if requested_width and requested_height:
        requested_area = requested_width * requested_height
        if requested_area <= width * height and max(requested_width, requested_height) <= 1344:
            width, height = requested_width, requested_height

    return width, height


def resolve_minimax_h3_frames(duration: Any, tier_gb: int) -> tuple[float, int]:
    """Clamp duration by tier and snap it to H3's required 17k+5 grid."""
    maximum = TIER_MAX_DURATION_SECONDS.get(tier_gb, TIER_MAX_DURATION_SECONDS[DEFAULT_TIER_GB])
    requested = _as_float(duration, min(5.0, float(maximum)), 5.0, float(maximum))
    target_frames = requested * FPS
    grid_index = math.ceil((target_frames - 5) / 17)
    frames = 5 + 17 * grid_index
    frames = max(MIN_TRAINED_FRAMES, min(MAX_TRAINED_FRAMES, frames))
    return requested, frames


def resolve_minimax_h3_seed(value: Any) -> tuple[int, str]:
    try:
        seed = int(value)
    except (TypeError, ValueError):
        seed = -1
    if seed < 0:
        return random.SystemRandom().randint(0, 2**63 - 1), "randomized"
    return min(seed, 2**63 - 1), "input"


def build_minimax_h3_workflow(
    workflow_data: dict,
    *,
    prompt: str,
    width: int,
    height: int,
    frames: int,
    steps: int,
    seed: int,
    filename_prefix: str,
    start_image_filename: Optional[str] = None,
    acceleration_mode: str = "native",
) -> dict:
    """Inject one API graph and optionally add Spectrum acceleration."""
    graph = copy.deepcopy(workflow_data["prompt"])
    graph["6"]["inputs"].update(
        {"prompt": prompt, "width": width, "height": height, "length": frames}
    )
    graph["7"]["inputs"]["noise_seed"] = seed
    graph["9"]["inputs"]["steps"] = steps
    graph["15"]["inputs"]["filename_prefix"] = filename_prefix

    if start_image_filename:
        graph["16"] = {
            "class_type": "LoadImage",
            "inputs": {"image": start_image_filename},
            "_meta": {"title": "Load first frame"},
        }
        graph["6"]["inputs"]["first_frame"] = ["16", 0]

    if acceleration_mode == "spectrum":
        graph["17"] = {
            "class_type": "SpectrumApplyMiniMaxH3",
            "inputs": {
                "model": ["5", 0],
                "enabled": True,
                "blend_weight": 0.5,
                "degree": 4,
                "ridge_lambda": 0.1,
                "window_size": 2.0,
                "flex_window": 0.75,
                "warmup_steps": 5,
                "tail_actual_steps": 1,
                "max_history": 8,
                "debug": False,
                "history_storage": "system_ram",
            },
            "_meta": {"title": "Spectrum (approximate, opt-in)"},
        }
        graph["9"]["inputs"]["model"] = ["17", 0]
        graph["10"]["inputs"]["model"] = ["17", 0]

    return graph


class MiniMaxH3BaseProcessor(ComfyUIProcessor, VideoOutputHandler):
    """Shared native H3 graph construction and output handling."""

    REQUIRE_START_IMAGE = False

    def process(self):
        if DEV_MODE:
            return
        if not self.job:
            self._fail_job("MiniMax-H3 job object is missing.")
            return
        if not _as_bool(
            os.environ.get("OPENFORK_MINIMAX_H3_LICENSE_ACKNOWLEDGED")
        ):
            self._fail_job(
                "MiniMax-H3 is license-gated. Its community license excludes the EU, "
                "UK, South Korea, and USA. Obtain applicable authorization, then set "
                "OPENFORK_MINIMAX_H3_LICENSE_ACKNOWLEDGED=true on this worker."
            )
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        inputs = self.job.get("inputs") or {}
        tier_source = (
            self.job.get("service_type")
            or inputs.get("model")
            or self.job.get("workflow_type")
        )
        tier_gb = resolve_minimax_h3_tier(tier_source)
        width, height = resolve_minimax_h3_dimensions(
            inputs.get("aspect_ratio", "16:9"),
            tier_gb,
            inputs.get("target_width") or inputs.get("width"),
            inputs.get("target_height") or inputs.get("height"),
        )
        requested_duration, frames = resolve_minimax_h3_frames(
            inputs.get("duration") or inputs.get("duration_seconds"), tier_gb
        )
        steps = _as_int(inputs.get("steps"), DEFAULT_STEPS, 10, 40)
        seed, seed_source = resolve_minimax_h3_seed(inputs.get("seed"))
        prompt, audio_prompt = build_ltx23_prompt(self.positive_prompt, inputs)
        acceleration_mode = str(inputs.get("acceleration_mode") or "native").strip().lower()
        if _as_bool(inputs.get("use_spectrum")):
            acceleration_mode = "spectrum"
        if acceleration_mode not in {"native", "sage", "spectrum"}:
            acceleration_mode = "native"

        start_image_path, start_image_owned = self._resolve_start_image(inputs)
        if self.REQUIRE_START_IMAGE and not start_image_path:
            self._fail_job("MiniMax-H3 image-to-video requires a start image or source video.")
            return

        start_image_filename = os.path.basename(start_image_path) if start_image_path else None
        container_input_path = None
        try:
            if start_image_path:
                container_input_path = f"/opt/ComfyUI/input/{start_image_filename}"
                self._copy_start_image(start_image_path, container_input_path)

            graph = build_minimax_h3_workflow(
                workflow_data,
                prompt=prompt,
                width=width,
                height=height,
                frames=frames,
                steps=steps,
                seed=seed,
                filename_prefix=f"video/minimax-h3/{self.job_id}",
                start_image_filename=start_image_filename,
                acceleration_mode=acceleration_mode,
            )
            timeout = 21600 if tier_gb <= 8 else 14400
            outputs = self._trigger_and_get_output({"prompt": graph}, timeout_sec=timeout)
            if not outputs:
                return
            result = self.handle_video_output(outputs)
            if not result:
                return

            video_storage_path, thumbnail_storage_path, actual_duration = result
            metadata = self._video_completion_metadata()
            metadata.update(
                {
                    "model": "MiniMax-H3",
                    "native_audio_video": True,
                    "audio_sample_rate_hz": 32000,
                    "audio_channels": 2,
                    "tier_gb": tier_gb,
                    "experimental_low_vram_offload": tier_gb <= 8,
                    "requested_resolution": f"{width}x{height}",
                    "requested_duration_seconds": requested_duration,
                    "requested_video_length_frames": frames,
                    "requested_fps": FPS,
                    "steps": steps,
                    "seed": seed,
                    "seed_source": seed_source,
                    "acceleration_mode": acceleration_mode,
                    "spectrum_is_approximate": acceleration_mode == "spectrum",
                }
            )
            if audio_prompt:
                metadata["audio_prompt"] = audio_prompt

            self.orchestrator_service.update_job_status(
                self.job_id,
                "completed",
                storage_path=video_storage_path,
                thumbnail_storage_path=thumbnail_storage_path,
                duration_seconds=actual_duration,
                completion_metadata=metadata,
                prompt=prompt,
            )
            logging.info("MiniMax-H3 job %s completed successfully", self.job_id)
        except Exception as exc:
            logging.error("MiniMax-H3 job %s failed: %s", self.job_id, exc, exc_info=True)
            self._fail_job(f"MiniMax-H3 generation failed: {exc}")
        finally:
            if container_input_path and (not HEADLESS_MODE or start_image_owned):
                self._cleanup_container_file(container_input_path, "MiniMax-H3 input image")
            if start_image_path and start_image_owned:
                self._cleanup_local_file(start_image_path, "MiniMax-H3 local input image")

    def _resolve_start_image(self, inputs: dict) -> tuple[Optional[str], bool]:
        if not self.REQUIRE_START_IMAGE:
            return None, False

        last_frame = materialize_last_frame_start_image(
            self,
            inputs,
            target_dimensions=resolve_minimax_h3_dimensions(
                inputs.get("aspect_ratio", "16:9"),
                resolve_minimax_h3_tier(
                    self.job.get("service_type")
                    or inputs.get("model")
                    or self.job.get("workflow_type")
                ),
            ),
        )
        if last_frame:
            return last_frame, True

        start_image_url = inputs.get("start_image_url")
        if start_image_url:
            downloaded = self.orchestrator_service.download_asset_by_url(
                start_image_url, self.input_dir
            )
            if downloaded:
                return downloaded, True

        filename = materialize_start_image(self.job, self.input_dir)
        if filename:
            owned = bool(
                self.job.get("start_image_base64")
                or inputs.get("start_image_base64")
            )
            return os.path.join(self.input_dir, filename), owned

        storage_path = (
            inputs.get("start_image_storage_path")
            or inputs.get("input_storage_path")
            or self.job.get("input_storage_path")
        )
        if storage_path:
            bucket = inputs.get("start_image_bucket") or self.job.get("bucket", "projects_public")
            downloaded = self.orchestrator_service.download_storage_asset(
                bucket, str(storage_path), self.input_dir
            )
            if downloaded:
                return downloaded, True
            supabase_url = os.environ.get("SUPABASE_URL", SUPABASE_URL)
            if supabase_url:
                downloaded = self.orchestrator_service.download_asset_by_url(
                    f"{supabase_url}/storage/v1/object/public/{bucket}/{storage_path}",
                    self.input_dir,
                )
                return downloaded, bool(downloaded)
        return None, False

    def _copy_start_image(self, source_path: str, container_path: str) -> None:
        if docker_manager:
            docker_manager.copy_file_to_container(
                service_type=self.client.active_service_type,
                source_on_host=source_path,
                dest_in_container=container_path,
                shutdown_event=self.shutdown_event,
            )
            return
        os.makedirs(os.path.dirname(container_path), exist_ok=True)
        if os.path.abspath(source_path) != os.path.abspath(container_path):
            shutil.copy2(source_path, container_path)


class MiniMaxH3TextToVideoProcessor(MiniMaxH3BaseProcessor):
    """MiniMax-H3 prompt-to-video with native synchronized audio."""


class MiniMaxH3ImageToVideoProcessor(MiniMaxH3BaseProcessor):
    """MiniMax-H3 first-frame/last-frame conditioned audio-video."""

    REQUIRE_START_IMAGE = True
