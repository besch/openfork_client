"""Helpers for image-to-video continuations from a source video's last frame."""

from __future__ import annotations

import logging
import os
from typing import Optional

from config import SUPABASE_URL
from utils.media_utils import VIDEO_EXTENSIONS, extract_last_frame

logger = logging.getLogger(__name__)


def workflow_uses_last_frame(job: dict, inputs: dict) -> bool:
    workflow_type = str(job.get("workflow_type") or inputs.get("model") or "").lower()
    return "from-last-frame" in workflow_type


def resolve_input_video_url(processor, inputs: dict) -> Optional[str]:
    job = processor.job or {}
    url = (
        inputs.get("input_video_url")
        or job.get("input_video_url")
        or inputs.get("source_video_url")
        or inputs.get("video_url")
    )
    if url:
        return str(url)

    if workflow_uses_last_frame(job, inputs):
        start_image_url = inputs.get("start_image_url")
        if start_image_url:
            logger.info(
                "Using start_image_url as source video URL for last-frame continuation."
            )
            return str(start_image_url)

    storage_path = (
        inputs.get("input_video_storage_path")
        or inputs.get("source_video_storage_path")
        or inputs.get("input_storage_path")
        or job.get("input_storage_path")
    )
    if not isinstance(storage_path, str) or not storage_path:
        return None

    storage_path_lower = storage_path.split("?", 1)[0].lower()
    if not (
        workflow_uses_last_frame(job, inputs)
        or storage_path_lower.endswith(VIDEO_EXTENSIONS)
    ):
        return None

    supabase_url = os.environ.get(
        "SUPABASE_URL",
        getattr(processor.client, "config", {}).get("SUPABASE_URL", SUPABASE_URL),
    )
    if not supabase_url:
        return None

    bucket = job.get("bucket", "projects_public")
    return f"{supabase_url}/storage/v1/object/public/{bucket}/{storage_path}"


def materialize_last_frame_start_image(
    processor,
    inputs: dict,
    *,
    target_dimensions: Optional[tuple[int, int]] = None,
) -> Optional[str]:
    """Return a local image path extracted from an input/source video, if present."""

    input_video_url = resolve_input_video_url(processor, inputs)
    if not input_video_url:
        return None

    video_path = processor.orchestrator_service.download_asset_by_url(
        input_video_url,
        processor.input_dir,
    )
    if not video_path:
        return None

    output_path = os.path.join(processor.input_dir, f"{processor.job_id}_last_frame.jpg")
    if not extract_last_frame(
        video_path,
        output_path,
        target_dimensions=target_dimensions,
    ):
        return None

    logger.info(
        "Extracted last frame from source video for job %s: %s",
        processor.job_id,
        os.path.basename(output_path),
    )
    return output_path
