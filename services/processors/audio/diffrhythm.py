"""
DiffRhythm Music Generation Processor (ComfyUI-based)
"""

import os
import json
import logging

from services.processors.comfyui_processor import ComfyUIProcessor
from services.processors.output_handlers import AudioOutputHandler
from services.orchestrator_service import TokenExpiredError
from utils.comfyui_workflow_utils import inject_prompt_into_diffrhythm_workflow


class DiffRhythmJobProcessor(ComfyUIProcessor, AudioOutputHandler):
    """Processor for DiffRhythm music generation via ComfyUI."""

    def process(self):
        if not self.job:
            self._fail_job(f"Job object is None for DiffRhythmJobProcessor. Cannot proceed.")
            return

        workflow_data = self._get_workflow_payload()
        if not workflow_data:
            return

        lyrics_or_edit_lyrics, style_prompt = self._parse_prompt()

        wf_ready = inject_prompt_into_diffrhythm_workflow(workflow_data, lyrics_or_edit_lyrics, style_prompt)
        payload = {"prompt": wf_ready}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        result = self.handle_audio_output(outputs)
        if not result:
            return

        audio_storage_path, duration = result

        try:
            completion_metadata = self.job.get("completion_metadata") or {}
            completion_metadata.update(
                {
                    "lyrics_length": len(lyrics_or_edit_lyrics),
                    "style_prompt_length": len(style_prompt),
                    "has_lyrics": bool(lyrics_or_edit_lyrics.strip()),
                }
            )

            self.orchestrator_service.update_job_status(
                self.job_id,
                "completed",
                storage_path=audio_storage_path,
                duration_seconds=duration,
                completion_metadata=completion_metadata,
            )
        except TokenExpiredError:
            raise

    def _parse_prompt(self) -> tuple:
        """Parse enhanced prompt structure to extract lyrics and style."""
        lyrics_or_edit_lyrics = ""
        style_prompt = self.positive_prompt

        try:
            if self.positive_prompt:
                parsed_prompt = json.loads(self.positive_prompt)
                lyrics_or_edit_lyrics = parsed_prompt.get("lyrics_or_edit_lyrics", "")
                style_prompt = parsed_prompt.get("style_prompt", self.positive_prompt)
                logging.info(
                    f"Parsed enhanced prompt - Lyrics: {len(lyrics_or_edit_lyrics)} chars, Style: {len(style_prompt)} chars"
                )
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse enhanced prompt structure for job {self.job_id}, using fallback")
            lyrics_or_edit_lyrics = ""
            style_prompt = self.positive_prompt

        return lyrics_or_edit_lyrics, style_prompt
