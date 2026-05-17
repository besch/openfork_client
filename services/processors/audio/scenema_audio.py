"""
Scenema Audio job processors.

Scenema exposes a synchronous REST endpoint that returns base64 WAV data. These
processors adapt OpenFork TTS and voice clone jobs into that API.
"""

import base64
import html
import logging
import os
import time
from typing import Dict, Optional

import requests

from config import SUPABASE_URL
from services.orchestrator_service import TokenExpiredError
from services.processors.base import BaseJobProcessor
from utils.media_utils import get_audio_duration


SCENEMA_MAX_WAIT_TIME = int(os.environ.get("SCENEMA_AUDIO_MAX_WAIT_TIME", "1800"))


def _input_alias(inputs: Dict, *keys: str, default=None):
    for key in keys:
        value = inputs.get(key)
        if value not in (None, ""):
            return value
    return default


def _as_url_list(value) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item.strip()]
    return []


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _looks_like_speak_xml(prompt: str) -> bool:
    stripped = prompt.strip()
    return stripped.startswith("<speak") and stripped.endswith("</speak>")


class ScenemaAudioBaseProcessor(BaseJobProcessor):
    """Shared Scenema Audio processor implementation."""

    API_PORT = 8000
    MAX_WAIT_TIME = SCENEMA_MAX_WAIT_TIME

    reference_required = False
    processor_name = "ScenemaAudioBaseProcessor"

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.api_base_url = f"http://localhost:{self.API_PORT}"

    def process(self):
        if not self.job:
            self._fail_job("Job object is None for Scenema Audio processor. Cannot proceed.")
            return

        inputs = self.job.get("inputs") or {}
        text = self.positive_prompt or inputs.get("text") or inputs.get("prompt", "")
        if not text:
            self._fail_job("No text provided for Scenema Audio generation")
            return

        reference_voice_url = self._resolve_reference_url(inputs)
        if self.reference_required and not reference_voice_url:
            self._fail_job("No voice clone reference audio provided")
            return

        if not self._wait_for_api():
            if self.is_cancelled():
                logging.info("Scenema Audio job %s cancelled while waiting for API", self.job_id)
                return
            self._fail_job(f"Scenema Audio API did not become available for job {self.job_id}")
            return

        local_path = None
        try:
            payload = self._build_payload(text, inputs, reference_voice_url)
            local_path = self._submit_generation(payload)
            if not local_path:
                self._fail_job(f"Failed to generate Scenema Audio output for job {self.job_id}")
                return

            audio_storage_path = self.orchestrator_service.upload_audio_output(
                local_path,
                self.job_id,
            )
            if not audio_storage_path:
                self._fail_job(f"Scenema Audio job {self.job_id} completed, but upload failed")
                return

            duration = get_audio_duration(local_path)
            completion_metadata = self.job.get("completion_metadata") or {}
            completion_metadata.update(
                {
                    "processor": self.processor_name,
                    "scenema_mode": payload.get("mode"),
                    "has_reference_voice": bool(reference_voice_url),
                    "background_sfx": payload.get("background_sfx"),
                    "validate": payload.get("validate"),
                }
            )

            self.orchestrator_service.update_job_status(
                self.job_id,
                "completed",
                storage_path=audio_storage_path,
                duration_seconds=duration,
                completion_metadata=completion_metadata,
            )
            logging.info("Scenema Audio job %s completed successfully", self.job_id)
        except TokenExpiredError:
            raise
        except Exception as exc:
            logging.error("Error processing Scenema Audio job %s: %s", self.job_id, exc, exc_info=True)
            self._fail_job(f"Error processing Scenema Audio job: {exc}")
        finally:
            if local_path and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except OSError:
                    pass

    def _wait_for_api(self, timeout: int = 600) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_cancelled():
                return False
            try:
                response = requests.get(f"{self.api_base_url}/health", timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        return False

    def _build_payload(self, text: str, inputs: Dict, reference_voice_url: Optional[str]) -> Dict:
        prompt_xml = self._build_prompt_xml(text, inputs)
        seed = _input_alias(inputs, "seed", default=-1)

        payload = {
            "prompt": prompt_xml,
            "mode": _input_alias(inputs, "scenema_mode", "mode", default="generate"),
            "background_sfx": _as_bool(
                _input_alias(inputs, "scenema_background_sfx", "background_sfx"),
                False,
            ),
            "validate": _as_bool(
                _input_alias(inputs, "scenema_validate", "validate"),
                True,
            ),
            "seed": _as_int(seed, -1),
            "pace": _as_float(_input_alias(inputs, "scenema_pace", "pace"), 1.5),
            "min_match_ratio": _as_float(
                _input_alias(inputs, "scenema_min_match_ratio", "min_match_ratio"),
                0.90,
            ),
            "vc_cfg_rate": _as_float(
                _input_alias(inputs, "scenema_vc_cfg_rate", "vc_cfg_rate"),
                0.5,
            ),
            "vc_steps": _as_int(
                _input_alias(inputs, "scenema_vc_steps", "vc_steps"),
                25,
            ),
            "skip_vc": _as_bool(
                _input_alias(inputs, "scenema_skip_vc", "skip_vc"),
                False,
            ),
        }

        if payload["mode"] not in {"generate", "voice_design"}:
            payload["mode"] = "generate"

        if reference_voice_url:
            payload["reference_voice_url"] = reference_voice_url
            payload["mode"] = "generate"

        return payload

    def _build_prompt_xml(self, text: str, inputs: Dict) -> str:
        if _looks_like_speak_xml(text):
            return text.strip()

        voice = html.escape(
            str(
                _input_alias(
                    inputs,
                    "scenema_voice",
                    "voice",
                    default="Warm expressive narrator, natural breath and clear emotion.",
                )
            ),
            quote=True,
        )
        scene = html.escape(
            str(_input_alias(inputs, "scenema_scene", "scene", default="Absolute silence.")),
            quote=True,
        )
        language = html.escape(
            str(_input_alias(inputs, "scenema_language", "language", default="en")),
            quote=True,
        )
        shot = html.escape(
            str(_input_alias(inputs, "scenema_shot", "shot", default="closeup")),
            quote=True,
        )
        line = html.escape(text.strip(), quote=False)

        return (
            f'<speak voice="{voice}" scene="{scene}" language="{language}" shot="{shot}">'
            f"{line}</speak>"
        )

    def _resolve_reference_url(self, inputs: Dict) -> Optional[str]:
        direct_reference = _input_alias(
            inputs,
            "reference_voice_url",
            "reference_audio_url",
        )
        if direct_reference:
            return str(direct_reference)

        voice_clone_urls = _as_url_list(inputs.get("voice_clone_urls", []))
        if voice_clone_urls:
            return voice_clone_urls[0]

        storage_path = _input_alias(
            inputs,
            "voice_clone_storage_path",
            "reference_audio",
            "reference_audio_storage_path",
        )
        if storage_path:
            bucket = self.job.get("bucket") or "projects_public"
            return f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{storage_path}"

        return None

    def _submit_generation(self, payload: Dict) -> Optional[str]:
        try:
            response = requests.post(
                f"{self.api_base_url}/generate",
                json=payload,
                timeout=self.MAX_WAIT_TIME,
            )
            if response.status_code != 200:
                logging.error(
                    "Scenema Audio submit failed: %s - %s",
                    response.status_code,
                    response.text,
                )
                return None

            data = response.json()
            if data.get("status") != "succeeded":
                logging.error("Scenema Audio generation failed: %s", data.get("error"))
                return None

            audio_b64 = data.get("audio")
            if not audio_b64:
                logging.error("Scenema Audio response did not include audio")
                return None

            os.makedirs(self.cache_dir, exist_ok=True)
            local_path = os.path.join(self.cache_dir, f"{self.job_id}.wav")
            with open(local_path, "wb") as handle:
                handle.write(base64.b64decode(audio_b64))
            return local_path
        except requests.exceptions.RequestException as exc:
            logging.error("Failed to submit Scenema Audio request: %s", exc)
            return None


class ScenemaAudioTTSProcessor(ScenemaAudioBaseProcessor):
    """Processor for Scenema text-to-speech and voice design."""

    processor_name = "ScenemaAudioTTSProcessor"
    reference_required = False


class ScenemaAudioVoiceCloneProcessor(ScenemaAudioBaseProcessor):
    """Processor for Scenema voice cloning."""

    processor_name = "ScenemaAudioVoiceCloneProcessor"
    reference_required = True
