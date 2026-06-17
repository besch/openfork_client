"""
dots.tts job processors.

dots.tts is exposed through a REST API wrapper with two OpenFork workflows:
- DotsTTSJobProcessor: text-to-speech / random voice sampling
- DotsTTSVoiceCloneJobProcessor: zero-shot voice cloning from reference audio
"""

import logging
import os
from typing import Dict, Optional

import requests

from config import SUPABASE_URL
from services.orchestrator_service import TokenExpiredError
from services.processors.base import BaseJobProcessor
from services.processors.rest_recovery import (
    poll_rest_job_with_clean_exit,
    recover_output_from_clean_container_exit,
)
from utils.media_utils import get_audio_duration


DOTS_TTS_MAX_WAIT_TIME = int(os.environ.get("DOTS_TTS_MAX_WAIT_TIME", "1800"))


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


def _coerce_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value, default: Optional[int] = None) -> Optional[int]:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_bool(value, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class DotsTTSBaseProcessor(BaseJobProcessor):
    API_PORT = 8000
    POLL_INTERVAL = 2
    MAX_WAIT_TIME = DOTS_TTS_MAX_WAIT_TIME
    SERVICE_LABEL = "dots.tts"

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.api_base_url = f"http://localhost:{self.API_PORT}"

    def _wait_for_api(self, timeout: int = 120) -> bool:
        import time

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_cancelled():
                return False
            try:
                response = requests.get(f"{self.api_base_url}/health", timeout=5)
                if response.status_code == 200:
                    logging.info("dots.tts API is ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

        logging.error("dots.tts API did not become available within %ss", timeout)
        return False

    def _poll_for_completion(self, remote_job_id: str) -> Dict:
        return poll_rest_job_with_clean_exit(
            self,
            self.api_base_url,
            remote_job_id,
            poll_interval=self.POLL_INTERVAL,
            max_wait_time=self.MAX_WAIT_TIME,
            service_label=self.SERVICE_LABEL,
        )

    def _download_output(self, remote_job_id: str) -> Optional[str]:
        os.makedirs(self.cache_dir, exist_ok=True)
        local_path = os.path.join(self.cache_dir, f"{self.job_id}.wav")

        try:
            response = requests.get(
                f"{self.api_base_url}/download/{remote_job_id}",
                timeout=60,
                stream=True,
            )

            if response.status_code == 200:
                with open(local_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=8192):
                        handle.write(chunk)
                logging.info("Downloaded dots.tts output to %s", local_path)
                return local_path

            logging.error("Failed to download dots.tts output: %s", response.status_code)

        except requests.exceptions.RequestException as exc:
            logging.error("Failed to download dots.tts output: %s", exc)

        return recover_output_from_clean_container_exit(
            self,
            local_path,
            container_output_path="/app/output",
            extensions=(".wav",),
        )

    def _cleanup_remote_job(self, remote_job_id: Optional[str]):
        if not remote_job_id:
            return
        try:
            requests.delete(f"{self.api_base_url}/job/{remote_job_id}", timeout=10)
        except requests.exceptions.RequestException:
            pass

    def _cleanup_local_output(self):
        local_path = os.path.join(self.cache_dir, f"{self.job_id}.wav")
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except OSError:
                pass

    def _download_reference(self, inputs: Dict) -> Optional[str]:
        voice_clone_urls = _as_url_list(inputs.get("voice_clone_urls", []))
        voice_clone_storage_path = _input_alias(
            inputs,
            "voice_clone_storage_path",
            "reference_audio",
            "reference_audio_storage_path",
        )

        for voice_clone_url in voice_clone_urls:
            clone_path = self.orchestrator_service.download_asset_by_url(
                voice_clone_url,
                self.input_dir,
            )
            if clone_path:
                return clone_path

        if voice_clone_storage_path:
            bucket = self.job.get("bucket") or "projects_public"
            source_url = (
                f"{SUPABASE_URL}/storage/v1/object/public/"
                f"{bucket}/{voice_clone_storage_path}"
            )
            logging.info(
                "Downloading dots.tts reference from storage path fallback: %s",
                voice_clone_storage_path,
            )
            return self.orchestrator_service.download_asset_by_url(
                source_url,
                self.input_dir,
            )

        return None

    def _base_payload(self, text: str, inputs: Dict) -> Dict:
        language = _input_alias(
            inputs,
            "language",
            "dots_tts_language",
            "qwen3_language",
        )
        return {
            "text": text,
            "language": language,
            "seed": _coerce_int(_input_alias(inputs, "seed", "dots_tts_seed")),
            "num_steps": _coerce_int(
                _input_alias(
                    inputs,
                    "num_steps",
                    "dots_tts_num_steps",
                    "dots_num_steps",
                    "nfe_step",
                )
            ),
            "guidance_scale": _coerce_float(
                _input_alias(
                    inputs,
                    "guidance_scale",
                    "dots_tts_guidance_scale",
                    "dots_guidance_scale",
                    "cfg_strength",
                )
            ),
            "normalize_text": _coerce_bool(
                _input_alias(inputs, "normalize_text", "dots_tts_normalize_text"),
                False,
            ),
        }

    def _submit_generation(self, payload: Dict) -> Optional[str]:
        try:
            response = requests.post(
                f"{self.api_base_url}/generate",
                json=payload,
                timeout=30,
            )
            if response.status_code == 200:
                remote_job_id = response.json().get("job_id")
                logging.info("dots.tts generation submitted: %s", remote_job_id)
                return remote_job_id

            logging.error(
                "Failed to submit dots.tts generation: %s - %s",
                response.status_code,
                response.text,
            )
            return None
        except requests.exceptions.RequestException as exc:
            logging.error("Failed to submit dots.tts generation request: %s", exc)
            return None

    def _submit_voice_clone(self, payload: Dict, ref_audio_path: str) -> Optional[str]:
        try:
            data = {
                key: str(value).lower() if isinstance(value, bool) else str(value)
                for key, value in payload.items()
                if value not in (None, "")
            }
            media_type = "audio/mpeg" if ref_audio_path.lower().endswith(".mp3") else "audio/wav"

            with open(ref_audio_path, "rb") as handle:
                files = {
                    "prompt_audio": (
                        os.path.basename(ref_audio_path),
                        handle,
                        media_type,
                    )
                }
                response = requests.post(
                    f"{self.api_base_url}/generate/voice-clone",
                    data=data,
                    files=files,
                    timeout=30,
                )

            if response.status_code == 200:
                remote_job_id = response.json().get("job_id")
                logging.info("dots.tts voice clone submitted: %s", remote_job_id)
                return remote_job_id

            logging.error(
                "Failed to submit dots.tts voice clone: %s - %s",
                response.status_code,
                response.text,
            )
            return None
        except Exception as exc:
            logging.error("Failed to submit dots.tts voice clone request: %s", exc)
            return None

    def _complete_with_audio(self, local_path: str, metadata: Dict) -> bool:
        audio_storage_path = self.orchestrator_service.upload_audio_output(
            local_path,
            self.job_id,
        )
        if not audio_storage_path:
            self._fail_job(f"dots.tts job {self.job_id} completed, but upload failed")
            return False

        duration = get_audio_duration(local_path)
        completion_metadata = dict(self.job.get("completion_metadata") or {})
        completion_metadata.update(metadata)

        self.orchestrator_service.update_job_status(
            self.job_id,
            "completed",
            storage_path=audio_storage_path,
            duration_seconds=duration,
            completion_metadata=completion_metadata,
        )
        return True


class DotsTTSJobProcessor(DotsTTSBaseProcessor):
    """Processor for dots.tts text-to-speech / random voice sampling."""

    def process(self):
        if not self.job:
            self._fail_job("Job object is None for DotsTTSJobProcessor. Cannot proceed.")
            return

        logging.info("Processing dots.tts job %s", self.job_id)
        inputs = self.job.get("inputs") or {}
        text = self.positive_prompt or inputs.get("text") or inputs.get("prompt") or ""
        if not text:
            self._fail_job("No text provided for dots.tts generation")
            return

        payload = self._base_payload(text, inputs)
        prompt_text = _input_alias(inputs, "prompt_text", "dots_tts_prompt_text", "qwen3_ref_text")
        prompt_audio_path = None
        remote_job_id = None

        try:
            prompt_audio_path = self._download_reference(inputs)
            if prompt_audio_path:
                payload["prompt_audio_path"] = prompt_audio_path
                payload["prompt_text"] = prompt_text

            if not self._wait_for_api():
                if self.is_cancelled():
                    logging.info("dots.tts job %s cancelled while waiting for API", self.job_id)
                    return
                self._fail_job(f"dots.tts API did not become available for job {self.job_id}")
                return

            remote_job_id = self._submit_generation(payload)
            if not remote_job_id:
                self._fail_job(f"Failed to submit dots.tts generation for job {self.job_id}")
                return

            result = self._poll_for_completion(remote_job_id)
            if result.get("status") == "cancelled":
                logging.info("dots.tts job %s cancelled during processing", self.job_id)
                return
            if result.get("status") != "completed":
                self._fail_job(
                    f"dots.tts generation failed: {result.get('error', 'Unknown error')}"
                )
                return

            local_path = self._download_output(remote_job_id)
            if not local_path:
                self._fail_job(f"Failed to download output for dots.tts job {self.job_id}")
                return

            if not self._complete_with_audio(
                local_path,
                {
                    "processor": "DotsTTSJobProcessor",
                    "model": "dots-tts",
                    "voice_clone": False,
                    "has_reference_audio": bool(prompt_audio_path),
                    "has_prompt_text": bool(prompt_text),
                    "dots_tts_model": result.get("model"),
                    "dots_tts_num_steps": result.get("num_steps") or payload.get("num_steps"),
                    "dots_tts_guidance_scale": result.get("guidance_scale")
                    or payload.get("guidance_scale"),
                    "language": payload.get("language"),
                    "seed": result.get("seed"),
                    "requested_seed": payload.get("seed"),
                },
            ):
                return
            logging.info("dots.tts job %s completed successfully", self.job_id)

        except TokenExpiredError:
            raise
        except Exception as exc:
            logging.error("Error processing dots.tts job %s: %s", self.job_id, exc, exc_info=True)
            self._fail_job(f"Error processing job: {exc}")
        finally:
            self._cleanup_remote_job(remote_job_id)
            self._cleanup_local_output()
            if prompt_audio_path and os.path.exists(prompt_audio_path):
                try:
                    os.remove(prompt_audio_path)
                except OSError:
                    pass


class DotsTTSVoiceCloneJobProcessor(DotsTTSBaseProcessor):
    """Processor for dots.tts zero-shot voice cloning."""

    def process(self):
        if not self.job:
            self._fail_job(
                "Job object is None for DotsTTSVoiceCloneJobProcessor. Cannot proceed."
            )
            return

        logging.info("Processing dots.tts voice clone job %s", self.job_id)
        inputs = self.job.get("inputs") or {}
        text = self.positive_prompt or inputs.get("text") or inputs.get("prompt") or ""
        if not text:
            self._fail_job("No text provided for dots.tts voice cloning")
            return

        prompt_text = _input_alias(inputs, "prompt_text", "dots_tts_prompt_text", "qwen3_ref_text")
        clone_path = None
        remote_job_id = None

        try:
            clone_path = self._download_reference(inputs)
            if not clone_path:
                self._fail_job("No dots.tts voice clone reference audio provided")
                return

            if not self._wait_for_api():
                if self.is_cancelled():
                    logging.info(
                        "dots.tts voice clone job %s cancelled while waiting for API",
                        self.job_id,
                    )
                    return
                self._fail_job(f"dots.tts API did not become available for job {self.job_id}")
                return

            payload = self._base_payload(text, inputs)
            payload["prompt_text"] = prompt_text

            remote_job_id = self._submit_voice_clone(payload, clone_path)
            if not remote_job_id:
                self._fail_job(f"Failed to submit dots.tts voice clone for job {self.job_id}")
                return

            result = self._poll_for_completion(remote_job_id)
            if result.get("status") == "cancelled":
                logging.info(
                    "dots.tts voice clone job %s cancelled during processing",
                    self.job_id,
                )
                return
            if result.get("status") != "completed":
                self._fail_job(
                    f"dots.tts voice clone failed: {result.get('error', 'Unknown error')}"
                )
                return

            local_path = self._download_output(remote_job_id)
            if not local_path:
                self._fail_job(
                    f"Failed to download output for dots.tts voice clone job {self.job_id}"
                )
                return

            if not self._complete_with_audio(
                local_path,
                {
                    "processor": "DotsTTSVoiceCloneJobProcessor",
                    "model": "dots-tts",
                    "voice_clone": True,
                    "has_prompt_text": bool(prompt_text),
                    "dots_tts_model": result.get("model"),
                    "dots_tts_num_steps": result.get("num_steps") or payload.get("num_steps"),
                    "dots_tts_guidance_scale": result.get("guidance_scale")
                    or payload.get("guidance_scale"),
                    "language": payload.get("language"),
                    "seed": result.get("seed"),
                    "requested_seed": payload.get("seed"),
                },
            ):
                return
            logging.info("dots.tts voice clone job %s completed successfully", self.job_id)

        except TokenExpiredError:
            raise
        except Exception as exc:
            logging.error(
                "Error processing dots.tts voice clone job %s: %s",
                self.job_id,
                exc,
                exc_info=True,
            )
            self._fail_job(f"Error processing job: {exc}")
        finally:
            self._cleanup_remote_job(remote_job_id)
            self._cleanup_local_output()
            if clone_path and os.path.exists(clone_path):
                try:
                    os.remove(clone_path)
                except OSError:
                    pass
