"""
F5-TTS job processors.

F5-TTS is exposed through a REST API wrapper with two OpenFork workflows:
- F5TTSJobProcessor: default-reference text-to-speech
- F5VoiceCloneJobProcessor: zero-shot voice cloning from reference audio
"""

import logging
import os
import time
from typing import Dict, Optional

import requests

from config import SUPABASE_URL
from services.docker_manager import docker_manager
from services.orchestrator_service import TokenExpiredError
from services.processors.base import BaseJobProcessor
from services.processors.rest_recovery import recover_output_from_clean_container_exit
from utils.media_utils import get_audio_duration


F5_MAX_WAIT_TIME = int(os.environ.get("F5_TTS_MAX_WAIT_TIME", "1800"))


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


def _coerce_float(value, default: float) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _coerce_bool(value, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _f5_service_type(processor: BaseJobProcessor) -> Optional[str]:
    service_type = processor.job.get("service_type")
    if service_type:
        return service_type
    try:
        return processor.client.get_service_type_for_workflow(processor.workflow_type)
    except Exception:
        return None


def _f5_container_state(processor: BaseJobProcessor) -> Optional[Dict]:
    service_type = _f5_service_type(processor)
    if not service_type:
        return None
    try:
        container_name = docker_manager.get_container_name(service_type)
        container = docker_manager.client.containers.get(container_name)
        container.reload()
        state = dict(container.attrs.get("State", {}) or {})
        state["container_name"] = container_name
        return state
    except Exception as exc:
        if exc.__class__.__name__ == "NotFound":
            logging.warning(
                "F5-TTS container for service %s disappeared while polling.",
                service_type,
            )
            return {"Status": "not_found", "ExitCode": None, "OOMKilled": False}
        logging.debug("Could not inspect F5-TTS container state: %s", exc)
        return None


def _f5_container_exited_cleanly(processor: BaseJobProcessor) -> bool:
    state = _f5_container_state(processor)
    if not state:
        return False
    return (
        state.get("Status") in ("exited", "dead")
        and state.get("ExitCode") == 0
        and not state.get("OOMKilled", False)
    )


def _f5_stopped_failure(processor: BaseJobProcessor) -> Optional[str]:
    state = _f5_container_state(processor)
    if not state:
        return None

    status = state.get("Status")
    if status == "not_found":
        return "F5-TTS API container disappeared before a final status response"

    if status not in ("exited", "dead"):
        return None

    exit_code = state.get("ExitCode")
    oom_killed = bool(state.get("OOMKilled", False))
    if exit_code == 0 and not oom_killed:
        return None

    if oom_killed:
        return f"F5-TTS API container stopped while polling (exit code {exit_code}, OOM killed)"
    return f"F5-TTS API container stopped while polling (exit code {exit_code})"


def _copy_f5_output_from_container(
    processor: BaseJobProcessor,
    local_path: str,
) -> Optional[str]:
    return recover_output_from_clean_container_exit(
        processor,
        local_path,
        container_output_path="/app/output",
        extensions=(".wav",),
    )


def _poll_f5_for_completion(
    processor: BaseJobProcessor,
    api_base_url: str,
    remote_job_id: str,
    poll_interval: int,
    max_wait_time: int,
) -> Dict:
    start_time = time.time()
    saw_remote_job = False
    consecutive_connection_errors = 0

    while time.time() - start_time < max_wait_time:
        if processor.is_cancelled():
            return {"status": "cancelled", "error": "Shutdown requested"}

        try:
            response = requests.get(f"{api_base_url}/status/{remote_job_id}", timeout=10)
            consecutive_connection_errors = 0

            if response.status_code == 200:
                saw_remote_job = True
                data = response.json()
                status = data.get("status")

                if status == "completed":
                    logging.info("F5-TTS remote job %s completed", remote_job_id)
                    return data
                if status == "failed":
                    logging.error(
                        "F5-TTS remote job %s failed: %s",
                        remote_job_id,
                        data.get("error"),
                    )
                    return data
                logging.debug("F5-TTS remote job %s status: %s", remote_job_id, status)
            else:
                logging.warning("F5-TTS status check returned %s", response.status_code)

        except requests.exceptions.RequestException as exc:
            consecutive_connection_errors += 1
            logging.warning("F5-TTS status check failed: %s", exc)
            if saw_remote_job and consecutive_connection_errors >= 3:
                if _f5_container_exited_cleanly(processor):
                    logging.warning(
                        "F5-TTS API exited cleanly before a final status response "
                        "for remote job %s; attempting output recovery.",
                        remote_job_id,
                    )
                    return {
                        "status": "completed",
                        "recovered_from_clean_container_exit": True,
                    }

                stopped_error = _f5_stopped_failure(processor)
                if stopped_error:
                    logging.error("%s for remote job %s", stopped_error, remote_job_id)
                    return {"status": "failed", "error": stopped_error}

        processor.shutdown_event.wait(poll_interval)

    return {"status": "failed", "error": "Timeout waiting for F5-TTS generation"}


class F5BaseProcessor(BaseJobProcessor):
    API_PORT = 8000
    POLL_INTERVAL = 2
    MAX_WAIT_TIME = F5_MAX_WAIT_TIME

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.api_base_url = f"http://localhost:{self.API_PORT}"

    def _wait_for_api(self, timeout: int = 90) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_cancelled():
                return False
            try:
                response = requests.get(f"{self.api_base_url}/health", timeout=5)
                if response.status_code == 200:
                    logging.info("F5-TTS API is ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

        logging.error("F5-TTS API did not become available within %ss", timeout)
        return False

    def _poll_for_completion(self, remote_job_id: str) -> Dict:
        return _poll_f5_for_completion(
            self,
            self.api_base_url,
            remote_job_id,
            self.POLL_INTERVAL,
            self.MAX_WAIT_TIME,
        )

    def _download_output(self, remote_job_id: str) -> Optional[str]:
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            local_path = os.path.join(self.cache_dir, f"{self.job_id}.wav")
            response = requests.get(
                f"{self.api_base_url}/download/{remote_job_id}",
                timeout=60,
                stream=True,
            )

            if response.status_code == 200:
                with open(local_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=8192):
                        handle.write(chunk)
                logging.info("Downloaded F5-TTS output to %s", local_path)
                return local_path

            logging.error("Failed to download F5-TTS output: %s", response.status_code)
            return _copy_f5_output_from_container(self, local_path)

        except requests.exceptions.RequestException as exc:
            logging.error("Failed to download F5-TTS output: %s", exc)
            return _copy_f5_output_from_container(self, local_path)

    def _cleanup_remote_job(self, remote_job_id: Optional[str]):
        if not remote_job_id:
            return
        try:
            requests.delete(f"{self.api_base_url}/job/{remote_job_id}", timeout=10)
        except requests.exceptions.RequestException:
            pass

    def _cleanup_local_file(self):
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
                "Downloading F5-TTS reference from storage path fallback: %s",
                voice_clone_storage_path,
            )
            return self.orchestrator_service.download_asset_by_url(
                source_url,
                self.input_dir,
            )

        return None

    def _submit_generation(self, payload: Dict) -> Optional[str]:
        try:
            response = requests.post(
                f"{self.api_base_url}/generate",
                json=payload,
                timeout=30,
            )
            if response.status_code == 200:
                remote_job_id = response.json().get("job_id")
                logging.info("F5-TTS generation submitted: %s", remote_job_id)
                return remote_job_id

            logging.error(
                "Failed to submit F5-TTS generation: %s - %s",
                response.status_code,
                response.text,
            )
            return None
        except requests.exceptions.RequestException as exc:
            logging.error("Failed to submit F5-TTS generation request: %s", exc)
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
                    "ref_audio": (
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
                logging.info("F5-TTS voice clone submitted: %s", remote_job_id)
                return remote_job_id

            logging.error(
                "Failed to submit F5-TTS voice clone: %s - %s",
                response.status_code,
                response.text,
            )
            return None
        except Exception as exc:
            logging.error("Failed to submit F5-TTS voice clone request: %s", exc)
            return None

    def _complete_with_audio(self, local_path: str, metadata: Dict) -> bool:
        audio_storage_path = self.orchestrator_service.upload_audio_output(
            local_path,
            self.job_id,
        )
        if not audio_storage_path:
            self._fail_job(f"F5-TTS job {self.job_id} completed, but upload failed")
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


class F5TTSJobProcessor(F5BaseProcessor):
    """Processor for F5-TTS default-reference speech generation."""

    def process(self):
        if not self.job:
            self._fail_job("Job object is None for F5TTSJobProcessor. Cannot proceed.")
            return

        logging.info("Processing F5-TTS job %s", self.job_id)
        inputs = self.job.get("inputs") or {}
        text = self.positive_prompt or inputs.get("text") or inputs.get("prompt") or ""
        if not text:
            self._fail_job("No text provided for F5-TTS generation")
            return

        payload = {
            "text": text,
            "speed": _coerce_float(_input_alias(inputs, "speed", "f5_speed"), 1.0),
            "seed": _coerce_int(_input_alias(inputs, "seed", "f5_seed")),
            "nfe_step": _coerce_int(_input_alias(inputs, "nfe_step", "f5_nfe_step")),
            "cfg_strength": _coerce_float(
                _input_alias(inputs, "cfg_strength", "f5_cfg_strength"),
                2.0,
            ),
            "remove_silence": _coerce_bool(
                _input_alias(inputs, "remove_silence", "f5_remove_silence"),
                False,
            ),
        }

        remote_job_id = None
        try:
            if not self._wait_for_api():
                if self.is_cancelled():
                    logging.info("F5-TTS job %s cancelled while waiting for API", self.job_id)
                    return
                self._fail_job(f"F5-TTS API did not become available for job {self.job_id}")
                return

            remote_job_id = self._submit_generation(payload)
            if not remote_job_id:
                self._fail_job(f"Failed to submit F5-TTS generation for job {self.job_id}")
                return

            result = self._poll_for_completion(remote_job_id)
            if result.get("status") == "cancelled":
                logging.info("F5-TTS job %s cancelled during processing", self.job_id)
                return
            if result.get("status") != "completed":
                self._fail_job(
                    f"F5-TTS generation failed: {result.get('error', 'Unknown error')}"
                )
                return

            local_path = self._download_output(remote_job_id)
            if not local_path:
                self._fail_job(f"Failed to download output for F5-TTS job {self.job_id}")
                return

            if not self._complete_with_audio(
                local_path,
                {
                    "processor": "F5TTSJobProcessor",
                    "model": "f5-tts",
                    "voice_clone": False,
                    "f5_speed": payload["speed"],
                    "seed": result.get("seed") or payload.get("seed"),
                },
            ):
                return
            logging.info("F5-TTS job %s completed successfully", self.job_id)

        except TokenExpiredError:
            raise
        except Exception as exc:
            logging.error("Error processing F5-TTS job %s: %s", self.job_id, exc, exc_info=True)
            self._fail_job(f"Error processing job: {exc}")
        finally:
            self._cleanup_remote_job(remote_job_id)
            self._cleanup_local_file()


class F5VoiceCloneJobProcessor(F5BaseProcessor):
    """Processor for F5-TTS zero-shot voice cloning."""

    def process(self):
        if not self.job:
            self._fail_job("Job object is None for F5VoiceCloneJobProcessor. Cannot proceed.")
            return

        logging.info("Processing F5-TTS voice clone job %s", self.job_id)
        inputs = self.job.get("inputs") or {}
        text = self.positive_prompt or inputs.get("text") or inputs.get("prompt") or ""
        if not text:
            self._fail_job("No text provided for F5-TTS voice cloning")
            return

        ref_text = _input_alias(inputs, "ref_text", "f5_ref_text", "qwen3_ref_text")
        clone_path = None
        remote_job_id = None

        try:
            clone_path = self._download_reference(inputs)
            if not clone_path:
                self._fail_job("No F5-TTS voice clone reference audio provided")
                return

            if not self._wait_for_api():
                if self.is_cancelled():
                    logging.info(
                        "F5-TTS voice clone job %s cancelled while waiting for API",
                        self.job_id,
                    )
                    return
                self._fail_job(f"F5-TTS API did not become available for job {self.job_id}")
                return

            payload = {
                "text": text,
                "ref_text": ref_text,
                "speed": _coerce_float(_input_alias(inputs, "speed", "f5_speed"), 1.0),
                "seed": _coerce_int(_input_alias(inputs, "seed", "f5_seed")),
                "nfe_step": _coerce_int(_input_alias(inputs, "nfe_step", "f5_nfe_step")),
                "cfg_strength": _coerce_float(
                    _input_alias(inputs, "cfg_strength", "f5_cfg_strength"),
                    2.0,
                ),
                "remove_silence": _coerce_bool(
                    _input_alias(inputs, "remove_silence", "f5_remove_silence"),
                    False,
                ),
            }

            remote_job_id = self._submit_voice_clone(payload, clone_path)
            if not remote_job_id:
                self._fail_job(f"Failed to submit F5-TTS voice clone for job {self.job_id}")
                return

            result = self._poll_for_completion(remote_job_id)
            if result.get("status") == "cancelled":
                logging.info(
                    "F5-TTS voice clone job %s cancelled during processing",
                    self.job_id,
                )
                return
            if result.get("status") != "completed":
                self._fail_job(
                    f"F5-TTS voice clone failed: {result.get('error', 'Unknown error')}"
                )
                return

            local_path = self._download_output(remote_job_id)
            if not local_path:
                self._fail_job(
                    f"Failed to download output for F5-TTS voice clone job {self.job_id}"
                )
                return

            if not self._complete_with_audio(
                local_path,
                {
                    "processor": "F5VoiceCloneJobProcessor",
                    "model": "f5-tts",
                    "voice_clone": True,
                    "has_ref_text": bool(ref_text),
                    "f5_speed": payload["speed"],
                    "seed": result.get("seed") or payload.get("seed"),
                },
            ):
                return
            logging.info("F5-TTS voice clone job %s completed successfully", self.job_id)

        except TokenExpiredError:
            raise
        except Exception as exc:
            logging.error(
                "Error processing F5-TTS voice clone job %s: %s",
                self.job_id,
                exc,
                exc_info=True,
            )
            self._fail_job(f"Error processing job: {exc}")
        finally:
            self._cleanup_remote_job(remote_job_id)
            self._cleanup_local_file()
            if clone_path and os.path.exists(clone_path):
                try:
                    os.remove(clone_path)
                except OSError:
                    pass
