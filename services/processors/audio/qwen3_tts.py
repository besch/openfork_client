"""
Qwen3-TTS Job Processors

This module contains processors for Qwen3-TTS text-to-speech generation:
- Qwen3TTSJobProcessor: Custom voice TTS with built-in speakers
- Qwen3VoiceDesignJobProcessor: Voice design with natural language descriptions
- Qwen3VoiceCloneJobProcessor: Voice cloning with reference audio
"""

import os
import json
import time
import logging
import requests
from typing import Optional, Dict

from config import SUPABASE_URL
from services.processors.base import BaseJobProcessor
from services.processors.rest_recovery import recover_output_from_clean_container_exit
from services.docker_manager import docker_manager
from services.orchestrator_service import TokenExpiredError
from utils.media_utils import get_audio_duration


QWEN3_MAX_WAIT_TIME = int(os.environ.get("QWEN3_TTS_MAX_WAIT_TIME", "1800"))

QWEN3_LANGUAGE_ALIASES = {
    "auto": "Auto",
    "cn": "Chinese",
    "de": "German",
    "deutsch": "German",
    "en": "English",
    "eng": "English",
    "english": "English",
    "es": "Spanish",
    "fr": "French",
    "french": "French",
    "german": "German",
    "it": "Italian",
    "italian": "Italian",
    "ja": "Japanese",
    "japanese": "Japanese",
    "jp": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "kr": "Korean",
    "mandarin": "Chinese",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "ru": "Russian",
    "russian": "Russian",
    "spanish": "Spanish",
    "zh": "Chinese",
    "zh_cn": "Chinese",
    "zhcn": "Chinese",
}


def _input_alias(inputs: Dict, *keys: str, default=None):
    for key in keys:
        value = inputs.get(key)
        if value not in (None, ""):
            return value
    return default


def _normalize_qwen3_language(language) -> str:
    if not isinstance(language, str):
        return "Auto"

    value = language.strip()
    if not value:
        return "Auto"

    supported = {
        "Auto",
        "Chinese",
        "English",
        "Japanese",
        "Korean",
        "German",
        "French",
        "Russian",
        "Portuguese",
        "Spanish",
        "Italian",
    }
    if value in supported:
        return value

    key = (
        value.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "")
    )
    compact_key = key.replace("_", "")
    return QWEN3_LANGUAGE_ALIASES.get(
        key,
        QWEN3_LANGUAGE_ALIASES.get(compact_key, "Auto"),
    )


def _as_url_list(value) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item.strip()]
    return []


def _qwen_service_type(processor: BaseJobProcessor) -> Optional[str]:
    service_type = processor.job.get("service_type")
    if service_type:
        return service_type
    try:
        return processor.client.get_service_type_for_workflow(processor.workflow_type)
    except Exception:
        return None


def _qwen_container_state(processor: BaseJobProcessor) -> Optional[Dict]:
    service_type = _qwen_service_type(processor)
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
                "Qwen container for service %s disappeared while polling.",
                service_type,
            )
            return {"Status": "not_found", "ExitCode": None, "OOMKilled": False}
        logging.debug("Could not inspect Qwen container state: %s", exc)
        return None


def _qwen_container_exited_cleanly(processor: BaseJobProcessor) -> bool:
    state = _qwen_container_state(processor)
    if not state:
        return False
    return (
        state.get("Status") in ("exited", "dead")
        and state.get("ExitCode") == 0
        and not state.get("OOMKilled", False)
    )


def _qwen_stopped_failure(processor: BaseJobProcessor) -> Optional[str]:
    state = _qwen_container_state(processor)
    if not state:
        return None

    status = state.get("Status")
    if status == "not_found":
        return "Qwen API container disappeared before a final status response"

    if status not in ("exited", "dead"):
        return None

    exit_code = state.get("ExitCode")
    oom_killed = bool(state.get("OOMKilled", False))
    if exit_code == 0 and not oom_killed:
        return None

    if oom_killed:
        return f"Qwen API container stopped while polling (exit code {exit_code}, OOM killed)"
    return f"Qwen API container stopped while polling (exit code {exit_code})"


def _copy_qwen_output_from_container(
    processor: BaseJobProcessor,
    local_path: str,
) -> Optional[str]:
    return recover_output_from_clean_container_exit(
        processor,
        local_path,
        container_output_path="/app/output",
        extensions=(".wav",),
        # Qwen3-TTS can finish the model job, close the HTTP server, and leave
        # the wav in /app/output while the container is still inspectable. Since
        # the desktop client runs this service as one-job-per-container, copying
        # the newest wav is safe and avoids marking a completed generation failed.
        require_clean_exit=False,
    )


def _poll_qwen_for_completion(
    processor: BaseJobProcessor,
    api_base_url: str,
    remote_job_id: str,
    poll_interval: int,
    max_wait_time: int,
) -> Dict:
    """Poll Qwen and recover when its API exits cleanly after writing output."""
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
                    logging.info("Remote job %s completed", remote_job_id)
                    return data
                if status == "failed":
                    logging.error(
                        "Remote job %s failed: %s",
                        remote_job_id,
                        data.get("error"),
                    )
                    return data
                logging.debug("Remote job %s status: %s", remote_job_id, status)
            else:
                logging.warning("Status check returned %s", response.status_code)

        except requests.exceptions.RequestException as exc:
            consecutive_connection_errors += 1
            logging.warning("Status check failed: %s", exc)
            if saw_remote_job and consecutive_connection_errors >= 3:
                if _qwen_container_exited_cleanly(processor):
                    logging.warning(
                        "Qwen API exited cleanly before a final status response for "
                        "remote job %s; attempting output recovery from container.",
                        remote_job_id,
                    )
                    return {
                        "status": "completed",
                        "recovered_from_clean_container_exit": True,
                    }

                stopped_error = _qwen_stopped_failure(processor)
                if stopped_error:
                    logging.error(
                        "%s for remote job %s",
                        stopped_error,
                        remote_job_id,
                    )
                    return {
                        "status": "failed",
                        "error": stopped_error,
                    }

        processor.shutdown_event.wait(poll_interval)

    return {"status": "failed", "error": "Timeout waiting for generation"}


class Qwen3TTSJobProcessor(BaseJobProcessor):
    """
    Job processor for Qwen3-TTS CustomVoice generation.
    Uses built-in speakers with optional emotional instructions.
    """

    API_PORT = 8000
    POLL_INTERVAL = 2
    MAX_WAIT_TIME = QWEN3_MAX_WAIT_TIME

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.api_base_url = f"http://localhost:{self.API_PORT}"

    def process(self):
        """Main processing method."""
        if not self.job:
            self._fail_job("Job object is None for Qwen3TTSJobProcessor. Cannot proceed.")
            return

        logging.info(f"Processing Qwen3-TTS job {self.job_id}")

        inputs = self.job.get("inputs") or {}
        text = self.positive_prompt or inputs.get("text", "")
        language = _normalize_qwen3_language(
            _input_alias(inputs, "language", "qwen3_language", default="Auto")
        )
        speaker = _input_alias(inputs, "speaker", "qwen3_speaker", default="vivian")
        instruct = inputs.get("instruct")

        if not text:
            self._fail_job("No text provided for TTS generation")
            return

        # IMPORTANT: Convert speaker to lowercase (model requirement)
        speaker = speaker.lower()

        if not self._wait_for_api():
            if self.is_cancelled():
                logging.info(f"Qwen3-TTS job {self.job_id} cancelled while waiting for API")
                return
            self._fail_job(f"Qwen3-TTS API did not become available for job {self.job_id}")
            return

        remote_job_id = self._submit_generation(text, language, speaker, instruct)
        if not remote_job_id:
            self._fail_job(f"Failed to submit generation for job {self.job_id}")
            return

        try:
            result = self._poll_for_completion(remote_job_id)

            if result.get("status") == "cancelled":
                logging.info(f"Qwen3-TTS job {self.job_id} cancelled during processing")
                return

            if result.get("status") != "completed":
                error_msg = result.get("error", "Unknown error")
                self._fail_job(f"Qwen3-TTS generation failed: {error_msg}")
                return

            local_path = self._download_output(remote_job_id)
            if not local_path:
                self._fail_job(f"Failed to download output for job {self.job_id}")
                return

            audio_storage_path = self.orchestrator_service.upload_audio_output(local_path, self.job_id)

            if audio_storage_path:
                duration = get_audio_duration(local_path)
                completion_metadata = dict(self.job.get("completion_metadata") or {})
                original_speaker = completion_metadata.get("speaker")
                completion_metadata.update({
                    "language": language,
                    "qwen3_speaker": speaker,
                    "model_speaker": speaker,
                    "has_instruct": bool(instruct),
                    "processor": "Qwen3TTSJobProcessor",
                })
                if original_speaker is not None:
                    completion_metadata["speaker"] = original_speaker

                self.orchestrator_service.update_job_status(
                    self.job_id,
                    "completed",
                    storage_path=audio_storage_path,
                    duration_seconds=duration,
                    completion_metadata=completion_metadata,
                )
                logging.info(f"Qwen3-TTS job {self.job_id} completed successfully")
            else:
                self._fail_job(f"Qwen3-TTS job {self.job_id} completed, but upload failed")

        except TokenExpiredError:
            raise
        except Exception as e:
            logging.error(f"Error processing Qwen3-TTS job {self.job_id}: {e}", exc_info=True)
            self._fail_job(f"Error processing job: {e}")
        finally:
            self._cleanup_remote_job(remote_job_id)
            self._cleanup_local_file()

    def _wait_for_api(self, timeout: int = 60) -> bool:
        """Wait for the Qwen3-TTS API to become available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_cancelled():
                return False
            try:
                response = requests.get(f"{self.api_base_url}/health", timeout=5)
                if response.status_code == 200:
                    logging.info("Qwen3-TTS API is ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

        logging.error(f"Qwen3-TTS API did not become available within {timeout}s")
        return False

    def _submit_generation(self, text: str, language: str, speaker: str, 
                           instruct: Optional[str] = None) -> Optional[str]:
        """Submit a generation request to the Qwen3-TTS API."""
        try:
            payload = {
                "text": text,
                "language": language,
                "speaker": speaker,
                "mode": "custom_voice",
            }
            if instruct:
                payload["instruct"] = instruct

            response = requests.post(f"{self.api_base_url}/generate", json=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()
                remote_job_id = data.get("job_id")
                logging.info(f"Generation job submitted: {remote_job_id}")
                return remote_job_id
            else:
                logging.error(f"Failed to submit generation: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to submit generation request: {e}")
            return None

    def _poll_for_completion(self, remote_job_id: str) -> Dict:
        """Poll the API for job completion."""
        return _poll_qwen_for_completion(
            self,
            self.api_base_url,
            remote_job_id,
            self.POLL_INTERVAL,
            self.MAX_WAIT_TIME,
        )

    def _download_output(self, remote_job_id: str) -> Optional[str]:
        """Download the generated audio file from the API."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            local_path = os.path.join(self.cache_dir, f"{self.job_id}.wav")

            response = requests.get(f"{self.api_base_url}/download/{remote_job_id}", timeout=60, stream=True)

            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logging.info(f"Downloaded output to {local_path}")
                return local_path
            else:
                logging.error(f"Failed to download output: {response.status_code}")
                return _copy_qwen_output_from_container(self, local_path)

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download output: {e}")
            return _copy_qwen_output_from_container(self, local_path)

    def _cleanup_remote_job(self, remote_job_id: str):
        """Clean up the remote job and its files."""
        if not remote_job_id:
            return
        try:
            requests.delete(f"{self.api_base_url}/job/{remote_job_id}", timeout=10)
        except requests.exceptions.RequestException:
            pass

    def _cleanup_local_file(self):
        """Clean up local temporary files."""
        local_path = os.path.join(self.cache_dir, f"{self.job_id}.wav")
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
                logging.info(f"Cleaned up temporary file: {local_path}")
            except OSError:
                pass


class Qwen3VoiceDesignJobProcessor(BaseJobProcessor):
    """
    Job processor for Qwen3-TTS VoiceDesign generation.
    Creates custom voices from natural language descriptions.
    """

    API_PORT = 8000
    POLL_INTERVAL = 2
    MAX_WAIT_TIME = QWEN3_MAX_WAIT_TIME

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.api_base_url = f"http://localhost:{self.API_PORT}"

    def process(self):
        """Main processing method."""
        if not self.job:
            self._fail_job("Job object is None for Qwen3VoiceDesignJobProcessor. Cannot proceed.")
            return

        logging.info(f"Processing Qwen3-TTS VoiceDesign job {self.job_id}")

        inputs = self.job.get("inputs") or {}
        text = self.positive_prompt or inputs.get("text", "")
        language = _normalize_qwen3_language(
            _input_alias(inputs, "language", "qwen3_language", default="Auto")
        )
        voice_design_instruct = inputs.get("voice_design_instruct", "")

        if not text:
            self._fail_job("No text provided for TTS generation")
            return

        if not voice_design_instruct:
            self._fail_job("No voice design instruction provided")
            return

        if not self._wait_for_api():
            if self.is_cancelled():
                logging.info(f"Qwen3-TTS VoiceDesign job {self.job_id} cancelled while waiting for API")
                return
            self._fail_job(f"Qwen3-TTS API did not become available for job {self.job_id}")
            return

        remote_job_id = self._submit_generation(text, language, voice_design_instruct)
        if not remote_job_id:
            self._fail_job(f"Failed to submit generation for job {self.job_id}")
            return

        try:
            result = self._poll_for_completion(remote_job_id)

            if result.get("status") == "cancelled":
                logging.info(f"Qwen3-TTS VoiceDesign job {self.job_id} cancelled during processing")
                return

            if result.get("status") != "completed":
                error_msg = result.get("error", "Unknown error")
                self._fail_job(f"Qwen3-TTS VoiceDesign generation failed: {error_msg}")
                return

            local_path = self._download_output(remote_job_id)
            if not local_path:
                self._fail_job(f"Failed to download output for job {self.job_id}")
                return

            audio_storage_path = self.orchestrator_service.upload_audio_output(local_path, self.job_id)

            if audio_storage_path:
                duration = get_audio_duration(local_path)
                completion_metadata = self.job.get("completion_metadata") or {}
                completion_metadata.update({
                    "language": language,
                    "voice_design": True,
                    "processor": "Qwen3VoiceDesignJobProcessor",
                })

                self.orchestrator_service.update_job_status(
                    self.job_id,
                    "completed",
                    storage_path=audio_storage_path,
                    duration_seconds=duration,
                    completion_metadata=completion_metadata,
                )
                logging.info(f"Qwen3-TTS VoiceDesign job {self.job_id} completed successfully")
            else:
                self._fail_job(f"Qwen3-TTS VoiceDesign job {self.job_id} completed, but upload failed")

        except TokenExpiredError:
            raise
        except Exception as e:
            logging.error(f"Error processing Qwen3-TTS VoiceDesign job {self.job_id}: {e}", exc_info=True)
            self._fail_job(f"Error processing job: {e}")
        finally:
            self._cleanup_remote_job(remote_job_id)
            self._cleanup_local_file()

    def _wait_for_api(self, timeout: int = 60) -> bool:
        """Wait for the Qwen3-TTS API to become available."""
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

    def _submit_generation(self, text: str, language: str, voice_design_instruct: str) -> Optional[str]:
        """Submit a generation request."""
        try:
            payload = {
                "text": text,
                "language": language,
                "mode": "voice_design",
                "voice_design_instruct": voice_design_instruct,
            }

            response = requests.post(f"{self.api_base_url}/generate", json=payload, timeout=30)

            if response.status_code == 200:
                return response.json().get("job_id")
            return None

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to submit generation request: {e}")
            return None

    def _poll_for_completion(self, remote_job_id: str) -> Dict:
        """Poll the API for job completion."""
        return _poll_qwen_for_completion(
            self,
            self.api_base_url,
            remote_job_id,
            self.POLL_INTERVAL,
            self.MAX_WAIT_TIME,
        )

    def _download_output(self, remote_job_id: str) -> Optional[str]:
        """Download the generated audio file."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            local_path = os.path.join(self.cache_dir, f"{self.job_id}.wav")
            response = requests.get(f"{self.api_base_url}/download/{remote_job_id}", timeout=60, stream=True)
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return local_path
            return _copy_qwen_output_from_container(self, local_path)
        except requests.exceptions.RequestException as exc:
            logging.error(f"Failed to download output: {exc}")
            return _copy_qwen_output_from_container(self, local_path)

    def _cleanup_remote_job(self, remote_job_id: str):
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


class Qwen3VoiceCloneJobProcessor(BaseJobProcessor):
    """
    Job processor for Qwen3-TTS voice cloning.
    Clones a voice from reference audio.
    """

    API_PORT = 8000
    POLL_INTERVAL = 2
    MAX_WAIT_TIME = QWEN3_MAX_WAIT_TIME

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.api_base_url = f"http://localhost:{self.API_PORT}"

    def process(self):
        """Main processing method."""
        if not self.job:
            self._fail_job("Job object is None for Qwen3VoiceCloneJobProcessor. Cannot proceed.")
            return

        logging.info(f"Processing Qwen3-TTS VoiceClone job {self.job_id}")

        inputs = self.job.get("inputs") or {}
        text = self.positive_prompt or inputs.get("text", "")
        language = _normalize_qwen3_language(
            _input_alias(inputs, "language", "qwen3_language", default="Auto")
        )
        voice_clone_urls = _as_url_list(inputs.get("voice_clone_urls", []))
        voice_clone_storage_path = _input_alias(
            inputs,
            "voice_clone_storage_path",
            "reference_audio",
            "reference_audio_storage_path",
        )
        ref_text = inputs.get("ref_text")

        if not text:
            self._fail_job("No text provided for TTS generation")
            return

        if not voice_clone_urls:
            self._fail_job("No voice clone reference audio provided")
            return

        clone_path = None
        remote_job_id = None
        try:
            # Download the voice reference audio
            for voice_clone_url in voice_clone_urls:
                clone_path = self.orchestrator_service.download_asset_by_url(
                    voice_clone_url, self.input_dir
                )
                if clone_path:
                    break

            if not clone_path and voice_clone_storage_path:
                bucket = self.job.get("bucket") or "projects_public"
                source_url = (
                    f"{SUPABASE_URL}/storage/v1/object/public/"
                    f"{bucket}/{voice_clone_storage_path}"
                )
                logging.info(
                    "Downloading voice clone reference from storage path fallback: %s",
                    voice_clone_storage_path,
                )
                clone_path = self.orchestrator_service.download_asset_by_url(
                    source_url, self.input_dir
                )

            if not clone_path:
                self._fail_job("Failed to download voice clone reference audio")
                return

            if not self._wait_for_api():
                if self.is_cancelled():
                    logging.info(f"Qwen3-TTS VoiceClone job {self.job_id} cancelled while waiting for API")
                    return
                self._fail_job(f"Qwen3-TTS API did not become available for job {self.job_id}")
                return

            remote_job_id = self._submit_voice_clone(text, language, clone_path, ref_text)
            if not remote_job_id:
                self._fail_job(f"Failed to submit voice clone for job {self.job_id}")
                return

            result = self._poll_for_completion(remote_job_id)

            if result.get("status") == "cancelled":
                logging.info(f"Qwen3-TTS VoiceClone job {self.job_id} cancelled during processing")
                return

            if result.get("status") != "completed":
                error_msg = result.get("error", "Unknown error")
                self._fail_job(f"Qwen3-TTS voice clone failed: {error_msg}")
                return

            local_path = self._download_output(remote_job_id)
            if not local_path:
                self._fail_job(f"Failed to download output for job {self.job_id}")
                return

            audio_storage_path = self.orchestrator_service.upload_audio_output(local_path, self.job_id)

            if audio_storage_path:
                duration = get_audio_duration(local_path)
                completion_metadata = self.job.get("completion_metadata") or {}
                completion_metadata.update({
                    "language": language,
                    "voice_clone": True,
                    "processor": "Qwen3VoiceCloneJobProcessor",
                })

                self.orchestrator_service.update_job_status(
                    self.job_id,
                    "completed",
                    storage_path=audio_storage_path,
                    duration_seconds=duration,
                    completion_metadata=completion_metadata,
                )
                logging.info(f"Qwen3-TTS VoiceClone job {self.job_id} completed successfully")
            else:
                self._fail_job(f"Qwen3-TTS VoiceClone job {self.job_id} completed, but upload failed")

        except TokenExpiredError:
            raise
        except Exception as e:
            logging.error(f"Error processing Qwen3-TTS VoiceClone job {self.job_id}: {e}", exc_info=True)
            self._fail_job(f"Error processing job: {e}")
        finally:
            self._cleanup_remote_job(remote_job_id)
            self._cleanup_local_file()
            # Clean up the reference audio file downloaded to host
            if clone_path and os.path.exists(clone_path):
                try:
                    os.remove(clone_path)
                    logging.info(f"Cleaned up local reference audio: {clone_path}")
                except OSError:
                    pass

    def _wait_for_api(self, timeout: int = 60) -> bool:
        """Wait for the Qwen3-TTS API to become available."""
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

    def _submit_voice_clone(self, text: str, language: str, ref_audio_path: str, 
                            ref_text: Optional[str] = None) -> Optional[str]:
        """Submit a voice clone request by uploading the reference audio."""
        try:
            # Prepare the form data
            data = {
                "text": text,
                "language": language,
                "x_vector_only": "true" if not ref_text else "false",
            }
            if ref_text:
                data["ref_text"] = ref_text

            # Prepare the file for upload
            with open(ref_audio_path, "rb") as f:
                files = {
                    "ref_audio": (os.path.basename(ref_audio_path), f, "audio/mpeg" if ref_audio_path.endswith(".mp3") else "audio/wav")
                }

                logging.info(f"Submitting voice clone with file upload: {ref_audio_path}")
                response = requests.post(
                    f"{self.api_base_url}/generate/voice-clone", 
                    data=data, 
                    files=files, 
                    timeout=30
                )

            if response.status_code == 200:
                data = response.json()
                remote_job_id = data.get("job_id")
                logging.info(f"Voice clone job submitted: {remote_job_id}")
                return remote_job_id
            else:
                logging.error(f"Failed to submit voice clone: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logging.error(f"Failed to submit voice clone request: {e}")
            return None

    def _poll_for_completion(self, remote_job_id: str) -> Dict:
        """Poll the API for job completion."""
        return _poll_qwen_for_completion(
            self,
            self.api_base_url,
            remote_job_id,
            self.POLL_INTERVAL,
            self.MAX_WAIT_TIME,
        )

    def _download_output(self, remote_job_id: str) -> Optional[str]:
        """Download the generated audio file."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            local_path = os.path.join(self.cache_dir, f"{self.job_id}.wav")
            response = requests.get(f"{self.api_base_url}/download/{remote_job_id}", timeout=60, stream=True)
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return local_path
            return _copy_qwen_output_from_container(self, local_path)
        except requests.exceptions.RequestException as exc:
            logging.error(f"Failed to download output: {exc}")
            return _copy_qwen_output_from_container(self, local_path)

    def _cleanup_remote_job(self, remote_job_id: str):
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
