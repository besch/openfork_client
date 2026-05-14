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

from services.processors.base import BaseJobProcessor
from services.orchestrator_service import TokenExpiredError
from utils.media_utils import get_audio_duration


class Qwen3TTSJobProcessor(BaseJobProcessor):
    """
    Job processor for Qwen3-TTS CustomVoice generation.
    Uses built-in speakers with optional emotional instructions.
    """

    API_PORT = 8000
    POLL_INTERVAL = 2
    MAX_WAIT_TIME = 300

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
        language = inputs.get("language", "Auto")
        speaker = inputs.get("speaker", "vivian")
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
                completion_metadata = self.job.get("completion_metadata") or {}
                completion_metadata.update({
                    "language": language,
                    "speaker": speaker,
                    "has_instruct": bool(instruct),
                    "processor": "Qwen3TTSJobProcessor",
                })

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
        start_time = time.time()

        while time.time() - start_time < self.MAX_WAIT_TIME:
            if self.is_cancelled():
                return {"status": "cancelled", "error": "Shutdown requested"}

            try:
                response = requests.get(f"{self.api_base_url}/status/{remote_job_id}", timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status")

                    if status == "completed":
                        logging.info(f"Remote job {remote_job_id} completed")
                        return data
                    elif status == "failed":
                        logging.error(f"Remote job {remote_job_id} failed: {data.get('error')}")
                        return data
                    else:
                        logging.debug(f"Remote job {remote_job_id} status: {status}")
                else:
                    logging.warning(f"Status check returned {response.status_code}")

            except requests.exceptions.RequestException as e:
                logging.warning(f"Status check failed: {e}")

            time.sleep(self.POLL_INTERVAL)

        return {"status": "failed", "error": "Timeout waiting for generation"}

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
                return None

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download output: {e}")
            return None

    def _cleanup_remote_job(self, remote_job_id: str):
        """Clean up the remote job and its files."""
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
    MAX_WAIT_TIME = 300

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
        language = inputs.get("language", "Auto")
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
        start_time = time.time()
        while time.time() - start_time < self.MAX_WAIT_TIME:
            if self.is_cancelled():
                return {"status": "cancelled"}
            try:
                response = requests.get(f"{self.api_base_url}/status/{remote_job_id}", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") in ["completed", "failed"]:
                        return data
            except requests.exceptions.RequestException:
                pass
            time.sleep(self.POLL_INTERVAL)
        return {"status": "failed", "error": "Timeout"}

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
            return None
        except requests.exceptions.RequestException:
            return None

    def _cleanup_remote_job(self, remote_job_id: str):
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
    MAX_WAIT_TIME = 300

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
        language = inputs.get("language", "Auto")
        voice_clone_urls = inputs.get("voice_clone_urls", [])
        ref_text = inputs.get("ref_text")

        if not text:
            self._fail_job("No text provided for TTS generation")
            return

        if not voice_clone_urls:
            self._fail_job("No voice clone reference audio provided")
            return

        clone_path = None
        try:
            # Download the voice reference audio
            clone_path = self.orchestrator_service.download_asset_by_url(
                voice_clone_urls[0], self.input_dir
            )
            if not clone_path:
                self._fail_job(f"Failed to download voice clone from {voice_clone_urls[0]}")
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
        start_time = time.time()
        while time.time() - start_time < self.MAX_WAIT_TIME:
            if self.is_cancelled():
                return {"status": "cancelled"}
            try:
                response = requests.get(f"{self.api_base_url}/status/{remote_job_id}", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") in ["completed", "failed"]:
                        return data
            except requests.exceptions.RequestException:
                pass
            time.sleep(self.POLL_INTERVAL)
        return {"status": "failed", "error": "Timeout"}

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
            return None
        except requests.exceptions.RequestException:
            return None

    def _cleanup_remote_job(self, remote_job_id: str):
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
