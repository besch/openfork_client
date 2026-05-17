"""
DramaBox job processors.

The DramaBox container exposes a small FastAPI wrapper with async job status,
download, and cleanup endpoints. These processors map OpenFork TTS params into
that wrapper.
"""

import logging
import os
import time
from typing import Dict, Optional

import requests

from config import SUPABASE_URL
from services.orchestrator_service import TokenExpiredError
from services.processors.base import BaseJobProcessor
from utils.media_utils import get_audio_duration


DRAMABOX_MAX_WAIT_TIME = int(os.environ.get("DRAMABOX_MAX_WAIT_TIME", "1800"))


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


class DramaboxBaseProcessor(BaseJobProcessor):
    """Shared DramaBox processor implementation."""

    API_PORT = 8000
    POLL_INTERVAL = 2
    MAX_WAIT_TIME = DRAMABOX_MAX_WAIT_TIME

    reference_required = False
    processor_name = "DramaboxBaseProcessor"

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.api_base_url = f"http://localhost:{self.API_PORT}"

    def process(self):
        if not self.job:
            self._fail_job("Job object is None for DramaBox processor. Cannot proceed.")
            return

        inputs = self.job.get("inputs") or {}
        prompt = self.positive_prompt or inputs.get("prompt") or inputs.get("text", "")
        if not prompt:
            self._fail_job("No prompt provided for DramaBox generation")
            return

        if not self._wait_for_api():
            if self.is_cancelled():
                logging.info("DramaBox job %s cancelled while waiting for API", self.job_id)
                return
            self._fail_job(f"DramaBox API did not become available for job {self.job_id}")
            return

        reference_path = None
        remote_job_id = None
        try:
            if self.reference_required:
                reference_path = self._download_reference(inputs)
                if not reference_path:
                    self._fail_job("No voice clone reference audio provided")
                    return

            payload = self._build_payload(prompt, inputs)
            remote_job_id = self._submit_generation(payload, reference_path)
            if not remote_job_id:
                self._fail_job(f"Failed to submit DramaBox generation for job {self.job_id}")
                return

            result = self._poll_for_completion(remote_job_id)
            if result.get("status") == "cancelled":
                return
            if result.get("status") != "completed":
                self._fail_job(f"DramaBox generation failed: {result.get('error', 'Unknown error')}")
                return

            local_path = self._download_output(remote_job_id)
            if not local_path:
                self._fail_job(f"Failed to download DramaBox output for job {self.job_id}")
                return

            audio_storage_path = self.orchestrator_service.upload_audio_output(
                local_path,
                self.job_id,
            )
            if not audio_storage_path:
                self._fail_job(f"DramaBox job {self.job_id} completed, but upload failed")
                return

            duration = get_audio_duration(local_path)
            completion_metadata = self.job.get("completion_metadata") or {}
            completion_metadata.update(
                {
                    "processor": self.processor_name,
                    "has_reference_voice": bool(reference_path),
                    "cfg_scale": payload["cfg_scale"],
                    "stg_scale": payload["stg_scale"],
                    "watermark": payload["watermark"],
                }
            )

            self.orchestrator_service.update_job_status(
                self.job_id,
                "completed",
                storage_path=audio_storage_path,
                duration_seconds=duration,
                completion_metadata=completion_metadata,
            )
            logging.info("DramaBox job %s completed successfully", self.job_id)
        except TokenExpiredError:
            raise
        except Exception as exc:
            logging.error("Error processing DramaBox job %s: %s", self.job_id, exc, exc_info=True)
            self._fail_job(f"Error processing DramaBox job: {exc}")
        finally:
            self._cleanup_remote_job(remote_job_id)
            cache_path = os.path.join(self.cache_dir, f"{self.job_id}.wav")
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
            if reference_path and os.path.exists(reference_path):
                try:
                    os.remove(reference_path)
                except OSError:
                    pass

    def _wait_for_api(self, timeout: int = 900) -> bool:
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

    def _build_payload(self, prompt: str, inputs: Dict) -> Dict:
        seed = _input_alias(inputs, "seed", default=42)
        return {
            "prompt": prompt,
            "cfg_scale": _as_float(
                _input_alias(inputs, "dramabox_cfg_scale", "cfg_scale"),
                2.5,
            ),
            "stg_scale": _as_float(
                _input_alias(inputs, "dramabox_stg_scale", "stg_scale"),
                1.5,
            ),
            "duration_multiplier": _as_float(
                _input_alias(inputs, "dramabox_duration_multiplier", "duration_multiplier"),
                1.1,
            ),
            "gen_duration": _as_float(
                _input_alias(inputs, "dramabox_gen_duration", "gen_duration"),
                0.0,
            ),
            "ref_duration": _as_float(
                _input_alias(inputs, "dramabox_ref_duration", "ref_duration"),
                10.0,
            ),
            "rescale_scale": str(
                _input_alias(inputs, "dramabox_rescale_scale", "rescale_scale", default="auto")
            ),
            "watermark": _as_bool(
                _input_alias(inputs, "dramabox_watermark", "watermark"),
                True,
            ),
            "seed": _as_int(seed, 42),
        }

    def _download_reference(self, inputs: Dict) -> Optional[str]:
        reference_urls = _as_url_list(inputs.get("voice_clone_urls", []))

        direct_reference = _input_alias(
            inputs,
            "reference_voice_url",
            "reference_audio_url",
        )
        if direct_reference:
            reference_urls.insert(0, str(direct_reference))

        storage_path = _input_alias(
            inputs,
            "voice_clone_storage_path",
            "reference_audio",
            "reference_audio_storage_path",
        )
        if storage_path:
            bucket = self.job.get("bucket") or "projects_public"
            reference_urls.append(
                f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{storage_path}"
            )

        for url in reference_urls:
            local_path = self.orchestrator_service.download_asset_by_url(
                url,
                self.input_dir,
            )
            if local_path:
                return local_path

        return None

    def _submit_generation(self, payload: Dict, reference_path: Optional[str]) -> Optional[str]:
        try:
            if reference_path:
                data = {
                    key: str(value).lower() if isinstance(value, bool) else str(value)
                    for key, value in payload.items()
                }
                with open(reference_path, "rb") as handle:
                    files = {
                        "voice_ref": (
                            os.path.basename(reference_path),
                            handle,
                            "audio/mpeg" if reference_path.lower().endswith(".mp3") else "audio/wav",
                        )
                    }
                    response = requests.post(
                        f"{self.api_base_url}/generate/voice-clone",
                        data=data,
                        files=files,
                        timeout=60,
                    )
            else:
                response = requests.post(
                    f"{self.api_base_url}/generate",
                    json=payload,
                    timeout=60,
                )

            if response.status_code == 200:
                return response.json().get("job_id")

            logging.error("DramaBox submit failed: %s - %s", response.status_code, response.text)
            return None
        except requests.exceptions.RequestException as exc:
            logging.error("Failed to submit DramaBox request: %s", exc)
            return None

    def _poll_for_completion(self, remote_job_id: str) -> Dict:
        start_time = time.time()
        while time.time() - start_time < self.MAX_WAIT_TIME:
            if self.is_cancelled():
                return {"status": "cancelled", "error": "Shutdown requested"}
            try:
                response = requests.get(
                    f"{self.api_base_url}/status/{remote_job_id}",
                    timeout=10,
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") in {"completed", "failed"}:
                        return data
                else:
                    logging.warning("DramaBox status check returned %s", response.status_code)
            except requests.exceptions.RequestException as exc:
                logging.warning("DramaBox status check failed: %s", exc)
            time.sleep(self.POLL_INTERVAL)

        return {"status": "failed", "error": "Timeout waiting for DramaBox generation"}

    def _download_output(self, remote_job_id: str) -> Optional[str]:
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            local_path = os.path.join(self.cache_dir, f"{self.job_id}.wav")
            response = requests.get(
                f"{self.api_base_url}/download/{remote_job_id}",
                timeout=60,
                stream=True,
            )
            if response.status_code != 200:
                logging.error("Failed to download DramaBox output: %s", response.status_code)
                return None

            with open(local_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    handle.write(chunk)
            return local_path
        except requests.exceptions.RequestException as exc:
            logging.error("Failed to download DramaBox output: %s", exc)
            return None

    def _cleanup_remote_job(self, remote_job_id: Optional[str]) -> None:
        if not remote_job_id:
            return
        try:
            requests.delete(f"{self.api_base_url}/job/{remote_job_id}", timeout=10)
        except requests.exceptions.RequestException:
            pass


class DramaboxTTSProcessor(DramaboxBaseProcessor):
    """Processor for DramaBox prompt-driven TTS."""

    processor_name = "DramaboxTTSProcessor"
    reference_required = False


class DramaboxVoiceCloneProcessor(DramaboxBaseProcessor):
    """Processor for DramaBox voice cloning."""

    processor_name = "DramaboxVoiceCloneProcessor"
    reference_required = True
