"""
ACE-Step-1.5 CLI Job Processor
"""

import os
import json
import time
import logging
import requests
from typing import Optional, Dict, Any
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from services.processors.base import BaseJobProcessor
from services.orchestrator_service import TokenExpiredError
from utils.media_utils import get_audio_duration

class AceStepCLIJobProcessor(BaseJobProcessor):
    """
    Job processor that communicates with ACE-Step 1.5 via its built-in REST API.

    Endpoints used:
      POST /release_task   — submit generation, returns task_id
      POST /query_result   — poll status (0=queued, 1=done, 2=failed)
      GET  /v1/audio       — download audio by URL or server-side file path
    """

    API_HOST = "127.0.0.1"
    API_PORT = 8000
    POLL_INTERVAL = 5
    MAX_WAIT_TIME = 900  # 15 minutes

    DEFAULT_MAX_DURATION = 60  # seconds

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        api_host = os.environ.get("ACESTEP_CLIENT_API_HOST", self.API_HOST)
        api_port = int(os.environ.get("ACESTEP_CLIENT_API_PORT", self.API_PORT))
        self.api_base_url = f"http://{api_host}:{api_port}"
        self.session = requests.Session()
        # Avoid corporate/system proxy settings intercepting localhost calls.
        self.session.trust_env = False

        workflow_config = client.config.get(self.workflow_type, {})
        self.max_audio_duration_seconds = workflow_config.get(
            "max_audio_duration_seconds",
            self.DEFAULT_MAX_DURATION
        )

    def process(self):
        """Main processing method."""
        if not self.job:
            self._fail_job("Job object is None. Cannot proceed.")
            return

        logging.info(f"Processing ACE-Step job {self.job_id}")

        lyrics, style_prompt, params = self._parse_prompt_and_params()
        inputs = self.job.get("inputs") or {}
        final_params = {**params, **inputs}
        local_path = None

        if not self._wait_for_api():
            self._fail_job("ACE-Step API did not become available")
            return

        task_id = self._submit_generation(lyrics, style_prompt, final_params)
        if not task_id:
            self._fail_job("Failed to submit generation")
            return

        try:
            result = self._poll_for_completion(task_id)

            if result.get("status") != "completed":
                error_msg = result.get("error", "Unknown error")
                self._fail_job(f"ACE-Step generation failed: {error_msg}")
                return

            local_path = self._download_output(task_id, result.get("download_target"))
            if not local_path:
                self._fail_job("Failed to download output")
                return

            audio_storage_path = self.orchestrator_service.upload_audio_output(local_path, self.job_id)

            if audio_storage_path:
                duration = get_audio_duration(local_path)
                completion_metadata = self.job.get("completion_metadata") or {}
                completion_metadata.update(
                    {
                        "lyrics_length": len(lyrics),
                        "style_prompt": style_prompt,
                        "has_lyrics": bool(lyrics.strip()),
                        "processor": "AceStepCLIJobProcessor",
                    }
                )

                self.orchestrator_service.update_job_status(
                    self.job_id,
                    "completed",
                    storage_path=audio_storage_path,
                    duration_seconds=duration,
                    completion_metadata=completion_metadata,
                )
                logging.info(f"ACE-Step job {self.job_id} completed successfully")
            else:
                self._fail_job("Job completed, but upload failed")

        except TokenExpiredError:
            raise
        except Exception as e:
            logging.error(f"Error processing ACE-Step job {self.job_id}: {e}", exc_info=True)
            self._fail_job(f"Error processing job: {e}")
        finally:
            if local_path and os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except OSError:
                    pass

    def _parse_prompt_and_params(self) -> tuple[str, str, Dict[str, Any]]:
        lyrics = ""
        style_prompt = ""
        params = {}

        try:
            if self.positive_prompt:
                parsed = json.loads(self.positive_prompt)
                lyrics = parsed.get("lyrics_or_edit_lyrics", "")
                style_prompt = parsed.get("style_prompt", "")
        except json.JSONDecodeError:
            style_prompt = self.positive_prompt or ""
            lyrics = ""

        return lyrics, style_prompt, params

    def _health_payload(self) -> Optional[Dict[str, Any]]:
        try:
            response = self.session.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code != 200:
                return None
            payload = response.json()
            return payload if isinstance(payload, dict) else None
        except (requests.exceptions.RequestException, ValueError):
            return None

    def _wait_for_api(self, timeout: int = 600) -> bool:
        start_time = time.monotonic()
        last_status_log = -30

        logging.info(f"Waiting for ACE-Step 1.5 API at {self.api_base_url} (timeout: {timeout}s)...")

        while time.monotonic() - start_time < timeout:
            if self.shutdown_event.is_set():
                return False

            elapsed = int(time.monotonic() - start_time)
            payload = self._health_payload()
            if payload:
                health = payload.get("data") if isinstance(payload.get("data"), dict) else payload
                if isinstance(health, dict):
                    models_initialized = health.get("models_initialized")
                    loaded_model = health.get("loaded_model")
                    llm_initialized = health.get("llm_initialized")
                    if health.get("status") == "ok" and (
                        models_initialized is None or models_initialized
                    ):
                        loaded_suffix = f", model={loaded_model}" if loaded_model else ""
                        llm_suffix = f", llm_initialized={llm_initialized}" if llm_initialized is not None else ""
                        logging.info(
                            f"ACE-Step API is ready after {elapsed}s{loaded_suffix}{llm_suffix}"
                        )
                        return True
                    if elapsed - last_status_log >= 30:
                        logging.info(
                            "ACE-Step API is reachable but still initializing "
                            f"(models_initialized={models_initialized}, loaded_model={loaded_model}, "
                            f"llm_initialized={llm_initialized})"
                        )
                        last_status_log = elapsed
                else:
                    logging.info(f"ACE-Step API is reachable after {elapsed}s")
                    return True
            elif elapsed - last_status_log >= 30:
                logging.info(f"ACE-Step API not reachable yet ({elapsed}/{timeout}s)")
                last_status_log = elapsed

            time.sleep(5)
        return False

    def _submit_generation(self, lyrics: str, style_prompt: str, params: Dict[str, Any]) -> Optional[str]:
        """Submit a generation task. Returns task_id or None on failure."""
        try:
            requested_duration = params.get("duration", params.get("duration_seconds", self.DEFAULT_MAX_DURATION))
            try:
                requested_duration = float(requested_duration)
            except (TypeError, ValueError):
                requested_duration = float(self.DEFAULT_MAX_DURATION)
            actual_duration = min(requested_duration, self.max_audio_duration_seconds)

            seed = params.get("seed")
            use_random_seed = seed in (None, "", -1, "-1")

            service_type = self.client.get_service_type_for_workflow(self.workflow_type)
            default_model_by_service = {
                "acestep-8gb": "acestep-v15-turbo",
                "acestep-16gb": "acestep-v15-xl-turbo",
            }
            requested_model = params.get("model") or default_model_by_service.get(service_type)

            payload = {
                "prompt": style_prompt,
                "lyrics": lyrics,
                "audio_duration": float(actual_duration),
                "audio_format": "wav",
                "inference_steps": 8,
                "use_random_seed": use_random_seed,
            }
            if not use_random_seed:
                try:
                    payload["seed"] = int(seed)
                except (TypeError, ValueError):
                    payload["seed"] = seed
            if requested_model:
                payload["model"] = requested_model

            response = self.session.post(f"{self.api_base_url}/release_task", json=payload, timeout=30)
            if response.status_code == 200:
                return response.json().get("data", {}).get("task_id")
            logging.error(f"release_task failed: {response.status_code} {response.text}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"release_task request error: {e}")
            return None

    def _parse_result_payload(self, result_str: Any) -> Optional[Dict[str, Any]]:
        if not result_str:
            return None
        try:
            result_data = json.loads(result_str) if isinstance(result_str, str) else result_str
        except (json.JSONDecodeError, TypeError):
            return None
        if isinstance(result_data, list):
            return result_data[0] if result_data and isinstance(result_data[0], dict) else None
        return result_data if isinstance(result_data, dict) else None

    def _extract_download_target(self, result_data: Optional[Dict[str, Any]]) -> Optional[str]:
        if not result_data:
            return None
        for key in ("file", "first_audio_path"):
            value = result_data.get(key)
            if value:
                return value
        for key in ("audio_paths", "raw_audio_paths"):
            value = result_data.get(key)
            if isinstance(value, list) and value:
                first_value = value[0]
                if isinstance(first_value, str) and first_value:
                    return first_value
        return None

    def _poll_for_completion(self, task_id: str) -> Dict:
        """
        Poll until done. Returns dict with:
          status: "completed" | "failed" | "cancelled"
          download_target: download URL or raw path to audio (on completed)
          error: error message (on failed)
        Status codes from API: 0=queued/running, 1=success, 2=failed
        """
        start_time = time.time()
        while time.time() - start_time < self.MAX_WAIT_TIME:
            if self.shutdown_event.is_set():
                return {"status": "cancelled", "error": "Shutdown requested"}
            try:
                response = self.session.post(
                    f"{self.api_base_url}/query_result",
                    json={"task_id_list": [task_id]},
                    timeout=10
                )
                if response.status_code == 200:
                    items = response.json().get("data", [])
                    if items:
                        item = items[0]
                        status_code = item.get("status")

                        if status_code == 1:  # Success
                            result_data = self._parse_result_payload(item.get("result"))
                            download_target = self._extract_download_target(result_data)
                            return {"status": "completed", "download_target": download_target}

                        elif status_code == 2:  # Failed
                            error = item.get("progress_text", "Generation failed")
                            result_data = self._parse_result_payload(item.get("result"))
                            if result_data:
                                error = result_data.get("error", error)
                            return {"status": "failed", "error": error}

                        # status_code == 0: still queued/running, keep polling

            except requests.exceptions.RequestException:
                pass
            time.sleep(self.POLL_INTERVAL)
        return {"status": "failed", "error": "Timeout"}

    def _infer_output_extension(self, download_target: str, content_type: Optional[str]) -> str:
        parsed = urlparse(download_target)
        query_path = parse_qs(parsed.query).get("path", [None])[0]
        path_candidate = unquote(query_path or parsed.path or download_target)
        _, ext = os.path.splitext(path_candidate)
        if ext:
            return ext

        content_type_map = {
            "audio/mpeg": ".mp3",
            "audio/mp3": ".mp3",
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
            "audio/flac": ".flac",
            "audio/ogg": ".ogg",
            "audio/opus": ".opus",
            "audio/aac": ".aac",
        }
        return content_type_map.get((content_type or "").split(";")[0].strip().lower(), ".wav")

    def _download_output(self, task_id: str, download_target: Optional[str]) -> Optional[str]:
        """Download audio from either a 1.5 download URL or a raw server-side file path."""
        if not download_target:
            logging.error(f"No server download target returned for task {task_id}")
            return None
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            if download_target.startswith(("http://", "https://")):
                download_url = download_target
                params = None
            elif download_target.startswith("/"):
                download_url = urljoin(f"{self.api_base_url}/", download_target.lstrip("/"))
                params = None
            elif download_target.startswith("v1/"):
                download_url = urljoin(f"{self.api_base_url}/", download_target)
                params = None
            else:
                download_url = f"{self.api_base_url}/v1/audio"
                params = {"path": download_target}

            response = self.session.get(download_url, params=params, timeout=60, stream=True)
            if response.status_code == 200:
                extension = self._infer_output_extension(download_target, response.headers.get("content-type"))
                local_path = os.path.join(self.cache_dir, f"{self.job_id}{extension}")
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return local_path
            logging.error(f"Audio download failed: {response.status_code} {response.text}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Audio download request error: {e}")
            return None
