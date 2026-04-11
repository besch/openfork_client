"""
ACE-Step-1.5 CLI Job Processor
"""

import os
import json
import time
import logging
import requests
from typing import Optional, Dict, Any

from services.processors.base import BaseJobProcessor
from services.orchestrator_service import TokenExpiredError
from utils.media_utils import get_audio_duration

class AceStepCLIJobProcessor(BaseJobProcessor):
    """
    Job processor that communicates with ACE-Step 1.5 via its built-in REST API.

    Endpoints used:
      POST /release_task   — submit generation, returns task_id
      POST /query_result   — poll status (0=queued, 1=done, 2=failed)
      GET  /v1/audio       — download audio by server-side file path
    """

    API_PORT = 8000
    POLL_INTERVAL = 5
    MAX_WAIT_TIME = 900  # 15 minutes

    DEFAULT_MAX_DURATION = 60  # seconds

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        self.api_base_url = f"http://localhost:{self.API_PORT}"

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

            local_path = self._download_output(task_id, result.get("file_path"))
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
            local_path = os.path.join(self.cache_dir, f"{self.job_id}.wav")
            if os.path.exists(local_path):
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

    def _wait_for_api(self, timeout: int = 600) -> bool:
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            if self.shutdown_event.is_set():
                return False
            try:
                response = requests.get(f"{self.api_base_url}/health", timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(5)
        return False

    def _submit_generation(self, lyrics: str, style_prompt: str, params: Dict[str, Any]) -> Optional[str]:
        """Submit a generation task. Returns task_id or None on failure."""
        try:
            requested_duration = params.get("duration", params.get("duration_seconds", self.DEFAULT_MAX_DURATION))
            actual_duration = min(requested_duration, self.max_audio_duration_seconds)

            seed = params.get("seed")

            payload = {
                "prompt": style_prompt,
                "lyrics": lyrics,
                "seed": seed if seed is not None else -1,
                "audio_duration": float(actual_duration),
                "audio_format": "wav",
                "inference_steps": 8,
            }

            response = requests.post(f"{self.api_base_url}/release_task", json=payload, timeout=30)
            if response.status_code == 200:
                return response.json().get("data", {}).get("task_id")
            logging.error(f"release_task failed: {response.status_code} {response.text}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"release_task request error: {e}")
            return None

    def _poll_for_completion(self, task_id: str) -> Dict:
        """
        Poll until done. Returns dict with:
          status: "completed" | "failed" | "cancelled"
          file_path: server-side path to audio (on completed)
          error: error message (on failed)
        Status codes from API: 0=queued/running, 1=success, 2=failed
        """
        start_time = time.time()
        while time.time() - start_time < self.MAX_WAIT_TIME:
            if self.shutdown_event.is_set():
                return {"status": "cancelled", "error": "Shutdown requested"}
            try:
                response = requests.post(
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
                            file_path = None
                            result_str = item.get("result")
                            if result_str:
                                try:
                                    result_data = json.loads(result_str)
                                    file_path = result_data.get("file")
                                except (json.JSONDecodeError, TypeError):
                                    pass
                            return {"status": "completed", "file_path": file_path}

                        elif status_code == 2:  # Failed
                            error = item.get("progress_text", "Generation failed")
                            result_str = item.get("result")
                            if result_str:
                                try:
                                    error = json.loads(result_str).get("error", error)
                                except (json.JSONDecodeError, TypeError):
                                    pass
                            return {"status": "failed", "error": error}

                        # status_code == 0: still queued/running, keep polling

            except requests.exceptions.RequestException:
                pass
            time.sleep(self.POLL_INTERVAL)
        return {"status": "failed", "error": "Timeout"}

    def _download_output(self, task_id: str, file_path: Optional[str]) -> Optional[str]:
        """Download audio from the server's file path via GET /v1/audio?path=..."""
        if not file_path:
            logging.error(f"No server file path returned for task {task_id}")
            return None
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            local_path = os.path.join(self.cache_dir, f"{self.job_id}.wav")
            response = requests.get(
                f"{self.api_base_url}/v1/audio",
                params={"path": file_path},
                timeout=60,
                stream=True
            )
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return local_path
            logging.error(f"Audio download failed: {response.status_code}")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Audio download request error: {e}")
            return None
