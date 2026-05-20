import logging
from typing import Dict, Any, Optional
import os
import requests
import time

from config import SUPABASE_URL
from services.processors.base import BaseJobProcessor
from services.processors.rest_recovery import (
    poll_rest_job_with_clean_exit,
    recover_output_from_clean_container_exit,
)
from utils.media_utils import get_audio_duration

logger = logging.getLogger(__name__)

class LavaSRJobProcessor(BaseJobProcessor):
    """
    Job processor for LavaSR speech restoration.
    Communicates with a REST API to perform audio enhancement.
    """

    API_PORT = 8000
    POLL_INTERVAL = 3
    MAX_WAIT_TIME = int(os.environ.get("LAVASR_MAX_WAIT_TIME", "1200"))
    API_WAIT_TIMEOUT = int(os.environ.get("LAVASR_API_WAIT_TIMEOUT", "600"))

    def process(self):
        """
        Processes a speech restoration job.
        1. Downloads the input audio clip.
        2. Sends it to the LavaSR REST API.
        3. Polls for completion.
        4. Downloads the restored audio.
        5. Uploads the restored audio to Supabase.
        6. Updates the job status.
        """
        audio_path = None
        output_path = None
        try:
            if self.job is None:
                self._fail_job("Job object is None for LavaSRJobProcessor. Cannot proceed.")
                return

            inputs = self.job.get("inputs") or {}

            # Get the input audio URL. Historically this reused input_video_url,
            # but website callers may provide audio-specific aliases.
            input_audio_url = (
                self.job.get("input_video_url")
                or inputs.get("input_video_url")
                or inputs.get("input_audio_url")
                or inputs.get("audio_url")
                or inputs.get("source_url")
            )
            input_storage_path = (
                self.job.get("input_storage_path")
                or inputs.get("input_storage_path")
                or inputs.get("input_audio_storage_path")
                or inputs.get("audio_storage_path")
                or inputs.get("source_storage_path")
            )
            if not input_audio_url and input_storage_path:
                bucket = self.job.get("bucket") or inputs.get("bucket") or "projects_public"
                input_audio_url = (
                    f"{SUPABASE_URL}/storage/v1/object/public/"
                    f"{bucket}/{input_storage_path}"
                )
                logger.info(
                    "Using LavaSR input storage path fallback for job %s: %s",
                    self.job_id,
                    input_storage_path,
                )

            if not input_audio_url:
                self._fail_job(
                    f"LavaSR job {self.job_id} missing audio input URL or storage path."
                )
                return

            # 1. Download input audio
            audio_path = self.orchestrator_service.download_asset_by_url(input_audio_url, self.input_dir)
            if not audio_path:
                self._fail_job(f"Failed to download input audio for LavaSR job {self.job_id}.")
                return

            # 2. Get API URL
            api_url = f"http://localhost:{self.API_PORT}"

            # 3. Wait for the LavaSR API to be ready
            if not self._wait_for_api(api_url):
                self._fail_job(f"LavaSR API did not become available for job {self.job_id}")
                return

            # 4. Start restoration job
            logger.info(f"Starting LavaSR restoration for job {self.job_id}")
            
            with open(audio_path, 'rb') as f:
                filename = os.path.basename(audio_path)
                files = {'audio': (filename, f, 'audio/wav')}
                response = requests.post(
                    f"{api_url}/enhance",
                    files=files,
                    timeout=30
                )

            if response.status_code != 200:
                self._fail_job(f"LavaSR API failed to start job: {response.text}")
                return

            restoration_job_id = response.json().get("job_id")
            if not restoration_job_id:
                self._fail_job("LavaSR API did not return a job_id")
                return

            # 5. Poll for completion
            logger.info(f"Polling LavaSR job {restoration_job_id} for job {self.job_id}")
            result = poll_rest_job_with_clean_exit(
                self,
                api_url,
                restoration_job_id,
                poll_interval=self.POLL_INTERVAL,
                max_wait_time=self.MAX_WAIT_TIME,
                service_label="LavaSR",
            )
            if result.get("status") == "cancelled":
                return
            if result.get("status") != "completed":
                self._fail_job(f"LavaSR restoration failed: {result.get('error', 'Unknown error')}")
                return

            # 6. Download restored audio
            output_filename = f"{self.job_id}_restored.wav"
            output_path = os.path.join(self.cache_dir, output_filename)

            try:
                download_resp = requests.get(
                    f"{api_url}/download/{restoration_job_id}",
                    stream=True,
                    timeout=120,
                )
                if download_resp.status_code == 200:
                    with open(output_path, 'wb') as f:
                        for chunk in download_resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    recovered_path = recover_output_from_clean_container_exit(
                        self,
                        output_path,
                        container_output_path=f"/app/output/{restoration_job_id}_restored.wav",
                        extensions=(".wav",),
                        prefer_name=restoration_job_id,
                    )
                    if not recovered_path:
                        self._fail_job(f"Failed to download restored audio from LavaSR: {download_resp.text}")
                        return
            except requests.exceptions.RequestException as e:
                recovered_path = recover_output_from_clean_container_exit(
                    self,
                    output_path,
                    container_output_path=f"/app/output/{restoration_job_id}_restored.wav",
                    extensions=(".wav",),
                    prefer_name=restoration_job_id,
                )
                if not recovered_path:
                    self._fail_job(f"Failed to download restored audio from LavaSR: {e}")
                    return

            # 7. Upload to Supabase
            if os.path.exists(output_path):
                # Clean up API job
                try:
                    requests.delete(f"{api_url}/job/{restoration_job_id}", timeout=5)
                except:
                    pass
                
                # Upload and complete
                audio_storage_path = self.orchestrator_service.upload_audio_output(output_path, self.job_id)
                if audio_storage_path:
                    duration_seconds = get_audio_duration(output_path)
                    completion_metadata = self.job.get("completion_metadata") or {}
                    completion_metadata.update({
                        "processor": "LavaSRJobProcessor",
                    })

                    self.orchestrator_service.update_job_status(
                        self.job_id,
                        "completed",
                        storage_path=audio_storage_path,
                        duration_seconds=duration_seconds,
                        completion_metadata=completion_metadata,
                    )
                    logger.info(f"LavaSR job {self.job_id} completed successfully")
                else:
                    self._fail_job("Failed to upload restored audio to storage.")
            else:
                self._fail_job("Restored audio file not found on disk.")

        except Exception as e:
            logger.exception(f"Unexpected error in LavaSRJobProcessor for job {self.job_id}: {e}")
            self._fail_job(str(e))
        finally:
            # Clean up local input
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
            # Clean up local output
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass

    def _wait_for_api(self, api_url: str, timeout: Optional[int] = None) -> bool:
        """Wait for the LavaSR API to become available."""
        timeout = timeout or self.API_WAIT_TIMEOUT
        logger.info(f"Waiting for LavaSR API at {api_url}...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_cancelled():
                return False
            try:
                response = requests.get(f"{api_url}/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy" and data.get("model_loaded"):
                        logger.info("LavaSR API is ready and model is loaded")
                        return True
                    elif data.get("status") == "error":
                        error_msg = data.get("model_error", "Unknown model loading error")
                        logger.error(f"LavaSR API reported error: {error_msg}")
                        return False
                    else:
                        logger.info(f"LavaSR API is up but not ready yet (loading...): {data}")
            except requests.exceptions.RequestException:
                pass
            time.sleep(5)

        logger.error(f"LavaSR API did not become available within {timeout}s")
        return False

