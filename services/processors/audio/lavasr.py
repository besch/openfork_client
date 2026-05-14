import logging
from typing import Dict, Any, Optional
import os
import requests
import time

from services.processors.base import BaseJobProcessor
from utils.media_utils import get_audio_duration

logger = logging.getLogger(__name__)

class LavaSRJobProcessor(BaseJobProcessor):
    """
    Job processor for LavaSR speech restoration.
    Communicates with a REST API to perform audio enhancement.
    """

    API_PORT = 8000
    POLL_INTERVAL = 3
    MAX_WAIT_TIME = 300  # 5 minutes

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

            # Get the input audio URL (reusing input_video_url field)
            input_audio_url = self.job.get("input_video_url")
            if not input_audio_url:
                self._fail_job(f"LavaSR job {self.job_id} missing 'input_video_url' (audio input).")
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
            completed = False
            start_time = time.time()
            
            while time.time() - start_time < self.MAX_WAIT_TIME:
                if self.is_cancelled():
                    return

                try:
                    poll_resp = requests.get(f"{api_url}/status/{restoration_job_id}", timeout=10)
                    if poll_resp.status_code == 200:
                        status_data = poll_resp.json()
                        status = status_data.get("status")

                        if status == "completed":
                            completed = True
                            break
                        elif status == "failed":
                            self._fail_job(f"LavaSR restoration failed: {status_data.get('error')}")
                            return
                    else:
                        logger.error(f"Failed to poll LavaSR status: {poll_resp.status_code}")
                except requests.exceptions.RequestException as e:
                    logger.warning(f"LavaSR status check failed: {e}")
                
                time.sleep(self.POLL_INTERVAL)

            if not completed:
                self._fail_job("LavaSR restoration timed out")
                return

            # 6. Download restored audio
            output_filename = f"{self.job_id}_restored.wav"
            output_path = os.path.join(self.cache_dir, output_filename)
            
            download_resp = requests.get(f"{api_url}/download/{restoration_job_id}", stream=True, timeout=30)
            if download_resp.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in download_resp.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                self._fail_job(f"Failed to download restored audio from LavaSR: {download_resp.text}")
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

    def _wait_for_api(self, api_url: str, timeout: int = 300) -> bool:
        """Wait for the LavaSR API to become available."""
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

