"""
DiagDistill Job Processor

This processor communicates with the DiagDistill REST API server for ultra-fast
HunyuanVideo generation using Diagonal Distillation.
"""

import os
import logging
import requests
from typing import Optional

from services.processors.video.turbodiffusion import TurboDiffusionBaseProcessor
from utils.comfyui_workflow_utils import materialize_start_image


class DiagDistillJobProcessor(TurboDiffusionBaseProcessor):
    """
    Processor for DiagDistill video generation.
    Uses the same pattern as TurboDiffusion (REST-based).
    """
    
    SERVICE_NAME = "DiagDistill"

    def __init__(self, client, job, shutdown_event):
        super().__init__(client, job, shutdown_event)
        # Try to get the port from the service config
        wf_type = job.get("workflow_type")
        self.api_port = 8000
        try:
            if wf_type in client.config:
                service_id = client.config[wf_type].get("service_name")
                if service_id and service_id in client.services_config:
                    self.api_port = client.services_config[service_id].get("port", 8000)
        except Exception as e:
            logging.debug(f"Could not determine port from config: {e}")
        
        self.api_base_url = f"http://localhost:{self.api_port}"

    def _wait_for_api(self, timeout: int = 600) -> bool:
        """Wait for the DiagDistill API to become available."""
        import time
        import requests
        start_time = time.time()
        logging.info(f"Waiting for {self.SERVICE_NAME} API at {self.api_base_url} (timeout={timeout}s)...")
        while time.time() - start_time < timeout:
            if self.shutdown_event.is_set():
                return False
            try:
                response = requests.get(f"{self.api_base_url}/health", timeout=10)
                if response.status_code == 200:
                    logging.info(f"{self.SERVICE_NAME} API is ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(10)

        logging.error(f"{self.SERVICE_NAME} API did not become available within {timeout}s at {self.api_base_url}")
        return False

    def process(self):
        """Main processing method."""
        if not self.job:
            self._fail_job(f"Job object is None for {self.SERVICE_NAME}Processor. Cannot proceed.")
            return

        logging.info(f"Processing {self.SERVICE_NAME} job {self.job_id}")

        if not self._wait_for_api(timeout=600):
            # The message is already logged in _wait_for_api
            self._fail_job(f"{self.SERVICE_NAME} API timeout for job {self.job_id}")
            return

        inputs = self.job.get("inputs") or {}
        workflow_type = self.job.get("workflow_type")
        
        # Handle I2V start image if needed
        start_image_full_path = None
        if "image-to-video" in workflow_type:
            start_image_filename = materialize_start_image(self.job, self.input_dir)
            if not start_image_filename:
                self._fail_job(f"Failed to materialize start image for job {self.job_id}.")
                return
            start_image_full_path = os.path.join(self.input_dir, start_image_filename)

        # DiagDistill only supports T2V (Text-to-Video)
        # I2V is NOT supported by the DiagDistill repository
        if start_image_full_path:
            self._fail_job(f"DiagDistill does not support image-to-video (I2V). Only T2V is supported.")
            return
        
        # DiagDistill usually has many fixed parameters, but we'll pass standard ones
        resolution = inputs.get("resolution", "720p")
        seed = inputs.get("seed", 0)
        
        if start_image_full_path:
            remote_job_id = self._submit_diagdistill_i2v(start_image_full_path, resolution, seed)
        else:
            remote_job_id = self._submit_diagdistill_t2v(resolution, seed)

        if not remote_job_id:
            self._fail_job(f"Failed to submit DiagDistill generation for job {self.job_id}")
            return

        result = self._poll_for_completion(remote_job_id)
        if result.get("status") != "completed":
            error_msg = result.get("error", "Unknown error")
            self._fail_job(f"DiagDistill generation failed: {error_msg}")
            return

        local_path = self._download_output(remote_job_id)
        if not local_path:
            self._fail_job(f"Failed to download output for job {self.job_id}")
            return

        self._finalize_job(local_path, remote_job_id)

    def _submit_diagdistill_t2v(self, resolution: str, seed: int) -> Optional[str]:
        """Submit a DiagDistill T2V generation request."""
        try:
            payload = {
                "prompt": self.positive_prompt,
                "negative_prompt": self.negative_prompt,
                "resolution": resolution,
                "seed": seed,
            }
            response = requests.post(f"{self.api_base_url}/generate/t2v", json=payload, timeout=30)
            if response.status_code == 200:
                return response.json().get("job_id")
            logging.error(f"DiagDistill T2V submission failed: {response.text}")
            return None
        except Exception as e:
            logging.error(f"Error submitting DiagDistill T2V: {e}")
            return None

    def _submit_diagdistill_i2v(self, image_path: str, resolution: str, seed: int) -> Optional[str]:
        """Submit a DiagDistill I2V generation request."""
        try:
            with open(image_path, "rb") as f:
                files = {"image": (os.path.basename(image_path), f, "image/png")}
                data = {
                    "prompt": self.positive_prompt,
                    "negative_prompt": self.negative_prompt,
                    "resolution": resolution,
                    "seed": seed,
                }
                response = requests.post(f"{self.api_base_url}/generate/i2v", data=data, files=files, timeout=60)
            if response.status_code == 200:
                return response.json().get("job_id")
            logging.error(f"DiagDistill I2V submission failed: {response.text}")
            return None
        except Exception as e:
            logging.error(f"Error submitting DiagDistill I2V: {e}")
            return None
