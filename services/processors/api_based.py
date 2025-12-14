import os
import json
import logging
import requests
import time
from typing import Union, Dict
from services.orchestrator_service import TokenExpiredError
from services.docker_manager import docker_manager
from utils.media_utils import generate_thumbnail, get_video_duration
from services.processors.base import BaseJobProcessor


class APIBasedJobProcessor(BaseJobProcessor):
    """
    Base class for API-based job processors that communicate with Docker containers
    via HTTP REST API instead of ComfyUI workflows.
    
    This is designed for images like RunPod-style containers that expose endpoints
    like POST /run or POST /runsync for job submission.
    """
    
    @property
    def api_port(self) -> int:
        """Override this to specify the port the API runs on in the container."""
        return 8000
    
    @property
    def api_endpoint(self) -> str:
        """Override this to specify the API endpoint path (e.g., '/run', '/generate')."""
        return "/run"
    
    @property
    def api_timeout(self) -> int:
        """Timeout in seconds for API requests."""
        return 1800  # 30 minutes default
    
    def get_api_url(self) -> str:
        """Constructs the full API URL for the running container."""
        return f"http://localhost:{self.api_port}{self.api_endpoint}"
    
    def prepare_api_payload(self) -> Dict:
        """
        Prepare the payload to send to the API.
        Override this method in subclasses to customize the request payload.
        """
        raise NotImplementedError("Subclasses must implement prepare_api_payload()")
    
    def wait_for_api_ready(self, max_wait_seconds: int = 180) -> bool:
        """Wait for the API to become ready."""
        health_url = f"http://localhost:{self.api_port}/health"
        start_time = time.time()
        
        logging.info(f"Waiting for API to be ready at {health_url}...")
        
        while time.time() - start_time < max_wait_seconds:
            if self.shutdown_event.is_set():
                logging.warning("Shutdown event detected while waiting for API.")
                return False
            
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    logging.info("API is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(3)
        
        return False
    
    def submit_job_to_api(self, payload: Dict) -> Union[Dict, None]:
        """
        Submit a job to the API and wait for the response.
        Returns the response JSON or None on failure.
        """
        api_url = self.get_api_url()
        logging.info(f"Submitting job to API at {api_url}")
        logging.debug(f"Payload: {json.dumps(payload, indent=2)}")
        
        try:
            response = requests.post(
                api_url,
                json=payload,
                timeout=self.api_timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logging.error(f"API request timed out after {self.api_timeout} seconds")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}", exc_info=True)
            return None
    
    def extract_output_from_response(self, response: Dict) -> Union[str, None]:
        """
        Extract the output file path from the API response.
        Override this method to handle different response formats.
        
        Returns:
            Path to the output file in the container, or None if not found.
        """
        raise NotImplementedError("Subclasses must implement extract_output_from_response()")
    
    def copy_output_from_container(self, container_path: str) -> Union[str, None]:
        """
        Copy the output file from the container to the host.
        
        Args:
            container_path: Path to the file inside the container
            
        Returns:
            Local path to the copied file, or None on failure
        """
        if not self.client.active_service_type:
            logging.error("Cannot copy from container: no active service type is set.")
            return None
        
        # Extract filename from container path
        filename = os.path.basename(container_path)
        
        # Create temporary destination
        os.makedirs(self.cache_dir, exist_ok=True)
        temp_filename = f"{self.job_id}_{filename}"
        dest_on_host = os.path.join(self.cache_dir, temp_filename)
        
        try:
            docker_manager.copy_file_from_container(
                service_type=self.client.active_service_type,
                source_in_container=container_path,
                dest_on_host=dest_on_host,
                shutdown_event=self.shutdown_event
            )
            
            if os.path.exists(dest_on_host):
                logging.info(f"Successfully copied file to: {dest_on_host}")
                return dest_on_host
            else:
                raise RuntimeError("docker cp command finished but destination file does not exist.")
        except Exception as e:
            logging.error(f"Failed to copy file from container: {e}", exc_info=True)
            return None


class WAN22APITextToVideoJobProcessor(APIBasedJobProcessor):
    """
    WAN22 Text-to-Video processor using REST API instead of ComfyUI.
    
    Note: This is a template implementation. You'll need to adjust it based on
    the actual API specification of the antilopax/wan22 Docker image.
    """
    
    @property
    def api_port(self) -> int:
        # Adjust this based on the actual port exposed by antilopax/wan22
        return 8000
    
    @property
    def api_endpoint(self) -> str:
        # Common endpoints: /run, /runsync, /generate, /text-to-video
        # Adjust based on actual image documentation
        return "/run"
    
    def prepare_api_payload(self) -> Dict:
        """
        Prepare the payload for WAN22 text-to-video generation.
        
        Typical RunPod format:
        {
            "input": {
                "prompt": "...",
                "negative_prompt": "...",
                "width": 832,
                "height": 480,
                "num_frames":81,
                ...
            }
        }
        """
        inputs = self.job.get('inputs', {})
        aspect_ratio = inputs.get('aspect_ratio', '16:9')
        
        # Convert aspect ratio to dimensions
        width, height = 832, 480  # Default
        if aspect_ratio == '16:9':
            width, height = 832, 480
        elif aspect_ratio == '1:1':
            width, height = 512, 512
        elif aspect_ratio == '9:16':
            width, height = 480, 832
        
        payload = {
            "input": {
                "prompt": self.positive_prompt,
                "negative_prompt": self.negative_prompt or "",
                "width": width,
                "height": height,
                "num_frames": inputs.get('num_frames', 81),
                "seed": inputs.get('seed', -1)
            }
        }
        
        return payload
    
    def extract_output_from_response(self, response: Dict) -> Union[str, None]:
        """
        Extract the output video path from the API response.
        
        Common response formats:
        - {"output": "/output/video.mp4"}
        - {"output": {"video": "/output/video.mp4"}}
        - {"result": "/output/video.mp4"}
        """
        # Try different common response formats
        if "output" in response:
            output = response["output"]
            if isinstance(output, str):
                return output
            elif isinstance(output, dict) and "video" in output:
                return output["video"]
        
        if "result" in response:
            return response["result"]
        
        if "video_path" in response:
            return response["video_path"]
        
        logging.error(f"Could not extract output path from response: {response}")
        return None
    
    def process(self):
        """Main processing logic for API-based WAN22 text-to-video generation."""
        logging.info(f"Starting API-based WAN22 text-to-video job {self.job_id}")
        
        # Wait for API to be ready
        if not self.wait_for_api_ready():
            logging.error(f"API did not become ready for job {self.job_id}")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return
        
        # Prepare and submit job
        payload = self.prepare_api_payload()
        response = self.submit_job_to_api(payload)
        
        if not response:
            logging.error(f"Failed to get response from API for job {self.job_id}")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return
        
        # Extract output path
        container_output_path = self.extract_output_from_response(response)
        if not container_output_path:
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return
        
        # Copy output from container
        temp_host_path = self.copy_output_from_container(container_output_path)
        if not temp_host_path:
            logging.error(f"Failed to copy output from container for job {self.job_id}")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return
        
        try:
            # Upload video
            video_storage_path = self.orchestrator_service.upload_output(
                temp_host_path, self.job_id, 'video/mp4'
            )
            
            if video_storage_path:
                # Generate and upload thumbnail
                thumbnail_local_path = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
                thumbnail_storage_path = None
                
                if generate_thumbnail(temp_host_path, thumbnail_local_path, width=100):
                    thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(
                        thumbnail_local_path, self.job_id
                    )
                    os.remove(thumbnail_local_path)
                
                # Get video duration
                duration = get_video_duration(temp_host_path)
                
                # Update job status
                self.orchestrator_service.update_job_status(
                    self.job_id,
                    'completed',
                    storage_path=video_storage_path,
                    thumbnail_storage_path=thumbnail_storage_path,
                    duration_seconds=duration
                )
                logging.info(f"Job {self.job_id} completed successfully")
            else:
                logging.error(f"Video upload failed for job {self.job_id}")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
        finally:
            # Cleanup
            if os.path.exists(temp_host_path):
                os.remove(temp_host_path)
                logging.info(f"Cleaned up temporary file: {temp_host_path}")
