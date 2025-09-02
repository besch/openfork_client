import requests
import logging
from typing import Union, Dict
from services.hardware_profiler import get_hardware_profile
import os

class OrchestratorService:
    def __init__(self, orchestrator_url: str, access_token: str):
        self.orchestrator_url = orchestrator_url
        self.access_token = access_token

    def _get_auth_headers(self) -> Dict[str, str]:
        """Returns the authorization headers for API requests."""
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }

    def get_next_job(self, provider_id: str) -> Union[Dict, None]:
        """Get the next available job for a provider."""
        try:
            response = requests.get(
                f"{self.orchestrator_url}/api/dgn/jobs/{provider_id}",
                headers=self._get_auth_headers()
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching next job: {e}")
            return None

    def download_asset(self, asset_id: str, cache_dir: str) -> Union[str, None]:
        """Download a specific asset from the orchestrator."""
        try:
            response = requests.get(
                f"{self.orchestrator_url}/api/dgn/assets/{asset_id}/download",
                headers=self._get_auth_headers()
            )
            response.raise_for_status()
            asset_info = response.json()
            # ... (rest of the logic is the same)
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading asset {asset_id}: {e}")
            return None
        return None # Add a return statement here

    def upload_output(self, file_path: str, job_id: str) -> Union[str, None]:
        """Upload the output file to the orchestrator."""
        try:
            with open(file_path, 'rb') as f:
                file_name = os.path.basename(file_path)
                # Note: requests will set the multipart/form-data header, so we don't set Content-Type here
                headers = {'Authorization': f'Bearer {self.access_token}'}
                files = {'file': (file_name, f.read(), 'video/mp4')}
                data = {'jobId': job_id}

                response = requests.post(
                    f"{self.orchestrator_url}/api/dgn/upload-output",
                    files=files, data=data, headers=headers
                )
                response.raise_for_status()
                # ... (rest of the logic is the same)
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not upload file {file_path}: {e}")
            return None
        return None # Add a return statement here

    def upload_thumbnail(self, file_path: str, job_id: str) -> Union[str, None]:
        """Upload the thumbnail file to the orchestrator."""
        try:
            with open(file_path, 'rb') as f:
                file_name = os.path.basename(file_path)
                headers = {'Authorization': f'Bearer {self.access_token}'}
                files = {'file': (file_name, f.read(), 'image/jpeg')}
                data = {'jobId': job_id}

                response = requests.post(
                    f"{self.orchestrator_url}/api/dgn/upload-thumbnail",
                    files=files, data=data, headers=headers
                )
                response.raise_for_status()
                # ... (rest of the logic is the same)
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not upload file {file_path}: {e}")
            return None
        return None # Add a return statement here

    def send_heartbeat(self, provider_id: str):
        """Sends a heartbeat to the orchestrator."""
        try:
            response = requests.post(
                f"{self.orchestrator_url}/api/dgn/heartbeat",
                json={"providerId": provider_id},
                headers=self._get_auth_headers()
            )
            response.raise_for_status()
            logging.info("Heartbeat sent successfully.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not send heartbeat: {e}")

    def update_job_status(self, job_id: str, status: str, output_path: Union[str, None] = None, thumbnail_path: Union[str, None] = None):
        """Update the status of a job."""
        try:
            payload = {"status": status}
            if output_path:
                payload["output_path"] = output_path
            if thumbnail_path:
                payload["thumbnail_path"] = thumbnail_path

            response = requests.put(
                f"{self.orchestrator_url}/api/dgn/job/{job_id}",
                json=payload,
                headers=self._get_auth_headers()
            )
            response.raise_for_status()
            logging.info(f"Job {job_id} status updated to {status}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not update job status: {e}")

    def update_provider_status(self, provider_id: str, status: str):
        """Update the status of a provider."""
        try:
            response = requests.put(
                f"{self.orchestrator_url}/api/dgn/provider-status/{provider_id}",
                json={"status": status},
                headers=self._get_auth_headers()
            )
            response.raise_for_status()
            logging.info(f"Provider {provider_id} status updated to {status}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not update provider status: {e}")

    def register_with_orchestrator(self) -> Union[str, None]:
        """Register the client with the orchestrator."""
        hardware_profile = get_hardware_profile()
        logging.info(f"Hardware Profile: {hardware_profile}")
        try:
            response = requests.post(
                f"{self.orchestrator_url}/api/dgn/register",
                json=hardware_profile,
                headers=self._get_auth_headers()
            )
            response.raise_for_status()
            logging.info("Successfully registered with the Orchestrator.")
            return response.json().get('provider_id')
        except requests.exceptions.RequestException as e:
            logging.error(f"Error registering with the Orchestrator: {e.response.text if e.response else e}")
            return None

    def deregister_from_orchestrator(self, provider_id: str) -> None:
        """Remove provider row when client stops."""
        try:
            response = requests.delete(
                f"{self.orchestrator_url}/api/dgn/register",
                params={"providerId": provider_id},
                headers=self._get_auth_headers()
            )
            response.raise_for_status()
            logging.info("Provider deregistered.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error deregistering provider: {e}")
