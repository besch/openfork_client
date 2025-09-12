import requests
import logging
import base64
import json
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

    def download_asset_by_url(self, asset_url: str, download_dir: str) -> Union[str, None]:
        """Download an asset from a given URL."""
        try:
            response = requests.get(asset_url, stream=True)
            response.raise_for_status()
            
            file_name = asset_url.split('/')[-1].split('?')[0]

            # If the file name has no extension, add .mp4
            if '.' not in file_name:
                file_name += '.mp4'

            file_path = os.path.join(download_dir, file_name)

            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logging.info(f"Asset downloaded to {file_path}")
            return file_path
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading asset from {asset_url}: {e}")
            return None

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
                response_data = response.json()
                storage_path = response_data.get('storagePath')
                if not storage_path:
                    logging.error(f"Upload response missing 'storagePath': {response_data}")
                    return None
                return storage_path
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not upload file {file_path}: {e}")
            return None
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from upload response: {response.text}")
            return None

    def upload_audio_output(self, file_path: str, job_id: str) -> Union[str, None]:
        """Upload the audio output file to the orchestrator."""
        try:
            with open(file_path, 'rb') as f:
                file_name = os.path.basename(file_path)
                headers = {'Authorization': f'Bearer {self.access_token}'}
                # Assuming the generated foley is in mp3 format
                files = {'file': (file_name, f.read(), 'audio/mpeg')}
                data = {'jobId': job_id}

                response = requests.post(
                    f"{self.orchestrator_url}/api/dgn/upload-output",
                    files=files, data=data, headers=headers
                )
                response.raise_for_status()
                response_data = response.json()
                storage_path = response_data.get('storagePath')
                if not storage_path:
                    logging.error(f"Upload response missing 'storagePath': {response_data}")
                    return None
                return storage_path
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not upload file {file_path}: {e}")
            return None
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from upload response: {response.text}")
            return None

    def upload_image_output(self, file_path: str, job_id: str) -> Union[str, None]:
        """Upload the image output file to the orchestrator."""
        try:
            with open(file_path, 'rb') as f:
                file_name = os.path.basename(file_path)
                headers = {'Authorization': f'Bearer {self.access_token}'}
                files = {'file': (file_name, f.read(), 'image/png')}
                data = {'jobId': job_id}

                response = requests.post(
                    f"{self.orchestrator_url}/api/dgn/upload-output",
                    files=files, data=data, headers=headers
                )
                response.raise_for_status()
                response_data = response.json()
                storage_path = response_data.get('storagePath')
                if not storage_path:
                    logging.error(f"Upload response missing 'storagePath': {response_data}")
                    return None
                return storage_path
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not upload file {file_path}: {e}")
            return None
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from upload response: {response.text}")
            return None

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
                response_data = response.json()
                storage_path = response_data.get('storagePath')
                if not storage_path:
                    logging.error(f"Upload response missing 'storagePath': {response_data}")
                    return None
                return storage_path
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not upload file {file_path}: {e}")
            return None
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from upload response: {response.text}")
            return None

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

    def update_job_status(self, job_id: str, status: str, output_path: Union[str, None] = None, thumbnail_path: Union[str, None] = None, duration_seconds: float = None, completion_metadata: Dict = None):
        """Update the status of a job."""
        try:
            payload = {"status": status}
            if output_path:
                payload["storage_path"] = output_path
            if thumbnail_path:
                payload["thumbnail_storage_path"] = thumbnail_path
            if duration_seconds:
                payload["duration_seconds"] = duration_seconds
            if completion_metadata:
                payload["completion_metadata"] = completion_metadata

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

    def _get_user_id_from_token(self) -> Union[str, None]:
        """Decodes the user ID (sub) from the JWT access token."""
        try:
            _, payload_b64, _ = self.access_token.split('.')
            payload_b64 += '=' * (-len(payload_b64) % 4)
            payload_json = base64.urlsafe_b64decode(payload_b64)
            payload = json.loads(payload_json)
            return payload.get('sub')
        except Exception as e:
            logging.error(f"Error decoding JWT to get user ID: {e}")
            return None

    def register_with_orchestrator(self, service_type: str = 'default') -> Union[str, None]:
        """Register the client with the orchestrator."""
        hardware_profile = get_hardware_profile()
        
        user_id = self._get_user_id_from_token()
        if not user_id:
            logging.error("Could not extract user_id from token. Cannot register.")
            return None
        
        payload = {**hardware_profile, "user_id": user_id, "service_type": service_type}

        logging.info(f"Registering with profile: {payload}")
        try:
            response = requests.post(
                f"{self.orchestrator_url}/api/dgn/register",
                json=payload,
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
        logging.info(f"OrchestratorService: Attempting to deregister provider {provider_id}.")
        try:
            response = requests.delete(
                f"{self.orchestrator_url}/api/dgn/register",
                params={"providerId": provider_id},
                headers=self._get_auth_headers()
            )
            response.raise_for_status()
            logging.info(f"OrchestratorService: Provider {provider_id} deregistered successfully. Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logging.error(f"OrchestratorService: Error deregistering provider {provider_id}: {e}")
            if e.response:
                logging.error(f"OrchestratorService: Response content: {e.response.text}")