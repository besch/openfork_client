import requests
import logging
import base64
import json
import time
import threading
from typing import Union, Dict
from services.hardware_profiler import get_hardware_profile
import os

# Custom exception for handling expired tokens
class TokenExpiredError(Exception):
    """Raised when the API returns a 401 Unauthorized error."""
    pass

class OrchestratorService:
    def __init__(self, orchestrator_url: str, access_token: str, refresh_token: str):
        self.orchestrator_url = orchestrator_url
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.token_update_lock = threading.Lock()

    def update_tokens(self, access_token: str, refresh_token: str):
        """Thread-safe method to update auth tokens."""
        with self.token_update_lock:
            self.access_token = access_token
            self.refresh_token = refresh_token
            logging.info("OrchestratorService tokens have been updated by the main process.")

    def _make_request(self, method, url, auth_required=True, **kwargs) -> requests.Response:
        """
        Makes an HTTP request, raising a custom error on 401 Unauthorized.
        """
        # Add a default timeout to all requests to prevent indefinite hangs
        kwargs.setdefault('timeout', 30)

        request_headers = kwargs.pop("headers", {}).copy()
        
        if auth_required:
            # Use the lock to ensure the token isn't being updated by another thread while we read it
            with self.token_update_lock:
                request_headers['Authorization'] = f'Bearer {self.access_token}'

        if 'Content-Type' not in request_headers and 'files' not in kwargs and 'data' not in kwargs and 'json' not in kwargs:
             request_headers['Content-Type'] = 'application/json'
        
        response = requests.request(method, url, headers=request_headers, **kwargs)

        if response.status_code == 401 and auth_required:
            logging.warning("Received 401 Unauthorized. Notifying main process to refresh token.")
            raise TokenExpiredError("Access token has expired.")
        
        return response

    def resolve_targets(self, targets: list[str], target_type: str = 'project') -> list[str]:
        """Resolves a list of target strings to UUIDs."""
        if not targets:
            return []
        
        try:
            response = self._make_request(
                'post',
                f"{self.orchestrator_url}/api/dgn/resolve-targets",
                json={"targets": targets, "type": target_type}
            )
            response.raise_for_status()
            data = response.json()
            
            resolved_ids = [item['id'] for item in data.get('resolved', []) if item['id']]
            
            unresolved_targets = [item['target'] for item in data.get('resolved', []) if not item['id']]
            if unresolved_targets:
                logging.warning(f"Some targets could not be resolved and will be ignored: {unresolved_targets}")

            return resolved_ids
        except requests.exceptions.RequestException as e:
            logging.error(f"Error resolving targets: {e}")
            if e.response:
                logging.error(f"Response content: {e.response.text}")
            return []
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from resolve_targets response: {response.text}")
            return []

    def get_next_job(self, provider_id: str, accept_policy: str, allowed_ids: list[str]) -> Union[Dict, None]:
        """Get the next available job for a provider based on their policy."""
        try:
            base_url = f"{self.orchestrator_url}/api/dgn/jobs/{provider_id}"
            params = {
                "ts": int(time.time()),
                "acceptPolicy": accept_policy,
            }

            if accept_policy == 'mine':
                user_id = self._get_user_id_from_token()
                if not user_id:
                    logging.error("Could not get user ID for 'own' policy. Aborting job fetch.")
                    return None
                params["userId"] = user_id
            
            if (accept_policy == 'project' or accept_policy == 'users') and allowed_ids:
                params["allowedIds"] = ",".join(allowed_ids)

            response = self._make_request('get', base_url, params=params)
            response.raise_for_status()
            if not response.content:
                return None
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching next job: {e}")
            return None
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from get_next_job response: {response.text}")
            return None

    def get_job(self, job_id: str) -> Union[Dict, None]:
        """Get a job by its ID. Returns a fabricated 'cancelled' status if the job is not found (404)."""
        try:
            # This endpoint needs to be created in the Next.js app: GET /api/dgn/job/{job_id}
            url = f"{self.orchestrator_url}/api/dgn/job/{job_id}"
            response = self._make_request('get', url)

            if response.status_code == 404:
                logging.warning(f"Job {job_id} not found (404). Assuming it was cancelled and deleted.")
                return {'status': 'cancelled'}

            response.raise_for_status()
            if not response.content:
                return None
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching job {job_id}: {e}")
            return None
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from get_job response: {response.text}")
            return None

    def download_asset_by_url(self, asset_url: str, download_dir: str) -> Union[str, None]:
        """Download an asset from a given URL."""
        try:
            response = requests.get(asset_url, stream=True)
            response.raise_for_status()
            
            file_name = asset_url.split('/')[-1].split('?')[0]

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

    def _get_signed_upload_url(self, job_id: str, file_name: str) -> Union[Dict, None]:
        """Get a presigned URL for uploading a file directly to storage."""
        try:
            response = self._make_request(
                'post',
                f"{self.orchestrator_url}/api/dgn/upload-url",
                json={"jobId": job_id, "fileName": file_name}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not get signed upload URL: {e}")
            if e.response:
                logging.error(f"Response content: {e.response.text}")
            return None
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from get_signed_upload_url response: {response.text}")
            return None

    def upload_output(self, file_path: str, job_id: str, content_type: str) -> Union[str, None]:
        """Uploads a file directly to storage using a presigned URL."""
        file_name = os.path.basename(file_path)
        
        upload_info = self._get_signed_upload_url(job_id, file_name)
        if not upload_info or not upload_info.get('success'):
            logging.error(f"Failed to get a presigned URL for job {job_id}")
            return None
        
        upload_url = upload_info['uploadUrl']
        storage_path = upload_info['storagePath']

        try:
            with open(file_path, 'rb') as f:
                response = self._make_request(
                    'put',
                    upload_url,
                    auth_required=False,
                    data=f,
                    headers={'Content-Type': content_type}
                )
                response.raise_for_status()
            
            logging.info(f"Successfully uploaded {file_name} for job {job_id}. Storage path: {storage_path}")
            return storage_path
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not upload file to presigned URL: {e}")
            if e.response:
                logging.error(f"Response content: {e.response.text}")
            return None

    def upload_audio_output(self, file_path: str, job_id: str) -> Union[str, None]:
        """Upload the audio output file to the orchestrator."""
        return self.upload_output(file_path, job_id, 'audio/flac')

    def upload_image_output(self, file_path: str, job_id: str) -> Union[str, None]:
        """Upload the image output file to the orchestrator."""
        return self.upload_output(file_path, job_id, 'image/png')

    def upload_thumbnail(self, file_path: str, job_id: str) -> Union[str, None]:
        """Upload the thumbnail file to the orchestrator."""
        return self.upload_output(file_path, job_id, 'image/jpeg')

    def send_heartbeat(self, provider_id: str):
        """Sends a heartbeat to the orchestrator."""
        try:
            response = self._make_request(
                'post',
                f"{self.orchestrator_url}/api/dgn/heartbeat",
                json={"providerId": provider_id}
            )
            response.raise_for_status()
            logging.info("Heartbeat sent successfully.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not send heartbeat: {e}")

    def update_job_status(self, job_id: str, status: str, storage_path: Union[str, None] = None, thumbnail_storage_path: Union[str, None] = None, duration_seconds: float = None, completion_metadata: Dict = None, prompt: Union[str, None] = None):
        """Update the status of a job."""
        try:
            payload = {"status": status}
            if storage_path:
                payload["storage_path"] = storage_path
            if thumbnail_storage_path:
                payload["thumbnail_storage_path"] = thumbnail_storage_path
            if duration_seconds:
                payload["duration_seconds"] = duration_seconds
            if completion_metadata:
                payload["completion_metadata"] = completion_metadata
            if prompt:
                payload["prompt"] = prompt

            response = self._make_request(
                'put',
                f"{self.orchestrator_url}/api/dgn/job/{job_id}",
                json=payload
            )
            response.raise_for_status()
            logging.info(f"Job {job_id} status updated to {status}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not update job status: {e}")

    def update_provider_status(self, provider_id: str, status: str):
        """Update the status of a provider."""
        try:
            response = self._make_request(
                'put',
                f"{self.orchestrator_url}/api/dgn/provider-status/{provider_id}",
                json={"status": status}
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

    def register_with_orchestrator(self, service_type: str) -> Union[str, None]:
        """Register the client with the orchestrator."""
        hardware_profile = get_hardware_profile()
        
        user_id = self._get_user_id_from_token()
        if not user_id:
            logging.error("Could not extract user_id from token. Cannot register.")
            return None
        
        payload = {**hardware_profile, "user_id": user_id, "service_type": service_type}

        logging.info(f"Registering with profile: {payload}")
        try:
            response = self._make_request(
                'post',
                f"{self.orchestrator_url}/api/dgn/register",
                json=payload
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
            response = self._make_request(
                'delete',
                f"{self.orchestrator_url}/api/dgn/register",
                params={"providerId": provider_id}
            )
            response.raise_for_status()
            logging.info(f"OrchestratorService: Provider {provider_id} deregistered successfully. Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logging.error(f"OrchestratorService: Error deregistering provider {provider_id}: {e}")
            if e.response:
                logging.error(f"OrchestratorService: Response content: {e.response.text}")

    def reset_interrupted_job(self, job_id: str):
        """Resets a job's status to 'pending' and clears its provider via a specific API endpoint."""
        try:
            logging.info(f"Requesting reset for job {job_id}")
            response = self._make_request(
                'put',
                f"{self.orchestrator_url}/api/dgn/job/{job_id}?action=reset"
            )
            response.raise_for_status()
            logging.info(f"Job {job_id} status reset successfully via API.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not reset job status for {job_id}: {e}")
            if e.response:
                logging.error(f"Reset API response: {e.response.text}")
            raise

    def get_workflow(self, workflow_name: str, cache_dir: str) -> Union[Dict, None]:
        """
        Fetches a workflow definition, using a local cache to avoid re-downloads.
        The workflow is assumed to be a JSON file.
        """
        workflow_cache_dir = os.path.join(cache_dir, 'workflows')
        os.makedirs(workflow_cache_dir, exist_ok=True)
        cached_workflow_path = os.path.join(workflow_cache_dir, workflow_name)

        # For now, we prioritize the cached version if it exists.
        # A more advanced implementation could involve version checking.
        if os.path.exists(cached_workflow_path):
            logging.info(f"Loading workflow '{workflow_name}' from cache.")
            try:
                with open(cached_workflow_path, 'r') as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                logging.error(f"Error reading cached workflow {workflow_name}: {e}. Will attempt to re-download.")

        # If not in cache or reading failed, download it
        logging.info(f"Downloading workflow '{workflow_name}' from orchestrator.")
        try:
            # Note: This API endpoint needs to be created in the Next.js backend.
            # It should serve the contents of the corresponding file from `dgn-client/workflows`.
            url = f"{self.orchestrator_url}/api/dgn/workflows/{workflow_name}"
            response = self._make_request('get', url)
            response.raise_for_status()
            
            workflow_data = response.json()

            # Save the newly downloaded workflow to the cache
            with open(cached_workflow_path, 'w') as f:
                json.dump(workflow_data, f)
            
            logging.info(f"Successfully downloaded and cached workflow '{workflow_name}'.")
            return workflow_data

        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading workflow {workflow_name}: {e}")
            return None
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from workflow response for {workflow_name}: {response.text}")
            return None
        except IOError as e:
            logging.error(f"Error saving downloaded workflow {workflow_name} to cache: {e}")
            # We can still return the data even if caching fails
            return workflow_data

