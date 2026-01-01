import requests
import logging
import base64
import json
import time
import threading
from typing import Union, Dict, Optional, Any, List

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import TimeoutConfig
from exceptions import AuthError, ProviderError, TransientError
from services.hardware_profiler import get_hardware_profile
import os

# Backward compatibility aliases
TokenExpiredError = AuthError
ProviderNotFoundError = ProviderError


class OrchestratorService:
    # Debounce interval for AUTH_EXPIRED signals (seconds)
    AUTH_EXPIRED_DEBOUNCE_SECONDS = 5

    def __init__(self, orchestrator_url: str, access_token: str = None, refresh_token: str = None, dgn_api_key: str = None):
        self.orchestrator_url = orchestrator_url
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.dgn_api_key = dgn_api_key
        self.use_api_key = dgn_api_key is not None
        self.token_update_lock = threading.Lock()
        self._last_auth_expired_signal = 0
        self._auth_failed_permanently = False

    def update_tokens(self, access_token: str, refresh_token: str):
        """Thread-safe method to update auth tokens."""
        with self.token_update_lock:
            self.access_token = access_token
            self.refresh_token = refresh_token
            self._auth_failed_permanently = False  # Reset permanent failure on new tokens
            logging.info("OrchestratorService tokens have been updated by the main process.")

    def mark_auth_failed_permanently(self):
        """Mark authentication as permanently failed (refresh token expired)."""
        with self.token_update_lock:
            self._auth_failed_permanently = True
            logging.error("Authentication marked as permanently failed. Client will stop.")

    def is_auth_failed_permanently(self) -> bool:
        """Thread-safe check if auth has permanently failed."""
        with self.token_update_lock:
            return self._auth_failed_permanently

    def signal_auth_expired(self):
        """
        Send AUTH_EXPIRED signal to main process with debouncing.
        Only sends one signal per AUTH_EXPIRED_DEBOUNCE_SECONDS to prevent spam.
        """
        current_time = time.time()
        with self.token_update_lock:
            if self._auth_failed_permanently:
                # Don't signal if already marked as permanently failed
                return
            if current_time - self._last_auth_expired_signal < self.AUTH_EXPIRED_DEBOUNCE_SECONDS:
                logging.debug("AUTH_EXPIRED signal debounced (within cooldown period).")
                return
            self._last_auth_expired_signal = current_time
        
        print(json.dumps({"status": "AUTH_EXPIRED"}), flush=True)
        logging.warning("AUTH_EXPIRED signal sent to main process.")

    @retry(
        stop=stop_after_attempt(TimeoutConfig.API_MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=TimeoutConfig.API_RETRY_MIN_WAIT,
            max=TimeoutConfig.API_RETRY_MAX_WAIT
        ),
        retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout, TransientError)),
        reraise=True
    )
    def _make_request(self, method: str, url: str, auth_required: bool = True, **kwargs) -> requests.Response:
        """
        Makes an HTTP request with automatic retry for transient failures.
        
        Supports both API key auth (headless) and Bearer token auth (Electron).
        Uses exponential backoff for retries on connection errors and timeouts.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            url: Target URL
            auth_required: Whether to include auth headers
            **kwargs: Additional arguments passed to requests.request()
            
        Returns:
            requests.Response object
            
        Raises:
            AuthError: When authentication fails (401)
            TransientError: When a retryable error occurs after max retries
        """
        # In API key mode, we don't need to check for token expiry
        if not self.use_api_key:
            # Check for permanent auth failure before making any authenticated request
            if auth_required and self.is_auth_failed_permanently():
                raise AuthError("Authentication has permanently failed. Please log in again.")

        # Use centralized timeout configuration
        kwargs.setdefault('timeout', TimeoutConfig.API_REQUEST_TIMEOUT)

        request_headers = kwargs.pop("headers", {}).copy()
        
        if auth_required:
            if self.use_api_key:
                # Headless mode: use API key
                request_headers['x-dgn-api-key'] = self.dgn_api_key
            else:
                # Electron/web mode: use Bearer token
                with self.token_update_lock:
                    request_headers['Authorization'] = f'Bearer {self.access_token}'

        if 'Content-Type' not in request_headers and 'files' not in kwargs and 'data' not in kwargs and 'json' not in kwargs:
             request_headers['Content-Type'] = 'application/json'
        
        try:
            response = requests.request(method, url, headers=request_headers, **kwargs)
        except requests.exceptions.ConnectionError as e:
            logging.warning(f"Connection error to {url}, will retry: {e}")
            raise  # Let tenacity handle the retry
        except requests.exceptions.Timeout as e:
            logging.warning(f"Request timeout to {url}, will retry: {e}")
            raise  # Let tenacity handle the retry

        # Handle server errors that are retryable
        if response.status_code == 503:
            raise TransientError(f"Server unavailable (503) at {url}")
        if response.status_code == 429:
            raise TransientError(f"Rate limited (429) at {url}")

        # Handle auth errors (not retryable)
        if response.status_code == 401 and auth_required:
            if self.use_api_key:
                logging.error("Received 401 Unauthorized with API key. Key may be invalid or revoked.")
                raise AuthError("DGN API key is invalid or revoked.")
            else:
                logging.warning("Received 401 Unauthorized. Notifying main process to refresh token.")
                raise AuthError("Access token has expired.")
        
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
            if e.response is not None:
                logging.error(f"Response content: {e.response.text}")
            return []
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from resolve_targets response: {response.text}")
            return []

    def get_next_job(self, provider_id: str, accept_policy: str, allowed_ids: list[str], job_id: str = None) -> Union[Dict, None]:
        """Get the next available job for a provider based on their policy."""
        try:
            base_url = f"{self.orchestrator_url}/api/dgn/jobs/{provider_id}"
            params = {
                "ts": int(time.time()),
                "acceptPolicy": accept_policy,
            }

            if job_id:
                params["jobId"] = job_id

            if accept_policy == 'mine':
                user_id = self._get_user_id_from_token()
                if not user_id:
                    logging.error("Could not get user ID for 'own' policy. Aborting job fetch.")
                    return None
                params["userId"] = user_id
            
            if (accept_policy == 'project' or accept_policy == 'users') and allowed_ids:
                params["allowedIds"] = ",".join(allowed_ids)

            response = self._make_request('get', base_url, params=params)
            
            # Check for provider expiration (404 with provider_not_found error)
            if response.status_code == 404:
                try:
                    error_data = response.json()
                    if error_data.get("error") == "provider_not_found":
                        raise ProviderNotFoundError("Provider registration has expired")
                except (json.JSONDecodeError, KeyError):
                    pass
            
            response.raise_for_status()
            if not response.content:
                return None
            return response.json()
        except ProviderNotFoundError:
            raise  # Re-raise to be handled by caller
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching next job: {e}")
            if e.response is not None:
                logging.error(f"Response content: {e.response.text}")
            return None
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from get_next_job response: {response.text}")
            return None

    def peek_available_jobs(self, provider_id: str, accept_policy: str, allowed_ids: list[str], limit: int = 10) -> list[Dict]:
        """
        Peek at available jobs without reserving any.
        
        Used for Docker image pre-fetching - allows the client to see what jobs
        are available and start downloading required images in the background.
        
        Args:
            provider_id: The provider's ID
            accept_policy: Job acceptance policy ('all', 'mine', 'project', 'users')
            allowed_ids: List of allowed IDs for project/users policies
            limit: Maximum number of jobs to return
            
        Returns:
            List of job dictionaries (without reserving them)
        """
        try:
            base_url = f"{self.orchestrator_url}/api/dgn/jobs/peek"
            params = {
                "ts": int(time.time()),
                "providerId": provider_id,
                "acceptPolicy": accept_policy,
                "limit": limit,
            }

            if accept_policy == 'mine':
                user_id = self._get_user_id_from_token()
                if not user_id:
                    logging.error("Could not get user ID for 'own' policy. Aborting job peek.")
                    return []
                params["userId"] = user_id
            
            if (accept_policy == 'project' or accept_policy == 'users') and allowed_ids:
                params["allowedIds"] = ",".join(allowed_ids)

            response = self._make_request('get', base_url, params=params)
            
            # Check for provider expiration (404 with provider_not_found error)
            if response.status_code == 404:
                try:
                    error_data = response.json()
                    if error_data.get("error") == "provider_not_found":
                        raise ProviderNotFoundError("Provider registration has expired")
                except (json.JSONDecodeError, KeyError):
                    pass
            
            response.raise_for_status()
            if not response.content:
                return []
            
            result = response.json()
            # Handle both array and object with 'jobs' key responses
            if isinstance(result, list):
                return result
            return result.get('jobs', [])
        except ProviderNotFoundError:
            raise  # Re-raise to be handled by caller
        except requests.exceptions.RequestException as e:
            logging.error(f"Error peeking at available jobs: {e}")
            if e.response is not None:
                logging.error(f"Response content: {e.response.text}")
            return []
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from peek_available_jobs response: {response.text}")
            return []


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
            if e.response is not None:
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
            if e.response is not None:
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
            
            # Check for provider expiration (404 with provider_not_found error)
            if response.status_code == 404:
                try:
                    error_data = response.json()
                    if error_data.get("error") == "provider_not_found":
                        raise ProviderNotFoundError("Provider registration has expired")
                except (json.JSONDecodeError, KeyError):
                    pass
            
            response.raise_for_status()
            logging.info("Heartbeat sent successfully.")
        except ProviderNotFoundError:
            raise  # Re-raise to be handled by caller
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

    def register_with_orchestrator(self, service_type: str, supported_services: list = None, cached_images: list = None, accept_policy: str = "all") -> Union[str, None]:
        """Register the client with the orchestrator."""
        hardware_profile = get_hardware_profile()
        
        # In API key mode, the server will get user_id from the API key
        # In OAuth mode, we extract it from the JWT token
        user_id = None
        if not self.use_api_key:
            user_id = self._get_user_id_from_token()
            if not user_id:
                logging.error(f"Could not extract user_id from token. Token prefix: {self.access_token[:10]}... Length: {len(self.access_token) if self.access_token else 0}. Cannot register.")
                return None
        
        payload = {
            **hardware_profile,
            "service_type": service_type,
            "supported_services": supported_services or [],
            "cached_images": cached_images or [],  # For smart job assignment
            "accept_policy": accept_policy,  # Job acceptance policy
        }
        
        # Only include user_id if we have it (OAuth mode)
        if user_id:
            payload["user_id"] = user_id
        
        # Detect cloud environment and include cloud instance info
        # This allows the orchestrator to correlate providers with cloud deployments
        cloud_instance_id = None
        cloud_provider = None
        
        # Vast.ai detection
        vast_container_id = os.environ.get("CONTAINER_ID") or os.environ.get("VAST_CONTAINERLABEL")
        if vast_container_id:
            # CONTAINER_ID is the numeric ID, VAST_CONTAINERLABEL is like "C.12345"
            cloud_instance_id = vast_container_id.replace("C.", "") if vast_container_id.startswith("C.") else vast_container_id
            cloud_provider = "vast.ai"
        
        # RunPod detection
        runpod_pod_id = os.environ.get("RUNPOD_POD_ID")
        if runpod_pod_id:
            cloud_instance_id = runpod_pod_id
            cloud_provider = "runpod"
        
        if cloud_instance_id:
            payload["cloud_instance_id"] = cloud_instance_id
            payload["cloud_provider"] = cloud_provider
            logging.info(f"Cloud environment detected: {cloud_provider} instance {cloud_instance_id}")

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
            if e.response is not None:
                logging.error(f"OrchestratorService: Response content: {e.response.text}")

    def reset_interrupted_job(self, job_id: str):
        """Resets a job's status to 'pending' and clears its provider via a specific API endpoint."""
        try:
            logging.info(f"Requesting reset for job {job_id}")
            response = self._make_request(
                'put',
                f"{self.orchestrator_url}/api/dgn/job/reset?jobId={job_id}"
            )
            response.raise_for_status()
            logging.info(f"Job {job_id} status reset successfully via API.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not reset job status for {job_id}: {e}")
            if e.response is not None:
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

        except IOError as e:
            logging.error(f"Error saving downloaded workflow {workflow_name} to cache: {e}")
            # We can still return the data even if caching fails
            return workflow_data

    def submit_job(self, job_data: Dict) -> Union[str, None]:
        """Submits a new job to the orchestrator."""
        try:
            response = self._make_request(
                'post',
                f"{self.orchestrator_url}/api/dgn/submit",
                json=job_data
            )
            response.raise_for_status()
            return response.json().get('job_id')
        except requests.exceptions.RequestException as e:
            logging.error(f"Error submitting job: {e}")
            if e.response is not None:
                logging.error(f"Response content: {e.response.text}")
            return None

    def report_cached_images(self, provider_id: str, cached_images: list[str], mode: str = "replace") -> bool:
        """
        Report cached Docker images to the server for smart job assignment.
        
        This doesn't affect credits - credits are calculated based on processing time
        and VRAM, not image caching. This optimization just helps route jobs to
        providers that can process them immediately.
        
        Args:
            provider_id: The provider's ID
            cached_images: List of service types with cached images
            mode: 'replace' to overwrite, 'add' to append new images
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self._make_request(
                'post',
                f"{self.orchestrator_url}/api/dgn/provider-cached-images",
                json={
                    "providerId": provider_id,
                    "cached_images": cached_images,
                    "mode": mode
                }
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logging.warning(f"Failed to report cached images: {e}")
            return False


