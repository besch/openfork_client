import requests
import logging
import base64
import json
import time
import threading
import ipaddress
import os
import re
import socket
import uuid
from typing import Union, Dict, Optional, Any, List, Tuple
from urllib.parse import urljoin, urlparse, unquote

from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from config import (
    TimeoutConfig,
    MAX_INPUT_ASSET_BYTES,
    MAX_INPUT_ASSET_REDIRECTS,
)
from exceptions import AuthError, ProviderError, TransientError
from services.hardware_profiler import get_hardware_profile
from utils.recent_logs import get_recent_logs_tail

# Backward compatibility aliases
TokenExpiredError = AuthError
ProviderNotFoundError = ProviderError


class OrchestratorService:
    # Debounce interval for AUTH_EXPIRED signals (seconds)
    AUTH_EXPIRED_DEBOUNCE_SECONDS = 5
    BLOCKED_ASSET_HOSTNAMES = {
        "localhost",
        "localhost.localdomain",
        "host.docker.internal",
        "gateway.docker.internal",
        "metadata.google.internal",
    }

    def __init__(self, orchestrator_url: str, access_token: str = None, refresh_token: str = None, dgn_api_key: str = None):
        self.orchestrator_url = orchestrator_url
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.dgn_api_key = dgn_api_key
        self.provider_id: Optional[str] = None
        self.use_api_key = dgn_api_key is not None
        self.token_update_lock = threading.Lock()
        self._last_auth_expired_signal = 0
        self._auth_failed_permanently = False
        self._active_job_id: Optional[str] = None
        self._active_execution_token: Optional[str] = None
        self.cloud_instance_id, self.cloud_provider = self._detect_cloud_instance()
        
        # PERFORMANCE: Use a Session for HTTP connection pooling
        # This reuses TCP connections across requests, reducing latency for
        # frequent operations like heartbeats and status updates
        self._session = requests.Session()
        # Configure connection pool size for parallel operations.
        # The client can have a heartbeat thread, cancellation checks, job peeks,
        # status updates, and asset downloads all active at once, so the default
        # pool size of 10 is easy to exhaust.
        pool_size = int(os.environ.get("OPENFORK_HTTP_POOL_SIZE", "32"))
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            max_retries=0  # We handle retries via tenacity
        )
        self._session.mount('http://', adapter)
        self._session.mount('https://', adapter)

    @staticmethod
    def _detect_cloud_instance() -> Tuple[Optional[str], Optional[str]]:
        # Vast.ai detection
        vast_container_id = os.environ.get("CONTAINER_ID") or os.environ.get("VAST_CONTAINERLABEL")
        if vast_container_id:
            normalized_id = vast_container_id.replace("C.", "") if vast_container_id.startswith("C.") else vast_container_id
            return normalized_id, "vast.ai"

        # RunPod detection
        runpod_pod_id = os.environ.get("RUNPOD_POD_ID")
        if runpod_pod_id:
            return runpod_pod_id, "runpod"

        return None, None

    def _set_active_job(self, job: Optional[Dict[str, Any]]) -> None:
        """Track the active leased job so heartbeats and status updates carry its token."""
        if not job:
            self._active_job_id = None
            self._active_execution_token = None
            return

        self._active_job_id = job.get("id")
        self._active_execution_token = job.get("execution_token")

    def clear_active_job(self, job_id: Optional[str] = None) -> None:
        """Clear tracked lease state when a job is completed, reset, or replaced."""
        if job_id is None or self._active_job_id == job_id:
            self._active_job_id = None
            self._active_execution_token = None

    def get_active_execution_token(self, job_id: Optional[str] = None) -> Optional[str]:
        """Return the current execution token, optionally scoped to one job."""
        if job_id is not None and self._active_job_id != job_id:
            return None
        return self._active_execution_token

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
        # wait_random_exponential adds full jitter so a fleet of clients does
        # not synchronise their retries during a shared orchestrator outage.
        wait=wait_random_exponential(
            multiplier=TimeoutConfig.API_RETRY_MIN_WAIT,
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
                if self.cloud_instance_id:
                    request_headers.setdefault('x-dgn-cloud-instance-id', self.cloud_instance_id)
            else:
                # Electron/web mode: use Bearer token
                with self.token_update_lock:
                    request_headers['Authorization'] = f'Bearer {self.access_token}'

        if 'Content-Type' not in request_headers and 'files' not in kwargs and 'data' not in kwargs and 'json' not in kwargs:
             request_headers['Content-Type'] = 'application/json'
        
        try:
            # Use session for connection pooling
            response = self._session.request(method, url, headers=request_headers, **kwargs)
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

    def get_next_job(self, provider_id: str, accept_policy: str, allowed_ids: list[str], job_id: str = None, monetize_mode: bool = False) -> Union[Dict, None]:
        """Get the next available job for a provider based on their policy."""
        try:
            base_url = f"{self.orchestrator_url}/api/dgn/jobs/{provider_id}"
            params = {
                "ts": int(time.time()),
                "acceptPolicy": accept_policy,
            }

            if monetize_mode:
                params["monetize"] = "true"

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
            job = response.json()
            if isinstance(job, dict) and job.get("id"):
                self._set_active_job(job)
            return job
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
        """Get a job by its ID.

        Returns fabricated local-stop statuses for cases where this worker
        should stop acting on the job even though the remote row may still
        exist:
        - 404 => cancelled/deleted elsewhere
        - 403 => lease lost / job reset or reassigned away from this provider
        """
        try:
            # This endpoint needs to be created in the Next.js app: GET /api/dgn/job/{job_id}
            url = f"{self.orchestrator_url}/api/dgn/job/{job_id}"
            response = self._make_request('get', url)

            if response.status_code == 404:
                logging.warning(f"Job {job_id} not found (404). Assuming it was cancelled and deleted.")
                self.clear_active_job(job_id)
                return {'status': 'cancelled'}
            if response.status_code == 403:
                logging.warning(
                    f"Job {job_id} is no longer accessible to this provider (403). "
                    "Assuming the job lease was lost, reset, or reassigned."
                )
                self.clear_active_job(job_id)
                return {'status': 'lease_lost'}

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

    @staticmethod
    def _is_blocked_asset_ip(ip_text: str) -> bool:
        try:
            ip_obj = ipaddress.ip_address(ip_text)
        except ValueError:
            return True

        return any(
            (
                ip_obj.is_private,
                ip_obj.is_loopback,
                ip_obj.is_link_local,
                ip_obj.is_multicast,
                ip_obj.is_reserved,
                ip_obj.is_unspecified,
            )
        )

    def _resolve_asset_host_ips(self, parsed) -> Tuple[str, set[str]]:
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").strip().lower().rstrip(".")

        try:
            literal_ip = ipaddress.ip_address(hostname)
        except ValueError:
            literal_ip = None

        if literal_ip is not None:
            return hostname, {hostname}

        try:
            resolved = {
                info[4][0]
                for info in socket.getaddrinfo(
                    hostname,
                    parsed.port or (443 if scheme == "https" else 80),
                    type=socket.SOCK_STREAM,
                )
            }
        except socket.gaierror as exc:
            raise ValueError(f"Unable to resolve asset hostname '{hostname}': {exc}") from exc

        if not resolved:
            raise ValueError(f"No IP addresses resolved for asset hostname '{hostname}'")

        return hostname, resolved

    @staticmethod
    def _get_response_peer_ip(response: requests.Response) -> Optional[str]:
        candidate_paths = [
            ("raw", "_connection", "sock"),
            ("raw", "connection", "sock"),
            ("raw", "_fp", "fp", "raw", "_sock"),
            ("raw", "_fp", "fp", "raw", "sock"),
        ]

        for path in candidate_paths:
            current: Any = response
            for attr in path:
                current = getattr(current, attr, None)
                if current is None:
                    break
            if current is None:
                continue
            try:
                peer = current.getpeername()
            except Exception:
                continue
            if peer:
                return peer[0]

        return None

    def _validate_asset_url(self, asset_url: str) -> Tuple[str, set[str]]:
        parsed = urlparse(asset_url)
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").strip().lower().rstrip(".")

        if scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported asset URL scheme: {scheme or 'missing'}")

        if not hostname:
            raise ValueError("Asset URL is missing a hostname")

        if parsed.username or parsed.password:
            raise ValueError("Asset URLs with embedded credentials are not allowed")

        if hostname in self.BLOCKED_ASSET_HOSTNAMES:
            raise ValueError(f"Blocked asset hostname: {hostname}")

        try:
            literal_ip = ipaddress.ip_address(hostname)
        except ValueError:
            literal_ip = None

        if literal_ip is not None:
            if self._is_blocked_asset_ip(hostname):
                raise ValueError(f"Blocked asset IP: {hostname}")
            return parsed.geturl(), {hostname}

        _, resolved = self._resolve_asset_host_ips(parsed)
        blocked_ips = sorted(ip for ip in resolved if self._is_blocked_asset_ip(ip))
        if blocked_ips:
            raise ValueError(
                f"Asset hostname '{hostname}' resolves to blocked address(es): {', '.join(blocked_ips)}"
            )

        return parsed.geturl(), resolved

    def _open_validated_asset_stream(self, asset_url: str) -> Tuple[str, requests.Response]:
        current_url = asset_url

        for redirect_count in range(MAX_INPUT_ASSET_REDIRECTS + 1):
            validated_url, allowed_ips = self._validate_asset_url(current_url)
            response = self._session.get(
                validated_url,
                stream=True,
                timeout=120,
                allow_redirects=False,
            )

            peer_ip = self._get_response_peer_ip(response)
            if not peer_ip:
                response.close()
                raise ValueError("Unable to verify asset peer address")
            if self._is_blocked_asset_ip(peer_ip):
                response.close()
                raise ValueError(f"Asset connection reached blocked address: {peer_ip}")
            if peer_ip not in allowed_ips:
                response.close()
                raise ValueError(
                    f"Asset host resolved to unexpected address after validation: {peer_ip}"
                )

            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    raise ValueError("Asset redirect response missing Location header")
                current_url = urljoin(validated_url, location)
                continue

            response.raise_for_status()
            return validated_url, response

        raise ValueError(
            f"Asset download exceeded redirect limit ({MAX_INPUT_ASSET_REDIRECTS})"
        )

    @staticmethod
    def _build_safe_asset_filename(asset_url: str, content_type: str) -> str:
        content_type_map = {
            'image/jpeg': '.jpeg',
            'image/jpg': '.jpeg',
            'image/png': '.png',
            'image/webp': '.webp',
            'image/gif': '.gif',
            'video/mp4': '.mp4',
            'video/webm': '.webm',
            'audio/mpeg': '.mp3',
            'audio/wav': '.wav',
            'audio/flac': '.flac',
        }

        parsed = urlparse(asset_url)
        raw_name = unquote(os.path.basename((parsed.path or "").replace("\\", "/"))).strip()
        if raw_name in {"", ".", ".."}:
            raw_name = "asset"

        sanitized_name = re.sub(r"[^A-Za-z0-9._-]", "_", raw_name)
        stem, ext = os.path.splitext(sanitized_name)
        stem = stem.strip("._") or "asset"
        ext = ext.lower()

        normalized_content_type = content_type.split(";", 1)[0].strip().lower()
        expected_ext = content_type_map.get(normalized_content_type)

        if expected_ext and ext not in {expected_ext, ".jpg", ".jpeg"}:
            ext = expected_ext
        elif not ext:
            ext = expected_ext or ".dat"

        safe_stem = stem[:80]
        return f"{safe_stem}-{uuid.uuid4().hex[:8]}{ext}"

    def download_asset_by_url(self, asset_url: str, download_dir: str) -> Union[str, None]:
        """Download an asset from a given URL with validation."""
        file_path: Optional[str] = None
        try:
            os.makedirs(download_dir, exist_ok=True)
            final_url, response = self._open_validated_asset_stream(asset_url)
            with response:
                content_type = response.headers.get('Content-Type', '').lower()
                content_length = response.headers.get("Content-Length")

                if content_length:
                    try:
                        expected_size = int(content_length)
                    except ValueError:
                        expected_size = None
                    else:
                        if expected_size < 0:
                            raise ValueError("Asset response reported a negative Content-Length")
                        if expected_size > MAX_INPUT_ASSET_BYTES:
                            raise ValueError(
                                f"Asset exceeds max allowed size ({expected_size} bytes > {MAX_INPUT_ASSET_BYTES} bytes)"
                            )

                file_name = self._build_safe_asset_filename(final_url, content_type)
                file_path = os.path.join(download_dir, file_name)

                downloaded_bytes = 0
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        downloaded_bytes += len(chunk)
                        if downloaded_bytes > MAX_INPUT_ASSET_BYTES:
                            raise ValueError(
                                f"Asset exceeds max allowed size while downloading ({downloaded_bytes} bytes)"
                            )
                        f.write(chunk)
            
            # Validate downloaded file
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logging.error(f"Downloaded file is empty (0 bytes): {file_path}")
                os.remove(file_path)
                return None
            
            # For images, validate they can be opened by PIL
            if content_type.startswith('image/'):
                try:
                    from PIL import Image
                    with Image.open(file_path) as img:
                        img.verify()  # Verify it's a valid image
                    logging.info(f"Image validated successfully: {file_path} ({file_size} bytes)")
                except Exception as e:
                    logging.error(f"Downloaded file is not a valid image: {file_path}. Error: {e}")
                    # Try to read the first few bytes to diagnose
                    with open(file_path, 'rb') as f:
                        header = f.read(20)
                    logging.error(f"File header (first 20 bytes): {header}")
                    os.remove(file_path)
                    return None
            
            logging.info(f"Asset downloaded to {file_path} ({file_size} bytes)")
            return file_path
        except ValueError as e:
            logging.error(f"Rejected asset download from {asset_url}: {e}")
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading asset from {asset_url}: {e}")
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            return None
        except Exception as e:
            logging.error(f"Unexpected error downloading asset: {e}")
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            return None

    def _get_signed_upload_url(self, job_id: str, file_name: str) -> Union[Dict, None]:
        """Get a presigned URL for uploading a file directly to storage."""
        try:
            payload = {"jobId": job_id, "fileName": file_name}
            execution_token = self.get_active_execution_token(job_id)
            if execution_token:
                payload["executionToken"] = execution_token

            response = self._make_request(
                'post',
                f"{self.orchestrator_url}/api/dgn/upload-url",
                json=payload
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
        file_size = os.path.getsize(file_path)
        
        upload_info = self._get_signed_upload_url(job_id, file_name)
        if not upload_info or not upload_info.get('success'):
            logging.error(f"Failed to get a presigned URL for job {job_id}")
            return None
        
        upload_url = upload_info['uploadUrl']
        storage_path = upload_info['storagePath']
        max_file_size = upload_info.get('maxFileSize')

        try:
            max_file_size = int(max_file_size) if max_file_size is not None else None
        except (TypeError, ValueError):
            max_file_size = None

        if max_file_size is not None and file_size > max_file_size:
            logging.error(
                f"Output file {file_name} for job {job_id} is {file_size} bytes, "
                f"which exceeds bucket limit {max_file_size} bytes."
            )
            return None

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

            logging.info(
                f"Successfully uploaded {file_name} ({file_size} bytes) for job {job_id}. "
                f"Storage path: {storage_path}"
            )

            # SECURITY: Defense-in-depth magic-byte verification. The presigned
            # PUT path stores whatever bytes arrive — there is no server-side
            # check that an .mp4 is really an mp4. We tell the orchestrator
            # the upload finished; it downloads the first 32 bytes back and
            # rejects anything that doesn't match the extension's magic
            # signature (deleting the object server-side).
            #
            # Failing this is a hard error: workflow outputs are produced by
            # trusted Docker images, so a magic-byte mismatch indicates either
            # a bug in our pipeline or a tampered upload — neither should
            # advance to job completion.
            finalize_ok = self._finalize_upload(job_id, storage_path)
            if not finalize_ok:
                logging.error(
                    f"Upload finalize rejected {storage_path} for job {job_id}; "
                    "the orchestrator deleted the object."
                )
                return None

            return storage_path
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not upload file to presigned URL: {e}")
            if e.response is not None:
                logging.error(f"Response content: {e.response.text}")
                if e.response.status_code == 413:
                    logging.error(
                        "Upload rejected by storage size limit. Increase the Supabase "
                        "bucket file_size_limit for project buckets or reduce output bitrate."
                    )
            return None

    def _finalize_upload(self, job_id: str, storage_path: str) -> bool:
        """Ask the orchestrator to magic-byte-verify a freshly uploaded object.

        Returns True on success or transport failure (we fail open on
        unreachable orchestrator so a brief outage doesn't fail the job).
        Returns False only when the orchestrator explicitly rejects the
        object — that means the file was deleted server-side and the
        worker must not advance the job to completion.
        """
        try:
            payload: Dict[str, Any] = {
                "jobId": job_id,
                "storagePath": storage_path,
            }
            execution_token = self.get_active_execution_token(job_id)
            if execution_token:
                payload["executionToken"] = execution_token

            response = self._make_request(
                'post',
                f"{self.orchestrator_url}/api/dgn/upload-finalize",
                json=payload,
            )
            if response.status_code >= 400 and response.status_code < 500:
                # Server explicitly rejected — magic byte mismatch, lease
                # expired, etc. Do not let the job complete with this output.
                logging.warning(
                    f"upload-finalize rejected job {job_id} path {storage_path}: "
                    f"HTTP {response.status_code} {response.text}"
                )
                return False
            response.raise_for_status()
            return True
        except (requests.exceptions.RequestException, TransientError) as e:
            # Transport-level failure — fail open. The cached output stays;
            # job completion proceeds. A persistent outage will be noticed
            # via heartbeat health, not by silently corrupting jobs.
            logging.warning(
                f"upload-finalize call failed for job {job_id} (failing open): {e}"
            )
            return True

    def upload_audio_output(self, file_path: str, job_id: str) -> Union[str, None]:
        """Upload the audio output file to the orchestrator."""
        return self.upload_output(file_path, job_id, 'audio/flac')

    def upload_image_output(self, file_path: str, job_id: str) -> Union[str, None]:
        """Upload the image output file to the orchestrator."""
        return self.upload_output(file_path, job_id, 'image/png')

    def upload_thumbnail(self, file_path: str, job_id: str) -> Union[str, None]:
        """Upload the thumbnail file to the orchestrator."""
        return self.upload_output(file_path, job_id, 'image/jpeg')

    def send_heartbeat(
        self,
        provider_id: str,
        compaction_pending: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Sends a heartbeat to the orchestrator.

        Returns the routing_config from the response, or None if no config was returned.
        """
        try:
            payload: Dict[str, Any] = {"providerId": provider_id}
            active_token = self.get_active_execution_token()
            if active_token:
                payload["executionToken"] = active_token
            if compaction_pending:
                payload["compactionPending"] = True

            response = self._make_request(
                'post',
                f"{self.orchestrator_url}/api/dgn/heartbeat",
                json=payload
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
            try:
                data = response.json()
                return data.get("routing_config")
            except (json.JSONDecodeError, AttributeError):
                return None
        except ProviderNotFoundError:
            raise  # Re-raise to be handled by caller
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not send heartbeat: {e}")
        return None


    def update_job_status(self, job_id: str, status: str, storage_path: Union[str, None] = None, thumbnail_storage_path: Union[str, None] = None, duration_seconds: float = None, completion_metadata: Dict = None, prompt: Union[str, None] = None, execution_token: Optional[str] = None) -> bool:
        """Update the status of a job."""
        try:
            payload = {"status": status}
            metadata_payload = dict(completion_metadata) if completion_metadata else None

            if status == "failed":
                metadata_payload = metadata_payload or {}
                metadata_payload.setdefault("provider_logs_tail", get_recent_logs_tail(100))

            if self.provider_id:
                payload["provider_id"] = self.provider_id
            token = execution_token or self.get_active_execution_token(job_id)
            if token:
                payload["execution_token"] = token
            if storage_path:
                payload["storage_path"] = storage_path
            if thumbnail_storage_path:
                payload["thumbnail_storage_path"] = thumbnail_storage_path
            if duration_seconds:
                payload["duration_seconds"] = duration_seconds
            if metadata_payload is not None:
                payload["completion_metadata"] = metadata_payload
            if prompt:
                payload["prompt"] = prompt

            response = self._make_request(
                'put',
                f"{self.orchestrator_url}/api/dgn/job/{job_id}",
                json=payload
            )
            response.raise_for_status()
            response_status = status
            try:
                response_data = response.json()
                if isinstance(response_data, dict):
                    response_status = response_data.get("status") or status
            except (ValueError, AttributeError):
                pass

            if response_status != status:
                logging.info(
                    f"Job {job_id} status update to {status} was accepted; "
                    f"orchestrator finalized status as {response_status}"
                )
            else:
                logging.info(f"Job {job_id} status updated to {status}")
            if status in ("completed", "failed", "cancelled"):
                self.clear_active_job(job_id)
            return True
        except requests.exceptions.RequestException as e:
            if (
                status == "cancelled"
                and e.response is not None
                and e.response.status_code == 404
            ):
                logging.info(
                    f"Job {job_id} was already missing while marking it "
                    "cancelled; treating as already cancelled/deleted."
                )
                self.clear_active_job(job_id)
                return False
            if (
                status in ("completed", "failed", "cancelled")
                and e.response is not None
                and e.response.status_code in (403, 409)
            ):
                logging.warning(
                    f"Could not update job {job_id} to {status}: "
                    "the job lease was lost or the job was already reset."
                )
                if e.response.text:
                    logging.warning(f"Job status update response: {e.response.text}")
                self.clear_active_job(job_id)
                return False
            logging.error(f"Could not update job status: {e}")
            if e.response is not None and e.response.text:
                logging.error(f"Job status update response: {e.response.text}")
            return False

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

    def register_with_orchestrator(
        self,
        service_type: str,
        supported_services: list = None,
        cached_images: list = None,
        process_own_jobs: bool = False,
        community_mode: str = "all",
        allowed_ids: list = None,
        monetize_mode: bool = False,
    ) -> Union[Dict[str, str], None]:
        """Register the client with the orchestrator.
        
        Returns:
            A dict with 'provider_id' and 'user_id' keys on success, or None on failure.
        """
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
            "cached_images": cached_images or [],
            "process_own_jobs": process_own_jobs,
            "community_mode": community_mode,
            "allowed_ids": allowed_ids or [],
            "monetize_mode": monetize_mode,
        }
        
        # Only include user_id if we have it (OAuth mode)
        if user_id:
            payload["user_id"] = user_id
        
        if self.cloud_instance_id:
            payload["cloud_instance_id"] = self.cloud_instance_id
            payload["cloud_provider"] = self.cloud_provider
            logging.info(f"Cloud environment detected: {self.cloud_provider} instance {self.cloud_instance_id}")

        logging.info(f"Registering with profile: {payload}")
        try:
            response = self._make_request(
                'post',
                f"{self.orchestrator_url}/api/dgn/register",
                json=payload
            )
            response.raise_for_status()
            logging.info("Successfully registered with the Orchestrator.")
            data = response.json()
            self.provider_id = data.get("provider_id")
            # Return both provider_id and user_id (user_id is returned for API key mode)
            return {
                "provider_id": data.get('provider_id'),
                "user_id": data.get('user_id') or user_id  # Fall back to local user_id for OAuth mode
            }
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

    def reset_interrupted_job(self, job_id: str, execution_token: Optional[str] = None, reason: str = "provider_interrupted"):
        """Resets a job's status to 'pending' and clears its provider via a specific API endpoint."""
        try:
            logging.info(f"Requesting reset for job {job_id}")
            params = {"jobId": job_id}
            if self.provider_id:
                params["providerId"] = self.provider_id
            token = execution_token or self.get_active_execution_token(job_id)
            if token:
                params["executionToken"] = token
            if reason:
                params["reason"] = reason
            response = self._make_request(
                'put',
                f"{self.orchestrator_url}/api/dgn/job/reset",
                params=params,
            )
            response.raise_for_status()
            logging.info(f"Job {job_id} status reset successfully via API.")
            self.clear_active_job(job_id)
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
            True if successful, False otherwise.
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

    def report_download_state(
        self,
        provider_id: str,
        service_type: str,
        action: str,
        accept_policy: Optional[str] = None,
        return_none_on_error: bool = False,
    ) -> Optional[bool]:
        """
        Report Docker image download state change to the server.
        
        This enables server-side tracking for smart job routing - jobs are prioritized
        to providers that have the required image cached or downloading.
        
        NOTE: This does NOT affect credits. Credits are calculated based on actual
        processing time and VRAM usage, not cache state.
        
        Args:
            provider_id: The provider's ID
            service_type: The service type being downloaded (e.g., 'wan22-12gb')
            action: One of 'start', 'finish', or 'cancel'
            accept_policy: Optional job policy for start requests. This lets
                own/trusted jobs bypass global public-pool coverage checks.
            return_none_on_error: Return None instead of False when the server
                cannot be reached.
            
        Returns:
            True if accepted, False if intentionally declined, None on
            transport error when requested.
        """
        try:
            payload = {
                "providerId": provider_id,
                "service_type": service_type,
                "action": action,
            }
            if accept_policy:
                payload["accept_policy"] = accept_policy

            response = self._make_request(
                'post',
                f"{self.orchestrator_url}/api/dgn/provider-download-state",
                json=payload,
            )
            response.raise_for_status()
            accepted = True
            if response.content:
                try:
                    data = response.json()
                    accepted = data.get("accepted", True) is not False
                except json.JSONDecodeError:
                    accepted = True
            logging.debug(f"Reported download state: {service_type} - {action}")
            return accepted
        except requests.exceptions.RequestException as e:
            # Non-critical - don't fail if reporting fails
            logging.warning(f"Failed to report download state: {e}")
            return None if return_none_on_error else False

    def get_prefetch_suggestions(
        self,
        provider_id: str,
        limit: int = 10,
        return_none_on_error: bool = False,
    ) -> Optional[list[str]]:
        """
        Get pre-fetch suggestions from the server based on network demand.

        The server analyzes pending jobs and cache coverage to suggest which
        Docker images this provider should download proactively.  The returned
        list only contains service types where effective demand is not already
        covered by cached + downloading providers, so callers can use it as a
        coordination gate: if a service type is absent, another provider is
        already handling the download.

        NOTE: This is purely for network efficiency. It does NOT affect credits.

        Args:
            provider_id: The provider's ID
            limit: Max number of suggestions to return.  Pass a larger value
                   (e.g. 20) when using the result to gate peeked-job downloads
                   so that all uncovered service types are visible.

            return_none_on_error: Return None instead of [] when the server
                cannot be reached, so download gates can fail open.

        Returns:
            List of service types with genuine uncovered demand, ordered by
            cache_deficit DESC. Empty list means coverage is sufficient; None
            means the coverage gate was unavailable when requested.
        """
        try:
            response = self._make_request(
                'get',
                f"{self.orchestrator_url}/api/dgn/provider-download-state",
                params={"providerId": provider_id, "limit": limit}
            )
            response.raise_for_status()
            data = response.json()
            return data.get('suggestions', [])
        except requests.exceptions.RequestException as e:
            logging.debug(f"Failed to get prefetch suggestions: {e}")
            return None if return_none_on_error else []


