"""
Docker Download Manager

Manages background Docker image downloads with concurrency limiting.
Downloads are performed in daemon threads that don't block app exit.
"""

import threading
import logging
import time
from typing import Dict, Optional, Set
from enum import Enum
import docker
from services.disk_space_utils import estimate_image_size_bytes, check_sufficient_space

_docker_errors = getattr(docker, "errors", None)
DockerImageNotFound = getattr(
    _docker_errors,
    "ImageNotFound",
    type("DockerImageNotFound", (Exception,), {}),
)
DockerNotFound = getattr(
    _docker_errors,
    "NotFound",
    type("DockerNotFound", (Exception,), {}),
)
DockerAPIError = getattr(
    _docker_errors,
    "APIError",
    type("DockerAPIError", (Exception,), {}),
)


class DownloadStatus(Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    PERMANENTLY_FAILED = "permanently_failed"  # e.g. image does not exist on registry


class DockerDownloadManager:
    """
    Manages background Docker image downloads with concurrency limiting.
    
    Key features:
    - Max 1 concurrent download to avoid resource contention
    - Thread-safe state management
    - Non-blocking: downloads run in daemon threads
    - Deduplication: won't start duplicate downloads for same image
    - Graceful shutdown: daemon threads don't block process exit
    - Reports completed downloads to server for smart job assignment
    """
    
    MAX_CONCURRENT_DOWNLOADS = 1
    
    def __init__(
        self,
        docker_manager,
        orchestrator_service=None,
        provider_id=None,
        wakeup_event=None,
        max_cached_images: Optional[int] = None,
    ):
        """
        Initialize the download manager.
        
        Args:
            docker_manager: The DockerProdManager instance for Docker operations
            orchestrator_service: Optional OrchestratorService for reporting cached images
            provider_id: Optional provider ID for server-side tracking
            wakeup_event: Optional threading.Event to set when a download completes
            max_cached_images: Optional hard cap for locally cached Docker images.
                When set, least-recently-used images are evicted before starting
                a new download so disk usage stays bounded.
        """
        self.docker_manager = docker_manager
        self.orchestrator_service = orchestrator_service
        self.provider_id = provider_id
        self.wakeup_event = wakeup_event
        self.max_cached_images = (
            max_cached_images if max_cached_images and max_cached_images > 0 else None
        )
        self._lock = threading.RLock()
        self._active_downloads: Set[str] = set()  # service_types currently downloading
        self._download_queue: list[str] = []  # service_types waiting to download
        self._download_status: Dict[str, DownloadStatus] = {}
        self._cancellation_events: Dict[str, threading.Event] = {}
        self._last_job_times: Dict[str, float] = {}
        self._shutdown = False
        
    def has_image(self, service_type: str) -> bool:
        """
        Check if Docker image for a service exists locally.
        
        Args:
            service_type: The service type (e.g., 'wan22', 'hunyuan')
            
        Returns:
            True if image exists locally, False otherwise
        """
        docker_client = getattr(self.docker_manager, "client", None) if self.docker_manager else None
        if not self.docker_manager or not docker_client:
            return False if self.docker_manager else True  # Headless mode - no Docker needed
            
        try:
            image_name = self.docker_manager.get_image_name(service_type)
            docker_client.images.get(image_name)
            return True
        except DockerImageNotFound:
            return False
        except Exception as e:
            logging.warning(f"Error checking image for {service_type}: {e}")
            return False

    def notify_job_complete(self, service_type: str):
        """Record that a service type was actually used to process a job this session."""
        if not service_type:
            return

        with self._lock:
            self._last_job_times[service_type] = time.time()

    def _get_known_service_types(self) -> list[str]:
        """Return all service types known to this client for cache accounting."""
        if not self.docker_manager:
            return []

        services_config = getattr(self.docker_manager, "services_config", None) or {}
        if services_config:
            return list(services_config.keys())

        docker_image_map = getattr(self.docker_manager, "docker_image_map", None) or {}
        return list(docker_image_map.keys())

    def _get_cached_service_types(self) -> list[str]:
        """Return cached service types using the manager's known service list."""
        cached = []
        for service_type in self._get_known_service_types():
            if self.has_image(service_type):
                cached.append(service_type)
        return cached

    def _get_running_service_types(self) -> Set[str]:
        """Return service types whose containers are currently running."""
        if (
            not self.docker_manager
            or not hasattr(self.docker_manager, "get_container_name")
            or not getattr(self.docker_manager, "client", None)
        ):
            return set()

        running = set()
        for service_type in self._get_known_service_types():
            try:
                container_name = self.docker_manager.get_container_name(service_type)
                container = self.docker_manager.client.containers.get(container_name)
                if getattr(container, "status", None) == "running":
                    running.add(service_type)
            except DockerNotFound:
                continue
            except Exception as e:
                logging.debug(f"Could not inspect running container for {service_type}: {e}")

        return running

    def _sync_cached_images_with_server(self):
        """Replace the server-side cached_images list after a local eviction."""
        if not self.orchestrator_service or not self.provider_id:
            return

        cached_service_types = self._get_cached_service_types()
        try:
            self.orchestrator_service.report_cached_images(
                provider_id=self.provider_id,
                cached_images=cached_service_types,
                mode="replace",
            )
            logging.debug(f"Synced cached images after eviction: {cached_service_types}")
        except Exception as e:
            logging.warning(f"Failed to sync cached images after eviction: {e}")

    def _evict_lru_image(
        self, exclude_service_types: Optional[Set[str]] = None
    ) -> Optional[str]:
        """
        Evict one cached image using session-local LRU ordering.

        Eviction priority:
        1. Cached images with no completed jobs this session (prefetched but unused)
        2. Otherwise, the image with the oldest last completed job time

        Images are never evicted if they are currently running, downloading, or queued.
        """
        if not self.docker_manager or not getattr(self.docker_manager, "client", None):
            return None

        exclude = exclude_service_types or set()
        busy_service_types = (
            set(self._active_downloads)
            | set(self._download_queue)
            | self._get_running_service_types()
            | exclude
        )

        candidates: list[tuple[tuple[int, float, str], str]] = []
        for service_type in self._get_cached_service_types():
            if service_type in busy_service_types:
                continue

            last_job_time = self._last_job_times.get(service_type)
            sort_key = (
                0 if last_job_time is None else 1,
                last_job_time if last_job_time is not None else 0.0,
                service_type,
            )
            candidates.append((sort_key, service_type))

        if not candidates:
            return None

        _, service_type_to_evict = min(candidates, key=lambda item: item[0])
        image_name = self.docker_manager.get_image_name(service_type_to_evict)

        try:
            referencing_containers = self.docker_manager.client.containers.list(
                all=True, filters={"ancestor": image_name}
            )
            for container in referencing_containers:
                if getattr(container, "status", None) == "running":
                    logging.info(
                        f"Skipping eviction for {service_type_to_evict}; container is running."
                    )
                    return None
                container.remove(force=True)

            self.docker_manager.client.images.remove(image=image_name, force=True)
            self._download_status.pop(service_type_to_evict, None)
            self._sync_cached_images_with_server()
            logging.info(
                f"Evicted cached image for {service_type_to_evict} to honor image cap."
            )
            return service_type_to_evict
        except DockerImageNotFound:
            self._download_status.pop(service_type_to_evict, None)
            self._sync_cached_images_with_server()
            return service_type_to_evict
        except Exception as e:
            logging.warning(f"Failed to evict cached image for {service_type_to_evict}: {e}")
            return None

    def _ensure_cache_capacity(self, incoming_service_type: str) -> bool:
        """Evict LRU images until a new download fits under the configured image cap."""
        if not self.max_cached_images:
            return True

        cached_service_types = self._get_cached_service_types()
        if incoming_service_type in cached_service_types:
            return True

        while len(cached_service_types) >= self.max_cached_images:
            evicted_service = self._evict_lru_image(
                exclude_service_types={incoming_service_type}
            )
            if not evicted_service:
                logging.warning(
                    f"Image cap reached ({self.max_cached_images}) but no cached image "
                    f"could be evicted for incoming service '{incoming_service_type}'."
                )
                return False
            cached_service_types = self._get_cached_service_types()

        return True
    
    def is_downloading(self, service_type: str) -> bool:
        """Check if a service's image is currently being downloaded."""
        with self._lock:
            return service_type in self._active_downloads
    
    def is_queued(self, service_type: str) -> bool:
        """Check if a service's image is queued for download."""
        with self._lock:
            return service_type in self._download_queue
    
    def get_download_status(self, service_type: str) -> Optional[DownloadStatus]:
        """Get the download status for a service type."""
        with self._lock:
            return self._download_status.get(service_type)
    
    def start_background_download(self, service_type: str) -> bool:
        """
        Start downloading the Docker image for a service in the background.
        
        If already downloading or queued, this is a no-op.
        If at max concurrent downloads, queues the download.
        
        Args:
            service_type: The service type to download image for
            
        Returns:
            True if download was started or queued, False if already in progress
        """
        if self._shutdown:
            return False
            
        if not self.docker_manager:
            return False  # Headless mode
        
        with self._lock:
            # Check if already downloading or queued
            if service_type in self._active_downloads:
                logging.debug(f"Image for {service_type} already downloading")
                return False
            if service_type in self._download_queue:
                logging.debug(f"Image for {service_type} already queued")
                return False
            
            # TOCTOU fix: Check inside lock to prevent race condition
            # where multiple threads could pass the has_image check simultaneously
            if self.has_image(service_type):
                logging.debug(f"Image for {service_type} already exists, skipping download")
                # Clear failed status if image exists now
                self._download_status.pop(service_type, None)
                return False
            
            # Permanently-failed images (e.g. 404 - image doesn't exist on registry) must
            # not be retried automatically; they require a config fix to resolve.
            if self._download_status.get(service_type) == DownloadStatus.PERMANENTLY_FAILED:
                logging.debug(f"Image for {service_type} permanently failed (image not found on registry); skipping retry")
                return False

            # Clear any previous FAILED status to allow retry
            # This is important for resuming after a cancelled download
            if service_type in self._download_status:
                prev_status = self._download_status[service_type]
                if prev_status == DownloadStatus.FAILED:
                    logging.info(f"Clearing previous FAILED status for {service_type} to allow retry")
                    del self._download_status[service_type]
            
            # Check if we can start a new download
            if len(self._active_downloads) < self.MAX_CONCURRENT_DOWNLOADS:
                if not self._ensure_cache_capacity(service_type):
                    self._download_status[service_type] = DownloadStatus.FAILED
                    return False

                # Check for sufficient disk space before starting.
                # Prefer disk_required_gb from services_config (authoritative, from services.json)
                # over the hardcoded estimate map, which can be significantly off (e.g. wan22-24gb:
                # estimate=120 GB, actual disk_required_gb=220 GB).
                image_name = self.docker_manager.get_image_name(service_type)
                service_config = (self.docker_manager.services_config or {}).get(service_type, {})
                disk_required_gb = service_config.get("disk_required_gb")
                if disk_required_gb:
                    required_bytes = int(disk_required_gb * 1024 ** 3)
                else:
                    # Fallback to name-based heuristic when config is unavailable
                    required_bytes = estimate_image_size_bytes(image_name)
                has_space, available, required = check_sufficient_space(required_bytes)

                if not has_space:
                    available_gb = available / 1024 ** 3
                    required_gb = required / 1024 ** 3
                    logging.error(
                        f"Insufficient disk space for {service_type}: "
                        f"Need ~{required_gb:.1f} GB, have {available_gb:.1f} GB"
                    )
                    self._emit_disk_space_error(image_name, required_gb, available_gb)
                    self._download_status[service_type] = DownloadStatus.FAILED
                    return False

                self._active_downloads.add(service_type)
                # Create and store a cancellation event for this download
                cancel_event = threading.Event()
                self._cancellation_events[service_type] = cancel_event
                
                # Start download thread (daemon=True so it doesn't block exit)
                thread = threading.Thread(
                    target=self._download_worker,
                    args=(service_type, cancel_event),
                    daemon=True,
                    name=f"docker-download-{service_type}"
                )
                thread.start()
                logging.info(f"Started background download for {service_type}")
                return True
            else:
                # Queue the download
                self._download_queue.append(service_type)
                self._download_status[service_type] = DownloadStatus.PENDING
                logging.info(f"Queued download for {service_type} (max concurrent reached)")
                return True

    def cancel_download(self, service_type: str):
        """
        Cancel an active or queued download.
        
        Args:
            service_type: The service type to cancel download for
        """
        logging.info(f"Request to cancel download for: {service_type}")
        with self._lock:
            # 1. Remove from queue if it's there
            if service_type in self._download_queue:
                self._download_queue.remove(service_type)
                self._download_status[service_type] = DownloadStatus.FAILED
                logging.info(f"Removed {service_type} from download queue")
                return

            # 2. Signal active download if it's running
            if service_type in self._cancellation_events:
                self._cancellation_events[service_type].set()
                logging.info(f"Signaled cancellation for active download: {service_type}")
                # The worker will handle cleanup in finally block when it detects the signal
            elif service_type in self._active_downloads:
                # Edge case: download is active but no cancellation event (shouldn't happen)
                logging.warning(f"Download for {service_type} is active but has no cancellation event")
                self._active_downloads.discard(service_type)
                self._download_status[service_type] = DownloadStatus.FAILED
    
    def _emit_disk_space_error(self, image_name: str, required_gb: float, available_gb: float):
        """Emit a DISK_SPACE_ERROR JSON event so the Electron UI can show an actionable alert."""
        import json, sys
        message = (
            f"Insufficient disk space for '{image_name}'. "
            f"Required: {required_gb:.1f} GB (including 5 GB buffer), "
            f"Available: {available_gb:.1f} GB"
        )
        print(json.dumps({
            "type": "DISK_SPACE_ERROR",
            "payload": {
                "image_name": image_name,
                "required_gb": round(required_gb, 1),
                "available_gb": round(available_gb, 1),
                "message": message,
            }
        }), flush=True)

    def _download_worker(self, service_type: str, cancel_event: threading.Event):
        """Worker function that runs in a background thread to download an image.
        
        Reports download state to server for 3-tier cache priority routing:
        - 'start' -> adds to downloading_images (tier 1 - downloading)
        - 'finish' -> moves to cached_images (tier 0 - cached)
        - 'cancel' -> removes from downloading_images (on failure)
        
        NOTE: This does NOT affect credits. Credits are based on processing
        time and VRAM, not cache state. This is purely for routing efficiency.
        """
        try:
            # Set status to DOWNLOADING immediately
            with self._lock:
                self._download_status[service_type] = DownloadStatus.DOWNLOADING
                logging.info(f"Download worker started for {service_type}, status set to DOWNLOADING")
            
            # Report download start to server (enables tier 1 routing)
            self._report_download_state(service_type, "start")
            
            image_name = self.docker_manager.get_image_name(service_type)
            logging.info(f"Background download starting for image: {image_name}")
            
            # Use the existing pull_image method which handles progress reporting
            self.docker_manager.pull_image(image_name, shutdown_event=cancel_event, service_type=service_type)
            
            with self._lock:
                self._download_status[service_type] = DownloadStatus.COMPLETED
                logging.info(f"Background download completed for {service_type}")
            
            # Report download completion to server (moves to tier 0 - cached)
            # This doesn't affect credits - credits are based on processing time, not caching
            self._report_download_state(service_type, "finish")
                
        except Exception as e:
            # Distinguish permanent failures (image does not exist on registry) from
            # transient ones (network blip, timeout).  A permanent failure must not be
            # retried automatically — only a config/image-name fix can resolve it.
            is_image_not_found = isinstance(e, DockerNotFound) or (
                isinstance(e, DockerAPIError) and getattr(e.response, "status_code", None) == 404
            )
            if is_image_not_found:
                image_name = self.docker_manager.get_image_name(service_type)
                logging.error(
                    f"Image '{image_name}' does not exist on the registry "
                    f"(404). Will not retry until config is corrected."
                )
                with self._lock:
                    self._download_status[service_type] = DownloadStatus.PERMANENTLY_FAILED
            else:
                logging.error(f"Background download failed for {service_type}: {e}")
                with self._lock:
                    self._download_status[service_type] = DownloadStatus.FAILED

            # Report download failure to server (removes from downloading)
            self._report_download_state(service_type, "cancel")

            # Case 2 — disk full MID-DOWNLOAD: Docker raises APIError with "no space left".
            # The pre-download check passes but the disk fills up during the long pull.
            # Emit DISK_SPACE_ERROR so the UI surfaces an actionable alert (not a silent failure).
            err_str = str(e).lower()
            is_disk_full = (
                "no space left" in err_str
                or "disk quota exceeded" in err_str
                or (isinstance(e, OSError) and getattr(e, "errno", None) == 28)
            )
            if is_disk_full:
                try:
                    image_name = self.docker_manager.get_image_name(service_type)
                    from .disk_space_utils import get_available_disk_space
                    available_gb = get_available_disk_space() / 1024 ** 3
                    service_config = (self.docker_manager.services_config or {}).get(service_type, {})
                    required_gb = service_config.get("disk_required_gb", 0)
                    self._emit_disk_space_error(image_name, required_gb, available_gb)
                except Exception:
                    pass

                # Case 4 — prune dangling partial layers to recover the space Docker
                # already wrote before running out. Without this, orphaned overlay2
                # entries stay on disk until the user manually runs "docker image prune".
                try:
                    if self.docker_manager and self.docker_manager.client:
                        logging.info(f"Disk-full failure for {service_type} — pruning dangling layers to recover space")
                        self.docker_manager.client.images.prune()
                except Exception as prune_err:
                    logging.warning(f"Could not prune after disk-full failure: {prune_err}")

        finally:
            # Clean up and start next queued download
            self._finish_download(service_type)
    
    def _report_download_state(self, service_type: str, action: str):
        """Report download state change to server for smart job routing.
        
        This enables 3-tier cache priority in job assignment:
        - Tier 0 (cached): Image ready, can process immediately
        - Tier 1 (downloading): Image being pulled, will be ready soon
        - Tier 2 (miss): Image not available, requires full download
        
        NOTE: This does NOT affect credits. Credits are calculated based on
        actual processing time and VRAM usage, not cache state.
        
        Args:
            service_type: The service type (e.g., 'wan22-12gb')
            action: One of 'start', 'finish', or 'cancel'
        """
        if not self.orchestrator_service or not self.provider_id:
            return
        
        try:
            self.orchestrator_service.report_download_state(
                provider_id=self.provider_id,
                service_type=service_type,
                action=action
            )
            logging.debug(f"Reported download state: {service_type} -> {action}")
        except Exception as e:
            # Non-critical - don't fail if reporting fails
            logging.warning(f"Failed to report download state to server: {e}")
    
    def _report_cached_image(self, service_type: str):
        """Legacy method - now uses report_download_state('finish').
        
        Kept for backward compatibility with existing code.
        """
        self._report_download_state(service_type, "finish")
    
    def _finish_download(self, service_type: str):
        """Clean up after a download finishes and start the next queued download."""
        with self._lock:
            self._active_downloads.discard(service_type)
            
            # Signal wakeup event if provided - this tells the job listener 
            # to check for new processable jobs immediately
            if self.wakeup_event:
                logging.info("Signaling job_wakeup_event after download completion.")
                self.wakeup_event.set()

            # Clean up cancellation event
            if service_type in self._cancellation_events:
                del self._cancellation_events[service_type]
                logging.debug(f"Cleaned up cancellation event for {service_type}")
            
            # Start the next queued download if any.
            # Re-check both image cap and disk space here because the prior download
            # may have changed local cache state significantly.
            while self._download_queue and not self._shutdown:
                next_service = self._download_queue.pop(0)

                if self.has_image(next_service):
                    self._download_status.pop(next_service, None)
                    continue

                if not self._ensure_cache_capacity(next_service):
                    self._download_status[next_service] = DownloadStatus.FAILED
                    self._report_download_state(next_service, "cancel")
                    continue

                next_image_name = self.docker_manager.get_image_name(next_service)
                next_config = (self.docker_manager.services_config or {}).get(next_service, {})
                next_disk_gb = next_config.get("disk_required_gb")
                if next_disk_gb:
                    next_required_bytes = int(next_disk_gb * 1024 ** 3)
                else:
                    next_required_bytes = estimate_image_size_bytes(next_image_name)

                has_space, available, required = check_sufficient_space(next_required_bytes)
                if not has_space:
                    avail_gb = available / 1024 ** 3
                    req_gb = required / 1024 ** 3
                    logging.error(
                        f"Insufficient disk space for queued download {next_service}: "
                        f"need ~{req_gb:.1f} GB, have {avail_gb:.1f} GB"
                    )
                    self._emit_disk_space_error(next_image_name, req_gb, avail_gb)
                    self._download_status[next_service] = DownloadStatus.FAILED
                    self._report_download_state(next_service, "cancel")
                    continue

                self._active_downloads.add(next_service)
                self._download_status[next_service] = DownloadStatus.DOWNLOADING

                # Create and store a cancellation event for this download
                cancel_event = threading.Event()
                self._cancellation_events[next_service] = cancel_event

                thread = threading.Thread(
                    target=self._download_worker,
                    args=(next_service, cancel_event),
                    daemon=True,
                    name=f"docker-download-{next_service}"
                )
                thread.start()
                logging.info(f"Started next queued download for {next_service}")
                break
    
    def get_all_statuses(self) -> Dict[str, str]:
        """Get download status for all tracked service types."""
        with self._lock:
            return {
                service: status.value 
                for service, status in self._download_status.items()
            }
    
    def get_cached_service_types(self, all_service_types: list[str]) -> list[str]:
        """
        Get list of service types that have their Docker images cached locally.
        
        Args:
            all_service_types: List of all known service types to check
            
        Returns:
            List of service types with locally cached images
        """
        if not self.docker_manager:
            return []
        
        cached = []
        for service_type in all_service_types:
            if self.has_image(service_type):
                cached.append(service_type)
        
        return cached
    
    def shutdown(self):
        """
        Signal shutdown to stop accepting new downloads.
        
        Note: Active downloads will continue in Docker daemon even after
        the Python process exits. On next startup, completed downloads
        will be detected, and partial downloads will resume from Docker's
        layer cache.
        """
        logging.info("DockerDownloadManager shutting down")
        with self._lock:
            self._shutdown = True
            self._download_queue.clear()
            # Signal all active downloads to stop
            for event in self._cancellation_events.values():
                event.set()
