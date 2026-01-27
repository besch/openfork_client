"""
Docker Download Manager

Manages background Docker image downloads with concurrency limiting.
Downloads are performed in daemon threads that don't block app exit.
"""

import threading
import logging
from typing import Dict, Optional, Set
from enum import Enum
import docker


class DownloadStatus(Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"


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
    
    def __init__(self, docker_manager, orchestrator_service=None, provider_id=None):
        """
        Initialize the download manager.
        
        Args:
            docker_manager: The DockerProdManager instance for Docker operations
            orchestrator_service: Optional OrchestratorService for reporting cached images
            provider_id: Optional provider ID for server-side tracking
        """
        self.docker_manager = docker_manager
        self.orchestrator_service = orchestrator_service
        self.provider_id = provider_id
        self._lock = threading.Lock()
        self._active_downloads: Set[str] = set()  # service_types currently downloading
        self._download_queue: list[str] = []  # service_types waiting to download
        self._download_status: Dict[str, DownloadStatus] = {}
        self._cancellation_events: Dict[str, threading.Event] = {}
        self._shutdown = False
        
    def has_image(self, service_type: str) -> bool:
        """
        Check if Docker image for a service exists locally.
        
        Args:
            service_type: The service type (e.g., 'wan22', 'hunyuan')
            
        Returns:
            True if image exists locally, False otherwise
        """
        if not self.docker_manager:
            return True  # Headless mode - no Docker needed
            
        try:
            image_name = self.docker_manager.get_image_name(service_type)
            self.docker_manager.client.images.get(image_name)
            return True
        except docker.errors.ImageNotFound:
            return False
        except Exception as e:
            logging.warning(f"Error checking image for {service_type}: {e}")
            return False
    
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
            
            # Clear any previous FAILED status to allow retry
            # This is important for resuming after a cancelled download
            if service_type in self._download_status:
                prev_status = self._download_status[service_type]
                if prev_status == DownloadStatus.FAILED:
                    logging.info(f"Clearing previous FAILED status for {service_type} to allow retry")
                    del self._download_status[service_type]
            
            # Check if we can start a new download
            if len(self._active_downloads) < self.MAX_CONCURRENT_DOWNLOADS:
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
            logging.error(f"Background download failed for {service_type}: {e}")
            with self._lock:
                self._download_status[service_type] = DownloadStatus.FAILED
            # Report download failure to server (removes from downloading)
            self._report_download_state(service_type, "cancel")
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
            
            # Clean up cancellation event
            if service_type in self._cancellation_events:
                del self._cancellation_events[service_type]
                logging.debug(f"Cleaned up cancellation event for {service_type}")
            
            # Start next queued download if any
            if self._download_queue and not self._shutdown:
                next_service = self._download_queue.pop(0)
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
