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
    """
    
    MAX_CONCURRENT_DOWNLOADS = 1
    
    def __init__(self, docker_manager):
        """
        Initialize the download manager.
        
        Args:
            docker_manager: The DockerProdManager instance for Docker operations
        """
        self.docker_manager = docker_manager
        self._lock = threading.Lock()
        self._active_downloads: Set[str] = set()  # service_types currently downloading
        self._download_queue: list[str] = []  # service_types waiting to download
        self._download_status: Dict[str, DownloadStatus] = {}
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
            
        # Check if already have the image
        if self.has_image(service_type):
            logging.debug(f"Image for {service_type} already exists, skipping download")
            return False
        
        with self._lock:
            # Check if already downloading or queued
            if service_type in self._active_downloads:
                logging.debug(f"Image for {service_type} already downloading")
                return False
            if service_type in self._download_queue:
                logging.debug(f"Image for {service_type} already queued")
                return False
            
            # Check if we can start a new download
            if len(self._active_downloads) < self.MAX_CONCURRENT_DOWNLOADS:
                self._active_downloads.add(service_type)
                self._download_status[service_type] = DownloadStatus.DOWNLOADING
                
                # Start download thread (daemon=True so it doesn't block exit)
                thread = threading.Thread(
                    target=self._download_worker,
                    args=(service_type,),
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
    
    def _download_worker(self, service_type: str):
        """Worker function that runs in a background thread to download an image."""
        try:
            image_name = self.docker_manager.get_image_name(service_type)
            logging.info(f"Background download starting for image: {image_name}")
            
            # Use the existing pull_image method which handles progress reporting
            self.docker_manager.pull_image(image_name)
            
            with self._lock:
                self._download_status[service_type] = DownloadStatus.COMPLETED
                logging.info(f"Background download completed for {service_type}")
                
        except Exception as e:
            logging.error(f"Background download failed for {service_type}: {e}")
            with self._lock:
                self._download_status[service_type] = DownloadStatus.FAILED
        finally:
            # Clean up and start next queued download
            self._finish_download(service_type)
    
    def _finish_download(self, service_type: str):
        """Clean up after a download finishes and start the next queued download."""
        with self._lock:
            self._active_downloads.discard(service_type)
            
            # Start next queued download if any
            if self._download_queue and not self._shutdown:
                next_service = self._download_queue.pop(0)
                self._active_downloads.add(next_service)
                self._download_status[next_service] = DownloadStatus.DOWNLOADING
                
                thread = threading.Thread(
                    target=self._download_worker,
                    args=(next_service,),
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
