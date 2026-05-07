"""
Docker Download Manager

Manages background Docker image downloads with concurrency limiting.
Downloads are performed in daemon threads that don't block app exit.
"""

import json
import os
import sys
import tempfile
import threading
import logging
import time
from typing import Dict, Optional, Set
from enum import Enum
import docker
from services.disk_space_utils import estimate_image_size_bytes, check_sufficient_space
from services import disk_pressure
from config import (
    DISK_PRESSURE_HEALTHY_GB,
    DISK_PRESSURE_CRITICAL_GB,
    DOCKER_IMAGE_CACHE_LIMIT_GB,
)

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


# How long (seconds) a freshly-downloaded image is shielded from LRU eviction.
# This prevents the race where _ensure_cache_capacity evicts an image that was
# just pulled, before the job listener has had a chance to run a job on it.
# Only bypassed when disk pressure reaches CRITICAL (force=True).
FRESH_IMAGE_PROTECTION_SECS = 120

# How long (seconds) to wait after a docker pull completes before signalling
# the job listener wakeup event.  After a large image pull (4-24 GB), Docker
# spends up to ~60 s extracting overlay2 layers during which:
#   - Disk stays at 100% I/O
#   - The Docker TCP API (port 2375) intermittently refuses connections
# Waking the job listener immediately causes image-availability checks to return
# UNKNOWN (WinError 10061), the job is skipped, and the client stalls until the
# next poll cycle.  A 30-second settle window is enough on most hardware while
# still being short relative to normal poll intervals.
POST_DOWNLOAD_SETTLE_SECS = 30


class DownloadStatus(Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    PERMANENTLY_FAILED = "permanently_failed"  # e.g. image does not exist on registry


class ImageAvailability(Enum):
    AVAILABLE = "available"
    MISSING = "missing"
    UNKNOWN = "unknown"


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

    # Cache metadata file format version — bump if schema changes.
    _CACHE_METADATA_VERSION = 1

    def __init__(
        self,
        docker_manager,
        orchestrator_service=None,
        provider_id=None,
        wakeup_event=None,
        cache_limit_gb: Optional[float] = None,
        community_mode: str = "all",
        monetize_mode: bool = False,
        data_dir: Optional[str] = None,
    ):
        """
        Initialize the download manager.

        Args:
            docker_manager: The DockerProdManager instance for Docker operations
            orchestrator_service: Optional OrchestratorService for reporting cached images
            provider_id: Optional provider ID for server-side tracking
            wakeup_event: Optional threading.Event to set when a download completes
            cache_limit_gb: Optional storage budget for locally cached OpenFork
                Docker images. When set, least-recently-used images are evicted
                before starting a new download so disk usage stays bounded.
            community_mode: Provider's current community routing mode. Used by
                logging and compatibility with routing config hot reloads.
            monetize_mode: True when the provider is currently in monetize mode.
            data_dir: Path of the OpenFork data directory. Cache metadata
                (per-image last-used timestamps) is persisted here so LRU
                ordering survives client restarts. When None, persistence is
                disabled and LRU stays session-local.
        """
        self.docker_manager = docker_manager
        self.orchestrator_service = orchestrator_service
        self.provider_id = provider_id
        self.wakeup_event = wakeup_event
        effective_cache_limit_gb = (
            cache_limit_gb
            if cache_limit_gb is not None
            else DOCKER_IMAGE_CACHE_LIMIT_GB
        )
        self.cache_limit_bytes = (
            int(float(effective_cache_limit_gb) * 1024 ** 3)
            if effective_cache_limit_gb and float(effective_cache_limit_gb) > 0
            else None
        )
        self.community_mode = community_mode or "all"
        self.monetize_mode = bool(monetize_mode)
        self._lock = threading.RLock()
        self._active_downloads: Set[str] = set()  # service_types currently downloading
        self._download_queue: list[str] = []  # service_types waiting to download
        self._download_status: Dict[str, DownloadStatus] = {}
        self._cancellation_events: Dict[str, threading.Event] = {}
        self._last_job_times: Dict[str, float] = {}
        # Timestamps of successfully completed downloads (session-local, not persisted).
        # Used by _evict_lru_image to shield fresh images from immediate eviction.
        self._recently_downloaded: Dict[str, float] = {}
        # Job-active gate: when True, no new downloads start and queue is frozen.
        self._job_active: bool = False
        # Service types whose downloads were cancelled because a job started;
        # re-inserted at the front of the queue when the job finishes.
        self._paused_downloads: Set[str] = set()
        self._shutdown = False

        # Persisted LRU metadata
        self._cache_metadata_path: Optional[str] = None
        if data_dir:
            try:
                os.makedirs(data_dir, exist_ok=True)
                self._cache_metadata_path = os.path.join(data_dir, "cache_metadata.json")
                self._load_cache_metadata()
            except OSError as e:
                logging.warning(f"Could not initialise cache metadata at {data_dir}: {e}")
                self._cache_metadata_path = None

        # Disk pressure tracking
        self._last_disk_tier: Optional[str] = None
        self._last_pressure_check_ts: float = 0.0
        
    def _check_image_once(
        self,
        docker_client,
        image_name: str,
    ) -> tuple[ImageAvailability, Optional[Exception]]:
        try:
            docker_client.images.get(image_name)
            return ImageAvailability.AVAILABLE, None
        except DockerImageNotFound:
            return ImageAvailability.MISSING, None
        except Exception as e:
            return ImageAvailability.UNKNOWN, e

    def get_image_availability(self, service_type: str) -> ImageAvailability:
        """
        Check Docker image availability for a service.

        Returns AVAILABLE only when the image can be verified locally.
        Transient Docker transport failures are reported as UNKNOWN so callers
        do not incorrectly treat a cached image as missing and trigger a pull.
        """
        if not self.docker_manager:
            return ImageAvailability.AVAILABLE  # Headless mode - no Docker needed

        docker_client = getattr(self.docker_manager, "client", None)
        if not docker_client:
            logging.warning(
                f"Docker client unavailable while checking image for {service_type}."
            )
            return ImageAvailability.UNKNOWN

        try:
            image_name = self.docker_manager.get_image_name(service_type)
        except Exception as e:
            logging.warning(f"Could not resolve image name for {service_type}: {e}")
            return ImageAvailability.UNKNOWN

        availability, error = self._check_image_once(docker_client, image_name)
        if error is None:
            return availability

        is_transient = False
        is_transient_transport_error = getattr(
            self.docker_manager, "_is_transient_transport_error", None
        )
        if callable(is_transient_transport_error):
            try:
                is_transient = is_transient_transport_error(error)
            except Exception:
                is_transient = False

        if is_transient:
            logging.warning(
                f"Transient Docker error checking image for {service_type}: {error}"
            )
            refresh_client_connection = getattr(
                self.docker_manager, "_refresh_client_connection", None
            )
            if callable(refresh_client_connection) and refresh_client_connection():
                refreshed_client = getattr(self.docker_manager, "client", None)
                if refreshed_client:
                    retry_availability, retry_error = self._check_image_once(
                        refreshed_client, image_name
                    )
                    if retry_error is None:
                        return retry_availability
                    error = retry_error

        logging.warning(f"Error checking image for {service_type}: {error}")
        return ImageAvailability.UNKNOWN

    def has_image(self, service_type: str) -> bool:
        """
        Return True only when the Docker image is confirmed to exist locally.
        """
        return self.get_image_availability(service_type) == ImageAvailability.AVAILABLE

    def notify_job_complete(self, service_type: str):
        """Record that a service type was actually used to process a job."""
        if not service_type:
            return

        with self._lock:
            self._last_job_times[service_type] = time.time()
            self._save_cache_metadata_locked()

    def set_job_active(self, active: bool) -> None:
        """Suspend or resume background downloads around an active job.

        When True: cancels all in-flight downloads so their disk I/O and the
        post-download Docker layer-extraction step don't compete with the
        running workflow.  Docker layer caching means the next pull will skip
        already-transferred layers, so progress is not fully lost.

        When False: re-queues any downloads that were cancelled for the job
        pause and kick-starts the queue.  This is idempotent — safe to call
        even if set_job_active(False) was already the current state.
        """
        next_to_start = None
        with self._lock:
            if self._job_active == active:
                return

            self._job_active = active

            if active:
                # Cancel every in-flight download.
                for st, cancel_event in list(self._cancellation_events.items()):
                    cancel_event.set()
                    self._paused_downloads.add(st)
                    logging.info(
                        f"Suspended download for '{st}' — job started; will resume after."
                    )
            else:
                # Re-insert paused downloads at the front of the queue so they
                # restart first, in their original order.
                paused = list(self._paused_downloads)
                self._paused_downloads.clear()
                for st in reversed(paused):
                    if st not in self._active_downloads and st not in self._download_queue:
                        self._download_queue.insert(0, st)
                        self._download_status.pop(st, None)  # Clear FAILED → allow retry
                        logging.info(f"Re-queued suspended download for '{st}' — job finished.")

                # Kick-start the queue if nothing is currently downloading.
                if (
                    self._download_queue
                    and not self._shutdown
                    and len(self._active_downloads) < self.MAX_CONCURRENT_DOWNLOADS
                ):
                    next_to_start = self._download_queue.pop(0)
                    self._download_status.pop(next_to_start, None)

        # Start outside the lock so start_background_download's own lock acquisition
        # is not a reentrant re-lock (still safe with RLock, but cleaner outside).
        if next_to_start is not None:
            logging.info(
                f"Kick-starting queued download for '{next_to_start}' — job finished."
            )
            self.start_background_download(next_to_start)

    # ── LRU persistence ───────────────────────────────────────────────────

    def _load_cache_metadata(self) -> None:
        """Load `_last_job_times` from disk. Silently ignores corrupted or absent files."""
        path = self._cache_metadata_path
        if not path or not os.path.exists(path):
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logging.warning(f"Could not read cache metadata at {path}: {e}. Starting fresh.")
            return

        if not isinstance(data, dict):
            return

        # Ignore version sentinel + non-numeric values; everything else is a timestamp.
        loaded = 0
        for key, value in data.items():
            if key == "_version":
                continue
            if isinstance(value, (int, float)) and value > 0:
                self._last_job_times[key] = float(value)
                loaded += 1
        if loaded:
            logging.info(f"Loaded {loaded} LRU entries from cache metadata.")

    def _save_cache_metadata_locked(self) -> None:
        """Atomically persist `_last_job_times`. Caller must hold `self._lock`."""
        path = self._cache_metadata_path
        if not path:
            return

        payload = {"_version": self._CACHE_METADATA_VERSION}
        payload.update(self._last_job_times)

        # Atomic write: temp file in the same directory, then rename.
        directory = os.path.dirname(path) or "."
        try:
            fd, tmp_path = tempfile.mkstemp(prefix="cache_metadata.", suffix=".tmp", dir=directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                os.replace(tmp_path, path)
            except Exception:
                # Best effort cleanup of the temp file
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as e:
            logging.debug(f"Could not persist cache metadata: {e}")

    # ── Routing config / storage budget ──────────────────────────────────

    def update_routing_config(
        self, community_mode: str, monetize_mode: bool
    ) -> None:
        """Hot-reload policy state used by routing-aware download decisions."""
        with self._lock:
            self.community_mode = community_mode or "all"
            self.monetize_mode = bool(monetize_mode)
            tier = disk_pressure.get_disk_pressure_tier()
            policy = disk_pressure.policy_key_for(self.community_mode, self.monetize_mode)
            logging.debug(
                f"Routing config applied to download manager: "
                f"policy={policy} tier={tier} "
                f"cache_limit_gb={self._cache_limit_gb_display()}"
            )

    def _current_policy_key(self) -> str:
        return disk_pressure.policy_key_for(self.community_mode, self.monetize_mode)

    def _cache_limit_gb_display(self) -> str:
        if not self.cache_limit_bytes:
            return "unlimited"
        return f"{self.cache_limit_bytes / 1024 ** 3:.0f}"

    def update_cache_limit_gb(self, cache_limit_gb: Optional[float]) -> None:
        """Hot-reload the user-facing Docker image storage budget."""
        with self._lock:
            if cache_limit_gb and float(cache_limit_gb) > 0:
                self.cache_limit_bytes = int(float(cache_limit_gb) * 1024 ** 3)
            else:
                self.cache_limit_bytes = None
            limit_label = self._cache_limit_gb_display()
            suffix = "" if limit_label == "unlimited" else " GB"
            logging.info(f"Docker image cache limit updated: {limit_label}{suffix}")

        self._evict_until_within_cache_limit(reason="storage_limit")

    # ── Disk-pressure-driven eviction ─────────────────────────────────────

    def check_and_evict_for_pressure(self) -> int:
        """
        Idle-loop hook: compute current disk tier, enforce the user's cache
        budget, and evict LRU images to bring disk back above the Pressure
        threshold when at Critical. Returns the number of images evicted on
        this call.

        Cheap when at Healthy unless the cache is over budget. Eviction skips
        running, downloading, queued, and freshly downloaded images.
        """
        if self._shutdown:
            return 0
        if not self.docker_manager or not getattr(self.docker_manager, "client", None):
            return 0

        tier = disk_pressure.get_disk_pressure_tier()
        free_gb = disk_pressure.get_free_disk_gb()
        disk_pressure.log_tier_transition(self._last_disk_tier, tier, free_gb)
        self._last_disk_tier = tier
        self._last_pressure_check_ts = time.time()

        if tier == disk_pressure.HEALTHY:
            # Honor the user's cache budget but don't do pressure cleanup.
            return self._evict_until_within_cache_limit()

        # PRESSURE / CRITICAL — additionally try to claw back to Healthy / Pressure.
        evicted = self._evict_until_within_cache_limit()

        if tier == disk_pressure.CRITICAL:
            # Aggressive eviction: keep going until free space exceeds Pressure threshold,
            # or we run out of evictable images. Each eviction may take a few seconds
            # (docker rmi + container removal) so we cap the loop to avoid wedging.
            # force=True bypasses freshness protection — disk safety takes priority.
            target_free_gb = max(DISK_PRESSURE_HEALTHY_GB, DISK_PRESSURE_CRITICAL_GB + 5)
            for _ in range(10):
                current_free = disk_pressure.get_free_disk_gb()
                if current_free >= target_free_gb:
                    break
                victim = self._evict_lru_image(reason="disk_critical", force=True)
                if not victim:
                    break
                evicted += 1

        return evicted

    def _evict_until_within_cache_limit(
        self,
        reason: str = "storage_limit",
    ) -> int:
        """Evict LRU images while cached OpenFork images exceed the GB budget."""
        if not self.cache_limit_bytes:
            return 0

        evicted = 0
        with self._lock:
            while self._get_cache_usage_bytes() > self.cache_limit_bytes:
                victim = self._evict_lru_image(reason=reason)
                if not victim:
                    break
                evicted += 1
        return evicted

    @staticmethod
    def _emit_image_evicted(service_type: str, image_name: str, freed_bytes: int, reason: str) -> None:
        """Print a JSON event so the Electron auto-compact manager can react."""
        payload = {
            "type": "IMAGE_EVICTED",
            "payload": {
                "service_type": service_type,
                "image": image_name,
                "freed_bytes": int(freed_bytes),
                "reason": reason,
            },
        }
        try:
            print(json.dumps(payload), flush=True)
        except Exception:
            # Never fail eviction because stdout is gone (test harnesses, etc.).
            pass

    @staticmethod
    def _emit_download_state(
        service_type: str,
        image_name: str,
        status: str,
    ) -> None:
        """
        Emit coarse download lifecycle state for Electron-side idle detection.

        Docker pull progress only starts once the worker reaches Docker. This
        event is emitted earlier so auto-compact can see accepted/queued
        downloads before image eviction opens a false idle window.
        """
        payload = {
            "type": "DOCKER_DOWNLOAD_STATE",
            "payload": {
                "service_type": service_type,
                "image": image_name,
                "status": status,
            },
        }
        try:
            print(json.dumps(payload), flush=True)
        except Exception:
            pass

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
            if self.get_image_availability(service_type) == ImageAvailability.AVAILABLE:
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

    def sync_cached_images_with_server(self):
        """Public wrapper used when Electron manually removes Docker images."""
        self._sync_cached_images_with_server()

    def _lookup_image_size_bytes(self, image_name: str) -> int:
        """Best-effort image size lookup. Falls back to the heuristic estimate."""
        try:
            client = getattr(self.docker_manager, "client", None)
            if client:
                image = client.images.get(image_name)
                size = getattr(image, "attrs", {}).get("Size")
                if isinstance(size, (int, float)) and size > 0:
                    return int(size)
        except Exception:
            pass
        return estimate_image_size_bytes(image_name)

    def _get_service_required_bytes(self, service_type: str) -> int:
        """Return the expected disk footprint for a service image."""
        service_config = (getattr(self.docker_manager, "services_config", None) or {}).get(
            service_type,
            {},
        )
        disk_required_gb = service_config.get("disk_required_gb")
        if isinstance(disk_required_gb, (int, float)) and disk_required_gb > 0:
            return int(float(disk_required_gb) * 1024 ** 3)

        image_name = self.docker_manager.get_image_name(service_type)
        return estimate_image_size_bytes(image_name)

    def _get_cached_image_size_bytes(self, service_type: str) -> int:
        """Return actual cached image size, falling back to configured requirement."""
        image_name = self.docker_manager.get_image_name(service_type)
        try:
            client = getattr(self.docker_manager, "client", None)
            if client:
                image = client.images.get(image_name)
                size = getattr(image, "attrs", {}).get("Size")
                if isinstance(size, (int, float)) and size > 0:
                    return int(size)
        except Exception:
            pass
        return self._get_service_required_bytes(service_type)

    def _get_cache_usage_bytes(
        self,
        exclude_service_types: Optional[Set[str]] = None,
    ) -> int:
        """Return total bytes used by cached OpenFork service images."""
        exclude = exclude_service_types or set()
        total = 0
        for service_type in self._get_cached_service_types():
            if service_type in exclude:
                continue
            total += self._get_cached_image_size_bytes(service_type)
        return total

    def service_fits_cache_budget(self, service_type: str) -> bool:
        """Whether one service image can fit inside the configured cache budget."""
        if not self.cache_limit_bytes:
            return True
        try:
            return self._get_service_required_bytes(service_type) <= self.cache_limit_bytes
        except Exception:
            return True

    def _evict_lru_image(
        self,
        exclude_service_types: Optional[Set[str]] = None,
        reason: str = "storage_limit",
        force: bool = False,
    ) -> Optional[str]:
        """
        Evict one cached image using session-local LRU ordering.

        Eviction priority:
        1. Cached images with no completed jobs this session (prefetched but unused)
        2. Otherwise, the image with the oldest last completed job time

        Images are never evicted if they are currently running, downloading, or queued.
        Images downloaded within FRESH_IMAGE_PROTECTION_SECS are also shielded unless
        force=True (reserved for CRITICAL disk pressure).

        On success, emits an `IMAGE_EVICTED` JSON event with the freed bytes so the
        Electron auto-compact manager can decide when to schedule VHDX compaction.
        """
        if not self.docker_manager or not getattr(self.docker_manager, "client", None):
            return None

        exclude = exclude_service_types or set()

        # Shield images that finished downloading recently so the job listener has
        # time to pick up a job before LRU eviction can remove the image.  This
        # prevents the race: download completes → next queued download evicts the
        # fresh image → job listener wakes up to find the image already gone.
        now = time.time()
        freshly_downloaded: Set[str] = set() if force else {
            st for st, ts in self._recently_downloaded.items()
            if now - ts < FRESH_IMAGE_PROTECTION_SECS
        }

        busy_service_types = (
            set(self._active_downloads)
            | set(self._download_queue)
            | self._get_running_service_types()
            | freshly_downloaded
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
        freed_bytes = self._lookup_image_size_bytes(image_name)

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
            self._emit_image_evicted(
                service_type=service_type_to_evict,
                image_name=image_name,
                freed_bytes=freed_bytes,
                reason=reason,
            )
            logging.info(
                f"Evicted cached image for {service_type_to_evict} "
                f"(reason={reason}, freed~{freed_bytes / 1024**3:.1f} GB)."
            )
            return service_type_to_evict
        except DockerImageNotFound:
            # Image already gone; still surface the cleanup so the compaction tracker
            # doesn't think nothing happened. Use 0 freed bytes since we can't measure.
            self._download_status.pop(service_type_to_evict, None)
            self._sync_cached_images_with_server()
            self._emit_image_evicted(
                service_type=service_type_to_evict,
                image_name=image_name,
                freed_bytes=0,
                reason=reason,
            )
            return service_type_to_evict
        except Exception as e:
            logging.warning(f"Failed to evict cached image for {service_type_to_evict}: {e}")
            return None

    def _ensure_cache_capacity(self, incoming_service_type: str) -> bool:
        """Evict LRU images until a new download fits under the storage budget."""
        if not self.cache_limit_bytes:
            return True

        cached_service_types = self._get_cached_service_types()
        if incoming_service_type in cached_service_types:
            return True

        try:
            incoming_required = self._get_service_required_bytes(incoming_service_type)
            incoming_image = self.docker_manager.get_image_name(incoming_service_type)
        except Exception as e:
            logging.warning(
                f"Could not calculate cache budget for {incoming_service_type}: {e}"
            )
            return True

        if incoming_required > self.cache_limit_bytes:
            required_gb = incoming_required / 1024 ** 3
            limit_gb = self.cache_limit_bytes / 1024 ** 3
            message = (
                f"OpenFork storage limit is too small for '{incoming_image}'. "
                f"This image needs about {required_gb:.1f} GB, but your Docker "
                f"image limit is {limit_gb:.1f} GB. Increase the limit in "
                "Docker Management settings to use this model."
            )
            logging.warning(message)
            self._emit_disk_space_error(
                incoming_image,
                required_gb=required_gb,
                available_gb=limit_gb,
                message=message,
            )
            return False

        while (
            self._get_cache_usage_bytes(exclude_service_types={incoming_service_type})
            + incoming_required
            > self.cache_limit_bytes
        ):
            evicted_service = self._evict_lru_image(
                exclude_service_types={incoming_service_type},
                reason="storage_limit",
            )
            if not evicted_service:
                current_gb = self._get_cache_usage_bytes() / 1024 ** 3
                incoming_gb = incoming_required / 1024 ** 3
                limit_gb = self.cache_limit_bytes / 1024 ** 3
                message = (
                    f"OpenFork needs about {incoming_gb:.1f} GB for "
                    f"'{incoming_image}', but the current image cache is "
                    f"{current_gb:.1f} GB and the limit is {limit_gb:.1f} GB. "
                    "No cached image can be safely removed right now; try again "
                    "after the current job/download finishes or increase the limit."
                )
                logging.info(
                    "Storage limit reached but no evictable candidate found for "
                    f"'{incoming_service_type}'."
                )
                self._emit_disk_space_error(
                    incoming_image,
                    required_gb=incoming_gb,
                    available_gb=max(0.0, limit_gb - current_gb),
                    message=message,
                )
                return False

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

    def _claim_download_slot(
        self,
        service_type: str,
        accept_policy: Optional[str] = None,
    ) -> bool:
        """
        Ask the orchestrator whether this provider should download this image.

        The server performs the network-wide cache-deficit check atomically for
        public/monetize pools. If the server is unreachable we fail open so a
        transient API problem cannot strand pending jobs forever.
        """
        if not self.orchestrator_service or not self.provider_id:
            return True

        accepted = self.orchestrator_service.report_download_state(
            provider_id=self.provider_id,
            service_type=service_type,
            action="start",
            accept_policy=accept_policy,
            return_none_on_error=True,
        )
        if accepted is None:
            logging.warning(
                f"Could not claim download slot for {service_type}; allowing download."
            )
            return True
        if not accepted:
            logging.info(
                f"Skipping download for {service_type}; network coverage is already sufficient."
            )
            return False
        return True
     
    def start_background_download(
        self,
        service_type: str,
        accept_policy: Optional[str] = None,
    ) -> bool:
        """
        Start downloading the Docker image for a service in the background.
        
        If already downloading or queued, this is a no-op.
        If at max concurrent downloads, queues the download.
        
        Args:
            service_type: The service type to download image for
            accept_policy: Optional job routing policy for this download. Own
                and trusted-policy jobs bypass global public-pool coverage.
            
        Returns:
            True if download was started or queued, False if already in progress
        """
        if self._shutdown:
            return False
            
        if not self.docker_manager:
            return False  # Headless mode
        
        with self._lock:
            if self._shutdown:
                return False

            # Check if already downloading or queued
            if service_type in self._active_downloads:
                logging.debug(f"Image for {service_type} already downloading")
                return False
            if service_type in self._download_queue:
                logging.debug(f"Image for {service_type} already queued")
                return False
            
            # TOCTOU fix: Check inside lock to prevent race condition
            # where multiple threads could pass the has_image check simultaneously
            availability = self.get_image_availability(service_type)
            if availability == ImageAvailability.AVAILABLE:
                logging.debug(f"Image for {service_type} already exists, skipping download")
                # Clear failed status if image exists now
                self._download_status.pop(service_type, None)
                return False
            if availability == ImageAvailability.UNKNOWN:
                logging.info(
                    f"Deferring background download for {service_type} because Docker image "
                    "availability could not be verified."
                )
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

            try:
                image_name = self.docker_manager.get_image_name(service_type)
            except Exception as e:
                logging.warning(f"Could not resolve image name for {service_type}: {e}")
                return False

            # While a job is processing, defer all new downloads to avoid
            # competing for disk I/O and CPU (Docker layer extraction).
            # The deferred items are restarted by set_job_active(False).
            if self._job_active:
                if service_type not in self._download_queue:
                    self._download_queue.append(service_type)
                    self._download_status[service_type] = DownloadStatus.PENDING
                    self._emit_download_state(service_type, image_name, "queued")
                    logging.info(
                        f"Deferred download for '{service_type}': job is actively processing."
                    )
                return True

            # Check if we can start a new download
            if len(self._active_downloads) < self.MAX_CONCURRENT_DOWNLOADS:
                self._emit_download_state(service_type, image_name, "starting")

                if not self._ensure_cache_capacity(service_type):
                    self._download_status[service_type] = DownloadStatus.FAILED
                    self._emit_download_state(service_type, image_name, "failed")
                    return False

                # Check for sufficient disk space before starting.
                # Prefer disk_required_gb from services_config (authoritative, from services.json)
                # over the hardcoded estimate map, which can be significantly off (e.g. wan22-24gb:
                # estimate=120 GB, actual disk_required_gb=220 GB).
                required_bytes = self._get_service_required_bytes(service_type)
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
                    self._emit_download_state(service_type, image_name, "failed")
                    return False

                if not self._claim_download_slot(service_type, accept_policy):
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
                if not self._claim_download_slot(service_type, accept_policy):
                    return False

                # Queue the download
                self._download_queue.append(service_type)
                self._download_status[service_type] = DownloadStatus.PENDING
                self._emit_download_state(service_type, image_name, "queued")
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
                self._report_download_state(service_type, "cancel")
                try:
                    image_name = self.docker_manager.get_image_name(service_type)
                    self._emit_download_state(service_type, image_name, "cancelled")
                except Exception:
                    pass
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
    
    def _emit_disk_space_error(
        self,
        image_name: str,
        required_gb: float,
        available_gb: float,
        message: Optional[str] = None,
    ):
        """Emit a DISK_SPACE_ERROR JSON event so the Electron UI can show an actionable alert."""
        import json
        if message is None:
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
        image_name = None
        try:
            image_name = self.docker_manager.get_image_name(service_type)

            # Set status to DOWNLOADING immediately, unless shutdown/cancel won
            # the race before this worker had a chance to touch Docker.
            with self._lock:
                if self._shutdown or cancel_event.is_set():
                    self._download_status[service_type] = DownloadStatus.FAILED
                    self._emit_download_state(service_type, image_name, "cancelled")
                    logging.info(
                        f"Download worker for {service_type} exiting before pull "
                        "because shutdown/cancellation was requested."
                    )
                    return
                self._download_status[service_type] = DownloadStatus.DOWNLOADING
                self._emit_download_state(service_type, image_name, "downloading")
                logging.info(f"Download worker started for {service_type}, status set to DOWNLOADING")
            
            # Report download start to server (enables tier 1 routing)
            self._report_download_state(service_type, "start")
            
            logging.info(f"Background download starting for image: {image_name}")
            
            # Use the existing pull_image method which handles progress reporting
            self.docker_manager.pull_image(image_name, shutdown_event=cancel_event, service_type=service_type)
            
            with self._lock:
                self._download_status[service_type] = DownloadStatus.COMPLETED
                self._recently_downloaded[service_type] = time.time()
                self._emit_download_state(service_type, image_name, "completed")
                logging.info(f"Background download completed for {service_type}")
            
            # Report download completion to server (moves to tier 0 - cached)
            # This doesn't affect credits - credits are based on processing time, not caching
            self._report_download_state(service_type, "finish")

            # The configured disk_required_gb is conservative; actual Docker
            # image size can still push the cache over the user's budget after
            # extraction. Evict older images immediately when possible.
            self._evict_until_within_cache_limit(reason="storage_limit")
                
        except Exception as e:
            # Distinguish permanent failures (image does not exist on registry) from
            # transient ones (network blip, timeout).  A permanent failure must not be
            # retried automatically — only a config/image-name fix can resolve it.
            is_image_not_found = isinstance(e, DockerNotFound) or (
                isinstance(e, DockerAPIError) and getattr(e.response, "status_code", None) == 404
            )
            if is_image_not_found:
                image_name = image_name or self.docker_manager.get_image_name(service_type)
                logging.error(
                    f"Image '{image_name}' does not exist on the registry "
                    f"(404). Will not retry until config is corrected."
                )
                with self._lock:
                    self._download_status[service_type] = DownloadStatus.PERMANENTLY_FAILED
                    self._emit_download_state(service_type, image_name, "failed")
            else:
                logging.error(f"Background download failed for {service_type}: {e}")
                with self._lock:
                    self._download_status[service_type] = DownloadStatus.FAILED
                    terminal_status = (
                        "cancelled"
                        if self._shutdown or cancel_event.is_set()
                        else "failed"
                    )
                    if image_name:
                        self._emit_download_state(
                            service_type,
                            image_name,
                            terminal_status,
                        )

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
    
    def _signal_wakeup_after_settle(self) -> None:
        """Signal the job wakeup event after a short settle delay.

        Large docker pulls leave the Docker daemon busy with overlay2 layer
        extraction for ~30-60 s.  During that window the TCP API returns
        ECONNREFUSED (WinError 10061), so any immediate image-availability
        check returns UNKNOWN.  We sleep briefly in a daemon thread so the
        job listener wakes up once Docker has settled.
        """
        shutdown = self._shutdown  # snapshot before sleeping
        if not shutdown:
            logging.info(
                f"Scheduling job_wakeup_event signal in {POST_DOWNLOAD_SETTLE_SECS}s "
                "to allow Docker overlay2 extraction to settle."
            )
            time.sleep(POST_DOWNLOAD_SETTLE_SECS)
        if self.wakeup_event and not self._shutdown:
            logging.info("Signaling job_wakeup_event after post-download settle period.")
            self.wakeup_event.set()

    def _finish_download(self, service_type: str):
        """Clean up after a download finishes and start the next queued download."""
        with self._lock:
            self._active_downloads.discard(service_type)

            # Signal wakeup event after a settle delay so Docker has time to
            # finish overlay2 layer extraction before the job listener checks
            # image availability.  Run in a daemon thread so we don't block
            # the download worker.
            if self.wakeup_event:
                logging.info(
                    "Scheduling delayed job_wakeup_event after download completion."
                )
                t = threading.Thread(
                    target=self._signal_wakeup_after_settle,
                    daemon=True,
                    name="download-settle-wakeup",
                )
                t.start()

            # Clean up cancellation event
            if service_type in self._cancellation_events:
                del self._cancellation_events[service_type]
                logging.debug(f"Cleaned up cancellation event for {service_type}")
            
            # Start the next queued download if any.
            # Re-check both cache budget and disk space here because the prior download
            # may have changed local cache state significantly.
            while self._download_queue and not self._shutdown:
                if self._job_active:
                    # Don't chain the next download while a job is processing;
                    # set_job_active(False) will kick the queue when the job ends.
                    logging.debug(
                        "Skipping next queued download: job is actively processing."
                    )
                    break
                next_service = self._download_queue.pop(0)

                availability = self.get_image_availability(next_service)
                if availability == ImageAvailability.AVAILABLE:
                    self._download_status.pop(next_service, None)
                    self._report_download_state(next_service, "finish")
                    try:
                        next_image_name = self.docker_manager.get_image_name(next_service)
                        self._emit_download_state(next_service, next_image_name, "completed")
                    except Exception:
                        pass
                    continue
                if availability == ImageAvailability.UNKNOWN:
                    self._download_queue.insert(0, next_service)
                    self._download_status[next_service] = DownloadStatus.PENDING
                    logging.info(
                        f"Deferring queued download for {next_service} because Docker image "
                        "availability could not be verified."
                    )
                    break

                if not self._ensure_cache_capacity(next_service):
                    self._download_status[next_service] = DownloadStatus.FAILED
                    try:
                        next_image_name = self.docker_manager.get_image_name(next_service)
                        self._emit_download_state(next_service, next_image_name, "failed")
                    except Exception:
                        pass
                    self._report_download_state(next_service, "cancel")
                    continue

                next_image_name = self.docker_manager.get_image_name(next_service)
                next_required_bytes = self._get_service_required_bytes(next_service)

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
                    self._emit_download_state(next_service, next_image_name, "failed")
                    self._report_download_state(next_service, "cancel")
                    continue

                self._active_downloads.add(next_service)
                self._download_status[next_service] = DownloadStatus.DOWNLOADING
                self._emit_download_state(next_service, next_image_name, "starting")

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
            if self.get_image_availability(service_type) == ImageAvailability.AVAILABLE:
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
            queued_services = list(self._download_queue)
            self._download_queue.clear()
            for service_type in queued_services:
                self._report_download_state(service_type, "cancel")
                try:
                    image_name = self.docker_manager.get_image_name(service_type)
                    self._emit_download_state(service_type, image_name, "cancelled")
                except Exception:
                    pass
            # Signal all active downloads to stop
            for event in self._cancellation_events.values():
                event.set()
