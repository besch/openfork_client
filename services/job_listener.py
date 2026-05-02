import logging
import json
import random
import threading
import time
import traceback
from typing import Dict, Any, Optional

from config import HEADLESS_MODE, TimeoutConfig
from services.docker_manager import docker_manager
from services.container_monitor import ContainerMonitor
from services.docker_download_manager import ImageAvailability
from exceptions import AuthError, InfrastructureError, ProviderError


# Cap on backoff so a long outage still polls every 5 minutes.
JOB_POLL_BACKOFF_CAP_SECONDS = 300
TERMINAL_JOB_STATUSES = ("completed", "failed", "cancelled", "deleted")
REMOTE_CANCELLATION_STATUSES = ("cancelled", "deleted")


def _compute_poll_interval(base_seconds: int, consecutive_empty_polls: int) -> float:
    """Exponential backoff with jitter for empty job polls.

    Reasoning: a fleet of clients all polling every 10 s during an orchestrator
    outage is a thundering herd. Backing off (capped at 5 min, ±20% jitter)
    spreads load and avoids hammering the API while still recovering quickly
    once jobs reappear.
    """
    if consecutive_empty_polls <= 0:
        return float(base_seconds)
    multiplier = min(1.5 ** consecutive_empty_polls, JOB_POLL_BACKOFF_CAP_SECONDS / max(base_seconds, 1))
    interval = base_seconds * multiplier
    jitter = interval * random.uniform(-0.2, 0.2)
    return max(float(base_seconds), min(interval + jitter, float(JOB_POLL_BACKOFF_CAP_SECONDS)))


class JobListener:
    """Listens for and processes DGN jobs from the orchestrator."""

    def __init__(self, client, provider_id: str, shutdown_event: threading.Event):
        self.client = client
        self.orchestrator_service = client.orchestrator_service
        self.provider_id = provider_id
        self.shutdown_event = shutdown_event

    def _job_has_terminal_status(self, job_id: Optional[str]) -> bool:
        """Check whether a job has already reached a terminal status."""
        return self._get_terminal_job_status(job_id) is not None

    @staticmethod
    def _remote_job_status(job_details: Any) -> Optional[str]:
        if not isinstance(job_details, dict):
            return None
        status = job_details.get("status")
        return status if isinstance(status, str) else None

    def _get_terminal_job_status(self, job_id: Optional[str]) -> Optional[str]:
        """Return the current terminal status for a job, if any."""
        if not job_id:
            return None

        try:
            job_details = self.orchestrator_service.get_job(job_id)
        except Exception as e:
            logging.warning(f"Could not verify terminal status for job {job_id}: {e}")
            return None

        if not isinstance(job_details, dict):
            return None

        status = self._remote_job_status(job_details)
        if status in TERMINAL_JOB_STATUSES:
            return status

        return None

    def _emit_job_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        print(json.dumps({"type": event_type, "payload": payload}), flush=True)

    def _emit_job_cleared(
        self,
        job_id: Optional[str],
        service_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> None:
        if not job_id:
            return

        payload: Dict[str, Any] = {"id": job_id}
        if service_type:
            payload["service_type"] = service_type
        if status:
            payload["status"] = status

        self._emit_job_event("JOB_CLEARED", payload)

    def _is_requeueable_infrastructure_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        transient_markers = (
            "connection aborted",
            "timed out",
            "timeout",
            "npipe",
            "named pipe",
            "docker daemon",
            "protocolerror",
            "out of memory",
            "oom",
            "cannot allocate memory",
            "container exited unexpectedly",
            "container killed",
        )
        return any(marker in text for marker in transient_markers)

    def _job_was_interrupted_by_infrastructure(self, job_id: Optional[str]) -> bool:
        return bool(
            job_id
            and getattr(self.client, "interrupted_job_id", None) == job_id
        )

    def _handle_service_startup_failure(
        self,
        job: Dict[str, Any],
        service_type: str,
    ) -> None:
        job_id = job.get("id")
        if self._job_was_interrupted_by_infrastructure(job_id):
            logging.info(
                f"Job {job_id} was already requeued after a local container "
                "failure during startup. Skipping failed status update."
            )
            return

        if self.shutdown_event.is_set():
            return

        logging.error(
            f"ComfyUI for service '{service_type}' failed to start. Failing job."
        )

        self._emit_job_event(
            "JOB_FAILED",
            {
                "id": job_id,
                "service_type": service_type,
                "error": "ComfyUI failed to start",
            },
        )

        self.orchestrator_service.update_job_status(job_id, "failed")
        self.client.current_job = None

    def _requeue_job_after_infrastructure_error(
        self,
        job: Dict[str, Any],
        exc: Exception,
        reason: str = "provider_docker_unavailable",
    ) -> None:
        job_id = job.get("id")
        error_message = f"Infrastructure error: {exc}"

        if not job_id:
            return

        logging.error(f"Requeueing job {job_id} after infrastructure error: {exc}")

        self._emit_job_event(
            "JOB_FAILED",
            {
                "id": job_id,
                "service_type": self._get_service_type_for_job(job),
                "error": f"{error_message} (job requeued)",
            },
        )

        try:
            self.orchestrator_service.reset_interrupted_job(
                job_id,
                execution_token=job.get("execution_token"),
                reason=reason,
            )
        except Exception as reset_error:
            logging.error(f"Failed to requeue job {job_id}: {reset_error}")

        if self.client.active_service_type and not HEADLESS_MODE:
            try:
                docker_manager.stop_container(
                    service_type=self.client.active_service_type
                )
            except Exception as stop_error:
                logging.debug(
                    f"Failed to stop container after infrastructure error: {stop_error}"
                )

        self.client.interrupted_job_id = job_id
        self.client.interrupted_job_execution_token = job.get("execution_token")
        self.client.current_job = None
        self.client.active_service_type = None

    def _process_job_safely(self, job: Dict[str, Any]) -> bool:
        """
        Process a job with proper error handling.

        Args:
            job: Job dictionary from the orchestrator

        Returns:
            True if processing completed (success or handled failure),
            False if a critical auth error occurred that should stop the listener.
        """
        job_id = job.get("id") if job else None
        service_type_for_cache = self._get_service_type_for_job(job) if job else None
        job_used_service = False

        try:
            # Safety check: Validate hardware requirements before processing
            # This is a belt-and-suspenders check in addition to server-side filtering
            workflow_type = job.get("workflow_type")
            if workflow_type:
                try:
                    from services.hardware_profiler import (
                        can_run_service,
                        get_service_incompatibility_reason,
                    )

                    service_type = (
                        service_type_for_cache
                        or self.client.get_service_type_for_workflow(workflow_type)
                    )
                    service_config = self.client.services_config.get(service_type, {})

                    if service_config and not can_run_service(
                        service_config, self.client.available_vram
                    ):
                        reason = get_service_incompatibility_reason(
                            service_config, self.client.available_vram
                        )
                        error_msg = f"Job requires service '{service_type}' but hardware is incompatible: {reason}"
                        logging.error(error_msg)

                        # Emit JOB_FAILED event
                        self._emit_job_event(
                            "JOB_FAILED",
                            {
                                "id": job.get("id"),
                                "service_type": service_type,
                                "error": error_msg,
                            },
                        )

                        self.orchestrator_service.update_job_status(
                            job.get("id"), "failed"
                        )
                        return True
                except ValueError:
                    # Unknown workflow type - let it proceed and fail later if needed
                    pass

            import time as _time

            _job_start_time = _time.monotonic()

            # Get processor and set up cancellation check
            processor = self.client._get_job_processor(job, self.shutdown_event)
            job_used_service = True

            # Check periodically during processing
            self._monitor_job_cancellation(job.get("id"), processor)

            try:
                processor.process()
            finally:
                processor.close()

            if (
                job_id
                and self._job_was_interrupted_by_infrastructure(job_id)
            ):
                logging.info(
                    f"Job {job_id} was interrupted by an infrastructure "
                    "failure. Skipping completion/failure notification because "
                    "the crash handler already reset it."
                )
                return True

            if (
                self.shutdown_event.is_set()
                and getattr(self.client, "stop_requested", False)
                and not self._job_has_terminal_status(job_id)
            ):
                if job_id:
                    self.client.interrupted_job_id = job_id
                    self.client.interrupted_job_execution_token = job.get(
                        "execution_token"
                    )
                logging.info(
                    f"Shutdown interrupted job {job_id}. "
                    "Skipping completion event and deferring reset to cleanup."
                )
                return True

            # Job may have been cancelled externally (e.g., by the cancellation monitor)
            # while the processor was still winding down.  Don't emit JOB_COMPLETE for
            # jobs that already reached a terminal state in the DB.
            terminal_status = self._get_terminal_job_status(job_id)
            if terminal_status:
                logging.info(
                    f"Job {job_id} reached terminal status during processing "
                    f"('{terminal_status}'). Clearing local job state without "
                    "emitting a completion/failure notification."
                )
                self._emit_job_cleared(job_id, service_type_for_cache, terminal_status)
                return True

            _job_duration = int(_time.monotonic() - _job_start_time)

            # Emit JOB_COMPLETE event
            self._emit_job_event(
                "JOB_COMPLETE",
                {
                    "id": job.get("id"),
                    "service_type": service_type_for_cache,
                },
            )

            # Emit MONETIZE_JOB_COMPLETE for wallet refresh in Electron
            if job.get("monetize_job"):
                _wf = job.get("workflow_type", "")
                try:
                    _svc = self.client.get_service_type_for_workflow(_wf)
                except Exception:
                    _svc = _wf
                print(
                    json.dumps(
                        {
                            "type": "MONETIZE_JOB_COMPLETE",
                            "payload": {
                                "id": job.get("id"),
                                "service_type": _svc,
                                "duration_seconds": _job_duration,
                            },
                        }
                    ),
                    flush=True,
                )

            return True
        except AuthError:
            self.orchestrator_service.signal_auth_expired()
            logging.warning(f"Auth expired during processing of job {job.get('id')}.")
            raise  # Re-raise to be caught by outer exception handler
        except InfrastructureError as e:
            self._requeue_job_after_infrastructure_error(
                job,
                e,
                reason="provider_gpu_oom",
            )
            return True
        except Exception as e:
            if (
                self.shutdown_event.is_set()
                and getattr(self.client, "stop_requested", False)
                and not self._job_has_terminal_status(job_id)
            ):
                if job_id:
                    self.client.interrupted_job_id = job_id
                    self.client.interrupted_job_execution_token = job.get(
                        "execution_token"
                    )
                logging.info(
                    f"Shutdown interrupted job {job_id} with in-flight error: {e}. "
                    "Deferring reset to cleanup."
                )
                return True

            # Processor threw because the container/workflow was stopped by the
            # cancellation monitor.  If the job is already terminal in the DB,
            # don't report it as a failure — just clean up quietly.
            if (
                job_id
                and self._job_was_interrupted_by_infrastructure(job_id)
            ):
                logging.info(
                    f"Job {job_id} raised after an infrastructure failure: {e}. "
                    "Skipping failure notification because the crash handler "
                    "already reset it."
                )
                return True

            terminal_status = self._get_terminal_job_status(job_id)
            if terminal_status:
                logging.info(
                    f"Job {job_id} was externally terminated with status "
                    f"'{terminal_status}'. Clearing local job state without "
                    "emitting a failure notification."
                )
                self._emit_job_cleared(job_id, service_type_for_cache, terminal_status)
                return True

            logging.error(
                f"An error occurred while processing job {job.get('id')}: {e}",
                exc_info=True,
            )
            trace = traceback.format_exc()
            # Emit JOB_FAILED event with traceback for remote debugging
            self._emit_job_event(
                "JOB_FAILED",
                {
                    "id": job.get("id"),
                    "service_type": service_type_for_cache,
                    "error": str(e),
                    "traceback": trace,
                },
            )

            if job and job.get("id"):
                self.orchestrator_service.update_job_status(
                    job.get("id"),
                    "failed",
                    completion_metadata={
                        "error": str(e),
                        "traceback": trace,
                    },
                )
            return True
        finally:
            download_manager = getattr(self.client, "download_manager", None)
            if download_manager and service_type_for_cache and job_used_service:
                download_manager.notify_job_complete(service_type_for_cache)

            if (
                self.shutdown_event.is_set()
                and getattr(self.client, "stop_requested", False)
                and job_id
                and not self.client.interrupted_job_id
                and not self._job_has_terminal_status(job_id)
            ):
                self.client.interrupted_job_id = job_id
                self.client.interrupted_job_execution_token = job.get("execution_token")
            self.client.current_job = None

    def _cleanup_remote_cancelled_job(
        self,
        job_id: str,
        processor,
        remote_status: str,
    ) -> None:
        logging.warning(
            f"Job {job_id} was {remote_status}. Interrupting processing."
        )

        # Interrupt before stopping the container so ComfyUI can abort cleanly
        # while its HTTP endpoint is still alive.
        comfyui_client = getattr(processor, "comfyui_client", None)
        if comfyui_client:
            try:
                comfyui_client.interrupt_workflow()
            except Exception as e:
                logging.debug(f"Could not interrupt workflow: {e}")

        # Stop any running Docker container for this job.
        if (
            docker_manager
            and hasattr(self.client, "active_service_type")
            and self.client.active_service_type
        ):
            try:
                logging.info(
                    f"Stopping Docker container for service "
                    f"'{self.client.active_service_type}' due to job cancellation..."
                )
                docker_manager.stop_container(
                    service_type=self.client.active_service_type
                )
                self.client.active_service_type = None
            except Exception as e:
                logging.debug(f"Could not stop container: {e}")

        logging.info(f"Job {job_id} cancellation cleanup completed.")

    def _monitor_job_cancellation(self, job_id: str, processor) -> None:
        """
        Monitor a job for cancellation during processing.

        This runs in a background thread and checks periodically if the job
        has been cancelled/deleted. If so, it interrupts the workflow and stops
        any running containers.

        Args:
            job_id: The ID of the job being processed
            processor: The job processor instance
        """
        import threading

        def _monitor_loop():
            check_interval = 5  # Check every 5 seconds
            while not self.shutdown_event.is_set():
                try:
                    job_details = self.orchestrator_service.get_job(job_id)
                    remote_status = self._remote_job_status(job_details)
                    if remote_status in REMOTE_CANCELLATION_STATUSES:
                        self._cleanup_remote_cancelled_job(
                            job_id,
                            processor,
                            remote_status,
                        )
                        return
                    if job_details is None:
                        logging.debug(
                            f"Could not verify cancellation state for job "
                            f"{job_id}; will retry."
                        )
                except Exception as e:
                    logging.debug(f"Error checking job status: {e}")

                # Wait for next check interval
                self.shutdown_event.wait(timeout=check_interval)

        monitor_thread = threading.Thread(target=_monitor_loop, daemon=True)
        monitor_thread.start()

    def _fetch_next_job_with_priority(self):
        """
        Poll for the next job using priority ordering:
        1. Own jobs first (if process_own_jobs=True) — mine policy
        2. Community jobs second (if community_mode != 'none')
        3. Monetize jobs (if monetize_mode=True, separate track)

        Returns the first job found, or None.
        """
        client = self.client

        # Monetize is a completely separate track — check it independently
        if getattr(client, "monetize_mode", False):
            job = self.orchestrator_service.get_next_job(
                provider_id=self.provider_id,
                accept_policy="monetize",
                allowed_ids=[],
                monetize_mode=True,
            )
            if job:
                return job

        # Priority 1: own jobs (mine policy)
        # Always poll own jobs when process_own_jobs is enabled OR when community
        # mode is "none" (Private) — without this, Private mode would never poll.
        community_mode = getattr(client, "community_mode", "none")
        if getattr(client, "process_own_jobs", False) or community_mode == "none":
            job = self.orchestrator_service.get_next_job(
                provider_id=self.provider_id,
                accept_policy="mine",
                allowed_ids=client.allowed_ids,
                monetize_mode=False,
            )
            if job:
                return job

        # Priority 2: community jobs
        if community_mode == "none":
            return None

        community_policy = {
            "trusted_users": "users",
            "trusted_projects": "project",
            "all": "all",
        }.get(community_mode, "all")

        return self.orchestrator_service.get_next_job(
            provider_id=self.provider_id,
            accept_policy=community_policy,
            allowed_ids=client.allowed_ids,
            monetize_mode=False,
        )

    def listen_for_jobs(self) -> None:
        """Listen for jobs from the orchestrator (for dedicated providers)."""
        consecutive_empty_polls = 0
        while not self.shutdown_event.is_set():
            # Clear wakeup event at start of loop
            if hasattr(self.client, "job_wakeup_event"):
                self.client.job_wakeup_event.clear()

            job = None
            try:
                # Acquire the processing lock BEFORE fetching a job to ensure
                # only one job is acquired and processed at a time
                with self.client.processing_lock:
                    logging.info(
                        f"Checking for new jobs for provider {self.provider_id}..."
                    )
                    job = self._fetch_next_job_with_priority()

                    if job and job.get("id"):
                        # Check if job was cancelled/deleted before processing
                        job_check = self.orchestrator_service.get_job(job.get("id"))
                        remote_status = self._remote_job_status(job_check)
                        if remote_status in REMOTE_CANCELLATION_STATUSES:
                            logging.info(
                                f"Job {job.get('id')} was cancelled/deleted before processing. Skipping."
                            )
                            self.client.current_job = None
                            continue

                        self.client.current_job = job
                        logging.info(f"Received job: {job['id']}")

                        # Emit JOB_START event
                        self._emit_job_event(
                            "JOB_START",
                            {
                                "id": job.get("id"),
                                "workflow_type": job.get("workflow_type", "unknown"),
                                "service_type": self._get_service_type_for_job(job),
                            },
                        )

                        # Process the job using the shared helper
                        self._process_job_safely(job)

                        if not self.shutdown_event.is_set():
                            self.orchestrator_service.update_provider_status(
                                self.provider_id, "available"
                            )
                            logging.info(
                                "Provider status set to available. Waiting for next job..."
                            )
                    else:
                        logging.info("No new jobs.")
            except AuthError:
                self.orchestrator_service.signal_auth_expired()
                logging.warning("Could not fetch job due to expired token.")
                if self.orchestrator_service.is_auth_failed_permanently():
                    logging.error("Auth permanently failed. Stopping job listener.")
                    break
            except ProviderError:
                logging.warning(
                    "Provider registration expired. Signaling main process for restart."
                )
                print(json.dumps({"status": "PROVIDER_EXPIRED"}), flush=True)
                break  # Exit the loop, Electron will restart the client
            except Exception as e:
                logging.error(f"Could not connect to the Orchestrator: {e}")

            if not (job and job.get("id")):
                consecutive_empty_polls += 1
                # Use faster polling frequency for headless cloud instances
                base_interval = (
                    TimeoutConfig.HEADLESS_JOB_POLL_INTERVAL
                    if HEADLESS_MODE
                    else TimeoutConfig.JOB_POLL_INTERVAL
                )
                poll_interval = _compute_poll_interval(base_interval, consecutive_empty_polls - 1)

                # Wait for poll interval, but interrupt if shutdown OR if a download
                # completes or routing config changes. Check in 0.5s increments.
                wait_until = time.time() + poll_interval
                while time.time() < wait_until and not self.shutdown_event.is_set():
                    if (
                        hasattr(self.client, "job_wakeup_event")
                        and self.client.job_wakeup_event.is_set()
                    ):
                        logging.info(
                            "Waking up job listener due to download completion or config change."
                        )
                        consecutive_empty_polls = 0
                        break
                    self.shutdown_event.wait(0.5)
            else:
                consecutive_empty_polls = 0
        logging.info("Shutdown event received. Exiting job listening loop.")

    def _get_service_type_for_job(self, job: Dict[str, Any]) -> Optional[str]:
        """Get the service type for a job based on its workflow type."""
        workflow_type = job.get("workflow_type", "image_to_video")
        try:
            return self.client.get_service_type_for_workflow(workflow_type)
        except ValueError:
            # Unknown workflow type, fall back to service_type from job
            return job.get("service_type")

    def _handle_prefetch_suggestions(self, download_manager) -> None:
        """Fetch and apply pre-fetch suggestions from the server.

        When idle (no processable jobs), this proactively downloads Docker images
        that the server identifies as high-demand based on network-wide analysis:
        - Jobs pending in queue by service_type
        - Current cache coverage across all providers
        - This provider's capabilities

        This improves network efficiency by ensuring images are ready before
        jobs need them. The server considers:
        - Number of pending jobs per service_type
        - Number of providers with that image cached vs downloading
        - Cache deficit (pending jobs - cached providers)

        NOTE: This does NOT affect credits in any way. Credits are calculated
        based on actual processing time and VRAM usage when jobs complete.
        Pre-fetching is purely for reducing job wait times.
        """
        if not download_manager:
            return

        community_mode = getattr(self.client, "community_mode", "all")
        monetize_mode = getattr(self.client, "monetize_mode", False)
        allow_network_prefetch = community_mode == "all" or monetize_mode
        if not allow_network_prefetch:
            logging.debug(
                f"Skipping global prefetch suggestions for private community_mode='{community_mode}'."
            )
            return

        # Don't fetch suggestions if we already have downloads in progress
        # This prevents queue explosion and respects MAX_CONCURRENT_DOWNLOADS
        if download_manager._active_downloads or download_manager._download_queue:
            return

        try:
            suggestions = self.orchestrator_service.get_prefetch_suggestions(
                self.provider_id
            )

            if suggestions:
                logging.info(
                    f"Received pre-fetch suggestions from server: {suggestions}"
                )

                # Start downloads for suggested service types (download manager handles queueing)
                for service_type in suggestions[
                    :2
                ]:  # Limit to 2 to avoid queue buildup
                    availability = download_manager.get_image_availability(
                        service_type
                    )
                    if (
                        availability == ImageAvailability.MISSING
                        and not download_manager.is_downloading(service_type)
                        and not download_manager.is_queued(service_type)
                    ):
                        logging.info(f"Pre-fetching suggested image: {service_type}")
                        download_manager.start_background_download(service_type)
                    elif availability == ImageAvailability.UNKNOWN:
                        logging.debug(
                            f"Skipping pre-fetch for '{service_type}' because Docker image "
                            "availability could not be verified."
                        )
        except Exception as e:
            # Non-critical - just log and continue
            logging.debug(f"Failed to get/apply pre-fetch suggestions: {e}")

    @staticmethod
    def _uses_global_download_gate(accept_policy: str) -> bool:
        return accept_policy in ("all", "monetize")

    def _get_download_gate_for_jobs(self, available_with_policy: list) -> Optional[set[str]]:
        """
        Return service types this provider may download for public/monetize jobs.

        Private/trusted/own jobs intentionally bypass this because global cache
        coverage cannot represent provider-specific trust visibility. None means
        "gate unavailable / fail open"; an empty set means "server says coverage
        is already sufficient."
        """
        should_gate = any(
            self._uses_global_download_gate(policy)
            for _, policy in available_with_policy
        )
        if not should_gate:
            return None

        try:
            suggestions = self.orchestrator_service.get_prefetch_suggestions(
                self.provider_id,
                limit=20,
                return_none_on_error=True,
            )
        except TypeError:
            # Test doubles or older service shims may not accept the explicit
            # fail-open flag. Keep the runtime behavior permissive.
            try:
                suggestions = self.orchestrator_service.get_prefetch_suggestions(
                    self.provider_id,
                    20,
                )
            except Exception as e:
                logging.debug(f"Download coverage gate failed: {e}")
                return None
        except Exception as e:
            logging.debug(f"Download coverage gate failed: {e}")
            return None

        if suggestions is None:
            return None
        return set(suggestions)

    def _should_download_missing_image(
        self,
        service_type: str,
        accept_policy: str,
        download_gate: Optional[set[str]],
    ) -> bool:
        if not self._uses_global_download_gate(accept_policy):
            return True
        if download_gate is None:
            return True
        if service_type in download_gate:
            return True

        logging.info(
            f"Skipping download for '{service_type}' because server-side "
            "cache coverage is already sufficient."
        )
        return False

    def listen_for_jobs_auto(self) -> None:
        """Listen for jobs and dynamically start/stop containers with image pre-fetching."""
        logging.info("Entering auto job listening loop with Docker image pre-fetching.")

        # Get download manager from client (may be None in headless mode)
        download_manager = getattr(self.client, "download_manager", None)

        # Sync actual local cache state with server once at startup.
        # The registration-time scan may be empty if Docker was briefly unavailable
        # after a WSL crash+restart, causing the server to skip tier-0 routing for
        # this provider. This corrects that stale state before the first job poll.
        if download_manager:
            try:
                download_manager._sync_cached_images_with_server()
                logging.info("Startup: synced local Docker image cache state with server.")
            except Exception as _sync_err:
                logging.warning(f"Startup: cache sync failed (non-fatal): {_sync_err}")

        consecutive_empty_polls = 0

        try:
            while not self.shutdown_event.is_set():
                # Clear wakeup event at start of loop
                if hasattr(self.client, "job_wakeup_event"):
                    self.client.job_wakeup_event.clear()

                logging.info("Top of auto-mode loop iteration.")
                job = None
                found_processable_job = False

                try:
                    # Acquire the processing lock BEFORE fetching a job
                    with self.client.processing_lock:
                        logging.info("Auto mode: Peeking at available jobs...")

                        # Step 1: Peek at available jobs without reserving.
                        # Build an ordered list of (job, policy) pairs — own jobs first
                        # when process_own_jobs=True so they aren't starved by community work.
                        _community_mode = getattr(self.client, "community_mode", "all")
                        _process_own_jobs = getattr(self.client, "process_own_jobs", False)
                        _monetize_mode = getattr(self.client, "monetize_mode", False)
                        if _monetize_mode:
                            _peek_policy = "monetize"
                        elif _community_mode == "none":
                            _peek_policy = "mine"
                        else:
                            _peek_policy = {
                                "trusted_users": "users",
                                "trusted_projects": "project",
                                "all": "all",
                            }.get(_community_mode, "all")

                        _available_with_policy: list = []
                        if _process_own_jobs and _peek_policy not in ("mine", "monetize"):
                            _mine_jobs = self.orchestrator_service.peek_available_jobs(
                                provider_id=self.provider_id,
                                accept_policy="mine",
                                allowed_ids=self.client.allowed_ids,
                                limit=10,
                            )
                            _mine_ids = {j.get("id") for j in _mine_jobs}
                            _available_with_policy.extend(
                                (j, "mine") for j in _mine_jobs
                            )
                            _community_jobs = self.orchestrator_service.peek_available_jobs(
                                provider_id=self.provider_id,
                                accept_policy=_peek_policy,
                                allowed_ids=self.client.allowed_ids,
                                limit=10,
                            )
                            _available_with_policy.extend(
                                (j, _peek_policy)
                                for j in _community_jobs
                                if j.get("id") not in _mine_ids
                            )
                        else:
                            _peeked = self.orchestrator_service.peek_available_jobs(
                                provider_id=self.provider_id,
                                accept_policy=_peek_policy,
                                allowed_ids=self.client.allowed_ids,
                                limit=10,
                            )
                            _available_with_policy.extend(
                                (j, _peek_policy) for j in _peeked
                            )

                        available_jobs = [j for j, _ in _available_with_policy]

                        if not available_jobs:
                            logging.info("No available jobs found in peek.")
                        else:
                            logging.info(
                                f"Peeked {len(available_jobs)} available jobs."
                            )

                            download_gate = self._get_download_gate_for_jobs(
                                _available_with_policy
                            )

                            # Step 2b: Find first job with available Docker image
                            for peeked_job, _job_policy in _available_with_policy:
                                service_type = self._get_service_type_for_job(
                                    peeked_job
                                )
                                if not service_type:
                                    continue

                                # Check if Docker image is available
                                image_availability = ImageAvailability.AVAILABLE
                                if not HEADLESS_MODE and download_manager:
                                    image_availability = (
                                        download_manager.get_image_availability(
                                            service_type
                                        )
                                    )
                                image_available = (
                                    image_availability == ImageAvailability.AVAILABLE
                                )

                                if image_available:
                                    # Image is ready - reserve and process this job
                                    logging.info(
                                        f"Found job {peeked_job.get('id')} with available image ({service_type}). Reserving..."
                                    )

                                    job = self.orchestrator_service.get_next_job(
                                        provider_id=self.provider_id,
                                        accept_policy=_job_policy,
                                        allowed_ids=self.client.allowed_ids,
                                        job_id=peeked_job.get("id"),
                                        monetize_mode=getattr(
                                            self.client, "monetize_mode", False
                                        ),
                                    )

                                    if job and job.get("id"):
                                        found_processable_job = True
                                        self.client.current_job = job
                                        job_id = job["id"]
                                        logging.info(f"Reserved job: {job_id}")

                                        workflow_type = job.get(
                                            "workflow_type", "image_to_video"
                                        )
                                        actual_service_type = (
                                            self.client.get_service_type_for_workflow(
                                                workflow_type
                                            )
                                        )
                                        self.client.active_service_type = (
                                            actual_service_type
                                        )
                                        logging.info(
                                            f"Job requires service '{actual_service_type}'."
                                        )

                                        # Emit JOB_START early to update UI while container starts/warms up
                                        self._emit_job_event(
                                            "JOB_START",
                                            {
                                                "id": job.get("id"),
                                                "workflow_type": job.get(
                                                    "workflow_type", "unknown"
                                                ),
                                                "service_type": actual_service_type,
                                            },
                                        )

                                        if not HEADLESS_MODE:
                                            logging.info("Starting container...")
                                            container_crash_event = threading.Event()
                                            service_cfg = (
                                                self.client.services_config.get(
                                                    actual_service_type, {}
                                                )
                                            )
                                            is_wan2gp = (
                                                service_cfg.get("backend") == "wan2gp"
                                            )

                                            if is_wan2gp:
                                                # LTX/Wan2GP images are expected to contain
                                                # their model files. If a dependency tries to
                                                # fetch from Hugging Face at runtime, fail
                                                # visibly instead of silently re-downloading on
                                                # every container start.
                                                wan2gp_env = {
                                                    "HF_HUB_OFFLINE": "1",
                                                    "TRANSFORMERS_OFFLINE": "1",
                                                    "HF_HUB_DISABLE_TELEMETRY": "1",
                                                    "CUDA_MODULE_LOADING": "LAZY",
                                                    "PYTORCH_CUDA_ALLOC_CONF": (
                                                        "expandable_segments:True,"
                                                        "max_split_size_mb:128"
                                                    ),
                                                    "MALLOC_ARENA_MAX": "2",
                                                }

                                                if "8gb" in actual_service_type.lower():
                                                    # WanGP's documented low-memory path:
                                                    # profile 4/4.5 + sdpa + low reserved RAM.
                                                    wan2gp_env["WAN2GP_CLI_ARGS"] = (
                                                        "--profile 4.5 --attention sdpa "
                                                        "--perc-reserved-mem-max 0.45"
                                                    )
                                                else:
                                                    wan2gp_env["WAN2GP_CLI_ARGS"] = (
                                                        "--profile 4 --attention sdpa"
                                                    )

                                                import os as _os

                                                server_src = _os.path.join(
                                                    self.client.root_dir,
                                                    "comfyui-storage",
                                                    "wan2gp_server.py",
                                                )
                                                pre_start_copies = []
                                                if _os.path.isfile(server_src):
                                                    pre_start_copies.append(
                                                        (
                                                            server_src,
                                                            "/opt/wan2gp/wan2gp_server.py",
                                                        )
                                                    )
                                                else:
                                                    logging.warning(
                                                        f"wan2gp_server.py not found at {server_src}; using baked-in version."
                                                    )

                                                docker_manager.run_container(
                                                    service_type=actual_service_type,
                                                    command=[
                                                        "python3",
                                                        "/opt/wan2gp/wan2gp_server.py",
                                                    ],
                                                    environment=wan2gp_env,
                                                    pre_start_copies=pre_start_copies,
                                                    shutdown_event=self.shutdown_event,
                                                )
                                                logging.info(
                                                    "Started wan2gp_server.py as the Wan2GP container main process."
                                                )
                                            else:
                                                docker_manager.run_container(
                                                    service_type=actual_service_type,
                                                )

                                            # Start container crash monitor
                                            # This detects OOM kills and unexpected container crashes
                                            def handle_container_crash(
                                                crashed_job_id: str,
                                                reason: str,
                                                container_logs: Optional[str],
                                            ):
                                                """Called when container crashes unexpectedly."""
                                                container_crash_event.set()
                                                logging.error(
                                                    f"Container crash detected for job {crashed_job_id}: {reason}"
                                                )

                                                self._requeue_job_after_infrastructure_error(
                                                    job,
                                                    InfrastructureError(reason),
                                                    reason="provider_container_crash",
                                                )

                                            container_monitor = ContainerMonitor(
                                                docker_client=docker_manager.client,
                                                container_name=docker_manager.get_container_name(
                                                    actual_service_type
                                                ),
                                                job_id=job_id,
                                                on_container_crash=handle_container_crash,
                                                shutdown_event=self.shutdown_event,
                                            )
                                            container_monitor.start()

                                            # Start log streaming in a background thread
                                            # This allows seeing ComfyUI/container logs in the DGN client console
                                            log_thread = threading.Thread(
                                                target=docker_manager.stream_logs,
                                                args=(
                                                    actual_service_type,
                                                    self.shutdown_event,
                                                ),
                                                daemon=True,
                                            )
                                            log_thread.start()
                                        else:
                                            logging.info(
                                                "Headless mode - container already running, skipping Docker management."
                                            )
                                            # Start log tailer for headless mode
                                            from utils.log_tailer import LogTailer

                                            tailer = LogTailer(
                                                "/tmp/comfyui.log",
                                                service_type=actual_service_type,
                                            )
                                            log_thread = threading.Thread(
                                                target=tailer.tail,
                                                args=(self.shutdown_event,),
                                                daemon=True,
                                            )
                                            log_thread.start()

                                        # Check if service uses ComfyUI backend from configuration
                                        service_config = (
                                            self.client.services_config.get(
                                                actual_service_type, {}
                                            )
                                        )
                                        uses_comfyui = (
                                            service_config.get("backend", "comfyui")
                                            == "comfyui"
                                        )

                                        if uses_comfyui:
                                            if self.client.comfyui_client.wait_for_ready(
                                                self.shutdown_event,
                                                abort_event=(
                                                    container_crash_event
                                                    if not HEADLESS_MODE
                                                    else None
                                                ),
                                            ):
                                                # Process the job using shared helper
                                                self._process_job_safely(job)
                                            else:
                                                self._handle_service_startup_failure(
                                                    job,
                                                    actual_service_type,
                                                )
                                        else:
                                            # For text_generation, directly process the job (no ComfyUI needed)
                                            self._process_job_safely(job)

                                        if not self.shutdown_event.is_set():
                                            logging.info(f"Job processing finished.")
                                            if not HEADLESS_MODE:
                                                # Stop container monitor before stopping container
                                                if "container_monitor" in locals():
                                                    container_monitor.stop()

                                                logging.info(
                                                    f"Stopping container for service '{actual_service_type}'..."
                                                )
                                                docker_manager.stop_container(
                                                    service_type=actual_service_type
                                                )
                                            self.client.active_service_type = None
                                            self.orchestrator_service.update_provider_status(
                                                self.provider_id, "available"
                                            )
                                            logging.info(
                                                "Provider status set to available. Waiting for next job..."
                                            )
                                    else:
                                        # Reservation failed - job may have been taken by another provider
                                        error_msg = (
                                            job.get("error")
                                            if isinstance(job, dict)
                                            else "Unknown"
                                        )
                                        logging.warning(
                                            f"Failed to reserve job {peeked_job.get('id')}. Server response: {error_msg}. Moving to next available job."
                                        )
                                        continue  # Try next peeked job

                                    break  # Exit the peeked jobs loop after processing one job
                                elif image_availability == ImageAvailability.UNKNOWN:
                                    logging.debug(
                                        f"Skipping job {peeked_job.get('id')} for '{service_type}' because "
                                        "Docker image availability could not be verified."
                                    )
                                else:
                                    # Image not available - start background download if
                                    # this provider should be the one to pull it.
                                    if download_manager:
                                        status = download_manager.get_download_status(
                                            service_type
                                        )
                                        is_downloading = (
                                            download_manager.is_downloading(
                                                service_type
                                            )
                                        )
                                        is_queued = download_manager.is_queued(
                                            service_type
                                        )

                                        if not is_downloading and not is_queued:
                                            if not self._should_download_missing_image(
                                                service_type,
                                                _job_policy,
                                                download_gate,
                                            ):
                                                continue

                                            if (
                                                status
                                                and status.value == "permanently_failed"
                                            ):
                                                logging.warning(
                                                    f"Image for service '{service_type}' does not exist on the registry (permanent failure). Skipping job."
                                                )
                                                continue
                                            elif status and status.value == "failed":
                                                logging.info(
                                                    f"Image for service '{service_type}' previously failed. Retrying download..."
                                                )

                                            logging.info(
                                                f"Image for service '{service_type}' not available locally. "
                                                f"Starting background download for peeked job..."
                                            )
                                            download_manager.start_background_download(
                                                service_type,
                                                accept_policy=_job_policy,
                                            )
                                        else:
                                            logging.debug(
                                                f"Image for service '{service_type}' already downloading/queued (status: {status.value if status else 'unknown'})."
                                            )
                                    # Continue to check next job

                        # Handle pre-fetch suggestions when idle
                        # This proactively downloads images for high-demand workflows
                        # NOTE: This does NOT affect credits - it's purely for network efficiency
                        if not found_processable_job:
                            self._handle_prefetch_suggestions(download_manager)

                            # Disk-pressure cleanup runs in the same idle window as
                            # prefetching. Cheap when at Healthy (one disk_usage call),
                            # only evicts at Pressure / Critical tiers.
                            if download_manager:
                                try:
                                    download_manager.check_and_evict_for_pressure()
                                except Exception as pressure_err:
                                    logging.debug(
                                        f"check_and_evict_for_pressure failed: {pressure_err}"
                                    )

                        if not found_processable_job:
                            if available_jobs:
                                # Count how many were actually ready vs downloading
                                ready_count = 0
                                downloading_count = 0
                                unknown_count = 0
                                for pj in available_jobs:
                                    st = self._get_service_type_for_job(pj)
                                    availability = (
                                        download_manager.get_image_availability(st)
                                        if download_manager
                                        else ImageAvailability.AVAILABLE
                                    )
                                    if availability == ImageAvailability.AVAILABLE:
                                        ready_count += 1
                                    elif availability == ImageAvailability.UNKNOWN:
                                        unknown_count += 1
                                    else:
                                        downloading_count += 1

                                if ready_count > 0:
                                    logging.info(
                                        f"{ready_count} jobs were ready but reservation failed. Provider may be busy in DB or jobs were taken. Waiting..."
                                    )
                                elif unknown_count > 0 and downloading_count > 0:
                                    logging.info(
                                        f"{downloading_count} jobs still need image downloads, and "
                                        f"{unknown_count} could not be checked because Docker was temporarily unavailable. Waiting..."
                                    )
                                elif unknown_count > 0:
                                    logging.info(
                                        f"{unknown_count} jobs could not be checked because Docker was temporarily unavailable. Waiting..."
                                    )
                                else:
                                    logging.info(
                                        f"All {len(available_jobs)} available jobs require images that are still downloading. Waiting..."
                                    )
                            else:
                                logging.info("No new jobs found in this check.")

                except AuthError:
                    self.orchestrator_service.signal_auth_expired()
                    logging.warning("Could not fetch job due to expired token.")
                    if self.orchestrator_service.is_auth_failed_permanently():
                        logging.error("Auth permanently failed. Stopping job listener.")
                        break
                except ProviderError:
                    logging.warning(
                        "Provider registration expired. Signaling main process for restart."
                    )
                    print(json.dumps({"status": "PROVIDER_EXPIRED"}), flush=True)
                    break  # Exit the loop, Electron will restart the client
                except Exception as e:
                    logging.error(
                        f"An error occurred in auto job listening loop: {e}",
                        exc_info=True,
                    )

                    if (
                        self.shutdown_event.is_set()
                        and getattr(self.client, "stop_requested", False)
                        and job and job.get("id")
                        and not self._job_has_terminal_status(job.get("id"))
                    ):
                        self.client.interrupted_job_id = job.get("id")
                        self.client.interrupted_job_execution_token = job.get("execution_token")
                        logging.info(
                            f"Shutdown interrupted job {job.get('id')} with error: {e}. "
                            "Deferring reset to cleanup."
                        )
                        continue

                    # Ensure provider status is reset to available on error
                    # This prevents the provider from being stuck in 'busy' state in the DB
                    try:
                        self.orchestrator_service.update_provider_status(
                            self.provider_id, "available"
                        )
                    except Exception as status_err:
                        logging.error(
                            f"Failed to reset provider status after error: {status_err}"
                        )

                    if (
                        job
                        and job.get("id")
                        and self._is_requeueable_infrastructure_error(e)
                    ):
                        self._requeue_job_after_infrastructure_error(job, e)
                        continue

                    if job and job.get("id"):
                        logging.error(
                            f"Failure occurred with active job {job.get('id')}. Cleaning up..."
                        )
                        # Emit JOB_FAILED event
                        self._emit_job_event(
                            "JOB_FAILED",
                            {
                                "id": job.get("id"),
                                "service_type": self._get_service_type_for_job(job),
                                "error": str(e),
                            },
                        )

                        try:
                            self.orchestrator_service.update_job_status(
                                job.get("id"), "failed"
                            )
                        except:
                            logging.error("Failed to update job status to failed.")

                        self.client.current_job = None

                if not found_processable_job:
                    consecutive_empty_polls += 1
                    # Use faster polling frequency for headless cloud instances
                    base_interval = (
                        TimeoutConfig.HEADLESS_JOB_POLL_INTERVAL
                        if HEADLESS_MODE
                        else TimeoutConfig.JOB_POLL_INTERVAL
                    )
                    poll_interval = _compute_poll_interval(base_interval, consecutive_empty_polls - 1)

                    # Wait for poll interval, but interrupt if shutdown OR if a download
                    # completes or routing config changes. Check in 0.5s increments.
                    wait_until = time.time() + poll_interval
                    while time.time() < wait_until and not self.shutdown_event.is_set():
                        if (
                            hasattr(self.client, "job_wakeup_event")
                            and self.client.job_wakeup_event.is_set()
                        ):
                            logging.info(
                                "Waking up auto job listener due to download completion or config change."
                            )
                            consecutive_empty_polls = 0
                            break
                        self.shutdown_event.wait(0.5)
                else:
                    consecutive_empty_polls = 0
        finally:
            logging.info(
                "Shutdown event received or loop exited. Exiting auto job listening loop."
            )

            # Shutdown the download manager
            if download_manager:
                download_manager.shutdown()

            if self.client.active_service_type and not HEADLESS_MODE:
                logging.info(
                    f"Ensuring container for service '{self.client.active_service_type}' is stopped."
                )
                docker_manager.stop_container(
                    service_type=self.client.active_service_type
                )
                self.client.active_service_type = None
