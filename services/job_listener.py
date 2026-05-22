import logging
import json
import random
import threading
import time
import traceback
from typing import Dict, Any, Optional

from config import (
    CONTAINER_RESTART_SETTLE_SECONDS,
    HEADLESS_MODE,
    TimeoutConfig,
)
from services.docker_manager import docker_manager
from services.container_monitor import ContainerMonitor
from services.docker_download_manager import ImageAvailability, POST_DOWNLOAD_SETTLE_SECS
from exceptions import AuthError, InfrastructureError, ProviderError


# Cap on backoff so a long outage still polls every 5 minutes.
JOB_POLL_BACKOFF_CAP_SECONDS = 300
# "lease_lost" is a local synthetic status returned when the server still has
# the job row, but this provider no longer has access to it because the lease
# was reset or reassigned.
TERMINAL_JOB_STATUSES = ("completed", "failed", "cancelled", "deleted", "lease_lost")
REMOTE_CANCELLATION_STATUSES = ("cancelled", "deleted", "lease_lost")


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
        # Tracks the active ContainerMonitor so it can be stopped consistently
        # from any cleanup path (normal, infrastructure error, cancellation, shutdown).
        # Using an instance attribute rather than locals() avoids the race where
        # the monitor thread fires a crash callback after the container has already
        # been stopped on an error path.
        self._active_container_monitor = None
        self._remote_stop_cleanup_jobs = set()
        self._remote_stop_cleanup_lock = threading.Lock()

    def _stop_container_monitor(self) -> None:
        """Stop and clear the active container monitor, if any.

        Must be called before every stop_container() call to prevent the
        monitor thread from firing a spurious crash callback on the next job
        after the container has been intentionally removed.
        """
        monitor = self._active_container_monitor
        if monitor is not None:
            monitor.stop()
            self._active_container_monitor = None

    def _wait_after_container_cleanup(self, service_type: str) -> None:
        """Let desktop GPU container cleanup settle before the next reservation."""
        delay = max(0, int(CONTAINER_RESTART_SETTLE_SECONDS or 0))
        if delay <= 0 or self.shutdown_event.is_set():
            return

        logging.info(
            "Waiting %ss after stopping service '%s' before polling for the next job.",
            delay,
            service_type,
        )
        deadline = time.time() + delay
        while time.time() < deadline and not self.shutdown_event.is_set():
            time.sleep(min(0.5, deadline - time.time()))

    @staticmethod
    def _should_keep_container_warm(service_config: Dict[str, Any]) -> bool:
        """Return whether this service explicitly opts into container reuse."""
        if HEADLESS_MODE:
            return False
        if service_config.get("backend") == "local":
            return False

        keep_warm = service_config.get("keep_container_warm", False)
        if isinstance(keep_warm, str):
            return keep_warm.strip().lower() in {"1", "true", "yes", "on"}
        return bool(keep_warm)

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

    def _wait_for_terminal_job_status(
        self,
        job_id: Optional[str],
        attempts: int = 3,
        delay_seconds: float = 1.0,
    ) -> Optional[str]:
        """Give the orchestrator a short grace period to reflect terminal status."""
        for attempt in range(max(1, attempts)):
            terminal_status = self._get_terminal_job_status(job_id)
            if terminal_status:
                return terminal_status
            if attempt < attempts - 1:
                self.shutdown_event.wait(delay_seconds)
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

    def _emit_job_completed(
        self,
        job: Dict[str, Any],
        service_type: Optional[str],
        duration_seconds: int,
    ) -> None:
        self._emit_job_event(
            "JOB_COMPLETE",
            {
                "id": job.get("id"),
                "service_type": service_type,
            },
        )

        if job.get("monetize_job"):
            workflow_type = job.get("workflow_type", "")
            try:
                monetize_service = self.client.get_service_type_for_workflow(workflow_type)
            except Exception:
                monetize_service = workflow_type
            print(
                json.dumps(
                    {
                        "type": "MONETIZE_JOB_COMPLETE",
                        "payload": {
                            "id": job.get("id"),
                            "service_type": monetize_service,
                            "duration_seconds": duration_seconds,
                        },
                    }
                ),
                flush=True,
            )

    def _is_requeueable_infrastructure_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        is_docker_api_500 = (
            ("127.0.0.1:2375" in text or "docker.errors.apierror" in text)
            and (
                "500 server error" in text
                or "internal server error" in text
                or "client connection is closing" in text
                or "context canceled" in text
                or "context cancelled" in text
            )
        )
        if is_docker_api_500:
            return True

        transient_markers = (
            "connection aborted",
            "connection refused",
            "actively refused",
            "failed to establish a new connection",
            "max retries exceeded",
            "timed out",
            "timeout",
            "npipe",
            "named pipe",
            "docker daemon",
            "protocolerror",
            "remote end closed connection",
            "cannot reach comfyui",
            "cannot reach wan2gp",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "out of memory",
            "oom",
            "cannot allocate memory",
            "container exited unexpectedly",
            "container killed",
        )
        return any(marker in text for marker in transient_markers)

    @staticmethod
    def _get_infrastructure_recovery_reason(exc: Exception) -> str:
        text = str(exc).lower()
        if any(
            marker in text
            for marker in (
                "out of memory",
                "oom",
                "cannot allocate memory",
                "cublas_status_alloc_failed",
                "torch.cuda.outofmemoryerror",
            )
        ):
            return "provider_gpu_oom"
        if any(
            marker in text
            for marker in (
                "failed to start",
                "did not become ready",
                "500 server error",
                "internal server error",
                "client connection is closing",
                "context canceled",
                "context cancelled",
                "cannot reach comfyui",
                "cannot reach wan2gp",
                "service unavailable",
                "bad gateway",
                "gateway timeout",
                "connection refused",
                "http 500",
                "http 502",
                "http 503",
                "http 504",
            )
        ):
            return "provider_service_startup_failed"
        return "provider_docker_unavailable"

    def _job_was_interrupted_by_infrastructure(
        self,
        job_id: Optional[str],
        execution_token: Optional[str] = None,
    ) -> bool:
        if not job_id or getattr(self.client, "interrupted_job_id", None) != job_id:
            return False

        interrupted_token = getattr(self.client, "interrupted_job_execution_token", None)
        if execution_token and interrupted_token and execution_token != interrupted_token:
            return False

        return True

    def _clear_stale_infrastructure_marker(self, job: Dict[str, Any]) -> None:
        job_id = job.get("id") if job else None
        execution_token = job.get("execution_token") if job else None
        interrupted_token = getattr(self.client, "interrupted_job_execution_token", None)

        if (
            job_id
            and execution_token
            and interrupted_token
            and getattr(self.client, "interrupted_job_id", None) == job_id
            and execution_token != interrupted_token
        ):
            logging.info(
                f"Clearing stale infrastructure marker for retried job {job_id} "
                "with a new execution token."
            )
            self.client.interrupted_job_id = None
            self.client.interrupted_job_execution_token = None

    def _handle_service_startup_failure(
        self,
        job: Dict[str, Any],
        service_type: str,
    ) -> None:
        job_id = job.get("id")
        if self._job_was_interrupted_by_infrastructure(
            job_id,
            job.get("execution_token"),
        ):
            logging.info(
                f"Job {job_id} was already requeued after a local container "
                "failure during startup. Skipping failed status update."
            )
            return

        if self.shutdown_event.is_set():
            return

        logging.error(
            f"ComfyUI for service '{service_type}' failed to start. "
            f"Requeueing job {job_id} so another provider can retry it."
        )
        self._requeue_job_after_infrastructure_error(
            job,
            InfrastructureError(
                f"ComfyUI failed to start for service '{service_type}'"
            ),
            reason="provider_service_startup_failed",
        )

    @staticmethod
    def _format_exec_output(output: Any, max_chars: int = 2000) -> str:
        if output is None:
            return ""
        if isinstance(output, bytes):
            text = output.decode("utf-8", errors="replace")
        else:
            text = str(output)
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    @staticmethod
    def _is_safe_ollama_model_name(model_name: str) -> bool:
        if not isinstance(model_name, str) or not model_name:
            return False
        allowed = set(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            "._:/-"
        )
        return all(ch in allowed for ch in model_name)

    def _warm_ollama_model_in_container(
        self,
        job: Dict[str, Any],
        service_type: str,
    ) -> bool:
        """Load the Ollama model from inside the container before host requests.

        Cold model loads can leave the Windows -> WSL localhost connection idle
        long enough for the client side to disconnect. Running a tiny in-container
        `ollama run` avoids that boundary and keeps the real processor request fast.
        """
        if HEADLESS_MODE:
            return True

        inputs = job.get("inputs") or {}
        model_name = inputs.get("model", "qwen3:4b")
        if not self._is_safe_ollama_model_name(model_name):
            logging.warning(
                "Skipping Ollama warmup for unsafe model name %r; processor will "
                "handle validation/failure.",
                model_name,
            )
            return True

        warmup_prompt = "Return the single word ready."
        for attempt in range(1, 4):
            if self.shutdown_event.is_set():
                return False

            logging.info(
                "Warming Ollama model '%s' inside container for service '%s' "
                "(attempt %s/3).",
                model_name,
                service_type,
                attempt,
            )
            result = docker_manager.exec_in_container(
                service_type,
                ["ollama", "run", model_name, warmup_prompt],
            )

            exit_code = getattr(result, "exit_code", None)
            output = self._format_exec_output(getattr(result, "output", None))
            if result is not None and exit_code == 0:
                if output:
                    logging.debug("Ollama warmup output tail: %s", output)
                logging.info("Ollama model '%s' warmed successfully.", model_name)
                return True

            logging.warning(
                "Ollama warmup failed for model '%s' (attempt %s/3, exit=%s): %s",
                model_name,
                attempt,
                exit_code,
                output or "no output",
            )
            self.shutdown_event.wait(min(10, 2 * attempt))

        self._requeue_job_after_infrastructure_error(
            job,
            InfrastructureError(
                f"Ollama model '{model_name}' did not warm up for service "
                f"'{service_type}'"
            ),
            reason="provider_service_startup_failed",
        )
        return False

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

        if self._job_was_interrupted_by_infrastructure(
            job_id,
            job.get("execution_token"),
        ):
            logging.info(
                f"Job {job_id} was already requeued after an infrastructure "
                "failure. Skipping duplicate reset."
            )
            return

        self.client.interrupted_job_id = job_id
        self.client.interrupted_job_execution_token = job.get("execution_token")

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

        active_service_type = getattr(self.client, "active_service_type", None)
        if active_service_type and not HEADLESS_MODE:
            # Stop the monitor BEFORE the container so it cannot fire a spurious
            # crash callback on the intentional removal.
            self._stop_container_monitor()
            try:
                docker_manager.stop_container(
                    service_type=active_service_type
                )
            except Exception as stop_error:
                logging.debug(
                    f"Failed to stop container after infrastructure error: {stop_error}"
                )

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
        self._clear_stale_infrastructure_marker(job)

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
                    hardware_profile = getattr(
                        self.client, "hardware_profile", None
                    )

                    if service_config and not can_run_service(
                        service_config,
                        self.client.available_vram,
                        hardware_profile,
                    ):
                        reason = get_service_incompatibility_reason(
                            service_config,
                            self.client.available_vram,
                            hardware_profile,
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
                and self._job_was_interrupted_by_infrastructure(
                    job_id,
                    job.get("execution_token"),
                )
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

            _job_duration = int(_time.monotonic() - _job_start_time)

            # Job may have been cancelled externally (e.g., by the cancellation monitor)
            # while the processor was still winding down.  If the processor itself already
            # marked the job completed, still emit the local completion notifications.
            terminal_status = self._wait_for_terminal_job_status(job_id)
            if terminal_status:
                self.orchestrator_service.clear_active_job(job_id)
                if terminal_status == "completed":
                    logging.info(
                        f"Job {job_id} reached completed status during processing. "
                        "Emitting completion notification without another status update."
                    )
                    self._emit_job_completed(
                        job,
                        service_type_for_cache,
                        _job_duration,
                    )
                    return True

                logging.info(
                    f"Job {job_id} reached terminal status during processing "
                    f"('{terminal_status}'). Clearing local job state without "
                    "emitting a completion/failure notification."
                )
                self._emit_job_cleared(job_id, service_type_for_cache, terminal_status)
                return True

            if job_id:
                message = (
                    f"Processor for job {job_id} returned without setting a terminal "
                    "job status. Marking the job failed so it cannot remain stuck "
                    "in processing."
                )
                logging.error(message)
                self._emit_job_event(
                    "JOB_FAILED",
                    {
                        "id": job_id,
                        "service_type": service_type_for_cache,
                        "error": message,
                    },
                )
                self.orchestrator_service.update_job_status(
                    job_id,
                    "failed",
                    completion_metadata={
                        "error": message,
                        "processor_returned_without_terminal_status": True,
                    },
                )
                return True

            self._emit_job_completed(job, service_type_for_cache, _job_duration)

            return True
        except AuthError:
            self.orchestrator_service.signal_auth_expired()
            logging.warning(f"Auth expired during processing of job {job.get('id')}.")
            raise  # Re-raise to be caught by outer exception handler
        except InfrastructureError as e:
            self._requeue_job_after_infrastructure_error(
                job,
                e,
                reason=self._get_infrastructure_recovery_reason(e),
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
                and self._job_was_interrupted_by_infrastructure(
                    job_id,
                    job.get("execution_token"),
                )
            ):
                logging.info(
                    f"Job {job_id} raised after an infrastructure failure: {e}. "
                    "Skipping failure notification because the crash handler "
                    "already reset it."
                )
                return True

            terminal_status = self._get_terminal_job_status(job_id)
            if terminal_status:
                self.orchestrator_service.clear_active_job(job_id)
                logging.info(
                    f"Job {job_id} was externally terminated with status "
                    f"'{terminal_status}'. Clearing local job state without "
                    "emitting a failure notification."
                )
                self._emit_job_cleared(job_id, service_type_for_cache, terminal_status)
                return True

            if job and job.get("id") and self._is_requeueable_infrastructure_error(e):
                self._requeue_job_after_infrastructure_error(
                    job,
                    e,
                    reason=self._get_infrastructure_recovery_reason(e),
                )
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
                and not self._job_was_interrupted_by_infrastructure(
                    job_id,
                    job.get("execution_token"),
                )
                and not self._job_has_terminal_status(job_id)
            ):
                self.client.interrupted_job_id = job_id
                self.client.interrupted_job_execution_token = job.get("execution_token")
            self.client.current_job = None
            if job_id:
                with self._remote_stop_cleanup_lock:
                    self._remote_stop_cleanup_jobs.discard(job_id)

    def _cleanup_remote_cancelled_job(
        self,
        job_id: str,
        processor,
        remote_status: str,
    ) -> None:
        with self._remote_stop_cleanup_lock:
            if job_id in self._remote_stop_cleanup_jobs:
                logging.debug(
                    "Remote-stop cleanup for job %s already completed; "
                    "skipping duplicate cleanup.",
                    job_id,
                )
                return
            self._remote_stop_cleanup_jobs.add(job_id)

        self.orchestrator_service.clear_active_job(job_id)
        logging.warning(
            f"Job {job_id} received remote stop status '{remote_status}'. "
            "Interrupting local processing."
        )

        request_cancel = getattr(processor, "request_cancel", None)
        if callable(request_cancel):
            request_cancel()

        # Interrupt before stopping the container so ComfyUI can abort cleanly
        # while its HTTP endpoint is still alive.
        comfyui_client = getattr(processor, "comfyui_client", None)
        if comfyui_client:
            try:
                comfyui_client.interrupt_workflow(quiet_if_unreachable=True)
            except Exception as e:
                logging.debug(f"Could not interrupt workflow: {e}")

        # Stop any running Docker container for this job.
        if (
            docker_manager
            and hasattr(self.client, "active_service_type")
            and self.client.active_service_type
        ):
            # Stop the monitor first so it cannot misinterpret the intentional
            # container removal as an unexpected crash for the cancelled job.
            self._stop_container_monitor()
            try:
                logging.info(
                    f"Stopping Docker container for service "
                    f"'{self.client.active_service_type}' due to remote job "
                    f"stop status '{remote_status}'..."
                )
                docker_manager.stop_container(
                    service_type=self.client.active_service_type
                )
                self.client.active_service_type = None
                self.client.active_container_crash_event = None
            except Exception as e:
                logging.debug(f"Could not stop container: {e}")

        logging.info(
            f"Job {job_id} remote-stop cleanup completed "
            f"(status='{remote_status}')."
        )

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

        _CANCEL_POLL_BASE = 5        # seconds between checks (normal)
        _CANCEL_POLL_MAX = 60        # cap so a fleet doesn't hammer during outage
        _CANCEL_POLL_BACKOFF = 1.5   # multiplier per consecutive error

        def _monitor_loop():
            consecutive_errors = 0
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
                        consecutive_errors += 1
                    else:
                        consecutive_errors = 0
                except Exception as e:
                    logging.debug(f"Error checking job status: {e}")
                    consecutive_errors += 1

                interval = min(
                    _CANCEL_POLL_BASE * (_CANCEL_POLL_BACKOFF ** consecutive_errors),
                    _CANCEL_POLL_MAX,
                )
                self.shutdown_event.wait(timeout=interval)

        monitor_thread = threading.Thread(target=_monitor_loop, daemon=True)
        monitor_thread.start()

    def _is_compaction_pending(self) -> bool:
        return bool(getattr(self.client, "compaction_pending", False))

    def _skip_new_work_for_compaction(self, context: str) -> bool:
        if not self._is_compaction_pending():
            return False
        logging.info(
            "Disk compaction is pending; skipping new job/download work "
            f"before {context}."
        )
        return True

    def _fetch_next_job_with_priority(self):
        """
        Poll for the next job using priority ordering:
        1. Own jobs first (if process_own_jobs=True) — mine policy
        2. Community jobs second (if community_mode != 'none')
        3. Monetize jobs (if monetize_mode=True, separate track)

        Returns the first job found, or None.
        """
        client = self.client

        if self._skip_new_work_for_compaction("job reservation"):
            return None

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

        if self._skip_new_work_for_compaction("own-job reservation"):
            return None

        # Priority 1: own jobs (mine policy). Private mode still requires the
        # explicit process_own_jobs toggle; otherwise "none" means accept no jobs.
        community_mode = getattr(client, "community_mode", "none")
        if getattr(client, "process_own_jobs", False):
            job = self.orchestrator_service.get_next_job(
                provider_id=self.provider_id,
                accept_policy="mine",
                allowed_ids=client.allowed_ids,
                monetize_mode=False,
            )
            if job:
                return job

        if self._skip_new_work_for_compaction("community-job reservation"):
            return None

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
                    if self._skip_new_work_for_compaction("job fetch"):
                        job = None
                        continue
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
                                "execution_token": job.get("execution_token"),
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

    @staticmethod
    def _format_reservation_failure(response: Any) -> str:
        if not isinstance(response, dict):
            return "Unknown"

        reason = response.get("reason") or response.get("error") or "Unknown"
        message = response.get("message")
        details = response.get("details")

        parts = [str(reason)]
        if message and message != reason:
            parts.append(str(message))
        if isinstance(details, dict) and details:
            compact_details = ", ".join(
                f"{key}={value}" for key, value in sorted(details.items())
            )
            parts.append(f"details: {compact_details}")

        return " | ".join(parts)

    def _known_service_types(self) -> set:
        """Return the set of service types declared in this client's services.json.

        Server-supplied service_type strings (prefetch suggestions, peeked jobs)
        are validated against this set before any Docker operation is initiated.
        An unknown service_type from a compromised or MITM'd server response
        cannot trigger a Docker pull for an arbitrary image.
        """
        services_config = getattr(self.client, "services_config", None) or {}
        return set(services_config.keys())

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
        if (
            hasattr(download_manager, "is_post_download_hold_active")
            and download_manager.is_post_download_hold_active()
        ):
            logging.debug(
                "Skipping prefetch suggestions during post-download settle hold."
            )
            return

        community_mode = getattr(self.client, "community_mode", "all")
        monetize_mode = getattr(self.client, "monetize_mode", False)
        allow_network_prefetch = community_mode == "all" or monetize_mode
        if not allow_network_prefetch:
            logging.debug(
                f"Skipping global prefetch suggestions for private community_mode='{community_mode}'."
            )
            return
        if self._skip_new_work_for_compaction("prefetch suggestions"):
            return

        # Don't fetch suggestions if we already have downloads in progress
        # This prevents queue explosion and respects MAX_CONCURRENT_DOWNLOADS
        if download_manager._active_downloads or download_manager._download_queue:
            return

        known = self._known_service_types()

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
                    if self._skip_new_work_for_compaction("prefetch download"):
                        return
                    # Reject service_type values not present in the local services.json.
                    # A malicious or compromised server cannot trigger a Docker pull for
                    # an image that is not part of this client's declared service set.
                    if known and service_type not in known:
                        logging.warning(
                            f"Ignoring prefetch suggestion for unknown service_type "
                            f"'{service_type}' (not in local services.json)."
                        )
                        continue
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
        compaction_pause_logged = False

        try:
            while not self.shutdown_event.is_set():
                # Clear wakeup event at start of loop
                if hasattr(self.client, "job_wakeup_event"):
                    self.client.job_wakeup_event.clear()

                if getattr(self.client, "compaction_pending", False):
                    if download_manager and hasattr(
                        download_manager, "set_compaction_paused"
                    ):
                        download_manager.set_compaction_paused(True)
                    if not compaction_pause_logged:
                        logging.info(
                            "Disk compaction is pending; pausing job polling "
                            "and new Docker image downloads until it completes."
                        )
                        compaction_pause_logged = True
                    self.shutdown_event.wait(1.0)
                    continue

                if compaction_pause_logged:
                    logging.info(
                        "Disk compaction pause cleared; resuming job polling."
                    )
                    if download_manager and hasattr(
                        download_manager, "set_compaction_paused"
                    ):
                        download_manager.set_compaction_paused(False)
                    compaction_pause_logged = False

                # Ensure downloads are always allowed at the start of each iteration.
                # This is idempotent, so it safely clears any True state left over
                # by an exception that fired during the previous job's processing.
                # Exception: skip the unfreeze during the post-download settle window
                # so queued downloads for other service types don't restart while
                # Docker is still extracting overlay2 layers (disk at 100%).
                post_download_hold_active = False
                if download_manager:
                    if hasattr(download_manager, "is_post_download_hold_active"):
                        post_download_hold_active = (
                            download_manager.is_post_download_hold_active()
                        )
                    else:
                        recently_downloaded = getattr(download_manager, "_recently_downloaded", {})
                        post_download_hold_active = any(
                            time.time() - ts < POST_DOWNLOAD_SETTLE_SECS
                            for ts in recently_downloaded.values()
                        )
                    if post_download_hold_active:
                        logging.debug(
                            "Post-download settle window active: keeping download queue frozen "
                            f"for up to {POST_DOWNLOAD_SETTLE_SECS}s to allow overlay2 extraction."
                        )
                    else:
                        download_manager.set_job_active(False)

                logging.info("Top of auto-mode loop iteration.")
                job = None
                found_processable_job = False
                compaction_requested = False

                try:
                    # Acquire the processing lock BEFORE fetching a job
                    with self.client.processing_lock:
                        logging.info("Auto mode: Peeking at available jobs...")
                        if self._skip_new_work_for_compaction("auto-mode job peek"):
                            compaction_requested = True
                            continue

                        # Step 1: Peek at available jobs without reserving.
                        # Build an ordered list of (job, policy) pairs — own jobs first
                        # when process_own_jobs=True so they aren't starved by community work.
                        _community_mode = getattr(self.client, "community_mode", "all")
                        _process_own_jobs = getattr(self.client, "process_own_jobs", False)
                        _monetize_mode = getattr(self.client, "monetize_mode", False)
                        if _monetize_mode:
                            _peek_policy = "monetize"
                        elif _community_mode == "none" and _process_own_jobs:
                            _peek_policy = "mine"
                        elif _community_mode == "none":
                            _peek_policy = None
                        else:
                            _peek_policy = {
                                "trusted_users": "users",
                                "trusted_projects": "project",
                                "all": "all",
                            }.get(_community_mode, "all")

                        _available_with_policy: list = []
                        if _peek_policy is None:
                            logging.info(
                                "Routing is disabled (community_mode='none', "
                                "process_own_jobs=False); skipping job peek."
                            )
                        elif _process_own_jobs and _peek_policy not in ("mine", "monetize"):
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
                            _known_pass1 = self._known_service_types()
                            for peeked_job, _job_policy in _available_with_policy:
                                workflow_type = peeked_job.get("workflow_type")
                                if not workflow_type:
                                    logging.warning(
                                        "Skipping peeked job %s with missing workflow_type.",
                                        peeked_job.get("id"),
                                    )
                                    continue

                                if workflow_type not in getattr(
                                    self.client, "processor_map", {}
                                ):
                                    logging.warning(
                                        "Peeked job %s has no local processor for "
                                        "workflow_type '%s'. Refreshing DGN config once "
                                        "before skipping.",
                                        peeked_job.get("id"),
                                        workflow_type,
                                    )
                                    try:
                                        self.client.load_config()
                                    except Exception as exc:
                                        logging.warning(
                                            "Failed to refresh DGN config before "
                                            "processor check for workflow_type '%s': %s",
                                            workflow_type,
                                            exc,
                                        )

                                    if workflow_type not in getattr(
                                        self.client, "processor_map", {}
                                    ):
                                        logging.warning(
                                            "Skipping peeked job %s with unsupported "
                                            "workflow_type '%s'.",
                                            peeked_job.get("id"),
                                            workflow_type,
                                        )
                                        continue

                                service_type = self._get_service_type_for_job(
                                    peeked_job
                                )
                                if not service_type:
                                    continue

                                # Validate service_type against locally-declared
                                # services before any Docker or processing operation.
                                if _known_pass1 and service_type not in _known_pass1:
                                    logging.warning(
                                        f"Skipping peeked job with unknown service_type "
                                        f"'{service_type}' (not in local services.json)."
                                    )
                                    continue

                                # Check if Docker image is available
                                image_availability = ImageAvailability.AVAILABLE
                                if not HEADLESS_MODE and download_manager:
                                    image_availability = (
                                        download_manager.get_image_availability(
                                            service_type
                                        )
                                    )
                                    # When Docker returns UNKNOWN right after a
                                    # large download, the daemon is likely still
                                    # extracting overlay2 layers (disk at 100%).
                                    # Retry a few times with short sleeps before
                                    # giving up, so we don't skip a job whose
                                    # image is actually already present.
                                    if image_availability == ImageAvailability.UNKNOWN:
                                        recently_downloaded = getattr(
                                            download_manager, "_recently_downloaded", {}
                                        )
                                        if service_type in recently_downloaded:
                                            logging.info(
                                                f"Docker API returned UNKNOWN for '{service_type}' "
                                                "which was recently downloaded. Retrying image check "
                                                "to allow overlay2 extraction to settle..."
                                            )
                                            for _retry in range(3):
                                                if self.shutdown_event.is_set():
                                                    break
                                                self.shutdown_event.wait(10)
                                                image_availability = (
                                                    download_manager.get_image_availability(
                                                        service_type
                                                    )
                                                )
                                                if image_availability != ImageAvailability.UNKNOWN:
                                                    logging.info(
                                                        f"Image check for '{service_type}' resolved "
                                                        f"after {_retry + 1} retry(ies): {image_availability.value}"
                                                    )
                                                    break
                                                logging.info(
                                                    f"Image check for '{service_type}' still UNKNOWN "
                                                    f"(retry {_retry + 1}/3), Docker may still be busy."
                                                )
                                image_available = (
                                    image_availability == ImageAvailability.AVAILABLE
                                )

                                if image_available:
                                    # Image is ready - reserve and process this job
                                    if self._skip_new_work_for_compaction(
                                        "auto-mode job reservation"
                                    ):
                                        compaction_requested = True
                                        break

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

                                        # Suspend background downloads for the
                                        # duration of this job.  Competing disk
                                        # I/O and Docker layer extraction can
                                        # overwhelm the machine during generation.
                                        # Downloads resume at the top of the next
                                        # loop iteration via set_job_active(False).
                                        if download_manager:
                                            download_manager.set_job_active(True)

                                        workflow_type = job.get(
                                            "workflow_type", "image_to_video"
                                        )
                                        actual_service_type = (
                                            self._get_service_type_for_job(job)
                                        )
                                        if not actual_service_type:
                                            raise ValueError(
                                                "Unknown service type for workflow: "
                                                f"{workflow_type}"
                                            )
                                        previous_service_type = getattr(
                                            self.client, "active_service_type", None
                                        )
                                        if (
                                            previous_service_type
                                            and previous_service_type
                                            != actual_service_type
                                            and not HEADLESS_MODE
                                        ):
                                            previous_service_config = (
                                                self.client.services_config.get(
                                                    previous_service_type, {}
                                                )
                                            )
                                            if (
                                                previous_service_config.get("backend")
                                                != "local"
                                            ):
                                                logging.info(
                                                    "Switching from service '%s' to '%s'; stopping warm container.",
                                                    previous_service_type,
                                                    actual_service_type,
                                                )
                                                self._stop_container_monitor()
                                                docker_manager.stop_container(
                                                    service_type=previous_service_type
                                                )

                                        self.client.active_service_type = (
                                            actual_service_type
                                        )
                                        service_config = (
                                            self.client.services_config.get(
                                                actual_service_type, {}
                                            )
                                        )
                                        keep_container_warm = (
                                            self._should_keep_container_warm(
                                                service_config
                                            )
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
                                                "execution_token": job.get(
                                                    "execution_token"
                                                ),
                                            },
                                        )

                                        self.client.active_container_crash_event = None

                                        if not HEADLESS_MODE:
                                            logging.info("Starting container...")
                                            container_crash_event = threading.Event()
                                            self.client.active_container_crash_event = (
                                                container_crash_event
                                            )
                                            is_wan2gp = (
                                                service_config.get("backend") == "wan2gp"
                                            )
                                            is_ollama = (
                                                service_config.get("backend") == "ollama"
                                            )
                                            is_local = (
                                                service_config.get("backend") == "local"
                                            )

                                            if is_local:
                                                logging.info(
                                                    "Service '%s' uses local backend; skipping Docker container start.",
                                                    actual_service_type,
                                                )
                                            elif is_wan2gp:
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

                                                lowered_service_type = (
                                                    actual_service_type.lower()
                                                )
                                                if "davinci" in lowered_service_type:
                                                    if "16gb" in lowered_service_type:
                                                        wan2gp_env[
                                                            "WAN2GP_CLI_ARGS"
                                                        ] = (
                                                            "--profile 4.5 --attention sdpa "
                                                            "--perc-reserved-mem-max 0.45 "
                                                            "--vram-safety-coefficient 0.7"
                                                        )
                                                    elif "32gb" in lowered_service_type:
                                                        wan2gp_env[
                                                            "WAN2GP_CLI_ARGS"
                                                        ] = (
                                                            "--profile 4 --attention sdpa "
                                                            "--perc-reserved-mem-max 0.55 "
                                                            "--vram-safety-coefficient 0.80"
                                                        )
                                                    else:
                                                        wan2gp_env[
                                                            "WAN2GP_CLI_ARGS"
                                                        ] = (
                                                            "--profile 4.5 --attention sdpa "
                                                            "--perc-reserved-mem-max 0.45 "
                                                            "--vram-safety-coefficient 0.70"
                                                        )
                                                elif "8gb" in lowered_service_type:
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
                                                    force_restart=not keep_container_warm,
                                                )
                                                logging.info(
                                                    "Started wan2gp_server.py as the Wan2GP container main process."
                                                )
                                            elif actual_service_type in {
                                                "qwen",
                                                "qwen-8gb",
                                                "qwen-turbo-8gb",
                                            }:
                                                qwen_args = [
                                                    "--listen",
                                                ]
                                                qwen_env = {
                                                    "CUDA_MODULE_LOADING": "LAZY",
                                                    "PYTORCH_CUDA_ALLOC_CONF": (
                                                        "expandable_segments:True,"
                                                        "max_split_size_mb:128"
                                                    ),
                                                    "MALLOC_ARENA_MAX": "2",
                                                }
                                                if "8gb" in actual_service_type:
                                                    qwen_args.extend(
                                                        [
                                                            "--lowvram",
                                                            "--cpu-vae",
                                                            "--reserve-vram",
                                                            "1.0",
                                                            "--cache-none",
                                                            "--preview-method",
                                                            "none",
                                                            "--use-split-cross-attention",
                                                        ]
                                                    )
                                                qwen_bootstrap = r"""
set -e
export PYTHONPATH="${PYTHONPATH:-}:/opt/ComfyUI/user/custom-packages"
cd /opt/ComfyUI
python3 - <<'PY'
from pathlib import Path
import re

config_path = Path("/opt/ComfyUI/user/__manager/config.ini")
config_path.parent.mkdir(parents=True, exist_ok=True)
if config_path.exists():
    config = config_path.read_text(encoding="utf-8", errors="ignore")
else:
    config = "[default]\n"

if not re.search(r"(?m)^\s*\[default\]\s*$", config):
    config = "[default]\n" + config

if re.search(r"(?m)^\s*network_mode\s*=", config):
    config = re.sub(
        r"(?m)^\s*network_mode\s*=.*$",
        "network_mode = offline",
        config,
    )
else:
    config = config.rstrip() + "\nnetwork_mode = offline\n"

config_path.write_text(config, encoding="utf-8")
print("Configured ComfyUI-Manager network_mode=offline for job runtime.")
PY
exec python main.py "$@"
"""
                                                docker_manager.run_container(
                                                    service_type=actual_service_type,
                                                    command=[
                                                        "bash",
                                                        "-lc",
                                                        qwen_bootstrap,
                                                        "openfork-qwen",
                                                        *qwen_args,
                                                    ],
                                                    environment=qwen_env,
                                                    force_restart=not keep_container_warm,
                                                )
                                                logging.info(
                                                    "Started Qwen ComfyUI container without runtime SageAttention install."
                                                )
                                            elif is_ollama:
                                                docker_manager.run_container(
                                                    service_type=actual_service_type,
                                                    entrypoint=["ollama"],
                                                    command=["serve"],
                                                    environment={
                                                        "OLLAMA_HOST": "0.0.0.0:11434",
                                                        "OLLAMA_MODELS": "/root/.ollama",
                                                        "OLLAMA_LOAD_TIMEOUT": "10m",
                                                        "OLLAMA_MAX_LOADED_MODELS": "1",
                                                        "OLLAMA_NUM_PARALLEL": "1",
                                                        "OLLAMA_ORIGINS": "*",
                                                    },
                                                    volumes={
                                                        "openfork-ollama-models": {
                                                            "bind": "/root/.ollama",
                                                            "mode": "rw",
                                                        },
                                                    },
                                                    force_restart=not keep_container_warm,
                                                )
                                                logging.info(
                                                    "Started Ollama as the container main process."
                                                )
                                            else:
                                                docker_manager.run_container(
                                                    service_type=actual_service_type,
                                                    force_restart=not keep_container_warm,
                                                )

                                            if not is_local:
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
                                                # Register on self so every cleanup path
                                                # (error, cancellation, shutdown) can stop it
                                                # without relying on locals().
                                                self._active_container_monitor = container_monitor

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
                                            # Start log tailers for headless mode.
                                            # Wan2GP backends write to their own log.
                                            from utils.log_tailer import (
                                                LogTailer,
                                                get_headless_log_paths,
                                            )

                                            for log_path in get_headless_log_paths(
                                                actual_service_type
                                            ):
                                                tailer = LogTailer(
                                                    log_path,
                                                    service_type=actual_service_type,
                                                )
                                                log_thread = threading.Thread(
                                                    target=tailer.tail,
                                                    args=(self.shutdown_event,),
                                                    daemon=True,
                                                )
                                                log_thread.start()

                                        # Check if service uses ComfyUI backend from configuration
                                        backend = service_config.get(
                                            "backend", "comfyui"
                                        )
                                        uses_comfyui = backend == "comfyui"
                                        logging.info(
                                            "Service '%s' uses backend '%s'.",
                                            actual_service_type,
                                            backend,
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
                                            logging.info(
                                                "Skipping ComfyUI readiness wait for "
                                                "backend '%s'; processor will manage "
                                                "its own readiness checks.",
                                                backend,
                                            )
                                            ollama_warmed = True
                                            if backend == "ollama":
                                                ollama_warmed = (
                                                    self._warm_ollama_model_in_container(
                                                        job,
                                                        actual_service_type,
                                                    )
                                                )

                                            if not ollama_warmed:
                                                logging.info(
                                                    "Skipping job processor because "
                                                    "Ollama warmup did not complete."
                                                )
                                            else:
                                                self._process_job_safely(job)

                                        if not self.shutdown_event.is_set():
                                            logging.info(f"Job processing finished.")
                                            if not HEADLESS_MODE:
                                                # Stop the per-job crash monitor before any
                                                # intentional container cleanup.
                                                self._stop_container_monitor()
                                                if service_config.get("backend") == "local":
                                                    self.client.active_service_type = None
                                                elif keep_container_warm:
                                                    logging.info(
                                                        "Keeping container for service '%s' warm for subsequent jobs.",
                                                        actual_service_type,
                                                    )
                                                elif (
                                                    getattr(self.client, "active_service_type", None)
                                                    == actual_service_type
                                                ):
                                                    logging.info(
                                                        "Stopping container for service '%s' after job; warm reuse is not enabled for this service.",
                                                        actual_service_type,
                                                    )
                                                    docker_manager.stop_container(
                                                        service_type=actual_service_type
                                                    )
                                                    self._wait_after_container_cleanup(
                                                        actual_service_type
                                                    )
                                                    self.client.active_service_type = None
                                                else:
                                                    logging.info(
                                                        "Container for service '%s' was already stopped during job cleanup.",
                                                        actual_service_type,
                                                    )
                                            else:
                                                self.client.active_service_type = None
                                            self.client.active_container_crash_event = None
                                            self.orchestrator_service.update_provider_status(
                                                self.provider_id, "available"
                                            )
                                            logging.info(
                                                "Provider status set to available. Waiting for next job..."
                                            )
                                    else:
                                        # Reservation failed - job may have been taken by another provider
                                        error_msg = self._format_reservation_failure(job)
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
                                # MISSING images: deferred to Pass 2 below.

                            # Pass 2: start downloads for missing images, but only when
                            # Pass 1 found no cached job to process. This prevents a
                            # cached job sitting later in the FIFO peek list from being
                            # blocked by downloads kicked off for earlier miss entries,
                            # and avoids unnecessary evictions caused by those downloads.
                            if (
                                not found_processable_job
                                and not compaction_requested
                                and download_manager
                                and not HEADLESS_MODE
                            ):
                                if post_download_hold_active:
                                    logging.info(
                                        "Skipping missing-image downloads during the "
                                        "post-download settle window so the freshly "
                                        "downloaded image can be checked first."
                                    )
                                else:
                                    _known = self._known_service_types()
                                    for peeked_job, _job_policy in _available_with_policy:
                                        if self._skip_new_work_for_compaction(
                                            "background image download"
                                        ):
                                            compaction_requested = True
                                            break

                                        service_type = self._get_service_type_for_job(peeked_job)
                                        if not service_type:
                                            continue
                                        # Validate against locally-declared services before
                                        # any Docker operation; rejects server-injected types.
                                        if _known and service_type not in _known:
                                            logging.warning(
                                                f"Ignoring peeked job with unknown service_type "
                                                f"'{service_type}' (not in local services.json)."
                                            )
                                            continue
                                        image_availability = download_manager.get_image_availability(
                                            service_type
                                        )
                                        if image_availability != ImageAvailability.MISSING:
                                            continue
                                        status = download_manager.get_download_status(service_type)
                                        is_downloading = download_manager.is_downloading(service_type)
                                        is_queued = download_manager.is_queued(service_type)
                                        if not is_downloading and not is_queued:
                                            if not self._should_download_missing_image(
                                                service_type, _job_policy, download_gate
                                            ):
                                                continue
                                            if status and status.value == "permanently_failed":
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
                                                "Starting background download for peeked job..."
                                            )
                                            download_manager.start_background_download(
                                                service_type,
                                                accept_policy=_job_policy,
                                            )
                                        else:
                                            logging.debug(
                                                f"Image for service '{service_type}' already downloading/queued "
                                                f"(status: {status.value if status else 'unknown'})."
                                            )

                        # Handle pre-fetch suggestions when idle
                        # This proactively downloads images for high-demand workflows
                        # NOTE: This does NOT affect credits - it's purely for network efficiency
                        if not found_processable_job and not compaction_requested:
                            if self._skip_new_work_for_compaction(
                                "idle downloads and cleanup"
                            ):
                                compaction_requested = True
                            elif post_download_hold_active:
                                logging.info(
                                    "Skipping queued and prefetch downloads during the "
                                    "post-download settle window."
                                )
                            else:
                                if download_manager and hasattr(
                                    download_manager, "start_next_queued_download"
                                ):
                                    if download_manager.start_next_queued_download(
                                        reason="no-processable-job"
                                    ):
                                        logging.info(
                                            "Started queued Docker image download after "
                                            "no processable job was found."
                                        )

                                self._handle_prefetch_suggestions(download_manager)

                            # Disk-pressure cleanup runs in the same idle window as
                            # prefetching. Cheap when at Healthy (one disk_usage call),
                            # only evicts at Pressure / Critical tiers.
                            if download_manager:
                                try:
                                    if not compaction_requested:
                                        download_manager.check_and_evict_for_pressure()
                                except Exception as pressure_err:
                                    logging.debug(
                                        f"check_and_evict_for_pressure failed: {pressure_err}"
                                    )

                        if not found_processable_job and not compaction_requested:
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
                        self._requeue_job_after_infrastructure_error(
                            job,
                            e,
                            reason=self._get_infrastructure_recovery_reason(e),
                        )
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

                if compaction_requested and self._is_compaction_pending():
                    consecutive_empty_polls = 0
                    continue

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
                # Always stop the monitor before the container — even on shutdown —
                # so a lingering monitor thread cannot misreport the clean removal
                # as a container crash.
                self._stop_container_monitor()
                logging.info(
                    f"Ensuring container for service '{self.client.active_service_type}' is stopped."
                )
                docker_manager.stop_container(
                    service_type=self.client.active_service_type
                )
                self.client.active_service_type = None
