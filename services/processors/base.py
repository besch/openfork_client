"""
Base Job Processor

Abstract base class for all job processors in the DGN client.
"""

import os
import json
import logging
import posixpath
import shutil
import threading
from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional

from exceptions import WorkflowError

_ALLOWED_CONTAINER_CLEANUP_PREFIXES = (
    "/opt/ComfyUI/input/",
    "/opt/ComfyUI/output/",
    "/opt/wan2gp/outputs/",
    "/opt/TurboDiffusion/outputs/",
    "/app/output/",
    "/data/inspatio_tasks/",
    "/tmp/",
)

_JOB_SCOPED_CONTAINER_CLEANUP_DIRS = (
    "/opt/ComfyUI/input",
    "/opt/ComfyUI/output",
    "/app/output",
    "/tmp",
)


class BaseJobProcessor(ABC):
    """Base class for all job processors.
    
    Attributes:
        client: The DGN client instance
        orchestrator_service: Service for communicating with the orchestrator
        job: The job dictionary from the orchestrator
        job_id: Unique identifier for this job
        shutdown_event: Event to signal shutdown
        root_dir: Root directory for the client
        input_dir: Directory for input files
        cache_dir: Directory for cached files
        positive_prompt: The positive prompt for generation
        negative_prompt: The negative prompt for generation
        workflow_type: Type of workflow to execute
    """

    def __init__(
        self,
        client,
        job: Dict[str, Any],
        shutdown_event: threading.Event
    ) -> None:
        self.client = client
        self.orchestrator_service = client.orchestrator_service
        self.job = job
        self.job_id: str = job["id"]
        self.shutdown_event = shutdown_event
        self.root_dir: str = client.root_dir
        self.input_dir: str = client.input_dir
        self.cache_dir: str = client.cache_dir
        self.positive_prompt: str = job.get("prompt") or ""
        self.negative_prompt: str = job.get("negative_prompt") or ""
        self.workflow_type: Optional[str] = job.get("workflow_type")
        self.cancel_event = threading.Event()

    def request_cancel(self) -> None:
        """Signal that this single job was cancelled remotely."""
        self.cancel_event.set()

    def is_cancelled(self) -> bool:
        """True when this job should stop without shutting down the client."""
        container_crash_event = getattr(
            self.client,
            "active_container_crash_event",
            None,
        )
        return (
            self.shutdown_event.is_set()
            or self.cancel_event.is_set()
            or (
                container_crash_event is not None
                and container_crash_event.is_set()
            )
        )

    @property
    def workflow_file(self) -> str:
        """The filename of the workflow to be used for this processor."""
        if self.workflow_type in self.client.config:
            return self.client.config[self.workflow_type]["workflow_file"]
        raise ValueError(f"Workflow file not found for type {self.workflow_type}")

    def _get_workflow_payload(self) -> Optional[Dict[str, Any]]:
        """Loads the workflow from the local filesystem.
        
        Returns:
            Workflow data dictionary, or None if loading failed.
            
        Raises:
            WorkflowError: If the workflow type is unknown (but handled internally).
        """
        try:
            local_filename = self.workflow_file
        except ValueError as e:
            self._fail_job(f"Cannot get workflow payload for job {self.job_id}: {e}")
            return None

        workflow_path = os.path.join(self.root_dir, "workflows", local_filename)

        logging.info(f"Loading local workflow for job {self.job_id} from: {workflow_path}")

        try:
            with open(workflow_path, "r") as f:
                workflow_data = json.load(f)
            return workflow_data
        except FileNotFoundError:
            self._fail_job(f"Local workflow file not found: {workflow_path}")
            return None
        except json.JSONDecodeError as e:
            self._fail_job(f"Invalid JSON in workflow file {local_filename}: {e}")
            return None
        except Exception as e:
            self._fail_job(
                f"An unexpected error occurred while loading local workflow {local_filename}: {e}"
            )
            return None

    def _check_interruption(self, outputs: Any) -> bool:
        """Check if processing was interrupted.
        
        Args:
            outputs: The workflow outputs (may be 'interrupted' string)
            
        Returns:
            True if processing was interrupted, False otherwise.
        """
        if outputs == "interrupted":
            logging.warning(f"Processing of job {self.job_id} was interrupted.")
            return True
        return False

    def _fail_job(self, message: str) -> None:
        """Mark job as failed with a log message.
        
        Args:
            message: Error message to log
        """
        if self.is_cancelled():
            logging.info(
                f"Skipping failure update for cancelled job {self.job_id}: {message}"
            )
            return

        logging.error(message)
        self.orchestrator_service.update_job_status(
            self.job_id,
            "failed",
            completion_metadata={"error": message},
        )

    def _cleanup_local_file(self, path: Optional[str], label: str = "temporary file") -> None:
        if not path:
            return
        try:
            if os.path.isfile(path):
                os.remove(path)
                logging.info("Cleaned up %s: %s", label, path)
        except OSError as exc:
            logging.debug("Could not clean up %s %s: %s", label, path, exc)

    def _cleanup_container_file(
        self,
        path: Optional[str],
        label: str = "container temporary file",
        recursive: bool = False,
    ) -> None:
        if not path:
            return

        normalized = posixpath.normpath(path.replace("\\", "/"))
        allowed_prefix = next(
            (
                prefix
                for prefix in _ALLOWED_CONTAINER_CLEANUP_PREFIXES
                if normalized.startswith(prefix)
            ),
            None,
        )
        if not allowed_prefix:
            logging.debug("Skipping cleanup for unexpected container path: %s", path)
            return

        if recursive and normalized.rstrip("/") == allowed_prefix.rstrip("/"):
            logging.debug("Skipping recursive cleanup for broad container path: %s", path)
            return

        try:
            from config import HEADLESS_MODE
            from services.docker_manager import docker_manager
        except Exception as exc:
            logging.debug("Could not import cleanup helpers for %s: %s", path, exc)
            return

        if HEADLESS_MODE:
            if recursive and os.path.isdir(normalized) and not os.path.islink(normalized):
                try:
                    shutil.rmtree(normalized)
                    logging.info("Cleaned up %s: %s", label, normalized)
                except OSError as exc:
                    logging.debug("Could not clean up %s %s: %s", label, normalized, exc)
            else:
                self._cleanup_local_file(normalized, label)
            return

        active_service_type = getattr(self.client, "active_service_type", None)
        if not active_service_type or not docker_manager:
            return

        try:
            result = docker_manager.exec_in_container(
                active_service_type,
                ["rm", "-rf" if recursive else "-f", "--", normalized],
            )
            if result is None:
                logging.debug(
                    "Container cleanup skipped because service '%s' is no longer reachable: %s",
                    active_service_type,
                    normalized,
                )
                return
            exit_code = getattr(result, "exit_code", None)
            if exit_code is None:
                exit_code = getattr(result, "returncode", None)
            if exit_code not in (None, 0):
                logging.debug(
                    "Container cleanup for %s exited with %s: %s",
                    normalized,
                    exit_code,
                    getattr(result, "output", ""),
                )
            else:
                logging.info("Requested cleanup for %s: %s", label, normalized)
        except Exception as exc:
            logging.debug("Could not clean up %s %s: %s", label, normalized, exc)

    def _cleanup_job_scoped_temp_files(self) -> None:
        job_id = self.job_id
        if not job_id:
            return

        prefixes = (f"{job_id}_", f"{job_id}.", f"{job_id}-", f"start_{job_id}")
        for directory in (self.input_dir, self.cache_dir):
            if not directory or not os.path.isdir(directory):
                continue
            try:
                for entry in os.scandir(directory):
                    if not entry.is_file():
                        continue
                    if entry.name.startswith(prefixes):
                        self._cleanup_local_file(entry.path)
            except OSError as exc:
                logging.debug(
                    "Could not scan %s for job-scoped temp cleanup: %s",
                    directory,
                    exc,
                )

    def _cleanup_job_scoped_files_in_local_dir(
        self,
        directory: str,
        prefixes: tuple[str, ...],
        *,
        max_depth: int = 2,
    ) -> None:
        if not directory or not os.path.isdir(directory):
            return

        root_depth = directory.rstrip(os.sep).count(os.sep)
        try:
            for root, dirs, files in os.walk(directory):
                depth = root.rstrip(os.sep).count(os.sep) - root_depth
                if depth >= max_depth:
                    dirs[:] = []
                for filename in files:
                    if filename.startswith(prefixes):
                        self._cleanup_local_file(os.path.join(root, filename))
        except OSError as exc:
            logging.debug(
                "Could not scan %s for job-scoped container cleanup: %s",
                directory,
                exc,
            )

    def _cleanup_job_scoped_container_files(self) -> None:
        job_id = self.job_id
        if not job_id:
            return

        prefixes = (f"{job_id}_", f"{job_id}.", f"{job_id}-", f"start_{job_id}")

        try:
            from config import HEADLESS_MODE
            from services.docker_manager import docker_manager
        except Exception as exc:
            logging.debug("Could not import container cleanup helpers: %s", exc)
            return

        if HEADLESS_MODE:
            for directory in _JOB_SCOPED_CONTAINER_CLEANUP_DIRS:
                self._cleanup_job_scoped_files_in_local_dir(directory, prefixes)
            return

        active_service_type = getattr(self.client, "active_service_type", None)
        if not active_service_type or not docker_manager:
            return

        for directory in _JOB_SCOPED_CONTAINER_CLEANUP_DIRS:
            for prefix in prefixes:
                try:
                    result = docker_manager.exec_in_container(
                        active_service_type,
                        [
                            "find",
                            directory,
                            "-maxdepth",
                            "2",
                            "-type",
                            "f",
                            "-name",
                            f"{prefix}*",
                            "-delete",
                        ],
                    )
                    if result is None:
                        return
                    exit_code = getattr(result, "exit_code", None)
                    if exit_code is None:
                        exit_code = getattr(result, "returncode", None)
                    if exit_code not in (None, 0):
                        logging.debug(
                            "Job-scoped container cleanup for %s/%s exited with %s: %s",
                            directory,
                            prefix,
                            exit_code,
                            getattr(result, "output", ""),
                        )
                except Exception as exc:
                    logging.debug(
                        "Could not clean job-scoped files in %s for prefix %s: %s",
                        directory,
                        prefix,
                        exc,
                    )

    def close(self) -> None:
        """Release any resources held by this processor (e.g. HTTP sessions)."""
        if hasattr(self, "session") and self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass
        self._cleanup_job_scoped_container_files()
        self._cleanup_job_scoped_temp_files()

    @abstractmethod
    def process(self) -> None:
        """Process the job. Must be implemented by subclasses."""
        pass
