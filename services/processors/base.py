"""
Base Job Processor

Abstract base class for all job processors in the DGN client.
"""

import os
import json
import logging
import threading
from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional

from exceptions import WorkflowError


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
        logging.error(message)
        self.orchestrator_service.update_job_status(
            self.job_id,
            "failed",
            completion_metadata={"error": message},
        )

    def close(self) -> None:
        """Release any resources held by this processor (e.g. HTTP sessions)."""
        if hasattr(self, "session") and self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass

    @abstractmethod
    def process(self) -> None:
        """Process the job. Must be implemented by subclasses."""
        pass
