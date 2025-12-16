"""
Base Job Processor

Abstract base class for all job processors in the DGN client.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Union, Dict


class BaseJobProcessor(ABC):
    """Base class for all job processors."""

    def __init__(self, client, job, shutdown_event):
        self.client = client
        self.orchestrator_service = client.orchestrator_service
        self.job = job
        self.job_id = job["id"]
        self.shutdown_event = shutdown_event
        self.root_dir = client.root_dir
        self.input_dir = client.input_dir
        self.cache_dir = client.cache_dir
        self.positive_prompt = job.get("prompt") or ""
        self.negative_prompt = job.get("negative_prompt") or ""
        self.workflow_type = job.get("workflow_type")

    @property
    def workflow_file(self) -> str:
        """The filename of the workflow to be used for this processor."""
        if self.workflow_type in self.client.config:
            return self.client.config[self.workflow_type]["workflow_file"]
        raise ValueError(f"Workflow file not found for type {self.workflow_type}")

    def _get_workflow_payload(self) -> Union[Dict, None]:
        """Loads the workflow from the local filesystem."""
        try:
            local_filename = self.workflow_file
        except ValueError as e:
            logging.error(f"Cannot get workflow payload for job {self.job_id}: {e}")
            self.orchestrator_service.update_job_status(self.job_id, "failed")
            return None

        workflow_path = os.path.join(self.root_dir, "workflows", local_filename)

        logging.info(f"Loading local workflow for job {self.job_id} from: {workflow_path}")

        try:
            import json
            with open(workflow_path, "r") as f:
                workflow_data = json.load(f)
            return workflow_data
        except FileNotFoundError:
            logging.error(f"Local workflow file not found: {workflow_path}")
            self.orchestrator_service.update_job_status(self.job_id, "failed")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading local workflow {local_filename}: {e}")
            self.orchestrator_service.update_job_status(self.job_id, "failed")
            return None

    def _check_interruption(self, outputs):
        """Check if processing was interrupted."""
        if outputs == "interrupted":
            logging.warning(f"Processing of job {self.job_id} was interrupted.")
            return True
        return False

    def _fail_job(self, message: str):
        """Mark job as failed with a log message."""
        logging.error(message)
        self.orchestrator_service.update_job_status(self.job_id, "failed")

    @abstractmethod
    def process(self):
        """Process the job. Must be implemented by subclasses."""
        pass
