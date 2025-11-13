import os
import logging
import threading
import requests

from services.orchestrator_service import OrchestratorService
from services.comfyui_service import ComfyUIClient
from services.docker_manager import docker_manager
import services.job_processors as job_processors_module


class DGNClient:
    def __init__(self, orchestrator_url: str, root_dir: str, data_dir: str, access_token: str, refresh_token: str, accept_policy: str = 'all', allowed_targets: list[str] = None):
        self.orchestrator_url = orchestrator_url
        self.orchestrator_service = OrchestratorService(orchestrator_url, access_token, refresh_token)
        self.comfyui_client = ComfyUIClient(os.environ.get("COMFYUI_WS_URL", "ws://127.0.0.1:8188/ws?clientId={}"))
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.cache_dir = os.path.join(data_dir, ".cache")
        self.input_dir = os.path.join(data_dir, "input")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.active_service_type = None
        self.current_job = None
        self.config = {}
        self.docker_image_map = {}
        self.accept_policy = accept_policy
        self.allowed_targets = allowed_targets or []
        self.allowed_ids = []
        
        self.processor_map = {}

        if self.accept_policy == 'mine':
            user_id = self.orchestrator_service._get_user_id_from_token()
            if user_id:
                self.allowed_ids.append(user_id)
        elif (self.accept_policy == 'project' or self.accept_policy == 'users') and self.allowed_targets:
            target_type = 'project' if self.accept_policy == 'project' else 'user'
            logging.info(f"Resolving {target_type} targets: {self.allowed_targets}")
            self.allowed_ids = self.orchestrator_service.resolve_targets(self.allowed_targets, target_type)
            logging.info(f"Resolved targets to IDs: {self.allowed_ids}")

    def _build_processor_map(self):
        proc_map = {}
        for workflow_type, config in self.config.items():
            processor_name = config.get("processor")
            if processor_name:
                processor_class = getattr(job_processors_module, processor_name, None)
                if processor_class:
                    proc_map[workflow_type] = processor_class
                else:
                    logging.warning(f"Processor class '{processor_name}' not found for workflow '{workflow_type}'")
        return proc_map

    def load_config(self):
        """Loads the configuration from the orchestrator."""
        try:
            config_url = f"{self.orchestrator_url}/api/config"
            logging.info(f"Fetching DGN configuration from {config_url}")
            response = requests.get(config_url)
            response.raise_for_status()
            
            self.config = response.json()
            
            # Create a map from service_name to prod_image for docker_manager
            unique_services = {
                (config["service_name"], config["prod_image"])
                for config in self.config.values() if 'service_name' in config and 'prod_image' in config
            }
            self.docker_image_map = {service_name: image for service_name, image in unique_services}

            docker_manager.set_docker_image_map(self.docker_image_map)
            
            self.processor_map = self._build_processor_map()

            logging.info("DGN configuration loaded successfully from orchestrator.")

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch configuration from orchestrator: {e}")
            raise

    def get_service_type_for_workflow(self, workflow_type: str) -> str:
        """Maps a workflow type to a service type using the loaded config."""
        if workflow_type not in self.config:
            raise ValueError(f"Unknown workflow type, cannot determine service: {workflow_type}")
        return self.config[workflow_type]["service_name"]

    def _get_job_processor(self, job, shutdown_event):
        workflow_type = job.get('workflow_type')

        ProcessorClass = self.processor_map.get(workflow_type)
        if not ProcessorClass:
            raise ValueError(f"No job processor found for workflow type: {workflow_type}")
        return ProcessorClass(self, job, shutdown_event)

    def _process_job(self, job, shutdown_event: threading.Event):
        """Processes a single DGN job by delegating to a specific job processor."""
        try:
            processor = self._get_job_processor(job, shutdown_event)
            processor.process()
        except Exception as e:
            logging.error(f"An error occurred while processing job {job.get('id')}: {e}", exc_info=True)
            if job and job.get('id'):
                self.orchestrator_service.update_job_status(job.get('id'), 'failed')

if __name__ == "__main__":
    from cli import main
    import sys
    from utils.shutdown_handler import SHUTDOWN_EVENT
    
    try:
        main()
        logging.info("Program exiting normally.")
        sys.exit(0)
    except KeyboardInterrupt:
        SHUTDOWN_EVENT.set()
        logging.info("Process interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"An unhandled exception occurred: {e}", exc_info=True)
        sys.exit(1)