import os
import logging
import threading
import requests

import json
from services.orchestrator_service import OrchestratorService, TokenExpiredError
from services.comfyui_service import ComfyUIClient
from services.docker_manager import docker_manager
from services.docker_download_manager import DockerDownloadManager
from services.hardware_profiler import get_available_vram, can_run_service, get_vram_requirement_display, get_service_incompatibility_reason, get_available_system_ram
import services.processors as job_processors_module


class DGNClient:
    def __init__(
        self,
        orchestrator_url: str,
        root_dir: str,
        data_dir: str,
        access_token: str = None,
        refresh_token: str = None,
        dgn_api_key: str = None,
        accept_policy: str = "all",
        allowed_targets: list[str] = None,
        monetize_mode: bool = False,
    ):
        self.orchestrator_url = orchestrator_url
        self.orchestrator_service = OrchestratorService(
            orchestrator_url, access_token, refresh_token, dgn_api_key=dgn_api_key
        )
        self.comfyui_client = ComfyUIClient(
            os.environ.get("COMFYUI_WS_URL", "ws://127.0.0.1:8188/ws?clientId={}"),
            access_token=access_token,
        )
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
        self.monetize_mode = monetize_mode
        self.allowed_targets = allowed_targets or []
        self.allowed_ids = []
        self.processing_lock = threading.Lock()

        self.processor_map = {}
        self.services_config = {}
        self.available_vram = get_available_vram()
        self.compatible_services = set()
        
        # Shared event to wake up the job listener immediately when a download completes
        self.job_wakeup_event = threading.Event()
        
        # Initialize download manager for Docker image pre-fetching (only when not headless)
        self.download_manager = DockerDownloadManager(docker_manager, wakeup_event=self.job_wakeup_event) if docker_manager else None

        if self.accept_policy == "mine" and not self.orchestrator_service.use_api_key:
            user_id = self.orchestrator_service._get_user_id_from_token()
            if user_id:
                self.allowed_ids.append(user_id)
        elif (
            self.accept_policy == "project" or self.accept_policy == "users"
        ) and self.allowed_targets:
            target_type = "project" if self.accept_policy == "project" else "user"
            logging.info(f"Resolving {target_type} targets: {self.allowed_targets}")
            self.allowed_ids = self.orchestrator_service.resolve_targets(
                self.allowed_targets, target_type
            )
            logging.info(f"Resolved targets to IDs: {self.allowed_ids}")

    def _build_processor_map(self):
        proc_map = {}
        # Auto-register LLMJobProcessor for 'llm' workflow
        # This allows it to work even if not explicitly in the server config yet
        if hasattr(job_processors_module, "LLMJobProcessor"):
            proc_map["llm"] = (
                job_processors_module.LLMJobProcessor
            )

        for workflow_type, config in self.config.items():
            processor_name = config.get("processor")
            if processor_name:
                processor_class = getattr(job_processors_module, processor_name, None)
                if processor_class:
                    proc_map[workflow_type] = processor_class
                else:
                    logging.warning(
                        f"Processor class '{processor_name}' not found for workflow '{workflow_type}'"
                    )
        return proc_map

    def load_config(self):
        """Loads the configuration from the orchestrator."""
        try:
            config_url = f"{self.orchestrator_url}/api/config"
            logging.info(f"Fetching DGN configuration from {config_url}")
            response = requests.get(config_url, timeout=15)
            response.raise_for_status()

            full_config = response.json()
            
            # Extract workflows and services from the config
            self.config = full_config.get("workflows", full_config)
            self.services_config = full_config.get("services", {})

            # Create a map from service_name to prod_image for docker_manager
            unique_services = {
                (config["service_name"], config["prod_image"])
                for config in self.config.values()
                if "service_name" in config and "prod_image" in config
            }
            self.docker_image_map = {
                service_name: image for service_name, image in unique_services
            }

            # Only configure docker_manager if not in headless mode
            if docker_manager:
                docker_manager.set_docker_image_map(self.docker_image_map)
                docker_manager.set_services_config(self.services_config)

            self.processor_map = self._build_processor_map()
            
            # Check VRAM compatibility for each service
            self.compatible_services = set()
            for service_name, service_config in self.services_config.items():
                if can_run_service(service_config, self.available_vram):
                    self.compatible_services.add(service_name)

            
            if self.compatible_services:
                from services.disk_space_utils import get_available_disk_space
                system_ram_mb = get_available_system_ram()
                available_disk_bytes = get_available_disk_space()
                logging.info(f"GPU VRAM: {get_vram_requirement_display(self.available_vram)}, System RAM: {get_vram_requirement_display(system_ram_mb)}, Disk Free: {available_disk_bytes / (1024**3):.1f}GB")
                logging.info(f"Compatible services: {', '.join(sorted(self.compatible_services))}")

            else:
                logging.warning("No compatible services found for current hardware!")
            
            # Log incompatible services for debugging
            incompatible_services = set(self.services_config.keys()) - self.compatible_services
            if incompatible_services:
                logging.debug("Incompatible services:")
                for service_name in sorted(incompatible_services):
                    reason = get_service_incompatibility_reason(self.services_config[service_name], self.available_vram)
                    if reason:
                        logging.debug(f"  - {service_name}: {reason}")
                    else:
                        logging.debug(f"  - {service_name}: unknown incompatibility")

            logging.info("DGN configuration loaded successfully from orchestrator.")
            
            # Note: Hardware-based job filtering is handled server-side in the SQL function
            # get_and_assign_next_dgn_job(). Jobs are only assigned to providers whose
            # supported_services array (calculated at registration) contains the job's service_type.
            # The client-side compatible_services provides additional validation for safety.

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch configuration from orchestrator: {e}")
            raise

    def get_service_type_for_workflow(self, workflow_type: str) -> str:
        """Maps a workflow type to a service type using the loaded config."""
        if workflow_type not in self.config:
            raise ValueError(
                f"Unknown workflow type, cannot determine service: {workflow_type}"
            )
        return self.config[workflow_type]["service_name"]

    def _get_job_processor(self, job, shutdown_event):
        workflow_type = job.get("workflow_type")

        ProcessorClass = self.processor_map.get(workflow_type)
        if not ProcessorClass:
            raise ValueError(
                f"No job processor found for workflow type: {workflow_type}"
            )
        return ProcessorClass(self, job, shutdown_event)

    # NOTE: _process_job was removed (it was deprecated and lacked lock management).
    # All job processing flows through JobListener._process_job_safely, which
    # correctly holds self.processing_lock before fetching and processing each job.


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
