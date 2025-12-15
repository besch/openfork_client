import os
import logging
import threading
import requests
import json

from services.orchestrator_service import OrchestratorService, TokenExpiredError
from services.comfyui_service import ComfyUIClient
from services.docker_comfyui_manager import DockerComfyUIManager
import services.job_processors as job_processors_module


class DGNClient:
    """DGN Client for processing AI generation jobs using Docker-based ComfyUI."""
    
    def __init__(
        self,
        orchestrator_url: str,
        root_dir: str,
        data_dir: str,
        access_token: str,
        refresh_token: str,
        accept_policy: str = "all",
        allowed_targets: list[str] = None,
        docker_image: str = None,
        host_port: int = 8188,
    ):
        self.orchestrator_url = orchestrator_url
        self.orchestrator_service = OrchestratorService(
            orchestrator_url, access_token, refresh_token
        )
        
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.cache_dir = os.path.join(data_dir, ".cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize Docker ComfyUI manager
        logging.info("Initializing Docker-based ComfyUI manager...")
        self.comfyui_manager = DockerComfyUIManager(
            image=docker_image,
            host_port=host_port,
            data_dir=data_dir,
        )
        
        # For backward compatibility - some code may reference docker_comfyui_manager
        self.docker_comfyui_manager = self.comfyui_manager
        
        # Build ComfyUI URL from host port
        comfyui_url = f"http://127.0.0.1:{host_port}"
        
        # Setup WebSocket client for ComfyUI
        ws_url = comfyui_url.replace("http://", "ws://").replace("https://", "wss://")
        if not ws_url.endswith("/ws"):
            ws_url += "/ws?clientId={}"
        else:
            if "clientId={}" not in ws_url:
                ws_url += "?clientId={}"

        self.comfyui_client = ComfyUIClient(
            ws_url,
            access_token=access_token,
        )
        
        # Start ComfyUI Docker container
        logging.info("Starting ComfyUI Docker container...")
        if not self.comfyui_manager.start_container():
            logging.error("Failed to start ComfyUI container!")
        
        # Set up input directory (from Docker volume mount)
        self.input_dir = self.comfyui_manager.get_input_directory()
        logging.info(f"Using input directory: {self.input_dir}")
        
        # Job tracking
        self.active_service_type = None
        self.current_job = None
        self.config = {}
        self.accept_policy = accept_policy
        self.allowed_targets = allowed_targets or []
        self.allowed_ids = []
        self.processing_lock = threading.Lock()
        
        # Scan available workflows from Docker volume
        self.available_workflows = self.comfyui_manager.get_all_workflows()
        logging.info(f"Found {len(self.available_workflows)} workflows.")
        
        # Build processor map
        self.processor_map = {}
        
        # Resolve allowed targets based on policy
        if self.accept_policy == "mine":
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
        """Build processor map with available processors."""
        proc_map = {}
        
        # TextGenerationJobProcessor for Ollama-based text generation
        if hasattr(job_processors_module, "TextGenerationJobProcessor"):
            proc_map["text_generation"] = job_processors_module.TextGenerationJobProcessor
        
        # GenericComfyWorkflowProcessor handles ALL ComfyUI workflows
        if hasattr(job_processors_module, "GenericComfyWorkflowProcessor"):
            proc_map["comfy-workflow"] = job_processors_module.GenericComfyWorkflowProcessor
        
        return proc_map

    def load_config(self):
        """Loads the configuration from the orchestrator."""
        try:
            config_url = f"{self.orchestrator_url}/api/config"
            logging.info(f"Fetching DGN configuration from {config_url}")
            response = requests.get(config_url)
            response.raise_for_status()

            full_config = response.json()
            
            # Extract workflows config
            self.config = full_config.get("workflows", {})
            
            # Build processor map
            self.processor_map = self._build_processor_map()

            logging.info("DGN configuration loaded successfully from orchestrator.")

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
        """Get the appropriate job processor for a job."""
        workflow_type = job.get("workflow_type")

        ProcessorClass = self.processor_map.get(workflow_type)
        
        # Fallback to GenericComfyWorkflowProcessor for unknown workflow types
        if not ProcessorClass:
            if hasattr(job_processors_module, "GenericComfyWorkflowProcessor"):
                logging.info(f"Using GenericComfyWorkflowProcessor for workflow type: {workflow_type}")
                ProcessorClass = job_processors_module.GenericComfyWorkflowProcessor
            else:
                raise ValueError(
                    f"No job processor found for workflow type: {workflow_type}"
                )
        
        # Pass the Docker manager to the processor
        return ProcessorClass(self, job, shutdown_event, docker_comfyui_manager=self.comfyui_manager)

    def _process_job(self, job, shutdown_event: threading.Event):
        """Processes a single DGN job by delegating to a specific job processor."""
        try:
            self.current_job = job
            processor = self._get_job_processor(job, shutdown_event)
            processor.process()
        except TokenExpiredError:
            print(json.dumps({"status": "AUTH_EXPIRED"}), flush=True)
            logging.warning(
                f"Auth expired during processing of job {job.get('id')}. Signaled main process."
            )
        except Exception as e:
            logging.error(
                f"An error occurred while processing job {job.get('id')}: {e}",
                exc_info=True,
            )
            if job and job.get("id"):
                self.orchestrator_service.update_job_status(job.get("id"), "failed")
        finally:
            self.current_job = None


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
