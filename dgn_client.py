import os
import logging
import threading
import requests

import json
from services.orchestrator_service import OrchestratorService, TokenExpiredError
from services.comfyui_service import ComfyUIClient
from services.local_comfyui_manager import LocalComfyUIManager
import services.job_processors as job_processors_module

try:
    from services.comfy_cli_manager import ComfyCliManager
except ImportError:
    ComfyCliManager = None

try:
    from services.workflow_importer import WorkflowImporter
except ImportError:
    WorkflowImporter = None


class DGNClient:
    def __init__(
        self,
        orchestrator_url: str,
        root_dir: str,
        data_dir: str,
        access_token: str,
        refresh_token: str,
        accept_policy: str = "all",
        allowed_targets: list[str] = None,
        comfyui_install_dir: str = None,
        comfyui_url: str = "http://127.0.0.1:8188",
    ):
        self.orchestrator_url = orchestrator_url
        self.orchestrator_service = OrchestratorService(
            orchestrator_url, access_token, refresh_token
        )
        # Ensure WS URL is derived or passed correctly. ComfyUIClient expects WS URL.
        # But DGNClient args usually get http url. 
        # Convert http to ws for ComfyUIClient if needed, or let ComfyUIClient handle it?
        # ComfyUIClient takes 'comfyui_ws_url'.
        
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
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.comfyui_install_dir = comfyui_install_dir
        self.cache_dir = os.path.join(data_dir, ".cache")
        # For local execution, input/output might be direct ComfyUI dirs or our temp dirs.
        # But LocalComfyUIManager provides get_input_directory().
        
        self.comfyui_manager = LocalComfyUIManager(
            comfyui_install_dir=comfyui_install_dir,
            comfyui_url=comfyui_url
        )
        
        # Start ComfyUI if needed
        self.comfyui_manager.start()

        # If we have a local install, prefer its input dir?
        # Actually, existing processors logic used self.input_dir then 'copied' to container.
        # Now we don't copy. The processors will just use the file paths.
        # But 'download content' needs a place.
        # If ComfyUI is local, we should download DIRECTLY to ComfyUI input folder if possible, 
        # or download to temp and ensure ComfyUI can read it (absolute paths work in ComfyUI for images usually).
        
        comfy_input_dir = self.comfyui_manager.get_input_directory()
        if comfy_input_dir and os.path.exists(comfy_input_dir):
             self.input_dir = comfy_input_dir
             logging.info(f"Using ComfyUI input directory: {self.input_dir}")
        else:
             self.input_dir = os.path.join(data_dir, "input")
             os.makedirs(self.input_dir, exist_ok=True)
             logging.info(f"Using internal input directory: {self.input_dir}")

        os.makedirs(self.cache_dir, exist_ok=True)
        self.active_service_type = None
        self.current_job = None
        self.config = {}
        # self.docker_image_map = {} # Removed
        self.accept_policy = accept_policy
        self.allowed_targets = allowed_targets or []
        self.allowed_ids = []
        self.processing_lock = threading.Lock()
        
        # Scan local workflows
        self.available_workflows = self.comfyui_manager.scan_workflows()
        logging.info(f"Found {len(self.available_workflows)} local workflows/templates.")
        
        # Initialize comfy-cli manager for programmatic node installation
        if ComfyCliManager:
            self.comfy_cli_manager = ComfyCliManager(comfyui_install_dir=comfyui_install_dir)
            if self.comfy_cli_manager.is_available():
                logging.info("comfy-cli is available for node management")
        else:
            self.comfy_cli_manager = None
        
        # Initialize workflow importer
        if WorkflowImporter and comfyui_install_dir:
            self.workflow_importer = WorkflowImporter(
                comfyui_install_dir=comfyui_install_dir,
                comfy_cli_manager=self.comfy_cli_manager
            )
        else:
            self.workflow_importer = None


        self.processor_map = {}

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
        """Build processor map with only two processors: TextGeneration and GenericComfyWorkflow."""
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
            
            # Extract workflows and services from the config
            self.config = full_config.get("workflows", {})
            services_config = full_config.get("services", {})

            # Create a map from service_name to prod_image for docker_manager
            # Now we need to get prod_image from services and map it via workflows
            # self.docker_image_map = {}
            # for workflow_type, workflow_config in self.config.items():
            #     service_name = workflow_config.get("service_name")
            #     if service_name and service_name in services_config:
            #         prod_image = services_config[service_name].get("prod_image")
            #         if prod_image:
            #             self.docker_image_map[service_name] = prod_image

            # docker_manager.set_docker_image_map(self.docker_image_map)


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
        workflow_type = job.get("workflow_type")

        ProcessorClass = self.processor_map.get(workflow_type)
        
        # Fallback to GenericComfyWorkflowProcessor for unknown workflow types
        if not ProcessorClass:
            if hasattr(job_processors_module, "GenericComfyWorkflowProcessor"):
                logging.info(f"Using GenericComfyWorkflowProcessor for unknown workflow type: {workflow_type}")
                ProcessorClass = job_processors_module.GenericComfyWorkflowProcessor
            else:
                raise ValueError(
                    f"No job processor found for workflow type: {workflow_type}"
                )
        
        if ProcessorClass == job_processors_module.GenericComfyWorkflowProcessor:
             return ProcessorClass(self, job, shutdown_event, local_comfyui_manager=self.comfyui_manager)
        
        return ProcessorClass(self, job, shutdown_event)

    def _process_job(self, job, shutdown_event: threading.Event):
        """Processes a single DGN job by delegating to a specific job processor.
        
        Note: This method is now deprecated and kept for backward compatibility.
        The job listener now handles processing directly to ensure proper locking.
        """
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
