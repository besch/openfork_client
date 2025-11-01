import os
import logging
import threading
from utils.shutdown_handler import SHUTDOWN_EVENT

from services.orchestrator_service import OrchestratorService
from services.comfyui_service import ComfyUIClient
from services.docker_manager import docker_manager
import sys
from services.job_processors import DynamicJobProcessor, MissingDependenciesError
from services.job_listener import JobListener

class DGNClient:
    def __init__(self, orchestrator_url: str, root_dir: str, data_dir: str, access_token: str, refresh_token: str, accept_policy: str = 'all', allowed_targets: list[str] = None):
        self.orchestrator_service = OrchestratorService(orchestrator_url, access_token, refresh_token)
        self.comfyui_client = ComfyUIClient(os.environ.get("COMFYUI_WS_URL", "ws://127.0.0.1:8188/ws?clientId={}"))
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.job_listener = JobListener(self, SHUTDOWN_EVENT)
        self.cache_dir = os.path.join(data_dir, ".cache")
        self.input_dir = os.path.join(data_dir, "input")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.current_job = None
        self.config = None
        self.accept_policy = accept_policy
        self.allowed_targets = allowed_targets or []
        self.allowed_ids = []

        if self.accept_policy == 'project' and self.allowed_targets:
            logging.info(f"Resolving targets: {self.allowed_targets}")
            self.allowed_ids = self.orchestrator_service.resolve_targets(self.allowed_targets)
            logging.info(f"Resolved targets to IDs: {self.allowed_ids}")

    def _get_job_processor(self, job, shutdown_event):
        return DynamicJobProcessor(self, job, shutdown_event)

    def _process_job(self, job, shutdown_event: threading.Event):
        """Processes a single DGN job by delegating to a specific job processor."""
        try:
            processor = self._get_job_processor(job, shutdown_event)
            processor.process()
        except MissingDependenciesError as e:
            logging.error("Job failed due to missing custom node dependencies.")
            print("[DEPENDENCY_ERROR] A required custom node is not installed.", file=sys.stderr)
            print("To fix this, please ask your AI assistant to run the following commands:", file=sys.stderr)
            
            for repo_url in e.missing_repos:
                print(f'[COMMAND] python3 /app/ComfyUI/custom_nodes/ComfyUI-Manager/cli/main.py --install-custom-node "{repo_url}"')
                
            print('[COMMAND] pkill -f "main.py --listen"')
            print("\nAfter the commands are run, please start the client again.", file=sys.stderr)
            # We need to fail the job in the orchestrator as well
            if job and job.get('id'):
                self.orchestrator_service.update_job_status(job.get('id'), 'failed', completion_metadata={'error_message': 'Missing custom node dependencies.'})
            SHUTDOWN_EVENT.set() # Signal all threads to shut down

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
