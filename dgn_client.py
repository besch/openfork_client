import os
import logging
import threading

from services.orchestrator_service import OrchestratorService
from services.comfyui_service import ComfyUIClient
from services.job_processors import (
    FoleyJobProcessor,
    TextToImageJobProcessor,
    VibeVoiceJobProcessor,
    VibeVoiceMultiCloneJobProcessor,
    TextToVideoJobProcessor,
    ImageToVideoJobProcessor,
    DiffRhythmJobProcessor,
    TextToVideoLightningJobProcessor,
    ImageToVideoLightningJobProcessor
)
from services.token_update_service import start_token_update_server


class DGNClient:
    def __init__(self, orchestrator_url: str, root_dir: str, cache_dir: str, access_token: str, refresh_token: str):
        self.orchestrator_service = OrchestratorService(orchestrator_url, access_token, refresh_token)
        self.comfyui_client = ComfyUIClient(os.environ.get("COMFYUI_WS_URL", "ws://127.0.0.1:8188/ws?clientId={}"))
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.input_dir = os.path.join(root_dir, "comfyui-storage", "storage", "ComfyUI", "input")
        self.output_dir = os.path.join(root_dir, "comfyui-storage", "storage", "ComfyUI", "output")
        self.models_dir = os.path.join(root_dir, "comfyui-storage", "storage", "ComfyUI", "models")
        self.active_service_type = None
        self.current_job = None
        self.httpd = None
        self.token_server_port = None
        self.config = None

        # Start the token update server and notify Electron of the port
        self.httpd, self.token_server_port = start_token_update_server(self.orchestrator_service)
        if self.token_server_port:
            print(f"DGN_CLIENT_TOKEN_SERVER_PORT: {self.token_server_port}", flush=True)

    def load_config(self):
        """Loads the configuration from the orchestrator."""
        self.config = self.orchestrator_service.get_dgn_config()
        if not self.config:
            raise RuntimeError("Failed to load DGN configuration from orchestrator.")

        wt = self.config.get('workflow_types', {})
        self.processor_map = {
            wt.get('HUNYUAN_VIDEO_FOLEY'): FoleyJobProcessor,
            wt.get('QWEN_TEXT_TO_IMAGE'): TextToImageJobProcessor,
            wt.get('VIBEVOICE_TTS'): VibeVoiceJobProcessor,
            wt.get('VIBEVOICE_TTS_MULTI_CLONE'): VibeVoiceMultiCloneJobProcessor,
            wt.get('DIFFRHYTHM_MUSIC_GENERATION'): DiffRhythmJobProcessor,
            wt.get('WAN22_TEXT_TO_VIDEO'): TextToVideoJobProcessor,
            wt.get('WAN22_IMAGE_TO_VIDEO'): ImageToVideoJobProcessor,
            wt.get('WAN22_LIGHTNING_TEXT_TO_VIDEO'): TextToVideoLightningJobProcessor,
            wt.get('WAN22_LIGHTNING_IMAGE_TO_VIDEO'): ImageToVideoLightningJobProcessor,
        }
        # Filter out None keys in case a workflow type is missing from config
        self.processor_map = {k: v for k, v in self.processor_map.items() if k}

        # TODO: Pass self.config['docker_image_map'] to docker_manager

    def get_service_type_for_workflow(self, workflow_type: str) -> str:
        """Maps a workflow type to a service type using the dynamic config."""
        if not self.config or 'workflow_to_service_map' not in self.config:
            raise ValueError("DGN configuration is not loaded or is invalid.")
        
        service_type = self.config['workflow_to_service_map'].get(workflow_type)
        if not service_type:
            raise ValueError(f"Unknown workflow type, cannot determine service: {workflow_type}")
        return service_type

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
