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
    DiffRhythmJobProcessor
)

class DGNClient:
    def __init__(self, orchestrator_url: str, root_dir: str, cache_dir: str, access_token: str):
        self.orchestrator_service = OrchestratorService(orchestrator_url, access_token)
        self.comfyui_client = ComfyUIClient(os.environ.get("COMFYUI_WS_URL", "ws://127.0.0.1:8188/ws?clientId={}"))
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.input_dir = os.path.join(root_dir, "comfyui-storage", "storage", "ComfyUI", "input")
        self.output_dir = os.path.join(root_dir, "comfyui-storage", "storage", "ComfyUI", "output")
        self.models_dir = os.path.join(root_dir, "comfyui-storage", "storage", "ComfyUI", "models")
        self.active_service_type = None
        self.current_job = None

    def get_service_type_for_workflow(self, workflow_type: str) -> str:
        """Maps a workflow type to a docker-compose service type."""
        if workflow_type == 'hunyuan_video_foley':
            return 'foley'
        elif workflow_type == 'text_to_image':
            return 'text_to_image'
        elif workflow_type == 'vibevoice':
            return 'vibevoice'
        elif workflow_type == 'vibevoice_multi_clone':
            return 'vibevoice'
        elif workflow_type == 'diffrhythm_music_generation':
            return 'diffrhythm'
        else:
            return 'default'

    def _get_job_processor(self, job, shutdown_event):
        workflow_type = job.get('workflow_type', 'image_to_video')
        
        processor_map = {
            'hunyuan_video_foley': FoleyJobProcessor,
            'text_to_image': TextToImageJobProcessor,
            'vibevoice': VibeVoiceJobProcessor,
            'vibevoice_multi_clone': VibeVoiceMultiCloneJobProcessor,
            'diffrhythm_music_generation': DiffRhythmJobProcessor,
            'wan-2.2-text-to-video': TextToVideoJobProcessor,
            'image_to_video': ImageToVideoJobProcessor,
        }

        ProcessorClass = processor_map.get(workflow_type, ImageToVideoJobProcessor)
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
