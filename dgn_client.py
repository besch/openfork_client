import os
import multiprocessing
import logging
import time
import requests
import subprocess
import threading
import http.server
import socketserver
import argparse
import sys
from config import ROOT_DIR, CACHE_DIR, DEV_MODE, DOCKER_COMPOSE_DIR, ORCHESTRATOR_URL_PROD, ORCHESTRATOR_URL_DEV
from services.orchestrator_service import OrchestratorService
from utils.comfyui_workflow_utils import materialize_start_image, inject_prompt_and_image_into_workflow, process_workflow_output, verify_workflow_nodes, inject_video_and_prompt_into_foley_workflow, inject_prompt_into_qwen_workflow, find_image_in_output, inject_prompt_into_text_to_video_workflow, inject_prompt_into_vibevoice_workflow, inject_script_and_clones_into_vibevoice_workflow
from services.comfyui_service import ComfyUIClient
from utils.video_utils import generate_thumbnail, find_video_in_output, find_audio_in_output, generate_placeholder_video, get_video_duration


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

# Global event for graceful shutdown
SHUTDOWN_EVENT = threading.Event()
SHUTDOWN_SERVER_PORT = 8000 # TODO: Make configurable
httpd_server = None # Global reference to the HTTP server

class ShutdownHandler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/shutdown':
            logging.info("Received shutdown request via HTTP.")
            SHUTDOWN_EVENT.set()
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"DGN Client shutting down.")
            if httpd_server:
                logging.info("ShutdownHandler: Attempting to shut down HTTP server.")
                threading.Thread(target=httpd_server.shutdown, daemon=True).start()
        else:
            self.send_response(404)
            self.end_headers()

def start_shutdown_server():
    """Starts a simple HTTP server in a new thread to listen for shutdown requests."""
    handler = ShutdownHandler
    global httpd_server
    with socketserver.TCPServer(('', SHUTDOWN_SERVER_PORT), handler, bind_and_activate=False) as httpd:
        httpd.allow_reuse_address = True
        httpd_server = httpd
        try:
            httpd.server_bind()
            httpd.server_activate()
            logging.info(f"Shutdown server started on port {SHUTDOWN_SERVER_PORT}")
            httpd.serve_forever()
        except Exception as e:
            if not SHUTDOWN_EVENT.is_set():
                logging.error(f"Failed to start shutdown server: {e}")
        finally:
            httpd.server_close()
            logging.info("Shutdown server stopped.")
            httpd_server = None

# --- Docker Management ---

def manage_docker(action: str, service_type: str = 'default'):
    """Starts or stops the ComfyUI Docker container using docker-compose."""
    if service_type == 'foley':
        compose_file = 'docker-compose.foley.yaml'
    elif service_type == 'text_to_image':
        compose_file = 'docker-compose.qwen.yaml'
    elif service_type == 'vibevoice':
        compose_file = 'docker-compose.vibevoice.yaml'
    else:
        compose_file = 'docker-compose.yaml'
    
    compose_file_path = os.path.join(DOCKER_COMPOSE_DIR, compose_file)
    if not os.path.exists(compose_file_path):
        logging.error(f"{compose_file} not found in {DOCKER_COMPOSE_DIR}")
        return

    command = ["docker-compose", "-f", compose_file_path, action]
    if action == "up":
        command.append("-d")
    
    logging.info(f"Running '{' '.join(command)}' for service '{service_type}'...")
    try:
        result = subprocess.run(
            command,
            cwd=DOCKER_COMPOSE_DIR,
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=False,
            timeout=1800
        )
        if result.returncode == 0:
            logging.info(f"Docker command '{action}' executed successfully.")
        else:
            logging.error(f"Docker command '{action}' failed with exit code {result.returncode}.")
            logging.error(f"Stderr: {result.stderr.strip()}")
    except Exception as e:
        logging.error(f"An exception occurred while running docker-compose: {e}")

# --- End Docker Management ---

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
        
    def start_heartbeat(self, provider_id: str):
        """Starts a background thread to send heartbeats."""
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop, args=(provider_id,), daemon=True)
        heartbeat_thread.start()
        logging.info("Heartbeat thread started.")

    def _heartbeat_loop(self, provider_id: str):
        """The loop that sends heartbeats periodically."""
        while not SHUTDOWN_EVENT.is_set():
            try:
                self.orchestrator_service.send_heartbeat(provider_id)
            except Exception as e:
                logging.error(f"An error occurred in the heartbeat loop: {e}")
            # Wait for 60 seconds or until shutdown event is set
            SHUTDOWN_EVENT.wait(60)

    def wait_for_comfyui(self, timeout=180):
        """Waits for the ComfyUI server to be available."""
        logging.info("Waiting for ComfyUI server to be ready...")
        start_time = time.time()
        url = "http://127.0.0.1:8188/queue"
        while time.time() - start_time < timeout:
            if SHUTDOWN_EVENT.is_set():
                logging.warning("Shutdown requested while waiting for ComfyUI.")
                return False
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logging.info("ComfyUI server is ready.")
                    return True
            except requests.exceptions.RequestException:
                SHUTDOWN_EVENT.wait(5)
        logging.error(f"ComfyUI server did not become ready in {timeout} seconds.")
        return False

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
        else:
            return 'default'

    def _process_job(self, job, shutdown_event: threading.Event):
        """Processes a single DGN job."""
        try:
            job_id = job['id']
            workflow_type = job.get('workflow_type', 'image_to_video')
            positive_prompt = job.get('prompt') or ""
            negative_prompt = job.get('negative_prompt') or ""

            def check_interruption(outputs):
                if outputs == "interrupted":
                    logging.warning(f"Processing of job {job_id} was interrupted by shutdown.")
                    return True
                return False

            if workflow_type == 'hunyuan_video_foley':
                # --- FOLEY WORKFLOW ---
                workflow_api_path = os.path.join(self.root_dir, 'workflows', 'hunyuan-video-foley.api.json')
                input_video_url = job.get('input_video_url')

                if not input_video_url:
                    logging.error(f"Foley job {job_id} missing 'input_video_url'.")
                    self.orchestrator_service.update_job_status(job_id, 'failed')
                    return

                video_path = self.orchestrator_service.download_asset_by_url(input_video_url, self.input_dir)
                if not video_path:
                    logging.error(f"Failed to download input video for foley job {job_id}.")
                    self.orchestrator_service.update_job_status(job_id, 'failed')
                    return
                
                video_filename = os.path.basename(video_path)
                wf_ready = inject_video_and_prompt_into_foley_workflow(workflow_api_path, video_filename, positive_prompt, negative_prompt)
                
                payload = {"prompt": wf_ready}
                prompt_id = self.comfyui_client.trigger_workflow(payload)

                if prompt_id:
                    outputs = self.comfyui_client.get_workflow_output(prompt_id, timeout_sec=7200, shutdown_event=shutdown_event)
                    if check_interruption(outputs): return

                    if outputs:
                        audio_info = find_audio_in_output(outputs)
                        if audio_info:
                            audio_filename, subfolder = audio_info
                            local_audio_path = os.path.join(self.output_dir, subfolder, audio_filename)
                            audio_storage_path = self.orchestrator_service.upload_audio_output(local_audio_path, job_id)
                            if audio_storage_path:
                                self.orchestrator_service.update_job_status(job_id, 'completed', output_path=audio_storage_path, completion_metadata=job.get('completion_metadata'))
                            else:
                                logging.error(f"Foley job {job_id} completed, but audio upload failed.")
                                self.orchestrator_service.update_job_status(job_id, 'failed')
                        else:
                            logging.error(f"Foley workflow for job {job_id} completed, but no audio file found.")
                            self.orchestrator_service.update_job_status(job_id, 'failed')
                    else:
                        logging.error(f"Foley workflow for job {job_id} failed to produce outputs.")
                        self.orchestrator_service.update_job_status(job_id, 'failed')
                else:
                    logging.error(f"Failed to trigger foley workflow for job {job_id}.")
                    self.orchestrator_service.update_job_status(job_id, 'failed')

            elif workflow_type == 'text_to_image':
                # --- TEXT TO IMAGE WORKFLOW (Qwen) ---
                workflow_api_path = os.path.join(self.root_dir, 'workflows', 'qwen.api.json')
                if not os.path.exists(workflow_api_path):
                    logging.error(f"Workflow API file not found at {workflow_api_path}")
                    self.orchestrator_service.update_job_status(job_id, 'failed')
                    return

                wf_ready = inject_prompt_into_qwen_workflow(workflow_api_path, positive_prompt, negative_prompt)
                payload = {"prompt": wf_ready}
                prompt_id = self.comfyui_client.trigger_workflow(payload)

                if prompt_id:
                    outputs = self.comfyui_client.get_workflow_output(prompt_id, timeout_sec=7200, shutdown_event=shutdown_event)
                    if check_interruption(outputs): return

                    if outputs:
                        image_info = find_image_in_output(outputs)
                        if image_info:
                            image_filename, subfolder = image_info
                            local_image_path = os.path.join(self.output_dir, subfolder, image_filename)
                            image_storage_path = self.orchestrator_service.upload_image_output(local_image_path, job_id)
                            if image_storage_path:
                                self.orchestrator_service.update_job_status(job_id, 'completed', output_path=image_storage_path, thumbnail_path=image_storage_path, prompt=positive_prompt)
                            else:
                                logging.error(f"Image upload failed for job {job_id}.")
                                self.orchestrator_service.update_job_status(job_id, 'failed')
                        else:
                            logging.error(f"Workflow for job {job_id} completed, but no image file found.")
                            self.orchestrator_service.update_job_status(job_id, 'failed')
                    else:
                        logging.error(f"Workflow for job {job_id} failed to produce outputs.")
                        self.orchestrator_service.update_job_status(job_id, 'failed')
                else:
                    logging.error(f"Failed to trigger workflow for job {job_id}.")
                    self.orchestrator_service.update_job_status(job_id, 'failed')

            elif workflow_type == 'vibevoice':
                # --- TEXT TO SPEECH WORKFLOW (VibeVoice) ---
                workflow_api_path = os.path.join(self.root_dir, 'workflows', 'vibevoice.api.json')
                if not os.path.exists(workflow_api_path):
                    logging.error(f"Workflow API file not found at {workflow_api_path}")
                    self.orchestrator_service.update_job_status(job_id, 'failed')
                    return

                wf_ready = inject_prompt_into_vibevoice_workflow(workflow_api_path, positive_prompt)
                payload = {"prompt": wf_ready}
                prompt_id = self.comfyui_client.trigger_workflow(payload)

                if prompt_id:
                    outputs = self.comfyui_client.get_workflow_output(prompt_id, timeout_sec=7200, shutdown_event=shutdown_event)
                    if check_interruption(outputs): return

                    if outputs:
                        audio_info = find_audio_in_output(outputs)

                        # The raw output log has shown the structure is {'audio': [...]}, not {'ui': {'audio': [...]}}
                        # Let's parse that structure directly if the original function fails.
                        if not audio_info:
                            logging.warning("find_audio_in_output failed. Looking for {'audio': [...]} pattern based on logs.")
                            for node_id, node_output in outputs.items():
                                if 'audio' in node_output and isinstance(node_output.get('audio'), list):
                                    for item in node_output['audio']:
                                        if isinstance(item, dict):
                                            filename = item.get('filename')
                                            if filename:
                                                audio_info = (filename, item.get('subfolder', ''))
                                                break
                                if audio_info:
                                    break
                        
                        if audio_info:
                            audio_filename, subfolder = audio_info
                            local_audio_path = os.path.join(self.output_dir, subfolder, audio_filename)
                            audio_storage_path = self.orchestrator_service.upload_audio_output(local_audio_path, job_id)
                            if audio_storage_path:
                                self.orchestrator_service.update_job_status(job_id, 'completed', output_path=audio_storage_path, completion_metadata=job.get('completion_metadata'))
                            else:
                                logging.error(f"VibeVoice job {job_id} completed, but audio upload failed.")
                                self.orchestrator_service.update_job_status(job_id, 'failed')
                        else:
                            logging.error(f"VibeVoice workflow for job {job_id} completed, but no audio file found.")
                            self.orchestrator_service.update_job_status(job_id, 'failed')
                    else:
                        logging.error(f"VibeVoice workflow for job {job_id} failed to produce outputs.")
                        self.orchestrator_service.update_job_status(job_id, 'failed')
                else:
                    logging.error(f"Failed to trigger VibeVoice workflow for job {job_id}.")
                    self.orchestrator_service.update_job_status(job_id, 'failed')

            elif workflow_type == 'vibevoice_multi_clone':
                # --- TEXT TO SPEECH MULTI-SPEAKER CLONE WORKFLOW (VibeVoice) ---
                workflow_api_path = os.path.join(self.root_dir, 'workflows', 'vibevoice-multi-speaker-clone.api.json')
                if not os.path.exists(workflow_api_path):
                    logging.error(f"Workflow API file not found at {workflow_api_path}")
                    self.orchestrator_service.update_job_status(job_id, 'failed')
                    return

                voice_clone_urls = job.get('voice_clone_urls', [])
                if not voice_clone_urls:
                    logging.error(f"VibeVoice multi-clone job {job_id} missing 'voice_clone_urls'.")
                    self.orchestrator_service.update_job_status(job_id, 'failed')
                    return

                clone_paths = []
                for url in voice_clone_urls:
                    clone_path = self.orchestrator_service.download_asset_by_url(url, self.input_dir)
                    if not clone_path:
                        logging.error(f"Failed to download voice clone from {url} for job {job_id}.")
                        self.orchestrator_service.update_job_status(job_id, 'failed')
                        return
                    clone_paths.append(os.path.basename(clone_path))

                wf_ready = inject_script_and_clones_into_vibevoice_workflow(workflow_api_path, positive_prompt, clone_paths)
                payload = {"prompt": wf_ready}
                prompt_id = self.comfyui_client.trigger_workflow(payload)

                if prompt_id:
                    outputs = self.comfyui_client.get_workflow_output(prompt_id, timeout_sec=7200, shutdown_event=shutdown_event)
                    if check_interruption(outputs): return

                    if outputs:
                        audio_info = find_audio_in_output(outputs)
                        if audio_info:
                            audio_filename, subfolder = audio_info
                            local_audio_path = os.path.join(self.output_dir, subfolder, audio_filename)
                            audio_storage_path = self.orchestrator_service.upload_audio_output(local_audio_path, job_id)
                            if audio_storage_path:
                                self.orchestrator_service.update_job_status(job_id, 'completed', output_path=audio_storage_path, completion_metadata=job.get('completion_metadata'))
                            else:
                                logging.error(f"VibeVoice multi-clone job {job_id} completed, but audio upload failed.")
                                self.orchestrator_service.update_job_status(job_id, 'failed')
                        else:
                            logging.error(f"VibeVoice multi-clone workflow for job {job_id} completed, but no audio file found.")
                            self.orchestrator_service.update_job_status(job_id, 'failed')
                    else:
                        logging.error(f"VibeVoice multi-clone workflow for job {job_id} failed to produce outputs.")
                        self.orchestrator_service.update_job_status(job_id, 'failed')
                else:
                    logging.error(f"Failed to trigger VibeVoice multi-clone workflow for job {job_id}.")
                    self.orchestrator_service.update_job_status(job_id, 'failed')

            elif workflow_type == 'wan-2.2-text-to-video':
                # --- TEXT TO VIDEO WORKFLOW (WAN 2.2) ---
                workflow_api_path = os.path.join(self.root_dir, 'workflows', 'wan2.2-text-to-video.api.json')
                if not os.path.exists(workflow_api_path):
                    logging.error(f"Workflow API file not found at {workflow_api_path}")
                    self.orchestrator_service.update_job_status(job_id, 'failed')
                    return

                wf_ready = inject_prompt_into_text_to_video_workflow(workflow_api_path, positive_prompt, negative_prompt)
                
                payload = {"prompt": wf_ready}
                prompt_id = self.comfyui_client.trigger_workflow(payload)

                if prompt_id:
                    outputs = self.comfyui_client.get_workflow_output(prompt_id, timeout_sec=7200, shutdown_event=shutdown_event)
                    if check_interruption(outputs): return

                    if outputs:
                        video_info = find_video_in_output(outputs)
                        if video_info:
                            video_filename, subfolder = video_info
                            local_video_path = os.path.join(self.output_dir, subfolder, video_filename)
                            video_storage_path = self.orchestrator_service.upload_output(local_video_path, job_id)
                            
                            if video_storage_path:
                                thumbnail_filename = os.path.splitext(video_filename)[0] + ".jpg"
                                thumbnail_local_path = os.path.join(self.output_dir, thumbnail_filename)
                                thumbnail_storage_path = None
                                
                                if generate_thumbnail(local_video_path, thumbnail_local_path):
                                    thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(thumbnail_local_path, job_id)
                                
                                duration = get_video_duration(local_video_path)
                                self.orchestrator_service.update_job_status(job_id, 'completed', output_path=video_storage_path, thumbnail_path=thumbnail_storage_path, duration_seconds=duration)
                            else:
                                logging.error(f"Video upload failed for job {job_id}.")
                                self.orchestrator_service.update_job_status(job_id, 'failed')
                        else:
                            logging.error(f"Workflow for job {job_id} completed, but no video file found.")
                            self.orchestrator_service.update_job_status(job_id, 'failed')
                    else:
                        logging.error(f"Workflow for job {job_id} failed to produce outputs.")
                        self.orchestrator_service.update_job_status(job_id, 'failed')
                else:
                    logging.error(f"Failed to trigger workflow for job {job_id}.")
                    self.orchestrator_service.update_job_status(job_id, 'failed')
            
            else: # Default to image_to_video
                # --- IMAGE TO VIDEO WORKFLOW (EXISTING LOGIC) ---
                workflow_api_path = os.path.join(self.root_dir, 'workflows', 'wan2.2-image-to-video.api.json')
                if not os.path.exists(workflow_api_path):
                    logging.error(f"Workflow API file not found at {workflow_api_path}")
                    self.orchestrator_service.update_job_status(job_id, 'failed')
                    return

                start_image_filename = materialize_start_image(job, self.input_dir)
                wf_ready = inject_prompt_and_image_into_workflow(workflow_api_path, positive_prompt, negative_prompt, start_image_filename)
                
                payload = {"prompt": wf_ready}
                prompt_id = self.comfyui_client.trigger_workflow(payload)

                if prompt_id:
                    outputs = self.comfyui_client.get_workflow_output(prompt_id, timeout_sec=7200, shutdown_event=shutdown_event)
                    if check_interruption(outputs): return

                    if outputs:
                        video_info = find_video_in_output(outputs)
                        if video_info:
                            video_filename, subfolder = video_info
                            local_video_path = os.path.join(self.output_dir, subfolder, video_filename)
                            video_storage_path = self.orchestrator_service.upload_output(local_video_path, job_id)
                            
                            if video_storage_path:
                                thumbnail_filename = os.path.splitext(video_filename)[0] + ".jpg"
                                thumbnail_local_path = os.path.join(self.output_dir, thumbnail_filename)
                                thumbnail_storage_path = None
                                
                                if generate_thumbnail(local_video_path, thumbnail_local_path):
                                    thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(thumbnail_local_path, job_id)
                                
                                duration = get_video_duration(local_video_path)
                                self.orchestrator_service.update_job_status(job_id, 'completed', output_path=video_storage_path, thumbnail_path=thumbnail_storage_path, duration_seconds=duration)
                            else:
                                logging.error(f"Video upload failed for job {job_id}.")
                                self.orchestrator_service.update_job_status(job_id, 'failed')
                        else:
                            logging.error(f"Workflow for job {job_id} completed, but no video file found.")
                            self.orchestrator_service.update_job_status(job_id, 'failed')
                    else:
                        logging.error(f"Workflow for job {job_id} failed to produce outputs.")
                        self.orchestrator_service.update_job_status(job_id, 'failed')
                else:
                    logging.error(f"Failed to trigger workflow for job {job_id}.")
                    self.orchestrator_service.update_job_status(job_id, 'failed')

        except Exception as e:
            logging.error(f"An error occurred while processing job {job.get('id')}: {e}", exc_info=True)
            if job and job.get('id'):
                self.orchestrator_service.update_job_status(job.get('id'), 'failed')

    def listen_for_jobs(self, provider_id: str):
        """Listen for jobs from the orchestrator (for dedicated providers)."""
        while not SHUTDOWN_EVENT.is_set():
            job = None
            try:
                logging.info(f"Checking for new jobs for provider {provider_id}...")
                job = self.orchestrator_service.get_next_job(provider_id)

                if job and job.get('id'):
                    self.current_job = job
                    logging.info(f"Received job: {job['id']}")
                    self._process_job(job, SHUTDOWN_EVENT)
                    
                    if not SHUTDOWN_EVENT.is_set():
                        self.orchestrator_service.update_provider_status(provider_id, 'available')
                        logging.info("Provider status set to available. Waiting for next job...")
                        self.current_job = None
                else:
                    logging.info("No new jobs.")
            except Exception as e:
                logging.error(f"Could not connect to the Orchestrator: {e}")

            if not (job and job.get('id')):
                SHUTDOWN_EVENT.wait(10)
        logging.info("Shutdown event received. Exiting job listening loop.")

    def listen_for_jobs_auto(self, provider_id: str):
        """Listen for jobs and dynamically start/stop containers."""
        while not SHUTDOWN_EVENT.is_set():
            job = None
            try:
                logging.info("Auto mode: Checking for new jobs...")
                job = self.orchestrator_service.get_next_job(provider_id)

                if job and job.get('id'):
                    self.current_job = job
                    job_id = job['id']
                    logging.info(f"Received job: {job_id}")

                    workflow_type = job.get('workflow_type', 'image_to_video')
                    service_type = self.get_service_type_for_workflow(workflow_type)
                    self.active_service_type = service_type
                    
                    logging.info(f"Job requires service '{service_type}'. Starting container...")
                    manage_docker("up", service_type=service_type)
                    
                    if self.wait_for_comfyui():
                        self._process_job(job, SHUTDOWN_EVENT)
                    else:
                        if not SHUTDOWN_EVENT.is_set():
                            logging.error(f"ComfyUI for service '{service_type}' failed to start. Failing job.")
                            self.orchestrator_service.update_job_status(job_id, 'failed')

                    # If a shutdown was not requested, proceed with normal cleanup
                    if not SHUTDOWN_EVENT.is_set():
                        logging.info(f"Job processing finished. Stopping container for service '{service_type}'...")
                        manage_docker("down", service_type=service_type)
                        self.active_service_type = None
                        self.orchestrator_service.update_provider_status(provider_id, 'available')
                        logging.info("Provider status set to available. Waiting for next job...")
                        self.current_job = None # Clear current job
                else:
                    logging.info("No new jobs.")
            except Exception as e:
                logging.error(f"An error occurred in auto job listening loop: {e}", exc_info=True)

            if not (job and job.get('id')):
                SHUTDOWN_EVENT.wait(10) # Wait for 10s or until shutdown
        logging.info("Shutdown event received. Exiting auto job listening loop.")


def main(args):
    """Main function to run the DGN client."""
    # Start docker container for non-auto services
    if args.service != 'auto':
        manage_docker("up", service_type=args.service)
    
    shutdown_thread = threading.Thread(target=start_shutdown_server, daemon=True)
    shutdown_thread.start()

    client = None
    provider_id = None
    try:
        determined_orchestrator_url = ORCHESTRATOR_URL_DEV if DEV_MODE else ORCHESTRATOR_URL_PROD
        
        logging.info(f"Attempting to connect to orchestrator URL: {determined_orchestrator_url}")
        try:
            response = requests.get(f"{determined_orchestrator_url}/api/dgn/provider-status/health", timeout=5)
            if response.status_code == 200:
                logging.info(f"Successfully connected to orchestrator URL: {determined_orchestrator_url}")
            else:
                logging.warning(f"Orchestrator URL {determined_orchestrator_url} returned status {response.status_code}.")
        except requests.exceptions.RequestException as e:
            logging.warning(f"Could not connect to orchestrator URL {determined_orchestrator_url}: {e}.")

        client = DGNClient(
            orchestrator_url=determined_orchestrator_url,
            root_dir=ROOT_DIR,
            cache_dir=CACHE_DIR,
            access_token=args.access_token
        )

        provider_id = client.orchestrator_service.register_with_orchestrator(service_type=args.service)

        if not provider_id:
            raise RuntimeError("Failed to register with orchestrator. Aborting startup.")

        client.start_heartbeat(provider_id)

        print("DGN_CLIENT_RUNNING", flush=True)
        logging.info(f"DGN Client is running in '{args.service}' mode and listening for jobs.")

        if args.service == 'auto':
            client.listen_for_jobs_auto(provider_id)
        else:
            client.listen_for_jobs(provider_id)

    except Exception as e:
        logging.error(f"A critical error occurred during client operation: {e}", exc_info=True)
    finally:
        logging.info("DGN Client: Initiating shutdown sequence.")
        
        if client and client.current_job:
            job_id = client.current_job.get('id')
            if job_id:
                logging.info(f"A job ({job_id}) was in progress. Attempting to reset its status to 'pending'.")
                try:
                    client.orchestrator_service.reset_interrupted_job(job_id)
                except Exception as e:
                    logging.error(f"Failed to reset job {job_id}: {e}", exc_info=True)

        if provider_id and client:
            logging.info("DGN Client: Attempting to deregister from orchestrator.")
            try:
                client.orchestrator_service.deregister_from_orchestrator(provider_id)
                logging.info("DGN Client: Successfully deregistered from orchestrator.")
            except Exception as e:
                logging.error(f"DGN Client: Failed to deregister from orchestrator: {e}", exc_info=True)
        
        logging.info("DGN Client: Stopping Docker container(s).")
        try:
            if args.service != 'auto':
                # If we started a service statically, we always try to shut it down.
                manage_docker("down", service_type=args.service)
            elif client and client.active_service_type:
                # In auto mode, only shut down a container if one was made active during an interrupted job.
                manage_docker("down", service_type=client.active_service_type)
            else:
                logging.info("DGN Client: No active container to stop.")
        except Exception as e:
            logging.error(f"DGN Client: Failed to stop Docker container: {e}", exc_info=True)

    logging.info("Main function completed.")

if __name__ == "__main__":
    import sys
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description='DGN Client')
    parser.add_argument('--access-token', type=str, required=True, help='Supabase Auth Access Token')
    parser.add_argument('--service', type=str, default='auto', help='Service to run (default, foley, text_to_image, or auto)')
    args = parser.parse_args()

    try:
        main(args)
        logging.info("Program exiting normally.")
        sys.exit(0)
    except KeyboardInterrupt:
        SHUTDOWN_EVENT.set()
        logging.info("Process interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"An unhandled exception occurred: {e}", exc_info=True)
        sys.exit(1)