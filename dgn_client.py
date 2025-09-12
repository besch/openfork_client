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
import shutil
from config import ROOT_DIR, CACHE_DIR, DEV_MODE, DOCKER_COMPOSE_DIR, ORCHESTRATOR_URL_PROD, ORCHESTRATOR_URL_DEV
from services.orchestrator_service import OrchestratorService
from utils.comfyui_workflow_utils import materialize_start_image, inject_prompt_and_image_into_workflow, process_workflow_output, verify_workflow_nodes, inject_video_and_prompt_into_foley_workflow, inject_prompt_into_flux_workflow, find_image_in_output
from services.comfyui_service import ComfyUIClient
from utils.video_utils import generate_thumbnail, find_video_in_output, find_audio_in_output, generate_placeholder_video, get_video_duration


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

# Global flag for graceful shutdown
SHUTDOWN_FLAG = False
SHUTDOWN_SERVER_PORT = 8000 # TODO: Make configurable
httpd_server = None # Global reference to the HTTP server

class ShutdownHandler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/shutdown':
            logging.info("Received shutdown request via HTTP.")
            global SHUTDOWN_FLAG
            SHUTDOWN_FLAG = True
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
            logging.info("Shutdown server: Now serving forever...")
            httpd.serve_forever()
        except Exception as e:
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
        compose_file = 'docker-compose.flux.yaml'
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
            check=False,
            timeout=30
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
        
    def start_heartbeat(self, provider_id: str):
        """Starts a background thread to send heartbeats."""
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop, args=(provider_id,), daemon=True)
        heartbeat_thread.start()
        logging.info("Heartbeat thread started.")

    def _heartbeat_loop(self, provider_id: str):
        """The loop that sends heartbeats periodically."""
        while True:
            try:
                self.orchestrator_service.send_heartbeat(provider_id)
            except Exception as e:
                logging.error(f"An error occurred in the heartbeat loop: {e}")
            time.sleep(60)

    def listen_for_jobs(self, provider_id: str):
        """Listen for jobs from the orchestrator."""
        while True:
            if SHUTDOWN_FLAG:
                logging.info("Shutdown flag received. Exiting job listening loop.")
                break
            try:
                logging.info("Checking for new jobs...")
                job = self.orchestrator_service.get_next_job(provider_id)

                if job and job.get('id'):
                    logging.info(f"Received job: {job['id']}")
                    try:
                        job_id = job['id']
                        workflow_type = job.get('workflow_type', 'image_to_video')
                        positive_prompt = job.get('prompt') or ""
                        negative_prompt = job.get('negative_prompt') or ""

                        if workflow_type == 'hunyuan_video_foley':
                            # --- FOLEY WORKFLOW ---
                            workflow_api_path = os.path.join(self.root_dir, 'workflows', 'hunyuan-video-foley.api.json')
                            input_video_url = job.get('input_video_url')

                            if not input_video_url:
                                logging.error(f"Foley job {job_id} missing 'input_video_url'.")
                                self.orchestrator_service.update_job_status(job_id, 'failed')
                                continue

                            # Download the video to be processed
                            video_path = self.orchestrator_service.download_asset_by_url(input_video_url, self.input_dir)
                            if not video_path:
                                logging.error(f"Failed to download input video for foley job {job_id}.")
                                self.orchestrator_service.update_job_status(job_id, 'failed')
                                continue
                            
                            video_filename = os.path.basename(video_path)

                            wf_ready = inject_video_and_prompt_into_foley_workflow(
                                workflow_api_path, video_filename, positive_prompt, negative_prompt
                            )
                            
                            payload = {"prompt": wf_ready}
                            prompt_id = self.comfyui_client.trigger_workflow(payload)

                            if prompt_id:
                                outputs = self.comfyui_client.get_workflow_output(prompt_id, timeout_sec=7200)
                                if outputs:
                                    audio_info = find_audio_in_output(outputs)
                                    if audio_info:
                                        audio_filename, subfolder = audio_info
                                        local_audio_path = os.path.join(self.output_dir, subfolder, audio_filename)
                                        
                                        audio_storage_path = self.orchestrator_service.upload_audio_output(local_audio_path, job_id)
                                        
                                        if audio_storage_path:
                                            self.orchestrator_service.update_job_status(
                                                job_id, 
                                                'completed',
                                                completion_metadata={'audio_storage_path': audio_storage_path}
                                            )
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

                        else:
                            # --- TEXT TO IMAGE WORKFLOW (FLUX) ---
                            if workflow_type == 'text_to_image':
                                workflow_api_path = os.path.join(self.root_dir, 'workflows', 'flux-text-to-image.api.json')
                                
                                if not os.path.exists(workflow_api_path):
                                    logging.error(f"Workflow API file not found at {workflow_api_path}")
                                    self.orchestrator_service.update_job_status(job_id, 'failed')
                                    continue

                                wf_ready = inject_prompt_into_flux_workflow(
                                    workflow_api_path, positive_prompt, negative_prompt
                                )
                                
                                payload = {"prompt": wf_ready}
                                prompt_id = self.comfyui_client.trigger_workflow(payload)

                                if prompt_id:
                                    outputs = self.comfyui_client.get_workflow_output(prompt_id, timeout_sec=7200)
                                    if outputs:
                                        image_info = find_image_in_output(outputs)
                                        if image_info:
                                            image_filename, subfolder = image_info
                                            local_image_path = os.path.join(self.output_dir, subfolder, image_filename)
                                            
                                            image_storage_path = self.orchestrator_service.upload_image_output(local_image_path, job_id)
                                            
                                            if image_storage_path:
                                                self.orchestrator_service.update_job_status(
                                                    job_id, 
                                                    'completed', 
                                                    output_path=image_storage_path
                                                )
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
                            # --- IMAGE TO VIDEO WORKFLOW (EXISTING LOGIC) ---
                            workflow_api_path = os.path.join(self.root_dir, 'workflows', 'wan2.2-image-to-video.api.json')
                            
                            if not os.path.exists(workflow_api_path):
                                logging.error(f"Workflow API file not found at {workflow_api_path}")
                                self.orchestrator_service.update_job_status(job_id, 'failed')
                                continue

                            start_image_filename = materialize_start_image(job, self.input_dir)
                            wf_ready = inject_prompt_and_image_into_workflow(
                                workflow_api_path, positive_prompt, negative_prompt, start_image_filename
                            )
                            
                            payload = {"prompt": wf_ready}
                            prompt_id = self.comfyui_client.trigger_workflow(payload)

                            if prompt_id:
                                outputs = self.comfyui_client.get_workflow_output(prompt_id, timeout_sec=7200)
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
                                            self.orchestrator_service.update_job_status(
                                                job_id, 
                                                'completed', 
                                                output_path=video_storage_path,
                                                thumbnail_path=thumbnail_storage_path,
                                                duration_seconds=duration
                                            )
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
                        self.orchestrator_service.update_job_status(job.get('id'), 'failed')
                    finally:
                        self.orchestrator_service.update_provider_status(provider_id, 'available')
                        logging.info("Provider status set to available. Waiting for next job...")
                else:
                    logging.info("No new jobs.")
            except Exception as e:
                logging.error(f"Could not connect to the Orchestrator: {e}")

            time.sleep(10)

def main(args):
    """Main function to run the DGN client."""
    manage_docker("up", service_type=args.service)
    shutdown_thread = threading.Thread(target=start_shutdown_server, daemon=True)
    shutdown_thread.start()

    if DEV_MODE:
        determined_orchestrator_url = ORCHESTRATOR_URL_DEV
    else:
        determined_orchestrator_url = ORCHESTRATOR_URL_PROD

    try:
        logging.info(f"Attempting to connect to orchestrator URL: {determined_orchestrator_url}")
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
        logging.error("Failed to register with orchestrator. Exiting.")
        return

    client.start_heartbeat(provider_id)

    print("DGN_CLIENT_RUNNING", flush=True)
    logging.info("DGN Client is running and listening for jobs.")

    try:
        client.listen_for_jobs(provider_id)
    except Exception as e:
        logging.error(f"An error occurred in job listening loop: {e}", exc_info=True)
    finally:
        logging.info("DGN Client: Initiating shutdown sequence.")
        logging.info("DGN Client: Attempting to deregister from orchestrator.")
        try:
            client.orchestrator_service.deregister_from_orchestrator(provider_id)
            logging.info("DGN Client: Successfully deregistered from orchestrator.")
        except Exception as e:
            logging.error(f"DGN Client: Failed to deregister from orchestrator: {e}", exc_info=True)
        
        logging.info("DGN Client: Stopping Docker container.")
        try:
            manage_docker("down", service_type=args.service)
            logging.info("DGN Client: Docker container stopped successfully.")
        except Exception as e:
            logging.error(f"DGN Client: Failed to stop Docker container: {e}", exc_info=True)

    logging.info("Main function completed.")

if __name__ == "__main__":
    import sys
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description='DGN Client')
    parser.add_argument('--access-token', type=str, required=True, help='Supabase Auth Access Token')
    parser.add_argument('--service', type=str, default='default', help='Service to run (default, foley, or text_to_image)')
    args = parser.parse_args()

    try:
        main(args)
        logging.info("Program exiting normally.")
        sys.exit(0)
    except KeyboardInterrupt:
        logging.info("Process interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"An unhandled exception occurred: {e}", exc_info=True)
        sys.exit(1)
