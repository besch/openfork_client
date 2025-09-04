import os
import multiprocessing
import logging
import time
import requests
import subprocess
import threading
import http.server
import socketserver
import ffmpeg
import argparse
import sys
from config import ROOT_DIR, PRIMARY_ORCHESTRATOR_URL, FALLBACK_ORCHESTRATOR_URL, CACHE_DIR, DEV_MODE, DOCKER_COMPOSE_DIR
from services.orchestrator_service import OrchestratorService
from utils.comfyui_workflow_utils import materialize_start_image, inject_prompt_and_image_into_workflow, process_workflow_output, verify_workflow_nodes
from services.comfyui_service import ComfyUIClient
from utils.video_utils import generate_thumbnail, find_video_in_output, generate_placeholder_video


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

def manage_docker(action: str):
    """Starts or stops the ComfyUI Docker container using docker-compose."""
    compose_file_path = os.path.join(DOCKER_COMPOSE_DIR, 'docker-compose.yaml')
    if not os.path.exists(compose_file_path):
        logging.error(f"docker-compose.yaml not found in {DOCKER_COMPOSE_DIR}")
        return

    command = ["docker-compose", "-f", compose_file_path, action]
    if action == "up":
        command.append("-d")
    
    logging.info(f"Running '{' '.join(command)}'...")
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
                        # The get_and_assign_next_dgn_job RPC is now responsible for setting the
                        # provider to 'busy' and the job to 'processing' atomically.
                        # These client-side calls were redundant and often failed due to RLS policies.
                        workflow = job.get('workflow')
                        required_assets = job.get('assets', [])
                        positive_prompt = job.get('prompt') or ""
                        negative_prompt = job.get('negative_prompt') or ""

                        workflow_api_path = os.path.join(self.root_dir, 'workflows', 'wan2.2-image-to-video.api.json')
                        if not os.path.exists(workflow_api_path):
                            logging.error(f"Workflow API file not found at {workflow_api_path}")
                            self.orchestrator_service.update_job_status(job['id'], 'failed')
                            self.orchestrator_service.update_provider_status(provider_id, 'available')
                            continue

                        if required_assets:
                            for asset_id in required_assets:
                                self.orchestrator_service.download_asset(asset_id, self.cache_dir)

                        start_image_filename = materialize_start_image(job, self.input_dir)
                        wf_ready = inject_prompt_and_image_into_workflow(
                            workflow_api_path, positive_prompt, negative_prompt, start_image_filename
                        )

                        # Emit targeted debug to catch missing class_type before sending.
                        # wf_ready is expected to be {"prompt": {...}}
                        graph = wf_ready.get("prompt") if isinstance(wf_ready, dict) else None

                        if isinstance(graph, dict):
                            # Extract node IDs and class types directly from the API graph
                            node_info = []
                            for node_id, node_obj in graph.items():
                                if isinstance(node_obj, dict):
                                    class_type = node_obj.get("class_type")
                                    if class_type:
                                        node_info.append({"id": node_id, "class_type": class_type})
                                    else:
                                        logging.error(f"Node {node_id} missing 'class_type'.")
                                else:
                                    logging.error(f"Node {node_id} is not a dictionary.")

                            if node_info:
                                logging.info(f"First nodes after normalization: {node_info[:5]}")
                                id_map = {info["id"]: info["class_type"] for info in node_info}
                                logging.info(f"ID->class_type map (count={len(id_map)}): {list(id_map.items())[:10]} ...")
                            else:
                                logging.warning("No nodes found in the workflow graph for logging.")
                        else:
                            logging.error("Workflow graph is not a dictionary for logging.")

                        # Normalize any accidental boolean enums in KSampler nodes just before sending
                        try:
                            def normalize_sampler_enums(graph_obj):
                                if isinstance(graph_obj, dict):
                                    # API dict: iterate node dicts
                                    for v in graph_obj.values():
                                        if isinstance(v, dict):
                                            ct = (v.get("class_type") or v.get("type") or "").strip()
                                            if ct in ("KSampler", "KSamplerAdvanced"):
                                                ins = v.setdefault("inputs", {})
                                                if isinstance(ins, dict):
                                                    if isinstance(ins.get("add_noise"), bool):
                                                        ins["add_noise"] = "enable" if ins["add_noise"] else "disable"
                                                if isinstance(ins.get("return_with_leftover_noise"), bool):
                                                    ins["return_with_leftover_noise"] = "enable" if ins["return_with_leftover_noise"] else "disable"
                                return graph_obj

                            if isinstance(graph, dict) and all(isinstance(k, (str, int)) and isinstance(v, dict) for k, v in graph.items()):
                                graph = normalize_sampler_enums(graph)
                            elif isinstance(graph, dict) and isinstance(graph.get("nodes"), list):
                                # litegraph list: try to fix in-place before conversion (defensive)
                                for n in graph.get("nodes", []):
                                    if isinstance(n, dict):
                                        ct = (n.get("class_type") or n.get("type") or "").strip()
                                        if ct in ("KSampler", "KSamplerAdvanced"):
                                            ins = n.setdefault("inputs", {})
                                            if isinstance(ins, dict):
                                                if isinstance(ins.get("add_noise"), bool):
                                                    ins["add_noise"] = "enable" if ins["add_noise"] else "disable"
                                                if isinstance(ins.get("return_with_leftover_noise"), bool):
                                                    ins["return_with_leftover_noise"] = "enable" if ins["return_with_leftover_noise"] else "disable"
                            # Rebuild payload from possibly normalized graph
                            payload = {"prompt": graph} if not (isinstance(wf_ready, dict) and "prompt" in wf_ready) else {"prompt": graph}
                        except Exception:
                            # Fallback to previous payload building if anything goes wrong
                            payload = {"prompt": wf_ready["prompt"]} if isinstance(wf_ready, dict) and "prompt" in wf_ready else {"prompt": wf_ready}

                        # Compute terminal node ids to wait for (prefer VHS_VideoCombine if present)
                        def _compute_terminal_nodes(api_graph: dict) -> list[str]:
                            try:
                                # Collect all node ids (as strings)
                                node_ids = [str(k) for k in api_graph.keys()]
                                # Find nodes referenced by others via inputs (these have incoming edges)
                                referenced = set()
                                for _, node in api_graph.items():
                                    if not isinstance(node, dict):
                                        continue
                                    inputs = node.get("inputs", {})
                                    if not isinstance(inputs, dict):
                                        continue
                                    for v in inputs.values():
                                        # Single ref form: ["<node_id>", output_idx]
                                        if isinstance(v, list) and len(v) >= 1 and isinstance(v[0], str):
                                            referenced.add(v[0])
                                        # List of refs form: [[nid, idx], [nid, idx], ...]| 
                                        if isinstance(v, list) and v and all(isinstance(x, list) for x in v):
                                            for item in v:
                                                if isinstance(item, list) and item and isinstance(item[0], str):
                                                    referenced.add(item[0])
                                # Prefer VHS_VideoCombine as explicit terminal(s)
                                vhs_ids = [str(k) for k, n in api_graph.items() if isinstance(n, dict) and n.get("class_type") == "VHS_VideoCombine"]
                                if vhs_ids:
                                    return vhs_ids
                                # Otherwise nodes that are not referenced downstream (no outgoing edges when viewed reversed)
                                terminals = [nid for nid in node_ids if nid not in referenced]
                                return terminals or node_ids
                            except Exception:
                                return []

                        if DEV_MODE:
                            logging.info("DEV_MODE is enabled. Generating placeholder video.")
                            video_filename = generate_placeholder_video(self.output_dir, job['id'])
                            if video_filename:
                                local_video_path = os.path.join(self.output_dir, video_filename)
                                video_storage_path = self.orchestrator_service.upload_output(local_video_path, job['id'])
                                if not video_storage_path:
                                    logging.error(f"Failed to upload video output for job {job['id']}.")
                                    self.orchestrator_service.update_job_status(job['id'], 'failed')
                                else:
                                    thumbnail_filename = os.path.splitext(video_filename)[0] + ".jpg"
                                    thumbnail_local_path = os.path.join(self.output_dir, thumbnail_filename)
                                    thumbnail_storage_path = None
                                    if generate_thumbnail(local_video_path, thumbnail_local_path):
                                        # Trusting generate_thumbnail, proceeding with upload.
                                        thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(thumbnail_local_path, job['id'])
                                        if not thumbnail_storage_path:
                                            logging.warning(f"Thumbnail upload failed for job {job['id']}. The file may not have been found or there was a server error.")
                                    else:
                                        logging.error(f"Thumbnail generation failed for job {job['id']}.")
                                    
                                    self.orchestrator_service.update_job_status(
                                        job['id'], 
                                        'completed', 
                                        output_path=video_storage_path,
                                        thumbnail_path=thumbnail_storage_path
                                    )
                            else:
                                logging.error(f"Failed to generate placeholder video for job {job['id']}.")
                                self.orchestrator_service.update_job_status(job['id'], 'failed')
                        else:
                            graph_for_compute = payload.get("prompt") if isinstance(payload, dict) else None
                            terminal_ids = []
                            if isinstance(graph_for_compute, dict) and graph_for_compute and all(isinstance(v, dict) for v in graph_for_compute.values()):
                                terminal_ids = _compute_terminal_nodes(graph_for_compute)
    
                            prompt_id = self.comfyui_client.trigger_workflow(payload)
                            if prompt_id:
                                outputs = self.comfyui_client.get_workflow_output(prompt_id, terminal_node_ids=terminal_ids, timeout_sec=7200)
                                if outputs:
                                    video_filename = find_video_in_output(outputs)
                                    if video_filename:
                                        local_video_path = os.path.join(self.output_dir, video_filename)
                                        
                                        video_storage_path = self.orchestrator_service.upload_output(local_video_path, job['id'])
                                        
                                        if not video_storage_path:
                                            logging.error(f"Failed to upload video output for job {job['id']}.")
                                            self.orchestrator_service.update_job_status(job['id'], 'failed')
                                        else:
                                            thumbnail_filename = os.path.splitext(video_filename)[0] + ".jpg"
                                            thumbnail_local_path = os.path.join(self.output_dir, thumbnail_filename)
                                            thumbnail_storage_path = None
                                            
                                            if generate_thumbnail(local_video_path, thumbnail_local_path):
                                                # Trusting generate_thumbnail, proceeding with upload.
                                                thumbnail_storage_path = self.orchestrator_service.upload_thumbnail(thumbnail_local_path, job['id'])
                                                if not thumbnail_storage_path:
                                                    logging.warning(f"Thumbnail upload failed for job {job['id']}. The file may not have been found or there was a server error.")
                                            else:
                                                logging.error(f"Thumbnail generation failed for job {job['id']}.")
                                            
                                            self.orchestrator_service.update_job_status(
                                                job['id'], 
                                                'completed', 
                                                output_path=video_storage_path,
                                                thumbnail_path=thumbnail_storage_path
                                            )
                                    else:
                                        logging.error(f"Workflow for job {job['id']} completed, but no video file found in output.")
                                        self.orchestrator_service.update_job_status(job['id'], 'failed')
                                else:
                                    logging.error(f"Workflow for job {job['id']} failed to produce outputs.")
                                    self.orchestrator_service.update_job_status(job['id'], 'failed')
                            else:
                                logging.error(f"Failed to trigger workflow for job {job['id']}.")
                                self.orchestrator_service.update_job_status(job['id'], 'failed')

                    except Exception as e:
                        logging.error(f"An error occurred while processing job {job['id']}: {e}")
                        self.orchestrator_service.update_job_status(job['id'], 'failed')
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
    manage_docker("up")
    shutdown_thread = threading.Thread(target=start_shutdown_server, daemon=True)
    shutdown_thread.start()

    primary_url = PRIMARY_ORCHESTRATOR_URL
    fallback_url = FALLBACK_ORCHESTRATOR_URL
    determined_orchestrator_url = fallback_url

    try:
        logging.info(f"Attempting to connect to primary orchestrator URL: {primary_url}")
        response = requests.get(f"{primary_url}/api/dgn/provider-status/health", timeout=5)
        if response.status_code == 200:
            determined_orchestrator_url = primary_url
            logging.info(f"Successfully connected to primary orchestrator URL: {determined_orchestrator_url}")
        else:
            logging.warning(f"Primary orchestrator URL {primary_url} returned status {response.status_code}. Falling back.")
    except requests.exceptions.RequestException as e:
        logging.warning(f"Could not connect to primary orchestrator URL {primary_url}: {e}. Falling back.")

    client = DGNClient(
        orchestrator_url=determined_orchestrator_url,
        root_dir=ROOT_DIR,
        cache_dir=CACHE_DIR,
        access_token=args.access_token
    )

    provider_id = client.orchestrator_service.register_with_orchestrator()

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
            manage_docker("down")
            logging.info("DGN Client: Docker container stopped successfully.")
        except Exception as e:
            logging.error(f"DGN Client: Failed to stop Docker container: {e}", exc_info=True)

    logging.info("Main function completed.")

if __name__ == "__main__":
    import sys
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description='DGN Client')
    parser.add_argument('--access-token', type=str, required=True, help='Supabase Auth Access Token')
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
