import os
import multiprocessing
import logging
import time
import requests
import subprocess
import threading
import http.server
import socketserver
from config import ROOT_DIR, PRIMARY_ORCHESTRATOR_URL, FALLBACK_ORCHESTRATOR_URL, CACHE_DIR
from services.orchestrator_service import OrchestratorService
from utils.comfyui_workflow_utils import materialize_start_image, inject_prompt_and_image_into_workflow, process_workflow_output, verify_workflow_nodes
from services.comfyui_service import ComfyUIClient


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CACHE_DIR is imported from config

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
            # Explicitly shut down the HTTP server
            if httpd_server:
                threading.Thread(target=httpd_server.shutdown, daemon=True).start() # Call shutdown in a new daemon thread to avoid deadlock
        else:
            self.send_response(404)
            self.end_headers()

def start_shutdown_server():
    """Starts a simple HTTP server in a new thread to listen for shutdown requests."""
    handler = ShutdownHandler
    global httpd_server # Assign to global variable
    with socketserver.TCPServer(("", SHUTDOWN_SERVER_PORT), handler, bind_and_activate=False) as httpd:
        httpd.allow_reuse_address = True
        httpd_server = httpd # Store reference
        try:
            httpd.server_bind()
            httpd.server_activate()
            logging.info(f"Shutdown server started on port {SHUTDOWN_SERVER_PORT}")
            httpd.serve_forever()
        except Exception as e:
            logging.error(f"Failed to start shutdown server: {e}")
        finally:
            httpd.server_close()
            logging.info("Shutdown server stopped.")
            httpd_server = None # Clear reference

# --- Docker Management ---
DOCKER_COMPOSE_DIR = os.path.join(ROOT_DIR, "comfyui-storage")

def manage_docker(action: str):
    """Starts or stops the ComfyUI Docker container using docker-compose."""
    compose_file_path = os.path.join(DOCKER_COMPOSE_DIR, 'docker-compose.yaml')
    if not os.path.exists(compose_file_path):
        logging.error(f"docker-compose.yaml not found in {DOCKER_COMPOSE_DIR}")
        return

    # Using -f to be explicit about the file, even with cwd set.
    command = ["docker-compose", "-f", compose_file_path, action]
    if action == "up":
        command.append("-d")  # Detached mode for starting
    
    logging.info(f"Running '{' '.join(command)}'...")
    try:
        result = subprocess.run(
            command,
            cwd=DOCKER_COMPOSE_DIR, # Running in the correct directory is still good practice
            capture_output=True,
            text=True,
            check=False,
            timeout=30 # Add a 30-second timeout
        )
        if result.returncode == 0:
            logging.info(f"Docker command '{action}' executed successfully.")
            if result.stdout.strip():
                logging.info(f"Docker stdout:\n{result.stdout.strip()}") # More explicit logging
            if result.stderr.strip():
                logging.warning(f"Docker stderr (even on success):\n{result.stderr.strip()}") # Log stderr even on success
        else:
            logging.error(f"Docker command '{action}' failed with exit code {result.returncode}.")
            if result.stdout.strip():
                logging.error(f"Docker stdout on failure:\n{result.stdout.strip()}")
            if result.stderr.strip():
                logging.error(f"Docker stderr on failure:\n{result.stderr.strip()}")

    except subprocess.TimeoutExpired:
        logging.error(f"Docker command '{action}' timed out after 30 seconds.")
        # If it timed out, try to kill the process group
        if result.returncode is None: # Process might still be running
            logging.error("Attempting to terminate hanging docker-compose process.")
            result.kill()
            result.wait() # Wait for it to actually terminate
            logging.error("Hanging docker-compose process terminated.")
    except FileNotFoundError:
        logging.error("'docker-compose' not found. Please ensure Docker Desktop is installed and running.")
    except Exception as e:
        logging.error(f"An exception occurred while running docker-compose: {e}")


# --- End Docker Management ---


class DGNClient:
    def __init__(self, orchestrator_url: str, root_dir: str, cache_dir: str):
        self.orchestrator_service = OrchestratorService(orchestrator_url)
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
            time.sleep(60) # Send heartbeat every 60 seconds

    def listen_for_jobs(self, provider_id: str):
        """Listen for jobs from the orchestrator."""
        while True:
            if SHUTDOWN_FLAG:
                logging.info("Shutdown flag received. Exiting job listening loop.")
                break
            try:
                logging.info("Checking for new jobs...")
                response = requests.get(f"{self.orchestrator_service.orchestrator_url}/api/dgn/jobs/{provider_id}")
                if response.status_code == 200:
                    job = response.json()
                    if job and job.get('id'):
                        logging.info(f"Received job: {job['id']}")
                        try:
                            self.orchestrator_service.update_provider_status(provider_id, 'busy')
                            self.orchestrator_service.update_job_status(job['id'], 'processing')

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

                            graph_for_compute = payload.get("prompt") if isinstance(payload, dict) else None
                            terminal_ids = []
                            if isinstance(graph_for_compute, dict) and graph_for_compute and all(isinstance(v, dict) for v in graph_for_compute.values()):
                                terminal_ids = _compute_terminal_nodes(graph_for_compute)

                            prompt_id = self.comfyui_client.trigger_workflow(payload)
                            if prompt_id:
                                outputs = self.comfyui_client.get_workflow_output(prompt_id, terminal_node_ids=terminal_ids, timeout_sec=7200)
                                if outputs:
                                    output_path = process_workflow_output(outputs, job['id'], self.output_dir, self.orchestrator_service.upload_output)
                                    if output_path:
                                        self.orchestrator_service.update_job_status(job['id'], 'completed', output_path=output_path)
                                    else:
                                        logging.error("Workflow completed, but output upload failed.")
                                        self.orchestrator_service.update_job_status(job['id'], 'failed')
                                else:
                                    logging.error("Workflow failed to produce outputs.")
                                    self.orchestrator_service.update_job_status(job['id'], 'failed')
                            else:
                                logging.error("Failed to trigger workflow.")
                                self.orchestrator_service.update_job_status(job['id'], 'failed')

                        except Exception as e:
                            logging.error(f"An error occurred while processing job {job['id']}: {e}")
                            self.orchestrator_service.update_job_status(job['id'], 'failed')
                        finally:
                            self.orchestrator_service.update_provider_status(provider_id, 'available')
                            logging.info("Provider status set to available. Waiting for next job...")

                    else:
                        logging.info("No new jobs.")
                else:
                    logging.error(f"Error checking for jobs: {response.text}")
            except requests.exceptions.RequestException as e:
                logging.error(f"Could not connect to the Orchestrator: {e}")

            time.sleep(10) # Poll every 10 seconds

def main():
    """Main function to run the DGN client."""
    # Start Docker container
    manage_docker("up")

    # Start the shutdown server in a separate thread
    shutdown_thread = threading.Thread(target=start_shutdown_server, daemon=True)
    shutdown_thread.start()

    # Determine ORCHESTRATOR_URL dynamically
    primary_url = PRIMARY_ORCHESTRATOR_URL
    fallback_url = FALLBACK_ORCHESTRATOR_URL
    determined_orchestrator_url = fallback_url # Default to fallback

    try:
        # Attempt to connect to the primary URL
        logging.info(f"Attempting to connect to primary orchestrator URL: {primary_url}")
        response = requests.get(f"{primary_url}/api/dgn/provider-status/health", timeout=5) # Use a health check endpoint
        if response.status_code == 200:
            determined_orchestrator_url = primary_url
            logging.info(f"Successfully connected to primary orchestrator URL: {determined_orchestrator_url}")
        else:
            logging.warning(f"Primary orchestrator URL {primary_url} returned status {response.status_code}. Falling back to {fallback_url}")
    except requests.exceptions.RequestException as e:
        logging.warning(f"Could not connect to primary orchestrator URL {primary_url}: {e}. Falling back to {fallback_url}")

    client = DGNClient(
        orchestrator_url=determined_orchestrator_url,
        root_dir=ROOT_DIR,
        cache_dir=CACHE_DIR
    )

    provider_id = client.orchestrator_service.register_with_orchestrator()

    if not provider_id:
        logging.error("Failed to register with orchestrator. Exiting.")
        return # Exit main function, which should lead to normal program exit

    # Start heartbeat thread
    client.start_heartbeat(provider_id)

    try:
        client.listen_for_jobs(provider_id)
    except Exception as e:
        logging.error(f"An error occurred in job listening loop: {e}", exc_info=True)
    finally:
        logging.info("Attempting to deregister from orchestrator.")
        try:
            client.orchestrator_service.deregister_from_orchestrator(provider_id)
            logging.info("Successfully deregistered from orchestrator.")
        except Exception as e:
            logging.error(f"Failed to deregister from orchestrator: {e}", exc_info=True)

        # ADD THIS LINE:
        logging.info("--- ABOUT TO CALL MANAGE_DOCKER('down') ---")
        # Explicitly stop Docker here
        logging.info("Explicitly stopping Docker container.")
        manage_docker("down")

    logging.info("Main function completed. Preparing for program exit.")

if __name__ == "__main__":
    import sys # Import sys
    multiprocessing.freeze_support()
    try:
        main()
        logging.info("Program exiting normally.")
        sys.exit(0) # Explicitly exit with code 0
    except KeyboardInterrupt:
        logging.info("Process interrupted by user.")
        sys.exit(0) # Exit with 0 on KeyboardInterrupt
    except Exception as e:
        logging.error(f"An unhandled exception occurred during program execution: {e}", exc_info=True)
        sys.exit(1) # Exit with 1 on unhandled exception
