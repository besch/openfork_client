import os
import logging
import argparse
import time
import requests
from config import ROOT_DIR, PRIMARY_ORCHESTRATOR_URL, FALLBACK_ORCHESTRATOR_URL, SUPABASE_URL, SUPABASE_ANON_KEY, CACHE_DIR
from services.supabase_service import SupabaseService
from services.orchestrator_service import OrchestratorService
from utils.comfyui_workflow_utils import materialize_start_image, inject_prompt_and_image_into_workflow, process_workflow_output, verify_workflow_nodes
from services.comfyui_service import ComfyUIClient


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define local input/output/models directories used for client I/O.
# These are used to materialize optional input image and read outputs if ComfyUI writes into a shared path.
# IMPORTANT: These paths must mirror the container mounts:
#   Host ./storage/ComfyUI/input  -> /opt/ComfyUI/input
#   Host ./storage/ComfyUI/output -> /opt/ComfyUI/output
#   Host ./storage/ComfyUI/models -> /opt/ComfyUI/models
#   Host ./storage/ComfyUI/user   -> /opt/ComfyUI/user
# To ensure the API points ComfyUI to files it can see, we write inputs under the mounted host input dir.
MOUNT_ROOT = os.path.join(ROOT_DIR, "comfyui-storage", "storage", "ComfyUI")
INPUT_DIR = os.path.join(MOUNT_ROOT, "input")
OUTPUT_DIR = os.path.join(MOUNT_ROOT, "output")
MODELS_DIR = os.path.join(MOUNT_ROOT, "models")
# CACHE_DIR is imported from config

def download_assets(assets: list[str]):
    """Download the assets required by the workflow from Supabase Storage."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

    for asset_id in assets:
        try:
            # Fetch asset metadata to get the storage_path
            response = supabase.from_('assets').select('storage_path').eq('id', asset_id).single()
            if response.error:
                logging.error(f"Error fetching asset {asset_id} metadata: {response.error.message}")
                continue
            
            storage_path = response.data['storage_path']
            file_name = os.path.basename(storage_path)
            asset_local_path = os.path.join(CACHE_DIR, file_name)

            if not os.path.exists(asset_local_path):
                logging.info(f"Downloading asset: {file_name} from {storage_path}")
                # Download the file from Supabase Storage
                download_response = supabase.storage.from_('dgn-assets').download(storage_path)
                if download_response.error:
                    logging.error(f"Error downloading asset {file_name}: {download_response.error.message}")
                    continue
                
                with open(asset_local_path, 'wb') as f:
                    f.write(download_response.data)
                logging.info(f"Asset {file_name} downloaded to {asset_local_path}")
            else:
                logging.info(f"Asset {file_name} already exists in cache.")
        except Exception as e:
            logging.error(f"An error occurred during asset download for {asset_id}: {e}")


def upload_output(file_path, job_id):
    """Upload the output file to Supabase Storage and return the storage path."""
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        with open(file_path, 'rb') as f:
            file_name = os.path.basename(file_path)
            storage_path = f"outputs/{job_id}/{file_name}"
            response = supabase.storage.from_('scene-videos').upload(storage_path, f.read(), {'content-type': 'video/mp4'})
            logging.info(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!! {response}")
            if response.path:
                logging.info(f"File {file_name} uploaded successfully to {response.path}.")
                return response.path # Return the full path from the response
            else:
                logging.error(f"Unexpected response from Supabase upload for file {file_name}: {response}")
                return None
    except Exception as e:
        logging.error(f"Could not upload file {file_path} to Supabase: {e}")
        return None







def listen_for_jobs(provider_id):
    """Listen for jobs from the orchestrator."""
    _ensure_dirs()
    while True:
        try:
            logging.info("Checking for new jobs...")
            response = requests.get(f"{ORCHESTRATOR_URL}/api/dgn/jobs/{provider_id}")
            if response.status_code == 200:
                job = response.json()
                if job and job.get('id'):
                    logging.info(f"Received job: {job['id']}")
                    try:
                        update_provider_status(provider_id, 'busy')
                        update_job_status(job['id'], 'processing')

                        # ... job processing logic ...
                        workflow = job.get('workflow')
                        required_assets = job.get('assets', [])
                        positive_prompt = job.get('prompt') or ""
                        negative_prompt = job.get('negative_prompt') or ""

                        workflow_api_path = os.path.join(ROOT_DIR, 'workflows', 'wan2.2-image-to-video.api.json')
                        if not os.path.exists(workflow_api_path):
                            logging.error(f"Workflow API file not found at {workflow_api_path}")
                            update_job_status(job['id'], 'failed')
                            continue

                        if required_assets:
                            download_assets(required_assets)

                        start_image_filename = _materialize_start_image(job)
                        wf_ready = _inject_prompt_and_image_into_workflow(
                            workflow_api_path, positive_prompt, negative_prompt, start_image_filename
                        )
                        
                        # ... (rest of the workflow preparation) ...

                        prompt_id = trigger_workflow(wf_ready)
                        if prompt_id:
                            outputs = get_workflow_output(prompt_id, timeout_sec=7200)
                            if outputs:
                                output_path = process_workflow_output(outputs, job['id'])
                                if output_path:
                                    update_job_status(job['id'], 'completed', output_path=output_path)
                                else:
                                    logging.error("Workflow completed, but output upload failed.")
                                    update_job_status(job['id'], 'failed')
                            else:
                                logging.error("Workflow failed to produce outputs.")
                                update_job_status(job['id'], 'failed')
                        else:
                            logging.error("Failed to trigger workflow.")
                            update_job_status(job['id'], 'failed')

                    except Exception as e:
                        logging.error(f"An error occurred while processing job {job['id']}: {e}")
                        update_job_status(job['id'], 'failed')
                    finally:
                        update_provider_status(provider_id, 'available')
                        logging.info("Provider status set to available. Waiting for next job...")

                else:
                    logging.info("No new jobs.")
            else:
                logging.error(f"Error checking for jobs: {response.text}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not connect to the Orchestrator: {e}")

        time.sleep(10) # Poll every 10 seconds


















class DGNClient:
    def __init__(self, orchestrator_url: str, supabase_url: str, supabase_anon_key: str, root_dir: str, cache_dir: str):
        self.orchestrator_service = OrchestratorService(orchestrator_url)
        self.supabase_service = SupabaseService(supabase_url, supabase_anon_key, cache_dir)
        self.comfyui_client = ComfyUIClient(os.environ.get("COMFYUI_WS_URL", "ws://127.0.0.1:8188/ws?clientId={}"))
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, "comfyui-storage", "storage", "ComfyUI", "input")
        self.output_dir = os.path.join(root_dir, "comfyui-storage", "storage", "ComfyUI", "output")
        self.models_dir = os.path.join(root_dir, "comfyui-storage", "storage", "ComfyUI", "models")
        self._ensure_dirs()

    def _ensure_dirs(self):
        for d in [self.input_dir, self.output_dir, self.models_dir, self.supabase_service.cache_dir]:
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

    def listen_for_jobs(self, provider_id: str):
        """Listen for jobs from the orchestrator."""
        while True:
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
                                self.supabase_service.download_assets(required_assets)

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
                                    output_path = process_workflow_output(outputs, job['id'], self.output_dir, self.supabase_service.upload_output)
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
    parser = argparse.ArgumentParser(description="CrowdMovie DGN Client")

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
        supabase_url=SUPABASE_URL,
        supabase_anon_key=SUPABASE_ANON_KEY,
        root_dir=ROOT_DIR,
        cache_dir=CACHE_DIR
    )

    provider_id = client.orchestrator_service.register_with_orchestrator()

    if not provider_id:
        return

    try:
        client.listen_for_jobs(provider_id)
    finally:
        client.orchestrator_service.deregister_from_orchestrator(provider_id)

if __name__ == "__main__":
    main()
