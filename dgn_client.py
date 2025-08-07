import time
import requests
import os
import logging
import argparse
import base64
import copy
import json

from comfyui_manager import trigger_workflow, get_workflow_output
from hardware_profiler import get_hardware_profile
from config import ROOT_DIR, ORCHESTRATOR_URL, SUPABASE_URL, SUPABASE_ANON_KEY, CACHE_DIR
from supabase import create_client, Client

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
    """Upload the output file to Supabase Storage."""
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        with open(file_path, 'rb') as f:
            file_name = os.path.basename(file_path)
            # Use job_id in the path to organize outputs
            storage_path = f"outputs/{job_id}/{file_name}"
            response = supabase.storage.from_('dgn-assets').upload(storage_path, f.read(), {'content-type': 'video/mp4'})
            if response.status_code == 200:
                logging.info(f"File {file_name} uploaded successfully to {storage_path}.")
            else:
                logging.error(f"Error uploading file {file_name}: {response.text}")
    except Exception as e:
        logging.error(f"Could not upload file {file_path} to Supabase: {e}")

def process_workflow_output(outputs, job_id):
    """Process the workflow output and upload the generated files."""
    for node_id, node_output in outputs.items():
        if 'filenames' in node_output:
            for filename in node_output['filenames']:
                # Prefer OUTPUT_DIR; ComfyUI default mount might be /opt/ComfyUI/output mirrored to our OUTPUT_DIR.
                file_path = os.path.join(OUTPUT_DIR, filename)
                if os.path.exists(file_path):
                    upload_output(file_path, job_id)
                else:
                    logging.warning(f"Output file not found: {file_path}")

def register_with_orchestrator():
    """Register the client with the orchestrator."""
    hardware_profile = get_hardware_profile()
    logging.info(f"Hardware Profile: {hardware_profile}")

    try:
        response = requests.post(f"{ORCHESTRATOR_URL}/api/dgn/register", json=hardware_profile)
        if response.status_code == 200:
            logging.info("Successfully registered with the Orchestrator.")
            return response.json().get('provider_id')
        else:
            logging.error(f"Error registering with the Orchestrator: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Could not connect to the Orchestrator: {e}")
        return None

def deregister_from_orchestrator(provider_id: str):
    """Remove provider row when client stops."""
    try:
        response = requests.delete(f"{ORCHESTRATOR_URL}/api/dgn/register", params={"providerId": provider_id})
        if response.status_code == 200:
            logging.info("Provider deregistered.")
        else:
            logging.error(f"Error deregistering provider: {response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Could not connect to the Orchestrator: {e}")

def update_job_status(job_id, status):
    """Update the status of a job."""
    try:
        # Repo has route at /api/dgn/job/[jobId]/route.ts (singular 'job')
        response = requests.put(f"{ORCHESTRATOR_URL}/api/dgn/job/{job_id}", json={"status": status})
        if response.status_code == 200:
            logging.info(f"Job {job_id} status updated to {status}")
        else:
            logging.error(f"Error updating job status: {response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Could not connect to the Orchestrator: {e}")

def _ensure_dirs():
    for d in [INPUT_DIR, OUTPUT_DIR, MODELS_DIR, CACHE_DIR]:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

def _materialize_start_image(job: dict):
    """
    Accepts:
      - job['start_image_base64']: 'data:image/png;base64,...' or plain base64 (preferred from Supabase)
      - job['start_image_filename']: stored file already present in mounted input dir

    Writes file into INPUT_DIR (host path mounted to /opt/ComfyUI/input) and returns the filename to use in workflow.
    Always prefers start_image_base64 when present.
    """
    try:
        # 1) Preferred path: Supabase provides base64 under 'start_image_base64'
        data_url = job.get('start_image_base64')
        if isinstance(data_url, str) and len(data_url) > 0:
            # Extract base64 regardless of data URL or raw base64
            b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
            try:
                binary = base64.b64decode(b64, validate=True)
            except Exception:
                # Fallback to non-strict decode if upstream added whitespace/newlines
                binary = base64.b64decode(b64)
            # Filename deterministic by job id unless explicit name provided
            fname = job.get('start_image_name') or f"start_{job.get('id', 'job')}.png"
            out_path = os.path.join(INPUT_DIR, fname)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(binary)
            logging.info(f"Start image (from base64) written to {out_path}")
            return fname

        # 2) Fallback: use provided filename that should already exist in mounted input
        fname = job.get('start_image_filename')
        if isinstance(fname, str) and len(fname) > 0:
            host_path = os.path.join(INPUT_DIR, fname)
            if not os.path.exists(host_path):
                logging.warning(f"Expected start image not found in mounted input: {host_path}")
            else:
                logging.info(f"Using existing start image from input mount: {fname}")
            return fname
    except Exception as e:
        logging.error(f"Failed to materialize start image: {e}")
    return None

def _inject_prompt_and_image_into_workflow(workflow_api_path: str, prompt: str, negative_prompt: str, start_image_filename: str):
    """
    Loads a ComfyUI API-formatted workflow, injects prompts and image filename.
    """
    with open(workflow_api_path, 'r') as f:
        workflow_api = json.load(f)

    # Deep copy to avoid modifying the cached workflow
    api_graph = copy.deepcopy(workflow_api["prompt"])

    # Inject prompts and image filename
    for node in api_graph.values():
        if node["class_type"] == "CLIPTextEncode":
            if "Positive" in node.get("title", ""):
                node["inputs"]["text"] = prompt
            elif "Negative" in node.get("title", ""):
                node["inputs"]["text"] = negative_prompt
        elif node["class_type"] == "LoadImage":
            node["inputs"]["image"] = start_image_filename

    return {"prompt": api_graph}

def listen_for_jobs(provider_id):
    """Listen for jobs from the orchestrator."""
    _ensure_dirs()
    while True:
        try:
            logging.info("Checking for new jobs...")
            response = requests.get(f"{ORCHESTRATOR_URL}/api/dgn/jobs/{provider_id}")
            if response.status_code == 200:
                job = response.json()
                if job:
                    logging.info(f"Received job: {job['id']}")
                    try:
                        update_job_status(job['id'], 'processing')

                        # Do NOT start or pull any Docker image; assume ComfyUI is already running.
                        # container = run_container()
                        # time.sleep(10) # Wait for ComfyUI to start

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

                        # Emit targeted debug to catch missing class_type before sending.
                        # wf_ready may be either {"prompt": {...}} or a bare graph.
                        graph = None
                        if isinstance(wf_ready, dict) and "prompt" in wf_ready and isinstance(wf_ready["prompt"], dict):
                            graph = wf_ready["prompt"]
                        else:
                            graph = wf_ready

                        # Determine nodes list if this is litegraph format; otherwise try API dict values.
                        nodes_ready = []
                        if isinstance(graph, dict) and isinstance(graph.get("nodes"), list):
                            nodes_ready = graph.get("nodes", [])
                        elif isinstance(graph, dict) and all(isinstance(k, (str, int)) and isinstance(v, dict) for k, v in graph.items()):
                            nodes_ready = list(graph.values())

                        missing = [getattr(n, "get", lambda k, d=None: None)("id") for n in nodes_ready if isinstance(n, dict) and not n.get("class_type")]
                        if missing:
                            logging.error(f"Normalization failed to assign class_type on nodes: {missing}")
                        else:
                            sample = []
                            for n in nodes_ready[:5]:
                                if isinstance(n, dict):
                                    sample.append({"id": n.get("id"), "class_type": n.get("class_type")})
                            logging.info(f"First nodes after normalization: {sample}")

                        try:
                            if nodes_ready:
                                id_map = {}
                                for n in nodes_ready:
                                    if isinstance(n, dict):
                                        nid = n.get('id')
                                        if not isinstance(nid, int):
                                            try:
                                                nid = int(str(nid)) if nid is not None else None
                                            except Exception:
                                                pass
                                        id_map[str(nid)] = (n.get('class_type') or '')
                                logging.info(f"ID->class_type map (count={len(id_map)}): {list(id_map.items())[:10]} ...")
                        except Exception:
                            pass

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
                                        # List of refs form: [[nid, idx], [nid, idx], ...]
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

                        prompt_id = trigger_workflow(payload)
                        if prompt_id:
                            outputs = get_workflow_output(prompt_id, terminal_node_ids=terminal_ids, timeout_sec=7200)
                            if outputs:
                                process_workflow_output(outputs, job['id']) # Pass job_id for storage path
                                update_job_status(job['id'], 'completed')
                            else:
                                logging.error("Workflow failed to produce outputs.")
                                update_job_status(job['id'], 'failed')
                        else:
                            logging.error("Failed to trigger workflow.")
                            update_job_status(job['id'], 'failed')

                        # No container management; ComfyUI is managed externally.
                    except Exception as e:
                        logging.error(f"An error occurred while processing job {job['id']}: {e}")
                        update_job_status(job['id'], 'failed')

                else:
                    logging.info("No new jobs.")
            else:
                logging.error(f"Error checking for jobs: {response.text}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Could not connect to the Orchestrator: {e}")

        time.sleep(10) # Poll every 10 seconds

def verify_workflow_nodes(workflow):
    """Verify nodes are approved regardless of workflow shape and prefer class_type over type."""
    APPROVED_NODES = [
        # Core samplers/loaders
        'KSampler',
        'KSamplerAdvanced',
        'VAELoader',
        'VAEDecode',
        'CLIPLoader',
        'CLIPTextEncode',
        'EmptyLatentImage',
        'Empty Latent Image',
        'LoraLoaderModelOnly',
        'ModelSamplingSD3',
        'LoadImage',
        'PreviewImage',
        # GGUF and UNet variants found in templates
        'UnetLoaderGGUF',
        'UNETLoader',  # allow standard UNet loader used by many templates
        # KJNodes / Video helpers
        'ImageResizeKJv2',
        'VHS_VideoCombine',
        'WanVideoNAG',
        'PathchSageAttentionKJ',
        'ModelPatchTorchSettings',
        'WanImageToVideo',
        # Misc / meta
        'Note',
    ]

    # Iterate nodes from multiple possible shapes: API dict, wrapped {"prompt": {...}}, or litegraph arrays.
    def iter_nodes(obj):
        # Wrapped API format
        if isinstance(obj, dict) and isinstance(obj.get("prompt"), dict):
            for n in obj["prompt"].values():
                if isinstance(n, dict):
                    yield n
            return
        # API dict format
        if isinstance(obj, dict) and all(isinstance(k, (str, int)) and isinstance(v, dict) for k, v in obj.items()):
            for n in obj.values():
                yield n
            return
        # Litegraph arrays
        if isinstance(obj, dict) and isinstance(obj.get("nodes"), list):
            for n in obj["nodes"]:
                if isinstance(n, dict):
                    yield n
            return
        if isinstance(obj, dict) and isinstance(obj.get("graph"), dict) and isinstance(obj["graph"].get("nodes"), list):
            for n in obj["graph"]["nodes"]:
                if isinstance(n, dict):
                    yield n
            return

    ok = True
    for node in iter_nodes(workflow):
        node_type = (node.get("class_type") or node.get("type") or "").strip()
        if not node_type:
            logging.error("Security/Validation: node missing both 'class_type' and 'type'.")
            ok = False
            continue
        if node_type not in APPROVED_NODES:
            logging.error(f"Security Alert: Workflow contains a non-approved node: {node_type}")
            ok = False
    return ok

def main():
    """Main function to run the DGN client."""
    parser = argparse.ArgumentParser(description="CrowdMovie DGN Client")
    parser.add_argument("--orchestrator-url", default="http://localhost:3000", help="The URL of the orchestrator")
    args = parser.parse_args()

    global ORCHESTRATOR_URL
    ORCHESTRATOR_URL = args.orchestrator_url

    provider_id = register_with_orchestrator()

    if not provider_id:
        return

    try:
        listen_for_jobs(provider_id)
    finally:
        deregister_from_orchestrator(provider_id)

if __name__ == "__main__":
    main()
