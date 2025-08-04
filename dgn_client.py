import time
import requests
import os
import logging
import argparse
import base64
import copy

from comfyui_manager import trigger_workflow, get_workflow_output
from hardware_profiler import get_hardware_profile
from config import ROOT_DIR, ORCHESTRATOR_URL, SUPABASE_URL, SUPABASE_ANON_KEY, CACHE_DIR
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define local input/output/models directories used for client I/O.
# These are used to materialize optional input image and read outputs if ComfyUI writes into a shared path.
INPUT_DIR = os.path.join(ROOT_DIR, "input")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
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
    Accepts either:
      - job['start_image_base64']: 'data:image/png;base64,...' or plain base64
      - job['start_image_filename']: stored file already present in mounted input dir
    Writes file into INPUT_DIR and returns filename used in workflow.
    """
    try:
        if 'start_image_base64' in job and job['start_image_base64']:
            data_url = job['start_image_base64']
            if "," in data_url:
                _, b64 = data_url.split(",", 1)
            else:
                b64 = data_url
            binary = base64.b64decode(b64)
            fname = job.get('start_image_name') or f"start_{job['id']}.png"
            out_path = os.path.join(INPUT_DIR, fname)
            with open(out_path, "wb") as f:
                f.write(binary)
            logging.info(f"Start image written to {out_path}")
            return fname
        if 'start_image_filename' in job and job['start_image_filename']:
            # Assume orchestrator instructed a known file name that already exists in INPUT_DIR
            fname = job['start_image_filename']
            logging.info(f"Using existing start image from input mount: {fname}")
            return fname
    except Exception as e:
        logging.error(f"Failed to materialize start image: {e}")
    return None

def _inject_prompt_and_image_into_workflow(workflow: dict, prompt: str, negative_prompt, start_image_filename):
    """
    Normalize a ComfyUI workflow graph and inject prompt/image:
      - Ensure every node has a class_type (ComfyUI requires this key)
      - Ensure inputs/outputs arrays exist
      - Update CLIPTextEncode text fields
      - Update LoadImage filename
      - Ensure a stable 'id' per node (ComfyUI expects numeric id)
      - Sanitize links to use numeric node ids (no '#id' placeholders)
    """
    wf = copy.deepcopy(workflow)
    # If payload is wrapped under {"prompt": {...}}, unwrap so we can sanitize uniformly.
    if isinstance(wf, dict) and "prompt" in wf and isinstance(wf["prompt"], dict):
        wf = wf["prompt"]

    # Some exports wrap under {'workflow': {...}} or similar; unwrap if needed
    if isinstance(wf, dict) and "workflow" in wf and isinstance(wf["workflow"], dict):
        wf = wf["workflow"]

    nodes = wf.get("nodes", [])
    # Build mapping from original id to normalized int id
    id_map: dict[str | int, int] = {}
    # If nodes is not a list, try common wrappers (e.g., {"graph": {"nodes": [...]}})
    if not isinstance(nodes, list):
        graph = wf.get("graph")
        if isinstance(graph, dict) and isinstance(graph.get("nodes"), list):
            nodes = graph.get("nodes")
            wf["nodes"] = nodes

    # 1) Normalize nodes to include required keys and build id_map
    for idx, n in enumerate(nodes):
        orig_id = n.get("id", idx)
        norm_id: int
        try:
            norm_id = int(orig_id)
        except Exception:
            norm_id = idx
        n["id"] = norm_id
        id_map[orig_id] = norm_id

        # class_type is required; fallback to 'type' or title-derived
        if not n.get("class_type"):
            if isinstance(n.get("type"), str) and n["type"]:
                n["class_type"] = n["type"]
            else:
                title = n.get("title") or ""
                n["class_type"] = title.replace(" ", "") or "Unknown"

        # Ensure structural arrays exist
        if not isinstance(n.get("inputs"), list):
            n["inputs"] = n.get("inputs", []) or []
        if not isinstance(n.get("outputs"), list):
            n["outputs"] = n.get("outputs", []) or []

    # 2) Sanitize links to ensure numeric ids and valid references
    # ComfyUI links format: [link_id, src_node_id, src_slot, dst_node_id, dst_slot, type]
    links = wf.get("links", [])
    # Try alternate location if not present (some editors store links on a 'graph' object)
    if not isinstance(links, list):
        graph = wf.get("graph")
        if isinstance(graph, dict) and isinstance(graph.get("links"), list):
            links = graph.get("links")
            wf["links"] = links
    sanitized_links = []
    for link in links if isinstance(links, list) else []:
        try:
            if not isinstance(link, list) or len(link) < 6:
                continue
            l_id, src_id, src_slot, dst_id, dst_slot, l_type = link[:6]
            # Convert src/dst ids through id_map (handles string placeholders like '#id')
            src_norm = id_map.get(src_id, int(src_id) if isinstance(src_id, (int, str)) and str(src_id).isdigit() else None)
            dst_norm = id_map.get(dst_id, int(dst_id) if isinstance(dst_id, (int, str)) and str(dst_id).isdigit() else None)
            if src_norm is None or dst_norm is None:
                # Drop links pointing to unknown ids (prevents ComfyUI '#id' errors)
                continue
            # Recompose link with normalized ids
            sanitized_links.append([l_id, src_norm, src_slot, dst_norm, dst_slot, l_type])
        except Exception:
            # Skip malformed link silently
            continue
    if links and not sanitized_links:
        logging.warning("All links were sanitized away due to invalid ids; ComfyUI may rebuild wiring implicitly.")
    wf["links"] = sanitized_links
    # Also reflect into 'graph' wrapper if present so ws/ui stays consistent
    if isinstance(wf.get("graph"), dict):
        wf["graph"]["links"] = sanitized_links

    # 3) Inject prompts and start image
    for n in nodes:
        t = (n.get("class_type") or n.get("type") or "").strip()
        # Set prompts
        if t == "CLIPTextEncode":
            title = n.get("title", "")
            if "Positive" in title or "Positive Prompt" in title or title == "CLIP Text Encode (Positive Prompt)":
                if n.get("widgets_values") and len(n["widgets_values"]) > 0:
                    n["widgets_values"][0] = prompt
            elif "Negative" in title or "Negative Prompt" in title or title == "CLIP Text Encode (Negative Prompt)":
                if negative_prompt is not None and n.get("widgets_values") and len(n["widgets_values"]) > 0:
                    n["widgets_values"][0] = negative_prompt
        # Set LoadImage file for i2v workflow
        if t in ("LoadImage", "Load Image") and start_image_filename:
            if n.get("widgets_values") and len(n["widgets_values"]) > 0:
                n["widgets_values"][0] = start_image_filename
        # Guardrail: ensure class_type is final non-empty
        if not n.get("class_type"):
            n["class_type"] = n.get("type") or (n.get("title") or "").replace(" ", "") or "Unknown"

    # 4) Final validation logging
    try:
        missing_cls = [n["id"] for n in nodes if not n.get("class_type")]
        if missing_cls:
            logging.error(f"Post-sanitize missing class_type on nodes: {missing_cls}")
        # Dump first few sanitized links to verify id normalization
        logging.info(f"Sanitized links (sample): {wf.get('links', [])[:5]}")
    except Exception:
        pass

    # If original payload was wrapped under {"prompt": ...}, rewrap to keep API compatibility
    if isinstance(workflow, dict) and "prompt" in workflow and isinstance(workflow["prompt"], dict):
        return {"prompt": wf}
    return wf

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

                        if not workflow:
                            logging.error("No workflow found in job.")
                            update_job_status(job['id'], 'failed')
                            continue

                        if not verify_workflow_nodes(workflow):
                            update_job_status(job['id'], 'failed')
                            continue

                        if required_assets:
                            download_assets(required_assets)

                        start_image_filename = _materialize_start_image(job)
                        wf_ready = _inject_prompt_and_image_into_workflow(
                            workflow, positive_prompt, negative_prompt, start_image_filename
                        )

                        # Emit targeted debug to catch missing class_type before sending
                        nodes_ready = wf_ready.get("nodes", [])
                        missing = [n.get("id") for n in nodes_ready if not n.get("class_type")]
                        if missing:
                            logging.error(f"Normalization failed to assign class_type on nodes: {missing}")
                        else:
                            sample = [
                                {"id": n.get("id"), "class_type": n.get("class_type")}
                                for n in nodes_ready[:5]
                            ]
                            logging.info(f"First nodes after normalization: {sample}")
                        try:
                            # Also log a compact id->class_type map to correlate with ComfyUI's '#id' error
                            id_map = { (n.get('id') if isinstance(n.get('id'), int) else str(n.get('id'))): (n.get('class_type') or '') for n in nodes_ready }
                            logging.info(f"ID->class_type map (count={len(id_map)}): {list(id_map.items())[:10]} ...")
                        except Exception:
                            pass

                        # If we re-wrapped under {"prompt": ...}, send that. Otherwise send the flat graph.
                        payload = wf_ready
                        if isinstance(wf_ready, dict) and "nodes" in wf_ready:
                            payload = {"prompt": wf_ready}
                        prompt_id = trigger_workflow(payload)
                        if prompt_id:
                            outputs = get_workflow_output(prompt_id)
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
    """Verify that all nodes in the workflow are on the approved list."""
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

    for node in workflow.get('nodes', []):
        if node.get('type') not in APPROVED_NODES:
            logging.error(f"Security Alert: Workflow contains a non-approved node: {node.get('type')}")
            return False
    return True

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
