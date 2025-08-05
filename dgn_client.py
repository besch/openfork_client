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
    Convert incoming workflow to ComfyUI API graph format and inject prompt/image.
    Guarantees returning {"prompt": { "<id>": { "class_type": "...", "inputs": {...} }, ... }}.
    Rejects nodes missing class_type to avoid ComfyUI 400 errors.
    """
    raw = copy.deepcopy(workflow)

    # 0) Unwrap common wrappers to get inner graph-like object
    if isinstance(raw, dict) and "prompt" in raw and isinstance(raw["prompt"], (dict, list)):
        inner = raw["prompt"]
    else:
        inner = raw
    if isinstance(inner, dict) and "workflow" in inner and isinstance(inner["workflow"], dict):
        inner = inner["workflow"]
    if isinstance(inner, dict) and "graph" in inner and isinstance(inner["graph"], dict) and "nodes" in inner["graph"]:
        # litegraph nested
        inner = {"nodes": inner["graph"]["nodes"], "links": inner["graph"].get("links", [])}

    # 1) If already API dict format (ids -> node dict with class_type), clone and modify fields
    if isinstance(inner, dict) and inner and all(isinstance(k, (str, int)) and isinstance(v, dict) for k, v in inner.items()) and not isinstance(inner.get("nodes"), list):
        api_graph = {}
        for k, node in inner.items():
            node_copy = copy.deepcopy(node)
            ctype = (node_copy.get("class_type") or "").strip()
            if not ctype:
                raise ValueError(f"Workflow node {k} missing 'class_type'.")
            # Inject prompts/image if fields exist (widgets_values is litegraph; API uses inputs usually)
            if ctype == "CLIPTextEncode":
                # Prefer API "text" input if present
                inputs = node_copy.setdefault("inputs", {})
                if isinstance(inputs, dict):
                    title = node_copy.get("title", "")
                    if "Negative" in title and negative_prompt is not None:
                        inputs["text"] = negative_prompt
                    elif "Positive" in title or not title:
                        inputs["text"] = prompt
            if ctype in ("LoadImage", "Load Image") and start_image_filename:
                inputs = node_copy.setdefault("inputs", {})
                if isinstance(inputs, dict):
                    inputs["image"] = start_image_filename
            api_graph[str(k)] = node_copy
        return {"prompt": api_graph}

    # 2) Handle litegraph array format: nodes/links arrays
    nodes = []
    links = []
    if isinstance(inner, dict):
        if isinstance(inner.get("nodes"), list):
            nodes = copy.deepcopy(inner["nodes"])
        if isinstance(inner.get("links"), list):
            links = copy.deepcopy(inner["links"])

    # Build id mapping to numeric ids and DROP non-executable/meta nodes (e.g., Note) to avoid server errors.
    id_map = {}
    filtered_nodes = []
    meta_node_types = {"Note"}  # extendable set of UI-only nodes to strip
    for idx, n in enumerate(nodes):
        n_type = (n.get("class_type") or n.get("type") or "").strip()
        # Skip meta/UI-only nodes (e.g., Note) entirely
        if n_type in meta_node_types:
            continue

        orig_id = n.get("id", idx)
        if isinstance(orig_id, str) and (orig_id.strip() == "#id" or not orig_id.strip().lstrip("-").isdigit()):
            orig_id_key = f"auto_{idx}"
        else:
            orig_id_key = orig_id
        try:
            norm_id = int(orig_id)
        except Exception:
            norm_id = idx
        n["id"] = norm_id
        id_map[orig_id_key] = norm_id

        # Ensure class_type
        if not n.get("class_type"):
            if isinstance(n.get("type"), str) and n["type"]:
                n["class_type"] = n["type"]
            else:
                title = n.get("title") or ""
                inferred = title.replace(" ", "")
                if not inferred:
                    raise ValueError(f"Workflow contains a node (idx {idx}) missing 'class_type' and 'type'.")
                n["class_type"] = inferred

        # Ensure widget arrays exist
        if "widgets_values" not in n or not isinstance(n.get("widgets_values"), list):
            n["widgets_values"] = []

        filtered_nodes.append(n)

    nodes = filtered_nodes

    # Sanitize links (optional; not required for API dict format we will build)
    sanitized_links = []
    # Build set of kept node ids for link pruning after dropping meta nodes
    kept_ids = {n["id"] for n in nodes if isinstance(n.get("id"), int)}
    for link in links:
        if not isinstance(link, list) or len(link) < 6:
            continue
        _, src_id, src_slot, dst_id, dst_slot, _ = link[:6]

        def resolve(v):
            if isinstance(v, str) and v.strip() == "#id":
                return None
            if v in id_map:
                return id_map[v]
            if isinstance(v, str) and v.strip().lstrip("-").isdigit():
                iv = int(v)
                return id_map.get(iv, iv)
            if isinstance(v, int):
                return id_map.get(v, v)
            return None

        s = resolve(src_id)
        d = resolve(dst_id)
        # Drop links if either endpoint is missing or pruned (e.g., Note)
        if s is None or d is None or s not in kept_ids or d not in kept_ids:
            continue
        sanitized_links.append([s, src_slot, d, dst_slot])
    # We won't attempt to map slots to named inputs; ComfyUI API allows direct value setting.

    # 3) Inject prompts/image in litegraph nodes (widgets_values commonly used)
    for n in nodes:
        t = (n.get("class_type") or n.get("type") or "").strip()
        if t == "CLIPTextEncode":
            title = n.get("title", "")
            if "Negative" in title and negative_prompt is not None:
                if n.get("widgets_values") and len(n["widgets_values"]) > 0:
                    n["widgets_values"][0] = negative_prompt
            else:
                if n.get("widgets_values") and len(n["widgets_values"]) > 0:
                    n["widgets_values"][0] = prompt
        if t in ("LoadImage", "Load Image") and start_image_filename:
            if n.get("widgets_values") and len(n["widgets_values"]) > 0:
                n["widgets_values"][0] = start_image_filename

    # 4) Convert to ComfyUI API dict format: {"<id>": {"class_type": ..., "inputs": {...}}}
    api_graph = {}

    # Build a quick lookup by id for later link-to-input reconstruction
    node_by_id = {n.get("id"): n for n in nodes if isinstance(n.get("id"), int)}

    for n in nodes:
        nid = n.get("id")
        ctype = (n.get("class_type") or "").strip()
        if not ctype:
            raise ValueError(f"Workflow node {nid} missing 'class_type' after normalization.")

        # Build inputs dict and preserve any existing API-like inputs
        inputs = {}
        if isinstance(n.get("inputs"), dict):
            inputs.update({k: v for k, v in n["inputs"].items()})

        # Known simple mappings
        if ctype == "CLIPTextEncode":
            if isinstance(n.get("widgets_values"), list) and n["widgets_values"]:
                inputs["text"] = n["widgets_values"][0]
        elif ctype in ("LoadImage", "Load Image"):
            if isinstance(n.get("widgets_values"), list) and n["widgets_values"]:
                inputs["image"] = n["widgets_values"][0]
        elif ctype == "VHS_VideoCombine":
            # Populate required VHS fields with sensible defaults
            inputs.setdefault("crf", 19)
            inputs.setdefault("format", "video/h264-mp4")
            inputs.setdefault("pix_fmt", "yuv420p")
            inputs.setdefault("pingpong", False)
            inputs.setdefault("frame_rate", 16)
            inputs.setdefault("loop_count", 0)
            inputs.setdefault("save_output", True)
            inputs.setdefault("save_metadata", True)
            inputs.setdefault("trim_to_audio", False)
            inputs.setdefault("filename_prefix", "%date:yyyy-MM-dd%/generated_")
        elif ctype == "WanImageToVideo":
            # Ensure Wan I2V node has mandatory scalar inputs and VAE/prompt wiring will be added via links
            # Pull width/height/length/batch_size from widgets if available; otherwise use sensible defaults
            wv = n.get("widgets_values") if isinstance(n.get("widgets_values"), list) else []
            if "width" not in inputs:
                inputs["width"] = wv[0] if len(wv) > 0 and isinstance(wv[0], int) else 832
            if "height" not in inputs:
                inputs["height"] = wv[1] if len(wv) > 1 and isinstance(wv[1], int) else 480
            if "length" not in inputs:
                inputs["length"] = wv[2] if len(wv) > 2 and isinstance(wv[2], int) else 81
            if "batch_size" not in inputs:
                inputs["batch_size"] = wv[3] if len(wv) > 3 and isinstance(wv[3], int) else 1
        elif ctype == "VAELoader":
            # Provide a default VAE selection if missing
            if "vae_name" not in inputs:
                if isinstance(n.get("widgets_values"), list) and n["widgets_values"]:
                    inputs["vae_name"] = n["widgets_values"][0]
                else:
                    inputs["vae_name"] = "wan_2.1_vae.safetensors"
        elif ctype in ("KSampler", "KSamplerAdvanced"):
            # Provide sensible defaults so validation passes; graph links will set model/positive/negative/latent_image
            inputs.setdefault("steps", 20)
            inputs.setdefault("cfg", 4.5)
            inputs.setdefault("sampler_name", "euler")
            inputs.setdefault("scheduler", "normal")
            inputs.setdefault("start_at_step", 0)
            inputs.setdefault("end_at_step", 20)
            # ComfyUI expects specific string options for these toggles, not booleans
            inputs.setdefault("add_noise", "enable")  # ['enable','disable']
            inputs.setdefault("return_with_leftover_noise", "disable")  # ['disable','enable']
            inputs.setdefault("noise_seed", 0)
        elif ctype in ("UnetLoaderGGUF", "UNETLoader"):
            # Ensure required UNet name is populated
            if "unet_name" not in inputs:
                if isinstance(n.get("widgets_values"), list) and n["widgets_values"]:
                    inputs["unet_name"] = n["widgets_values"][0]
        elif ctype == "CLIPLoader":
            # Normalize optional CLIPLoader params from widgets if not explicitly present
            wv = n.get("widgets_values") if isinstance(n.get("widgets_values"), list) else []
            if "clip_name" not in inputs and wv:
                inputs["clip_name"] = wv[0]
            if "type" not in inputs and len(wv) > 1:
                inputs["type"] = wv[1]
            if "device" not in inputs and len(wv) > 2:
                inputs["device"] = wv[2]

        api_graph[str(nid)] = {
            "class_type": ctype,
            "inputs": inputs
        }

    # 5) Reconstruct critical connections from sanitized_links into API inputs:
    # Map upstream connections into API references: ["<node_id>", output_index]
    # Handle:
    #   - VAELoader -> VAEDecode.vae
    #   - KSampler/KSamplerAdvanced -> VAEDecode.samples
    #   - VAEDecode.IMAGE (or other IMAGE producers) -> VHS_VideoCombine.images
    #   - CLIPTextEncode -> KSampler/KSamplerAdvanced.positive/negative
    #   - MODEL producers (UnetLoaderGGUF/UNETLoader) -> KSampler/KSamplerAdvanced.model
    #   - EmptyLatentImage (or latent producers) -> KSampler/KSamplerAdvanced.latent_image
    #   - CLIPLoader -> KSampler/KSamplerAdvanced.clip (some templates wire it explicitly)
    for link in sanitized_links:
        try:
            src_id, src_slot, dst_id, dst_slot = link
            dst_node = node_by_id.get(dst_id)
            src_node = node_by_id.get(src_id)
            if not dst_node or not src_node:
                continue
            dst_type = (dst_node.get("class_type") or dst_node.get("type") or "").strip()
            src_type = (src_node.get("class_type") or src_node.get("type") or "").strip()

            # Bind VAELoader -> VAEDecode.vae
            if dst_type == "VAEDecode" and src_type in ("VAELoader", "VAE", "VAELoaderNode"):
                dst_key = str(dst_id)
                api_inputs = api_graph.get(dst_key, {}).get("inputs", {})
                if "vae" not in api_inputs:
                    api_inputs["vae"] = [str(src_id), 0]
                    api_graph[dst_key]["inputs"] = api_inputs

            # Bind KSampler/KSamplerAdvanced (LATENT) -> VAEDecode.samples
            if dst_type == "VAEDecode" and src_type in ("KSampler", "KSamplerAdvanced", "Latent", "LatentNode"):
                dst_key = str(dst_id)
                api_inputs = api_graph.get(dst_key, {}).get("inputs", {})
                if "samples" not in api_inputs:
                    api_inputs["samples"] = [str(src_id), 0]
                    api_graph[dst_key]["inputs"] = api_inputs

            # Bind IMAGE producer -> VHS_VideoCombine.images
            if dst_type == "VHS_VideoCombine" and src_type in ("VAEDecode", "LoadImage", "Image", "ImageNode"):
                dst_key = str(dst_id)
                api_inputs = api_graph.get(dst_key, {}).get("inputs", {})
                if "images" not in api_inputs:
                    api_inputs["images"] = [str(src_id), 0]
                    api_graph[dst_key]["inputs"] = api_inputs

            # Bind VAELoader -> WanImageToVideo.vae
            if dst_type == "WanImageToVideo" and src_type in ("VAELoader", "VAE", "VAELoaderNode"):
                dst_key = str(dst_id)
                api_inputs = api_graph.get(dst_key, {}).get("inputs", {})
                if "vae" not in api_inputs:
                    api_inputs["vae"] = [str(src_id), 0]
                    api_graph[dst_key]["inputs"] = api_inputs

            # Bind CLIPTextEncode -> WanImageToVideo positive/negative
            if dst_type == "WanImageToVideo" and src_type == "CLIPTextEncode":
                dst_key = str(dst_id)
                api_inputs = api_graph.get(dst_key, {}).get("inputs", {})
                src_title = (node_by_id.get(src_id, {}) or {}).get("title", "") if isinstance(node_by_id.get(src_id, {}), dict) else ""
                is_negative = ("Negative" in src_title) or (dst_slot == 1)
                if is_negative:
                    if "negative" not in api_inputs:
                        api_inputs["negative"] = [str(src_id), 0]
                else:
                    if "positive" not in api_inputs:
                        api_inputs["positive"] = [str(src_id), 0]
                api_graph[dst_key]["inputs"] = api_inputs

            # Bind ImageResizeKJv2 width/height -> WanImageToVideo.width/height
            if dst_type == "WanImageToVideo" and src_type == "ImageResizeKJv2":
                dst_key = str(dst_id)
                api_inputs = api_graph.get(dst_key, {}).get("inputs", {})
                # According to the workflow, ImageResizeKJv2 outputs width at slot 1 and height at slot 2
                if dst_slot == 5 and "width" not in api_inputs:
                    api_inputs["width"] = [str(src_id), 1]
                if dst_slot == 6 and "height" not in api_inputs:
                    api_inputs["height"] = [str(src_id), 2]
                api_graph[dst_key]["inputs"] = api_inputs

            # Bind CLIP encoders to KSampler/KSamplerAdvanced positive/negative
            if dst_type in ("KSampler", "KSamplerAdvanced") and src_type == "CLIPTextEncode":
                dst_key = str(dst_id)
                api_inputs = api_graph.get(dst_key, {}).get("inputs", {})
                # Some exported graphs don't preserve slot semantics; infer by node title/label if available.
                src_title = (node_by_id.get(src_id, {}) or {}).get("title", "") if isinstance(node_by_id.get(src_id, {}), dict) else ""
                is_negative = ("Negative" in src_title) or (dst_slot == 1)
                if is_negative:
                    if "negative" not in api_inputs:
                        api_inputs["negative"] = [str(src_id), 0]
                else:
                    if "positive" not in api_inputs:
                        api_inputs["positive"] = [str(src_id), 0]
                api_graph[dst_key]["inputs"] = api_inputs

            # Bind CLIPLoader to CLIPTextEncode.clip
            if dst_type == "CLIPTextEncode" and src_type == "CLIPLoader":
                dst_key = str(dst_id)
                api_inputs = api_graph.get(dst_key, {}).get("inputs", {})
                if "clip" not in api_inputs:
                    api_inputs["clip"] = [str(src_id), 0]
                    api_graph[dst_key]["inputs"] = api_inputs

            # Bind model loaders to KSampler/KSamplerAdvanced.model
            if dst_type in ("KSampler", "KSamplerAdvanced") and src_type in ("UnetLoaderGGUF", "UNETLoader"):
                dst_key = str(dst_id)
                api_inputs = api_graph.get(dst_key, {}).get("inputs", {})
                if "model" not in api_inputs:
                    api_inputs["model"] = [str(src_id), 0]
                    api_graph[dst_key]["inputs"] = api_inputs

            # Bind CLIPLoader to KSampler/KSamplerAdvanced.clip (if present in graph)
            if dst_type in ("KSampler", "KSamplerAdvanced") and src_type == "CLIPLoader":
                dst_key = str(dst_id)
                api_inputs = api_graph.get(dst_key, {}).get("inputs", {})
                if "clip" not in api_inputs:
                    api_inputs["clip"] = [str(src_id), 0]
                    api_graph[dst_key]["inputs"] = api_inputs

            # Bind latent providers to KSampler/KSamplerAdvanced.latent_image
            if dst_type in ("KSampler", "KSamplerAdvanced") and src_type in ("EmptyLatentImage", "Latent", "LatentNode", "WanImageToVideo"):
                dst_key = str(dst_id)
                api_inputs = api_graph.get(dst_key, {}).get("inputs", {})
                if "latent_image" not in api_inputs:
                    # WanImageToVideo provides latent on slot index 2
                    if src_type == "WanImageToVideo":
                        api_inputs["latent_image"] = [str(src_id), 2]
                    else:
                        api_inputs["latent_image"] = [str(src_id), 0]
                    api_graph[dst_key]["inputs"] = api_inputs
        except Exception:
            continue

    # 6) Fallback wiring if required sampler inputs are still missing but producers exist without links.
    # This covers cases where links are lost in transport (e.g., UI export glitches).
    # Try to heuristically pick producers for model/positive/negative/latent_image.
    try:
        # Collect candidates by type
        ks_ids = [nid for nid, n in ((int(k), v) for k, v in api_graph.items()) if isinstance(n, dict) and (n.get("class_type") in ("KSampler", "KSamplerAdvanced"))]
        model_ids = [nid for nid, n in ((int(k), v) for k, v in api_graph.items()) if isinstance(n, dict) and (n.get("class_type") in ("UnetLoaderGGUF", "UNETLoader"))]
        clip_ids = [nid for nid, n in ((int(k), v) for k, v in api_graph.items()) if isinstance(n, dict) and (n.get("class_type") == "CLIPTextEncode")]
        latent_ids = [nid for nid, n in ((int(k), v) for k, v in api_graph.items()) if isinstance(n, dict) and (n.get("class_type") in ("EmptyLatentImage", "Latent", "LatentNode", "WanImageToVideo"))]

        # Prefer the first of each type if present
        model_src = str(model_ids[0]) if model_ids else None
        # Split CLIPTextEncode into positive/negative by node title if possible, else assign first to positive
        pos_src = None
        neg_src = None
        for cid in clip_ids:
            n = node_by_id.get(cid)
            title = (n.get("title") or "") if isinstance(n, dict) else ""
            if "Negative" in title and not neg_src:
                neg_src = str(cid)
            elif not pos_src:
                pos_src = str(cid)
        latent_src = str(latent_ids[0]) if latent_ids else None

        for ks in ks_ids:
            inputs = api_graph[str(ks)].setdefault("inputs", {})
            if model_src and "model" not in inputs:
                inputs["model"] = [model_src, 0]
            if pos_src and "positive" not in inputs:
                inputs["positive"] = [pos_src, 0]
            if neg_src and "negative" not in inputs and len(clip_ids) > 1:
                inputs["negative"] = [neg_src, 0]
            if latent_src and "latent_image" not in inputs:
                # If the selected latent source is WanImageToVideo, its latent is output slot 2
                src_node = node_by_id.get(int(latent_src)) if latent_src is not None and latent_src.lstrip("-").isdigit() else None
                src_type = (src_node.get("class_type") or src_node.get("type") or "").strip() if isinstance(src_node, dict) else ""
                if src_type == "WanImageToVideo":
                    inputs["latent_image"] = [latent_src, 2]
                else:
                    inputs["latent_image"] = [latent_src, 0]
    except Exception:
        pass

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
