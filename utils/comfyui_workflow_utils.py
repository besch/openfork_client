import os
import json
import base64
import copy
import logging
from datetime import datetime
from typing import Union

# Assuming OUTPUT_DIR, INPUT_DIR are passed or imported from config
# from config import OUTPUT_DIR, INPUT_DIR

def materialize_start_image(job: dict, input_dir: str) -> Union[str, None]:
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
            out_path = os.path.join(input_dir, fname)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(binary)
            logging.info(f"Start image (from base64) written to {out_path}")
            return fname

        # 2) Fallback: use provided filename that should already exist in mounted input
        fname = job.get('start_image_filename')
        if isinstance(fname, str) and len(fname) > 0:
            host_path = os.path.join(input_dir, fname)
            if not os.path.exists(host_path):
                logging.warning(f"Expected start image not found in mounted input: {host_path}")
            else:
                logging.info(f"Using existing start image from input mount: {fname}")
            return fname
    except Exception as e:
        logging.error(f"Failed to materialize start image: {e}")
    return None

def inject_prompt_and_image_into_workflow(workflow_api_path: str, prompt: str, negative_prompt: str, start_image_filename: str):
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
        elif node["class_type"] == "VHS_VideoCombine":
            # Replace date token in filename_prefix
            prefix = node["inputs"].get("filename_prefix", "")
            if "%date:yyyy-MM-dd%" in prefix:
                datestr = datetime.now().strftime("%Y-%m-%d")
                node["inputs"]["filename_prefix"] = prefix.replace("%date:yyyy-MM-dd%", datestr)

    return api_graph

def inject_video_and_prompt_into_foley_workflow(workflow_api_path: str, video_filename: str, prompt: str, negative_prompt: str):
    """
    Loads the Foley ComfyUI API-formatted workflow, injects video filename and prompts.
    """
    with open(workflow_api_path, 'r') as f:
        workflow_api = json.load(f)

    # Deep copy to avoid modifying the cached workflow
    api_graph = copy.deepcopy(workflow_api)

    # Inject video filename and prompts
    for node in api_graph.values():
        if node["class_type"] == "HunyuanVideoFoleyGeneratorAdvanced":
            node["inputs"]["video"] = video_filename
            node["inputs"]["text_prompt"] = prompt
            node["inputs"]["negative_prompt"] = negative_prompt

    return api_graph

def process_workflow_output(outputs: dict, job_id: str, output_dir: str, upload_output_func) -> Union[str, None]:
    """Process the workflow output, upload the generated files, and return the first successful upload path."""
    logging.info(f"Processing workflow outputs for job {job_id}. Outputs received: {json.dumps(outputs, indent=2)}")

    for node_id, node_output in outputs.items():
        logging.info(f"Checking node_id: {node_id}, node_output keys: {node_output.keys()}")

        # Check for 'ui' key (from websocket 'executed' messages)
        if 'ui' in node_output and isinstance(node_output['ui'], dict):
            logging.info(f"Found 'ui' in node_output for node {node_id}. UI keys: {node_output['ui'].keys()}")

            if 'images' in node_output['ui'] and isinstance(node_output['ui']['images'], list):
                logging.info(f"Found 'images' in node_output['ui'] for node {node_id}. Number of images: {len(node_output['ui']['images'])}")
                for img_info in node_output['ui']['images']:
                    if 'filename' in img_info:
                        filename = img_info['filename']
                        file_path = os.path.join(output_dir, filename)
                        logging.info(f"Checking image file: {file_path}")
                        if os.path.exists(file_path):
                            logging.info(f"Image file found: {file_path}. Attempting upload.")
                            storage_path = upload_output_func(file_path, job_id)
                            if storage_path:
                                return storage_path
                        else:
                            logging.warning(f"Output image file not found: {file_path}")
                    else:
                        logging.warning(f"Image info missing 'filename' for node {node_id}: {img_info}")
            else:
                logging.info(f"No 'images' found or not a list in node_output['ui'] for node {node_id}.")

            if 'videos' in node_output['ui'] and isinstance(node_output['ui']['videos'], list):
                logging.info(f"Found 'videos' in node_output['ui'] for node {node_id}. Number of videos: {len(node_output['ui']['videos'])}")
                for video_info in node_output['ui']['videos']:
                    if 'filename' in video_info:
                        filename = video_info['filename']
                        file_path = os.path.join(output_dir, filename)
                        logging.info(f"Checking video file: {file_path}")
                        if os.path.exists(file_path):
                            logging.info(f"Video file found: {file_path}. Attempting upload.")
                            storage_path = upload_output_func(file_path, job_id)
                            if storage_path:
                                return storage_path
                        else:
                            logging.warning(f"Output video file not found: {file_path}")
                    else:
                        logging.warning(f"Video info missing 'filename' for node {node_id}: {video_info}")
            else:
                logging.info(f"No 'videos' found or not a list in node_output['ui'] for node {node_id}.")

        # New: Check for 'gifs' key (from history endpoint)
        elif 'gifs' in node_output and isinstance(node_output['gifs'], list):
            logging.info(f"Found 'gifs' in node_output for node {node_id}. Number of gifs: {len(node_output['gifs'])}")
            for gif_info in node_output['gifs']:
                if 'filename' in gif_info:
                    filename = gif_info['filename']
                    # The fullpath is also available, but let's stick to filename and OUTPUT_DIR for consistency
                    file_path = os.path.join(output_dir, gif_info.get('subfolder', ''), filename)
                    logging.info(f"Checking gif/video file from history: {file_path}")
                    if os.path.exists(file_path):
                        logging.info(f"Gif/video file found: {file_path}. Attempting upload.")
                        storage_path = upload_output_func(file_path, job_id)
                        if storage_path:
                            return storage_path
                    else:
                        logging.warning(f"Output gif/video file not found: {file_path}")
                else:
                    logging.warning(f"Gif info missing 'filename' for node {node_id}: {gif_info}")
        else:
            logging.info(f"No 'ui' or 'gifs' found or not a dict/list in node_output for node {node_id}.")

    logging.error(f"Workflow failed to produce outputs for job {job_id}. No valid output files found after checking all nodes.")
    return None

def verify_workflow_nodes(workflow: dict) -> bool:
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