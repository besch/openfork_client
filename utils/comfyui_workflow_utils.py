import os
import json
import base64
import copy
import logging
import random
from datetime import datetime
from typing import Union, Dict, Optional
import random
import logging
import copy
from datetime import datetime

# Assuming OUTPUT_DIR, INPUT_DIR are passed or imported from config
# from config import OUTPUT_DIR, INPUT_DIR


def get_dimensions(aspect_ratio: str, default_width: int = 768, default_height: int = 432) -> tuple[int, int]:
    """
    Returns (width, height) based on the aspect ratio string.
    Using smaller dimensions suitable for GPUs with less VRAM.
    All dimensions are divisible by 16.
    """
    if aspect_ratio == "16:9":
        return 768, 432  # 432p
    elif aspect_ratio == "9:16":
        return 432, 768
    elif aspect_ratio == "1:1":
        return 512, 512
    elif aspect_ratio == "4:3":
        return 640, 480  # 480p
    elif aspect_ratio == "3:4":
        return 480, 640
    elif aspect_ratio == "21:9":
        return 896, 384
    else:
        return default_width, default_height

def materialize_start_image(job: dict, input_dir: str) -> Union[str, None]:
    """
    Accepts:
      - job['start_image_base64'] or job['inputs']['start_image_base64']: 'data:image/png;base64,...' or plain base64
      - job['start_image_filename'] or job['inputs']['start_image_filename']: stored file already present in mounted input dir

    Writes file into INPUT_DIR (host path mounted to /opt/ComfyUI/input) and returns the filename to use in workflow.
    Always prefers start_image_base64 when present.
    """
    try:
        # Get inputs dict for fallback lookup
        inputs = job.get('inputs', {})
        
        # 1) Preferred path: base64 image (check both root and inputs)
        data_url = job.get('start_image_base64') or inputs.get('start_image_base64')
        if isinstance(data_url, str) and len(data_url) > 0:
            # Extract base64 regardless of data URL or raw base64
            b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
            try:
                binary = base64.b64decode(b64, validate=True)
            except Exception:
                # Fallback to non-strict decode if upstream added whitespace/newlines
                binary = base64.b64decode(b64)
            # Filename deterministic by job id unless explicit name provided
            fname = job.get('start_image_name') or inputs.get('start_image_name') or f"start_{job.get('id', 'job')}.png"
            out_path = os.path.join(input_dir, fname)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(binary)
            logging.info(f"Start image (from base64) written to {out_path}")
            return fname

        # 2) Fallback: use provided filename that should already exist in mounted input
        fname = job.get('start_image_filename') or inputs.get('start_image_filename')
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

def inject_prompt_and_image_into_workflow(
    workflow_api_data: Dict, 
    prompt: str, 
    negative_prompt: str, 
    start_image_filename: str, 
    aspect_ratio: str = "16:9",
    cfg_scale: Optional[float] = None,
    steps: Optional[int] = None,
    flow_shift: Optional[float] = None,
    sampler: Optional[str] = None,
    scheduler: Optional[str] = None,
    seed: Optional[int] = None
):
    """
    Loads a ComfyUI API-formatted workflow, injects prompts and image filename.
    Also sets seed for image-to-video generation (random if not provided).
    Optionally injects cfg_scale and steps into KSampler nodes.
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])

    # Inject prompts and image filename
    for node in api_graph.values():
        if node["class_type"] == "CLIPTextEncode":
            if "Positive" in node.get("title", ""):
                node["inputs"]["text"] = prompt
            elif "Negative" in node.get("title", ""):
                node["inputs"]["text"] = negative_prompt
        elif node["class_type"] == "LoadImage":
            node["inputs"]["image"] = start_image_filename
        elif node["class_type"] == "WanImageToVideo":
            width, height = get_dimensions(aspect_ratio)
            node["inputs"]["width"] = width
            node["inputs"]["height"] = height
        elif node["class_type"] == "ImageResizeKJv2":
            width, height = get_dimensions(aspect_ratio)
            node["inputs"]["width"] = width
            node["inputs"]["height"] = height
        elif node["class_type"] == "VHS_VideoCombine":
            # Replace date token in filename_prefix
            prefix = node["inputs"].get("filename_prefix", "")
            if "%date:yyyy-MM-dd%" in prefix:
                datestr = datetime.now().strftime("%Y-%m-%d")
                node["inputs"]["filename_prefix"] = prefix.replace("%date:yyyy-MM-dd%", datestr)

    # Set seed for KSamplerAdvanced nodes (use provided seed or generate random)
    actual_seed = seed if seed is not None else random.randint(0, 2**63 - 1)
    if '57' in api_graph and 'inputs' in api_graph['57']:
        api_graph['57']['inputs']['noise_seed'] = actual_seed

    # Inject cfg and steps into KSampler nodes
    if cfg_scale is not None or steps is not None:
        for node in api_graph.values():
            class_type = node.get("class_type", "")
            if "KSampler" in class_type and "inputs" in node:
                if cfg_scale is not None and "cfg" in node["inputs"]:
                    node["inputs"]["cfg"] = cfg_scale
                if steps is not None and "steps" in node["inputs"]:
                    node["inputs"]["steps"] = steps
    
    # Inject flow_shift, sampler, and scheduler (V2)
    for node in api_graph.values():
        class_type = node.get("class_type", "")
        if class_type == "ModelSamplingSD3" and flow_shift is not None:
            node["inputs"]["shift"] = flow_shift
        elif class_type == "KSamplerSelect" and sampler is not None:
            node["inputs"]["sampler_name"] = sampler
        elif class_type == "BasicScheduler" and scheduler is not None:
            node["inputs"]["scheduler"] = scheduler
        elif class_type == "CFGGuider" and cfg_scale is not None:
            node["inputs"]["cfg"] = cfg_scale

    return api_graph

def inject_prompt_into_text_to_video_workflow(
    workflow_api_data: Dict, 
    prompt: str, 
    negative_prompt: str, 
    aspect_ratio: str = "16:9",
    cfg_scale: Optional[float] = None,
    steps: Optional[int] = None,
    flow_shift: Optional[float] = None,
    sampler: Optional[str] = None,
    scheduler: Optional[str] = None,
    seed: Optional[int] = None
):
    """
    Loads a ComfyUI API-formatted workflow, injects prompts for text-to-video.
    Also sets seed for varied outputs (random if not provided).
    Optionally injects cfg_scale and steps into KSamplerAdvanced nodes.
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])

    # Node 6 is positive, Node 7 is negative
    if '6' in api_graph and 'inputs' in api_graph['6'] and 'text' in api_graph['6']['inputs']:
        api_graph['6']['inputs']['text'] = prompt
    else:
        logging.warning("Could not find positive prompt node 6 in text-to-video workflow")

    if '7' in api_graph and 'inputs' in api_graph['7'] and 'text' in api_graph['7']['inputs']:
        api_graph['7']['inputs']['text'] = negative_prompt
    else:
        logging.warning("Could not find negative prompt node 7 in text-to-video workflow")

    # Inject dimensions into WanImageToVideo
    for node in api_graph.values():
        if node.get("class_type") == "WanImageToVideo":
            width, height = get_dimensions(aspect_ratio)
            node["inputs"]["width"] = width
            node["inputs"]["height"] = height

    # Set seed for node 57 (use provided seed or generate random)
    actual_seed = seed if seed is not None else random.randint(0, 2**63 - 1)
    if '57' in api_graph and 'inputs' in api_graph['57']:
        api_graph['57']['inputs']['noise_seed'] = actual_seed
    else:
        logging.warning("Could not find sampler node 57 to set seed")

    # Inject cfg and steps into KSamplerAdvanced nodes (57 and 58)
    if cfg_scale is not None or steps is not None:
        for node_id in ['57', '58']:
            if node_id in api_graph and api_graph[node_id].get("class_type") == "KSamplerAdvanced":
                if cfg_scale is not None:
                    api_graph[node_id]['inputs']['cfg'] = cfg_scale
                    logging.info(f"Injected cfg={cfg_scale} into KSamplerAdvanced node {node_id}")
                if steps is not None:
                    api_graph[node_id]['inputs']['steps'] = steps
                    logging.info(f"Injected steps={steps} into KSamplerAdvanced node {node_id}")
        
        # Also search for other KSampler variants by class_type
        for node in api_graph.values():
            class_type = node.get("class_type", "")
            if "KSampler" in class_type and "inputs" in node:
                if cfg_scale is not None and "cfg" in node["inputs"]:
                    node["inputs"]["cfg"] = cfg_scale
                if steps is not None and "steps" in node["inputs"]:
                    node["inputs"]["steps"] = steps
        
        # Also handle V2 parameters (flow_shift, sampler, scheduler)
        for node in api_graph.values():
            class_type = node.get("class_type", "")
            if class_type == "ModelSamplingSD3" and flow_shift is not None:
                node["inputs"]["shift"] = flow_shift
            elif class_type == "KSamplerSelect" and sampler is not None:
                node["inputs"]["sampler_name"] = sampler
            elif class_type == "BasicScheduler" and scheduler is not None:
                node["inputs"]["scheduler"] = scheduler
            elif class_type == "CFGGuider" and cfg_scale is not None:
                node["inputs"]["cfg"] = cfg_scale
    # Replace date token in filename_prefix for VHS_VideoCombine node
    for node in api_graph.values():
        if node.get("class_type") == "VHS_VideoCombine":
            prefix = node["inputs"].get("filename_prefix", "")
            if "%date:yyyy-MM-dd%" in prefix:
                datestr = datetime.now().strftime("%Y-%m-%d")
                node["inputs"]["filename_prefix"] = prefix.replace("%date:yyyy-MM-dd%", datestr)

    return api_graph

def inject_prompt_into_ltx_video_workflow(
    workflow_api_data: Dict, 
    prompt: str, 
    negative_prompt: str, 
    aspect_ratio: str = "16:9",
    cfg_scale: Optional[float] = None,
    steps: Optional[int] = None,
    flow_shift: Optional[float] = None,  # Accepted but unused - LTX workflow doesn't use it
    sampler: Optional[str] = None,
    scheduler: Optional[str] = None,
    seed: Optional[int] = None
):
    """
    Loads a ComfyUI API-formatted workflow, injects prompts for LTX-Video.
    LTX-Video uses a different node structure than other models:
    - Node 3: Positive prompt (CLIPTextEncode)
    - Node 4: Negative prompt (CLIPTextEncode)
    - Node 6: EmptyLTXVLatentVideo (for dimensions)
    - Node 7: RandomNoise (for seed)
    - Node 8: KSamplerSelect (for sampler)
    - Node 9: BasicScheduler (for scheduler and steps)
    - Node 10: CFGGuider (for cfg_scale)
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])

    # Node 3 is positive prompt, Node 4 is negative prompt (different from generic workflows!)
    if '3' in api_graph and 'inputs' in api_graph['3'] and 'text' in api_graph['3']['inputs']:
        api_graph['3']['inputs']['text'] = prompt
        logging.info(f"Injected positive prompt into LTX node 3: {prompt[:50]}...")
    else:
        logging.warning("Could not find positive prompt node 3 in LTX workflow")

    if '4' in api_graph and 'inputs' in api_graph['4'] and 'text' in api_graph['4']['inputs']:
        api_graph['4']['inputs']['text'] = negative_prompt
        logging.info(f"Injected negative prompt into LTX node 4")
    else:
        logging.warning("Could not find negative prompt node 4 in LTX workflow")

    # Inject dimensions into EmptyLTXVLatentVideo (Node 6)
    if '6' in api_graph and api_graph['6'].get("class_type") == "EmptyLTXVLatentVideo":
        width, height = get_dimensions(aspect_ratio)
        api_graph['6']['inputs']['width'] = width
        api_graph['6']['inputs']['height'] = height
        logging.info(f"Injected dimensions into LTX node 6: {width}x{height}")

    # Set seed for RandomNoise node (Node 7) - use provided or generate random
    actual_seed = seed if seed is not None else random.randint(0, 2**63 - 1)
    if '7' in api_graph and api_graph['7'].get("class_type") == "RandomNoise":
        api_graph['7']['inputs']['noise_seed'] = actual_seed
        logging.info(f"Set seed in LTX node 7: {actual_seed}")
    else:
        logging.warning("Could not find RandomNoise node 7 to set seed")

    # Inject sampler into KSamplerSelect (Node 8)
    if sampler is not None and '8' in api_graph and api_graph['8'].get("class_type") == "KSamplerSelect":
        api_graph['8']['inputs']['sampler_name'] = sampler
        logging.info(f"Injected sampler into LTX node 8: {sampler}")

    # Inject scheduler and steps into BasicScheduler (Node 9)
    if '9' in api_graph and api_graph['9'].get("class_type") == "BasicScheduler":
        if scheduler is not None:
            api_graph['9']['inputs']['scheduler'] = scheduler
            logging.info(f"Injected scheduler into LTX node 9: {scheduler}")
        if steps is not None:
            api_graph['9']['inputs']['steps'] = steps
            logging.info(f"Injected steps into LTX node 9: {steps}")

    # Inject cfg_scale into CFGGuider (Node 10)
    if cfg_scale is not None and '10' in api_graph and api_graph['10'].get("class_type") == "CFGGuider":
        api_graph['10']['inputs']['cfg'] = cfg_scale
        logging.info(f"Injected cfg into LTX node 10: {cfg_scale}")

    # Replace date token in filename_prefix for VHS_VideoCombine node
    for node in api_graph.values():
        if node.get("class_type") == "VHS_VideoCombine":
            prefix = node["inputs"].get("filename_prefix", "")
            if "%date:yyyy-MM-dd%" in prefix:
                datestr = datetime.now().strftime("%Y-%m-%d")
                node["inputs"]["filename_prefix"] = prefix.replace("%date:yyyy-MM-dd%", datestr)

    return api_graph

def inject_prompt_and_image_into_ltx_video_workflow(
    workflow_api_data: Dict, 
    prompt: str, 
    negative_prompt: str, 
    start_image_filename: str, 
    aspect_ratio: str = "16:9",
    cfg_scale: Optional[float] = None,
    steps: Optional[int] = None,
    sampler: Optional[str] = None,
    scheduler: Optional[str] = None,
    seed: Optional[int] = None
):
    """
    Loads a ComfyUI API-formatted workflow, injects prompts and image for LTX-Video image-to-video.
    LTX-Video image-to-video uses this node structure:
    - Node 3: Positive prompt (CLIPTextEncode)
    - Node 4: Negative prompt (CLIPTextEncode)
    - Node 6: LoadImage (for start image)
    - Node 7: RandomNoise (for seed)
    - Node 8: KSamplerSelect (for sampler)
    - Node 9: BasicScheduler (for scheduler and steps)
    - Node 12: CFGGuider (for cfg_scale) - different node than text-to-video!
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])

    # Node 3 is positive prompt, Node 4 is negative prompt
    if '3' in api_graph and 'inputs' in api_graph['3'] and 'text' in api_graph['3']['inputs']:
        api_graph['3']['inputs']['text'] = prompt
        logging.info(f"Injected positive prompt into LTX i2v node 3: {prompt[:50]}...")
    else:
        logging.warning("Could not find positive prompt node 3 in LTX i2v workflow")

    if '4' in api_graph and 'inputs' in api_graph['4'] and 'text' in api_graph['4']['inputs']:
        api_graph['4']['inputs']['text'] = negative_prompt
        logging.info(f"Injected negative prompt into LTX i2v node 4")
    else:
        logging.warning("Could not find negative prompt node 4 in LTX i2v workflow")

    # Inject start image into LoadImage node (Node 6)
    if '6' in api_graph and api_graph['6'].get("class_type") == "LoadImage":
        api_graph['6']['inputs']['image'] = start_image_filename
        logging.info(f"Injected start image into LTX i2v node 6: {start_image_filename}")
    else:
        logging.warning("Could not find LoadImage node 6 in LTX i2v workflow")

    # Set seed for RandomNoise node (Node 7) - use provided or generate random
    actual_seed = seed if seed is not None else random.randint(0, 2**63 - 1)
    if '7' in api_graph and api_graph['7'].get("class_type") == "RandomNoise":
        api_graph['7']['inputs']['noise_seed'] = actual_seed
        logging.info(f"Set seed in LTX i2v node 7: {actual_seed}")
    else:
        logging.warning("Could not find RandomNoise node 7 to set seed")

    # Inject sampler into KSamplerSelect (Node 8)
    if sampler is not None and '8' in api_graph and api_graph['8'].get("class_type") == "KSamplerSelect":
        api_graph['8']['inputs']['sampler_name'] = sampler
        logging.info(f"Injected sampler into LTX i2v node 8: {sampler}")

    # Inject scheduler and steps into BasicScheduler (Node 9)
    if '9' in api_graph and api_graph['9'].get("class_type") == "BasicScheduler":
        if scheduler is not None:
            api_graph['9']['inputs']['scheduler'] = scheduler
            logging.info(f"Injected scheduler into LTX i2v node 9: {scheduler}")
        if steps is not None:
            api_graph['9']['inputs']['steps'] = steps
            logging.info(f"Injected steps into LTX i2v node 9: {steps}")

    # Inject cfg_scale into CFGGuider (Node 12 for i2v, not 10!)
    if cfg_scale is not None and '12' in api_graph and api_graph['12'].get("class_type") == "CFGGuider":
        api_graph['12']['inputs']['cfg'] = cfg_scale
        logging.info(f"Injected cfg into LTX i2v node 12: {cfg_scale}")

    # Replace date token in filename_prefix for VHS_VideoCombine node
    for node in api_graph.values():
        if node.get("class_type") == "VHS_VideoCombine":
            prefix = node["inputs"].get("filename_prefix", "")
            if "%date:yyyy-MM-dd%" in prefix:
                datestr = datetime.now().strftime("%Y-%m-%d")
                node["inputs"]["filename_prefix"] = prefix.replace("%date:yyyy-MM-dd%", datestr)

    return api_graph

def inject_prompt_into_ltx2_video_workflow(
    workflow_api_data: Dict, 
    prompt: str, 
    negative_prompt: str, 
    aspect_ratio: str = "16:9",
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    camera_movement: Optional[str] = None,
    camera_movement_strength: float = 1.0
):
    """
    Loads a ComfyUI API-formatted workflow, injects prompts for LTX-2.
    Also handles Camera Control LoRA injection if camera_movement is specified.
    
    New workflow structure:
    - Node 3: Positive prompt (CLIPTextEncode)
    - Node 9: BasicScheduler (steps)
    - Node 10: RandomNoise (seed)
    - Node 11: LTXVBaseSampler (width, height, num_frames)
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])

    # Node 3 is the positive prompt node (CLIPTextEncode)
    if '3' in api_graph and 'inputs' in api_graph['3'] and 'text' in api_graph['3']['inputs']:
        api_graph['3']['inputs']['text'] = prompt
        logging.info(f"Injected positive prompt into LTX-2 node 3: {prompt[:50]}...")
    else:
        logging.warning("Could not find prompt node 3 in LTX-2 workflow")

    # Generate seed
    actual_seed = seed if seed is not None else random.randint(0, 2**63 - 1)
    
    # Inject seed into RandomNoise (Node 10)
    if '10' in api_graph and api_graph['10'].get("class_type") == "RandomNoise":
        api_graph['10']['inputs']['noise_seed'] = actual_seed
        logging.info(f"Injected seed into LTX-2 RandomNoise node 10: {actual_seed}")

    # Inject steps into BasicScheduler (Node 9)
    if steps is not None and '9' in api_graph and api_graph['9'].get("class_type") == "BasicScheduler":
        api_graph['9']['inputs']['steps'] = steps
        logging.info(f"Injected steps into LTX-2 BasicScheduler node 9: {steps}")

    # Inject dimensions into LTXVBaseSampler (Node 11)
    if '11' in api_graph and api_graph['11'].get("class_type") == "LTXVBaseSampler":
        width, height = get_dimensions(aspect_ratio)
        api_graph['11']['inputs']['width'] = width
        api_graph['11']['inputs']['height'] = height
        logging.info(f"Injected dimensions into LTX-2 LTXVBaseSampler node 11: {width}x{height}")

    # Inject Camera Control LoRA
    if camera_movement and camera_movement != "none":
        lora_filename = get_camera_lora_filename(camera_movement)
        if lora_filename:
            inject_lora_ltx2(api_graph, lora_filename, strength=camera_movement_strength)

    return api_graph

def inject_prompt_and_image_into_ltx2_video_workflow(
    workflow_api_data: Dict, 
    prompt: str, 
    negative_prompt: str, 
    start_image_filename: str, 
    aspect_ratio: str = "16:9",
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    camera_movement: Optional[str] = None,
    camera_movement_strength: float = 1.0
):
    """
    Loads a ComfyUI API-formatted workflow, injects prompts and image for LTX-2 image-to-video.
    Also handles Camera Control LoRA injection.
    
    New workflow structure:
    - Node 3: Positive prompt (CLIPTextEncode)
    - Node 9: BasicScheduler (steps)
    - Node 10: RandomNoise (seed)
    - Node 11: LTXVBaseSampler (width, height, num_frames)
    - Node 15: LoadImage (start image)
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])

    # Node 3 is the prompt node (CLIPTextEncode)
    if '3' in api_graph and 'inputs' in api_graph['3'] and 'text' in api_graph['3']['inputs']:
        api_graph['3']['inputs']['text'] = prompt
        logging.info(f"Injected positive prompt into LTX-2 i2v node 3: {prompt[:50]}...")
    else:
        logging.warning("Could not find prompt node 3 in LTX-2 i2v workflow")

    # Inject start image into LoadImage node (Node 15)
    if '15' in api_graph and api_graph['15'].get("class_type") == "LoadImage":
        api_graph['15']['inputs']['image'] = start_image_filename
        logging.info(f"Injected start image into LTX-2 i2v node 15: {start_image_filename}")
    else:
        logging.warning("Could not find LoadImage node 15 in LTX-2 i2v workflow")

    # Generate seed
    actual_seed = seed if seed is not None else random.randint(0, 2**63 - 1)
    
    # Inject seed into RandomNoise (Node 10)
    if '10' in api_graph and api_graph['10'].get("class_type") == "RandomNoise":
        api_graph['10']['inputs']['noise_seed'] = actual_seed
        logging.info(f"Injected seed into LTX-2 i2v RandomNoise node 10: {actual_seed}")

    # Inject steps into BasicScheduler (Node 9)
    if steps is not None and '9' in api_graph and api_graph['9'].get("class_type") == "BasicScheduler":
        api_graph['9']['inputs']['steps'] = steps
        logging.info(f"Injected steps into LTX-2 i2v BasicScheduler node 9: {steps}")

    # Inject dimensions into LTXVBaseSampler (Node 11)
    if '11' in api_graph and api_graph['11'].get("class_type") == "LTXVBaseSampler":
        width, height = get_dimensions(aspect_ratio)
        api_graph['11']['inputs']['width'] = width
        api_graph['11']['inputs']['height'] = height
        logging.info(f"Injected dimensions into LTX-2 i2v LTXVBaseSampler node 11: {width}x{height}")

    # Inject Camera Control LoRA
    if camera_movement and camera_movement != "none":
        lora_filename = get_camera_lora_filename(camera_movement)
        if lora_filename:
            inject_lora_ltx2(api_graph, lora_filename, strength=camera_movement_strength)

    return api_graph

def get_camera_lora_filename(movement: str) -> Optional[str]:
    mapping = {
        "dolly-in": "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
        "dolly-out": "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
        "dolly-left": "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
        "dolly-right": "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
        "jib-up": "ltx-2-19b-lora-camera-control-jib-up.safetensors",
        "jib-down": "ltx-2-19b-lora-camera-control-jib-down.safetensors",
        "static": "ltx-2-19b-lora-camera-control-static.safetensors"
    }
    return mapping.get(movement)

def inject_lora_ltx2(api_graph: Dict, lora_name: str, strength: float = 1.0):
    """
    Injects a LoraLoader node into the LTX-2 workflow.
    Intercepts connection between ModelLoader (Node 1) and Sampler (Node 9 or 10).
    Also handles CLIP path from GemmaLoader (Node 2) to Prompt (Node 3).
    """
    lora_id = "100"
    
    # Find Model Loader (Node 1 - CheckpointLoaderSimple)
    # and Gemma Loader (Node 2 - LTXVGemmaCLIPModelLoader)
    # This is specific to our LTX-2 workflows
    
    # Add LoraLoader node
    api_graph[lora_id] = {
        "inputs": {
            "lora_name": lora_name,
            "strength_model": strength,
            "strength_clip": 1.0,
            "model": ["1", 0], # From CheckpointLoaderSimple
            "clip": ["2", 0]   # From GemmaLoader
        },
        "class_type": "LoraLoader",
        "_meta": {"title": "Camera Control LoRA"}
    }
    logging.info(f"Added LoraLoader node {lora_id} for {lora_name}")

    # Re-route Sampler (Node 9 in T2V, Node 10 in I2V) model input
    # Check T2V sampler
    if '9' in api_graph and api_graph['9'].get("class_type") == "LTXVSampler":
        api_graph['9']['inputs']['model'] = [lora_id, 0]
        logging.info(f"Rerouted T2V Sampler model input to LoRA")
    
    # Check I2V sampler
    if '10' in api_graph and api_graph['10'].get("class_type") == "LTXVSampler":
        api_graph['10']['inputs']['model'] = [lora_id, 0]
        logging.info(f"Rerouted I2V Sampler model input to LoRA")

    # Re-route CLIP for Prompt Node (Node 3)
    # Prompt node usually takes CLIP from Node 2. Now it should take from LoraLoader (output 1)
    if '3' in api_graph and 'inputs' in api_graph['3']:
        api_graph['3']['inputs']['clip'] = [lora_id, 1]
        logging.info(f"Rerouted Prompt Node CLIP input to LoRA")

def inject_prompt_into_hunyuan_video_workflow(
    workflow_api_data: Dict, 
    prompt: str, 
    negative_prompt: str, 
    aspect_ratio: str = "16:9",
    cfg_scale: Optional[float] = None,
    steps: Optional[int] = None,
    flow_shift: Optional[float] = None,
    sampler: Optional[str] = None,
    scheduler: Optional[str] = None,
    seed: Optional[int] = None
):
    """
    Loads a ComfyUI API-formatted workflow, injects prompts for HunyuanVideo 1.5.
    HunyuanVideo uses this node structure:
    - Node 44: Positive prompt (CLIPTextEncode)
    - Node 93: Negative prompt (CLIPTextEncode)
    - Node 129: RandomNoise (for seed)
    - Node 130: KSamplerSelect (for sampler)
    - Node 128: BasicScheduler (for scheduler and steps)
    - Node 131: CFGGuider (for cfg_scale)
    - Node 132: ModelSamplingSD3 (for flow_shift)
    - Node 133: EmptyHunyuanLatentVideo (for dimensions)
    - Node 101: CreateVideo (needs fps parameter)
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])

    # Node 44 is positive prompt, Node 93 is negative prompt
    if '44' in api_graph and 'inputs' in api_graph['44'] and 'text' in api_graph['44']['inputs']:
        api_graph['44']['inputs']['text'] = prompt
        logging.info(f"Injected positive prompt into HunyuanVideo node 44: {prompt[:50]}...")
    else:
        logging.warning("Could not find positive prompt node 44 in HunyuanVideo workflow")

    if '93' in api_graph and 'inputs' in api_graph['93'] and 'text' in api_graph['93']['inputs']:
        api_graph['93']['inputs']['text'] = negative_prompt
        logging.info(f"Injected negative prompt into HunyuanVideo node 93")
    else:
        logging.warning("Could not find negative prompt node 93 in HunyuanVideo workflow")

    # Inject dimensions into EmptyHunyuanLatentVideo (Node 133)
    if '133' in api_graph and api_graph['133'].get("class_type") == "EmptyHunyuanLatentVideo":
        width, height = get_dimensions(aspect_ratio)
        api_graph['133']['inputs']['width'] = width
        api_graph['133']['inputs']['height'] = height
        logging.info(f"Injected dimensions into HunyuanVideo node 133: {width}x{height}")

    # Set seed for RandomNoise node (Node 129) - use provided or generate random
    actual_seed = seed if seed is not None else random.randint(0, 2**63 - 1)
    if '129' in api_graph and api_graph['129'].get("class_type") == "RandomNoise":
        api_graph['129']['inputs']['noise_seed'] = actual_seed
        logging.info(f"Set seed in HunyuanVideo node 129: {actual_seed}")
    else:
        logging.warning("Could not find RandomNoise node 129 to set seed")

    # Inject sampler into KSamplerSelect (Node 130)
    if sampler is not None and '130' in api_graph and api_graph['130'].get("class_type") == "KSamplerSelect":
        api_graph['130']['inputs']['sampler_name'] = sampler
        logging.info(f"Injected sampler into HunyuanVideo node 130: {sampler}")

    # Inject scheduler and steps into BasicScheduler (Node 128)
    if '128' in api_graph and api_graph['128'].get("class_type") == "BasicScheduler":
        if scheduler is not None:
            api_graph['128']['inputs']['scheduler'] = scheduler
            logging.info(f"Injected scheduler into HunyuanVideo node 128: {scheduler}")
        if steps is not None:
            api_graph['128']['inputs']['steps'] = steps
            logging.info(f"Injected steps into HunyuanVideo node 128: {steps}")

    # Inject cfg_scale into CFGGuider (Node 131)
    if cfg_scale is not None and '131' in api_graph and api_graph['131'].get("class_type") == "CFGGuider":
        api_graph['131']['inputs']['cfg'] = cfg_scale
        logging.info(f"Injected cfg into HunyuanVideo node 131: {cfg_scale}")

    # Inject flow_shift into ModelSamplingSD3 (Node 132)
    if flow_shift is not None and '132' in api_graph and api_graph['132'].get("class_type") == "ModelSamplingSD3":
        api_graph['132']['inputs']['shift'] = flow_shift
        logging.info(f"Injected flow_shift into HunyuanVideo node 132: {flow_shift}")

    # Fix CreateVideo node (Node 101) - add missing fps parameter
    if '101' in api_graph and api_graph['101'].get("class_type") == "CreateVideo":
        # Ensure fps is set (defaults to frame_rate if not present)
        if 'fps' not in api_graph['101']['inputs']:
            frame_rate = api_graph['101']['inputs'].get('frame_rate', 24)
            api_graph['101']['inputs']['fps'] = frame_rate
            logging.info(f"Added missing fps parameter to HunyuanVideo CreateVideo node 101: {frame_rate}")

    # Update SaveVideo filename_prefix if present
    for node_id, node in api_graph.items():
        if node.get("class_type") == "SaveVideo":
            prefix = node["inputs"].get("filename_prefix", "")
            if "%date:yyyy-MM-dd%" in prefix:
                datestr = datetime.now().strftime("%Y-%m-%d")
                node["inputs"]["filename_prefix"] = prefix.replace("%date:yyyy-MM-dd%", datestr)
                logging.info(f"Updated filename_prefix in SaveVideo node {node_id}")
            break

    return api_graph

def inject_prompt_and_image_into_hunyuan_video_workflow(
    workflow_api_data: Dict, 
    prompt: str, 
    negative_prompt: str, 
    start_image_filename: str,
    aspect_ratio: str = "16:9",
    cfg_scale: Optional[float] = None,
    steps: Optional[int] = None,
    flow_shift: Optional[float] = None,
    sampler: Optional[str] = None,
    scheduler: Optional[str] = None,
    seed: Optional[int] = None
):
    """
    Loads a ComfyUI API-formatted workflow, injects prompts and image for HunyuanVideo 1.5 image-to-video.
    HunyuanVideo i2v uses this node structure:
    - Node 1: LoadImage (for start image)
    - Node 44: Positive prompt (CLIPTextEncode)
    - Node 93: Negative prompt (CLIPTextEncode)
    - Node 78: HunyuanImageToVideo (handles dimensions and conditioning)
    - Node 129: RandomNoise (for seed)
    - Node 130: KSamplerSelect (for sampler)
    - Node 128: BasicScheduler (for scheduler and steps)
    - Node 131: CFGGuider (for cfg_scale)
    - Node 132: ModelSamplingSD3 (for flow_shift)
    - Node 101: CreateVideo (needs fps parameter)
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])

    # Node 44 is positive prompt, Node 93 is negative prompt
    if '44' in api_graph and 'inputs' in api_graph['44'] and 'text' in api_graph['44']['inputs']:
        api_graph['44']['inputs']['text'] = prompt
        logging.info(f"Injected positive prompt into HunyuanVideo i2v node 44: {prompt[:50]}...")
    else:
        logging.warning("Could not find positive prompt node 44 in HunyuanVideo i2v workflow")

    if '93' in api_graph and 'inputs' in api_graph['93'] and 'text' in api_graph['93']['inputs']:
        api_graph['93']['inputs']['text'] = negative_prompt
        logging.info(f"Injected negative prompt into HunyuanVideo i2v node 93")
    else:
        logging.warning("Could not find negative prompt node 93 in HunyuanVideo i2v workflow")

    # Inject start image into LoadImage node (Node 1)
    if '1' in api_graph and api_graph['1'].get("class_type") == "LoadImage":
        api_graph['1']['inputs']['image'] = start_image_filename
        logging.info(f"Injected start image into HunyuanVideo i2v node 1: {start_image_filename}")
    else:
        logging.warning("Could not find LoadImage node 1 in HunyuanVideo i2v workflow")

    # Inject dimensions into HunyuanImageToVideo (Node 78)
    if '78' in api_graph and api_graph['78'].get("class_type") == "HunyuanImageToVideo":
        width, height = get_dimensions(aspect_ratio)
        api_graph['78']['inputs']['width'] = width
        api_graph['78']['inputs']['height'] = height
        logging.info(f"Injected dimensions into HunyuanVideo i2v node 78: {width}x{height}")

    # Set seed for RandomNoise node (Node 129) - use provided or generate random
    actual_seed = seed if seed is not None else random.randint(0, 2**63 - 1)
    if '129' in api_graph and api_graph['129'].get("class_type") == "RandomNoise":
        api_graph['129']['inputs']['noise_seed'] = actual_seed
        logging.info(f"Set seed in HunyuanVideo i2v node 129: {actual_seed}")
    else:
        logging.warning("Could not find RandomNoise node 129 to set seed")

    # Inject sampler into KSamplerSelect (Node 130)
    if sampler is not None and '130' in api_graph and api_graph['130'].get("class_type") == "KSamplerSelect":
        api_graph['130']['inputs']['sampler_name'] = sampler
        logging.info(f"Injected sampler into HunyuanVideo i2v node 130: {sampler}")

    # Inject scheduler and steps into BasicScheduler (Node 128)
    if '128' in api_graph and api_graph['128'].get("class_type") == "BasicScheduler":
        if scheduler is not None:
            api_graph['128']['inputs']['scheduler'] = scheduler
            logging.info(f"Injected scheduler into HunyuanVideo i2v node 128: {scheduler}")
        if steps is not None:
            api_graph['128']['inputs']['steps'] = steps
            logging.info(f"Injected steps into HunyuanVideo i2v node 128: {steps}")

    # Inject cfg_scale into CFGGuider (Node 131)
    if cfg_scale is not None and '131' in api_graph and api_graph['131'].get("class_type") == "CFGGuider":
        api_graph['131']['inputs']['cfg'] = cfg_scale
        logging.info(f"Injected cfg into HunyuanVideo i2v node 131: {cfg_scale}")

    # Inject flow_shift into ModelSamplingSD3 (Node 132)
    if flow_shift is not None and '132' in api_graph and api_graph['132'].get("class_type") == "ModelSamplingSD3":
        api_graph['132']['inputs']['shift'] = flow_shift
        logging.info(f"Injected flow_shift into HunyuanVideo i2v node 132: {flow_shift}")

    # Fix CreateVideo node (Node 101) - add missing fps parameter
    if '101' in api_graph and api_graph['101'].get("class_type") == "CreateVideo":
        # Ensure fps is set (defaults to frame_rate if not present)
        if 'fps' not in api_graph['101']['inputs']:
            frame_rate = api_graph['101']['inputs'].get('frame_rate', 24)
            api_graph['101']['inputs']['fps'] = frame_rate
            logging.info(f"Added missing fps parameter to HunyuanVideo i2v CreateVideo node 101: {frame_rate}")

    # Update SaveVideo filename_prefix if present
    for node_id, node in api_graph.items():
        if node.get("class_type") == "SaveVideo":
            prefix = node["inputs"].get("filename_prefix", "")
            if "%date:yyyy-MM-dd%" in prefix:
                datestr = datetime.now().strftime("%Y-%m-%d")
                node["inputs"]["filename_prefix"] = prefix.replace("%date:yyyy-MM-dd%", datestr)
                logging.info(f"Updated filename_prefix in SaveVideo node {node_id}")
            break

    return api_graph

def inject_prompt_into_qwen_workflow(workflow_api_data: Dict, prompt: str, negative_prompt: str, aspect_ratio: str = "1:1"):
    """
    Loads a ComfyUI API-formatted workflow, injects prompts for Qwen.
    Also randomizes seed for varied outputs.
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])

    # In the qwen-text-to-image.api.json:
    # Node 6 is positive prompt
    # Node 7 is negative prompt
    if '6' in api_graph and 'inputs' in api_graph['6'] and 'text' in api_graph['6']['inputs']:
        api_graph['6']['inputs']['text'] = prompt
    else:
        logging.warning("Could not find positive prompt node 6 in Qwen workflow")

    if '7' in api_graph and 'inputs' in api_graph['7'] and 'text' in api_graph['7']['inputs']:
        api_graph['7']['inputs']['text'] = negative_prompt
    else:
        logging.warning("Could not find negative prompt node 7 in Qwen workflow")

    # Inject dimensions into EmptySD3LatentImage (Node 58)
    if '58' in api_graph and 'inputs' in api_graph['58']:
        width, height = get_dimensions(aspect_ratio, default_width=1024, default_height=1024)
        api_graph['58']['inputs']['width'] = width
        api_graph['58']['inputs']['height'] = height
    else:
        # Fallback: search for EmptySD3LatentImage by class type
        for node in api_graph.values():
            if node.get("class_type") == "EmptySD3LatentImage":
                width, height = get_dimensions(aspect_ratio, default_width=1024, default_height=1024)
                node["inputs"]["width"] = width
                node["inputs"]["height"] = height
                break

    # Randomize seed for node 3 (KSampler)
    if '3' in api_graph and 'inputs' in api_graph['3']:
        new_seed = random.randint(0, 2**63 - 1)
        api_graph['3']['inputs']['seed'] = new_seed

    return api_graph

def inject_prompt_into_flux_workflow(workflow_api_data: Dict, prompt: str, negative_prompt: str):
    """
    Loads a ComfyUI API-formatted workflow, injects prompts for FLUX.
    Also randomizes seed for varied outputs.
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])

    # In the new flux-text-to-image.api.json:
    # Node 7 is positive prompt
    # Node 6 is negative prompt
    if '7' in api_graph and 'inputs' in api_graph['7'] and 'text' in api_graph['7']['inputs']:
        api_graph['7']['inputs']['text'] = prompt
    else:
        logging.warning("Could not find positive prompt node 7 in FLUX workflow")

    if '6' in api_graph and 'inputs' in api_graph['6'] and 'text' in api_graph['6']['inputs']:
        api_graph['6']['inputs']['text'] = negative_prompt
    else:
        logging.warning("Could not find negative prompt node 6 in FLUX workflow")

    # Randomize seed for node 3 (KSampler)
    if '3' in api_graph and 'inputs' in api_graph['3']:
        new_seed = random.randint(0, 2**63 - 1)
        api_graph['3']['inputs']['seed'] = new_seed

    return api_graph

def inject_video_and_prompt_into_foley_workflow(workflow_api_data: Dict, video_filename: str, prompt: str, negative_prompt: str):
    """
    Loads the Foley ComfyUI API-formatted workflow, injects video filename and prompts.
    Also randomizes seed for varied outputs.
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])

    # Inject video filename and prompts
    for node in api_graph.values():
        if node["class_type"] == "HunyuanVideoFoleyGeneratorAdvanced":
            node["inputs"]["video"] = video_filename
            node["inputs"]["text_prompt"] = prompt
            node["inputs"]["negative_prompt"] = negative_prompt
            # Randomize seed

            new_seed = random.randint(0, 2**31 - 1)
            node["inputs"]["seed"] = new_seed

    return api_graph

def inject_prompt_into_vibevoice_workflow(
    workflow_api_data: Dict, 
    prompt: str,
    cfg_scale: float = 3.5,
    diffusion_steps: int = 10,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: Union[int, None] = None,
    voice_id: str = "Alice"
):
    """
    Loads the VibeVoice ComfyUI API-formatted workflow, injects the prompt
    and ensures all required inputs for VibeVoiceSingleSpeakerNode are present.
    Also randomizes seed for varied outputs.
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])

    # Generate random seed if not provided
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    # Find the VibeVoice node and explicitly set its inputs
    for node in api_graph.values():
        if node["class_type"] == "VibeVoiceSingleSpeakerNode":
            # Overwrite the entire 'inputs' dictionary to ensure correctness
            node["inputs"] = {
                "text": prompt,
                "model": "VibeVoice-1.5B",
                "attention_type": "auto",
                "free_memory_after_generate": True,
                "diffusion_steps": diffusion_steps,
                "seed": seed,
                "cfg_scale": cfg_scale,
                "use_sampling": False,
                "temperature": temperature,
                "top_p": top_p,
                "quantize_llm": "full precision",
                "voice": voice_id
            }
            break  # Assuming only one such node

    return api_graph

def inject_prompt_into_diffrhythm_workflow(
    workflow_api_data: Dict, 
    lyrics_or_edit_lyrics: str,
    style_prompt: str
):
    """
    Loads the DiffRhythm ComfyUI API-formatted workflow.
    
    Args:
        lyrics_or_edit_lyrics: Timestamped lyrics for vocals, or "" for instrumental
        style_prompt: Musical style description (genre, tempo, instruments, vocal type)
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])
    new_seed = random.randint(0, 2**31 - 1)

    for node in api_graph.values():
        if node["class_type"] == "DiffRhythmRun":
            node["inputs"]["lyrics_or_edit_lyrics"] = lyrics_or_edit_lyrics
            node["inputs"]["style_prompt"] = style_prompt
            node["inputs"]["edit"] = False
            node["inputs"]["seed"] = new_seed
            logging.info(f"DiffRhythm configured - Style: {style_prompt}, Has lyrics: {bool(lyrics_or_edit_lyrics)}, Seed: {new_seed}")

    return api_graph

def inject_script_and_clones_into_vibevoice_workflow(
    workflow_api_data: Dict, 
    script: str, 
    clone_paths: list[str],
    cfg_scale: float = 3.5,
    diffusion_steps: int = 10,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: Union[int, None] = None
):
    """
    Loads the VibeVoice ComfyUI API-formatted workflow, injects the script and clone paths.
    Also randomizes seed for varied outputs.
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])

    # Generate random seed if not provided
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    # Inject script and clone paths
    for node in api_graph.values():
        if node["class_type"] == "VibeVoiceMultipleSpeakersNode":
            node["inputs"]["text"] = script
            node["inputs"]["seed"] = seed
            node["inputs"]["cfg_scale"] = cfg_scale
            node["inputs"]["diffusion_steps"] = diffusion_steps
            node["inputs"]["temperature"] = temperature
            node["inputs"]["top_p"] = top_p
            # Required params defaults
            if "model" not in node["inputs"]: node["inputs"]["model"] = "VibeVoice-1.5B"
            if "attention_type" not in node["inputs"]: node["inputs"]["attention_type"] = "auto"
            if "quantize_llm" not in node["inputs"]: node["inputs"]["quantize_llm"] = "full precision"
            if "free_memory_after_generate" not in node["inputs"]: node["inputs"]["free_memory_after_generate"] = True
            if "use_sampling" not in node["inputs"]: node["inputs"]["use_sampling"] = False
            
            for i, clone_path in enumerate(clone_paths):
                node["inputs"][f"speaker{i+1}_voice"] = clone_path

    return api_graph

def inject_prompt_into_chatterbox_workflow(
    workflow_api_data: Dict, 
    prompt: str,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
    max_new_tokens: int = 2048,
    seed: Union[int, None] = None,
):
    """
    Loads the Chatterbox ComfyUI API-formatted workflow, injects the prompt
    and generation parameters.
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])

    # Generate random seed if not provided
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    # Find the ChatterboxTTS node and set its inputs
    for node in api_graph.values():
        if node["class_type"] == "ChatterboxTTS":
            node["inputs"]["text"] = prompt
            node["inputs"]["exaggeration"] = exaggeration
            node["inputs"]["cfg_weight"] = cfg_weight
            node["inputs"]["temperature"] = temperature
            node["inputs"]["max_new_tokens"] = max_new_tokens
            node["inputs"]["seed"] = seed
            break

    logging.info(f"Chatterbox TTS configured - Prompt: '{prompt[:50]}...', Exaggeration: {exaggeration}, CFG Weight: {cfg_weight}")
    
    return api_graph

def inject_prompt_and_clone_into_chatterbox_workflow(
    workflow_api_data: Dict, 
    prompt: str,
    audio_prompt_filename: str,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
    max_new_tokens: int = 2048,
    seed: Union[int, None] = None,
):
    """
    Loads the Chatterbox voice clone ComfyUI API-formatted workflow, 
    injects the prompt and audio prompt for voice cloning.
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])

    # Generate random seed if not provided
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    # Find the LoadAudio node and set the audio filename
    for node in api_graph.values():
        if node["class_type"] == "LoadAudio":
            node["inputs"]["audio"] = audio_prompt_filename
            break

    # Find the ChatterboxTTS node and set its inputs
    for node in api_graph.values():
        if node["class_type"] == "ChatterboxTTS":
            node["inputs"]["text"] = prompt
            node["inputs"]["exaggeration"] = exaggeration
            node["inputs"]["cfg_weight"] = cfg_weight
            node["inputs"]["temperature"] = temperature
            node["inputs"]["max_new_tokens"] = max_new_tokens
            node["inputs"]["seed"] = seed
            break

    logging.info(f"Chatterbox Voice Clone configured - Prompt: '{prompt[:50]}...', Audio: {audio_prompt_filename}")
    
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

    # Check for audio files in output directory (for workflows that save directly to disk)
    audio_extensions = ['.wav', '.flac', '.mp3', '.ogg', '.m4a']
    try:
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                if any(filename.lower().endswith(ext) for ext in audio_extensions):
                    file_path = os.path.join(output_dir, filename)
                    logging.info(f"Audio file found: {file_path}. Attempting upload.")
                    storage_path = upload_output_func(file_path, job_id)
                    if storage_path:
                        return storage_path
    except Exception as e:
        logging.warning(f"Error checking for audio files in output directory: {e}")

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

def inject_video_into_upscaler_workflow(
    workflow_api_data: Dict, 
    video_filename: str,
    upscale_model: str = "Stream-DiffVSR",
    frame_rate: int = 30,
    target_width: Union[int, None] = None,
    target_height: Union[int, None] = None,
    scale_by: Union[float, None] = None
):
    """
    Injects video filename and upscale settings into the upscaler workflow.
    
    Args:
        workflow_api_data: The workflow JSON structure
        video_filename: Name of the video file in input directory
        upscale_model: Model name to use (default: Stream-DiffVSR)
        frame_rate: Output video frame rate
        target_width: The final width of the video (optional if scale_by is used)
        target_height: The final height of the video (optional if scale_by is used)
        scale_by: Factor to scale the image by (optional, overrides width/height)
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])

    # Node 1: VHS_LoadVideo - Same as before
    if '1' in api_graph and api_graph['1']['class_type'] == 'VHS_LoadVideo':
        api_graph['1']['inputs']['video'] = video_filename
        api_graph['1']['inputs']['force_size'] = "Disabled"
        api_graph['1']['inputs']['custom_width'] = 0
        api_graph['1']['inputs']['custom_height'] = 0
        logging.info(f"Injected video filename: {video_filename}")
    
    # Node 2: StreamDiffVSR_Node - inject steps if provided or default?
    # The current 'upscale_model' param might be repurposed or ignored if we only have one model.
    # But let's check for Steps param injection if we want to support it.
    if '2' in api_graph and api_graph['2']['class_type'] == 'StreamDiffVSR_Node':
        # If upscale_model is passed as an int-like string, maybe treat it as steps?
        # Or we can add a 'steps' argument to this function. For now, strict compatibility.
        # Let's assume defaults are fine or we can inject if needed.
        pass

    # Node 6: ImageScale - REMOVED in new workflow (Stream-DiffVSR doesn't use it, it upscales 4x by default?)
    # Wait, Stream-DiffVSR is usually 4x.
    # The new workflow JSON I wrote DOES NOT have node 6.
    # So we can remove the Node 6 injection logic here or wrap it in a check.

    # Node 4: VHS_VideoCombine - inject frame rate and filename
    if '4' in api_graph and api_graph['4']['class_type'] == 'VHS_VideoCombine':
        api_graph['4']['inputs']['frame_rate'] = frame_rate
        datestr = datetime.now().strftime("%Y-%m-%d")
        prefix = api_graph['4']['inputs'].get('filename_prefix', 'upscaled_video')
        if '%date:yyyy-MM-dd%' in prefix:
            prefix = prefix.replace('%date:yyyy-MM-dd%', datestr)
        else:
            prefix = f"{prefix}_{datestr}"
        api_graph['4']['inputs']['filename_prefix'] = prefix
        logging.info(f"Configured video output: {prefix} at {frame_rate}fps")

    return api_graph

def inject_prompt_into_stable_audio_workflow(workflow_data, prompt, duration_seconds, seed=None):
    """
    Inject prompt and duration into Stable Audio workflow using comfyui-sound-lab's StableAudio_ node.
    
    Args:
        workflow_data: The workflow JSON structure (dict with 'prompt' key containing nodes)
        prompt: Text prompt for audio generation (e.g., "a dog barking")
        duration_seconds: Duration in seconds (default: 5, max depends on model)
        seed: Random seed (optional, defaults to random if None)
    
    Returns:
        Modified workflow data (dict)
    
    Example:
        workflow = inject_prompt_into_stable_audio_workflow(
            workflow_data=loaded_json,
            prompt="thunderstorm with heavy rain",
            duration_seconds=10,
            seed=42
        )
    """
    
    # Deep copy to avoid modifying original
    api_graph = copy.deepcopy(workflow_data.get('prompt', workflow_data))
    
    # Find the StableAudio_ node (usually node "1")
    stable_audio_node_id = None
    for node_id, node in api_graph.items():
        if node.get('class_type') == 'StableAudio_':
            stable_audio_node_id = node_id
            break
    
    if not stable_audio_node_id:
        raise ValueError("StableAudio_ node not found in workflow. Ensure the workflow contains a StableAudio_ node.")
    
    # Update the StableAudio_ node inputs
    api_graph[stable_audio_node_id]['inputs']['prompt'] = prompt
    api_graph[stable_audio_node_id]['inputs']['seconds'] = duration_seconds
    
    # Set seed - use provided seed or generate random one
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    api_graph[stable_audio_node_id]['inputs']['seed'] = seed
    
    # Ensure device is set to 'auto' (accepts 'auto' or 'cpu' only)
    api_graph[stable_audio_node_id]['inputs']['device'] = 'auto'

    logging.info(f"Stable Audio configured - Prompt: '{prompt}', Duration: {duration_seconds}s, Seed: {seed}")
    
    # Return just the node graph (consistent with other inject functions)
    return api_graph


def get_zimage_dimensions(aspect_ratio: str) -> tuple[int, int]:
    """
    Returns (width, height) for Z-Image based on aspect ratio.
    Z-Image works best at 1024px base resolution.
    All dimensions are divisible by 16.
    """
    if aspect_ratio == "16:9":
        return 1024, 576
    elif aspect_ratio == "9:16":
        return 576, 1024
    elif aspect_ratio == "1:1":
        return 1024, 1024
    elif aspect_ratio == "4:3":
        return 1024, 768
    elif aspect_ratio == "3:4":
        return 768, 1024
    elif aspect_ratio == "21:9":
        return 1024, 448
    else:
        return 1024, 1024


def inject_prompt_into_zimage_workflow(
    workflow_api_data: Dict, 
    prompt: str, 
    aspect_ratio: str = "1:1",
    advanced_settings: Optional[Dict] = None,
    seed: Optional[int] = None
):
    """
    Loads a Z-Image ComfyUI API-formatted workflow, injects prompt and dimensions.
    Also randomizes seed for varied outputs and applies advanced settings.
    
    Args:
        workflow_api_data: The workflow JSON structure with 'prompt' key
        prompt: Text prompt for image generation
        aspect_ratio: Aspect ratio string (e.g., "1:1", "16:9", "9:16")
        advanced_settings: Optional dict with keys: steps, cfg, sampler_name, scheduler, shift
        seed: Optional seed for reproduction
    
    Returns:
        Modified workflow graph (dict)
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])
    
    # Node 45 is CLIPTextEncode (positive prompt)
    if '45' in api_graph and 'inputs' in api_graph['45'] and 'text' in api_graph['45']['inputs']:
        api_graph['45']['inputs']['text'] = prompt
    else:
        # Fallback: search for CLIPTextEncode by class type
        for node in api_graph.values():
            if node.get("class_type") == "CLIPTextEncode":
                node["inputs"]["text"] = prompt
                break
        else:
            logging.warning("Could not find CLIPTextEncode node in Z-Image workflow")
    
    # Inject dimensions into EmptySD3LatentImage (Node 41)
    width, height = get_zimage_dimensions(aspect_ratio)
    if '41' in api_graph and 'inputs' in api_graph['41']:
        api_graph['41']['inputs']['width'] = width
        api_graph['41']['inputs']['height'] = height
    else:
        # Fallback: search by class type
        for node in api_graph.values():
            if node.get("class_type") == "EmptySD3LatentImage":
                node["inputs"]["width"] = width
                node["inputs"]["height"] = height
                break
    
    # Randomize seed for KSampler (Node 44)
    new_seed = seed if seed is not None else random.randint(0, 2**63 - 1)
    if '44' in api_graph and 'inputs' in api_graph['44']:
        api_graph['44']['inputs']['seed'] = new_seed
    else:
        for node in api_graph.values():
            if node.get("class_type") == "KSampler":
                node["inputs"]["seed"] = new_seed
                break
    
    # Apply advanced settings to KSampler (Node 44)
    if advanced_settings:
        ksampler_node = api_graph.get('44', {}).get('inputs')
        if not ksampler_node:
            # Fallback search
            for node in api_graph.values():
                if node.get("class_type") == "KSampler":
                    ksampler_node = node.get("inputs", {})
                    break
        
        if ksampler_node:
            if 'steps' in advanced_settings:
                ksampler_node['steps'] = advanced_settings['steps']
            if 'cfg' in advanced_settings:
                ksampler_node['cfg'] = advanced_settings['cfg']
            if 'sampler_name' in advanced_settings:
                ksampler_node['sampler_name'] = advanced_settings['sampler_name']
            if 'scheduler' in advanced_settings:
                ksampler_node['scheduler'] = advanced_settings['scheduler']
        
        # Apply shift to ModelSamplingAuraFlow (Node 47)
        if 'shift' in advanced_settings:
            if '47' in api_graph and 'inputs' in api_graph['47']:
                api_graph['47']['inputs']['shift'] = advanced_settings['shift']
            else:
                for node in api_graph.values():
                    if node.get("class_type") == "ModelSamplingAuraFlow":
                        node["inputs"]["shift"] = advanced_settings['shift']
                        break
        
        logging.info(f"Advanced settings applied: {advanced_settings}")
    
    logging.info(f"Z-Image configured - Prompt: '{prompt[:50]}...', Size: {width}x{height}, Seed: {new_seed}")
    
    return api_graph


def inject_image_into_zimage_controlnet_workflow(
    workflow_api_data: Dict, 
    prompt: str, 
    image_filename: str,
    control_mode: str = "pose",
    strength: float = 1.0,
    aspect_ratio: str = "1:1",
    seed: Optional[int] = None
):
    """
    Loads a Z-Image ControlNet ComfyUI API-formatted workflow, injects prompt, image, and control settings.
    
    Args:
        workflow_api_data: The workflow JSON structure with 'prompt' key
        prompt: Text prompt for image generation
        image_filename: Filename of the reference image in input directory
        control_mode: ControlNet mode - 'pose', 'depth', or 'canny'
        strength: ControlNet strength (0.0 to 1.0)
        aspect_ratio: Aspect ratio string for output dimensions
    
    Returns:
        Modified workflow graph (dict)
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])
    
    # Node 45 is CLIPTextEncode (positive prompt)
    if '45' in api_graph and 'inputs' in api_graph['45'] and 'text' in api_graph['45']['inputs']:
        api_graph['45']['inputs']['text'] = prompt
    else:
        for node in api_graph.values():
            if node.get("class_type") == "CLIPTextEncode":
                node["inputs"]["text"] = prompt
                break
        else:
            logging.warning("Could not find CLIPTextEncode node in Z-Image ControlNet workflow")
    
    # Node 58 is LoadImage - inject reference image filename
    if '58' in api_graph and 'inputs' in api_graph['58']:
        api_graph['58']['inputs']['image'] = image_filename
    else:
        for node in api_graph.values():
            if node.get("class_type") == "LoadImage":
                node["inputs"]["image"] = image_filename
                break
        else:
            logging.warning("Could not find LoadImage node in Z-Image ControlNet workflow")
    
    # Node 60 is ZImageFunControlnet - inject control mode and strength
    if '60' in api_graph and 'inputs' in api_graph['60']:
        api_graph['60']['inputs']['control_mode'] = control_mode
        api_graph['60']['inputs']['strength'] = strength
    else:
        for node in api_graph.values():
            if node.get("class_type") == "ZImageFunControlnet":
                node["inputs"]["control_mode"] = control_mode
                node["inputs"]["strength"] = strength
                break
        else:
            logging.warning("Could not find ZImageFunControlnet node in Z-Image ControlNet workflow")
    
    # Randomize seed for KSampler (Node 44)
    new_seed = seed if seed is not None else random.randint(0, 2**63 - 1)
    if '44' in api_graph and 'inputs' in api_graph['44']:
        api_graph['44']['inputs']['seed'] = new_seed
    else:
        for node in api_graph.values():
            if node.get("class_type") == "KSampler":
                node["inputs"]["seed"] = new_seed
                break
    
    logging.info(f"Z-Image ControlNet configured - Mode: {control_mode}, Strength: {strength}, Image: {image_filename}, Seed: {new_seed}")
    
    return api_graph


def inject_image_into_zimage_inpaint_workflow(
    workflow_api_data: Dict, 
    prompt: str, 
    image_filename: str,
    mask_filename: str,
    denoise_strength: float = 0.8,
    seed: Optional[int] = None
):
    """
    Loads a Z-Image Inpaint ComfyUI API-formatted workflow, injects prompt, image, and mask.
    
    Args:
        workflow_api_data: The workflow JSON structure with 'prompt' key
        prompt: Text prompt describing what to generate in masked area
        image_filename: Filename of the source image in input directory
        mask_filename: Filename of the mask image (white = edit, black = keep)
        denoise_strength: How much to change the masked area (0.0 to 1.0)
    
    Returns:
        Modified workflow graph (dict)
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])
    
    # Node 45 is CLIPTextEncode (positive prompt)
    if '45' in api_graph and 'inputs' in api_graph['45'] and 'text' in api_graph['45']['inputs']:
        api_graph['45']['inputs']['text'] = prompt
    else:
        for node in api_graph.values():
            if node.get("class_type") == "CLIPTextEncode":
                node["inputs"]["text"] = prompt
                break
        else:
            logging.warning("Could not find CLIPTextEncode node in Z-Image inpaint workflow")
    
    # Node 58 is LoadImage for source image
    if '58' in api_graph and 'inputs' in api_graph['58']:
        api_graph['58']['inputs']['image'] = image_filename
    else:
        load_image_nodes = [n for n in api_graph.values() if n.get("class_type") == "LoadImage"]
        if load_image_nodes:
            load_image_nodes[0]["inputs"]["image"] = image_filename
        else:
            logging.warning("Could not find LoadImage node for source image in Z-Image inpaint workflow")
    
    # Node 59 is LoadImage for mask
    if '59' in api_graph and 'inputs' in api_graph['59']:
        api_graph['59']['inputs']['image'] = mask_filename
    else:
        load_image_nodes = [n for n in api_graph.values() if n.get("class_type") == "LoadImage"]
        if len(load_image_nodes) > 1:
            load_image_nodes[1]["inputs"]["image"] = mask_filename
        else:
            logging.warning("Could not find LoadImage node for mask in Z-Image inpaint workflow")
    
    # Randomize seed for KSampler (Node 44) and set denoise strength
    new_seed = seed if seed is not None else random.randint(0, 2**63 - 1)
    if '44' in api_graph and 'inputs' in api_graph['44']:
        api_graph['44']['inputs']['seed'] = new_seed
        api_graph['44']['inputs']['denoise'] = denoise_strength
    else:
        for node in api_graph.values():
            if node.get("class_type") == "KSampler":
                node["inputs"]["seed"] = new_seed
                node["inputs"]["denoise"] = denoise_strength
                break
    
    logging.info(f"Z-Image Inpaint configured - Image: {image_filename}, Mask: {mask_filename}, Denoise: {denoise_strength}, Seed: {new_seed}")
    
    return api_graph


def inject_prompt_into_yume_workflow(
    workflow_api_data: Dict, 
    prompt: str,
    negative_prompt: str = "low quality, distorted, bad animation, blurry, watermark",
    frame_rate: int = 24,
    steps: int = 30,
    cfg: float = 7.0,
    seed: Optional[int] = None,
    width: int = 1280,
    height: int = 720,
    num_frames: int = 49
):
    """
    Loads a YUME 1.5 ComfyUI API-formatted workflow, injects parameters for text-to-video generation.
    
    Args:
        workflow_api_data: The workflow JSON structure (raw, not wrapped in 'prompt' key since YUME uses flat structure)
        prompt: Text prompt for world generation
        negative_prompt: Negative prompt to avoid unwanted elements (default: quality-focused)
        frame_rate: Output video frame rate (default 24)
        steps: Number of sampling steps (default 30)
        cfg: Classifier-free guidance scale (default 7.0)
        seed: Random seed for reproducibility (None = random)
        width: Output video width (default 1280)
        height: Output video height (default 720)
        num_frames: Number of frames to generate (default 49)
    
    Returns:
        Modified workflow graph (dict)
    """
    # YUME workflows are flat (no 'prompt' wrapper), so handle both cases
    if "prompt" in workflow_api_data:
        api_graph = copy.deepcopy(workflow_api_data["prompt"])
    else:
        api_graph = copy.deepcopy(workflow_api_data)
    
    # Generate random seed if not provided
    if seed is None or seed == 0:
        seed = random.randint(0, 2**63 - 1)
    
    # Find YUME_Node and inject parameters
    for node_id, node in api_graph.items():
        if node.get("class_type") == "YUME_Node":
            node["inputs"]["prompt"] = prompt
            node["inputs"]["n_prompt"] = negative_prompt
            node["inputs"]["steps"] = steps
            node["inputs"]["cfg"] = cfg
            node["inputs"]["seed"] = seed
            node["inputs"]["width"] = width
            node["inputs"]["height"] = height
            node["inputs"]["num_frames"] = num_frames
            logging.info(f"YUME T2V configured - Prompt: '{prompt[:50]}...', Steps: {steps}, CFG: {cfg}, Seed: {seed}")
            break
    else:
        logging.warning("Could not find YUME_Node in workflow")
    
    # Update VHS_VideoCombine frame rate and filename_prefix
    for node_id, node in api_graph.items():
        if node.get("class_type") == "VHS_VideoCombine":
            node["inputs"]["frame_rate"] = frame_rate
            
            # Replace date token in filename_prefix
            prefix = node["inputs"].get("filename_prefix", "")
            if "%date:yyyy-MM-dd%" in prefix:
                datestr = datetime.now().strftime("%Y-%m-%d")
                node["inputs"]["filename_prefix"] = prefix.replace("%date:yyyy-MM-dd%", datestr)
            break
    
    return api_graph


def inject_image_into_yume_workflow(
    workflow_api_data: Dict, 
    image_path: str,
    prompt: str,
    negative_prompt: str = "low quality, distorted, bad animation, blurry, watermark",
    frame_rate: int = 24,
    steps: int = 30,
    cfg: float = 7.0,
    seed: Optional[int] = None,
    width: int = 1280,
    height: int = 720,
    num_frames: int = 49
):
    """
    Loads a YUME 1.5 ComfyUI API-formatted workflow, injects parameters for image-to-video generation.
    
    Args:
        workflow_api_data: The workflow JSON structure
        image_path: Path/filename of the start image in input directory
        prompt: Text prompt for world generation
        negative_prompt: Negative prompt to avoid unwanted elements (default: quality-focused)
        frame_rate: Output video frame rate (default 24)
        steps: Number of sampling steps (default 30)
        cfg: Classifier-free guidance scale (default 7.0)
        seed: Random seed for reproducibility (None = random)
        width: Output video width (default 1280)
        height: Output video height (default 720)
        num_frames: Number of frames to generate (default 49)
    
    Returns:
        Modified workflow graph (dict)
    """
    # YUME workflows are flat (no 'prompt' wrapper), so handle both cases
    if "prompt" in workflow_api_data:
        api_graph = copy.deepcopy(workflow_api_data["prompt"])
    else:
        api_graph = copy.deepcopy(workflow_api_data)
    
    # Generate random seed if not provided
    if seed is None or seed == 0:
        seed = random.randint(0, 2**63 - 1)
    
    # Find LoadImage node and inject the start image
    for node_id, node in api_graph.items():
        if node.get("class_type") == "LoadImage":
            node["inputs"]["image"] = image_path
            logging.info(f"YUME I2V: Injected start image: {image_path}")
            break
    else:
        logging.warning("Could not find LoadImage node in YUME I2V workflow")
    
    # Find YUME_Node and inject parameters
    for node_id, node in api_graph.items():
        if node.get("class_type") == "YUME_Node":
            node["inputs"]["prompt"] = prompt
            node["inputs"]["n_prompt"] = negative_prompt
            node["inputs"]["steps"] = steps
            node["inputs"]["cfg"] = cfg
            node["inputs"]["seed"] = seed
            node["inputs"]["width"] = width
            node["inputs"]["height"] = height
            node["inputs"]["num_frames"] = num_frames
            logging.info(f"YUME I2V configured - Prompt: '{prompt[:50]}...', Steps: {steps}, CFG: {cfg}, Seed: {seed}")
            break
    else:
        logging.warning("Could not find YUME_Node in workflow")
    
    # Update VHS_VideoCombine frame rate and filename_prefix
    for node_id, node in api_graph.items():
        if node.get("class_type") == "VHS_VideoCombine":
            node["inputs"]["frame_rate"] = frame_rate
            
            # Replace date token in filename_prefix
            prefix = node["inputs"].get("filename_prefix", "")
            if "%date:yyyy-MM-dd%" in prefix:
                datestr = datetime.now().strftime("%Y-%m-%d")
                node["inputs"]["filename_prefix"] = prefix.replace("%date:yyyy-MM-dd%", datestr)
            
            break
    
    return api_graph
