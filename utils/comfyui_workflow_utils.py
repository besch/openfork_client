import os
import json
import base64
import copy
import logging
import random
from datetime import datetime
from typing import Union, Dict
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

def inject_prompt_and_image_into_workflow(workflow_api_data: Dict, prompt: str, negative_prompt: str, start_image_filename: str, aspect_ratio: str = "16:9"):
    """
    Loads a ComfyUI API-formatted workflow, injects prompts and image filename.
    Also randomizes seed for image-to-video generation.
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

    # Randomize seed for KSamplerAdvanced nodes (typically node 57 for high noise)
    if '57' in api_graph and 'inputs' in api_graph['57']:
        api_graph['57']['inputs']['noise_seed'] = random.randint(0, 2**63 - 1)

    return api_graph

def inject_prompt_into_text_to_video_workflow(workflow_api_data: Dict, prompt: str, negative_prompt: str, aspect_ratio: str = "16:9"):
    """
    Loads a ComfyUI API-formatted workflow, injects prompts for text-to-video.
    Also randomizes seed for varied outputs.
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

    # Randomize seed for node 57 (high noise sampler)
    if '57' in api_graph and 'inputs' in api_graph['57']:
        new_seed = random.randint(0, 2**63 - 1)
        api_graph['57']['inputs']['noise_seed'] = new_seed
    else:
        logging.warning("Could not find sampler node 57 to randomize seed")

    # Replace date token in filename_prefix for VHS_VideoCombine node
    for node in api_graph.values():
        if node.get("class_type") == "VHS_VideoCombine":
            prefix = node["inputs"].get("filename_prefix", "")
            if "%date:yyyy-MM-dd%" in prefix:
                datestr = datetime.now().strftime("%Y-%m-%d")
                node["inputs"]["filename_prefix"] = prefix.replace("%date:yyyy-MM-dd%", datestr)

    return api_graph

def get_safe_ltx_resolution(aspect_ratio: str, quality_preset: str = "standard") -> tuple[int, int]:
    """
    Returns resolution (width, height) optimized for LTX Video.
    """
    width, height = get_dimensions(aspect_ratio, default_width=768, default_height=512)
    
    if quality_preset == "low_vram":
        # Optimization for 8GB cards with --lowvram and --cpu-vae enabled
        # We can push to ~480p/512p now
        if aspect_ratio == "16:9":
            return 768, 448 # ~448p, multiple of 32
        elif aspect_ratio == "9:16":
            return 448, 768
        elif aspect_ratio == "1:1":
            return 512, 512
        elif aspect_ratio == "21:9":
            return 832, 352
        else:
             return 640, 480
             
    return width, height

def inject_prompt_into_ltx_video_workflow(workflow_api_data: Dict, prompt: str, negative_prompt: str, aspect_ratio: str = "16:9", quality_preset: str = "standard"):
    """
    Loads a ComfyUI API-formatted workflow, injects prompts for LTX Video.
    Also randomizes seed for varied outputs.
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])

    # Node 2 is positive, Node 3 is negative
    if '2' in api_graph and 'inputs' in api_graph['2']:
        api_graph['2']['inputs']['text'] = prompt
    if '3' in api_graph and 'inputs' in api_graph['3']:
        api_graph['3']['inputs']['text'] = negative_prompt

    # Configuration based on quality_preset
    frame_count = 121 # Default (Standard) - ~5 seconds
    steps = 35 # Default steps
    width, height = get_safe_ltx_resolution(aspect_ratio, quality_preset)
    
    ckpt_name = "ltx-video-2b-v0.9.1.safetensors"
    
    if quality_preset == "low_vram":
        # Optimization for 8GB VRAM with flags
        # 65 frames = ~2.7s @ 24fps
        frame_count = 65 
        steps = 30
        
        # Try to use GGUF model if available (user must download it)
        ckpt_name = "ltx-video-2b-v0.9.1.safetensors" 
        # ckpt_name = "ltx-video-2b-v0.9.1-q4_k_m.gguf" # TODO: Uncomment if GGUF is verified present

    elif quality_preset == "high_quality":
        frame_count = 169 # ~7 seconds
        steps = 50

    # Inject dimensions and frame count (batch_size) into EmptyLatentImage (Node 4)
    if '4' in api_graph: 
        # Prefer EmptyLTXVLatentVideo if possible for correct video latents
        api_graph['4']['class_type'] = "EmptyLTXVLatentVideo"
        
        # Ensure inputs are correct for EmptyLTXVLatentVideo
        if 'inputs' not in api_graph['4']: api_graph['4']['inputs'] = {}
        
        api_graph['4']['inputs']['width'] = width
        api_graph['4']['inputs']['height'] = height
        api_graph['4']['inputs']['length'] = frame_count
        api_graph['4']['inputs']['batch_size'] = 1
        
        logging.info(f"LTX T2V Configured: {width}x{height}, {frame_count} frames, {steps} steps, Preset: {quality_preset}, Node: EmptyLTXVLatentVideo")

    # Inject steps and randomized seed for node 5 (KSampler)
    if '5' in api_graph and 'inputs' in api_graph['5']:
        new_seed = random.randint(0, 2**63 - 1)
        api_graph['5']['inputs']['seed'] = new_seed
        api_graph['5']['inputs']['steps'] = steps
    
    if '1' in api_graph and 'inputs' in api_graph['1']:
         api_graph['1']['inputs']['ckpt_name'] = ckpt_name

    return api_graph

def inject_prompt_and_image_into_ltx_video_workflow(workflow_api_data: Dict, prompt: str, negative_prompt: str, start_image_filename: str, aspect_ratio: str = "16:9", quality_preset: str = "standard"):
    """
    Loads a ComfyUI API-formatted workflow for LTX I2V.
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])

    # Text Prompts (2 & 3)
    if '2' in api_graph: api_graph['2']['inputs']['text'] = prompt
    if '3' in api_graph: api_graph['3']['inputs']['text'] = negative_prompt

    # Image Load (4)
    if '4' in api_graph and api_graph['4'].get("class_type") == "LoadImage":
        api_graph['4']['inputs']['image'] = start_image_filename

    # Image Resize (10)
    width, height = get_safe_ltx_resolution(aspect_ratio, quality_preset)
    if '10' in api_graph and api_graph['10'].get("class_type") == "ImageResizeKJv2":
        api_graph['10']['inputs']['width'] = width
        api_graph['10']['inputs']['height'] = height
        logging.info(f"LTX I2V Configured: {width}x{height}, Preset: {quality_preset}")

    # Determine steps based on quality preset
    steps = 35 # Default
    if quality_preset == "low_vram":
        steps = 30
    elif quality_preset == "high_quality":
        steps = 50

    # Randomize seed (6) and inject steps
    if '6' in api_graph and 'inputs' in api_graph['6']:
        new_seed = random.randint(0, 2**63 - 1)
        api_graph['6']['inputs']['seed'] = new_seed
        api_graph['6']['inputs']['steps'] = steps

    # Checkpoint configuration
    ckpt_name = "ltx-video-2b-v0.9.1.safetensors"
    if '1' in api_graph and 'inputs' in api_graph['1']:
         api_graph['1']['inputs']['ckpt_name'] = ckpt_name

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
        # LTX Nodes (adding them here)
        'CheckpointLoaderSimple', 
        'VAEEncode',
        'EmptyLTXVLatentVideo',
        
        # Misc / meta
        'Note',
        # Hunyuan Nodes
        'HunyuanVideoModelLoaderGGUF',
        'EmptyHunyuanLatentVideo',
        'HyVideoI2V',
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
            # Temporarily log warning instead of failing block for development
            logging.warning(f"Security Alert: Workflow contains a non-approved node: {node_type}")
            # ok = False 
    return ok

def inject_video_into_upscaler_workflow(
    workflow_api_data: Dict, 
    video_filename: str,
    upscale_model: str = "RealESRGAN_x4plus.pth",
    frame_rate: int = 30,
    target_width: Union[int, None] = None,
    target_height: Union[int, None] = None,
    scale_by: Union[float, None] = None
):
    """
    Injects video filename and upscale settings into Real-ESRGAN workflow.
    
    Args:
        workflow_api_data: The workflow JSON structure
        video_filename: Name of the video file in input directory
        upscale_model: Which Real-ESRGAN model to use
        frame_rate: Output video frame rate
        target_width: The final width of the video (optional if scale_by is used)
        target_height: The final height of the video (optional if scale_by is used)
        scale_by: Factor to scale the image by (optional, overrides width/height)
    """
    api_graph = copy.deepcopy(workflow_api_data["prompt"])

    # Node 1: VHS_LoadVideo - inject video filename
    if '1' in api_graph and api_graph['1']['class_type'] == 'VHS_LoadVideo':
        api_graph['1']['inputs']['video'] = video_filename
        # Force disable resizing to ensure original dimensions are used
        api_graph['1']['inputs']['force_size'] = "Disabled"
        api_graph['1']['inputs']['custom_width'] = 0
        api_graph['1']['inputs']['custom_height'] = 0
        logging.info(f"Injected video filename: {video_filename} and disabled force_size")
    
    # Node 2: UpscaleModelLoader - inject model selection
    if '2' in api_graph and api_graph['2']['class_type'] == 'UpscaleModelLoader':
        api_graph['2']['inputs']['model_name'] = upscale_model
        logging.info(f"Injected upscale model: {upscale_model}")

    # Node 6: ImageScale - inject dimensions
    if '6' in api_graph:
        if scale_by is not None:
            # Use ImageScaleBy for factor-based scaling
            api_graph['6']['class_type'] = 'ImageScaleBy'
            api_graph['6']['inputs'] = {
                'scale_by': scale_by,
                'image': ['3', 0],
                'upscale_method': 'lanczos'
            }
            logging.info(f"Using ImageScaleBy with factor: {scale_by}")
        elif target_width is not None and target_height is not None:
            # Ensure it's ImageScale
            if api_graph['6']['class_type'] != 'ImageScale':
                    api_graph['6']['class_type'] = 'ImageScale'
            
            api_graph['6']['inputs']['width'] = target_width
            api_graph['6']['inputs']['height'] = target_height
            # Ensure other inputs are correct
            api_graph['6']['inputs']['upscale_method'] = 'lanczos'
            api_graph['6']['inputs']['crop'] = 'disabled'
            api_graph['6']['inputs']['image'] = ['3', 0]

            logging.info(f"Injected target dimensions: {target_width}x{target_height}")
        else:
                logging.warning("No target dimensions or scale_by provided for upscale workflow. Using default/existing values.")
    
    # Node 4: VHS_VideoCombine - inject frame rate and filename
    if '4' in api_graph and api_graph['4']['class_type'] == 'VHS_VideoCombine':
        api_graph['4']['inputs']['frame_rate'] = frame_rate
        # Add timestamp to filename
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

def inject_prompt_into_hunyuan_workflow(workflow_api_data: Dict, prompt: str, steps: int = 30, guidance: float = 6.0, strength: float = 1.0, width: int = 854, height: int = 480, frame_count: int = 49):
    """
    Injects parameters into the HunyuanVideo workflow.
    """
    api_graph = copy.deepcopy(workflow_api_data.get("prompt", workflow_api_data))
    
    # Generate random seed
    seed = random.randint(0, 2**63 - 1)

    for node_id, node in api_graph.items():
        class_type = node.get("class_type")
        inputs = node.get("inputs", {})

        # KSampler for steps and guidance (cfg)
        if class_type == "KSampler":
            inputs["steps"] = int(steps)
            inputs["cfg"] = float(guidance)
            inputs["seed"] = seed
            if "denoise" in inputs:
                inputs["denoise"] = float(strength)

        # EmptyHunyuanLatentVideo for resolution and frames
        if class_type == "EmptyHunyuanLatentVideo":
            inputs["width"] = int(width)
            inputs["height"] = int(height)
            inputs["length"] = int(frame_count)
            inputs["batch_size"] = 1

        # CLIPTextEncode (Positive)
        if class_type == "CLIPTextEncode":
             title = node.get("_meta", {}).get("title", "")
             if "Positive" in title:
                 inputs["text"] = prompt
             # Also check if it's the positive prompt node 6
             elif node_id == "6": 
                 inputs["text"] = prompt

    return api_graph

def inject_prompt_and_image_into_hunyuan_workflow(workflow_api_data: Dict, prompt: str, start_image_filename: str, steps: int = 30, guidance: float = 6.0, strength: float = 1.0, width: int = 854, height: int = 480, frame_count: int = 49):
    """
    Injects parameters and image into the HunyuanVideo I2V workflow.
    """
    api_graph = copy.deepcopy(workflow_api_data.get("prompt", workflow_api_data))
    
    # Generate random seed
    seed = random.randint(0, 2**63 - 1)

    for node_id, node in api_graph.items():
        class_type = node.get("class_type")
        inputs = node.get("inputs", {})

        # KSampler
        if class_type == "KSampler":
            inputs["steps"] = int(steps)
            inputs["cfg"] = float(guidance)
            inputs["seed"] = seed
            if "denoise" in inputs:
                inputs["denoise"] = float(strength)

        # EmptyHunyuanLatentVideo
        if class_type == "EmptyHunyuanLatentVideo":
            inputs["width"] = int(width)
            inputs["height"] = int(height)
            inputs["length"] = int(frame_count)
            inputs["batch_size"] = 1

        # CLIPTextEncode (Positive)
        if class_type == "CLIPTextEncode":
             title = node.get("_meta", {}).get("title", "")
             if "Positive" in title:
                 inputs["text"] = prompt
             elif node_id == "6": 
                 inputs["text"] = prompt

        # HyVideoI2V (Positive with Image)
        if class_type == "HyVideoI2V":
            inputs["prompt"] = prompt

        # LoadImage
        if class_type == "LoadImage":
            inputs["image"] = start_image_filename

    return api_graph