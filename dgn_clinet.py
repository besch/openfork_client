import subprocess
import json
import psutil
import platform
import os
import sys
from uuid import uuid4

def check_requirements():
    """Check if Docker and NVIDIA drivers are installed."""
    try:
        # Check Docker
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, check=True)
        print(f"Docker found: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Docker is not installed. Please install Docker Desktop.")
        sys.exit(1)

    try:
        # Check NVIDIA drivers
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        print("NVIDIA drivers found.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: NVIDIA drivers not found. Please install NVIDIA drivers and CUDA toolkit.")
        sys.exit(1)

def profile_hardware():
    """Profile the system hardware and return a profile dictionary."""
    profile = {
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": psutil.cpu_count(),
        "ram_total_mb": psutil.virtual_memory().total // (1024 * 1024)
    }

    try:
        nvidia_smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], 
                                   capture_output=True, text=True, check=True)
        lines = nvidia_smi.stdout.strip().split('\n')[1:]  # Skip header
        profile["gpus"] = [
            {"name": line.split(',')[0].strip(), "vram_mb": int(line.split(',')[1].strip().split()[0])}
            for line in lines
        ]
    except (subprocess.CalledProcessError, FileNotFoundError):
        profile["gpus"] = []

    return profile

def save_workflow_json(workflow_data, filename="workflow.json"):
    """Save the provided workflow JSON to a file."""
    with open(filename, 'w') as f:
        json.dump(workflow_data, f, indent=2)
    return os.path.abspath(filename)

def pull_docker_image(image_name="comfyui-worker:latest"):
    """Pull or build the Docker image."""
    try:
        subprocess.run(["docker", "pull", image_name], check=True)
        print(f"Pulled Docker image: {image_name}")
    except subprocess.CalledProcessError:
        print(f"Image {image_name} not found in registry. Building locally...")
        subprocess.run(["docker", "build", "-t", image_name, "."], check=True)
        print(f"Built Docker image: {image_name}")

def run_comfyui_job(workflow_path, output_dir="output"):
    """Run the ComfyUI job in a Docker container."""
    os.makedirs(output_dir, exist_ok=True)
    image_name = "comfyui-worker:latest"
    
    # Ensure Docker image is available
    pull_docker_image(image_name)

    # Run Docker container
    try:
        cmd = [
            "docker", "run", "--gpus", "all",
            "-v", f"{os.path.abspath(output_dir)}:/app/ComfyUI/output",
            "-v", f"{workflow_path}:/app/ComfyUI/workflow.json",
            image_name
        ]
        subprocess.run(cmd, check=True)
        print(f"Job completed. Output saved in {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error running Docker container: {e}")
        sys.exit(1)

def main():
    # Check system requirements
    check_requirements()

    # Profile hardware
    hardware_profile = profile_hardware()
    print("Hardware Profile:", json.dumps(hardware_profile, indent=2))

    # Workflow JSON (from user input)
    workflow_data = {
        "id": "79268578-a2ac-4bb4-b7aa-d1e963e57124",
        "revision": 0,
        "last_node_id": 142,
        "last_link_id": 249,
        "nodes": [
            {"id": 39, "type": "VAELoader", "pos": [-1290, 510], "size": [390, 60], "flags": {}, "order": 0, "mode": 0, "inputs": [{"localized_name": "vae_name", "name": "vae_name", "type": "COMBO", "widget": {"name": "vae_name"}, "link": null}], "outputs": [{"localized_name": "VAE", "name": "VAE", "type": "VAE", "slot_index": 0, "links": [76, 210]}], "properties": {"cnr_id": "comfy-core", "ver": "0.3.34", "Node name for S&R": "VAELoader", "models": [{"name": "wan_2.1_vae.safetensors", "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors", "hash": "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b", "hash_type": "SHA256", "directory": "vae"}], "widget_ue_connectable": {}}, "widgets_values": ["wan_2.1_vae.safetensors"]},
            {"id": 125, "type": "UnetLoaderGGUF", "pos": [-1290, 50], "size": [370, 60], "flags": {}, "order": 1, "mode": 0, "inputs": [{"localized_name": "unet_name", "name": "unet_name", "type": "COMBO", "widget": {"name": "unet_name"}, "link": null}], "outputs": [{"localized_name": "MODEL", "name": "MODEL", "type": "MODEL", "links": [229]}], "properties": {"cnr_id": "comfyui-gguf", "ver": "6570efec6992015085f11b84e42d32f6cc71e8b7", "Node name for S&R": "UnetLoaderGGUF", "widget_ue_connectable": {}}, "widgets_values": ["wan2.1-i2v-14b-480p-Q5_K_S.gguf"]},
            {"id": 3, "type": "KSampler", "pos": [960, 190], "size": [329.5036926269531, 262], "flags": {}, "order": 17, "mode": 0, "inputs": [{"localized_name": "model", "name": "model", "type": "MODEL", "link": 95}, {"localized_name": "positive", "name": "positive", "type": "CONDITIONING", "link": 215}, {"localized_name": "negative", "name": "negative", "type": "CONDITIONING", "link": 216}, {"localized_name": "latent_image", "name": "latent_image", "type": "LATENT", "link": 214}, {"localized_name": "seed", "name": "seed", "type": "INT", "widget": {"name": "seed"}, "link": null}, {"localized_name": "steps", "name": "steps", "type": "INT", "widget": {"name": "steps"}, "link": null}, {"localized_name": "cfg", "name": "cfg", "type": "FLOAT", "widget": {"name": "cfg"}, "link": null}, {"localized_name": "sampler_name", "name": "sampler_name", "type": "COMBO", "widget": {"name": "sampler_name"}, "link": null}, {"localized_name": "scheduler", "name": "scheduler", "type": "COMBO", "widget": {"name": "scheduler"}, "link": null}, {"localized_name": "denoise", "name": "denoise", "type": "FLOAT", "widget": {"name": "denoise"}, "link": null}], "outputs": [{"localized_name": "LATENT", "name": "LATENT", "type": "LATENT", "slot_index": 0, "links": [241]}], "properties": {"cnr_id": "comfy-core", "ver": "0.3.34", "Node name for S&R": "KSampler", "widget_ue_connectable": {}}, "widgets_values": [342046170243252, "randomize", 6, 1, "euler", "simple", 1]},
            {"id": 48, "type": "ModelSamplingSD3", "pos": [960, 90], "size": [210, 58], "flags": {}, "order": 16, "mode": 0, "inputs": [{"localized_name": "model", "name": "model", "type": "MODEL", "link": 247}, {"localized_name": "shift", "name": "shift", "type": "FLOAT", "widget": {"name": "shift"}, "link": null}], "outputs": [{"localized_name": "MODEL", "name": "MODEL", "type": "MODEL", "slot_index": 0, "links": [95]}], "properties": {"cnr_id": "comfy-core", "ver": "0.3.34", "Node name for S&R": "ModelSamplingSD3", "widget_ue_connectable": {}}, "widgets_values": [2.0000000000000004]},
            {"id": 8, "type": "VAEDecode", "pos": [960, 500], "size": [210, 46], "flags": {"collapsed": false}, "order": 18, "mode": 0, "inputs": [{"localized_name": "samples", "name": "samples", "type": "LATENT", "link": 241}, {"localized_name": "vae", "name": "vae", "type": "VAE", "link": 76}], "outputs": [{"localized_name": "IMAGE", "name": "IMAGE", "type": "IMAGE", "slot_index": 0, "links": [189]}], "properties": {"cnr_id": "comfy-core", "ver": "0.3.34", "Node name for S&R": "VAEDecode", "widget_ue_connectable": {}}, "widgets_values": []},
            {"id": 73, "type": "LoadImage", "pos": [180, 710], "size": [274.080078125, 314.00006103515625], "flags": {}, "order": 2, "mode": 0, "inputs": [{"localized_name": "image", "name": "image", "type": "COMBO", "widget": {"name": "image"}, "link": null}, {"localized_name": "choose file to upload", "name": "upload", "type": "IMAGEUPLOAD", "widget": {"name": "upload"}, "link": null}], "outputs": [{"localized_name": "IMAGE", "name": "IMAGE", "type": "IMAGE", "links": [235]}, {"localized_name": "MASK", "name": "MASK", "type": "MASK", "links": null}], "properties": {"cnr_id": "comfy-core", "ver": "0.3.34", "Node name for S&R": "LoadImage", "widget_ue_connectable": {}}, "widgets_values": ["ComfyUI_00154_.png", "image"]},
            {"id": 75, "type": "PreviewImage", "pos": [760, 690], "size": [510, 350], "flags": {}, "order": 9, "mode": 0, "inputs": [{"localized_name": "images", "name": "images", "type": "IMAGE", "link": 240}], "outputs": [], "properties": {"cnr_id": "comfy-core", "ver": "0.3.34", "Node name for S&R": "PreviewImage", "widget_ue_connectable": {}}, "widgets_values": []},
            {"id": 140, "type": "ImageResizeKJv2", "pos": [480, 690], "size": [260, 348], "flags": {}, "order": 5, "mode": 0, "inputs": [{"localized_name": "image", "name": "image", "type": "IMAGE", "link": 235}, {"localized_name": "mask", "name": "mask", "shape": 7, "type": "MASK", "link": null}, {"localized_name": "width", "name": "width", "type": "INT", "widget": {"name": "width"}, "link": null}, {"localized_name": "height", "name": "height", "type": "INT", "widget": {"name": "height"}, "link": null}, {"localized_name": "upscale_method", "name": "upscale_method", "type": "COMBO", "widget": {"name": "upscale_method"}, "link": null}, {"localized_name": "keep_proportion", "name": "keep_proportion", "type": "COMBO", "widget": {"name": "keep_proportion"}, "link": null}, {"localized_name": "pad_color", "name": "pad_color", "type": "STRING", "widget": {"name": "pad_color"}, "link": null}, {"localized_name": "crop_position", "name": "crop_position", "type": "COMBO", "widget": {"name": "crop_position"}, "link": null}, {"localized_name": "divisible_by", "name": "divisible_by", "type": "INT", "widget": {"name": "divisible_by"}, "link": null}, {"localized_name": "device", "name": "device", "shape": 7, "type": "COMBO", "widget": {"name": "device"}, "link": null}], "outputs": [{"localized_name": "IMAGE", "name": "IMAGE", "type": "IMAGE", "links": [236, 240]}, {"localized_name": "width", "name": "width", "type": "INT", "links": [237]}, {"localized_name": "height", "name": "height", "type": "INT", "links": [238]}, {"localized_name": "mask", "name": "mask", "type": "MASK", "links": null}], "properties": {"cnr_id": "comfyui-kjnodes", "ver": "0d909572e226a49cae540cfe436551e93836db20", "Node name for S&R": "ImageResizeKJv2"}, "widgets_values": [480, 480, "lanczos", "crop", "0, 0, 0", "center", 16, "cpu"]},
            {"id": 112, "type": "VHS_VideoCombine", "pos": [1390, 70], "size": [520, 848], "flags": {}, "order": 19, "mode": 0, "inputs": [{"localized_name": "images", "name": "images", "type": "IMAGE", "link": 189}, {"localized_name": "audio", "name": "audio", "shape": 7, "type": "AUDIO", "link": null}, {"localized_name": "meta_batch", "name": "meta_batch", "shape": 7, "type": "VHS_BatchManager", "link": null}, {"localized_name": "vae", "name": "vae", "shape": 7, "type": "VAE", "link": null}, {"localized_name": "frame_rate", "name": "frame_rate", "type": "FLOAT", "widget": {"name": "frame_rate"}, "link": null}, {"localized_name": "loop_count", "name": "loop_count", "type": "INT", "widget": {"name": "loop_count"}, "link": null}, {"localized_name": "filename_prefix", "name": "filename_prefix", "type": "STRING", "widget": {"name": "filename_prefix"}, "link": null}, {"localized_name": "format", "name": "format", "type": "COMBO", "widget": {"name": "format"}, "link": null}, {"localized_name": "pingpong", "name": "pingpong", "type": "BOOLEAN", "widget": {"name": "pingpong"}, "link": null}, {"localized_name": "save_output", "name": "save_output", "type": "BOOLEAN", "widget": {"name": "save_output"}, "link": null}], "outputs": [{"localized_name": "Filenames", "name": "Filenames", "type": "VHS_FILENAMES", "links": null}], "properties": {"cnr_id": "comfyui-videohelpersuite", "ver": "1.6.1", "Node name for S&R": "VHS_VideoCombine", "widget_ue_connectable": {}}, "widgets_values": {"frame_rate": 16, "loop_count": 0, "filename_prefix": "%date:yyyy-MM-dd%/wanvid", "format": "video/nvenc_h264-mp4", "pix_fmt": "yuv420p", "bitrate": 10, "megabit": true, "save_metadata": true, "pingpong": false, "save_output": true, "videopreview": {"hidden": false, "paused": false, "params": {"filename": "wanvid_00002.mp4", "subfolder": "2025-07-22", "type": "output", "format": "video/nvenc_h264-mp4", "frame_rate": 16, "workflow": "wanvid_00002.png", "fullpath": "C:\\Users\\panal\\Documents\\ComfyUI\\output\\2025-07-22\\wanvid_00002.mp4"}}}},
            {"id": 142, "type": "WanVideoNAG", "pos": [640, 70], "size": [270, 126], "flags": {}, "order": 15, "mode": 0, "inputs": [{"localized_name": "model", "name": "model", "type": "MODEL", "link": 246}, {"localized_name": "conditioning", "name": "conditioning", "type": "CONDITIONING", "link": 248}, {"localized_name": "nag_scale", "name": "nag_scale", "type": "FLOAT", "widget": {"name": "nag_scale"}, "link": null}, {"localized_name": "nag_alpha", "name": "nag_alpha", "type": "FLOAT", "widget": {"name": "nag_alpha"}, "link": null}, {"localized_name": "nag_tau", "name": "nag_tau", "type": "FLOAT", "widget": {"name": "nag_tau"}, "link": null}], "outputs": [{"localized_name": "model", "name": "model", "type": "MODEL", "links": [247]}], "properties": {"cnr_id": "comfyui-kjnodes", "ver": "0d909572e226a49cae540cfe436551e93836db20", "Node name for S&R": "WanVideoNAG"}, "widgets_values": [11, 0.25, 2.5]},
            {"id": 6, "type": "CLIPTextEncode", "pos": [150, 280], "size": [420, 290], "flags": {}, "order": 6, "mode": 0, "inputs": [{"localized_name": "clip", "name": "clip", "type": "CLIP", "link": 233}, {"localized_name": "text", "name": "text", "type": "STRING", "widget": {"name": "text"}, "link": null}], "outputs": [{"localized_name": "CONDITIONING", "name": "CONDITIONING", "type": "CONDITIONING", "slot_index": 0, "links": [208]}], "title": "CLIP Text Encode (Positive Prompt)", "properties": {"cnr_id": "comfy-core", "ver": "0.3.34", "Node name for S&R": "CLIPTextEncode", "widget_ue_connectable": {}}, "widgets_values": ["the woman aims her pistol and shoots three shots. muzzle flashes on each shot"], "color": "#232", "bgcolor": "#353"},
            {"id": 7, "type": "CLIPTextEncode", "pos": [150, 60], "size": [425.27801513671875, 180.6060791015625], "flags": {}, "order": 7, "mode": 0, "inputs": [{"localized_name": "clip", "name": "clip", "type": "CLIP", "link": 234}, {"localized_name": "text", "name": "text", "type": "STRING", "widget": {"name": "text"}, "link": null}], "outputs": [{"localized_name": "CONDITIONING", "name": "CONDITIONING", "type": "CONDITIONING", "slot_index": 0, "links": [209, 248]}], "title": "CLIP Text Encode (Negative Prompt)", "properties": {"cnr_id": "comfy-core", "ver": "0.3.34", "Node name for S&R": "CLIPTextEncode", "widget_ue_connectable": {}}, "widgets_values": ["poorly drawn, bad anatomy, bad hands, bad eyes, missing fingers, extra fingers, ugly, deformed, disfigured, blurry, grainy, out of focus, low resolution, amateur, poorly lit, oversaturated, undersaturated, watermark, signature, text, writing, noise, artifacts, cgi, 3d, illustration"], "color": "#322", "bgcolor": "#533"},
            {"id": 108, "type": "PathchSageAttentionKJ", "pos": [-860, 60], "size": [270, 58], "flags": {}, "order": 4, "mode": 0, "inputs": [{"localized_name": "model", "name": "model", "type": "MODEL", "link": 229}, {"localized_name": "sage_attention", "name": "sage_attention", "type": "COMBO", "widget": {"name": "sage_attention"}, "link": null}], "outputs": [{"localized_name": "MODEL", "name": "MODEL", "type": "MODEL", "links": [187]}], "properties": {"cnr_id": "comfyui-kjnodes", "ver": "5dcda71011870278c35d92ff77a677ed2e538f2d", "Node name for S&R": "PathchSageAttentionKJ", "widget_ue_connectable": {}}, "widgets_values": ["sageattn_qk_int8_pv_fp8_cuda++"]},
            {"id": 111, "type": "ModelPatchTorchSettings", "pos": [-860, 170], "size": [307.443359375, 58], "flags": {}, "order": 8, "mode": 0, "inputs": [{"localized_name": "model", "name": "model", "type": "MODEL", "link": 187}, {"localized_name": "enable_fp16_accumulation", "name": "enable_fp16_accumulation", "type": "BOOLEAN", "widget": {"name": "enable_fp16_accumulation"}, "link": null}], "outputs": [{"localized_name": "MODEL", "name": "MODEL", "type": "MODEL", "links": [217]}], "properties": {"cnr_id": "comfyui-kjnodes", "ver": "5dcda71011870278c35d92ff77a677ed2e538f2d", "Node name for S&R": "ModelPatchTorchSettings", "widget_ue_connectable": {}}, "widgets_values": [true]},
            {"id": 118, "type": "CLIPLoader", "pos": [-1290, 330], "size": [350, 130], "flags": {}, "order": 3, "mode": 0, "inputs": [{"localized_name": "clip_name", "name": "clip_name", "type": "COMBO", "widget": {"name": "clip_name"}, "link": null}, {"localized_name": "type", "name": "type", "type": "COMBO", "widget": {"name": "type"}, "link": null}, {"localized_name": "device", "name": "device", "shape": 7, "type": "COMBO", "widget": {"name": "device"}, "link": null}], "outputs": [{"localized_name": "CLIP", "name": "CLIP", "type": "CLIP", "links": [233, 234]}], "properties": {"cnr_id": "comfy-core", "ver": "0.3.32", "Node name for S&R": "CLIPLoader", "widget_ue_connectable": {}}, "widgets_values": ["umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "cpu"]},
            {"id": 130, "type": "LoraLoaderModelOnly", "pos": [-450, 50], "size": [494.1808776855469, 82], "flags": {}, "order": 11, "mode": 0, "inputs": [{"localized_name": "model", "name": "model", "type": "MODEL", "link": 217}, {"localized_name": "lora_name", "name": "lora_name", "type": "COMBO", "widget": {"name": "lora_name"}, "link": null}, {"localized_name": "strength_model", "name": "strength_model", "type": "FLOAT", "widget": {"name": "strength_model"}, "link": null}], "outputs": [{"localized_name": "MODEL", "name": "MODEL", "type": "MODEL", "links": [223]}], "properties": {"cnr_id": "comfy-core", "ver": "0.3.36", "Node name for S&R": "LoraLoaderModelOnly"}, "widgets_values": ["wan21_lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors", 1.0000000000000002]},
            {"id": 129, "type": "WanImageToVideo", "pos": [640, 290], "size": [280, 230], "flags": {}, "order": 10, "mode": 0, "inputs": [{"localized_name": "positive", "name": "positive", "type": "CONDITIONING", "link": 208}, {"localized_name": "negative", "name": "negative", "type": "CONDITIONING", "link": 209}, {"localized_name": "vae", "name": "vae", "type": "VAE", "link": 210}, {"localized_name": "clip_vision_output", "name": "clip_vision_output", "shape": 7, "type": "CLIP_VISION_OUTPUT", "link": null}, {"localized_name": "start_image", "name": "start_image", "shape": 7, "type": "IMAGE", "link": 236}, {"localized_name": "width", "name": "width", "type": "INT", "widget": {"name": "width"}, "link": 237}, {"localized_name": "height", "name": "height", "type": "INT", "widget": {"name": "height"}, "link": 238}, {"localized_name": "length", "name": "length", "type": "INT", "widget": {"name": "length"}, "link": null}, {"localized_name": "batch_size", "name": "batch_size", "type": "INT", "widget": {"name": "batch_size"}, "link": null}], "outputs": [{"localized_name": "positive", "name": "positive", "type": "CONDITIONING", "links": [215]}, {"localized_name": "negative", "name": "negative", "type": "CONDITIONING", "links": [216]}, {"localized_name": "latent", "name": "latent", "type": "LATENT", "links": [214]}], "properties": {"cnr_id": "comfy-core", "ver": "0.3.36", "Node name for S&R": "WanImageToVideo"}, "widgets_values": [480, 480, 81, 1]},
            {"id": 131, "type": "LoraLoaderModelOnly", "pos": [-450, 290], "size": [487.6258544921875, 82], "flags": {}, "order": 13, "mode": 4, "inputs": [{"localized_name": "model", "name": "model", "type": "MODEL", "link": 224}, {"localized_name": "lora_name", "name": "lora_name", "type": "COMBO", "widget": {"name": "lora_name"}, "link": null}, {"localized_name": "strength_model", "name": "strength_model", "type": "FLOAT", "widget": {"name": "strength_model"}, "link": null}], "outputs": [{"localized_name": "MODEL", "name": "MODEL", "type": "MODEL", "links": [225]}], "properties": {"cnr_id": "comfy-core", "ver": "0.3.36", "Node name for S&R": "LoraLoaderModelOnly"}, "widgets_values": ["Wan21_AccVid_I2V_480P_14B_lora_rank32_fp16.safetensors", 1.0000000000000002]},
            {"id": 134, "type": "LoraLoaderModelOnly", "pos": [-450, 410], "size": [490, 90], "flags": {}, "order": 14, "mode": 4, "inputs": [{"localized_name": "model", "name": "model", "type": "MODEL", "link": 225}, {"localized_name": "lora_name", "name": "lora_name", "type": "COMBO", "widget": {"name": "lora_name"}, "link": null}, {"localized_name": "strength_model", "name": "strength_model", "type": "FLOAT", "widget": {"name": "strength_model"}, "link": null}], "outputs": [{"localized_name": "MODEL", "name": "MODEL", "type": "MODEL", "links": [246]}], "properties": {"cnr_id": "comfy-core", "ver": "0.3.36", "Node name for S&R": "LoraLoaderModelOnly"}, "widgets_values": ["Wan2.1-Fun-14B-InP-MPS.safetensors", 1.0000000000000002]},
            {"id": 133, "type": "LoraLoaderModelOnly", "pos": [-450, 170], "size": [496.8028869628906, 82], "flags": {}, "order": 12, "mode": 0, "inputs": [{"localized_name": "model", "name": "model", "type": "MODEL", "link": 223}, {"localized_name": "lora_name", "name": "lora_name", "type": "COMBO", "widget": {"name": "lora_name"}, "link": null}, {"localized_name": "strength_model", "name": "strength_model", "type": "FLOAT", "widget": {"name": "strength_model"}, "link": null}], "outputs": [{"localized_name": "MODEL", "name": "MODEL", "type": "MODEL", "links": [224]}], "properties": {"cnr_id": "comfy-core", "ver": "0.3.36", "Node name for S&R": "LoraLoaderModelOnly"}, "widgets_values": ["wan21_pusa_v1.safetensors", 1.2000000000000002]}
        ],
        "links": [
            [76, 39, 0, 8, 1, "VAE"],
            [95, 48, 0, 3, 0, "MODEL"],
            [187, 108, 0, 111, 0, "MODEL"],
            [189, 8, 0, 112, 0, "IMAGE"],
            [208, 6, 0, 129, 0, "CONDITIONING"],
            [209, 7, 0, 129, 1, "CONDITIONING"],
            [210, 39, 0, 129, 2, "VAE"],
            [214, 129, 2, 3, 3, "LATENT"],
            [215, 129, 0, 3, 1, "CONDITIONING"],
            [216, 129, 1, 3, 2, "CONDITIONING"],
            [217, 111, 0, 130, 0, "MODEL"],
            [223, 130, 0, 133, 0, "MODEL"],
            [224, 133, 0, 131, 0, "MODEL"],
            [225, 131, 0, 134, 0, "MODEL"],
            [229, 125, 0, 108, 0, "MODEL"],
            [233, 118, 0, 6, 0, "CLIP"],
            [234, 118, 0, 7, 0, "CLIP"],
            [235, 73, 0, 140, 0, "IMAGE"],
            [236, 140, 0, 129, 4, "IMAGE"],
            [237, 140, 1, 129, 5, "INT"],
            [238, 140, 2, 129, 6, "INT"],
            [240, 140, 0, 75, 0, "IMAGE"],
            [241, 3, 0, 8, 0, "LATENT"],
            [246, 134, 0, 142, 0, "MODEL"],
            [247, 142, 0, 48, 0, "MODEL"],
            [248, 7, 0, 142, 1, "CONDITIONING"]
        ],
        "groups": [
            {"id": 1, "title": "Load models here", "bounding": [-1310, -30, 410, 620], "color": "#b58b2a", "font_size": 24, "flags": {}},
            {"id": 2, "title": "Prompt", "bounding": [140, -20, 450, 610], "color": "#3f789e", "font_size": 24, "flags": {}},
            {"id": 3, "title": "Sampling & Decoding", "bounding": [610, -20, 740, 610], "color": "#3f789e", "font_size": 24, "flags": {}},
            {"id": 4, "title": "Save Video(Mp4)", "bounding": [1370, -10, 790, 1070], "color": "#3f789e", "font_size": 24, "flags": {}},
            {"id": 7, "title": "Load reference image", "bounding": [140, 610, 1210, 450], "color": "#3f789e", "font_size": 24, "flags": {}},
            {"id": 11, "title": "Speed", "bounding": [-880, -30, 340, 300], "color": "#3f789e", "font_size": 24, "flags": {}},
            {"id": 12, "title": "LORAs", "bounding": [-520, -30, 630, 730], "color": "#3f789e", "font_size": 24, "flags": {}}
        ],
        "config": {},
        "extra": {
            "ds": {"scale": 1.11678157794254, "offset": [-685.0725120370064, -300.190254804462]},
            "frontendVersion": "1.18.10",
            "node_versions": {"comfy-core": "0.3.34"},
            "VHS_latentpreview": false,
            "VHS_latentpreviewrate": 0,
            "VHS_MetadataImage": true,
            "VHS_KeepIntermediate": true,
            "ue_links": [],
            "links_added_by_ue": []
        },
        "version": 0.4
    }

    # Save workflow JSON
    workflow_path = save_workflow_json(workflow_data)

    # Run the ComfyUI job
    run_comfyui_job(workflow_path)

if __name__ == "__main__":
    main()