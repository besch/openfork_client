'''
Configuration for the DGN Client
'''
import os
import sys
from dotenv import load_dotenv


load_dotenv()

# --- General Configuration ---
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle (e.g., by PyInstaller)
    ROOT_DIR = os.path.dirname(sys.executable)
else:
    # If running as a script in a normal Python environment
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(ROOT_DIR)

CACHE_DIR = os.path.join(ROOT_DIR, '.cache')
DEV_MODE = False

# --- Supabase Configuration ---
SUPABASE_URL = "https://vmuylzvwqravkmdmcpgv.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZtdXlsenZ3cXJhdmttZG1jcGd2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIxNDM3MjAsImV4cCI6MjA2NzcxOTcyMH0.f2USQOkuKhPksSLSXhTlyl5zTstyCyYvzdiHV9HQUKw"

# --- Orchestrator Configuration ---
ORCHESTRATOR_URL_PROD = os.getenv("ORCHESTRATOR_URL_PROD", "https://www.openfork.video")
ORCHESTRATOR_URL_DEV = os.getenv("ORCHESTRATOR_URL_DEV", "http://localhost:3000")

# --- Docker Image Configuration ---
# Maps a service type to a full Docker Hub image name.
DOCKER_HUB_USERNAME = "beschiak"

WORKFLOW_CONFIG = {
    "WAN22_TEXT_TO_VIDEO": {
        "service_name": "wan22",
        "workflow_file": "wan22-text-to-video.api.json",
        "docker_image_name": "openfork-wan22-rtx4060",
        "processor": "WAN22TextToVideoJobProcessor"
    },
    "WAN22_IMAGE_TO_VIDEO": {
        "service_name": "wan22",
        "workflow_file": "wan22-image-to-video.api.json",
        "docker_image_name": "openfork-wan22-rtx4060",
        "processor": "WAN22ImageToVideoJobProcessor"
    },
    "WAN22_LIGHTNING_TEXT_TO_VIDEO": {
        "service_name": "wan22-lightning",
        "workflow_file": "wan22-text-to-video-lightning.api.json",
        "docker_image_name": "openfork-wan22-lightning-rtx4060",
        "processor": "TextToVideoLightningJobProcessor"
    },
    "WAN22_LIGHTNING_IMAGE_TO_VIDEO": {
        "service_name": "wan22-lightning",
        "workflow_file": "wan22-image-to-video-lightning.api.json",
        "docker_image_name": "openfork-wan22-lightning-rtx4060",
        "processor": "ImageToVideoLightningJobProcessor"
    },
    "HUNYUAN_VIDEO_FOLEY": {
        "service_name": "foley",
        "workflow_file": "hunyuan-video-foley.api.json",
        "docker_image_name": "openfork-foley-rtx4060",
        "processor": "FoleyJobProcessor"
    },
    "QWEN_TEXT_TO_IMAGE": {
        "service_name": "qwen",
        "workflow_file": "qwen.api.json",
        "docker_image_name": "openfork-qwen-rtx4060",
        "processor": "TextToImageJobProcessor"
    },
    "VIBEVOICE_TTS": {
        "service_name": "vibevoice",
        "workflow_file": "vibevoice.api.json",
        "docker_image_name": "openfork-vibevoice-rtx4060",
        "processor": "VibeVoiceJobProcessor"
    },
    "VIBEVOICE_TTS_MULTI_CLONE": {
        "service_name": "vibevoice",
        "workflow_file": "vibevoice-multi-speaker-clone.api.json",
        "docker_image_name": "openfork-vibevoice-rtx4060",
        "processor": "VibeVoiceMultiCloneJobProcessor"
    },
    "DIFFRHYTHM_MUSIC_GENERATION": {
        "service_name": "diffrhythm",
        "workflow_file": "diffrhythm.api.json",
        "docker_image_name": "openfork-diffrhythm-rtx4060",
        "processor": "DiffRhythmJobProcessor"
    },
    "ESRGAN_UPSCALER": {
        "service_name": "esrgan-upscaler",
        "workflow_file": "esrgan-video-upscale.api.json",
        "docker_image_name": "openfork-realesrgan-upscaler-rtx4060",
        "processor": "VideoUpscalerJobProcessor"
    }
}

# Dynamically create DOCKER_IMAGE_MAP from WORKFLOW_CONFIG
DOCKER_IMAGE_MAP = {
    config["service_name"]: f"{DOCKER_HUB_USERNAME}/{config['docker_image_name']}:latest"
    for config in WORKFLOW_CONFIG.values()
}