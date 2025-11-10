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

DOCKER_IMAGE_MAP = {
    "WAN22": f"{DOCKER_HUB_USERNAME}/openfork-wan22-rtx4060:latest",
    "FOLEY": f"{DOCKER_HUB_USERNAME}/openfork-foley-rtx4060:latest",
    "QWEN": f"{DOCKER_HUB_USERNAME}/openfork-qwen-rtx4060:latest",
    "VIBEVOICE": f"{DOCKER_HUB_USERNAME}/openfork-vibevoice-rtx4060:latest",
    "DIFFRHYTHM": f"{DOCKER_HUB_USERNAME}/openfork-diffrhythm-rtx4060:latest",
    "WAN22_LIGHTNING": f"{DOCKER_HUB_USERNAME}/openfork-wan22-lightning-rtx4060:latest",
    "ESRGAN_UPSCALER": f"{DOCKER_HUB_USERNAME}/openfork-realesrgan-upscaler-rtx4060:latest",
}