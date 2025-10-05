'''
Configuration for the DGN Client
'''
import os
from dotenv import load_dotenv
from shared_types import *

load_dotenv()

# --- General Configuration ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT_DIR, '.cache')
DEV_MODE = True

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
    SERVICE_TYPE_WAN22: f"{DOCKER_HUB_USERNAME}/openfork-wan22-rtx4060:latest",
    SERVICE_TYPE_FOLEY: f"{DOCKER_HUB_USERNAME}/openfork-foley-rtx4060:latest",
    SERVICE_TYPE_QWEN: f"{DOCKER_HUB_USERNAME}/openfork-qwen-rtx4060:latest",
    SERVICE_TYPE_VIBEVOICE: f"{DOCKER_HUB_USERNAME}/openfork-vibevoice-rtx4060:latest",
    SERVICE_TYPE_DIFFRHYTHM: f"{DOCKER_HUB_USERNAME}/openfork-diffrhythm-rtx4060:latest",
}