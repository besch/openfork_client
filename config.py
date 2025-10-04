'''
Configuration for the DGN Client
'''
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- General Configuration ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT_DIR, '.cache')
DEV_MODE = os.getenv('DEV_MODE', 'False').lower() in ('true', '1', 't')

# --- Supabase Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

# --- Orchestrator Configuration ---
ORCHESTRATOR_URL_PROD = os.getenv("ORCHESTRATOR_URL_PROD", "https://your-prod-url.com")
ORCHESTRATOR_URL_DEV = os.getenv("ORCHESTRATOR_URL_DEV", "http://localhost:3000")

# --- Docker Image Configuration ---
# Maps a service type to a full Docker Hub image name.
# Replace 'yourusername' with your actual Docker Hub username.
DOCKER_HUB_USERNAME = os.getenv("DOCKER_HUB_USERNAME", "yourusername")

DOCKER_IMAGE_MAP = {
    "default": f"{DOCKER_HUB_USERNAME}/crowdmovie-default:latest",
    "foley": f"{DOCKER_HUB_USERNAME}/crowdmovie-foley:latest",
    "text_to_image": f"{DOCKER_HUB_USERNAME}/crowdmovie-qwen:latest", # Corresponds to qwen workflow
    "vibevoice": f"{DOCKER_HUB_USERNAME}/crowdmovie-vibevoice:latest",
    "diffrhythm": f"{DOCKER_HUB_USERNAME}/crowdmovie-diffrhythm:latest",
    # Add other mappings here as you create more images
}