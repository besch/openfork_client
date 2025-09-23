import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file

# Get the absolute path of the project's root directory
def get_root_dir():
    """Determines the root directory for the application, handling both script and frozen exe."""
    if getattr(sys, 'frozen', False):
        # Running as a PyInstaller bundle
        executable_path = os.path.dirname(sys.executable)
        # Check if we are in the electron app's bin directory
        if 'dgn_client_desktop' in executable_path:
            return os.path.abspath(os.path.join(executable_path, '..', '..', 'dgn-client'))
        return executable_path
    else:
        # Running as a normal script
        return os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = get_root_dir()


load_dotenv(os.path.join(ROOT_DIR, '.env'))

# Orchestrator-related configurations
ORCHESTRATOR_URL_PROD = "https://www.openfork.video"
ORCHESTRATOR_URL_DEV = "http://localhost:3000"

SUPABASE_URL = "https://vmuylzvwqravkmdmcpgv.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZtdXlsenZ3cXJhdmttZG1jcGd2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTIxNDM3MjAsImV4cCI6MjA2NzcxOTcyMH0.f2USQOkuKhPksSLSXhTlyl5zTstyCyYvzdiHV9HQUKw"



# Cache directory for assets and outputs
CACHE_DIR = os.path.join(ROOT_DIR, "cache")

# Development mode switch
DEV_MODE = False # Set to True to use placeholder video instead of ComfyUI

# Docker-related configurations
DOCKER_COMPOSE_DIR = os.path.join(ROOT_DIR, "comfyui-storage")
