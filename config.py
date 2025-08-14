import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file

# Get the absolute path of the project's root directory
def get_root_dir():
    """Determines the root directory for the application, handling both script and frozen exe."""
    if getattr(sys, 'frozen', False):
        # Running as a PyInstaller bundle
        return os.path.dirname(sys.executable)
    else:
        # Running as a normal script
        return os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = get_root_dir()


load_dotenv(os.path.join(ROOT_DIR, '.env'))

# Orchestrator-related configurations
PRIMARY_ORCHESTRATOR_URL = "https://crowdmovie.vercel.app/"
FALLBACK_ORCHESTRATOR_URL = "http://localhost:3000"



# Cache directory for assets and outputs
CACHE_DIR = os.path.join(ROOT_DIR, "cache")