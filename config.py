
import os
from dotenv import load_dotenv

# Load environment variables from .env file

# Get the absolute path of the project's root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(ROOT_DIR, '.env'))

# Orchestrator-related configurations
PRIMARY_ORCHESTRATOR_URL = "https://crowdmovie.vercel.app/"
FALLBACK_ORCHESTRATOR_URL = "http://localhost:3000"



# Cache directory for assets and outputs
CACHE_DIR = os.path.join(ROOT_DIR, "cache")
