
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the absolute path of the project's root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Orchestrator-related configurations
ORCHESTRATOR_URL = "http://localhost:3000"

# Supabase-related configurations (placeholders - replace with actual values)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

# Cache directory for assets and outputs
CACHE_DIR = os.path.join(ROOT_DIR, "cache")
