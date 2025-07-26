
import os

# Get the absolute path of the project's root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Docker-related configurations
DOCKER_IMAGE_NAME = "dgn-client"
DOCKERFILE_PATH = os.path.join(ROOT_DIR, "Dockerfile")

# ComfyUI-related configurations
COMFYUI_PORT = 8188
COMFYUI_URL = f"http://localhost:{COMFYUI_PORT}"

# Orchestrator-related configurations
ORCHESTRATOR_URL = "http://localhost:3000"

# Supabase-related configurations (placeholders - replace with actual values)
SUPABASE_URL = "YOUR_SUPABASE_URL"
SUPABASE_ANON_KEY = "YOUR_SUPABASE_ANON_KEY"

# Cache directory for assets and outputs
CACHE_DIR = os.path.join(ROOT_DIR, "cache")

# Workflow-related configurations
WORKFLOW_FILE_PATH = os.path.join(ROOT_DIR, "workflow.json")


