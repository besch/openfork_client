
import os

# Get the absolute path of the project's root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Docker-related configurations
DOCKER_IMAGE_NAME = "dgn-client"
DOCKERFILE_PATH = os.path.join(ROOT_DIR, "Dockerfile")

# ComfyUI-related configurations
COMFYUI_PORT = 8188
COMFYUI_URL = f"http://localhost:{COMFYUI_PORT}"

# Workflow-related configurations
WORKFLOW_FILE_PATH = os.path.join(ROOT_DIR, "workflow.json")
