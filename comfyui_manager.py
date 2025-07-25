
import requests
import json
from config import COMFYUI_URL, WORKFLOW_FILE_PATH

def trigger_workflow():
    """Triggers the ComfyUI workflow."""
    with open(WORKFLOW_FILE_PATH, 'r') as f:
        workflow = json.load(f)

    response = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow})

    if response.status_code == 200:
        print("Workflow triggered successfully.")
        return response.json()
    else:
        print(f"Error triggering workflow: {response.text}")
        return None
