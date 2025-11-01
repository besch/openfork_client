import os
import time
import logging
import subprocess
import requests

# Paths (inside container)
COMFY_ROOT = "/app/ComfyUI"
MANAGER_CLI = os.path.join(COMFY_ROOT, "custom_nodes", "ComfyUI-Manager", "cli", "main.py")
MODEL_DIR = os.path.join(COMFY_ROOT, "models")
TEMP_DL = os.path.join(MODEL_DIR, "_temp")
os.makedirs(TEMP_DL, exist_ok=True)

# ------------------------------------------------------- 
# Helpers
# ------------------------------------------------------- 
def comfy_refresh_nodes():
    try:
        requests.post("http://127.0.0.1:8188/refresh_nodes", timeout=5)
    except Exception:
        logging.debug("Could not request /refresh_nodes")

def comfy_refresh_models():
    try:
        requests.post("http://127.0.0.1:8188/refresh_models", timeout=5)
    except Exception:
        logging.debug("Could not request /refresh_models")

# ------------------------------------------------------- 
# Install via ComfyUI-Manager CLI
# ------------------------------------------------------- 
def manager_install_custom_node(repo_url: str):
    if not os.path.exists(MANAGER_CLI):
        logging.warning(f"Manager CLI missing: {MANAGER_CLI}")
        return False
    try:
        result = subprocess.run(
            ["python3", MANAGER_CLI, "--install-custom-node", repo_url],
            check=True, capture_output=True, text=True, timeout=300
        )
        logging.info(f"Installed {repo_url}\n{result.stdout}")
        time.sleep(8)  # CRITICAL: Restart + load nodes
        comfy_refresh_nodes()
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Install failed: {e.stderr}")
        return False

def manager_install_model(url: str):
    if not os.path.exists(MANAGER_CLI):
        return False
    try:
        subprocess.run(["python3", MANAGER_CLI, "--install-model", url], check=True)
        time.sleep(1)
        comfy_refresh_models()
        return True
    except subprocess.CalledProcessError as e:
        logging.warning(f"Manager CLI model install failed for {url}: {e}")
        return False

# ------------------------------------------------------- 
# Top-level orchestration: given a workflow template dict,
# install nodes & models (safe, idempotent).
# ------------------------------------------------------- 
def auto_install_all(template: dict):
    installed_nodes = []
    installed_models = []

    # 1. NODES: EXPLICIT URLs (from sync!)
    for dep in template.get('custom_node_dependencies', []):
        url = dep.get('url')
        if url and manager_install_custom_node(url):
            installed_nodes.append(url)

    # 2. HARD-MAPS (backup for undocumented)
    # This part requires a get_missing_nodes function, which is not defined yet.
    # For now, I will skip this part and assume all nodes are explicitly defined or handled by manager_install_custom_node.
    # If get_missing_nodes is needed, it would involve querying ComfyUI for installed nodes and comparing.
    # The user's prompt implies that the custom_node_dependencies from workflow_sync should be sufficient.

    # 3. MODELS: MANAGER CLI (handles ALL!)
    for url in template.get('model_urls', []):
        if manager_install_model(url):  # CivitAI/HF/GH/direct → MAGIC!
            installed_models.append(url)

    comfy_refresh_nodes()
    comfy_refresh_models()
    return installed_nodes, installed_models