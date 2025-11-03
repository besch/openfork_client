"""
Improved dependency installer using ComfyUI Manager's cm-cli for reliable installations.
This approach is more robust than manual git cloning.
"""
import logging
import subprocess
import requests
from services.docker_manager import docker_manager

# Known repository redirects/replacements
REPO_REDIRECTS = {
    "https://github.com/ltdrdata/ComfyUI-Documentation-Nodes": None,  # Deprecated/removed
    "https://github.com/ltdrdata/ComfyUI-Documentation-Nodes.git": None,
}

def validate_github_repo(repo_url: str) -> bool:
    """
    Validates that a GitHub repository exists by checking the API.
    """
    try:
        parts = repo_url.rstrip('/').replace('.git', '').split('github.com/')
        if len(parts) < 2:
            return False
        
        owner_repo = parts[1].strip('/')
        api_url = f"https://api.github.com/repos/{owner_repo}"
        response = requests.head(api_url, timeout=10)
        
        if response.status_code == 200:
            return True
        elif response.status_code == 404:
            logging.warning(f"Repository not found: {repo_url}")
            return False
        else:
            logging.warning(f"Could not validate {repo_url} (status {response.status_code}), assuming it exists")
            return True
            
    except Exception as e:
        logging.warning(f"Error validating repository {repo_url}: {e}")
        return True


def get_node_name_from_url(repo_url: str) -> str:
    """Extract the repository name from a GitHub URL."""
    return repo_url.rstrip('/').split('/')[-1].replace('.git', '')


def is_custom_node_installed(node_name: str) -> bool:
    """
    Check if a custom node is installed using cm-cli.
    """
    container_name = docker_manager.get_container_name()
    if not container_name:
        return False
    
    check_script = f"""
cd /app/ComfyUI/custom_nodes/ComfyUI-Manager
python3 cm-cli.py simple-show installed 2>/dev/null | grep -q "{node_name}"
"""
    
    cmd = ["docker", "exec", container_name, "bash", "-c", check_script]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0
    except Exception as e:
        logging.debug(f"Error checking if {node_name} is installed: {e}")
        return False


def manager_install_custom_node_via_cli(repo_url: str) -> bool:
    """
    Install a custom node using ComfyUI Manager's cm-cli tool.
    This is the most reliable method as it uses the same installation
    logic as the ComfyUI Manager UI.
    """
    normalized_url = repo_url.rstrip('/')
    if normalized_url in REPO_REDIRECTS:
        replacement = REPO_REDIRECTS[normalized_url]
        if replacement is None:
            logging.warning(f"[INSTALL] Repository {repo_url} is deprecated/removed. Skipping.")
            return True
        else:
            logging.info(f"[INSTALL] Redirecting {repo_url} to {replacement}")
            repo_url = replacement
    
    if not validate_github_repo(repo_url):
        logging.error(f"[INSTALL] Repository does not exist: {repo_url}")
        return True  # Don't fail the job
    
    container_name = docker_manager.get_container_name()
    if not container_name:
        logging.error("No active ComfyUI container found for node installation")
        return False
    
    repo_name = get_node_name_from_url(repo_url)
    
    # Check if already installed
    if is_custom_node_installed(repo_name):
        logging.info(f"[INSTALL] [OK] {repo_name} is already installed")
        return True
    
    logging.info(f"[INSTALL] Installing custom node via cm-cli: {repo_name}")
    logging.info(f"[INSTALL]   Repository: {repo_url}")
    
    # Use cm-cli to install the node
    install_script = f"""
set -e
cd /app/ComfyUI/custom_nodes/ComfyUI-Manager

echo "[INSTALL] Installing {repo_name} using cm-cli..."
python3 cm-cli.py install "{repo_url}" --mode remote 2>&1 | while IFS= read -r line; do
    echo "[INSTALL]   $line"
done

# Verify installation
if python3 cm-cli.py simple-show installed 2>/dev/null | grep -q "{repo_name}"; then
    echo "[INSTALL] [OK] {repo_name} installed successfully"
    exit 0
else
    echo "[INSTALL] [ERROR] {repo_name} installation verification failed"
    exit 1
fi
"""
    
    cmd = ["docker", "exec", container_name, "bash", "-c", install_script]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Stream output in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                logging.info(line.rstrip())
        
        return_code = process.wait(timeout=600)
        
        if return_code == 0:
            logging.info(f"[INSTALL] [OK] Successfully installed {repo_name}")
            return True
        else:
            logging.error(f"[INSTALL] [ERROR] Failed to install {repo_name} (exit code: {return_code})")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"[INSTALL] [ERROR] Timeout installing {repo_name}")
        process.kill()
        return False
    except Exception as e:
        logging.error(f"[INSTALL] [ERROR] Exception installing {repo_name}: {e}", exc_info=True)
        return False


def fix_all_custom_node_dependencies() -> bool:
    """
    Run cm-cli fix to reinstall all dependencies for installed custom nodes.
    This is useful after container restarts or when nodes fail to load.
    """
    container_name = docker_manager.get_container_name()
    if not container_name:
        logging.error("No active ComfyUI container found")
        return False
    
    logging.info("[FIX] Running cm-cli fix to ensure all dependencies are installed...")
    
    fix_script = """
cd /app/ComfyUI/custom_nodes/ComfyUI-Manager
python3 cm-cli.py fix all 2>&1 | while IFS= read -r line; do
    echo "[FIX]   $line"
done
"""
    
    cmd = ["docker", "exec", container_name, "bash", "-c", fix_script]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            encoding='utf-8'
        )
        
        if result.stdout:
            for line in result.stdout.split('\n'):
                if line.strip():
                    logging.info(line)
        
        if result.returncode == 0:
            logging.info("[FIX] [OK] Successfully fixed all custom node dependencies")
            return True
        else:
            logging.warning(f"[FIX] [WARNING] Fix command completed with code {result.returncode}")
            return True  # Don't fail - fix might report issues but still work
            
    except Exception as e:
        logging.error(f"[FIX] [ERROR] Exception running fix: {e}", exc_info=True)
        return False


def manager_install_model(model_url: str) -> bool:
    """
    Install a model inside the running ComfyUI container.
    """
    container_name = docker_manager.get_container_name()
    
    if not container_name:
        logging.error("No active ComfyUI container found for model installation")
        return False
    
    import urllib.parse
    filename = urllib.parse.urlparse(model_url).path.split('/')[-1]
    
    logging.info(f"[DOWNLOAD] Downloading model: {filename}")
    logging.info(f"[DOWNLOAD]   URL: {model_url}")
    
    download_script = f"""
set -e
MODEL_URL='{model_url}'
FILENAME='{filename}'

# Determine target directory
if [[ "$FILENAME" == *"lora"* ]] || [[ "$FILENAME" == *"LoRA"* ]]; then
    TARGET_DIR="/app/ComfyUI/models/loras"
elif [[ "$FILENAME" == *"vae"* ]] || [[ "$FILENAME" == *"VAE"* ]]; then
    TARGET_DIR="/app/ComfyUI/models/vae"
elif [[ "$FILENAME" == *"controlnet"* ]] || [[ "$FILENAME" == *"ControlNet"* ]]; then
    TARGET_DIR="/app/ComfyUI/models/controlnet"
else
    TARGET_DIR="/app/ComfyUI/models/checkpoints"
fi

mkdir -p "$TARGET_DIR"
TARGET_PATH="$TARGET_DIR/$FILENAME"

if [ -f "$TARGET_PATH" ]; then
    FILE_SIZE=$(du -h "$TARGET_PATH" | cut -f1)
    echo "[DOWNLOAD]   [OK] Model already exists ($FILE_SIZE)"
    exit 0
fi

echo "[DOWNLOAD]   Starting download..."

# Try aria2c first (faster)
if command -v aria2c &> /dev/null; then
    aria2c -x 16 -s 16 -k 1M --console-log-level=warn --summary-interval=2 \
        -d "$TARGET_DIR" -o "$FILENAME" "$MODEL_URL" 2>&1 | \
        grep -E "%(|Download complete)" | while read line; do echo "[DOWNLOAD]     $line"; done
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        FILE_SIZE=$(du -h "$TARGET_PATH" | cut -f1)
        echo "[DOWNLOAD]   [OK] Download complete ($FILE_SIZE)"
        exit 0
    fi
fi

# Fallback to wget
wget --progress=bar:force -O "$TARGET_PATH" "$MODEL_URL" 2>&1 | \
    grep "%" | while read line; do echo "[DOWNLOAD]     $line"; done

if [ $? -eq 0 ]; then
    FILE_SIZE=$(du -h "$TARGET_PATH" | cut -f1)
    echo "[DOWNLOAD]   [OK] Download complete ($FILE_SIZE)"
else
    echo "[DOWNLOAD]   [ERROR] Download failed"
    exit 1
fi
"""
    
    cmd = ["docker", "exec", container_name, "bash", "-c", download_script]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        for line in iter(process.stdout.readline, ''):
            if line:
                logging.info(line.rstrip())
        
        return_code = process.wait(timeout=1800)
        
        if return_code == 0:
            logging.info(f"[DOWNLOAD] [OK] Successfully downloaded {filename}")
            return True
        else:
            logging.error(f"[DOWNLOAD] [ERROR] Failed to download {filename}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"[DOWNLOAD] [ERROR] Timeout downloading {filename}")
        process.kill()
        return False
    except Exception as e:
        logging.error(f"[DOWNLOAD] [ERROR] Exception: {e}", exc_info=True)
        return False