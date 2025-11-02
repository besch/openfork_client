"""
Dynamic dependency installer for ComfyUI nodes running in Docker.
Executes installation commands inside the container with real-time progress logging.
"""
import logging
import subprocess
import threading
import time
from services.docker_manager import docker_manager

def _stream_output(process, prefix=""):
    """Stream subprocess output in real-time to logging."""
    def stream_stdout():
        for line in iter(process.stdout.readline, b''):
            if line:
                decoded = line.decode('utf-8', errors='replace').rstrip()
                if decoded:
                    logging.info(f"{prefix}{decoded}")
    
    def stream_stderr():
        for line in iter(process.stderr.readline, b''):
            if line:
                decoded = line.decode('utf-8', errors='replace').rstrip()
                if decoded:
                    logging.warning(f"{prefix}{decoded}")
    
    stdout_thread = threading.Thread(target=stream_stdout, daemon=True)
    stderr_thread = threading.Thread(target=stream_stderr, daemon=True)
    
    stdout_thread.start()
    stderr_thread.start()
    
    return stdout_thread, stderr_thread


def manager_install_custom_node(repo_url: str) -> bool:
    """
    Installs a custom node inside the running ComfyUI container with progress logging.
    
    Args:
        repo_url: Git repository URL of the custom node
        
    Returns:
        True if installation succeeded, False otherwise
    """
    container_name = docker_manager.get_container_name()
    
    if not container_name:
        logging.error("No active ComfyUI container found for node installation")
        return False
    
    # Extract repo name for logging
    repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
    
    logging.info(f"[INSTALL] Installing custom node: {repo_name}")
    logging.info(f"[INSTALL]   Repository: {repo_url}")
    
    # Ensure repo URL ends with .git
    if not repo_url.endswith('.git'):
        repo_url = f"{repo_url}.git"
    
    # Use bash script with fallback to zip download
    install_script = f"""
set -e
set -o pipefail

REPO_URL='{repo_url}'
REPO_NAME='{repo_name}'
TARGET_PATH="/app/ComfyUI/custom_nodes/$REPO_NAME"

echo "[INSTALL]   Checking if already installed..."
if [ -d "$TARGET_PATH" ]; then
    echo "[INSTALL]   [OK] Repository already exists at $TARGET_PATH"
    exit 0
fi

echo "[INSTALL]   Attempting repository installation..."

# Method 1: Try git clone with sanitized environment
echo "[INSTALL]   Method 1: Trying git clone..."
git_success=false

(
    # Create completely clean environment
    cd /tmp
    unset GIT_ASKPASS SSH_ASKPASS
    export GIT_TERMINAL_PROMPT=0
    export HOME=/tmp
    
    # Clone with no config files
    if GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null git clone --depth 1 "$REPO_URL" "/tmp/$REPO_NAME" 2>&1; then
        mv "/tmp/$REPO_NAME" "$TARGET_PATH"
        echo "[INSTALL]   [OK] Git clone succeeded"
        exit 0
    else
        exit 1
    fi
) && git_success=true || git_success=false

# Method 2: Fallback to zip download if git fails
if [ "$git_success" = "false" ]; then
    echo "[INSTALL]   Method 2: Falling back to zip download..."
    
    # Extract owner and repo from URL
    REPO_PATH=$(echo "$REPO_URL" | sed 's|https://github.com/||' | sed 's|.git$||')
    
    # Try main branch first
    ZIP_URL="https://github.com/$REPO_PATH/archive/refs/heads/main.zip"
    echo "[INSTALL]   Downloading: $ZIP_URL"
    
    if wget -q -O /tmp/repo.zip "$ZIP_URL" 2>&1; then
        echo "[INSTALL]   Extracting archive..."
        unzip -q /tmp/repo.zip -d /tmp/ 2>&1
        
        # Find extracted directory (usually repo-main)
        EXTRACTED_DIR=$(find /tmp -maxdepth 1 -type d -name "*-main" -o -name "$REPO_NAME-main" | head -n 1)
        
        if [ -n "$EXTRACTED_DIR" ] && [ -d "$EXTRACTED_DIR" ]; then
            mv "$EXTRACTED_DIR" "$TARGET_PATH"
            rm -f /tmp/repo.zip
            echo "[INSTALL]   [OK] Downloaded and extracted successfully (main branch)"
        else
            # Try master branch as fallback
            rm -f /tmp/repo.zip
            ZIP_URL="https://github.com/$REPO_PATH/archive/refs/heads/master.zip"
            echo "[INSTALL]   Trying master branch: $ZIP_URL"
            
            if wget -q -O /tmp/repo.zip "$ZIP_URL" 2>&1; then
                echo "[INSTALL]   Extracting archive..."
                unzip -q /tmp/repo.zip -d /tmp/ 2>&1
                
                EXTRACTED_DIR=$(find /tmp -maxdepth 1 -type d -name "*-master" -o -name "$REPO_NAME-master" | head -n 1)
                
                if [ -n "$EXTRACTED_DIR" ] && [ -d "$EXTRACTED_DIR" ]; then
                    mv "$EXTRACTED_DIR" "$TARGET_PATH"
                    rm -f /tmp/repo.zip
                    echo "[INSTALL]   [OK] Downloaded and extracted successfully (master branch)"
                else
                    echo "[INSTALL]   [ERROR] Could not find extracted directory"
                    rm -f /tmp/repo.zip
                    exit 1
                fi
            else
                echo "[INSTALL]   [ERROR] Failed to download repository from both main and master branches"
                exit 1
            fi
        fi
    else
        echo "[INSTALL]   [ERROR] Failed to download from main branch, trying master..."
        ZIP_URL="https://github.com/$REPO_PATH/archive/refs/heads/master.zip"
        
        if wget -q -O /tmp/repo.zip "$ZIP_URL" 2>&1; then
            echo "[INSTALL]   Extracting archive..."
            unzip -q /tmp/repo.zip -d /tmp/ 2>&1
            
            EXTRACTED_DIR=$(find /tmp -maxdepth 1 -type d -name "*-master" -o -name "$REPO_NAME-master" | head -n 1)
            
            if [ -n "$EXTRACTED_DIR" ] && [ -d "$EXTRACTED_DIR" ]; then
                mv "$EXTRACTED_DIR" "$TARGET_PATH"
                rm -f /tmp/repo.zip
                echo "[INSTALL]   [OK] Downloaded and extracted successfully (master branch)"
            else
                echo "[INSTALL]   [ERROR] Could not find extracted directory"
                rm -f /tmp/repo.zip
                exit 1
            fi
        else
            echo "[INSTALL]   [ERROR] Failed to download repository"
            exit 1
        fi
    fi
fi

if [ ! -d "$TARGET_PATH" ]; then
    echo "[INSTALL]   [ERROR] Installation failed - directory not created"
    exit 1
fi

echo "[INSTALL]   [OK] Repository installed successfully"

# Check for requirements
REQ_FILE="$TARGET_PATH/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    echo "[INSTALL]   Found requirements.txt, installing dependencies..."
    pip3 install -q -r "$REQ_FILE" 2>&1 | while IFS= read -r line; do
        if [[ "$line" == *"Requirement already satisfied"* ]]; then
            : # Skip these to reduce noise
        elif [[ "$line" == *"Successfully installed"* ]] || [[ "$line" == *"Collecting"* ]] || [[ "$line" == *"Downloading"* ]]; then
            echo "[INSTALL]     $line"
        fi
    done
    echo "[INSTALL]   [OK] Dependencies installed"
else
    echo "[INSTALL]   No requirements.txt found, skipping dependency installation"
fi

echo "[INSTALL]   [OK] Installation complete"
"""
    
    cmd = [
        "docker", "exec", container_name,
        "bash", "-c", install_script
    ]
    
    try:
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1
        )
        
        # Stream output in real-time
        stdout_thread, stderr_thread = _stream_output(process)
        
        # Wait for completion
        return_code = process.wait(timeout=600)
        
        # Wait for output threads to finish
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        
        if return_code == 0:
            logging.info(f"[INSTALL] [OK] Successfully installed {repo_name}")
            return True
        else:
            logging.error(f"[INSTALL] [ERROR] Failed to install {repo_name} (exit code: {return_code})")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"[INSTALL] [ERROR] Timeout installing {repo_name} (exceeded 10 minutes)")
        process.kill()
        return False
    except Exception as e:
        logging.error(f"[INSTALL] [ERROR] Exception installing {repo_name}: {e}", exc_info=True)
        return False


def manager_install_model(model_url: str) -> bool:
    """
    Installs a model inside the running ComfyUI container with progress logging.
    
    Args:
        model_url: URL to download the model from
        
    Returns:
        True if installation succeeded, False otherwise
    """
    container_name = docker_manager.get_container_name()
    
    if not container_name:
        logging.error("No active ComfyUI container found for model installation")
        return False
    
    # Extract filename for logging
    import urllib.parse
    filename = urllib.parse.urlparse(model_url).path.split('/')[-1]
    
    logging.info(f"[DOWNLOAD] Downloading model: {filename}")
    logging.info(f"[DOWNLOAD]   URL: {model_url}")
    
    # Bash script with progress bars
    download_script = f"""
set -e
set -o pipefail

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

echo "[DOWNLOAD]   Target: $TARGET_PATH"

if [ -f "$TARGET_PATH" ]; then
    FILE_SIZE=$(du -h "$TARGET_PATH" | cut -f1)
    echo "[DOWNLOAD]   [OK] Model already exists ($FILE_SIZE)"
    exit 0
fi

echo "[DOWNLOAD]   Starting download..."

# Try aria2c first (much faster, shows progress)
if command -v aria2c &> /dev/null; then
    echo "[DOWNLOAD]   Using aria2c (multi-connection download)"
    aria2c -x 16 -s 16 -k 1M \
        --console-log-level=warn \
        --summary-interval=2 \
        -d "$TARGET_DIR" -o "$FILENAME" \
        "$MODEL_URL" 2>&1 | while IFS= read -r line; do
        # Filter for progress lines only
        if [[ "$line" == *"("*"%)"* ]] || [[ "$line" == *"Download complete"* ]]; then
            echo "[DOWNLOAD]     $line"
        fi
    done
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        FILE_SIZE=$(du -h "$TARGET_PATH" | cut -f1)
        echo "[DOWNLOAD]   [OK] Download complete ($FILE_SIZE)"
        exit 0
    fi
fi

# Fallback to wget with progress bar
echo "[DOWNLOAD]   Using wget (fallback)"
wget --progress=bar:force \
    -O "$TARGET_PATH" \
    "$MODEL_URL" 2>&1 | while IFS= read -r line; do
    if [[ "$line" == *"%"* ]]; then
        echo "[DOWNLOAD]     $line"
    fi
done

if [ $? -eq 0 ]; then
    FILE_SIZE=$(du -h "$TARGET_PATH" | cut -f1)
    echo "[DOWNLOAD]   [OK] Download complete ($FILE_SIZE)"
    exit 0
else
    echo "[DOWNLOAD]   [ERROR] Download failed"
    exit 1
fi
"""
    
    cmd = [
        "docker", "exec", container_name,
        "bash", "-c", download_script
    ]
    
    try:
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1
        )
        
        # Stream output in real-time
        stdout_thread, stderr_thread = _stream_output(process)
        
        # Wait for completion (30 minutes for large models)
        return_code = process.wait(timeout=1800)
        
        # Wait for output threads to finish
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        
        if return_code == 0:
            logging.info(f"[DOWNLOAD] [OK] Successfully downloaded {filename}")
            return True
        else:
            logging.error(f"[DOWNLOAD] [ERROR] Failed to download {filename} (exit code: {return_code})")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"[DOWNLOAD] [ERROR] Timeout downloading {filename} (exceeded 30 minutes)")
        process.kill()
        return False
    except Exception as e:
        logging.error(f"[DOWNLOAD] [ERROR] Exception downloading {filename}: {e}", exc_info=True)
        return False