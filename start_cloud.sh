#!/bin/bash
# OpenFork DGN Client Cloud Startup Script (start_cloud.sh)
# This script is fetched and run by cloud containers (RunPod, Vast.ai)

set -e

# Redirect all output to a log file AND stdout so we can see it in cloud logs
LOG_FILE="/tmp/dgn_init.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# --- Helper Functions ---

# Log with timestamp
log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

# Find Python executable with priority
find_python() {
  if [ -x "/usr/bin/python" ]; then
    echo "/usr/bin/python"
  elif command -v python &> /dev/null; then
    command -v python
  elif command -v python3 &> /dev/null; then
    command -v python3
  else
    echo "NOT_FOUND"
  fi
}

# Wait for a URL to return 200 OK
wait_for_url() {
  local name="$1"
  local url="$2"
  local max_wait="${3:-120}"
  local log_file="$4"
  local waited=0
  
  log "Waiting for $name to be ready at $url..."
  while [ $waited -lt $max_wait ]; do
    if curl -s "$url" > /dev/null 2>&1; then
      log "$name is ready!"
      return 0
    fi
    [ $((waited % 10)) -eq 0 ] && log "  Waiting... ($waited/$max_wait seconds)"
    sleep 2
    waited=$((waited + 2))
  done
  
  log "WARNING: $name did not become ready within $max_wait seconds."
  [ -n "$log_file" ] && log "Check $log_file for errors."
  return 1
}

# Ensure pip is installed for the given Python executable
install_pip() {
  local py_exe="$1"
  if ! "$py_exe" -m pip --version &> /dev/null; then
    log "pip not found, attempting to install..."
    if "$py_exe" -m ensurepip --upgrade 2>/dev/null; then
      log "pip installed via ensurepip"
    else
      log "ensurepip failed, downloading get-pip.py..."
      curl -sL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
      "$py_exe" /tmp/get-pip.py --quiet --break-system-packages 2>/dev/null || \
      "$py_exe" /tmp/get-pip.py --quiet
      log "pip installed via get-pip.py"
    fi
  fi
  
  if ! "$py_exe" -m pip --version &> /dev/null; then
    log "ERROR: Failed to install pip!"
    return 1
  fi
  return 0
}

# --- Initialization ---

log "========================================"
log "=== OpenFork DGN Worker Initialization ==="
log "========================================"
log "User: $(whoami)"
log "Path: $PATH"

PYTHON_EXE=$(find_python)
log "Python Executable: $PYTHON_EXE"
"$PYTHON_EXE" --version 2>&1 || true

if [ "$PYTHON_EXE" = "NOT_FOUND" ]; then
  log "ERROR: Python not found! Exiting."
  exit 1
fi

# Verify PyTorch is available (critical for ComfyUI)
if ! "$PYTHON_EXE" -c "import torch" 2>/dev/null; then
  log "WARNING: PyTorch not found in $PYTHON_EXE. ComfyUI will likely fail."
fi

install_pip "$PYTHON_EXE" || exit 1

# Install ffmpeg for thumbnail generation and video duration detection
log "Installing ffmpeg..."
apt-get update -qq 2>/dev/null || true
apt-get install -y -qq ffmpeg 2>/dev/null || \
  (log "apt-get failed, trying alternative..." && \
   apt-get install -y ffmpeg 2>&1 || log "Warning: ffmpeg installation failed")

# Verify ffmpeg installed
if command -v ffmpeg &> /dev/null; then
  log "ffmpeg installed: $(ffmpeg -version 2>&1 | head -1)"
else
  log "WARNING: ffmpeg not available. Thumbnails and duration detection will fail."
fi

# Install critical dependencies
log "Installing base dependencies..."
DEP_LIST="setuptools pyyaml requests python-dotenv websocket-client py-cpuinfo GPUtil psutil transformers Pillow typing_extensions aiohttp einops safetensors scipy tqdm"
"$PYTHON_EXE" -m pip install --quiet --break-system-packages $DEP_LIST 2>/dev/null || \
"$PYTHON_EXE" -m pip install --quiet $DEP_LIST || true

# --- Background Services ---

# Start ComfyUI
if [ -d "/opt/ComfyUI" ]; then
  log "Starting ComfyUI in background..."
  (cd /opt/ComfyUI && "$PYTHON_EXE" main.py --listen > /tmp/comfyui.log 2>&1) &
  log "ComfyUI startup initiated (logging to /tmp/comfyui.log)"
else
  log "Info: /opt/ComfyUI not found."
fi

# Start YUME REST API
if [ -f "/opt/dgn-client/yume_api.py" ]; then
  log "Found YUME API script. Starting..."
  (cd /opt/dgn-client && "$PYTHON_EXE" yume_api.py > /tmp/yume_api.log 2>&1) &
  wait_for_url "YUME API" "http://127.0.0.1:8000/health" 120 "/tmp/yume_api.log"
fi

# Start DiffRhythm REST API
if [ -f "/app/diffrhythm_api.py" ]; then
  log "Found DiffRhythm API script. Starting..."
  (cd /app && "$PYTHON_EXE" diffrhythm_api.py > /tmp/diffrhythm_api.log 2>&1) &
  wait_for_url "DiffRhythm API" "http://127.0.0.1:8000/health" 120 "/tmp/diffrhythm_api.log"
fi

# Start TurboDiffusion REST API
if [ -f "/opt/TurboDiffusion/api_server.py" ]; then
  log "Found TurboDiffusion API script. Starting..."
  (cd /opt/TurboDiffusion && "$PYTHON_EXE" api_server.py > /tmp/turbodiffusion_api.log 2>&1) &
  wait_for_url "TurboDiffusion API" "http://127.0.0.1:8000/health" 120 "/tmp/turbodiffusion_api.log"
fi

# Start Ollama server
if command -v ollama &> /dev/null; then
  log "Found Ollama. Starting server..."
  export OLLAMA_HOST=0.0.0.0
  export OLLAMA_ORIGINS="*"
  ollama serve > /tmp/ollama.log 2>&1 &
  wait_for_url "Ollama" "http://127.0.0.1:11434/api/tags" 60 "/tmp/ollama.log"
fi

# --- Client Setup ---

mkdir -p /opt/dgn-client /data/.cache /data/input

# Download DGN client from GitHub using bootstrap script
cd /opt/dgn-client
log "Downloading DGN client files..."
export INSTALL_DEPS=true
curl -sL https://raw.githubusercontent.com/besch/openfork_client/main/bootstrap.sh -o bootstrap.sh
bash bootstrap.sh

# Wait for ComfyUI to be ready (if it was started)
if [ -d "/opt/ComfyUI" ]; then
  wait_for_url "ComfyUI" "http://127.0.0.1:8188/system_stats" 120 "/tmp/comfyui.log"
fi

# Save restart configuration
log "Saving restart configuration..."
cat > /opt/dgn-client/.restart-config << RESTART_CONFIG_EOF
export DGN_CLIENT_ARGS="--dgn-api-key \"$DGN_API_KEY\" --service \"${SERVICE_TYPE:-auto}\" --accept-policy all --root-dir /opt/dgn-client --data-dir /data"
export ORCHESTRATOR_URL_PROD="${DGN_ORCHESTRATOR_URL:-https://openfork.video}"
RESTART_CONFIG_EOF

chmod +x /opt/dgn-client/.restart-config

# --- Execution ---

log "Starting DGN client..."
export ORCHESTRATOR_URL_PROD="${DGN_ORCHESTRATOR_URL:-https://openfork.video}"

# Test that imports work before running
log "Testing Python imports..."
"$PYTHON_EXE" -c "
import sys
sys.path.insert(0, '/opt/dgn-client')
try:
    from config import HEADLESS_MODE
    from dgn_client import DGNClient
    print('DGNClient import successful')
except Exception as e:
    print(f'Import error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" || (log "ERROR: Python imports failed." && exit 1)

"$PYTHON_EXE" cli.py \
  --dgn-api-key "$DGN_API_KEY" \
  --service "${SERVICE_TYPE:-auto}" \
  --accept-policy all \
  --root-dir /opt/dgn-client \
  --data-dir /data 2>&1 | tee -a /tmp/dgn_client.log
