#!/bin/bash
# OpenFork DGN Client Cloud Startup Script (start_cloud.sh)
# This script is fetched and run by cloud containers (RunPod, Vast.ai)

set -e

# Redirect all output to a log file AND stdout so we can see it in cloud logs
LOG_FILE="/tmp/dgn_init.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo "=== OpenFork DGN Worker Initialization ==="
echo "========================================"
echo "Timestamp: $(date)"
echo "User: $(whoami)"
echo "Path: $PATH"

# Python detection priority:
# 1. /usr/bin/python (Docker images symlink this to Python 3.11 with PyTorch)
# 2. python (general fallback)
# 3. python3 (system Python, may not have PyTorch)
if [ -x "/usr/bin/python" ]; then
  PYTHON_EXE="/usr/bin/python"
elif command -v python &> /dev/null; then
  PYTHON_EXE=$(command -v python)
elif command -v python3 &> /dev/null; then
  PYTHON_EXE=$(command -v python3)
else
  PYTHON_EXE="NOT_FOUND"
fi

echo "Python Executable: $PYTHON_EXE"
$PYTHON_EXE --version 2>&1 || true

if [ "$PYTHON_EXE" = "NOT_FOUND" ]; then
  echo "ERROR: Python not found! Exiting."
  exit 1
fi

# Verify PyTorch is available (critical for ComfyUI)
if ! $PYTHON_EXE -c "import torch" 2>/dev/null; then
  echo "WARNING: PyTorch not found in $PYTHON_EXE"
  echo "ComfyUI will likely fail to start. Check Docker image installation."
fi

# Ensure pip is installed
echo "Checking for pip..."
if ! $PYTHON_EXE -m pip --version &> /dev/null; then
  echo "pip not found, attempting to install..."
  
  # Try ensurepip first (built into Python 3.4+)
  if $PYTHON_EXE -m ensurepip --upgrade 2>/dev/null; then
    echo "pip installed via ensurepip"
  else
    # Fall back to get-pip.py (use --break-system-packages for PEP 668 / Ubuntu 24+)
    echo "ensurepip failed, downloading get-pip.py..."
    curl -sL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    $PYTHON_EXE /tmp/get-pip.py --quiet --break-system-packages 2>/dev/null || \
    $PYTHON_EXE /tmp/get-pip.py --quiet
    echo "pip installed via get-pip.py"
  fi
fi

# Verify pip works
if $PYTHON_EXE -m pip --version; then
  echo "pip is ready"
else
  echo "ERROR: Failed to install pip! Exiting."
  exit 1
fi

# Install ffmpeg for thumbnail generation and video duration detection
echo "Installing ffmpeg..."
apt-get update -qq 2>/dev/null || true
apt-get install -y -qq ffmpeg 2>/dev/null || \
  (echo "apt-get failed, trying alternative..." && \
   apt-get install -y ffmpeg 2>&1 || echo "Warning: ffmpeg installation failed")

# Verify ffmpeg installed
if command -v ffmpeg &> /dev/null; then
  echo "ffmpeg installed: $(ffmpeg -version 2>&1 | head -1)"
else
  echo "WARNING: ffmpeg not available. Thumbnails and duration detection will fail."
fi

# Install critical dependencies
# - setuptools: for distutils compatibility
# - pyyaml: for ComfyUI config files
# - transformers, Pillow, typing_extensions: ComfyUI requires these
# - Other deps: for DGN client operation
echo "Installing base dependencies..."
$PYTHON_EXE -m pip install --quiet --break-system-packages \
  setuptools pyyaml requests python-dotenv websocket-client \
  py-cpuinfo GPUtil psutil transformers Pillow typing_extensions \
  aiohttp einops safetensors scipy tqdm 2>/dev/null || \
$PYTHON_EXE -m pip install --quiet \
  setuptools pyyaml requests python-dotenv websocket-client \
  py-cpuinfo GPUtil psutil transformers Pillow typing_extensions \
  aiohttp einops safetensors scipy tqdm || true

# Start ComfyUI in the background
if [ -d "/opt/ComfyUI" ]; then
  echo "Starting ComfyUI in background..."
  (cd /opt/ComfyUI && $PYTHON_EXE main.py --listen > /tmp/comfyui.log 2>&1) &
  echo "ComfyUI startup initiated (logging to /tmp/comfyui.log)"
else
  echo "Warning: /opt/ComfyUI not found. Pod might be LLM-only."
fi

# Create directories
mkdir -p /opt/dgn-client /data/.cache /data/input

# Download DGN client from GitHub using bootstrap script
cd /opt/dgn-client
echo "Downloading DGN client files..."
export INSTALL_DEPS=true
curl -sL https://raw.githubusercontent.com/besch/openfork_client/main/bootstrap.sh -o bootstrap.sh
bash bootstrap.sh

echo "DGN client downloaded successfully"

# Wait for ComfyUI to be ready (if it exists)
if [ -d "/opt/ComfyUI" ]; then
  echo "Waiting for ComfyUI to be ready..."
  MAX_WAIT=120
  WAITED=0
  while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
      echo "ComfyUI is ready!"
      break
    fi
    [ $((WAITED % 10)) -eq 0 ] && echo "  Waiting... ($WAITED/$MAX_WAIT seconds)"
    sleep 2
    WAITED=$((WAITED + 2))
  done
fi

# Save restart configuration
echo "Saving restart configuration..."
cat > /opt/dgn-client/.restart-config << RESTART_CONFIG_EOF
export DGN_CLIENT_ARGS="--dgn-api-key \"$DGN_API_KEY\" --service \"${SERVICE_TYPE:-auto}\" --accept-policy all --root-dir /opt/dgn-client --data-dir /data"
export ORCHESTRATOR_URL_PROD="${DGN_ORCHESTRATOR_URL:-https://openfork.video}"
RESTART_CONFIG_EOF

chmod +x /opt/dgn-client/.restart-config

# Run the client
echo "Starting DGN client..."
export ORCHESTRATOR_URL_PROD="${DGN_ORCHESTRATOR_URL:-https://openfork.video}"

# Test that imports work before running
echo "Testing Python imports..."
$PYTHON_EXE -c "
import sys
sys.path.insert(0, '/opt/dgn-client')
try:
    from config import HEADLESS_MODE
    print(f'HEADLESS_MODE = {HEADLESS_MODE}')
    from dgn_client import DGNClient
    print('DGNClient import successful')
except Exception as e:
    print(f'Import error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -ne 0 ]; then
  echo "ERROR: Python imports failed. Check the error above."
  exit 1
fi

echo "Imports OK. Starting client..."
$PYTHON_EXE cli.py \
  --dgn-api-key "$DGN_API_KEY" \
  --service "${SERVICE_TYPE:-auto}" \
  --accept-policy all \
  --root-dir /opt/dgn-client \
  --data-dir /data 2>&1 | tee -a /tmp/dgn_client.log
