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

# Ensure python3 is available as 'python'
if ! command -v python &> /dev/null; then
  if command -v python3 &> /dev/null; then
    echo "Creating python symlink to python3..."
    ln -sf $(which python3) /usr/local/bin/python 2>/dev/null || true
  fi
fi

# Ensure pip is available
if ! command -v pip &> /dev/null; then
  if command -v pip3 &> /dev/null; then
    echo "Creating pip symlink to pip3..."
    ln -sf $(which pip3) /usr/local/bin/pip 2>/dev/null || true
  fi
fi

PYTHON_EXE=$(command -v python3 || command -v python || echo "NOT_FOUND")
echo "Python Executable: $PYTHON_EXE"

if [ "$PYTHON_EXE" = "NOT_FOUND" ]; then
  echo "ERROR: Python not found! Exiting."
  exit 1
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

# Install critical dependencies
echo "Installing base dependencies..."
$PYTHON_EXE -m pip install --quiet --break-system-packages requests python-dotenv websocket-client 2>/dev/null || \
$PYTHON_EXE -m pip install --quiet requests python-dotenv websocket-client || true

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

$PYTHON_EXE cli.py \
  --dgn-api-key "$DGN_API_KEY" \
  --service "${SERVICE_TYPE:-auto}" \
  --accept-policy all \
  --root-dir /opt/dgn-client \
  --data-dir /data 2>&1 | tee -a /tmp/dgn_client.log
