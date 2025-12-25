#!/bin/bash
# OpenFork DGN Client Cloud Startup Script (start_cloud.sh)
# This script is fetched and run by cloud containers (RunPod, Vast.ai)
# It downloads the DGN client and starts it after ComfyUI is ready

set -e

echo "=== OpenFork DGN Worker Initialization ==="
echo "Service: ${SERVICE_TYPE:-auto}"
echo "Selected Workflows: ${SELECTED_WORKFLOWS:-auto}"
echo "Orchestrator: ${DGN_ORCHESTRATOR_URL:-https://openfork.video}"

# Start ComfyUI in the background
# We assume ComfyUI is in /opt/ComfyUI as per our Dockerfiles
if [ -d "/opt/ComfyUI" ]; then
  echo "Starting ComfyUI in background..."
  (cd /opt/ComfyUI && python main.py --listen > /var/log/comfyui.log 2>&1) &
  echo "ComfyUI startup initiated (logging to /var/log/comfyui.log)"
else
  echo "Warning: /opt/ComfyUI not found. Skipping ComfyUI startup."
fi

# Install dependencies
echo "Installing Python dependencies..."
pip install --quiet requests python-dotenv websocket-client 2>/dev/null || true

# Create directories
mkdir -p /opt/dgn-client /data/.cache /data/input

# Download DGN client from GitHub using bootstrap script
cd /opt/dgn-client
echo "Downloading DGN client files..."
export INSTALL_DEPS=true
curl -sL https://raw.githubusercontent.com/besch/openfork_client/main/bootstrap.sh | bash

echo "DGN client downloaded successfully"

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to be ready..."
MAX_WAIT=120
WAIT_INTERVAL=2
WAITED=0

while [ $WAITED -lt $MAX_WAIT ]; do
  if curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
    echo "ComfyUI is ready!"
    break
  fi
  echo "  Waiting... ($WAITED/$MAX_WAIT seconds)"
  sleep $WAIT_INTERVAL
  WAITED=$((WAITED + WAIT_INTERVAL))
done

if [ $WAITED -ge $MAX_WAIT ]; then
  echo "Warning: ComfyUI did not become ready within $MAX_WAIT seconds. Proceeding anyway..."
fi

# Save restart configuration for Remote Restart feature
echo "Saving restart configuration..."
cat > /opt/dgn-client/.restart-config << RESTART_CONFIG_EOF
export DGN_CLIENT_ARGS="--dgn-api-key \"$DGN_API_KEY\" --service \"${SERVICE_TYPE:-auto}\" --accept-policy all --root-dir /opt/dgn-client --data-dir /data"
export ORCHESTRATOR_URL_PROD="${DGN_ORCHESTRATOR_URL:-https://openfork.video}"
RESTART_CONFIG_EOF

chmod +x /opt/dgn-client/.restart-config

# Start DGN client
echo "Starting DGN client..."
cd /opt/dgn-client

# Export orchestrator URL from env var
export ORCHESTRATOR_URL_PROD="${DGN_ORCHESTRATOR_URL:-https://openfork.video}"

# Run the client
python cli.py \
  --dgn-api-key "$DGN_API_KEY" \
  --service "${SERVICE_TYPE:-auto}" \
  --accept-policy all \
  --root-dir /opt/dgn-client \
  --data-dir /data
