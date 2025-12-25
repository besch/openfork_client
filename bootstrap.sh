#!/bin/bash
# OpenFork DGN Client Bootstrap Script
# Downloads all client files from GitHub repository
# Usage: curl -sL https://raw.githubusercontent.com/besch/openfork_client/main/bootstrap.sh | bash

set -e

BASE_URL="https://raw.githubusercontent.com/besch/openfork_client/main"

echo "=== OpenFork DGN Client Bootstrap ==="
echo "Downloading client files from GitHub..."

# Core files
echo "Downloading core files..."
curl -sL $BASE_URL/cli.py -o cli.py
curl -sL $BASE_URL/dgn_client.py -o dgn_client.py
curl -sL $BASE_URL/config.py -o config.py

# Services directory
echo "Downloading services..."
mkdir -p services/processors

SERVICES_FILES=(
  "__init__.py"
  "orchestrator_service.py"
  "comfyui_service.py"
  "docker_manager.py"
  "heartbeat_manager.py"
  "job_listener.py"
  "hardware_profiler.py"
)

for file in "${SERVICES_FILES[@]}"; do
  curl -sL $BASE_URL/services/$file -o services/$file
done

# Processors
echo "Downloading processors..."
PROCESSOR_FILES=(
  "__init__.py"
  "base.py"
  "comfyui.py"
  "video.py"
  "audio.py"
  "image.py"
  "text.py"
)

for file in "${PROCESSOR_FILES[@]}"; do
  curl -sL $BASE_URL/services/processors/$file -o services/processors/$file
done

# Utils directory
echo "Downloading utils..."
mkdir -p utils

UTILS_FILES=(
  "__init__.py"
  "comfyui_workflow_utils.py"
  "hardware_utils.py"
  "shutdown_handler.py"
)

for file in "${UTILS_FILES[@]}"; do
  curl -sL $BASE_URL/utils/$file -o utils/$file
done

# Download requirements.txt
echo "Downloading requirements.txt..."
curl -sL $BASE_URL/requirements.txt -o requirements.txt

# Install Python dependencies if we're in a fresh environment
if [ "$INSTALL_DEPS" = "true" ]; then
  echo "Installing Python dependencies..."
  pip install --quiet -r requirements.txt 2>/dev/null || true
fi

echo "✓ DGN client files downloaded successfully"
