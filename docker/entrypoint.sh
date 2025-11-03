#!/bin/bash
set -e

echo "[Entrypoint] Starting ComfyUI initialization..."

# Navigate to ComfyUI directory
cd /app/ComfyUI

# Ensure ComfyUI Manager is properly installed
if [ ! -d "custom_nodes/ComfyUI-Manager" ]; then
    echo "[Entrypoint] ComfyUI Manager not found. Installing..."
    cd custom_nodes
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git
    cd ComfyUI-Manager
    pip3 install -r requirements.txt
    cd /app/ComfyUI
fi

# Run cm-cli fix to ensure all existing custom nodes have their dependencies
# This is crucial after container restarts
if [ -d "custom_nodes/ComfyUI-Manager" ]; then
    echo "[Entrypoint] Checking and fixing custom node dependencies..."
    cd custom_nodes/ComfyUI-Manager
    
    # Run fix command, but don't fail if it has issues
    python3 cm-cli.py fix all 2>&1 || echo "[Entrypoint] Warning: cm-cli fix reported issues, but continuing..."
    
    cd /app/ComfyUI
fi

# Update ComfyUI if it's a git repo
if [ -d ".git" ]; then
    echo "[Entrypoint] Updating ComfyUI to latest version..."
    git pull || echo "[Entrypoint] Warning: Could not update ComfyUI"
fi

# Ensure output directory exists and has correct permissions
mkdir -p output input
chmod -R 777 output input

echo "[Entrypoint] ComfyUI initialization complete. Starting server..."
echo "[Entrypoint] ComfyUI will be available at http://0.0.0.0:8188"

# Start ComfyUI with proper options
exec python3 main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --enable-cors-header \
    2>&1