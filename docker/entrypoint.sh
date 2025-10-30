#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Function to log messages
log() {
    echo "[Entrypoint] $1"
}

log "Starting dependency installation..."

# Install custom nodes if the environment variable is set
if [ -n "$CUSTOM_NODES_GIT_URLS" ]; then
    log "Found custom node URLs. Installing..."
    # Split the comma-separated string into an array
    IFS=',' read -r -a urls <<< "$CUSTOM_NODES_GIT_URLS"
    for url in "${urls[@]}"; do
        # Trim leading/trailing whitespace
        trimmed_url=$(echo "$url" | xargs)
        if [ -n "$trimmed_url" ]; then
            log "Installing custom node from: $trimmed_url"
            python3 /app/ComfyUI/custom_nodes/ComfyUI-Manager/cli/main.py --install-custom-node "$trimmed_url"
        fi
    done
    log "Custom node installation finished."
else
    log "No custom node URLs provided. Skipping."
fi

# Install models if the environment variable is set
if [ -n "$MODEL_URLS" ]; then
    log "Found model URLs. Installing..."
    # Split the comma-separated string into an array
    IFS=',' read -r -a urls <<< "$MODEL_URLS"
    for url in "${urls[@]}"; do
        # Trim leading/trailing whitespace
        trimmed_url=$(echo "$url" | xargs)
        if [ -n "$trimmed_url" ]; then
            log "Installing model from: $trimmed_url"
            python3 /app/ComfyUI/custom_nodes/ComfyUI-Manager/cli/main.py --install-model "$trimmed_url"
        fi
    done
    log "Model installation finished."
else
    log "No model URLs provided. Skipping."
fi

log "All dependencies installed."
log "Starting ComfyUI..."

# Execute the main ComfyUI process
exec python3 main.py --listen --port 8188
