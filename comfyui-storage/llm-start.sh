#!/bin/bash
set -e

echo "=== OpenFork LLM Worker Startup ==="
echo "Timestamp: $(date)"

# Start Ollama server in background
echo "Starting Ollama server..."
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=1
ollama serve &
OLLAMA_PID=$!
sleep 5

# Setup directories
mkdir -p /data/.cache /data/input

# DGN client is pre-installed in Docker image
# Only update if explicitly requested via environment variable
cd /opt/dgn-client

if [ "$DGN_UPDATE_CLIENT" = "true" ]; then
  echo "Updating DGN client (DGN_UPDATE_CLIENT=true)..."
  curl -sL https://raw.githubusercontent.com/besch/openfork_client/main/bootstrap.sh -o bootstrap.sh
  bash bootstrap.sh
  rm -f bootstrap.sh
else
  echo "Using pre-installed DGN client (set DGN_UPDATE_CLIENT=true to force update)"
  if [ -f /opt/dgn-client/.installed ]; then
    echo "Installed: $(cat /opt/dgn-client/.installed)"
  fi
fi

# Start DGN client
echo "Starting DGN client..."
export ORCHESTRATOR_URL_PROD="${DGN_ORCHESTRATOR_URL:-https://openfork.video}"

python3 cli.py \
  --dgn-api-key "$DGN_API_KEY" \
  --service "${SERVICE_TYPE:-auto}" \
  --accept-policy all \
  --root-dir /opt/dgn-client \
  --data-dir /data &

# Wait for Ollama to exit (keeps container running)
wait $OLLAMA_PID
