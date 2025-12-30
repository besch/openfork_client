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

# Download and setup DGN client
mkdir -p /opt/dgn-client /data/.cache /data/input
cd /opt/dgn-client

echo "Downloading DGN client..."
curl -sL https://raw.githubusercontent.com/besch/openfork_client/main/bootstrap.sh -o bootstrap.sh
export INSTALL_DEPS=true
bash bootstrap.sh

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
