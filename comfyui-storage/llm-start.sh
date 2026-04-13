#!/bin/bash
set -e

echo "=== OpenFork LLM Worker Startup ==="
echo "Timestamp: $(date)"

# Start Ollama server in background
echo "Starting Ollama server..."
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_HOST=0.0.0.0
export OLLAMA_ORIGINS="*"
ollama serve &
OLLAMA_PID=$!
# Wait for Ollama to be responsive
echo "Waiting for Ollama to be ready..."
MAX_ATTEMPTS=30
ATTEMPT=0
while ! curl -s http://127.0.0.1:11434/api/tags > /dev/null; do
  if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
    echo "Ollama failed to start in time"
    exit 1
  fi
  ATTEMPT=$((ATTEMPT+1))
  sleep 0.5
done
echo "Ollama is ready!"

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
  --community-mode all \
  --root-dir /opt/dgn-client \
  --data-dir /data &

# Wait for Ollama to exit (keeps container running)
wait $OLLAMA_PID
