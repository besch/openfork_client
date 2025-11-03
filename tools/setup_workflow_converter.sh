#!/bin/bash
#
# Setup script for workflow-to-api-converter-endpoint integration
#
# This script:
# 1. Ensures ComfyUI container is running
# 2. Installs workflow-to-api-converter-endpoint custom node
# 3. Restarts ComfyUI to load the endpoint
# 4. Verifies the /workflow/convert endpoint is available
#

set -e

echo "=================================================="
echo "Setting up workflow-to-api-converter-endpoint"
echo "=================================================="

# Configuration
COMPOSE_FILE="${COMPOSE_FILE:-../docker/docker-compose.unified.yaml}"
COMFYUI_URL="${COMFYUI_URL:-http://127.0.0.1:8188}"
CONVERTER_REPO="https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    log_error "docker-compose not found. Please install it first."
    exit 1
fi

# Step 1: Start ComfyUI container if not running
log_info "Checking ComfyUI container status..."

CONTAINER_NAME=$(docker ps --filter "name=comfyui" --format "{{.Names}}" | head -n 1)

if [ -z "$CONTAINER_NAME" ]; then
    log_warn "ComfyUI container not running. Starting it..."
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Wait for container to be ready
    log_info "Waiting for ComfyUI to be ready..."
    for i in {1..30}; do
        if curl -s "$COMFYUI_URL/system_stats" > /dev/null 2>&1; then
            log_info "ComfyUI is ready!"
            break
        fi
        echo -n "."
        sleep 2
        
        if [ $i -eq 30 ]; then
            log_error "ComfyUI did not become ready in time"
            exit 1
        fi
    done
    echo ""
    
    # Get container name after starting
    CONTAINER_NAME=$(docker ps --filter "name=comfyui" --format "{{.Names}}" | head -n 1)
else
    log_info "ComfyUI container is running: $CONTAINER_NAME"
fi

# Step 2: Check if converter is already installed
log_info "Checking if workflow-to-api-converter-endpoint is installed..."

CONVERTER_CHECK=$(docker exec "$CONTAINER_NAME" bash -c \
    "[ -d /app/ComfyUI/custom_nodes/comfyui-workflow-to-api-converter-endpoint ] && echo 'EXISTS' || echo 'NOT_FOUND'")

if echo "$CONVERTER_CHECK" | grep -q "EXISTS"; then
    log_info "workflow-to-api-converter-endpoint is already installed"
else
    log_info "Installing workflow-to-api-converter-endpoint..."
    
    docker exec "$CONTAINER_NAME" bash -c "
        cd /app/ComfyUI/custom_nodes && \
        git clone $CONVERTER_REPO
    "
    
    if [ $? -eq 0 ]; then
        log_info "Successfully cloned converter repository"
    else
        log_error "Failed to clone converter repository"
        exit 1
    fi
fi

# Step 3: Restart ComfyUI to load the endpoint
log_info "Restarting ComfyUI to load the converter endpoint..."
docker-compose -f "$COMPOSE_FILE" restart

log_info "Waiting for ComfyUI to restart..."
for i in {1..30}; do
    if curl -s "$COMFYUI_URL/system_stats" > /dev/null 2>&1; then
        log_info "ComfyUI restarted successfully!"
        break
    fi
    echo -n "."
    sleep 2
    
    if [ $i -eq 30 ]; then
        log_error "ComfyUI did not restart in time"
        exit 1
    fi
done
echo ""

# Step 4: Verify the endpoint is available
log_info "Verifying /workflow/convert endpoint..."

# Wait a bit for endpoint to be registered
sleep 5

# Test endpoint (GET should return 405, which means it exists)
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$COMFYUI_URL/workflow/convert")

if [ "$HTTP_CODE" = "405" ] || [ "$HTTP_CODE" = "200" ]; then
    log_info "/workflow/convert endpoint is available!"
elif [ "$HTTP_CODE" = "404" ]; then
    log_error "Endpoint not found. The custom node may not have loaded correctly."
    log_error "Check ComfyUI logs: docker-compose -f $COMPOSE_FILE logs comfyui"
    exit 1
else
    log_warn "Unexpected response code: $HTTP_CODE"
    log_warn "The endpoint may still work for POST requests"
fi

# Step 5: Test conversion with a simple workflow
log_info "Testing workflow conversion..."

TEST_WORKFLOW='{
  "nodes": [
    {
      "id": 1,
      "type": "CheckpointLoaderSimple",
      "pos": [0, 0],
      "size": [315, 98],
      "flags": {},
      "order": 0,
      "mode": 0,
      "outputs": [
        {"name": "MODEL", "type": "MODEL", "links": [1]},
        {"name": "CLIP", "type": "CLIP", "links": [2]},
        {"name": "VAE", "type": "VAE", "links": [3]}
      ],
      "properties": {},
      "widgets_values": ["model.safetensors"]
    },
    {
      "id": 2,
      "type": "SaveImage",
      "pos": [100, 0],
      "size": [315, 270],
      "flags": {},
      "order": 1,
      "mode": 0,
      "inputs": [
        {"name": "images", "type": "IMAGE", "link": 1}
      ],
      "properties": {},
      "widgets_values": ["output"]
    }
  ],
  "links": [
    [1, 1, 0, 2, 0, "IMAGE"]
  ]
}'

CONVERSION_RESPONSE=$(curl -s -X POST "$COMFYUI_URL/workflow/convert" \
    -H "Content-Type: application/json" \
    -d "$TEST_WORKFLOW" \
    -w "\nHTTP_CODE:%{http_code}")

HTTP_CODE=$(echo "$CONVERSION_RESPONSE" | grep "HTTP_CODE:" | cut -d':' -f2)
RESPONSE_BODY=$(echo "$CONVERSION_RESPONSE" | sed '/HTTP_CODE:/d')

if [ "$HTTP_CODE" = "200" ]; then
    log_info "Conversion test successful!"
    echo "$RESPONSE_BODY" | python3 -m json.tool > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        log_info "Response is valid JSON"
    else
        log_warn "Response is not valid JSON"
    fi
else
    log_error "Conversion test failed with HTTP code: $HTTP_CODE"
    log_error "Response: $RESPONSE_BODY"
    exit 1
fi

# Final summary
echo ""
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "The workflow converter is now ready to use."
echo ""
echo "Next steps:"
echo "1. Update your workflow_sync.py to use the new converter"
echo "2. Run: python workflow_sync.py"
echo ""
echo "Converter endpoint: $COMFYUI_URL/workflow/convert"
echo "Container name: $CONTAINER_NAME"
echo ""