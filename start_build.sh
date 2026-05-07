#!/bin/bash
# OpenFork Docker Image Builder Script (start_build.sh)
# This script is fetched and run by cloud containers for building Docker images

OPENFORK_CLIENT_SCRIPT_REF="${OPENFORK_CLIENT_SCRIPT_REF:-main}"
OPENFORK_RAW_BASE="https://raw.githubusercontent.com/besch/openfork_client/${OPENFORK_CLIENT_SCRIPT_REF}"

download_openfork_script() {
  local script_name="$1"
  local dest="$2"
  local sha_var="$3"
  curl --fail --location --proto '=https' --tlsv1.2 --silent --show-error \
    "${OPENFORK_RAW_BASE}/${script_name}" -o "$dest"

  local expected_sha="${!sha_var:-}"
  if [ -n "$expected_sha" ]; then
    echo "${expected_sha}  ${dest}" | sha256sum -c -
  fi
}

# --- Help ---
show_help() {
  cat << EOF
OpenFork Docker Image Builder

USAGE:
  curl -sL https://raw.githubusercontent.com/besch/openfork_client/main/start_build.sh | bash

  Or with environment variables:
  DOCKERFILE_NAME=Dockerfile.ltx23-wan2gp-8gb DOCKER_TAG=user/image:tag ./start_build.sh

ENVIRONMENT VARIABLES:
  Required:
    DOCKERFILE_NAME       Dockerfile to build (e.g., Dockerfile.ltx23-wan2gp-8gb)
    DOCKER_TAG            Docker image tag (e.g., beschiak/openfork-ltx23-wan2gp-8gb:latest)

  Optional:
    BUILD_JOB_ID          UUID of the build job for webhook reporting
    ORCHESTRATOR_URL      Webhook URL for status updates (default: https://openfork.video)
    PUSH_TO_DOCKERHUB     Set to "true" to push image after build
    DOCKERHUB_USER        DockerHub username (required if pushing)
    DOCKERHUB_TOKEN       DockerHub access token (required if pushing)
    HF_TOKEN              HuggingFace token for gated models
    RUN_DGN_CLIENT        Set to "true" to start DGN client after build
    DGN_API_KEY           DGN API key (required if running client)
    SERVICE_TYPE          Service type for DGN client (e.g., ltx23-video-8gb)
    SAVE_LOGS             Set to "true" to stream logs to webhook (default: false)

EXAMPLES:
  # Build only
  DOCKERFILE_NAME=Dockerfile.ltx23-wan2gp-8gb DOCKER_TAG=myuser/ltx23-q4:test ./start_build.sh

  # Build and push
  DOCKERFILE_NAME=Dockerfile.ltx23-wan2gp-8gb DOCKER_TAG=myuser/ltx23-q4:test \\
    PUSH_TO_DOCKERHUB=true DOCKERHUB_USER=myuser DOCKERHUB_TOKEN=xxx \\
    ./start_build.sh

  # Build, push, and run DGN client
  DOCKERFILE_NAME=Dockerfile.ltx23-wan2gp-8gb DOCKER_TAG=myuser/ltx23-q4:test \\
    PUSH_TO_DOCKERHUB=true RUN_DGN_CLIENT=true DGN_API_KEY=xxx \\
    ./start_build.sh
EOF
  exit 0
}

# Check for --help flag
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
  show_help
fi

set -e

# --- Log Configuration ---
LOG_FILE="/tmp/build.log"
LOG_STREAM_INTERVAL=30  # Seconds between log uploads
LAST_LOG_POSITION=0

# Redirect all output to a log file AND stdout
exec > >(tee -a "$LOG_FILE") 2>&1

# --- Helper Functions ---

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

# Report status to webhook
report_status() {
  local status="$1"
  local log_chunk="$2"
  local error="$3"

  if [ -z "$BUILD_JOB_ID" ] || [ -z "$ORCHESTRATOR_URL" ]; then
    return
  fi

  local payload="{\"job_id\": \"$BUILD_JOB_ID\", \"status\": \"$status\""
  
  if [ -n "$log_chunk" ]; then
    # Escape special characters for JSON
    local escaped_log=$(echo "$log_chunk" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g' | tr '\n' ' ')
    payload="$payload, \"log_chunk\": \"$escaped_log\""
  fi
  
  if [ -n "$error" ]; then
    local escaped_error=$(echo "$error" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g' | tr '\n' ' ')
    payload="$payload, \"error\": \"$escaped_error\""
  fi
  
  payload="$payload}"

  curl -s -X POST "$ORCHESTRATOR_URL/api/build-webhook" \
    -H "Content-Type: application/json" \
    -d "$payload" || true
}

# Stream logs to webhook (non-blocking)
stream_logs() {
  if [ "$SAVE_LOGS" != "true" ]; then
    return
  fi
  
  if [ ! -f "$LOG_FILE" ]; then
    return
  fi
  
  # Get new log content since last position
  local current_size=$(wc -c < "$LOG_FILE")
  if [ "$current_size" -gt "$LAST_LOG_POSITION" ]; then
    # Extract new content, filter spam, limit size
    local new_logs=$(tail -c +$((LAST_LOG_POSITION + 1)) "$LOG_FILE" | \
      grep -vE '^\s*$|^\[=+\]$|^##+$|^=+$|^\s+[0-9.]+%' | \
      head -c 50000)
    
    if [ -n "$new_logs" ]; then
      report_status "" "$new_logs" ""
    fi
    
    LAST_LOG_POSITION=$current_size
  fi
}

# Background log streamer
start_log_streamer() {
  if [ "$SAVE_LOGS" != "true" ]; then
    return
  fi
  
  log "Log streaming enabled (interval: ${LOG_STREAM_INTERVAL}s)"
  
  (
    while true; do
      sleep "$LOG_STREAM_INTERVAL"
      stream_logs
    done
  ) &
  LOG_STREAMER_PID=$!
}

stop_log_streamer() {
  if [ -n "$LOG_STREAMER_PID" ]; then
    kill "$LOG_STREAMER_PID" 2>/dev/null || true
  fi
}

# Send final logs on exit
send_final_logs() {
  if [ "$SAVE_LOGS" = "true" ] && [ -f "$LOG_FILE" ]; then
    log "Uploading final logs..."
    # Send last 100KB of logs
    local final_logs=$(tail -c 100000 "$LOG_FILE" | \
      grep -vE '^\s*$|^\[=+\]$|^##+$|^=+$' | \
      head -c 100000)
    report_status "" "$final_logs" ""
  fi
}

# Cleanup on exit
cleanup() {
  stop_log_streamer
  send_final_logs
}
trap cleanup EXIT

# --- Initialization ---

log "========================================"
log "=== OpenFork Docker Image Builder ==="
log "========================================"
log "Build Job ID: $BUILD_JOB_ID"
log "Dockerfile: $DOCKERFILE_NAME"
log "Docker Tag: $DOCKER_TAG"
log "Push to DockerHub: $PUSH_TO_DOCKERHUB"
log "Run DGN Client: $RUN_DGN_CLIENT"
log "Save Logs: $SAVE_LOGS"

# Check required environment variables
if [ -z "$DOCKERFILE_NAME" ] || [ -z "$DOCKER_TAG" ]; then
  log "ERROR: DOCKERFILE_NAME and DOCKER_TAG are required"
  log "Run with --help for usage information"
  report_status "failed" "" "Missing DOCKERFILE_NAME or DOCKER_TAG"
  exit 1
fi

# Start log streaming if enabled
start_log_streamer

# --- Install Docker ---

log "Checking for Docker..."
if ! command -v docker &> /dev/null; then
  log "Installing Docker..."
  apt-get update -qq
  apt-get install -y -qq ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin
  log "Docker installed successfully"
fi

# Install NVIDIA Container Toolkit for GPU support in Docker
if ! command -v nvidia-container-toolkit &> /dev/null && [ -n "$(command -v nvidia-smi 2>/dev/null)" ]; then
  log "Installing NVIDIA Container Toolkit..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
  apt-get update -qq
  apt-get install -y -qq nvidia-container-toolkit
  # Configure Docker daemon for NVIDIA runtime
  nvidia-ctk runtime configure --runtime=docker || true
  log "NVIDIA Container Toolkit installed"
fi

# Start Docker daemon if not running
if ! docker info &> /dev/null; then
  log "Starting Docker daemon..."
  dockerd &
  sleep 5
  
  # Wait for Docker to be ready
  for i in {1..30}; do
    if docker info &> /dev/null; then
      log "Docker daemon is ready"
      break
    fi
    sleep 1
  done
fi

docker --version

# --- Clone Repository ---

log "Cloning openfork_client repository..."
mkdir -p /opt/build
cd /opt/build

# Clone the client repo which contains Dockerfiles
git clone --depth 1 https://github.com/besch/openfork_client.git client
cd client/comfyui-storage

# Check if dockerfile exists
if [ ! -f "$DOCKERFILE_NAME" ]; then
  log "ERROR: Dockerfile not found: $DOCKERFILE_NAME"
  report_status "failed" "" "Dockerfile not found: $DOCKERFILE_NAME"
  exit 1
fi

log "Found Dockerfile: $DOCKERFILE_NAME"

# --- Build Image ---

report_status "building" "Starting Docker build..."

log "Building image: $DOCKER_TAG"
log "Using Dockerfile: $DOCKERFILE_NAME"

BUILD_ARGS=""
if [ -n "$HF_TOKEN" ]; then
  BUILD_ARGS="--build-arg HF_TOKEN=$HF_TOKEN"
  log "HuggingFace token provided"
fi

# Run build and capture output
BUILD_OUTPUT="/tmp/docker_build.log"
if docker build --no-cache $BUILD_ARGS -f "$DOCKERFILE_NAME" -t "$DOCKER_TAG" . 2>&1 | tee "$BUILD_OUTPUT"; then
  log "Build completed successfully!"
  report_status "building" "Build completed successfully"
else
  log "ERROR: Build failed!"
  BUILD_ERROR=$(tail -50 "$BUILD_OUTPUT")
  report_status "failed" "" "Docker build failed: $BUILD_ERROR"
  exit 1
fi

# --- Push to DockerHub (if enabled) ---

if [ "$PUSH_TO_DOCKERHUB" = "true" ]; then
  if [ -z "$DOCKERHUB_USER" ] || [ -z "$DOCKERHUB_TOKEN" ]; then
    log "ERROR: DockerHub credentials required for push"
    report_status "failed" "" "DockerHub credentials not provided"
    exit 1
  fi

  report_status "pushing" "Logging into DockerHub..."

  log "Logging into DockerHub as $DOCKERHUB_USER..."
  echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USER" --password-stdin

  log "Pushing image: $DOCKER_TAG"
  if docker push "$DOCKER_TAG" 2>&1 | tee /tmp/docker_push.log; then
    log "Push completed successfully!"
    report_status "pushing" "Push completed successfully"
  else
    log "ERROR: Push failed!"
    PUSH_ERROR=$(tail -20 /tmp/docker_push.log)
    report_status "failed" "" "Docker push failed: $PUSH_ERROR"
    exit 1
  fi
fi

# --- Run DGN Client (if enabled) ---

if [ "$RUN_DGN_CLIENT" = "true" ]; then
  log "Starting DGN Client using the newly built Docker image..."
  report_status "running" "Starting DGN client container..."

  # Create data directory for persistent storage
  mkdir -p /data

  # Run the newly built image as a container with GPU access
  # The container will run start_cloud.sh which starts ComfyUI and dgn_client
  log "Starting container from image: $DOCKER_TAG"
  docker run --rm \
    --gpus all \
    --name openfork-dgn-client \
    -e DGN_API_KEY="$DGN_API_KEY" \
    -e DGN_ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-https://openfork.video}" \
    -e SERVICE_TYPE="$SERVICE_TYPE" \
    -e HEADLESS_MODE="true" \
    -e ACCEPT_POLICY="all" \
    -e PYTHONUNBUFFERED="1" \
    -e OPENFORK_CLIENT_SCRIPT_REF="$OPENFORK_CLIENT_SCRIPT_REF" \
    ${OPENFORK_BOOTSTRAP_SHA256:+-e OPENFORK_BOOTSTRAP_SHA256="$OPENFORK_BOOTSTRAP_SHA256"} \
    ${OPENFORK_START_CLOUD_SHA256:+-e OPENFORK_START_CLOUD_SHA256="$OPENFORK_START_CLOUD_SHA256"} \
    -v /data:/data \
    "$DOCKER_TAG" \
    bash -c 'set -euo pipefail; raw_base="https://raw.githubusercontent.com/besch/openfork_client/${OPENFORK_CLIENT_SCRIPT_REF:-main}"; curl --fail --location --proto "=https" --tlsv1.2 --silent --show-error "$raw_base/start_cloud.sh" -o /tmp/start_cloud.sh; if [ -n "${OPENFORK_START_CLOUD_SHA256:-}" ]; then echo "${OPENFORK_START_CLOUD_SHA256}  /tmp/start_cloud.sh" | sha256sum -c -; fi; bash /tmp/start_cloud.sh'
else
  log "Build complete. No DGN client requested."
  report_status "completed" "Build completed successfully"
fi

log "========================================"
log "=== Build Script Complete ==="
log "========================================"
