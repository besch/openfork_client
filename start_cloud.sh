#!/bin/bash
# OpenFork DGN Client Cloud Startup Script (start_cloud.sh)
# This script is fetched and run by cloud containers (RunPod, Vast.ai)

OPENFORK_CLIENT_SCRIPT_REF="${OPENFORK_CLIENT_SCRIPT_REF:-main}"
OPENFORK_RAW_BASE="https://raw.githubusercontent.com/besch/openfork_client/${OPENFORK_CLIENT_SCRIPT_REF}"

download_openfork_script() {
  local script_name="$1"
  local dest="$2"
  local sha_var="$3"
  curl --fail --location --proto '=https' --tlsv1.2 --max-time 60 --progress-bar \
    "${OPENFORK_RAW_BASE}/${script_name}" -o "$dest"

  local expected_sha="${!sha_var:-}"
  if [ -n "$expected_sha" ]; then
    echo "${expected_sha}  ${dest}" | sha256sum -c -
  fi
}

# --- Help ---
show_help() {
  cat << EOF
OpenFork DGN Client Cloud Startup Script

USAGE:
  curl -sL https://raw.githubusercontent.com/besch/openfork_client/main/start_cloud.sh | bash

  Or with environment variables:
  DGN_API_KEY=xxx SERVICE_TYPE=ltx23-video-8gb ./start_cloud.sh

ENVIRONMENT VARIABLES:
  Required:
    DGN_API_KEY           API key for DGN client authentication

  Optional:
    SERVICE_TYPE          Service type to advertise (default: auto)
    DGN_ORCHESTRATOR_URL  Orchestrator URL (default: https://openfork.video)
    HEADLESS_MODE         Set to "true" for headless operation
    ACCEPT_POLICY         Job acceptance policy: all, mine, users, project, monetize (default: all)
    SAVE_LOGS             Set to "true" to stream logs to webhook
    BUILD_JOB_ID          Build job ID for log association (uses provider ID if not set)

EXAMPLES:
  # Basic startup
  DGN_API_KEY=dgn_xxx ./start_cloud.sh

  # With specific service type
  DGN_API_KEY=dgn_xxx SERVICE_TYPE=ltx23-video-16gb ./start_cloud.sh

  # With log streaming
  DGN_API_KEY=dgn_xxx SAVE_LOGS=true ./start_cloud.sh
EOF
  exit 0
}

# Check for --help flag
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
  show_help
fi

set -e

# --- Log Configuration ---
# Attempt to load credentials if missing (reboot persistence)
if [ -z "$DGN_API_KEY" ] && [ -f "/etc/dgn-api-key" ]; then
  export DGN_API_KEY=$(cat /etc/dgn-api-key)
  echo "Restored DGN_API_KEY from /etc/dgn-api-key"
fi

LOG_FILE="/tmp/dgn_init.log"
LOG_STREAM_INTERVAL=60  # Seconds between log uploads (longer for runtime)
LAST_LOG_POSITION=0
ORCHESTRATOR_URL="${DGN_ORCHESTRATOR_URL:-https://openfork.video}"

# Redirect all output to a log file AND stdout so we can see it in cloud logs
exec > >(tee -a "$LOG_FILE") 2>&1

# --- Helper Functions ---

# Log with timestamp
log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

# Stream logs to webhook (for SAVE_LOGS mode)
stream_logs() {
  if [ "$SAVE_LOGS" != "true" ]; then
    return
  fi
  
  if [ ! -f "$LOG_FILE" ]; then
    return
  fi
  
  # Need a job ID to report to
  local job_id="${BUILD_JOB_ID:-}"
  if [ -z "$job_id" ]; then
    return
  fi
  
  # Get new log content since last position
  local current_size=$(wc -c < "$LOG_FILE")
  if [ "$current_size" -gt "$LAST_LOG_POSITION" ]; then
    # Extract new content, filter spam, limit size
    local new_logs=$(tail -c +$((LAST_LOG_POSITION + 1)) "$LOG_FILE" | \
      grep -vE '^\s*$|^\[=+\]$|^##+$|^=+$|^\s+[0-9.]+%|ComfyUI.*progress|Downloading.*:.*%' | \
      head -c 50000)
    
    if [ -n "$new_logs" ]; then
      local escaped_log=$(echo "$new_logs" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g' | tr '\n' ' ')
      curl -s -X POST "$ORCHESTRATOR_URL/api/build-webhook" \
        -H "Content-Type: application/json" \
        -d "{\"job_id\": \"$job_id\", \"log_chunk\": \"$escaped_log\"}" || true
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
  if [ "$SAVE_LOGS" = "true" ] && [ -f "$LOG_FILE" ] && [ -n "$BUILD_JOB_ID" ]; then
    log "Uploading final logs..."
    local final_logs=$(tail -c 100000 "$LOG_FILE" | \
      grep -vE '^\s*$|^\[=+\]$|^##+$|^=+$' | \
      head -c 100000)
    local escaped_log=$(echo "$final_logs" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g' | tr '\n' ' ')
    curl -s -X POST "$ORCHESTRATOR_URL/api/build-webhook" \
      -H "Content-Type: application/json" \
      -d "{\"job_id\": \"$BUILD_JOB_ID\", \"log_chunk\": \"$escaped_log\"}" || true
  fi
}

# Cleanup on exit
cleanup() {
  stop_log_streamer
  send_final_logs
}
trap cleanup EXIT

# Find Python executable with priority
find_python() {
  if [ -x "/usr/bin/python" ]; then
    echo "/usr/bin/python"
  elif command -v python &> /dev/null; then
    command -v python
  elif command -v python3 &> /dev/null; then
    command -v python3
  else
    echo "NOT_FOUND"
  fi
}

# Wait for a URL to return 200 OK
wait_for_url() {
  local name="$1"
  local url="$2"
  local max_wait="${3:-120}"
  local log_file="$4"
  local waited=0
  
  log "Waiting for $name to be ready at $url..."
  while [ $waited -lt $max_wait ]; do
    if curl -fsS "$url" > /dev/null 2>&1; then
      log "$name is ready!"
      return 0
    fi
    [ $((waited % 10)) -eq 0 ] && log "  Waiting... ($waited/$max_wait seconds)"
    # Output last lines of log if waiting too long
    if [ $waited -gt 30 ] && [ $((waited % 30)) -eq 0 ] && [ -f "$log_file" ]; then
        log "  Last 15 lines of $log_file:"
        tail -n 15 "$log_file" | sed 's/^/    /' || true
        # Highlight any import/load errors
        local errs
        errs=$(grep -i "cannot import\|error\|exception\|failed to\|traceback" "$log_file" 2>/dev/null | tail -n 10 || true)
        if [ -n "$errs" ]; then
            log "  === Errors/Warnings in log ==="
            echo "$errs" | sed 's/^/    /' || true
        fi
    fi
    sleep 2
    waited=$((waited + 2))
  done
  
  log "WARNING: $name did not become ready within $max_wait seconds."
  [ -n "$log_file" ] && log "Check $log_file for errors."
  return 1
}

# Ensure pip is installed for the given Python executable
install_pip() {
  local py_exe="$1"
  if ! "$py_exe" -m pip --version &> /dev/null; then
    log "pip not found, attempting to install..."
    if "$py_exe" -m ensurepip --upgrade 2>/dev/null; then
      log "pip installed via ensurepip"
    else
      log "ensurepip failed, downloading get-pip.py..."
      if curl --max-time 30 -L https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py; then
        "$py_exe" /tmp/get-pip.py --quiet --break-system-packages 2>/dev/null || \
        "$py_exe" /tmp/get-pip.py --quiet
        log "pip installed via get-pip.py"
      else
        log "ERROR: Failed to download get-pip.py"
        return 1
      fi
    fi
  fi
  
  if ! "$py_exe" -m pip --version &> /dev/null; then
    log "ERROR: Failed to install pip!"
    return 1
  fi
  return 0
}

# --- Initialization ---

log "========================================"
log "=== OpenFork DGN Worker Initialization ==="
log "========================================"
log "User: $(whoami)"
log "Path: $PATH"
log "OpenFork client script ref: ${OPENFORK_CLIENT_SCRIPT_REF}"
log "OpenFork raw base: ${OPENFORK_RAW_BASE}"
log "Save Logs: $SAVE_LOGS"
log "TIP: Large models like LTX-2 require a large swap file (128GB recommended) for stability when offloading."
RESR=$(free -g | awk '/Swap/ {print $2}')
log "Current Swap Space: ${RESR}GB"
if [ "$RESR" -lt 64 ]; then
  log "WARNING: Low swap space detected. If models stall or OOM, increase system virtual memory."
fi

# Start log streaming if enabled
start_log_streamer

PYTHON_EXE=$(find_python)
log "Python Executable: $PYTHON_EXE"
"$PYTHON_EXE" --version 2>&1 || true

if [ "$PYTHON_EXE" = "NOT_FOUND" ]; then
  log "ERROR: Python not found! Exiting."
  exit 1
fi

# Verify PyTorch is available (critical for ComfyUI)
if ! "$PYTHON_EXE" -c "import torch" 2>/dev/null; then
  log "WARNING: PyTorch not found in $PYTHON_EXE. ComfyUI will likely fail."
fi

install_pip "$PYTHON_EXE" || exit 1

# Install ffmpeg for thumbnail generation and video duration detection, and psmisc for fuser
log "Installing ffmpeg and system tools..."
apt-get update -qq 2>/dev/null || true
apt-get install -y -qq ffmpeg psmisc net-tools 2>/dev/null || \
  (log "apt-get failed, trying alternative..." && \
   apt-get install -y ffmpeg psmisc net-tools 2>&1 || log "Warning: installation failed")

# Verify ffmpeg installed
if command -v ffmpeg &> /dev/null; then
  log "ffmpeg installed: $(ffmpeg -version 2>&1 | head -1)"
else
  log "WARNING: ffmpeg not available. Thumbnails and duration detection will fail."
fi

# Install critical dependencies
log "Installing base dependencies..."
DEP_LIST="setuptools pyyaml requests python-dotenv websocket-client py-cpuinfo GPUtil psutil transformers Pillow typing_extensions aiohttp einops safetensors scipy tqdm tenacity"
"$PYTHON_EXE" -m pip install --quiet --break-system-packages $DEP_LIST 2>/dev/null || \
"$PYTHON_EXE" -m pip install --quiet $DEP_LIST || true

# --- Client Setup (Download FIRST) ---
# NOTE: We download DGN client files BEFORE starting services

mkdir -p /opt/dgn-client /data/.cache /data/input

# Download DGN client from GitHub using bootstrap script
cd /opt/dgn-client
log "Downloading DGN client files..."
export INSTALL_DEPS=true
if download_openfork_script bootstrap.sh bootstrap.sh OPENFORK_BOOTSTRAP_SHA256; then
  log "✓ bootstrap.sh downloaded successfully"
  bash bootstrap.sh
  if [ -f ".installed-ref" ]; then
    log "DGN client install metadata:"
    sed 's/^/  /' .installed-ref || true
  fi
else
  log "ERROR: Failed to download bootstrap.sh. Cannot continue."
  exit 1
fi

  # Determine source directory for overrides (repo structure can vary)
  DGN_SOURCE_DIR=""
  if [ -d "/opt/dgn-client/client/comfyui-storage" ]; then
      DGN_SOURCE_DIR="/opt/dgn-client/client/comfyui-storage"
  elif [ -d "/opt/dgn-client/comfyui-storage" ]; then
      DGN_SOURCE_DIR="/opt/dgn-client/comfyui-storage"
  fi

  if [ -n "$DGN_SOURCE_DIR" ]; then
      log "Found DGN source files at: $DGN_SOURCE_DIR"
      log "Syncing dynamic API and CLI files from GitHub repo..."
      
      # 1. HeartMuLa API & Tools (runs in /app)
      if [ -d "/app" ]; then
          [ -f "$DGN_SOURCE_DIR/heartmula_api.py" ] && cp -v "$DGN_SOURCE_DIR/heartmula_api.py" /app/
          [ -f "$DGN_SOURCE_DIR/diagnose_heartmula.sh" ] && cp -v "$DGN_SOURCE_DIR/diagnose_heartmula.sh" /app/ && chmod +x /app/diagnose_heartmula.sh
          [ -f "$DGN_SOURCE_DIR/diffrhythm_api.py" ] && cp -v "$DGN_SOURCE_DIR/diffrhythm_api.py" /app/
          [ -f "$DGN_SOURCE_DIR/audiox_api.py" ] && cp -v "$DGN_SOURCE_DIR/audiox_api.py" /app/
          [ -f "$DGN_SOURCE_DIR/qwen3_tts_api.py" ] && cp -v "$DGN_SOURCE_DIR/qwen3_tts_api.py" /app/
          [ -f "$DGN_SOURCE_DIR/diagdistill_api.py" ] && cp -v "$DGN_SOURCE_DIR/diagdistill_api.py" /app/
          [ -f "$DGN_SOURCE_DIR/stream_diffvsr_wrapper.py" ] && cp -v "$DGN_SOURCE_DIR/stream_diffvsr_wrapper.py" /app/
          [ -f "$DGN_SOURCE_DIR/sparkvsr_api.py" ] && cp -v "$DGN_SOURCE_DIR/sparkvsr_api.py" /app/
          [ -f "$DGN_SOURCE_DIR/ernie_image_api.py" ] && cp -v "$DGN_SOURCE_DIR/ernie_image_api.py" /app/
      fi
      
      # 2. TurboDiffusion (runs in /opt/TurboDiffusion)
      if [ -d "/opt/TurboDiffusion" ] && [ -f "$DGN_SOURCE_DIR/turbodiffusion_api_server.py" ]; then
          cp -v "$DGN_SOURCE_DIR/turbodiffusion_api_server.py" /opt/TurboDiffusion/api_server.py
      fi

      # 3. Wan2GP HTTP server (runs in /opt/wan2gp)
      if [ -d "/opt/wan2gp" ] && [ -f "$DGN_SOURCE_DIR/wan2gp_server.py" ]; then
          cp -v "$DGN_SOURCE_DIR/wan2gp_server.py" /opt/wan2gp/
      fi

      log "✓ Dynamic file sync complete"
  else
      log "WARNING: Could not find comfyui-storage in DGN client. Skipping file sync."
  fi

  # Start ComfyUI

# Fix VHS h264-mp4.json: force full PC color range to prevent washed-out colors
VHS_FMT="/opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/video_formats/h264-mp4.json"
if [ -f "$VHS_FMT" ]; then
    python3 -c "
import json
with open('$VHS_FMT') as f: d = json.load(f)
mp = d.get('main_pass', [])
color_flags = ['-color_range','2','-colorspace','1','-color_primaries','1','-color_trc','1']
if '-color_range' not in mp:
    mp.extend(color_flags)
    d['main_pass'] = mp
    with open('$VHS_FMT','w') as f: json.dump(d, f, indent=2)
    print('Patched h264-mp4.json: full color range enabled')
else:
    print('h264-mp4.json already patched')
" || log "WARNING: Failed to patch h264-mp4.json"
else
    log "INFO: VHS h264-mp4.json not found at $VHS_FMT (skipping color range patch)"
fi

# CRITICAL FIX for PyTorch / CUDA startup on GeForce cloud nodes.
#
# Some NVIDIA container runtimes inject /usr/local/cuda-*/compat into ldconfig.
# That forward-compat libcuda shim is intended for data-center GPUs; on GeForce
# cards (RTX 3090/4090/etc.) it can fail with CUDA error 804 even though the host
# driver is healthy. Prefer the host-mounted driver libcuda and the pip-installed
# CUDA component libraries before probing torch.cuda.
prefer_host_libcuda() {
  local host_libcuda="/usr/lib/x86_64-linux-gnu/libcuda.so.1"
  local disabled_compat="false"

  if [ -f "$host_libcuda" ]; then
    for conf in /etc/ld.so.conf.d/*.conf; do
      [ -f "$conf" ] || continue
      if grep -Eq '/usr/local/cuda(-[0-9.]+)?/compat' "$conf"; then
        if mv "$conf" "${conf}.disabled.$(date +%s).$$" 2>/dev/null; then
          disabled_compat="true"
        fi
      fi
    done

    if [ "$disabled_compat" = "true" ]; then
      ldconfig 2>/dev/null || true
      log "Disabled CUDA forward-compat libcuda path so PyTorch uses the host driver."
    fi
  fi
}

prefer_host_libcuda

PYTORCH_NVIDIA_LIBRARY_PATHS=$(python -c "import site; p=site.getsitepackages()[0]; print(':'.join([p + '/nvidia/nvjitlink/lib', p + '/nvidia/cusparse/lib', p + '/nvidia/cublas/lib', p + '/nvidia/cuda_runtime/lib']))")
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${PYTORCH_NVIDIA_LIBRARY_PATHS}:${LD_LIBRARY_PATH:-}"
log "Updated LD_LIBRARY_PATH for PyTorch compatibility: $LD_LIBRARY_PATH"

# --- Service Selection & Resource Management ---

# Defaults
START_HEARTMULA="false"
START_DIFFRHYTHM="false"
START_AUDIOX="false"
START_QWEN3TTS="false"
START_DIAGDISTILL="false"
START_WAN2GP="false"
START_COMFYUI="true"
START_SPARKVSR="false"
START_INSPATIO="false"
START_ERNIE_IMAGE="false"
ENABLE_4BIT="false"

TOTAL_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
log "Detected Total VRAM: ${TOTAL_VRAM_MB} MB"

# Determine which services to run
if [[ "${SERVICE_TYPE:-auto}" == "auto" ]]; then
  # AUTO MODE: Strict Priority Selection
  if [ -f "/app/heartmula_api.py" ]; then
      log "Auto-mode: Detected HeartMuLa image. Selecting HeartMuLa service."
      START_HEARTMULA="true"
      if [ "$TOTAL_VRAM_MB" -gt 22000 ]; then
          SERVICE_TYPE="heartmula-24gb"
          log "Auto-selected 24GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      else
          SERVICE_TYPE="heartmula-16gb"
          log "Auto-selected 16GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      fi
  elif [ -f "/app/diffrhythm_api.py" ]; then
      log "Auto-mode: Detected DiffRhythm image. Selecting DiffRhythm service."
      START_DIFFRHYTHM="true"
  elif [ -f "/app/audiox_api.py" ]; then
      log "Auto-mode: Detected AudioX image. Selecting AudioX service."
      START_AUDIOX="true"
      if [ "$TOTAL_VRAM_MB" -gt 22000 ]; then
          SERVICE_TYPE="audiox-24gb"
          log "Auto-selected AudioX 24GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      else
          SERVICE_TYPE="audiox-16gb"
          log "Auto-selected AudioX 16GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      fi
  elif [ -f "/app/qwen3_tts_api.py" ]; then
      log "Auto-mode: Detected Qwen3-TTS image. Selecting Qwen3-TTS service."
      START_QWEN3TTS="true"
      SERVICE_TYPE="qwen3-tts"
  elif [ -d "/opt/wan2gp" ]; then
      log "Auto-mode: Detected Wan2GP installation. Selecting Wan2GP backend."
      START_WAN2GP="true"
      MAGIHUMAN_DISTILL_TRANSFORMER="/opt/wan2gp/ckpts/magi_human_distill_quanto_bf16_int8.safetensors"
      MAGIHUMAN_BASE_TRANSFORMER="/opt/wan2gp/ckpts/magi_human_quanto_bf16_int8.safetensors"
      MAGIHUMAN_SR_TRANSFORMER="/opt/wan2gp/ckpts/magi_human_sr1080_quanto_bf16_int8.safetensors"
      SCAIL_TRANSFORMER_BF16="/opt/wan2gp/ckpts/wan2.1_scail_preview_14B_bf16.safetensors"
      SCAIL_TRANSFORMER_INT8="/opt/wan2gp/ckpts/wan2.1_scail_preview_14B_quanto_bf16_int8.safetensors"
      SCAIL_TRANSFORMER_FP16_INT8="/opt/wan2gp/ckpts/wan2.1_scail_preview_14B_quanto_fp16_int8.safetensors"
      VISTA4D_TRANSFORMER_BF16="/opt/wan2gp/ckpts/wan2.1_vista4d_384p49_14B_bf16.safetensors"
      VISTA4D_TRANSFORMER_INT8="/opt/wan2gp/ckpts/wan2.1_vista4d_384p49_14B_quanto_bf16_int8.safetensors"
      LTX23_Q4_TRANSFORMER="/opt/wan2gp/ckpts/ltx-2.3-22b-distilled-Q4_K_M_light.gguf"
      LTX23_Q6_TRANSFORMER="/opt/wan2gp/ckpts/ltx-2.3-22b-distilled-Q6_K_light.gguf"
      LTX23_Q8_TRANSFORMER="/opt/wan2gp/ckpts/ltx-2.3-22b-distilled-Q8_0_light.gguf"
      if [ -f "$VISTA4D_TRANSFORMER_BF16" ] || [ -f "$VISTA4D_TRANSFORMER_INT8" ]; then
          SERVICE_TYPE="vista4d-wan2gp-24gb"
          log "Auto-selected Vista4D Wan2GP 24GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      elif [ -f "$SCAIL_TRANSFORMER_BF16" ] || [ -f "$SCAIL_TRANSFORMER_INT8" ] || [ -f "$SCAIL_TRANSFORMER_FP16_INT8" ]; then
          if [ "$TOTAL_VRAM_MB" -gt 22000 ]; then
              SERVICE_TYPE="scail-wan2gp-24gb"
              log "Auto-selected SCAIL Wan2GP 24GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
          else
              SERVICE_TYPE="scail-wan2gp-16gb"
              log "Auto-selected SCAIL Wan2GP 16GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
          fi
      elif [ -f "$MAGIHUMAN_SR_TRANSFORMER" ] && { [ -f "$MAGIHUMAN_DISTILL_TRANSFORMER" ] || [ -f "$MAGIHUMAN_BASE_TRANSFORMER" ]; }; then
          if [ "$TOTAL_VRAM_MB" -gt 28000 ] && [ -f "$MAGIHUMAN_BASE_TRANSFORMER" ]; then
              SERVICE_TYPE="davinci-magihuman-32gb"
              log "Auto-selected daVinci-MagiHuman Wan2GP 32GB tier (base SR1080, VRAM: ${TOTAL_VRAM_MB}MB)"
          elif [ "$TOTAL_VRAM_MB" -gt 22000 ] && [ -f "$MAGIHUMAN_BASE_TRANSFORMER" ]; then
              SERVICE_TYPE="davinci-magihuman-24gb"
              log "Auto-selected daVinci-MagiHuman Wan2GP 24GB tier (base SR1080, VRAM: ${TOTAL_VRAM_MB}MB)"
          else
              SERVICE_TYPE="davinci-magihuman-16gb"
              log "Auto-selected daVinci-MagiHuman Wan2GP 16GB tier (distill SR1080, VRAM: ${TOTAL_VRAM_MB}MB)"
          fi
      elif [ -f "$LTX23_Q4_TRANSFORMER" ] && [ ! -f "$LTX23_Q6_TRANSFORMER" ] && [ ! -f "$LTX23_Q8_TRANSFORMER" ]; then
          SERVICE_TYPE="ltx23-video-8gb"
          log "Auto-selected LTX-2.3 8GB tier (Q4_K_M image detected, VRAM: ${TOTAL_VRAM_MB}MB)"
      elif [ -f "$LTX23_Q6_TRANSFORMER" ] && [ ! -f "$LTX23_Q8_TRANSFORMER" ]; then
          SERVICE_TYPE="ltx23-video-12gb"
          log "Auto-selected LTX-2.3 12GB tier (Q6_K image detected, VRAM: ${TOTAL_VRAM_MB}MB)"
      elif [ "$TOTAL_VRAM_MB" -gt 28000 ]; then
          SERVICE_TYPE="ltx23-video-32gb"
          log "Auto-selected LTX-2.3 32GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      elif [ "$TOTAL_VRAM_MB" -gt 18000 ]; then
          SERVICE_TYPE="ltx23-video-24gb"
          log "Auto-selected LTX-2.3 24GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      elif [ "$TOTAL_VRAM_MB" -lt 10000 ] && [ -f "$LTX23_Q4_TRANSFORMER" ]; then
          SERVICE_TYPE="ltx23-video-8gb"
          log "Auto-selected LTX-2.3 8GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      elif [ "$TOTAL_VRAM_MB" -lt 14000 ] && [ -f "$LTX23_Q6_TRANSFORMER" ]; then
          SERVICE_TYPE="ltx23-video-12gb"
          log "Auto-selected LTX-2.3 12GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      else
          SERVICE_TYPE="ltx23-video-16gb"
          log "Auto-selected LTX-2.3 16GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      fi
  elif [ -f "/opt/ComfyUI/models/checkpoints/ltx-2.3-22b-distilled.safetensors" ]; then
      log "Auto-mode: Detected LTX-2.3 ComfyUI image (BF16 safetensors present)."
      if [ "$TOTAL_VRAM_MB" -gt 20000 ]; then
          SERVICE_TYPE="ltx23-comfyui-video-24gb"
          log "Auto-selected LTX-2.3 ComfyUI 24GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      elif [ "$TOTAL_VRAM_MB" -gt 14000 ]; then
          SERVICE_TYPE="ltx23-comfyui-video-16gb"
          log "Auto-selected LTX-2.3 ComfyUI 16GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      elif [ "$TOTAL_VRAM_MB" -gt 10000 ]; then
          SERVICE_TYPE="ltx23-comfyui-video-12gb"
          log "Auto-selected LTX-2.3 ComfyUI 12GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      else
          SERVICE_TYPE="ltx23-comfyui-video-8gb"
          log "Auto-selected LTX-2.3 ComfyUI 8GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      fi
  elif [ -f "/opt/ComfyUI/models/DreamID-Omni/DreamID_Omni/dreamid_omni.fp8_e4m3fn.safetensors" ]; then
      log "Auto-mode: Detected DreamID-Omni FP8 ComfyUI image."
      SERVICE_TYPE="dreamid-omni-24gb"
      if [ "$TOTAL_VRAM_MB" -lt 22000 ]; then
          log "WARNING: DreamID-Omni is configured as a 24GB tier; detected only ${TOTAL_VRAM_MB}MB VRAM."
      else
          log "Auto-selected DreamID-Omni 24GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      fi
   elif [ -f "/app/sparkvsr_api.py" ] || [ -f "/opt/SparkVSR/sparkvsr_api.py" ]; then
        log "Auto-mode: Detected SparkVSR image. Selecting SparkVSR 24GB service."
        START_SPARKVSR="true"
        SERVICE_TYPE="sparkvsr-upscaler-24gb"
   elif [ -f "/app/inspatio_api.py" ] || [ -f "/opt/inspatio-world/inspatio_api.py" ]; then
       log "Auto-mode: Detected InSpatio-World image."
       START_INSPATIO="true"
       if [ "$TOTAL_VRAM_MB" -gt 20000 ]; then
           SERVICE_TYPE="inspatio-world-24gb"
           log "Auto-selected InSpatio-World 24GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
       else
           SERVICE_TYPE="inspatio-world-16gb"
           log "Auto-selected InSpatio-World 16GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
       fi
  elif [ -f "/app/ernie_image_api.py" ]; then
      log "Auto-mode: Detected ERNIE-Image image. Selecting ERNIE-Image service."
      START_ERNIE_IMAGE="true"
      if [ "$TOTAL_VRAM_MB" -gt 22000 ]; then
          SERVICE_TYPE="ernie-image-24gb"
          log "Auto-selected ERNIE-Image 24GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      elif [ "$TOTAL_VRAM_MB" -gt 14000 ]; then
          SERVICE_TYPE="ernie-image-16gb"
          log "Auto-selected ERNIE-Image 16GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      else
          SERVICE_TYPE="ernie-image-8gb"
          log "Auto-selected ERNIE-Image 8GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      fi
  else
      log "Auto-mode: No specialized API found. Defaulting to ComfyUI only."
  fi
else
  # MANUAL MODE: Check for keywords
  if [[ "$SERVICE_TYPE" == *"heartmula"* ]]; then START_HEARTMULA="true"; fi
  if [[ "$SERVICE_TYPE" == *"diffrhythm"* ]]; then START_DIFFRHYTHM="true"; fi
  if [[ "$SERVICE_TYPE" == *"audiox"* ]]; then START_AUDIOX="true"; fi
  if [[ "$SERVICE_TYPE" == *"qwen3-tts"* ]]; then START_QWEN3TTS="true"; fi
  if [[ "$SERVICE_TYPE" == *"diagdistill"* ]]; then START_DIAGDISTILL="true"; fi
  if [[ "$SERVICE_TYPE" == *"sparkvsr"* ]]; then START_SPARKVSR="true"; fi
  if [[ "$SERVICE_TYPE" == *"inspatio"* ]]; then START_INSPATIO="true"; fi
  if [[ "$SERVICE_TYPE" == *"ernie-image"* ]]; then START_ERNIE_IMAGE="true"; fi
  if [[ "$SERVICE_TYPE" == *"anima"* ]]; then START_COMFYUI="true"; fi
  # Wan2GP backend for LTX-2.3 Audio-Video services (NOT the ComfyUI variants),
  # daVinci-MagiHuman, SCAIL, and Vista4D services.
  if { [[ "$SERVICE_TYPE" == *"ltx23"* ]] && [[ "$SERVICE_TYPE" != *"comfyui"* ]]; } || [[ "$SERVICE_TYPE" == *"davinci"* ]] || [[ "$SERVICE_TYPE" == *"scail"* ]] || [[ "$SERVICE_TYPE" == *"vista4d"* ]]; then
      START_WAN2GP="true"
      log "Wan2GP service requested for ${SERVICE_TYPE}."
  fi
fi

# Resource constraints and VRAM management
if [ "$START_HEARTMULA" = "true" ]; then
  # HeartMuLa VRAM checks
  if [ "$TOTAL_VRAM_MB" -lt 20000 ]; then
      log "VRAM < 20GB. Disabling ComfyUI to reserve memory for HeartMuLa."
      START_COMFYUI="false"
  fi
  if [ "$TOTAL_VRAM_MB" -lt 16000 ]; then
      log "VRAM < 16GB. Enabling 4-bit quantization for HeartMuLa."
      ENABLE_4BIT="true"
  fi
  # If manually requested 8GB/4bit
  if [[ "${SERVICE_TYPE:-auto}" == *"8gb"* ]] || [[ "${SERVICE_TYPE:-auto}" == *"4bit"* ]]; then
      ENABLE_4BIT="true"
  fi
fi

if [ "$START_DIAGDISTILL" = "true" ]; then
  # DiagDistill (HunyuanVideo) needs full VRAM — always disable ComfyUI
  log "DiagDistill selected. Disabling ComfyUI to reserve VRAM for HunyuanVideo."
  START_COMFYUI="false"
fi

if [ "$START_AUDIOX" = "true" ]; then
  log "AudioX selected. Disabling ComfyUI to reserve VRAM."
  START_COMFYUI="false"
fi

if [ "$START_SPARKVSR" = "true" ]; then
  # SparkVSR needs full 24GB VRAM — always disable ComfyUI
  log "SparkVSR selected. Disabling ComfyUI to reserve VRAM."
  START_COMFYUI="false"
fi

if [ "$START_INSPATIO" = "true" ]; then
  log "InSpatio-World selected. Disabling ComfyUI to reserve VRAM."
  START_COMFYUI="false"
  if [[ "${SERVICE_TYPE:-auto}" == *"16gb"* ]]; then
      export INSPATIO_VRAM_GB=16
      log "InSpatio-World: INSPATIO_VRAM_GB=16 (CPU offloading enabled)"
  fi
fi

if [ "$START_ERNIE_IMAGE" = "true" ]; then
  log "ERNIE-Image selected. Disabling ComfyUI to reserve VRAM."
  START_COMFYUI="false"
fi

if [ "$START_QWEN3TTS" = "true" ]; then
  # Qwen3-TTS Model Selection
  # If 16GB service requested OR VRAM > 16GB, use 1.7B model
  if [[ "${SERVICE_TYPE:-auto}" == *"16gb"* ]] || [ "$TOTAL_VRAM_MB" -gt 16000 ]; then
      export QWEN_MODEL_SIZE="1.7B"
      log "Selecting Qwen3-TTS 1.7B model (Capacity: ${TOTAL_VRAM_MB}MB | Request: ${SERVICE_TYPE})"
  else
      export QWEN_MODEL_SIZE="0.6B"
      log "Selecting Qwen3-TTS 0.6B model for 8GB VRAM (Capacity: ${TOTAL_VRAM_MB}MB)"
  fi
fi

# Wan2GP backend (LTX-2.3 Audio-Video, daVinci-MagiHuman, SCAIL, and Vista4D)
if [ "$START_WAN2GP" = "true" ]; then
    # Wan2GP replaces ComfyUI for this service type
    log "Wan2GP backend selected. Disabling ComfyUI to reserve VRAM for Wan2GP."
    START_COMFYUI="false"

    if [[ "${SERVICE_TYPE:-}" == *"scail"* ]] && [[ "${SERVICE_TYPE:-}" == *"16gb"* ]]; then
        export WAN2GP_CLI_ARGS="${WAN2GP_CLI_ARGS:---profile 4.5 --attention sdpa --perc-reserved-mem-max 0.35 --vram-safety-coefficient 0.60}"
    elif [[ "${SERVICE_TYPE:-}" == *"scail"* ]] || [[ "${SERVICE_TYPE:-}" == *"vista4d"* ]]; then
        export WAN2GP_CLI_ARGS="${WAN2GP_CLI_ARGS:---profile 4.5 --attention sdpa --perc-reserved-mem-max 0.45 --vram-safety-coefficient 0.70}"
    elif [[ "${SERVICE_TYPE:-}" == *"davinci"* ]]; then
        if [[ "${SERVICE_TYPE:-}" == *"16gb"* ]]; then
            export WAN2GP_CLI_ARGS="${WAN2GP_CLI_ARGS:---profile 4.5 --attention sdpa --perc-reserved-mem-max 0.45 --vram-safety-coefficient 0.7}"
        elif [[ "${SERVICE_TYPE:-}" == *"32gb"* ]]; then
            export WAN2GP_CLI_ARGS="${WAN2GP_CLI_ARGS:---profile 4 --attention sdpa --perc-reserved-mem-max 0.55 --vram-safety-coefficient 0.80}"
        else
            export WAN2GP_CLI_ARGS="${WAN2GP_CLI_ARGS:---profile 4.5 --attention sdpa --perc-reserved-mem-max 0.45 --vram-safety-coefficient 0.70}"
        fi
    fi

    # Wan2GP images use PyTorch cu128 wheels and quantized model files.
    # Minimum requirement is CC >= 7.5 — the floor for the PyTorch 2.7 cu128 wheel
    # (sm_75 = T4/RTX 20xx, sm_80 = A100, sm_86 = RTX 30xx/A10G, sm_89 = RTX 40xx, sm_90 = H100).
    # Blackwell GPUs (SM 12.0) are forward-compatible via PTX JIT in the cu128 wheel.
    # Use Python (which has the actual PyTorch build info) rather than a raw CC check.
    WAN2GP_GPU_CHECK=$("$PYTHON_EXE" -c "
import sys
try:
    import torch
    if not torch.cuda.is_available():
        print('NO_CUDA')
        sys.exit(0)
    major, minor = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name(0)
    # NOTE: We intentionally do NOT check get_arch_list() here.
    # cu128 wheels use PTX intermediate code for forward-compat architectures,
    # so an SM may not appear in the SASS arch list yet still be fully compatible.
    # Hard minimum is CC 7.5 (PyTorch cu128 wheel floor).
    if major < 7 or (major == 7 and minor < 5):
        print('BELOW_MIN:{}.{}'.format(major, minor))
    else:
        print('OK:{}.{}:{}'.format(major, minor, gpu_name))
except Exception as e:
    print('ERROR:{}'.format(e))
" 2>/dev/null || echo "CHECK_FAILED")

    case "$WAN2GP_GPU_CHECK" in
        OK:*)
            _info="${WAN2GP_GPU_CHECK#OK:}"
            _cc=$(echo "$_info" | cut -d: -f1-2)
            _gpu=$(echo "$_info" | cut -d: -f3-)
            log "GPU '${_gpu}' CC ${_cc} — compatible with Wan2GP (PyTorch cu128 minimum: CC 7.5)."
            ;;
        BELOW_MIN:*)
            _cc="${WAN2GP_GPU_CHECK#BELOW_MIN:}"
            log "ERROR: Wan2GP requires compute capability 7.5+ (detected: ${_cc})."
            log "Supported GPUs: T4 (CC 7.5), A100 (CC 8.0), RTX 30xx/A10G (CC 8.6), RTX 40xx/L40S (CC 8.9), H100 (CC 9.0)."
            exit 1
            ;;
        NO_CUDA)
            log "ERROR: PyTorch cannot access CUDA, but Wan2GP requires a CUDA GPU."
            log "This is often a host driver/runtime mismatch or a CUDA forward-compat libcuda issue on GeForce hosts."
            exit 1
            ;;
        ERROR:*)
            _err="${WAN2GP_GPU_CHECK#ERROR:}"
            log "ERROR: Wan2GP GPU compatibility check failed: ${_err}"
            log "Refusing to register this provider because Wan2GP would not be able to start reliably."
            exit 1
            ;;
        *)
            log "ERROR: GPU compatibility check failed (${WAN2GP_GPU_CHECK})."
            log "Refusing to register this provider because Wan2GP would not be able to start reliably."
            exit 1
            ;;
    esac

    # Set Wan2GP environment variables
    export WAN2GP_ROOT="/opt/wan2gp"
    export WAN2GP_OUTPUT="/opt/wan2gp/outputs"
    export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
    if [ "${WAN2GP_STRICT_OFFLINE:-false}" = "true" ]; then
        export HF_HUB_OFFLINE="1"
        export TRANSFORMERS_OFFLINE="1"
        log "Wan2GP strict offline mode enabled."
    else
        export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
        export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
        log "Wan2GP runtime downloads allowed when baked cache files are missing (set WAN2GP_STRICT_OFFLINE=true to disable)."
    fi
    
    # Detailed diagnostics for Wan2GP installation
    log "Checking Wan2GP installation at $WAN2GP_ROOT..."
    if [ ! -d "$WAN2GP_ROOT" ]; then
        log "ERROR: Wan2GP directory not found at $WAN2GP_ROOT."
        log "This container image does not include Wan2GP installation."
        log "Please rebuild the image using:"
        log "  python client/comfyui-storage/build_and_push.py --build --push --hf-token <your-hf-token>"
        exit 1
    fi
    
    # Check for critical Wan2GP files/directories
    WAN2GP_CHECK_FAILED=0
    if [ ! -f "$WAN2GP_ROOT/shared/api.py" ] && [ ! -f "$WAN2GP_ROOT/shared/api/__init__.py" ]; then
        log "WARNING: Wan2GP shared.api module not found in $WAN2GP_ROOT/shared/."
        log "The processor imports 'from shared.api import init' — this path is required."
        WAN2GP_CHECK_FAILED=1
    fi
    
    if [ ! -d "$WAN2GP_ROOT/ckpts" ]; then
        log "WARNING: Wan2GP checkpoints directory not found at $WAN2GP_ROOT/ckpts."
        log "Models may not have been downloaded during build."
        WAN2GP_CHECK_FAILED=1
    fi

    if [[ "${SERVICE_TYPE:-}" == *"ltx23"* ]]; then
        LTX23_Q4_TRANSFORMER="$WAN2GP_ROOT/ckpts/ltx-2.3-22b-distilled-Q4_K_M_light.gguf"
        LTX23_Q6_TRANSFORMER="$WAN2GP_ROOT/ckpts/ltx-2.3-22b-distilled-Q6_K_light.gguf"
        LTX23_Q8_TRANSFORMER="$WAN2GP_ROOT/ckpts/ltx-2.3-22b-distilled-Q8_0_light.gguf"
        if [[ "$SERVICE_TYPE" == *"8gb"* ]]; then
            LTX23_REQUIRED_TRANSFORMER="$LTX23_Q4_TRANSFORMER"
            LTX23_EXPECTED_IMAGE="beschiak/openfork-ltx23-wan2gp-8gb:latest"
        elif [[ "$SERVICE_TYPE" == *"12gb"* ]]; then
            LTX23_REQUIRED_TRANSFORMER="$LTX23_Q6_TRANSFORMER"
            LTX23_EXPECTED_IMAGE="beschiak/openfork-ltx23-wan2gp-12gb:latest"
        # elif [[ "$SERVICE_TYPE" == *"32gb"* ]]; then
        #     LTX23_REQUIRED_TRANSFORMER="$LTX23_Q8_TRANSFORMER"
        #     LTX23_EXPECTED_IMAGE="beschiak/openfork-ltx23-wan2gp:latest"
        else
            LTX23_REQUIRED_TRANSFORMER="$LTX23_Q8_TRANSFORMER"
            LTX23_EXPECTED_IMAGE="beschiak/openfork-ltx23-wan2gp:latest"
        fi
        # LTX23_REQUIRED_TRANSFORMER="$LTX23_Q8_TRANSFORMER"
        # LTX23_EXPECTED_IMAGE="beschiak/openfork-ltx23-wan2gp-original:latest"

        if [ ! -f "$LTX23_REQUIRED_TRANSFORMER" ]; then
            log "ERROR: $SERVICE_TYPE requires $(basename "$LTX23_REQUIRED_TRANSFORMER"), but this image does not contain it."
            log "Use beschiak/openfork-ltx23-wan2gp-8gb:latest for the 8GB tier."
            log "Use beschiak/openfork-ltx23-wan2gp-12gb:latest for the 12GB tier."
            log "Use beschiak/openfork-ltx23-wan2gp-original:latest for the 16GB and 24GB tiers."
            log "Use beschiak/openfork-ltx23-wan2gp:latest for the 32GB tier."
            log "Expected image for this service: $LTX23_EXPECTED_IMAGE"
            WAN2GP_CHECK_FAILED=1
        fi
    fi

    if [[ "${SERVICE_TYPE:-}" == *"davinci"* ]]; then
        MAGIHUMAN_DISTILL_TRANSFORMER="$WAN2GP_ROOT/ckpts/magi_human_distill_quanto_bf16_int8.safetensors"
        MAGIHUMAN_BASE_TRANSFORMER="$WAN2GP_ROOT/ckpts/magi_human_quanto_bf16_int8.safetensors"
        MAGIHUMAN_SR_TRANSFORMER="$WAN2GP_ROOT/ckpts/magi_human_sr1080_quanto_bf16_int8.safetensors"
        MAGIHUMAN_TEXT_ENCODER="$WAN2GP_ROOT/ckpts/t5gemma-9b-9b-ul2/t5gemma-9b-9b-ul2_quanto_bf16_int8.safetensors"
        MAGIHUMAN_VAE="$WAN2GP_ROOT/ckpts/Wan2.2_VAE.safetensors"

        if [[ "$SERVICE_TYPE" == *"16gb"* ]]; then
            MAGIHUMAN_REQUIRED_TRANSFORMER="$MAGIHUMAN_DISTILL_TRANSFORMER"
            MAGIHUMAN_EXPECTED_IMAGE="beschiak/openfork-davinci-magihuman-wan2gp-16gb:latest"
        elif [[ "$SERVICE_TYPE" == *"32gb"* ]]; then
            MAGIHUMAN_REQUIRED_TRANSFORMER="$MAGIHUMAN_BASE_TRANSFORMER"
            MAGIHUMAN_EXPECTED_IMAGE="beschiak/openfork-davinci-magihuman-wan2gp-32gb:latest"
        else
            MAGIHUMAN_REQUIRED_TRANSFORMER="$MAGIHUMAN_BASE_TRANSFORMER"
            MAGIHUMAN_EXPECTED_IMAGE="beschiak/openfork-davinci-magihuman-wan2gp-24gb:latest"
        fi

        for required_file in "$MAGIHUMAN_REQUIRED_TRANSFORMER" "$MAGIHUMAN_SR_TRANSFORMER" "$MAGIHUMAN_TEXT_ENCODER" "$MAGIHUMAN_VAE"; do
            if [ ! -f "$required_file" ]; then
                log "ERROR: $SERVICE_TYPE requires $(basename "$required_file"), but this image does not contain it."
                log "Expected image for this service: $MAGIHUMAN_EXPECTED_IMAGE"
                WAN2GP_CHECK_FAILED=1
            fi
        done
    fi

    if [[ "${SERVICE_TYPE:-}" == *"scail"* ]]; then
        SCAIL_TRANSFORMER_BF16="$WAN2GP_ROOT/ckpts/wan2.1_scail_preview_14B_bf16.safetensors"
        SCAIL_TRANSFORMER_INT8="$WAN2GP_ROOT/ckpts/wan2.1_scail_preview_14B_quanto_bf16_int8.safetensors"
        SCAIL_TRANSFORMER_FP16_INT8="$WAN2GP_ROOT/ckpts/wan2.1_scail_preview_14B_quanto_fp16_int8.safetensors"
        SCAIL_VAE="$WAN2GP_ROOT/ckpts/Wan2.1_VAE.safetensors"
        SCAIL_TEXT_ENCODER="$WAN2GP_ROOT/ckpts/umt5-xxl/models_t5_umt5-xxl-enc-bf16.safetensors"
        SCAIL_TEXT_ENCODER_INT8="$WAN2GP_ROOT/ckpts/umt5-xxl/models_t5_umt5-xxl-enc-quanto_int8.safetensors"
        SCAIL_POSE_MODEL="$WAN2GP_ROOT/ckpts/pose/nlf_l_multi_0.3.2.eager.safetensors"
        SCAIL_POSE_META="$WAN2GP_ROOT/ckpts/pose/nlf_l_multi_0.3.2.eager.meta.json"
        if [[ "$SERVICE_TYPE" == *"16gb"* ]]; then
            SCAIL_EXPECTED_IMAGE="beschiak/openfork-scail-wan2gp-16gb:latest"
        else
            SCAIL_EXPECTED_IMAGE="beschiak/openfork-scail-wan2gp-24gb:latest"
        fi

        if [ ! -f "$SCAIL_TRANSFORMER_BF16" ] && [ ! -f "$SCAIL_TRANSFORMER_INT8" ] && [ ! -f "$SCAIL_TRANSFORMER_FP16_INT8" ]; then
            log "ERROR: $SERVICE_TYPE requires a SCAIL preview transformer, but this image does not contain one."
            log "Expected image for this service: $SCAIL_EXPECTED_IMAGE"
            WAN2GP_CHECK_FAILED=1
        fi

        if [ ! -f "$SCAIL_TEXT_ENCODER" ] && [ ! -f "$SCAIL_TEXT_ENCODER_INT8" ]; then
            log "ERROR: $SERVICE_TYPE requires a UMT5 text encoder, but this image does not contain one."
            log "Expected image for this service: $SCAIL_EXPECTED_IMAGE"
            WAN2GP_CHECK_FAILED=1
        fi

        for required_file in "$SCAIL_VAE" "$SCAIL_POSE_MODEL" "$SCAIL_POSE_META"; do
            if [ ! -f "$required_file" ]; then
                log "ERROR: $SERVICE_TYPE requires $(basename "$required_file"), but this image does not contain it."
                log "Expected image for this service: $SCAIL_EXPECTED_IMAGE"
                WAN2GP_CHECK_FAILED=1
            fi
        done
    fi

    if [[ "${SERVICE_TYPE:-}" == *"vista4d"* ]]; then
        VISTA4D_TRANSFORMER_BF16="$WAN2GP_ROOT/ckpts/wan2.1_vista4d_384p49_14B_bf16.safetensors"
        VISTA4D_TRANSFORMER_INT8="$WAN2GP_ROOT/ckpts/wan2.1_vista4d_384p49_14B_quanto_bf16_int8.safetensors"
        VISTA4D_VAE="$WAN2GP_ROOT/ckpts/Wan2.1_VAE.safetensors"
        VISTA4D_TEXT_ENCODER="$WAN2GP_ROOT/ckpts/umt5-xxl/models_t5_umt5-xxl-enc-bf16.safetensors"
        VISTA4D_TEXT_ENCODER_INT8="$WAN2GP_ROOT/ckpts/umt5-xxl/models_t5_umt5-xxl-enc-quanto_int8.safetensors"
        VISTA4D_DEPTH_MODEL="$WAN2GP_ROOT/ckpts/depth/depth_anything_v3_vitl_bf16.safetensors"
        VISTA4D_SAM_MODEL="$WAN2GP_ROOT/ckpts/sam3/sam3.1_multiplex_bf16.safetensors"
        VISTA4D_SAM_VOCAB="$WAN2GP_ROOT/ckpts/sam3/bpe_simple_vocab_16e6.txt.gz"
        VISTA4D_EXPECTED_IMAGE="beschiak/openfork-vista4d-wan2gp-24gb:latest"

        if [ ! -f "$VISTA4D_TRANSFORMER_BF16" ] && [ ! -f "$VISTA4D_TRANSFORMER_INT8" ]; then
            log "ERROR: $SERVICE_TYPE requires a Vista4D 384p49 transformer, but this image does not contain one."
            log "Expected image for this service: $VISTA4D_EXPECTED_IMAGE"
            WAN2GP_CHECK_FAILED=1
        fi

        if [ ! -f "$VISTA4D_TEXT_ENCODER" ] && [ ! -f "$VISTA4D_TEXT_ENCODER_INT8" ]; then
            log "ERROR: $SERVICE_TYPE requires a UMT5 text encoder, but this image does not contain one."
            log "Expected image for this service: $VISTA4D_EXPECTED_IMAGE"
            WAN2GP_CHECK_FAILED=1
        fi

        for required_file in "$VISTA4D_VAE" "$VISTA4D_DEPTH_MODEL" "$VISTA4D_SAM_MODEL" "$VISTA4D_SAM_VOCAB"; do
            if [ ! -f "$required_file" ]; then
                log "ERROR: $SERVICE_TYPE requires $(basename "$required_file"), but this image does not contain it."
                log "Expected image for this service: $VISTA4D_EXPECTED_IMAGE"
                WAN2GP_CHECK_FAILED=1
            fi
        done
    fi

    if [ "$WAN2GP_CHECK_FAILED" = "1" ]; then
        log "ERROR: Wan2GP installation is incomplete."
        log "Please rebuild the image with proper HF_TOKEN for model downloads."
        exit 1
    fi
    
    log "Wan2GP environment configured (WAN2GP_ROOT=$WAN2GP_ROOT)"

    # Start the Wan2GP HTTP server in the background.
    # It loads the model at startup (can take 10-20 min for the 22B model).
    # The DGN client processor polls /health and waits up to 30 min for it.
    WAN2GP_SERVER="/opt/wan2gp/wan2gp_server.py"
    WAN2GP_LOG_FILE="/tmp/wan2gp_server.log"
    if [ -f "$WAN2GP_SERVER" ]; then
        log "Starting Wan2GP HTTP server in background (logging to ${WAN2GP_LOG_FILE})..."
        : > "$WAN2GP_LOG_FILE"
        (cd "$WAN2GP_ROOT" && "$PYTHON_EXE" "$WAN2GP_SERVER" > "$WAN2GP_LOG_FILE" 2>&1) &
        WAN2GP_SERVER_PID=$!
        WAN2GP_STARTUP_TIMEOUT="${WAN2GP_STARTUP_TIMEOUT:-1800}"
        if ! wait_for_url "Wan2GP" "http://127.0.0.1:8188/health" "$WAN2GP_STARTUP_TIMEOUT" "$WAN2GP_LOG_FILE"; then
            if kill -0 "$WAN2GP_SERVER_PID" 2>/dev/null; then
                log "ERROR: Wan2GP server process is still running but did not expose /health."
            else
                log "ERROR: Wan2GP server process exited before readiness."
            fi
            log "ERROR: Wan2GP failed to become ready; not starting the DGN client."
            exit 1
        fi
    else
        log "ERROR: wan2gp_server.py not found at $WAN2GP_SERVER. Cannot start Wan2GP HTTP server."
        exit 1
    fi
fi

# Fix AudioVAE(sd) → AudioVAE(sd, metadata) in low_vram_loaders.py.
# ComfyUI's audio_vae.py now asserts metadata is present; older built images
# had a Dockerfile patch that stripped the metadata arg — restore it here.
LTX_LOADERS="/opt/ComfyUI/custom_nodes/ComfyUI-LTXVideo/low_vram_loaders.py"
if [ -f "$LTX_LOADERS" ] && grep -q "AudioVAE(sd)" "$LTX_LOADERS"; then
    sed -i 's/AudioVAE(sd)/AudioVAE(sd, metadata)/g' "$LTX_LOADERS"
    log "Patched AudioVAE call in low_vram_loaders.py to include metadata"
fi

# Start ComfyUI
if [ -d "/opt/ComfyUI" ] && [ "$START_COMFYUI" = "true" ]; then
  # Determine ComfyUI launch flags based on SERVICE_TYPE
  COMFY_FLAGS="--listen 0.0.0.0 --port 8188"
  
  case "$SERVICE_TYPE" in
    # LTX-2.3 ComfyUI variants — must appear BEFORE generic ltx2*/8gb patterns
    # Uses --use-pytorch-cross-attention (better than xformers for offloaded 46GB model)
    # and --cpu-vae on 8/16GB to keep VRAM free for attention activations
    *ltx23*comfyui*8gb*|*ltx23-comfyui*-8gb*)
      log "Applying 8GB optimizations for LTX-2.3 ComfyUI (512x288, 65 frames)"
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --cpu-vae --reserve-vram 0.5 --use-pytorch-cross-attention --cache-none"
      ;;
    *ltx23*comfyui*12gb*|*ltx23-comfyui*-12gb*)
      log "Applying 12GB optimizations for LTX-2.3 ComfyUI (544x304, 81 frames)"
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --cpu-vae --reserve-vram 0.5 --use-pytorch-cross-attention --cache-none"
      ;;
    *ltx23*comfyui*16gb*|*ltx23-comfyui*-16gb*)
      log "Applying 16GB optimizations for LTX-2.3 ComfyUI (576x320, 97 frames)"
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --cpu-vae --reserve-vram 0.5 --use-pytorch-cross-attention --cache-none"
      ;;
    *ltx23*comfyui*24gb*|*ltx23-comfyui*-24gb*)
      log "Applying 24GB optimizations for LTX-2.3 ComfyUI (768x432, 121 frames)"
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --reserve-vram 1.0 --use-pytorch-cross-attention"
      ;;
    *dreamid*omni*|*dreamid-omni*)
      log "Applying 24GB optimizations for DreamID-Omni FP8"
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --reserve-vram 1.0 --use-pytorch-cross-attention --cache-none"
      ;;
    *ltx23*-8gb*|*ltx2*-8gb*|*8gb*)
      log "Applying AGGRESSIVE 8GB VRAM optimizations for ComfyUI"
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --fp32-vae --disable-smart-memory --reserve-vram 1.0 --cache-none --force-fp16 --use-split-cross-attention --preview-method none"
      ;;
    *ltx23*-16gb*|*ltx2*-16gb*|*16gb*)
      log "Applying 16GB VRAM optimizations for ComfyUI (lowvram mode for model offloading)"
      # IMPORTANT: Use --lowvram because total model size (~29.5GB) far exceeds 16GB VRAM
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --reserve-vram 1.0 --use-split-cross-attention --cache-none"
      ;;
    *ltx23*-24gb*|*ltx2*-24gb*|*24gb*)
      log "Applying 24GB VRAM optimizations for ComfyUI (GGUF Q8_0 model)"
      # GGUF Q8_0 (~20.4GB) + Gemma FP8 (~6GB) = ~26.4GB total, slightly over 24GB VRAM
      # Use --lowvram to allow CPU offloading when needed
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --reserve-vram 1.5 --use-pytorch-cross-attention"
      ;;
    *hunyuan*)
      log "Applying Hunyuan optimizations (16GB+)"
      # Hunyuan is very heavy (especially FP16 T2V).
      # Disable smart memory to force aggressive unloading.
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --reserve-vram 1.0 --use-split-cross-attention --cache-none"
      ;;
    *anima*)
      log "Applying Anima optimizations"
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --reserve-vram 1.0"
      ;;
    *)
      # Default to lowvram if service type is unknown but potentially heavy
      if [[ "$SERVICE_TYPE" == *"video"* ]]; then
        COMFY_FLAGS="$COMFY_FLAGS --lowvram"
      else
        COMFY_FLAGS="$COMFY_FLAGS --highvram"
      fi
      ;;
  esac

  # Check if ComfyUI is already starting (port 8188 bound)
  PORT_BOUND=false
  if command -v netstat &> /dev/null; then
    if netstat -tln | grep -q ":8188 "; then PORT_BOUND=true; fi
  elif command -v ss &> /dev/null; then
     if ss -tln | grep -q ":8188 "; then PORT_BOUND=true; fi
  else
     # Fallback: check /proc/net/tcp for port 1FFC (8188 in hex)
     if grep -q "1FFC" /proc/net/tcp; then PORT_BOUND=true; fi
  fi

  if [ "$PORT_BOUND" = "true" ]; then
    log "Port 8188 is already bound. Assuming ComfyUI is starting or running. Skipping redundant startup."
  else
    log "Starting ComfyUI in background..."
    log "ComfyUI Flags: $COMFY_FLAGS"
    (cd /opt/ComfyUI && "$PYTHON_EXE" main.py $COMFY_FLAGS > /tmp/comfyui.log 2>&1) &
    log "ComfyUI startup initiated (logging to /tmp/comfyui.log)"
  fi
else
  log "Info: /opt/ComfyUI not found."
fi

# Start DiffRhythm REST API
if [ "$START_DIFFRHYTHM" = "true" ] && [ -f "/app/diffrhythm_api.py" ]; then
  log "Found DiffRhythm API script. Starting..."
  (cd /app && "$PYTHON_EXE" diffrhythm_api.py > /tmp/diffrhythm_api.log 2>&1) &
  wait_for_url "DiffRhythm API" "http://127.0.0.1:8000/health" 120 "/tmp/diffrhythm_api.log"
fi

# Start AudioX REST API
if [ "$START_AUDIOX" = "true" ] && [ -f "/app/audiox_api.py" ]; then
  log "Found AudioX API script. Starting..."
  export AUDIOX_MODEL_HALF="${AUDIOX_MODEL_HALF:-true}"
  export AUDIOX_MAX_DURATION_SECONDS="${AUDIOX_MAX_DURATION_SECONDS:-10}"
  (cd /app && "$PYTHON_EXE" audiox_api.py > /tmp/audiox_api.log 2>&1) &
  wait_for_url "AudioX API" "http://127.0.0.1:8000/health" 600 "/tmp/audiox_api.log"
fi

# Start Qwen3-TTS REST API
if [ "$START_QWEN3TTS" = "true" ] && [ -f "/app/qwen3_tts_api.py" ]; then
  log "Found Qwen3-TTS API script. Starting..."
  (cd /app && "$PYTHON_EXE" qwen3_tts_api.py > /tmp/qwen3_tts_api.log 2>&1) &
  wait_for_url "Qwen3-TTS API" "http://127.0.0.1:8000/health" 300 "/tmp/qwen3_tts_api.log"
fi

# Start HeartMuLa REST API
if [ "$START_HEARTMULA" = "true" ] && [ -f "/app/heartmula_api.py" ]; then
  log "Found HeartMuLa API script. Starting..."

  # Check if heartlib is installed, if not install it
  if ! "$PYTHON_EXE" -c "import heartlib" 2>/dev/null; then
      log "HeartLib not found. Installing..."
      if [ ! -d "/app/heartlib_repo" ]; then
          log "Cloning HeartLib..."
          git clone https://github.com/HeartMuLa/heartlib.git /app/heartlib_repo || true
      fi
      
      if [ -d "/app/heartlib_repo" ]; then
          log "Installing HeartLib from /app/heartlib_repo..."
          (cd /app/heartlib_repo && "$PYTHON_EXE" -m pip install -e .) || log "WARNING: HeartLib install failed"
      else
          log "ERROR: Could not clone HeartLib"
      fi
  else
      log "HeartLib is already installed."
  fi

  
  # CUDA memory allocation optimization - reduces fragmentation for tight VRAM scenarios
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"
  log "Setting PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
  
  # Check if 4-bit quantization is requested or auto-enabled
  # NOTE: 16GB VRAM REQUIRES 4-bit quantization! 8-bit uses ~15.64GB leaving no room for inference.
  if [ "$ENABLE_4BIT" = "true" ]; then
      export HEARTMULA_QUANTIZATION="4bit"
      log "Enabling 4-bit quantization for HeartMuLa (Auto or Requested)"
  elif [[ "$SERVICE_TYPE" == *"heartmula-16gb"* ]]; then
      export HEARTMULA_QUANTIZATION="4bit"
      log "Enabling 4-bit quantization for HeartMuLa 16GB service (required for 16GB VRAM)"
  fi
  
  (cd /app && "$PYTHON_EXE" heartmula_api.py > /tmp/heartmula_api.log 2>&1) &
  
  # Wait for API with diagnostic fallback
  if ! wait_for_url "HeartMuLa API" "http://127.0.0.1:8000/health" 900 "/tmp/heartmula_api.log"; then
      log "❌ HeartMuLa failed to start within timeout."
      if [ -x "/app/diagnose_heartmula.sh" ]; then
          log "Running diagnostic script..."
          /app/diagnose_heartmula.sh | tee -a "$LOG_FILE"
      elif [ -f "/app/diagnose_heartmula.sh" ]; then
          bash /app/diagnose_heartmula.sh | tee -a "$LOG_FILE"
      else
          log "Diagnostic script not found at /app/diagnose_heartmula.sh"
      fi
      exit 1
  fi
fi

# Start DiagDistill REST API
if [ "$START_DIAGDISTILL" = "true" ]; then
  log "Starting DiagDistill API service..."
  
  # Check if DiagDistill API exists in either location
  DIAGDISTILL_API=""
  if [ -f "/app/diagdistill_api.py" ]; then
    DIAGDISTILL_API="/app/diagdistill_api.py"
    DIAGDISTILL_CD="/app"
  elif [ -f "/opt/DiagDistill/diagdistill_api.py" ]; then
    DIAGDISTILL_API="/opt/DiagDistill/diagdistill_api.py"
    DIAGDISTILL_CD="/opt/DiagDistill"
  fi
  
  if [ -n "$DIAGDISTILL_API" ]; then
    log "Found DiagDistill API at $DIAGDISTILL_API. Starting..."
    (cd "$DIAGDISTILL_CD" && "$PYTHON_EXE" diagdistill_api.py > /tmp/diagdistill_api.log 2>&1) &
    wait_for_url "DiagDistill API" "http://127.0.0.1:8000/health" 600 "/tmp/diagdistill_api.log"
  else
    log "ERROR: DiagDistill API not found at /app/diagdistill_api.py or /opt/DiagDistill/diagdistill_api.py"
  fi
fi

# Start TurboDiffusion REST API (skip if DiagDistill already claimed port 8000)
if [ "$START_DIAGDISTILL" != "true" ] && [ -f "/opt/TurboDiffusion/api_server.py" ]; then
  log "Found TurboDiffusion API script. Starting..."
  (cd /opt/TurboDiffusion && "$PYTHON_EXE" api_server.py > /tmp/turbodiffusion_api.log 2>&1) &
  wait_for_url "TurboDiffusion API" "http://127.0.0.1:8000/health" 120 "/tmp/turbodiffusion_api.log"
fi

# daVinci-MagiHuman is served by the Wan2GP block above. The legacy REST/FP8
# startup path was intentionally removed because its FP8 image is not a
# reliable 24GB-tier deployment target.

# Start SparkVSR REST API
if [ "$START_SPARKVSR" = "true" ]; then
  log "Starting SparkVSR API service..."
  SPARKVSR_API=""
  if [ -f "/app/sparkvsr_api.py" ]; then
    SPARKVSR_API="/app/sparkvsr_api.py"
    SPARKVSR_CD="/app"
  elif [ -f "/opt/SparkVSR/sparkvsr_api.py" ]; then
    SPARKVSR_API="/opt/SparkVSR/sparkvsr_api.py"
    SPARKVSR_CD="/opt/SparkVSR"
  fi

  if [ -n "$SPARKVSR_API" ]; then
    # ── Download model weights if not already cached ──────────────────────────
    # Weights live in HF_HOME (default /models/hf_cache). On a vast.ai instance
    # with a persistent volume mounted at /models this download only happens once.
    HF_CACHE_DIR="${HF_HOME:-/models/hf_cache}/hub"
    SPARKVSR_CACHE="$HF_CACHE_DIR/models--JiongzeYu--SparkVSR"
    COGVIDEO_CACHE="$HF_CACHE_DIR/models--THUDM--CogVideoX1.5-5B-I2V"

    mkdir -p "$HF_CACHE_DIR"

    if [ ! -d "$SPARKVSR_CACHE" ] || [ -z "$(ls -A "$SPARKVSR_CACHE" 2>/dev/null)" ]; then
      log "SparkVSR weights not found in cache. Downloading JiongzeYu/SparkVSR (~3 GB)..."
      huggingface-cli download JiongzeYu/SparkVSR \
        --cache-dir "$HF_CACHE_DIR" 2>&1 | tee -a "$LOG_FILE" || \
        log "WARNING: SparkVSR weight download failed — API will retry at startup."
    else
      log "SparkVSR weights found at $SPARKVSR_CACHE. Skipping download."
    fi

    if [ ! -d "$COGVIDEO_CACHE" ] || [ -z "$(ls -A "$COGVIDEO_CACHE" 2>/dev/null)" ]; then
      log "CogVideoX1.5-5B-I2V weights not found in cache. Downloading THUDM/CogVideoX1.5-5B-I2V (~10–15 GB, safetensors only)..."
      huggingface-cli download THUDM/CogVideoX1.5-5B-I2V \
        --cache-dir "$HF_CACHE_DIR" \
        --ignore-patterns "*.bin" 2>&1 | tee -a "$LOG_FILE" || \
        log "WARNING: CogVideoX weight download failed — API will retry at startup."
    else
      log "CogVideoX1.5-5B-I2V weights found at $COGVIDEO_CACHE. Skipping download."
    fi
    # ─────────────────────────────────────────────────────────────────────────

    # Check if port 8000 is already bound (e.g. by the Docker CMD running sparkvsr_api.py)
    PORT8000_BOUND=false
    if command -v netstat &> /dev/null; then
      if netstat -tln | grep -q ":8000 "; then PORT8000_BOUND=true; fi
    elif command -v ss &> /dev/null; then
      if ss -tln | grep -q ":8000 "; then PORT8000_BOUND=true; fi
    fi

    if [ "$PORT8000_BOUND" = "true" ]; then
      log "Port 8000 already bound — SparkVSR API is already starting (Docker CMD). Skipping redundant launch."
    else
      log "Found SparkVSR API at $SPARKVSR_API. Starting..."
      (cd "$SPARKVSR_CD" && "$PYTHON_EXE" sparkvsr_api.py > /tmp/sparkvsr_api.log 2>&1) &
    fi
    # Extended timeout: first-run model load can take several minutes after the download above
    wait_for_url "SparkVSR API" "http://127.0.0.1:8000/health" 600 "/tmp/sparkvsr_api.log"
  else
    log "ERROR: SparkVSR API not found at /app/sparkvsr_api.py or /opt/SparkVSR/sparkvsr_api.py"
  fi
fi

# Start InSpatio-World REST API
if [ "$START_INSPATIO" = "true" ]; then
  log "Starting InSpatio-World API service..."
  INSPATIO_API=""
  if [ -f "/app/inspatio_api.py" ]; then
    INSPATIO_API="/app/inspatio_api.py"
    INSPATIO_CD="/app"
  elif [ -f "/opt/inspatio-world/inspatio_api.py" ]; then
    INSPATIO_API="/opt/inspatio-world/inspatio_api.py"
    INSPATIO_CD="/opt/inspatio-world"
  fi

  if [ -n "$INSPATIO_API" ]; then
    export INSPATIO_ROOT="${INSPATIO_ROOT:-/opt/inspatio-world}"
    export INSPATIO_CHECKPOINTS="${INSPATIO_CHECKPOINTS:-${INSPATIO_ROOT}/checkpoints}"

    if [ ! -d "$INSPATIO_CHECKPOINTS" ] || [ -z "$(ls -A "$INSPATIO_CHECKPOINTS" 2>/dev/null)" ]; then
      log "InSpatio-World checkpoints not found. Please ensure the Docker image includes model weights."
    else
      log "InSpatio-World checkpoints found at $INSPATIO_CHECKPOINTS"
    fi

    log "Found InSpatio-World API at $INSPATIO_API. Starting..."
    (cd "$INSPATIO_CD" && "$PYTHON_EXE" inspatio_api.py > /tmp/inspatio_api.log 2>&1) &
    wait_for_url "InSpatio-World API" "http://127.0.0.1:8000/health" 600 "/tmp/inspatio_api.log"
  else
    log "ERROR: InSpatio-World API not found at /app/inspatio_api.py or /opt/inspatio-world/inspatio_api.py"
  fi
fi

# Start ERNIE-Image REST API
if [ "$START_ERNIE_IMAGE" = "true" ]; then
  log "Starting ERNIE-Image API service..."
  if [ -f "/app/ernie_image_api.py" ]; then
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export ERNIE_ALLOW_MODEL_DOWNLOAD="${ERNIE_ALLOW_MODEL_DOWNLOAD:-false}"
    export ERNIE_MODEL_ID="${ERNIE_MODEL_ID:-baidu/ERNIE-Image}"
    export ERNIE_DEFAULT_STEPS="${ERNIE_DEFAULT_STEPS:-50}"
    if [[ "${SERVICE_TYPE}" == *"8gb"* ]]; then
      export ERNIE_MODEL_ID="baidu/ERNIE-Image-Turbo"
      export ERNIE_DEFAULT_STEPS="8"
      # bf16 is required: fp16 causes NaN latents in the transformer on this model,
      # producing solid black output. bf16 upcasts the affected ops to fp32 internally.
      export ERNIE_DTYPE="bf16"
      export ERNIE_USE_PE="false"
      export ERNIE_ENABLE_CPU_OFFLOAD="true"
      export ERNIE_ENABLE_ATTENTION_SLICING="true"
      export ERNIE_ENABLE_VAE_TILING="true"
    elif [[ "${SERVICE_TYPE}" == *"16gb"* ]]; then
      export ERNIE_MODEL_ID="baidu/ERNIE-Image"
      export ERNIE_DEFAULT_STEPS="38"
      # bf16 is required: fp16 causes NaN latents in the transformer on this model,
      # producing solid black output. bf16 upcasts the affected ops to fp32 internally.
      export ERNIE_DTYPE="bf16"
      export ERNIE_USE_PE="false"
      export ERNIE_ENABLE_CPU_OFFLOAD="true"
      export ERNIE_ENABLE_ATTENTION_SLICING="true"
      export ERNIE_ENABLE_VAE_TILING="true"
    else
      export ERNIE_DTYPE="bf16"
      export ERNIE_USE_PE="true"
      export ERNIE_ENABLE_CPU_OFFLOAD="false"
      export ERNIE_ENABLE_ATTENTION_SLICING="false"
      export ERNIE_ENABLE_VAE_TILING="true"
    fi
    log "ERNIE-Image config: model=$ERNIE_MODEL_ID dtype=$ERNIE_DTYPE steps=$ERNIE_DEFAULT_STEPS use_pe=${ERNIE_USE_PE:-true} cpu_offload=${ERNIE_ENABLE_CPU_OFFLOAD:-false}"
    (cd /app && "$PYTHON_EXE" ernie_image_api.py > /tmp/ernie_image_api.log 2>&1) &
    wait_for_url "ERNIE-Image API" "http://127.0.0.1:8000/health" 600 "/tmp/ernie_image_api.log"
  else
    log "ERROR: ERNIE-Image API not found at /app/ernie_image_api.py"
  fi
fi

# Start Ollama server
if command -v ollama &> /dev/null; then
  log "Found Ollama. Starting server..."
  export OLLAMA_HOST=0.0.0.0
  export OLLAMA_ORIGINS="*"
  ollama serve > /tmp/ollama.log 2>&1 &
  wait_for_url "Ollama" "http://127.0.0.1:11434/api/tags" 60 "/tmp/ollama.log"
fi

# --- Final Checks & Execution ---

# Wait for ComfyUI to be ready (if it was started or is running)
if [ -d "/opt/ComfyUI" ]; then
  # Determine timeout based on service tier
  WAIT_TIME=120
  if [[ "$SERVICE_TYPE" == *"24gb"* ]] || [[ "$SERVICE_TYPE" == *"ltx23"* ]] || [[ "$SERVICE_TYPE" == *"ltx2"* ]]; then
    WAIT_TIME=600
    log "Large model detected ($SERVICE_TYPE). Extending ComfyUI readiness timeout to ${WAIT_TIME}s."
  fi
  
  wait_for_url "ComfyUI" "http://127.0.0.1:8188/system_stats" "$WAIT_TIME" "/tmp/comfyui.log"

  # Dump ComfyUI startup log so import errors are visible in cloud logs
  if [ -f "/tmp/comfyui.log" ]; then
    log "=== ComfyUI startup log (first 100 lines) ==="
    head -n 100 /tmp/comfyui.log | sed 's/^/  [comfyui] /' || true
    log "=== ComfyUI import errors (if any) ==="
    grep -i "cannot import\|failed to import\|error importing\|traceback\|modulenotfounderror\|importerror" /tmp/comfyui.log | sed 's/^/  [comfyui] /' || log "  (no import errors found)"
    log "=== End ComfyUI startup log ==="
  fi

  # Verify critical LTX-2.3 audio nodes are registered (diagnose missing_node_type errors)
  if [[ "${SERVICE_TYPE:-}" == *"ltx23"* ]]; then
    log "Checking LTX-2.3 audio node availability..."
    for node in LTXAVTextEncoderLoader LTXVAudioVAELoader LTXVEmptyLatentAudio LTXVAudioVAEDecode LTXVConcatAVLatent LTXVSeparateAVLatent; do
      result=$(curl -s "http://127.0.0.1:8188/object_info/$node" 2>/dev/null || echo "{}")
      if echo "$result" | grep -q "\"$node\""; then
        log "  [OK] $node is registered"
      else
        log "  [MISSING] $node is NOT registered — check ComfyUI import errors above"
      fi
    done
  fi

  if [[ "${SERVICE_TYPE:-}" == *"dreamid"* ]]; then
    log "Checking DreamID-Omni node availability..."
    for node_path in "ComfyUI%20DreamID-Omni%20Loader" "ComfyUI%20DreamID-Omni%20Sampler"; do
      result=$(curl -s "http://127.0.0.1:8188/object_info/$node_path" 2>/dev/null || echo "{}")
      if echo "$result" | grep -q "DreamID-Omni"; then
        log "  [OK] ${node_path//%20/ } is registered"
      else
        log "  [MISSING] ${node_path//%20/ } is NOT registered — check ComfyUI import errors above"
      fi
    done
  fi
fi

# Save restart configuration
log "Saving restart configuration..."

ACCEPT_POLICY="${ACCEPT_POLICY:-all}"
DGN_CLIENT_ARGS=(
  --dgn-api-key "$DGN_API_KEY"
  --service "${SERVICE_TYPE:-auto}"
  --root-dir /opt/dgn-client
  --data-dir /data
)

case "$ACCEPT_POLICY" in
  monetize)
    log "Routing policy: monetize"
    DGN_CLIENT_ARGS+=(--community-mode none --monetize-mode)
    ;;
  all)
    log "Routing policy: all"
    DGN_CLIENT_ARGS+=(--community-mode all)
    ;;
  mine)
    log "Routing policy: mine"
    DGN_CLIENT_ARGS+=(--community-mode none --process-own-jobs)
    ;;
  users)
    log "Routing policy: users"
    DGN_CLIENT_ARGS+=(--community-mode trusted_users)
    if [ -n "$ALLOWED_TARGETS" ]; then
      DGN_CLIENT_ARGS+=(--allowed-targets "$ALLOWED_TARGETS")
    fi
    ;;
  project)
    log "Routing policy: project"
    DGN_CLIENT_ARGS+=(--community-mode trusted_projects)
    if [ -n "$ALLOWED_TARGETS" ]; then
      DGN_CLIENT_ARGS+=(--allowed-targets "$ALLOWED_TARGETS")
    fi
    ;;
  *)
    log "WARNING: Unknown ACCEPT_POLICY='$ACCEPT_POLICY', defaulting to all"
    DGN_CLIENT_ARGS+=(--community-mode all)
    ;;
esac

printf -v DGN_CLIENT_ARGS_STRING '%q ' "${DGN_CLIENT_ARGS[@]}"
cat > /opt/dgn-client/.restart-config << RESTART_CONFIG_EOF
export DGN_CLIENT_ARGS="$DGN_CLIENT_ARGS_STRING"
export ORCHESTRATOR_URL_PROD="${DGN_ORCHESTRATOR_URL:-https://openfork.video}"
RESTART_CONFIG_EOF

chmod +x /opt/dgn-client/.restart-config

# Save credentials for future reboot persistence
if [ -n "$DGN_API_KEY" ]; then
    echo "$DGN_API_KEY" > /etc/dgn-api-key
fi

log "Starting DGN client..."
cd /opt/dgn-client
export ORCHESTRATOR_URL_PROD="${DGN_ORCHESTRATOR_URL:-https://openfork.video}"

# Test that imports work before running
log "Testing Python imports..."
"$PYTHON_EXE" -c "
import sys
import os
sys.path.insert(0, '/opt/dgn-client')
try:
    from config import HEADLESS_MODE
    from dgn_client import DGNClient
    import services.processors as processors
    print('DGNClient import successful')
    print(f'Client cwd: {os.getcwd()}')
    print(f'Processor module: {getattr(processors, \"__file__\", \"unknown\")}')
    processor_allowlist = getattr(processors, '__all__', [])
    print(f'Processor allowlist count: {len(processor_allowlist)}')
    print(f'Processor allowlist has SCAILImageToVideoProcessor: {\"SCAILImageToVideoProcessor\" in processor_allowlist}')
    print(f'Processor allowlist has Vista4DVideoToVideoProcessor: {\"Vista4DVideoToVideoProcessor\" in processor_allowlist}')
except Exception as e:
    print(f'Import error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" || (log "ERROR: Python imports failed." && exit 1)

"$PYTHON_EXE" cli.py \
  "${DGN_CLIENT_ARGS[@]}" 2>&1 | tee -a /tmp/dgn_client.log
