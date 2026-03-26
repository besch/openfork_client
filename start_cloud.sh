#!/bin/bash
# OpenFork DGN Client Cloud Startup Script (start_cloud.sh)
# This script is fetched and run by cloud containers (RunPod, Vast.ai)

# --- Help ---
show_help() {
  cat << EOF
OpenFork DGN Client Cloud Startup Script

USAGE:
  curl -sL https://raw.githubusercontent.com/besch/openfork_client/main/start_cloud.sh | bash

  Or with environment variables:
  DGN_API_KEY=xxx SERVICE_TYPE=ltx2-video-8gb ./start_cloud.sh

ENVIRONMENT VARIABLES:
  Required:
    DGN_API_KEY           API key for DGN client authentication

  Optional:
    SERVICE_TYPE          Service type to advertise (default: auto)
    DGN_ORCHESTRATOR_URL  Orchestrator URL (default: https://openfork.video)
    HEADLESS_MODE         Set to "true" for headless operation
    ACCEPT_POLICY         Job acceptance policy: all, mine, users, project (default: all)
    SAVE_LOGS             Set to "true" to stream logs to webhook
    BUILD_JOB_ID          Build job ID for log association (uses provider ID if not set)

EXAMPLES:
  # Basic startup
  DGN_API_KEY=dgn_xxx ./start_cloud.sh

  # With specific service type
  DGN_API_KEY=dgn_xxx SERVICE_TYPE=ltx2-video-16gb ./start_cloud.sh

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
    if curl -s "$url" > /dev/null 2>&1; then
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
if curl --max-time 60 --progress-bar -L https://raw.githubusercontent.com/besch/openfork_client/main/bootstrap.sh -o bootstrap.sh; then
  log "✓ bootstrap.sh downloaded successfully"
  bash bootstrap.sh
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
          [ -f "$DGN_SOURCE_DIR/qwen3_tts_api.py" ] && cp -v "$DGN_SOURCE_DIR/qwen3_tts_api.py" /app/
          [ -f "$DGN_SOURCE_DIR/diagdistill_api.py" ] && cp -v "$DGN_SOURCE_DIR/diagdistill_api.py" /app/
          [ -f "$DGN_SOURCE_DIR/stream_diffvsr_wrapper.py" ] && cp -v "$DGN_SOURCE_DIR/stream_diffvsr_wrapper.py" /app/
      fi
      
      # 2. TurboDiffusion (runs in /opt/TurboDiffusion)
      if [ -d "/opt/TurboDiffusion" ] && [ -f "$DGN_SOURCE_DIR/turbodiffusion_api_server.py" ]; then
          cp -v "$DGN_SOURCE_DIR/turbodiffusion_api_server.py" /opt/TurboDiffusion/api_server.py
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

# CRITICAL FIX for PyTorch 2.4+ / CUDA 12 mismatch errors
# We must prioritize the pip-installed nvidia libraries over system libraries
export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib:' + site.getsitepackages()[0] + '/nvidia/cusparse/lib:' + site.getsitepackages()[0] + '/nvidia/cublas/lib:' + site.getsitepackages()[0] + '/nvidia/cuda_runtime/lib')"):$LD_LIBRARY_PATH
log "Updated LD_LIBRARY_PATH for PyTorch compatibility: $LD_LIBRARY_PATH"

# --- Service Selection & Resource Management ---

# Defaults
START_HEARTMULA="false"
START_DIFFRHYTHM="false"
START_QWEN3TTS="false"
START_DIAGDISTILL="false"
START_WAN2GP="false"
START_COMFYUI="true"
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
  elif [ -f "/app/qwen3_tts_api.py" ]; then
      log "Auto-mode: Detected Qwen3-TTS image. Selecting Qwen3-TTS service."
      START_QWEN3TTS="true"
      SERVICE_TYPE="qwen3-tts"
  elif [ -f "/app/diagdistill_api.py" ]; then
      log "Auto-mode: Detected DiagDistill image. Selecting DiagDistill service."
      START_DIAGDISTILL="true"
      if [ "$TOTAL_VRAM_MB" -gt 22000 ]; then
          SERVICE_TYPE="diagdistill-24gb"
          log "Auto-selected 24GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      else
          SERVICE_TYPE="diagdistill-16gb"
          log "Auto-selected 16GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      fi
  elif [ -d "/opt/wan2gp" ]; then
      log "Auto-mode: Detected Wan2GP installation. Selecting Wan2GP backend."
      START_WAN2GP="true"
      SERVICE_TYPE="ltx23-video-24gb"
      log "Auto-selected Wan2GP backend (LTX-2.3 Audio-Video 24GB)"
  else
      log "Auto-mode: No specialized API found. Defaulting to ComfyUI only."
  fi
else
  # MANUAL MODE: Check for keywords
  if [[ "$SERVICE_TYPE" == *"heartmula"* ]]; then START_HEARTMULA="true"; fi
  if [[ "$SERVICE_TYPE" == *"diffrhythm"* ]]; then START_DIFFRHYTHM="true"; fi
  if [[ "$SERVICE_TYPE" == *"qwen3-tts"* ]]; then START_QWEN3TTS="true"; fi
  if [[ "$SERVICE_TYPE" == *"diagdistill"* ]]; then START_DIAGDISTILL="true"; fi
  # Wan2GP backend for all LTX-2.3 Audio-Video services
  if [[ "$SERVICE_TYPE" == *"ltx23"* ]]; then
      START_WAN2GP="true"
      log "LTX-2.3 service requested. Using Wan2GP backend."
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

# Wan2GP backend (LTX-2.3 Audio-Video 24GB)
if [ "$START_WAN2GP" = "true" ]; then
    # Wan2GP replaces ComfyUI for this service type
    log "Wan2GP backend selected. Disabling ComfyUI to reserve VRAM for Wan2GP."
    START_COMFYUI="false"

    # LTX-2.3 uses the FP8 Gemma 3 12B text encoder which requires:
    #   1. CC >= 8.9 for FP8 support (RTX 40xx / Ada Lovelace, Hopper)
    #   2. The GPU's SM version must be in this PyTorch build's arch list.
    #      Blackwell GPUs (RTX 5060 Ti = SM 12.0) crash with "no kernel image"
    #      if PyTorch was compiled only up to SM 9.0.
    # Use Python (which has the actual PyTorch build info) rather than a raw CC check.
    WAN2GP_GPU_CHECK=$("$PYTHON_EXE" -c "
import sys
try:
    import torch
    if not torch.cuda.is_available():
        print('NO_CUDA')
        sys.exit(0)
    major, minor = torch.cuda.get_device_capability()
    cc_str = 'sm_{}{}'.format(major, minor)
    gpu_name = torch.cuda.get_device_name(0)
    supported = torch.cuda.get_arch_list()
    if cc_str not in supported:
        print('UNSUPPORTED:{}:{}'.format(major, minor, gpu_name))
    elif major < 8 or (major == 8 and minor < 9):
        print('NO_FP8:{}.{}'.format(major, minor))
    else:
        print('OK:{}.{}:{}'.format(major, minor, gpu_name))
except Exception as e:
    print('ERROR:{}'.format(e))
" 2>/dev/null || echo "CHECK_FAILED")

    case "$WAN2GP_GPU_CHECK" in
        OK:*)
            _cc="${WAN2GP_GPU_CHECK#OK:}"
            log "GPU ${_cc} — compute capability and PyTorch arch OK for LTX-2.3."
            ;;
        NO_FP8:*)
            _cc="${WAN2GP_GPU_CHECK#NO_FP8:}"
            log "ERROR: LTX-2.3 requires compute capability 8.9+ for FP8 (detected: ${_cc})."
            log "Supported GPUs: RTX 4090/4080/4070, RTX 4000/5000 Ada, L40S, H100, H200."
            exit 1
            ;;
        UNSUPPORTED:*)
            _cc=$(echo "$WAN2GP_GPU_CHECK" | cut -d: -f2-3)
            log "ERROR: GPU CC ${_cc} is not supported by the installed PyTorch build."
            log "Rebuild the Docker image with a PyTorch version that supports this GPU,"
            log "or use a supported GPU: RTX 4090/4080/4070, L40S, H100."
            exit 1
            ;;
        NO_CUDA)
            log "WARNING: No CUDA device detected — skipping GPU CC check."
            ;;
        *)
            log "WARNING: GPU compatibility check failed (${WAN2GP_GPU_CHECK}) — proceeding anyway."
            ;;
    esac

    # Set Wan2GP environment variables
    export WAN2GP_ROOT="/opt/wan2gp"
    export WAN2GP_OUTPUT="/opt/wan2gp/outputs"
    
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
    
    if [ "$WAN2GP_CHECK_FAILED" = "1" ]; then
        log "ERROR: Wan2GP installation is incomplete."
        log "Please rebuild the image with proper HF_TOKEN for model downloads."
        exit 1
    fi
    
    log "Wan2GP environment configured (WAN2GP_ROOT=$WAN2GP_ROOT)"
fi

# Start ComfyUI
if [ -d "/opt/ComfyUI" ] && [ "$START_COMFYUI" = "true" ]; then
  # Determine ComfyUI launch flags based on SERVICE_TYPE
  COMFY_FLAGS="--listen 0.0.0.0 --port 8188"
  
  case "$SERVICE_TYPE" in
    *ltx2*-8gb*|*8gb*)
      log "Applying AGGRESSIVE 8GB VRAM optimizations for ComfyUI"
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --fp32-vae --disable-smart-memory --reserve-vram 1.0 --cache-none --force-fp16 --use-split-cross-attention --preview-method none"
      ;;
    *ltx2*-16gb*|*16gb*)
      log "Applying 16GB VRAM optimizations for ComfyUI (lowvram mode for model offloading)"
      # IMPORTANT: Use --lowvram because total model size (~29.5GB) far exceeds 16GB VRAM
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --reserve-vram 1.0 --use-split-cross-attention --cache-none"
      ;;
    *ltx2*-24gb*|*24gb*)
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
  if [[ "$SERVICE_TYPE" == *"24gb"* ]] || [[ "$SERVICE_TYPE" == *"ltx2"* ]]; then
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
fi

# Save restart configuration
log "Saving restart configuration..."
cat > /opt/dgn-client/.restart-config << RESTART_CONFIG_EOF
export DGN_CLIENT_ARGS="--dgn-api-key \"$DGN_API_KEY\" --service \"${SERVICE_TYPE:-auto}\" --accept-policy all --root-dir /opt/dgn-client --data-dir /data"
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
sys.path.insert(0, '/opt/dgn-client')
try:
    from config import HEADLESS_MODE
    from dgn_client import DGNClient
    print('DGNClient import successful')
except Exception as e:
    print(f'Import error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" || (log "ERROR: Python imports failed." && exit 1)

"$PYTHON_EXE" cli.py \
  --dgn-api-key "$DGN_API_KEY" \
  --service "${SERVICE_TYPE:-auto}" \
  --accept-policy all \
  --root-dir /opt/dgn-client \
  --data-dir /data 2>&1 | tee -a /tmp/dgn_client.log
