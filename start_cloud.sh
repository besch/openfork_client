#!/bin/bash
# OpenFork DGN Client Cloud Startup Script (start_cloud.sh)
# This script is fetched and run by Vast.ai cloud containers.

OPENFORK_CLIENT_SCRIPT_REF="${OPENFORK_CLIENT_SCRIPT_REF:-main}"
OPENFORK_RAW_BASE="https://raw.githubusercontent.com/besch/openfork_client/${OPENFORK_CLIENT_SCRIPT_REF}"

download_openfork_script() {
  local script_name="$1"
  local dest="$2"
  curl --fail --location --proto '=https' --tlsv1.2 --max-time 60 --progress-bar \
    "${OPENFORK_RAW_BASE}/${script_name}" -o "$dest"
}

refresh_openfork_file() {
  local source_path="$1"
  local dest="$2"
  local temp_path="${dest}.openfork-refresh"

  if download_openfork_script "$source_path" "$temp_path"; then
    chmod --reference="$dest" "$temp_path" 2>/dev/null || true
    mv "$temp_path" "$dest"
    log "Refreshed $dest from ${OPENFORK_CLIENT_SCRIPT_REF}:${source_path}"
    return 0
  fi

  rm -f "$temp_path"
  log "WARNING: Could not refresh $dest from ${OPENFORK_CLIENT_SCRIPT_REF}:${source_path}; using image copy."
  return 1
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
    OPENFORK_ALLOW_DISABLED_SERVICE_TEST
                          Set to "1" only for admin smoke tests of disabled services
    HF_TOKEN / HUGGINGFACE_TOKEN
                          Hugging Face token for gated model downloads

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
RESOURCE_LOG_FILE="/tmp/dgn_resource_usage.log"
RESOURCE_MONITOR_INTERVAL_SECONDS="${RESOURCE_MONITOR_INTERVAL_SECONDS:-30}"
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

normalize_huggingface_token_env() {
  local token="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}}}"
  if [ -z "$token" ]; then
    log "Hugging Face token: not configured"
    return
  fi

  export HF_TOKEN="$token"
  export HUGGINGFACE_TOKEN="$token"
  export HUGGING_FACE_HUB_TOKEN="$token"
  export HUGGINGFACE_HUB_TOKEN="$token"
  log "Hugging Face token: configured"
}

normalize_huggingface_token_env

log_resource_snapshot() {
  local label="${1:-snapshot}"
  {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] resource_snapshot label=$label service=${SERVICE_TYPE:-unknown}"
    if command -v nvidia-smi >/dev/null 2>&1; then
      echo "gpu_csv: timestamp,name,memory.used_mib,memory.total_mib,utilization.gpu_pct,utilization.memory_pct,power.draw_w"
      nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits 2>&1 || true
      echo "gpu_processes:"
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1 || true
    else
      echo "gpu_csv: nvidia-smi unavailable"
    fi

    if command -v free >/dev/null 2>&1; then
      echo "memory_mib:"
      free -m 2>&1 || true
    fi

    echo "cpu_mem_processes:"
    ps -eo pid,ppid,pcpu,pmem,rss,vsz,etime,args --sort=-pcpu 2>/dev/null | \
      grep -E 'PID|python.*(main.py|cli.py|wan2gp_server.py)|ComfyUI|wan2gp|server.py' | \
      grep -v grep | \
      head -n 20 || true
    echo
  } >> "$RESOURCE_LOG_FILE"
}

start_resource_monitor() {
  if [ "${ENABLE_RESOURCE_MONITOR:-true}" != "true" ]; then
    log "Resource monitor disabled by ENABLE_RESOURCE_MONITOR=$ENABLE_RESOURCE_MONITOR"
    return
  fi

  if [ -n "${RESOURCE_MONITOR_PID:-}" ] && kill -0 "$RESOURCE_MONITOR_PID" 2>/dev/null; then
    return
  fi

  log "Resource monitor writing snapshots to $RESOURCE_LOG_FILE every ${RESOURCE_MONITOR_INTERVAL_SECONDS}s"
  log_resource_snapshot "startup"
  (
    while true; do
      sleep "$RESOURCE_MONITOR_INTERVAL_SECONDS"
      log_resource_snapshot "interval"
    done
  ) &
  RESOURCE_MONITOR_PID=$!
}

stop_resource_monitor() {
  if [ -n "${RESOURCE_MONITOR_PID:-}" ]; then
    kill "$RESOURCE_MONITOR_PID" 2>/dev/null || true
    wait "$RESOURCE_MONITOR_PID" 2>/dev/null || true
  fi
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
  stop_resource_monitor
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

wait_for_model_loaded() {
  local name="$1"
  local url="$2"
  local max_wait="${3:-900}"
  local log_file="$4"
  local waited=0
  local health=""

  log "Waiting for $name model to finish loading at $url..."
  while [ $waited -lt $max_wait ]; do
    health=$(curl -fsS "$url" 2>/dev/null || true)

    if echo "$health" | grep -Eq '"model_loaded"[[:space:]]*:[[:space:]]*true'; then
      log "$name model is loaded."
      return 0
    fi

    if echo "$health" | grep -Eq '"status"[[:space:]]*:[[:space:]]*"error"'; then
      log "ERROR: $name model reported an error: $health"
      if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        log "Last 30 lines of $log_file:"
        tail -n 30 "$log_file" | sed 's/^/    /' || true
      fi
      return 1
    fi

    if [ $((waited % 30)) -eq 0 ]; then
      if [ -n "$health" ]; then
        log "  $name model still loading ($waited/$max_wait seconds): $health"
      else
        log "  $name health endpoint not reachable yet ($waited/$max_wait seconds)"
      fi
      if [ "$waited" -gt 0 ] && [ -n "$log_file" ] && [ -f "$log_file" ]; then
        log "  Last 15 lines of $log_file:"
        tail -n 15 "$log_file" | sed 's/^/    /' || true
      fi
    fi

    sleep 5
    waited=$((waited + 5))
  done

  log "ERROR: $name model did not finish loading within $max_wait seconds."
  [ -n "$log_file" ] && log "Check $log_file for errors."
  return 1
}

ffmpeg_supports_fps_mode() {
  command -v ffmpeg >/dev/null 2>&1 &&
    ffmpeg -hide_banner -h full 2>/dev/null | grep -q -- "-fps_mode"
}

install_static_ffmpeg_71() {
  local tmp_dir="/tmp/openfork-ffmpeg-71"
  rm -rf "$tmp_dir"
  mkdir -p "$tmp_dir"

  log "Installing static FFmpeg 7.1 for Wan2GP video decode compatibility..."
  if ! curl -fSL --retry 3 --retry-delay 5 \
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-gpl-7.1.tar.xz" \
    -o "$tmp_dir/ffmpeg.tar.xz"; then
    log "ERROR: Failed to download static FFmpeg 7.1."
    return 1
  fi

  if ! tar -xJf "$tmp_dir/ffmpeg.tar.xz" -C "$tmp_dir"; then
    log "ERROR: Failed to extract static FFmpeg 7.1."
    return 1
  fi

  local bin_dir
  bin_dir="$(find "$tmp_dir" -type d -path "*/bin" | head -n 1)"
  if [ -z "$bin_dir" ] || [ ! -x "$bin_dir/ffmpeg" ]; then
    log "ERROR: Static FFmpeg archive did not contain an executable ffmpeg."
    return 1
  fi

  cp -f "$bin_dir/ffmpeg" /usr/local/bin/ffmpeg
  cp -f "$bin_dir/ffprobe" /usr/local/bin/ffprobe
  chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe
  hash -r
  log "Static FFmpeg installed: $(ffmpeg -version 2>&1 | head -1)"
}

ensure_ffmpeg_fps_mode_support() {
  if ffmpeg_supports_fps_mode; then
    log "FFmpeg supports -fps_mode: $(ffmpeg -version 2>&1 | head -1)"
    return 0
  fi

  log "WARNING: Current FFmpeg does not support -fps_mode, which Wan2GP uses for video-guide decode."
  install_static_ffmpeg_71 && ffmpeg_supports_fps_mode
}

verify_wan2gp_stable_thread_wrapper() {
  local server_file="$1"
  [ -f "$server_file" ] || return 1

  grep -q "_session = _init_session_sync()" "$server_file" &&
    grep -q "process main thread" "$server_file" &&
    grep -q "WAN2GP_EXIT_AFTER_JOB" "$server_file" &&
    grep -q "_schedule_process_exit" "$server_file" &&
    grep -q "_generate_sync" "$server_file" &&
    ! grep -q "run_in_executor" "$server_file"
}

start_wan2gp_server_supervisor() {
  local server_file="$1"
  local root_dir="$2"
  local log_file="$3"
  local restart_delay="${WAN2GP_RESTART_DELAY_SECONDS:-2}"
  local rc

  while true; do
    log "Wan2GP supervisor launching HTTP server..."
    rc=0
    (cd "$root_dir" && "$PYTHON_EXE" "$server_file" >> "$log_file" 2>&1) || rc=$?
    log "Wan2GP HTTP server exited with status ${rc}; restarting in ${restart_delay}s..."
    sleep "$restart_delay"
  done
}

ensure_swap_space() {
  local target_gb="${1:-0}"
  [ "$target_gb" -gt 0 ] 2>/dev/null || return 0
  command -v swapon >/dev/null 2>&1 || {
    log "WARNING: swapon not available; cannot create swap for Wan2GP."
    return 0
  }

  local current_gb
  current_gb=$(free -g | awk '/Swap/ {print $2}')
  current_gb="${current_gb:-0}"
  if [ "$current_gb" -ge "$target_gb" ] 2>/dev/null; then
    log "Swap already sufficient for Wan2GP: ${current_gb}GB >= ${target_gb}GB"
    return 0
  fi

  local swap_file="${WAN2GP_SWAP_FILE:-/swapfile}"
  if [ -e "$swap_file" ]; then
    log "WARNING: $swap_file already exists but active swap is ${current_gb}GB; leaving it untouched."
    return 0
  fi

  log "Creating best-effort ${target_gb}GB swap file for Wan2GP offload stability..."
  local created=0
  if command -v fallocate >/dev/null 2>&1; then
    fallocate -l "${target_gb}G" "$swap_file" && created=1
  fi
  if [ "$created" != "1" ]; then
    dd if=/dev/zero of="$swap_file" bs=1G count="$target_gb" status=none && created=1
  fi
  if [ "$created" != "1" ]; then
    log "WARNING: Could not allocate swap file; continuing without extra swap."
    rm -f "$swap_file" 2>/dev/null || true
    return 0
  fi

  chmod 600 "$swap_file" || true
  if mkswap "$swap_file" >/dev/null 2>&1 && swapon "$swap_file" >/dev/null 2>&1; then
    log "Enabled Wan2GP swap file: $(swapon --show --bytes | tail -n +2 | awk '{sum += $3} END {printf \"%.1fGB\", sum/1024/1024/1024}')"
  else
    log "WARNING: Could not enable swap file inside this container; continuing without extra swap."
    rm -f "$swap_file" 2>/dev/null || true
  fi
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
if download_openfork_script bootstrap.sh bootstrap.sh; then
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
          [ -f "$DGN_SOURCE_DIR/mmaudio_api.py" ] && cp -v "$DGN_SOURCE_DIR/mmaudio_api.py" /app/
          [ -f "$DGN_SOURCE_DIR/qwen3_tts_api.py" ] && cp -v "$DGN_SOURCE_DIR/qwen3_tts_api.py" /app/
          [ -f "$DGN_SOURCE_DIR/f5_tts_api.py" ] && cp -v "$DGN_SOURCE_DIR/f5_tts_api.py" /app/
          [ -f "$DGN_SOURCE_DIR/dramabox_api.py" ] && cp -v "$DGN_SOURCE_DIR/dramabox_api.py" /app/
          [ -f "$DGN_SOURCE_DIR/diagdistill_api.py" ] && cp -v "$DGN_SOURCE_DIR/diagdistill_api.py" /app/
          [ -f "$DGN_SOURCE_DIR/stream_diffvsr_wrapper.py" ] && cp -v "$DGN_SOURCE_DIR/stream_diffvsr_wrapper.py" /app/
          [ -f "$DGN_SOURCE_DIR/sparkvsr_api.py" ] && cp -v "$DGN_SOURCE_DIR/sparkvsr_api.py" /app/
          [ -f "$DGN_SOURCE_DIR/ernie_image_api.py" ] && cp -v "$DGN_SOURCE_DIR/ernie_image_api.py" /app/
          [ -d "/app/PiD" ] && [ -f "$DGN_SOURCE_DIR/pid_image_api.py" ] && cp -v "$DGN_SOURCE_DIR/pid_image_api.py" /app/PiD/
      fi
      
      # 2. TurboDiffusion (runs in /opt/TurboDiffusion)
      if [ -d "/opt/TurboDiffusion" ] && [ -f "$DGN_SOURCE_DIR/turbodiffusion_api_server.py" ]; then
          cp -v "$DGN_SOURCE_DIR/turbodiffusion_api_server.py" /opt/TurboDiffusion/api_server.py
      fi

      # 3. Wan2GP HTTP server (runs in /opt/wan2gp)
      if [ -d "/opt/wan2gp" ] && [ -f "$DGN_SOURCE_DIR/wan2gp_server.py" ]; then
          cp -v "$DGN_SOURCE_DIR/wan2gp_server.py" /opt/wan2gp/
          if verify_wan2gp_stable_thread_wrapper "/opt/wan2gp/wan2gp_server.py"; then
              log "Wan2GP stable main-thread recycle wrapper is installed."
          else
              log "WARNING: Synced wan2gp_server.py does not contain the stable main-thread recycle wrapper."
          fi
      fi

      log "✓ Dynamic file sync complete"
  else
      log "WARNING: Could not find comfyui-storage in DGN client. Skipping file sync."
  fi

repair_torch_audio_stack() {
  if "$PYTHON_EXE" - <<'PY' >/tmp/torch_stack_check.log 2>&1
import torch
import torchvision
import torchaudio
print(f"torch={torch.__version__} torchvision={torchvision.__version__} torchaudio={torchaudio.__version__}")
PY
  then
    log "Torch stack OK: $(cat /tmp/torch_stack_check.log)"
    return 0
  fi

  log "WARNING: Torch stack import failed; attempting version-matched repair."
  sed 's/^/  /' /tmp/torch_stack_check.log || true

  local torch_version
  torch_version=$("$PYTHON_EXE" - <<'PY' 2>/dev/null || true
import torch
print(torch.__version__)
PY
)
  torch_version="${torch_version%%$'\n'*}"

  local index_url=""
  local torch_pkg=""
  local vision_pkg=""
  local audio_pkg=""

  case "$torch_version" in
    2.8.0+cu128*)
      index_url="https://download.pytorch.org/whl/cu128"
      torch_pkg="torch==2.8.0+cu128"
      vision_pkg="torchvision==0.23.0+cu128"
      audio_pkg="torchaudio==2.8.0+cu128"
      ;;
    2.7.1+cu128*)
      index_url="https://download.pytorch.org/whl/cu128"
      torch_pkg="torch==2.7.1+cu128"
      vision_pkg="torchvision==0.22.1+cu128"
      audio_pkg="torchaudio==2.7.1+cu128"
      ;;
    2.7.0+cu128*)
      index_url="https://download.pytorch.org/whl/cu128"
      torch_pkg="torch==2.7.0+cu128"
      vision_pkg="torchvision==0.22.0+cu128"
      audio_pkg="torchaudio==2.7.0+cu128"
      ;;
    2.4.0*)
      index_url="https://download.pytorch.org/whl/cu124"
      torch_pkg="torch==2.4.0"
      vision_pkg="torchvision==0.19.0"
      audio_pkg="torchaudio==2.4.0"
      ;;
  esac

  if [ -z "$torch_pkg" ]; then
    log "WARNING: No known torch repair mapping for version '${torch_version}'. Continuing without repair."
    return 0
  fi

  log "Repairing torch stack to ${torch_pkg}, ${vision_pkg}, ${audio_pkg}"
  "$PYTHON_EXE" -m pip install --quiet --no-cache-dir --force-reinstall \
    --index-url "$index_url" \
    --extra-index-url "https://pypi.org/simple" \
    "$torch_pkg" "$vision_pkg" "$audio_pkg" || {
      log "WARNING: Torch stack repair failed."
      return 0
    }

  if "$PYTHON_EXE" - <<'PY' >/tmp/torch_stack_check.log 2>&1
import torch
import torchvision
import torchaudio
print(f"torch={torch.__version__} torchvision={torchvision.__version__} torchaudio={torchaudio.__version__}")
PY
  then
    log "Torch stack repaired: $(cat /tmp/torch_stack_check.log)"
  else
    log "WARNING: Torch stack still fails after repair."
    sed 's/^/  /' /tmp/torch_stack_check.log || true
  fi
}

repair_torch_audio_stack

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

PYTORCH_NVIDIA_LIBRARY_PATHS=$("$PYTHON_EXE" -c "import site; p=site.getsitepackages()[0]; print(':'.join([p + '/torch/lib', p + '/nvidia/nvjitlink/lib', p + '/nvidia/cusparse/lib', p + '/nvidia/cublas/lib', p + '/nvidia/cuda_runtime/lib']))" 2>/dev/null || true)
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${PYTORCH_NVIDIA_LIBRARY_PATHS}:${LD_LIBRARY_PATH:-}"
log "Updated LD_LIBRARY_PATH for PyTorch compatibility: $LD_LIBRARY_PATH"

# TorchScript can crash native extensions during ComfyUI startup on some tight
# CUDA/Python combinations (observed in kornia import on 8GB FLUX Kontext).
export PYTORCH_JIT="${PYTORCH_JIT:-0}"

# --- Service Selection & Resource Management ---

# Defaults
START_HEARTMULA="false"
START_DIFFRHYTHM="false"
START_AUDIOX="false"
START_QWEN3TTS="false"
START_F5TTS="false"
START_WAVTTS="false"
START_SCENEMA_AUDIO="false"
START_DRAMABOX="false"
START_DIAGDISTILL="false"
START_WAN2GP="false"
START_COMFYUI="true"
START_SPARKVSR="false"
START_INSPATIO="false"
START_ERNIE_IMAGE="false"
START_IDEOGRAM4="false"
START_PID_IMAGE="false"
START_PRISMAUDIO="false"
START_MMAUDIO="false"
START_ACESTEP="false"
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
  elif [ -d "/app/acestep_repo" ] || command -v acestep-api >/dev/null 2>&1; then
      log "Auto-mode: Detected ACE-Step image. Selecting ACE-Step service."
      START_ACESTEP="true"
      if [ "$TOTAL_VRAM_MB" -gt 14000 ]; then
          SERVICE_TYPE="acestep-16gb"
          log "Auto-selected ACE-Step 16GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      else
          SERVICE_TYPE="acestep-8gb"
          log "Auto-selected ACE-Step 8GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      fi
  elif [ -f "/app/qwen3_tts_api.py" ]; then
      log "Auto-mode: Detected Qwen3-TTS image. Selecting Qwen3-TTS service."
      START_QWEN3TTS="true"
      SERVICE_TYPE="qwen3-tts"
  elif [ -f "/app/f5_tts_api.py" ]; then
      log "Auto-mode: Detected F5-TTS image. Selecting F5-TTS service."
      START_F5TTS="true"
      SERVICE_TYPE="f5-tts"
  elif [ -f "/app/wavtts_api.py" ]; then
      log "Auto-mode: Detected WavTTS image. Selecting WavTTS service."
      START_WAVTTS="true"
      SERVICE_TYPE="wavtts"
  elif [ -f "/app/prismaudio_api.py" ]; then
      log "Auto-mode: Detected PRiSM Audio image. Selecting PRiSM Audio service."
      START_PRISMAUDIO="true"
      if [ "$TOTAL_VRAM_MB" -gt 14000 ]; then
          SERVICE_TYPE="prismaudio-16gb"
          log "Auto-selected PRiSM Audio 16GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      else
          SERVICE_TYPE="prismaudio-8gb"
          log "Auto-selected PRiSM Audio 8GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      fi
  elif [ -f "/app/mmaudio_api.py" ]; then
      log "Auto-mode: Detected MMAudio image. Selecting MMAudio service."
      START_MMAUDIO="true"
      if [ "$TOTAL_VRAM_MB" -gt 14000 ]; then
          SERVICE_TYPE="mmaudio-16gb"
          log "Auto-selected MMAudio 16GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      else
          SERVICE_TYPE="mmaudio-8gb"
          log "Auto-selected MMAudio 8GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      fi
  elif [ -f "/app/dramabox_api.py" ]; then
      log "Auto-mode: Detected DramaBox image. Selecting DramaBox service."
      START_DRAMABOX="true"
      SERVICE_TYPE="dramabox"
  elif [ -f "/app/src/server.py" ] && [ -f "/app/src/audio_core/processor.py" ]; then
      log "Auto-mode: Detected Scenema Audio image. Selecting Scenema Audio service."
      START_SCENEMA_AUDIO="true"
      SERVICE_TYPE="scenema-audio"
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
      WAN22_T2V_HIGH_MBF16="/opt/wan2gp/ckpts/wan2.2_text2video_14B_high_mbf16.safetensors"
      WAN22_T2V_LOW_MBF16="/opt/wan2gp/ckpts/wan2.2_text2video_14B_low_mbf16.safetensors"
      WAN22_I2V_HIGH_MBF16="/opt/wan2gp/ckpts/wan2.2_image2video_14B_high_mbf16.safetensors"
      WAN22_I2V_LOW_MBF16="/opt/wan2gp/ckpts/wan2.2_image2video_14B_low_mbf16.safetensors"
      WAN22_T2V_HIGH_MBF16_INT8="/opt/wan2gp/ckpts/wan2.2_text2video_14B_high_quanto_mbf16_int8.safetensors"
      WAN22_T2V_LOW_MBF16_INT8="/opt/wan2gp/ckpts/wan2.2_text2video_14B_low_quanto_mbf16_int8.safetensors"
      WAN22_I2V_HIGH_MBF16_INT8="/opt/wan2gp/ckpts/wan2.2_image2video_14B_high_quanto_mbf16_int8.safetensors"
      WAN22_I2V_LOW_MBF16_INT8="/opt/wan2gp/ckpts/wan2.2_image2video_14B_low_quanto_mbf16_int8.safetensors"
      WAN22_T2V_HIGH_MFP16_INT8="/opt/wan2gp/ckpts/wan2.2_text2video_14B_high_quanto_mfp16_int8.safetensors"
      WAN22_T2V_LOW_MFP16_INT8="/opt/wan2gp/ckpts/wan2.2_text2video_14B_low_quanto_mfp16_int8.safetensors"
      WAN22_I2V_HIGH_MFP16_INT8="/opt/wan2gp/ckpts/wan2.2_image2video_14B_high_quanto_mfp16_int8.safetensors"
      WAN22_I2V_LOW_MFP16_INT8="/opt/wan2gp/ckpts/wan2.2_image2video_14B_low_quanto_mfp16_int8.safetensors"
      if { [ -f "$WAN22_T2V_HIGH_MFP16_INT8" ] && [ -f "$WAN22_T2V_LOW_MFP16_INT8" ]; } || { [ -f "$WAN22_I2V_HIGH_MFP16_INT8" ] && [ -f "$WAN22_I2V_LOW_MFP16_INT8" ]; }; then
          if [ "$TOTAL_VRAM_MB" -lt 9000 ]; then
              SERVICE_TYPE="wan22-wan2gp-8gb"
              log "Auto-selected WAN 2.2 Wan2GP 8GB tier (quanto mfp16 int8 image detected, VRAM: ${TOTAL_VRAM_MB}MB)"
          elif [ "$TOTAL_VRAM_MB" -lt 12000 ]; then
              SERVICE_TYPE="wan22-wan2gp-10gb"
              log "Auto-selected WAN 2.2 Wan2GP 10GB tier (quanto mfp16 int8 image detected, VRAM: ${TOTAL_VRAM_MB}MB)"
          else
              SERVICE_TYPE="wan22-wan2gp-12gb"
              log "Auto-selected WAN 2.2 Wan2GP 12GB tier (quanto mfp16 int8 image detected, VRAM: ${TOTAL_VRAM_MB}MB)"
          fi
      elif { [ -f "$WAN22_T2V_HIGH_MBF16_INT8" ] && [ -f "$WAN22_T2V_LOW_MBF16_INT8" ]; } || { [ -f "$WAN22_I2V_HIGH_MBF16_INT8" ] && [ -f "$WAN22_I2V_LOW_MBF16_INT8" ]; }; then
          SERVICE_TYPE="wan22-wan2gp-16gb"
          log "Auto-selected WAN 2.2 Wan2GP 16GB tier (quanto mbf16 int8 image detected, VRAM: ${TOTAL_VRAM_MB}MB)"
      elif { [ -f "$WAN22_T2V_HIGH_MBF16" ] && [ -f "$WAN22_T2V_LOW_MBF16" ]; } || { [ -f "$WAN22_I2V_HIGH_MBF16" ] && [ -f "$WAN22_I2V_LOW_MBF16" ]; }; then
          SERVICE_TYPE="wan22-wan2gp-24gb"
          log "Auto-selected WAN 2.2 Wan2GP 24GB tier (mbf16 image detected, VRAM: ${TOTAL_VRAM_MB}MB)"
      elif [ -f "$VISTA4D_TRANSFORMER_BF16" ] || [ -f "$VISTA4D_TRANSFORMER_INT8" ]; then
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
  elif ls /opt/ComfyUI/models/unet/flux1-kontext-dev-*.gguf >/dev/null 2>&1; then
      log "Auto-mode: Detected FLUX.1 Kontext [dev] GGUF ComfyUI image."
      if [ -f "/opt/ComfyUI/models/unet/flux1-kontext-dev-Q8_0.gguf" ]; then
          SERVICE_TYPE="flux-kontext-dev-24gb"
          if [ "$TOTAL_VRAM_MB" -lt 22000 ]; then
              log "WARNING: FLUX Kontext Q8_0 is configured as a 24GB tier; detected only ${TOTAL_VRAM_MB}MB VRAM."
          else
              log "Auto-selected FLUX Kontext 24GB tier (Q8_0, VRAM: ${TOTAL_VRAM_MB}MB)"
          fi
      elif [ -f "/opt/ComfyUI/models/unet/flux1-kontext-dev-Q6_K.gguf" ]; then
          SERVICE_TYPE="flux-kontext-dev-16gb"
          log "Auto-selected FLUX Kontext 16GB tier (Q6_K, VRAM: ${TOTAL_VRAM_MB}MB)"
      elif [ -f "/opt/ComfyUI/models/unet/flux1-kontext-dev-Q5_K_M.gguf" ]; then
          SERVICE_TYPE="flux-kontext-dev-12gb"
          log "Auto-selected FLUX Kontext 12GB tier (Q5_K_M, VRAM: ${TOTAL_VRAM_MB}MB)"
      else
          SERVICE_TYPE="flux-kontext-dev-8gb"
          log "Auto-selected FLUX Kontext 8GB tier (Q4_K_M, VRAM: ${TOTAL_VRAM_MB}MB)"
      fi
  elif [ -f "/opt/ComfyUI/models/DreamID-Omni/DreamID_Omni/dreamid_omni.fp8_e4m3fn.safetensors" ]; then
      log "Auto-mode: Detected DreamID-Omni FP8 ComfyUI image."
      SERVICE_TYPE="dreamid-omni-24gb"
      if [ "$TOTAL_VRAM_MB" -lt 22000 ]; then
          log "WARNING: DreamID-Omni is configured as a 24GB tier; detected only ${TOTAL_VRAM_MB}MB VRAM."
      else
          log "Auto-selected DreamID-Omni 24GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      fi
   elif [ -f "/app/PiD/pid_image_api.py" ] && [ -d "/app/PiD/pid" ]; then
        log "Auto-mode: Detected PiD image upscaler. Selecting PiD Z-Image service."
        START_PID_IMAGE="true"
        SERVICE_TYPE="pid-zimage-upscaler-16gb"
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
  elif [ -f "/app/ideogram4_api.py" ]; then
      log "Auto-mode: Detected Ideogram 4 image. Selecting Ideogram 4 service."
      START_IDEOGRAM4="true"
      if [ "${IDEOGRAM_QUANTIZATION:-}" = "fp8" ] || [ "$TOTAL_VRAM_MB" -gt 30000 ]; then
          SERVICE_TYPE="ideogram4-32gb"
          log "Auto-selected Ideogram 4 32GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      elif [ "$TOTAL_VRAM_MB" -gt 22000 ]; then
          SERVICE_TYPE="ideogram4-24gb"
          log "Auto-selected Ideogram 4 24GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
      else
          SERVICE_TYPE="ideogram4-16gb"
          log "Auto-selected Ideogram 4 16GB tier (VRAM: ${TOTAL_VRAM_MB}MB)"
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
  if [[ "$SERVICE_TYPE" == *"f5-tts"* ]]; then START_F5TTS="true"; fi
  if [[ "$SERVICE_TYPE" == *"wavtts"* ]]; then START_WAVTTS="true"; fi
  if [[ "$SERVICE_TYPE" == *"prismaudio"* ]]; then START_PRISMAUDIO="true"; fi
  if [[ "$SERVICE_TYPE" == *"mmaudio"* ]]; then START_MMAUDIO="true"; fi
  if [[ "$SERVICE_TYPE" == *"acestep"* ]]; then START_ACESTEP="true"; fi
  if [[ "$SERVICE_TYPE" == *"scenema-audio"* ]]; then START_SCENEMA_AUDIO="true"; fi
  if [[ "$SERVICE_TYPE" == *"dramabox"* ]]; then START_DRAMABOX="true"; fi
  if [[ "$SERVICE_TYPE" == *"diagdistill"* ]]; then START_DIAGDISTILL="true"; fi
  if [[ "$SERVICE_TYPE" == *"sparkvsr"* ]]; then START_SPARKVSR="true"; fi
  if [[ "$SERVICE_TYPE" == *"inspatio"* ]]; then START_INSPATIO="true"; fi
  if [[ "$SERVICE_TYPE" == *"ernie-image"* ]]; then START_ERNIE_IMAGE="true"; fi
  if [[ "$SERVICE_TYPE" == *"ideogram4"* ]]; then START_IDEOGRAM4="true"; fi
  if [[ "$SERVICE_TYPE" == *"pid-zimage"* ]]; then START_PID_IMAGE="true"; fi
  if [[ "$SERVICE_TYPE" == *"anima"* ]]; then START_COMFYUI="true"; fi
  # Wan2GP backend for LTX-2.3 Audio-Video, WAN 2.2, daVinci-MagiHuman, SCAIL, and Vista4D services.
  if [[ "$SERVICE_TYPE" == *"ltx23"* ]] || [[ "$SERVICE_TYPE" == *"wan22-wan2gp"* ]] || [[ "$SERVICE_TYPE" == *"davinci"* ]] || [[ "$SERVICE_TYPE" == *"scail"* ]] || [[ "$SERVICE_TYPE" == *"vista4d"* ]]; then
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

if [ "$START_SCENEMA_AUDIO" = "true" ]; then
  log "Scenema Audio selected. Disabling ComfyUI to reserve VRAM."
  START_COMFYUI="false"
fi

if [ "$START_DRAMABOX" = "true" ]; then
  log "DramaBox selected. Disabling ComfyUI to reserve VRAM."
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

if [ "$START_PID_IMAGE" = "true" ]; then
  log "PiD image upscaler selected. Disabling ComfyUI to reserve VRAM."
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

  export QWEN3_ALLOW_ONLINE_FALLBACK="${QWEN3_ALLOW_ONLINE_FALLBACK:-1}"
  if [ "${QWEN3_STRICT_OFFLINE:-false}" = "true" ]; then
      export HF_HUB_OFFLINE="1"
      export TRANSFORMERS_OFFLINE="1"
      log "Qwen3-TTS strict offline mode enabled."
  elif [ -n "${HF_TOKEN:-}" ]; then
      export HF_HUB_OFFLINE="0"
      export TRANSFORMERS_OFFLINE="0"
      log "Qwen3-TTS online cache repair enabled when baked model snapshots are missing."
  else
      log "Qwen3-TTS has no Hugging Face token; using baked cache only."
  fi
fi

if [ "$START_F5TTS" = "true" ]; then
  log "F5-TTS selected. Disabling ComfyUI to reserve VRAM."
  START_COMFYUI="false"
fi

if [ "$START_WAVTTS" = "true" ]; then
  log "WavTTS selected. Disabling ComfyUI to reserve VRAM."
  START_COMFYUI="false"
fi

if [ "$START_IDEOGRAM4" = "true" ]; then
  log "Ideogram 4 selected. Disabling ComfyUI to reserve VRAM."
  START_COMFYUI="false"
fi

if [ "$START_PRISMAUDIO" = "true" ]; then
  log "PRiSM Audio selected. Disabling ComfyUI to reserve VRAM."
  START_COMFYUI="false"
fi

if [ "$START_MMAUDIO" = "true" ]; then
  log "MMAudio selected. Disabling ComfyUI to reserve VRAM."
  START_COMFYUI="false"
fi

if [ "$START_ACESTEP" = "true" ]; then
  log "ACE-Step selected. Disabling ComfyUI to reserve VRAM."
  START_COMFYUI="false"
fi

ensure_zimage_full_models() {
  local service="${SERVICE_TYPE:-}"
  local model_name=""
  local source_url=""
  local min_bytes=1000000000

  case "$service" in
    *zimage-full-24gb*)
      model_name="z_image_bf16.safetensors"
      source_url="https://huggingface.co/Comfy-Org/z_image/resolve/main/split_files/diffusion_models/z_image_bf16.safetensors"
      min_bytes=10000000000
      ;;
    *zimage-full-16gb*)
      model_name="z_image_fp8.safetensors"
      source_url="https://huggingface.co/Kijai/Z-Image_comfy_fp8_scaled/resolve/main/z-image-turbo_fp8_scaled_e4m3fn_KJ.safetensors"
      min_bytes=5000000000
      ;;
    *)
      return 0
      ;;
  esac

  if [ "${ZIMAGE_SKIP_MODEL_REPAIR:-false}" = "true" ]; then
    log "Z-Image model repair skipped by ZIMAGE_SKIP_MODEL_REPAIR=true."
    return 0
  fi

  if [ ! -d "/opt/ComfyUI" ]; then
    log "WARNING: Z-Image service selected, but /opt/ComfyUI is missing."
    return 1
  fi

  local diffusion_dir="/opt/ComfyUI/models/diffusion_models"
  local unet_dir="/opt/ComfyUI/models/unet"
  local target="${diffusion_dir}/${model_name}"
  mkdir -p "$diffusion_dir" "$unet_dir"

  for candidate in \
    "$unet_dir/$model_name" \
    "$unet_dir/split_files/diffusion_models/$model_name" \
    "$diffusion_dir/split_files/diffusion_models/$model_name"; do
    if [ -f "$candidate" ] && [ ! -f "$target" ]; then
      log "Moving misplaced Z-Image model from $candidate to $target"
      mv "$candidate" "$target" || cp "$candidate" "$target"
    fi
  done

  if [ -s "$target" ]; then
    local existing_bytes
    existing_bytes=$(wc -c < "$target" 2>/dev/null || echo 0)
    if [ "$existing_bytes" -lt "$min_bytes" ]; then
      log "Removing incomplete Z-Image model $target (${existing_bytes} bytes)."
      rm -f "$target"
    fi
  fi

  if [ ! -s "$target" ]; then
    log "Z-Image model $model_name is missing; downloading from Hugging Face."
    rm -f "${target}.tmp"
    if ! curl --fail --location --retry 5 --retry-delay 20 --continue-at - \
      --output "${target}.tmp" "$source_url"; then
      log "ERROR: Failed to download $model_name."
      return 1
    fi
    mv "${target}.tmp" "$target"
  fi

  local actual_bytes
  actual_bytes=$(wc -c < "$target" 2>/dev/null || echo 0)
  if [ "$actual_bytes" -lt "$min_bytes" ]; then
    log "ERROR: $model_name is too small (${actual_bytes} bytes); expected at least ${min_bytes} bytes."
    return 1
  fi

  ln -sf "../diffusion_models/$model_name" "$unet_dir/$model_name"
  log "Z-Image model ready: $target (${actual_bytes} bytes)."
}

if [[ "${SERVICE_TYPE:-}" == *"zimage-full-"* ]]; then
  if ! ensure_zimage_full_models; then
    log "ERROR: Z-Image Full model repair failed; refusing to start an unusable provider."
    exit 1
  fi
fi

# Wan2GP backend (LTX-2.3 Audio-Video, WAN 2.2, daVinci-MagiHuman, SCAIL, and Vista4D)
if [ "$START_WAN2GP" = "true" ]; then
    # Wan2GP replaces ComfyUI for this service type
    log "Wan2GP backend selected. Disabling ComfyUI to reserve VRAM for Wan2GP."
    START_COMFYUI="false"
    export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-2}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

    if [ "${ENABLE_WAN2GP_SWAP:-true}" = "true" ]; then
        if [[ "${SERVICE_TYPE:-}" == *"ltx23"* ]]; then
            ensure_swap_space "${WAN2GP_SWAP_GB:-64}"
        elif [[ "${SERVICE_TYPE:-}" == *"24gb"* ]]; then
            ensure_swap_space "${WAN2GP_SWAP_GB:-48}"
        else
            ensure_swap_space "${WAN2GP_SWAP_GB:-32}"
        fi
    fi

    if ! ensure_ffmpeg_fps_mode_support; then
        if [[ "${SERVICE_TYPE:-}" == *"scail"* ]] || [[ "${SERVICE_TYPE:-}" == *"vista4d"* ]]; then
            log "ERROR: $SERVICE_TYPE requires FFmpeg with -fps_mode support for driving/source video decode."
            exit 1
        fi
        log "WARNING: Continuing with FFmpeg that may not support Wan2GP video-guide decode."
    fi

    if [[ "${SERVICE_TYPE:-}" == *"wan22-wan2gp"* ]]; then
        if [[ "${SERVICE_TYPE:-}" == *"8gb"* ]]; then
            export WAN2GP_CLI_ARGS="${WAN2GP_CLI_ARGS:---profile 5 --attention sdpa --preload 0 --perc-reserved-mem-max 0.20 --vram-safety-coefficient 0.35}"
        elif [[ "${SERVICE_TYPE:-}" == *"10gb"* ]]; then
            export WAN2GP_CLI_ARGS="${WAN2GP_CLI_ARGS:---profile 4.5 --attention sdpa --perc-reserved-mem-max 0.35 --vram-safety-coefficient 0.60}"
        elif [[ "${SERVICE_TYPE:-}" == *"12gb"* ]]; then
            export WAN2GP_CLI_ARGS="${WAN2GP_CLI_ARGS:---profile 4.5 --attention sdpa --perc-reserved-mem-max 0.35 --vram-safety-coefficient 0.65}"
        elif [[ "${SERVICE_TYPE:-}" == *"24gb"* ]]; then
            export WAN2GP_CLI_ARGS="${WAN2GP_CLI_ARGS:---profile 4 --attention sdpa --perc-reserved-mem-max 0.55 --vram-safety-coefficient 0.80}"
        else
            export WAN2GP_CLI_ARGS="${WAN2GP_CLI_ARGS:---profile 4.5 --attention sdpa --perc-reserved-mem-max 0.45 --vram-safety-coefficient 0.70}"
        fi
    elif [[ "${SERVICE_TYPE:-}" == *"scail"* ]] && [[ "${SERVICE_TYPE:-}" == *"16gb"* ]]; then
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
    # PyTorch can load on CC >= 7.5, but public Wan2GP workers require CC >= 8.0
    # (sm_80 = A100, sm_86 = RTX 30xx/A10G, sm_89 = RTX 40xx, sm_90 = H100).
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
    # Public Wan2GP video workers need Ampere-class throughput or better.
    # PyTorch cu128 can load on Turing (SM 7.5), but these hosts are too slow
    # for production network video and may hang before DGN registration.
    if major < 8:
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
            log "ERROR: Wan2GP public video workers require compute capability 8.0+ (detected: ${_cc})."
            log "Supported production GPUs: A100 (CC 8.0), RTX 30xx/A10G (CC 8.6), RTX 40xx/L40S (CC 8.9), H100 (CC 9.0), Blackwell (CC 12.0)."
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
            LTX23_EXPECTED_IMAGE="beschiak/openfork-ltx23-wan2gp-hdr:latest"
        fi
        # LTX23_REQUIRED_TRANSFORMER="$LTX23_Q8_TRANSFORMER"
        # LTX23_EXPECTED_IMAGE="beschiak/openfork-ltx23-wan2gp-original:latest"

        if [ ! -f "$LTX23_REQUIRED_TRANSFORMER" ]; then
            log "ERROR: $SERVICE_TYPE requires $(basename "$LTX23_REQUIRED_TRANSFORMER"), but this image does not contain it."
            log "Use beschiak/openfork-ltx23-wan2gp-8gb:latest for the 8GB tier."
            log "Use beschiak/openfork-ltx23-wan2gp-12gb:latest for the 12GB tier."
            log "Use beschiak/openfork-ltx23-wan2gp-hdr:latest for the 16GB and 24GB tiers."
            log "Use beschiak/openfork-ltx23-wan2gp:latest for the 32GB tier."
            log "Expected image for this service: $LTX23_EXPECTED_IMAGE"
            WAN2GP_CHECK_FAILED=1
        fi

        if [[ "$LTX23_EXPECTED_IMAGE" == *"-hdr:"* ]]; then
            LTX23_HDR_LORA="$WAN2GP_ROOT/ckpts/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors"
            LTX23_HDR_SCENE_EMB="$WAN2GP_ROOT/ckpts/ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors"
            if [ ! -f "$LTX23_HDR_LORA" ] || [ ! -f "$LTX23_HDR_SCENE_EMB" ]; then
                log "ERROR: $SERVICE_TYPE requires HDR IC-LoRA files, but this image does not contain them."
                log "Missing HDR LoRA: $LTX23_HDR_LORA"
                log "Missing HDR scene embedding: $LTX23_HDR_SCENE_EMB"
                log "Expected image for this service: $LTX23_EXPECTED_IMAGE"
                WAN2GP_CHECK_FAILED=1
            fi
        fi
    fi

    if [[ "${SERVICE_TYPE:-}" == *"wan22-wan2gp"* ]]; then
        if [[ "$SERVICE_TYPE" == *"8gb"* ]]; then
            WAN22_TRANSFORMER_VARIANT="quanto_mfp16_int8"
            WAN22_EXPECTED_IMAGE="beschiak/openfork-wan22-wan2gp-8gb:latest"
        elif [[ "$SERVICE_TYPE" == *"10gb"* ]]; then
            WAN22_TRANSFORMER_VARIANT="quanto_mfp16_int8"
            WAN22_EXPECTED_IMAGE="beschiak/openfork-wan22-wan2gp-10gb:latest"
        elif [[ "$SERVICE_TYPE" == *"12gb"* ]]; then
            WAN22_TRANSFORMER_VARIANT="quanto_mfp16_int8"
            WAN22_EXPECTED_IMAGE="beschiak/openfork-wan22-wan2gp-12gb:latest"
        elif [[ "$SERVICE_TYPE" == *"24gb"* ]]; then
            WAN22_TRANSFORMER_VARIANT="quanto_mbf16_int8"
            WAN22_EXPECTED_IMAGE="beschiak/openfork-wan22-wan2gp-24gb:latest"
        else
            WAN22_TRANSFORMER_VARIANT="quanto_mbf16_int8"
            WAN22_EXPECTED_IMAGE="beschiak/openfork-wan22-wan2gp-16gb:latest"
        fi

        WAN22_T2V_HIGH_TRANSFORMER="$WAN2GP_ROOT/ckpts/wan2.2_text2video_14B_high_${WAN22_TRANSFORMER_VARIANT}.safetensors"
        WAN22_T2V_LOW_TRANSFORMER="$WAN2GP_ROOT/ckpts/wan2.2_text2video_14B_low_${WAN22_TRANSFORMER_VARIANT}.safetensors"
        WAN22_I2V_HIGH_TRANSFORMER="$WAN2GP_ROOT/ckpts/wan2.2_image2video_14B_high_${WAN22_TRANSFORMER_VARIANT}.safetensors"
        WAN22_I2V_LOW_TRANSFORMER="$WAN2GP_ROOT/ckpts/wan2.2_image2video_14B_low_${WAN22_TRANSFORMER_VARIANT}.safetensors"
        WAN22_VAE_21="$WAN2GP_ROOT/ckpts/Wan2.1_VAE.safetensors"
        WAN22_VAE_22="$WAN2GP_ROOT/ckpts/Wan2.2_VAE.safetensors"
        WAN22_TEXT_ENCODER="$WAN2GP_ROOT/ckpts/umt5-xxl/models_t5_umt5-xxl-enc-bf16.safetensors"
        WAN22_TEXT_ENCODER_INT8="$WAN2GP_ROOT/ckpts/umt5-xxl/models_t5_umt5-xxl-enc-quanto_int8.safetensors"
        WAN22_TOKENIZER_MODEL="$WAN2GP_ROOT/ckpts/umt5-xxl/spiece.model"
        WAN22_TOKENIZER_CONFIG="$WAN2GP_ROOT/ckpts/umt5-xxl/tokenizer_config.json"

        for required_file in "$WAN22_T2V_HIGH_TRANSFORMER" "$WAN22_T2V_LOW_TRANSFORMER" "$WAN22_I2V_HIGH_TRANSFORMER" "$WAN22_I2V_LOW_TRANSFORMER" "$WAN22_VAE_21" "$WAN22_VAE_22" "$WAN22_TOKENIZER_MODEL" "$WAN22_TOKENIZER_CONFIG"; do
            if [ ! -f "$required_file" ]; then
                log "ERROR: $SERVICE_TYPE requires $(basename "$required_file"), but this image does not contain it."
                log "Expected image for this service: $WAN22_EXPECTED_IMAGE"
                WAN2GP_CHECK_FAILED=1
            fi
        done

        if [ ! -f "$WAN22_TEXT_ENCODER" ] && [ ! -f "$WAN22_TEXT_ENCODER_INT8" ]; then
            log "ERROR: $SERVICE_TYPE requires a UMT5 text encoder, but this image does not contain one."
            log "Expected image for this service: $WAN22_EXPECTED_IMAGE"
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
        SCAIL_SCRIBBLE="$WAN2GP_ROOT/ckpts/scribble/netG_A_latest.pth"
        SCAIL_FLOW="$WAN2GP_ROOT/ckpts/flow/raft-things.pth"
        SCAIL_DEPTH="$WAN2GP_ROOT/ckpts/depth/depth_anything_v2_vitl.pth"
        SCAIL_DEPTH_VITB="$WAN2GP_ROOT/ckpts/depth/depth_anything_v2_vitb.pth"
        SCAIL_WAV2VEC_CONFIG="$WAN2GP_ROOT/ckpts/wav2vec/config.json"
        SCAIL_WAV2VEC_FEATURES="$WAN2GP_ROOT/ckpts/wav2vec/feature_extractor_config.json"
        SCAIL_WAV2VEC_MODEL="$WAN2GP_ROOT/ckpts/wav2vec/model.safetensors"
        SCAIL_WAV2VEC_PREPROCESSOR="$WAN2GP_ROOT/ckpts/wav2vec/preprocessor_config.json"
        SCAIL_WAV2VEC_SPECIAL_TOKENS="$WAN2GP_ROOT/ckpts/wav2vec/special_tokens_map.json"
        SCAIL_WAV2VEC_TOKENIZER="$WAN2GP_ROOT/ckpts/wav2vec/tokenizer_config.json"
        SCAIL_WAV2VEC_VOCAB="$WAN2GP_ROOT/ckpts/wav2vec/vocab.json"
        SCAIL_CHINESE_WAV2VEC_CONFIG="$WAN2GP_ROOT/ckpts/chinese-wav2vec2-base/config.json"
        SCAIL_CHINESE_WAV2VEC_MODEL="$WAN2GP_ROOT/ckpts/chinese-wav2vec2-base/pytorch_model.bin"
        SCAIL_CHINESE_WAV2VEC_PREPROCESSOR="$WAN2GP_ROOT/ckpts/chinese-wav2vec2-base/preprocessor_config.json"
        SCAIL_ROFORMER_MODEL="$WAN2GP_ROOT/ckpts/roformer/model_bs_roformer_ep_317_sdr_12.9755.ckpt"
        SCAIL_ROFORMER_CONFIG="$WAN2GP_ROOT/ckpts/roformer/model_bs_roformer_ep_317_sdr_12.9755.yaml"
        SCAIL_ROFORMER_CHECKS="$WAN2GP_ROOT/ckpts/roformer/download_checks.json"
        SCAIL_PYANNOTE_WESPEAKER="$WAN2GP_ROOT/ckpts/pyannote/pyannote_model_wespeaker-voxceleb-resnet34-LM.bin"
        SCAIL_PYANNOTE_SEGMENTATION="$WAN2GP_ROOT/ckpts/pyannote/pytorch_model_segmentation-3.0.bin"
        SCAIL_DET_ALIGN="$WAN2GP_ROOT/ckpts/det_align/detface.pt"
        SCAIL_MASK_SAM="$WAN2GP_ROOT/ckpts/mask/sam_vit_h_4b8939_fp16.safetensors"
        SCAIL_MASK_MATANYONE="$WAN2GP_ROOT/ckpts/mask/matanyone.safetensors"
        SCAIL_MASK_CONFIG="$WAN2GP_ROOT/ckpts/mask/config.json"
        SCAIL_RIFE="$WAN2GP_ROOT/ckpts/rife4.26.pkl"
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

        for required_file in "$SCAIL_VAE" "$SCAIL_POSE_MODEL" "$SCAIL_POSE_META" "$SCAIL_SCRIBBLE" "$SCAIL_FLOW" "$SCAIL_DEPTH" "$SCAIL_DEPTH_VITB" "$SCAIL_WAV2VEC_CONFIG" "$SCAIL_WAV2VEC_FEATURES" "$SCAIL_WAV2VEC_MODEL" "$SCAIL_WAV2VEC_PREPROCESSOR" "$SCAIL_WAV2VEC_SPECIAL_TOKENS" "$SCAIL_WAV2VEC_TOKENIZER" "$SCAIL_WAV2VEC_VOCAB" "$SCAIL_CHINESE_WAV2VEC_CONFIG" "$SCAIL_CHINESE_WAV2VEC_MODEL" "$SCAIL_CHINESE_WAV2VEC_PREPROCESSOR" "$SCAIL_ROFORMER_MODEL" "$SCAIL_ROFORMER_CONFIG" "$SCAIL_ROFORMER_CHECKS" "$SCAIL_PYANNOTE_WESPEAKER" "$SCAIL_PYANNOTE_SEGMENTATION" "$SCAIL_DET_ALIGN" "$SCAIL_MASK_SAM" "$SCAIL_MASK_MATANYONE" "$SCAIL_MASK_CONFIG" "$SCAIL_RIFE"; do
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
    if [[ "${SERVICE_TYPE:-}" == *"scail"* ]] || [[ "${SERVICE_TYPE:-}" == *"ltx23"* ]] || [[ "${SERVICE_TYPE:-}" == *"wan22-wan2gp-24gb"* ]]; then
        export WAN2GP_EXIT_AFTER_JOB="${WAN2GP_EXIT_AFTER_JOB:-1}"
        export WAN2GP_EXIT_DELAY_SECONDS="${WAN2GP_EXIT_DELAY_SECONDS:-1}"
    fi
    if verify_wan2gp_stable_thread_wrapper "$WAN2GP_SERVER"; then
        log "Verified Wan2GP stable main-thread recycle wrapper."
    elif [[ "${SERVICE_TYPE:-}" == *"scail"* ]]; then
        log "ERROR: SCAIL requires the stable main-thread recycle Wan2GP wrapper."
        log "ERROR: $WAN2GP_SERVER is stale; update OPENFORK_CLIENT_SCRIPT_REF or rebuild the SCAIL image/client source before accepting SCAIL jobs."
        exit 1
    else
        log "WARNING: Wan2GP stable main-thread recycle wrapper was not detected; continuing for non-SCAIL service ${SERVICE_TYPE:-auto}."
    fi
    if [ -f "$WAN2GP_SERVER" ]; then
        log "Starting Wan2GP HTTP server supervisor (logging to ${WAN2GP_LOG_FILE})..."
        : > "$WAN2GP_LOG_FILE"
        start_wan2gp_server_supervisor "$WAN2GP_SERVER" "$WAN2GP_ROOT" "$WAN2GP_LOG_FILE" &
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
    *dreamid*omni*|*dreamid-omni*)
      log "Applying 24GB optimizations for DreamID-Omni FP8"
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --reserve-vram 1.0 --use-pytorch-cross-attention --cache-none"
      ;;
    *flux-kontext*8gb*)
      log "Applying FLUX Kontext 8GB GGUF optimizations"
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --cpu-vae --fp16-unet --reserve-vram 1.5 --cache-none --use-split-cross-attention --preview-method none --disable-dynamic-vram --disable-async-offload --disable-pinned-memory"
      ;;
    *flux-kontext*12gb*)
      log "Applying FLUX Kontext 12GB GGUF optimizations"
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --fp16-vae --reserve-vram 1.0 --use-split-cross-attention --cache-none --preview-method none --disable-pinned-memory"
      ;;
    *flux-kontext*16gb*)
      log "Applying FLUX Kontext 16GB GGUF optimizations"
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --cpu-vae --reserve-vram 1.0 --use-split-cross-attention --cache-none --preview-method none --disable-pinned-memory"
      ;;
    *flux-kontext*24gb*)
      log "Applying FLUX Kontext 24GB GGUF optimizations"
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --fp16-vae --reserve-vram 1.5 --use-pytorch-cross-attention --preview-method none"
      ;;
    qwen-8gb|qwen-turbo-8gb)
      log "Applying Qwen Image 8GB VRAM optimizations"
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --fp16-vae --reserve-vram 1.0 --use-split-cross-attention --cache-none --preview-method none --disable-async-offload --disable-pinned-memory"
      ;;
    qwen)
      log "Applying Qwen Image 12GB VRAM optimizations"
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --fp16-vae --reserve-vram 1.0 --use-split-cross-attention --cache-none --preview-method none --disable-pinned-memory"
      ;;
    wan22|wan22-8gb)
      log "Applying WAN 2.2 8GB Q4 ComfyUI optimizations"
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --fp16-vae --reserve-vram 0.5 --cache-none --preview-method none --use-split-cross-attention --disable-async-offload --disable-cuda-malloc --disable-pinned-memory"
      ;;
    wan22-16gb)
      log "Applying WAN 2.2 16GB Q6 ComfyUI optimizations"
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --fp16-vae --reserve-vram 0.75 --cache-none --preview-method none --use-split-cross-attention --disable-async-offload --disable-cuda-malloc --disable-pinned-memory"
      ;;
    wan22-24gb)
      log "Applying WAN 2.2 24GB Q8 ComfyUI optimizations"
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --fp16-vae --reserve-vram 1.0 --cache-none --preview-method none --use-split-cross-attention --disable-async-offload --disable-cuda-malloc --disable-pinned-memory"
      ;;
    *wan22*)
      log "Applying WAN 2.2 fallback ComfyUI optimizations"
      COMFY_FLAGS="$COMFY_FLAGS --lowvram --fp16-vae --reserve-vram 0.75 --cache-none --preview-method none --use-split-cross-attention --disable-async-offload --disable-cuda-malloc --disable-pinned-memory"
      ;;
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

  if [[ "$SERVICE_TYPE" == *"flux-kontext"* ]] || [[ "$SERVICE_TYPE" == qwen* ]] || [[ "$SERVICE_TYPE" == *"wan22"* ]]; then
    mkdir -p /opt/ComfyUI/user/__manager
    cat > /opt/ComfyUI/user/__manager/config.ini <<'EOF'
[default]
network_mode = offline
EOF
    log "Configured ComfyUI-Manager network_mode=offline for ComfyUI runtime."
  fi

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

# Start F5-TTS REST API
if [ "$START_F5TTS" = "true" ] && [ -f "/app/f5_tts_api.py" ]; then
  log "Found F5-TTS API script. Starting..."
  (cd /app && "$PYTHON_EXE" f5_tts_api.py > /tmp/f5_tts_api.log 2>&1) &
  wait_for_url "F5-TTS API" "http://127.0.0.1:8000/health" 300 "/tmp/f5_tts_api.log"
fi

# Start WavTTS REST API
if [ "$START_WAVTTS" = "true" ] && [ -f "/app/wavtts_api.py" ]; then
  log "Found WavTTS API script. Starting..."
  (cd /app && "$PYTHON_EXE" wavtts_api.py > /tmp/wavtts_api.log 2>&1) &
  wait_for_url "WavTTS API" "http://127.0.0.1:8000/health" 300 "/tmp/wavtts_api.log"
fi

# Start PRiSM Audio REST API
if [ "$START_PRISMAUDIO" = "true" ] && [ -f "/app/prismaudio_api.py" ]; then
  log "Found PRiSM Audio API script. Starting..."
  if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "prismaudio"; then
    log "Using prismaudio conda environment for PRiSM Audio API."
    (cd /app && HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" conda run --no-capture-output -n prismaudio python prismaudio_api.py > /tmp/prismaudio_api.log 2>&1) &
  else
    log "WARNING: prismaudio conda environment not found. Falling back to $PYTHON_EXE."
    (cd /app && HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" "$PYTHON_EXE" prismaudio_api.py > /tmp/prismaudio_api.log 2>&1) &
  fi
  wait_for_url "PRiSM Audio API" "http://127.0.0.1:8000/health" 900 "/tmp/prismaudio_api.log"
fi

# Start MMAudio REST API
if [ "$START_MMAUDIO" = "true" ] && [ -f "/app/mmaudio_api.py" ]; then
  log "Found MMAudio API script. Starting..."
  (cd /app && HF_HUB_OFFLINE="${MMAUDIO_HF_HUB_OFFLINE:-0}" "$PYTHON_EXE" mmaudio_api.py > /tmp/mmaudio_api.log 2>&1) &
  wait_for_url "MMAudio API" "http://127.0.0.1:8000/health" 900 "/tmp/mmaudio_api.log"
fi

# Start ACE-Step REST API
if [ "$START_ACESTEP" = "true" ]; then
  log "Starting ACE-Step API service..."
  if [ -d "/app/acestep_repo" ]; then
    (cd /app/acestep_repo && HF_HUB_OFFLINE="${ACESTEP_HF_HUB_OFFLINE:-1}" uv run acestep-api > /tmp/acestep_api.log 2>&1) &
  else
    (HF_HUB_OFFLINE="${ACESTEP_HF_HUB_OFFLINE:-1}" uv run acestep-api > /tmp/acestep_api.log 2>&1) &
  fi
  wait_for_url "ACE-Step API" "http://127.0.0.1:8000/health" 900 "/tmp/acestep_api.log"
fi

# Start Scenema Audio REST API
if [ "$START_SCENEMA_AUDIO" = "true" ] && [ -f "/app/src/server.py" ]; then
  log "Found Scenema Audio server. Starting..."
  (cd /app/src && "$PYTHON_EXE" -m server > /tmp/scenema_audio_api.log 2>&1) &
  wait_for_url "Scenema Audio API" "http://127.0.0.1:8000/health" 900 "/tmp/scenema_audio_api.log"
fi

# Start DramaBox REST API
if [ "$START_DRAMABOX" = "true" ] && [ -f "/app/dramabox_api.py" ]; then
  log "Found DramaBox API script. Starting..."
  export DRAMABOX_MODEL_CACHE="${DRAMABOX_MODEL_CACHE:-/app/models}"
  (cd /app && "$PYTHON_EXE" dramabox_api.py > /tmp/dramabox_api.log 2>&1) &
  wait_for_url "DramaBox API" "http://127.0.0.1:8000/health" 900 "/tmp/dramabox_api.log"
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
  if ! "$PYTHON_EXE" - <<'PY' >/tmp/turbodiffusion_dependency_check.log 2>&1
import importlib

modules = [
    "loguru",
    "pandas",
    "yaml",
    "omegaconf",
    "attr",
    "fvcore",
    "ftfy",
    "regex",
    "transformers",
    "pynvml",
    "accelerate",
    "diffusers",
    "sentencepiece",
    "cv2",
    "av",
    "scipy",
    "skimage",
    "lmdb",
    "tensorboard",
    "lpips",
    "matplotlib",
    "tqdm",
    "requests",
    "iopath",
    "prompt_toolkit",
    "rich",
    "turbo_diffusion_ops",
]

missing = []
for module in modules:
    try:
        importlib.import_module(module)
    except Exception:
        missing.append(module)

if missing:
    raise SystemExit("missing " + ", ".join(missing))
PY
  then
    log "TurboDiffusion runtime dependencies missing; installing repair set."
    sed 's/^/  /' /tmp/turbodiffusion_dependency_check.log || true
    "$PYTHON_EXE" -m pip install --quiet --no-cache-dir \
      loguru pandas pyyaml omegaconf attrs fvcore ftfy regex transformers nvidia-ml-py \
      accelerate diffusers sentencepiece opencv-python av scipy scikit-image lmdb tensorboard \
      lpips matplotlib tqdm requests iopath prompt-toolkit rich || \
      log "WARNING: Failed to install one or more TurboDiffusion runtime dependencies; inference may fail."
  fi
  if ! "$PYTHON_EXE" - <<'PY' >/tmp/turbodiffusion_ops_check.log 2>&1
import turbo_diffusion_ops  # noqa: F401
PY
  then
    log "TurboDiffusion CUDA extension is missing; attempting runtime rebuild."
    sed 's/^/  /' /tmp/turbodiffusion_ops_check.log || true
    (cd /opt/TurboDiffusion && MAX_JOBS=2 "$PYTHON_EXE" -m pip install --no-cache-dir -e . --no-build-isolation) || \
      log "WARNING: TurboDiffusion CUDA extension rebuild command failed."
  fi
  if ! "$PYTHON_EXE" - <<'PY' >/tmp/turbodiffusion_ops_check_after.log 2>&1
import turbo_diffusion_ops  # noqa: F401
PY
  then
    log "ERROR: TurboDiffusion CUDA extension is still unavailable; refusing to start the API."
    sed 's/^/  /' /tmp/turbodiffusion_ops_check_after.log || true
    case ",${SELECTED_WORKFLOWS:-}," in
      *,turbodiffusion-image-to-video,*|*,turbodiffusion-text-to-video,*)
        exit 78
        ;;
    esac
  else
    (cd /opt/TurboDiffusion && "$PYTHON_EXE" api_server.py > /tmp/turbodiffusion_api.log 2>&1) &
    wait_for_url "TurboDiffusion API" "http://127.0.0.1:8000/health" 120 "/tmp/turbodiffusion_api.log"
  fi
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
      # 2026-06-09 paid image smoke: full-GPU ERNIE 24GB on RTX 3090 filled
      # ~23.5GiB and failed on a small extra allocation during inference. Keep
      # prompt enhancement disabled and default to model CPU offload for real
      # 24GB compatibility; this is slower, but avoids stranding paid jobs.
      export ERNIE_USE_PE="${ERNIE_USE_PE:-false}"
      export ERNIE_ENABLE_CPU_OFFLOAD="${ERNIE_ENABLE_CPU_OFFLOAD:-true}"
      export ERNIE_ENABLE_ATTENTION_SLICING="${ERNIE_ENABLE_ATTENTION_SLICING:-true}"
      export ERNIE_ENABLE_VAE_TILING="true"
    fi
    log "ERNIE-Image config: model=$ERNIE_MODEL_ID dtype=$ERNIE_DTYPE steps=$ERNIE_DEFAULT_STEPS use_pe=${ERNIE_USE_PE:-true} cpu_offload=${ERNIE_ENABLE_CPU_OFFLOAD:-false}"
    (cd /app && "$PYTHON_EXE" ernie_image_api.py > /tmp/ernie_image_api.log 2>&1) &
    wait_for_url "ERNIE-Image API" "http://127.0.0.1:8000/health" 600 "/tmp/ernie_image_api.log"
  else
    log "ERROR: ERNIE-Image API not found at /app/ernie_image_api.py"
  fi
fi

# Start Ideogram 4 REST API
if [ "$START_IDEOGRAM4" = "true" ]; then
  log "Starting Ideogram 4 API service..."
  if [ -f "/app/ideogram4_api.py" ]; then
    refresh_openfork_file "comfyui-storage/ideogram4_api.py" "/app/ideogram4_api.py" || true
    if [ -f "/app/services/processors/image/ideogram4.py" ]; then
      refresh_openfork_file "services/processors/image/ideogram4.py" "/app/services/processors/image/ideogram4.py" || true
    fi
    if [[ "${SERVICE_TYPE:-}" == *"32gb"* ]] || [ "${IDEOGRAM_QUANTIZATION:-}" = "fp8" ]; then
      export IDEOGRAM_QUANTIZATION="${IDEOGRAM_QUANTIZATION:-fp8}"
      export IDEOGRAM_SAMPLER_PRESET="${IDEOGRAM_SAMPLER_PRESET:-V4_QUALITY_48}"
    elif [[ "${SERVICE_TYPE:-}" == *"24gb"* ]]; then
      export IDEOGRAM_QUANTIZATION="${IDEOGRAM_QUANTIZATION:-nf4}"
      export IDEOGRAM_SAMPLER_PRESET="${IDEOGRAM_SAMPLER_PRESET:-V4_QUALITY_48}"
    else
      export IDEOGRAM_QUANTIZATION="${IDEOGRAM_QUANTIZATION:-nf4}"
      export IDEOGRAM_SAMPLER_PRESET="${IDEOGRAM_SAMPLER_PRESET:-V4_DEFAULT_20}"
    fi
    export IDEOGRAM_USE_MAGIC_PROMPT="${IDEOGRAM_USE_MAGIC_PROMPT:-0}"
    export IDEOGRAM_MODEL_LOAD_TIMEOUT="${IDEOGRAM_MODEL_LOAD_TIMEOUT:-1800}"
    export IDEOGRAM4_API_WAIT_TIMEOUT="${IDEOGRAM4_API_WAIT_TIMEOUT:-1800}"
    log "Ideogram 4 config: quantization=$IDEOGRAM_QUANTIZATION preset=$IDEOGRAM_SAMPLER_PRESET magic_prompt=$IDEOGRAM_USE_MAGIC_PROMPT model_load_timeout=$IDEOGRAM_MODEL_LOAD_TIMEOUT"
    (cd /app && "$PYTHON_EXE" ideogram4_api.py > /tmp/ideogram4_api.log 2>&1) &
    if ! wait_for_url "Ideogram 4 API" "http://127.0.0.1:8000/health" 900 "/tmp/ideogram4_api.log"; then
      log "ERROR: Ideogram 4 API did not expose /health; not starting the DGN client."
      exit 1
    fi
    if ! wait_for_model_loaded "Ideogram 4" "http://127.0.0.1:8000/health" "$IDEOGRAM_MODEL_LOAD_TIMEOUT" "/tmp/ideogram4_api.log"; then
      log "ERROR: Ideogram 4 model did not become ready; not starting the DGN client."
      exit 1
    fi
  else
    log "ERROR: Ideogram 4 API not found at /app/ideogram4_api.py"
    exit 1
  fi
fi

# Start PiD image upscale REST API
if [ "$START_PID_IMAGE" = "true" ]; then
  log "Starting PiD image upscale API service..."
  PID_API=""
  PID_CD=""
  if [ -f "/app/PiD/pid_image_api.py" ]; then
    PID_API="/app/PiD/pid_image_api.py"
    PID_CD="/app/PiD"
  elif [ -f "/app/pid_image_api.py" ]; then
    PID_API="/app/pid_image_api.py"
    PID_CD="/app"
  fi

  if [ -n "$PID_API" ]; then
    export PID_REPO_DIR="${PID_REPO_DIR:-/app/PiD}"
    export PID_BACKBONE="${PID_BACKBONE:-zimage}"
    export PID_CKPT_TYPE="${PID_CKPT_TYPE:-2k}"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

    if [ ! -d "${PID_REPO_DIR}/checkpoints/PiD_res2k_sr4x_official_flux_distill_4step" ]; then
      log "WARNING: PiD 2k Flux/Z-Image checkpoint directory was not found under ${PID_REPO_DIR}/checkpoints."
    fi

    PORT8000_BOUND=false
    if command -v netstat &> /dev/null; then
      if netstat -tln | grep -q ":8000 "; then PORT8000_BOUND=true; fi
    elif command -v ss &> /dev/null; then
      if ss -tln | grep -q ":8000 "; then PORT8000_BOUND=true; fi
    fi

    if [ "$PORT8000_BOUND" = "true" ]; then
      log "Port 8000 already bound — PiD API is already starting. Skipping redundant launch."
    else
      log "Found PiD API at $PID_API. Starting..."
      (cd "$PID_CD" && "$PYTHON_EXE" "$PID_API" > /tmp/pid_image_api.log 2>&1) &
    fi

    wait_for_url "PiD image API" "http://127.0.0.1:8000/health" 120 "/tmp/pid_image_api.log"

    PID_WAITED=0
    PID_MAX_WAIT=900
    while [ "$PID_WAITED" -lt "$PID_MAX_WAIT" ]; do
      PID_HEALTH=$(curl -fsS "http://127.0.0.1:8000/health" 2>/dev/null || true)
      if echo "$PID_HEALTH" | grep -Eq '"model_loaded"[[:space:]]*:[[:space:]]*true'; then
        log "PiD image API model is loaded."
        break
      fi
      if echo "$PID_HEALTH" | grep -Eq '"status"[[:space:]]*:[[:space:]]*"error"'; then
        log "ERROR: PiD image API reported a model load error: $PID_HEALTH"
        [ -f "/tmp/pid_image_api.log" ] && tail -n 80 /tmp/pid_image_api.log | sed 's/^/  [pid] /'
        exit 1
      fi
      [ $((PID_WAITED % 60)) -eq 0 ] && log "PiD model still loading... (${PID_WAITED}/${PID_MAX_WAIT}s)"
      sleep 10
      PID_WAITED=$((PID_WAITED + 10))
    done

    if [ "$PID_WAITED" -ge "$PID_MAX_WAIT" ]; then
      log "ERROR: PiD image API model did not load within ${PID_MAX_WAIT}s."
      [ -f "/tmp/pid_image_api.log" ] && tail -n 80 /tmp/pid_image_api.log | sed 's/^/  [pid] /'
      exit 1
    fi
  else
    log "ERROR: PiD API not found at /app/PiD/pid_image_api.py or /app/pid_image_api.py"
    exit 1
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
  if [[ "$SERVICE_TYPE" == *"24gb"* ]] || [[ "$SERVICE_TYPE" == *"ltx23"* ]] || [[ "$SERVICE_TYPE" == *"ltx2"* ]] || [[ "$SERVICE_TYPE" == *"flux-kontext"* ]]; then
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

if [ "${OPENFORK_ALLOW_DISABLED_SERVICE_TEST:-0}" = "1" ] || [ "${OPENFORK_ALLOW_DISABLED_SERVICE_TEST:-false}" = "true" ]; then
  log "Disabled-service smoke-test override enabled for ${SERVICE_TYPE:-auto}."
  DGN_CLIENT_ARGS+=(--allow-disabled-service-test)
fi

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

start_resource_monitor

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
