'''
Configuration for the DGN Client
'''
import os
import sys

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        """Gracefully degrade when python-dotenv is not installed."""
        return False


load_dotenv()

# --- General Configuration ---
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle (e.g., by PyInstaller)
    ROOT_DIR = os.path.dirname(sys.executable)
else:
    # If running as a script in a normal Python environment
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(ROOT_DIR)

CACHE_DIR = os.path.join(ROOT_DIR, '.cache')
DEV_MODE = False
THUMBNAIL_WIDTH = int(os.getenv("THUMBNAIL_WIDTH", "512"))
MAX_INPUT_ASSET_BYTES = int(
    os.getenv("MAX_INPUT_ASSET_BYTES", "1073741824")
)
MAX_INPUT_ASSET_REDIRECTS = int(
    os.getenv("MAX_INPUT_ASSET_REDIRECTS", "3")
)

# Policy-specific cache caps for local Docker images at the Healthy disk-pressure tier.
# None means uncapped for that policy at Healthy. Caps may shrink under disk pressure
# (see services.disk_pressure.get_effective_cap).
POLICY_MAX_CACHED_IMAGES = {
    "monetize": 3,
    "all": 4,
    "project": 6,
    "users": 6,
    "mine": None,
}

# Policy-specific idle timeouts for the Electron-side cleanup notifier (minutes).
# Mirrored on the desktop side via /api/config so both layers stay in sync.
# None means idle eviction is disabled at Healthy tier.
POLICY_IDLE_TIMEOUT_MINUTES = {
    "monetize": 90,
    "all": 120,
    "project": 240,
    "users": 240,
    "mine": None,
}

# Disk-pressure thresholds (GB free at the Docker storage path).
# These trigger LRU eviction independent of policy caps.
#   Healthy:  free > DISK_PRESSURE_HEALTHY_GB         → honor per-policy caps as-is.
#   Pressure: CRITICAL < free <= HEALTHY              → cap × 0.6 (min 2), idle × 0.5.
#   Critical: free <= DISK_PRESSURE_CRITICAL_GB        → evict until back above pressure;
#                                                        block new pulls; signal compaction.
DISK_PRESSURE_HEALTHY_GB = int(os.getenv("DISK_PRESSURE_HEALTHY_GB", "50"))
DISK_PRESSURE_CRITICAL_GB = int(os.getenv("DISK_PRESSURE_CRITICAL_GB", "20"))

# Effective cap for "mine" policy at non-Healthy tiers. Cannot be None because we
# need a finite number to drive eviction. At Pressure use this directly; at Critical
# use floor(this × 0.6) (so eviction is more aggressive even for the user's own images).
MINE_POLICY_PRESSURE_CAP = int(os.getenv("MINE_POLICY_PRESSURE_CAP", "8"))

# Headless mode detection - when running inside a cloud container (RunPod/Vast.ai),
# Docker operations should be skipped as ComfyUI is already running in the same container.
# Detection based on:
# 1. Environment vars set by cloud providers or deploy scripts
# 2. File markers created by cloud providers (Vast.ai creates ~/.vast_containerlabel)
def _is_vast_container():
    """Check if running inside a Vast.ai container by looking for marker files."""
    return (
        os.path.exists(os.path.expanduser("~/.vast_containerlabel")) or
        os.path.exists(os.path.expanduser("~/.vast_api_key"))
    )

HEADLESS_MODE = any([
    os.environ.get("RUNPOD_POD_ID"),      # RunPod sets this automatically
    os.environ.get("VAST_CONTAINERLABEL"), # Vast.ai container environment (via env var)
    _is_vast_container(),                  # Vast.ai container (via file detection)
    os.path.exists("/.dockerenv"),         # Generic Docker container detection
    os.environ.get("HEADLESS_MODE", "").lower() in ("1", "true", "yes"),  # Explicit flag
])

# --- Supabase Configuration ---
# Production Supabase project
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://lhwcmiialdwsmtoikgqb.supabase.co")

# Supabase publishable key for Realtime WebSocket connections.
# This is the project's publishable/anon key (format: sb_publishable_... or legacy anon JWT).
# Required for connecting to Supabase Realtime via WebSocket (the apikey param in the URL).
# WARNING: Do NOT use a user's access token JWT here - use the project's publishable/anon key.
SUPABASE_PUBLISHABLE_KEY = os.getenv(
    "SUPABASE_PUBLISHABLE_KEY",
    os.getenv(
        "NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY",
        "sb_publishable_zJTLSH1mMUStNoKD19veEw_ZXslKovE",
    ),
)

# Legacy anon key (JWT format) - fallback for older Supabase projects.
# Use SUPABASE_PUBLISHABLE_KEY for new projects with sb_publishable_ keys.
SUPABASE_ANON_KEY = os.getenv(
    "SUPABASE_ANON_KEY",
    "",
)

# --- Orchestrator Configuration ---
ORCHESTRATOR_URL_PROD = os.getenv("ORCHESTRATOR_URL_PROD", "https://www.openfork.video")
ORCHESTRATOR_URL_DEV = os.getenv("ORCHESTRATOR_URL_DEV", "http://localhost:3000")


# --- Timeout Configuration ---
class TimeoutConfig:
    """
    Centralized timeout configuration with environment variable overrides.
    
    All timeouts are in seconds unless otherwise specified.
    """
    # ComfyUI wait time for the server to become ready
    COMFYUI_READY_TIMEOUT = int(os.getenv("COMFYUI_READY_TIMEOUT", "180"))
    
    # Maximum time to wait for a workflow to complete.
    # 30 minutes is sufficient for the longest video jobs; 2 hours would leave a hung
    # ComfyUI process tying up the GPU for far too long. Overridable via WORKFLOW_TIMEOUT env.
    WORKFLOW_TIMEOUT = int(os.getenv("WORKFLOW_TIMEOUT", "1800"))
    
    # Default timeout for HTTP API requests
    API_REQUEST_TIMEOUT = int(os.getenv("API_REQUEST_TIMEOUT", "30"))
    
    # Interval between job polling requests when no job is available
    JOB_POLL_INTERVAL = int(os.getenv("JOB_POLL_INTERVAL", "10"))
    
    # Faster polling interval for headless cloud clients (they are dedicated & cheap to check)
    HEADLESS_JOB_POLL_INTERVAL = int(os.getenv("HEADLESS_JOB_POLL_INTERVAL", "2"))
    
    # WebSocket connection timeout
    WEBSOCKET_TIMEOUT = int(os.getenv("WEBSOCKET_TIMEOUT", "600"))
    
    # Interval between heartbeat signals to orchestrator
    HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "30"))
    
    # Maximum retries for transient API failures
    API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "3"))
    
    # Minimum wait time between retries (exponential backoff)
    API_RETRY_MIN_WAIT = int(os.getenv("API_RETRY_MIN_WAIT", "1"))
    
    # Maximum wait time between retries
    API_RETRY_MAX_WAIT = int(os.getenv("API_RETRY_MAX_WAIT", "10"))


def _apply_overrides() -> None:
    """Apply user config overrides from a JSON file pointed to by env var.

    This runs at import time so downstream modules (disk_pressure, etc.)
    pick up the overridden values automatically.
    """
    import json

    path = os.environ.get("OPENFORK_CONFIG_OVERRIDES_PATH")
    if not path or not os.path.isfile(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logging.warning(f"Could not read config overrides from {path}: {e}")
        return

    if not isinstance(overrides, dict):
        return

    # Scalar overrides
    _scalar_keys = {
        "DISK_PRESSURE_HEALTHY_GB": int,
        "DISK_PRESSURE_CRITICAL_GB": int,
        "MINE_POLICY_PRESSURE_CAP": int,
    }
    for key, cast in _scalar_keys.items():
        val = overrides.get(key)
        if val is not None:
            try:
                globals()[key] = cast(val)
            except (ValueError, TypeError):
                logging.warning(f"Invalid config override for {key}: {val}")

    # Dict overrides — merge with defaults so partial overrides work
    _dict_keys = ["POLICY_MAX_CACHED_IMAGES", "POLICY_IDLE_TIMEOUT_MINUTES"]
    for key in _dict_keys:
        val = overrides.get(key)
        if isinstance(val, dict):
            base = globals().get(key, {})
            merged = {**base, **val}
            # Normalize explicit null back to Python None
            merged = {
                k: (None if v is None else v)
                for k, v in merged.items()
            }
            globals()[key] = merged

    logging.info(f"Applied config overrides from {path}.")


_apply_overrides()

