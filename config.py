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

# Policy-specific cache caps for local Docker images in auto mode.
# None means uncapped for that policy.
POLICY_MAX_CACHED_IMAGES = {
    "monetize": 3,
    "project": 3,
    "users": 3,
}

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
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imxod2NtaWlhbGR3c210b2lrZ3FiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDIxNzMxMzAsImV4cCI6MjA1Nzc0OTEzMH0.nZuZL4sD-4fsP5ZO2UpJKFcxsWM9kGfJjbhzKiIvnJA")

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

