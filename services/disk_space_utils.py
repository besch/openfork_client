"""
Disk Space Utility Module

Provides cross-platform utilities for checking available disk space and 
estimating Docker image sizes before download.
"""

import os
import re
import shutil
import logging
import platform
import subprocess
from typing import Optional, Tuple


def _get_wsl_distro_base_path() -> Optional[str]:
    """
    Get the WSL distro base path from Windows registry.
    Returns the path where the WSL VHDX is stored (e.g., D:\\WSL\\OpenFork).
    Returns None if not found.
    """
    if platform.system() != "win32":
        return None

    distro = os.environ.get("OPENFORK_WSL_DISTRO") or "OpenFork"

    try:
        ps_command = f"Get-ItemProperty 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss\\*' | Where-Object DistributionName -eq '{distro}' | Select-Object -ExpandProperty BasePath"
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_command],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception as e:
        logging.debug(f"Could not get WSL distro base path: {e}")

    return None


def get_docker_storage_path() -> str:
    """
    Get the Docker storage directory path based on the platform and mode.

    Returns:
        str: Path to Docker storage directory
    """
    system = platform.system()

    if system == "Windows":
        # OpenFork on Windows stores Docker data inside its dedicated WSL distro.
        wsl_base = _get_wsl_distro_base_path()
        if wsl_base:
            # Extract drive letter from base path (e.g., D:\WSL\OpenFork -> D:)
            match = re.match(r"([a-zA-Z]):", wsl_base)
            if match:
                drive = match.group(1).upper()
                return f"{drive}:\\"
        # Fallback: check C: drive
        return "C:\\"
    elif system == "Darwin":  # macOS
        # Docker Desktop on macOS uses a VM disk image
        return os.path.expanduser("~/Library/Containers/com.docker.docker/Data")
    else:  # Linux
        # Standard Docker location on Linux
        return "/var/lib/docker"


def get_available_disk_space(path: Optional[str] = None) -> int:
    """
    Get available disk space in bytes for the given path.

    Args:
        path: Path to check disk space for. If None, uses Docker storage path.

    Returns:
        int: Available disk space in bytes
    """
    if path is None:
        path = get_docker_storage_path()

    # Ensure path exists, otherwise use root/home
    if not os.path.exists(path):
        logging.debug(f"Path '{path}' does not exist, checking root directory instead")
        if platform.system() == "Windows":
            path = "C:\\"
        else:
            path = "/"

    try:
        stat = shutil.disk_usage(path)
        return stat.free
    except Exception as e:
        logging.error(f"Failed to get disk space for '{path}': {e}")
        # Return a large value to avoid blocking downloads if check fails
        return 1000 * 1024**3  # 1TB fallback


def estimate_image_size_bytes(image_name: str) -> int:
    """
    Estimate Docker image size based on the image name pattern.

    This uses heuristics based on the services.json disk_required_gb values
    and common image naming patterns (8gb, 16gb, 24gb suffixes).

    Args:
        image_name: Docker image name (e.g., "beschiak/openfork-ltx2-8gb:latest")

    Returns:
        int: Estimated size in bytes
    """
    image_lower = image_name.lower()

    # Map of service patterns to estimated sizes (in GB)
    # These are conservative estimates based on services.json disk_required_gb
    size_map = {
        "llm": 20,
        "mmaudio": 60,
        "vibevoice": 80,
        "chatterbox": 80,
        "stream-diffvsr": 80,
        "audiox": 120,
        "stable-audio": 40,
        "dreamid-omni": 140,
        "ltx-video": 100,  # LTX v1
        "ltxvideo": 100,
        "diffrhythm": 100,
        "zimage": 120,
        "qwen": 120,
        "svi-12gb": 140,
        "wan22-24gb": 120,
        "svi-16gb": 150,
        "svi-24gb": 150,
        "wan22-8gb": 160,
        "wan22": 160,  # Default WAN22
        "hunyuan-video-16gb": 160,
        "hunyuan-16gb": 160,
        "turbodiffusion": 160,
        "hunyuan-video-24gb": 180,
        "hunyuan-24gb": 180,
        "hunyuan": 180,  # Default Hunyuan
        "ltx23-wan2gp-8gb": 160,
        "ltx23-wan2gp": 170,
        "ltx2-8gb": 180,
        "ltx2-16gb": 180,
        "wan22-16gb": 190,
        "ltx2-24gb": 210,
    }

    # Try to find a matching pattern
    estimated_gb = 100  # Default conservative estimate

    for pattern, size_gb in size_map.items():
        if pattern in image_lower:
            estimated_gb = size_gb
            break

    # Convert GB to bytes
    estimated_bytes = estimated_gb * 1024**3

    logging.debug(f"Estimated size for '{image_name}': {estimated_gb} GB ({estimated_bytes} bytes)")
    return estimated_bytes


def check_sufficient_space(
    required_bytes: int, 
    buffer_gb: float = 5.0,
    path: Optional[str] = None
) -> Tuple[bool, int, int]:
    """
    Check if there is sufficient disk space for a Docker image download.

    Args:
        required_bytes: Required space in bytes for the image
        buffer_gb: Additional safety buffer in GB (default: 5.0)
        path: Path to check disk space for (default: Docker storage path)

    Returns:
        Tuple of (has_sufficient_space, available_bytes, required_with_buffer_bytes)
    """
    available_bytes = get_available_disk_space(path)
    buffer_bytes = int(buffer_gb * 1024**3)
    required_with_buffer = required_bytes + buffer_bytes

    has_space = available_bytes >= required_with_buffer

    if has_space:
        logging.debug(
            f"Disk space check passed: {available_bytes / 1024**3:.1f} GB available, "
            f"{required_with_buffer / 1024**3:.1f} GB required (including {buffer_gb} GB buffer)"
        )
    else:
        logging.warning(
            f"Disk space check failed: {available_bytes / 1024**3:.1f} GB available, "
            f"{required_with_buffer / 1024**3:.1f} GB required (including {buffer_gb} GB buffer)"
        )

    return has_space, available_bytes, required_with_buffer


def get_disk_space_info(path: Optional[str] = None) -> dict:
    """
    Get comprehensive disk space information.

    Args:
        path: Path to check disk space for. If None, uses Docker storage path.

    Returns:
        dict: Dictionary with 'total', 'used', 'free' in bytes
    """
    if path is None:
        path = get_docker_storage_path()

    # Ensure path exists, otherwise use root/home
    if not os.path.exists(path):
        if platform.system() == "Windows":
            path = "C:\\"
        else:
            path = "/"

    try:
        stat = shutil.disk_usage(path)
        return {
            "total": stat.total,
            "used": stat.used,
            "free": stat.free,
            "path": path
        }
    except Exception as e:
        logging.error(f"Failed to get disk space info for '{path}': {e}")
        # Return default values
        return {
            "total": 0,
            "used": 0,
            "free": 0,
            "path": path
        }
