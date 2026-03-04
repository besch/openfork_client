import cpuinfo
import GPUtil
import psutil
from typing import Optional, Tuple, List
from services.disk_space_utils import get_disk_space_info

try:
    from config import HEADLESS_MODE
except ImportError:
    HEADLESS_MODE = False


def get_hardware_profile():
    """Get hardware profile of the system."""
    # Get CPU information
    cpu_info = cpuinfo.get_cpu_info()
    cpu = {
        "brand": cpu_info["brand_raw"],
        "cores": cpu_info["count"],
    }

    # Get GPU information
    gpus = GPUtil.getGPUs()
    gpu_list = []
    for gpu in gpus:
        gpu_list.append({
            "name": gpu.name,
            "vram": gpu.memoryTotal,
        })

    # Get RAM information
    ram = {
        "total": psutil.virtual_memory().total,
    }

    # Get Disk information
    disk_info = get_disk_space_info()
    disk = {
        "total": disk_info["total"],
        "free": disk_info["free"],
        "used": disk_info["used"],
        "path": disk_info["path"]
    }

    return {
        "cpu": cpu,
        "gpus": gpu_list,
        "ram": ram,
        "disk": disk,
    }



def get_available_vram() -> int:
    """Get the total VRAM of the primary GPU in MB. Returns 0 if no GPU found."""
    gpus = GPUtil.getGPUs()
    if gpus:
        return int(gpus[0].memoryTotal)
    return 0


def get_available_system_ram() -> int:
    """Get the total system RAM in MB. Returns 0 if unable to query."""
    try:
        return int(psutil.virtual_memory().total / (1024 * 1024))  # bytes to MB
    except Exception:
        return 0


def get_cpu_core_count() -> int:
    """Get the number of CPU cores (logical). Returns 0 if unable to query."""
    try:
        cpu_info = cpuinfo.get_cpu_info()
        return cpu_info.get("count", 0)
    except Exception:
        return 0


def can_run_service(service_config: dict, available_vram_mb: Optional[int] = None) -> bool:
    """
    Check if the current system can run a service based on all hardware requirements.
    
    Args:
        service_config: Service configuration dict containing:
            - 'vram_required_mb': GPU VRAM required in MB
            - 'cpu_ram_required_mb': System RAM required in MB (optional)
            - 'cpu_cores_required': CPU cores required (optional)
        available_vram_mb: Optional pre-fetched VRAM in MB. If None, will query GPU.
    
    Returns:
        True if system meets all requirements, False otherwise
    """
    # Check VRAM
    if available_vram_mb is None:
        available_vram_mb = get_available_vram()
    
    required_vram = service_config.get("vram_required_mb", 0)
    if available_vram_mb < required_vram:
        return False
    
    # Check CPU RAM (only if specified in config)
    required_cpu_ram = service_config.get("cpu_ram_required_mb")
    if required_cpu_ram:
        available_cpu_ram = get_available_system_ram()
        if available_cpu_ram < required_cpu_ram:
            return False
    
    # Check CPU cores (only if specified in config)
    required_cores = service_config.get("cpu_cores_required")
    if required_cores:
        available_cores = get_cpu_core_count()
        if available_cores < required_cores:
            return False

    # Check Disk Space (only if specified in config)
    # In headless/cloud mode the Docker image is pre-baked into the container,
    # so there is nothing to download and disk space is irrelevant.
    required_disk_gb = service_config.get("disk_required_gb")
    if required_disk_gb and not HEADLESS_MODE:
        from services.disk_space_utils import get_available_disk_space
        available_disk_bytes = get_available_disk_space()
        available_disk_gb = available_disk_bytes / (1024**3)
        if available_disk_gb < required_disk_gb:
            return False
    
    return True




def get_service_incompatibility_reason(service_config: dict, available_vram_mb: Optional[int] = None) -> Optional[str]:
    """
    Returns a human-readable reason why a service can't run, or None if it can run.
    
    Args:
        service_config: Service configuration dict
        available_vram_mb: Optional pre-fetched VRAM in MB
    
    Returns:
        String describing the incompatibility, or None if compatible
    """
    if available_vram_mb is None:
        available_vram_mb = get_available_vram()
    
    required_vram = service_config.get("vram_required_mb", 0)
    if available_vram_mb < required_vram:
        return f"requires {required_vram}MB VRAM, have {available_vram_mb}MB"
    
    # Check CPU RAM
    required_cpu_ram = service_config.get("cpu_ram_required_mb")
    if required_cpu_ram:
        available_cpu_ram = get_available_system_ram()
        if available_cpu_ram < required_cpu_ram:
            return f"requires {required_cpu_ram}MB system RAM, have {available_cpu_ram}MB"

    # Check CPU cores
    required_cores = service_config.get("cpu_cores_required")
    if required_cores:
        available_cores = get_cpu_core_count()
        if available_cores < required_cores:
            return f"requires {required_cores} CPU cores, have {available_cores}"

    
    # Check disk space
    # In headless/cloud mode the Docker image is pre-baked into the container,
    # so there is nothing to download and disk space is irrelevant.
    required_disk_gb = service_config.get("disk_required_gb")
    if required_disk_gb and not HEADLESS_MODE:
        from services.disk_space_utils import get_available_disk_space
        available_disk_bytes = get_available_disk_space()
        available_disk_gb = available_disk_bytes / (1024**3)
        if available_disk_gb < required_disk_gb:
            return f"requires {required_disk_gb}GB disk space, have {available_disk_gb:.1f}GB"
            
    return None



def get_compatible_services(services_config: dict) -> Tuple[List[str], List[str]]:
    """
    Get lists of compatible and incompatible services based on available VRAM.
    
    Args:
        services_config: Dict of service_name -> service_config
    
    Returns:
        Tuple of (compatible_services, incompatible_services) as lists of service names
    """
    available_vram = get_available_vram()
    compatible = []
    incompatible = []
    
    for service_name, config in services_config.items():
        if can_run_service(config, available_vram):
            compatible.append(service_name)
        else:
            incompatible.append(service_name)
    
    return compatible, incompatible


def get_vram_requirement_display(vram_mb: int) -> str:
    """Convert VRAM in MB to a human-readable string like '8GB'."""
    gb = vram_mb / 1000
    if gb >= 1:
        return f"{int(gb)}GB"
    return f"{vram_mb}MB"

