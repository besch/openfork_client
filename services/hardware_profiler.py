import cpuinfo
import GPUtil
import psutil
from typing import Optional, Tuple, List

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

    return {
        "cpu": cpu,
        "gpus": gpu_list,
        "ram": ram,
    }


def get_available_vram() -> int:
    """Get the total VRAM of the primary GPU in MB. Returns 0 if no GPU found."""
    gpus = GPUtil.getGPUs()
    if gpus:
        return int(gpus[0].memoryTotal)
    return 0


def can_run_service(service_config: dict, available_vram_mb: Optional[int] = None) -> bool:
    """
    Check if the current GPU can run a service based on VRAM requirements.
    
    Args:
        service_config: Service configuration dict containing 'vram_required_mb'
        available_vram_mb: Optional pre-fetched VRAM in MB. If None, will query GPU.
    
    Returns:
        True if GPU has enough VRAM, False otherwise
    """
    if available_vram_mb is None:
        available_vram_mb = get_available_vram()
    
    required_vram = service_config.get("vram_required_mb", 0)
    return available_vram_mb >= required_vram


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

