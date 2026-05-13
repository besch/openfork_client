import cpuinfo
import csv
import GPUtil
import psutil
import re
import shutil
import subprocess
from typing import Optional, Tuple, List, Dict, Any
from services.disk_space_utils import get_disk_space_info

try:
    from config import HEADLESS_MODE
except ImportError:
    HEADLESS_MODE = False


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _run_nvidia_smi(args: List[str], timeout: int = 5) -> Optional[str]:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _get_nvidia_smi_cuda_version() -> Optional[float]:
    output = _run_nvidia_smi([], timeout=5)
    if not output:
        return None
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", output)
    return _parse_float(match.group(1)) if match else None


def _infer_compute_capability_from_name(name: Optional[str]) -> Optional[float]:
    if not name:
        return None
    upper = name.upper()
    if re.search(r"\b(B200|GB200)\b", upper) or re.search(r"\bRTX\s*(50\d{2}|PRO\s*6000)\b", upper):
        return 12.0
    if re.search(r"\b(H100|H200|GH200)\b", upper):
        return 9.0
    if re.search(r"\b(L40S?|L4)\b", upper) or re.search(r"\bRTX\s*40\d{2}", upper):
        return 8.9
    if re.search(r"\bA100\b", upper):
        return 8.0
    if (
        re.search(r"\b(A10G?|A16|A30|A40)\b", upper)
        or re.search(r"\bRTX\s*(30\d{2}|A[2465]000|A6000)\b", upper)
    ):
        return 8.6
    if re.search(r"\b(T4|RTX\s*20\d{2}|QUADRO\s*RTX)\b", upper):
        return 7.5
    if re.search(r"\bV100\b", upper):
        return 7.0
    if re.search(r"\b(P40|P100|P4|GTX\s*10\d{2}|TITAN\s*X|QUADRO\s*P)\b", upper):
        return 6.0
    if re.search(r"\b(K80|K40|M40)\b", upper):
        return 3.7
    if re.search(r"\b(M60|M6)\b", upper):
        return 5.2
    return None


def _infer_architecture(name: Optional[str], compute_capability: Optional[float]) -> Optional[str]:
    if compute_capability is None:
        return None
    if compute_capability >= 12.0:
        return "blackwell"
    if compute_capability >= 9.0:
        return "hopper"
    if compute_capability >= 8.9:
        return "ada"
    if compute_capability >= 8.0:
        return "ampere"
    if compute_capability >= 7.5:
        return "turing"
    if compute_capability >= 7.0:
        return "volta"
    if compute_capability >= 6.0:
        return "pascal"
    return "legacy" if name else None


def _infer_gpu_features(compute_capability: Optional[float]) -> Dict[str, bool]:
    if compute_capability is None:
        return {}
    return {
        "tensor_cores": compute_capability >= 7.0,
        "bf16": compute_capability >= 8.0,
        "fp8": compute_capability >= 8.9,
    }


def _query_nvidia_smi_gpu_profiles() -> Dict[int, Dict[str, Any]]:
    query = "index,name,memory.total,driver_version,compute_cap"
    output = _run_nvidia_smi(
        [f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        timeout=5,
    )
    has_compute_capability = True
    if not output:
        query = "index,name,memory.total,driver_version"
        output = _run_nvidia_smi(
            [f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            timeout=5,
        )
        has_compute_capability = False
    if not output:
        return {}

    cuda_version = _get_nvidia_smi_cuda_version()
    profiles: Dict[int, Dict[str, Any]] = {}
    for row in csv.reader(output.splitlines()):
        if len(row) < 4:
            continue
        try:
            index = int(str(row[0]).strip())
        except ValueError:
            continue
        name = str(row[1]).strip()
        vram = int(float(str(row[2]).strip()))
        driver_version = str(row[3]).strip() or None
        compute_capability = (
            _parse_float(row[4]) if has_compute_capability and len(row) >= 5 else None
        ) or _infer_compute_capability_from_name(name)
        profiles[index] = {
            "name": name,
            "vram": vram,
            "driver_version": driver_version,
            "cuda_version": cuda_version,
            "compute_capability": compute_capability,
            "architecture": _infer_architecture(name, compute_capability),
            "features": _infer_gpu_features(compute_capability),
        }
    return profiles


def get_hardware_profile():
    """Get hardware profile of the system."""
    # Get CPU information
    cpu_info = cpuinfo.get_cpu_info()
    cpu = {
        "brand": cpu_info["brand_raw"],
        "cores": cpu_info["count"],
    }

    # Get GPU information
    smi_profiles = _query_nvidia_smi_gpu_profiles()
    gpus = GPUtil.getGPUs()
    gpu_list = []
    if gpus:
        for index, gpu in enumerate(gpus):
            smi_profile = smi_profiles.get(index, {})
            name = smi_profile.get("name") or gpu.name
            compute_capability = (
                smi_profile.get("compute_capability")
                or _infer_compute_capability_from_name(name)
            )
            gpu_list.append({
                **smi_profile,
                "name": name,
                "vram": int(smi_profile.get("vram") or gpu.memoryTotal),
                "compute_capability": compute_capability,
                "architecture": smi_profile.get("architecture")
                or _infer_architecture(name, compute_capability),
                "features": smi_profile.get("features")
                or _infer_gpu_features(compute_capability),
            })
    else:
        for index in sorted(smi_profiles):
            gpu_list.append(smi_profiles[index])

    cuda_version = None
    driver_version = None
    if gpu_list:
        cuda_version = gpu_list[0].get("cuda_version")
        driver_version = gpu_list[0].get("driver_version")

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
        "cuda": {
            "version": cuda_version,
            "driver_version": driver_version,
        },
        "ram": ram,
        "disk": disk,
    }



def get_available_vram(hardware_profile: Optional[dict] = None) -> int:
    """Get the total VRAM of the primary GPU in MB. Returns 0 if no GPU found."""
    if hardware_profile:
        gpus = hardware_profile.get("gpus") or []
        if gpus:
            return int(gpus[0].get("vram") or gpus[0].get("memoryTotal") or 0)
    gpus = GPUtil.getGPUs()
    if gpus:
        return int(gpus[0].memoryTotal)
    return 0


def get_primary_gpu_profile(hardware_profile: Optional[dict] = None) -> dict:
    if hardware_profile:
        gpus = hardware_profile.get("gpus") or []
        if gpus:
            return gpus[0] or {}
    profile = get_hardware_profile()
    gpus = profile.get("gpus") or []
    return gpus[0] if gpus else {}


def get_compute_capability(hardware_profile: Optional[dict] = None) -> Optional[float]:
    gpu = get_primary_gpu_profile(hardware_profile)
    return _parse_float(
        gpu.get("compute_capability")
        or gpu.get("computeCapability")
        or gpu.get("cuda_compute_capability")
    ) or _infer_compute_capability_from_name(gpu.get("name"))


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


def _default_min_compute_capability(service_config: dict) -> Optional[float]:
    if service_config.get("min_compute_capability") is not None:
        return _parse_float(service_config.get("min_compute_capability"))
    category = service_config.get("category")
    if category in ("video", "image"):
        return 7.5
    if category in ("audio", "utils"):
        return 6.0
    return None


def _service_gpu_feature_requirements(service_config: dict) -> List[str]:
    features = service_config.get("required_gpu_features") or []
    return [str(feature) for feature in features if feature]


def can_run_service(
    service_config: dict,
    available_vram_mb: Optional[int] = None,
    hardware_profile: Optional[dict] = None,
) -> bool:
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
        available_vram_mb = get_available_vram(hardware_profile)
    
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

    primary_gpu = get_primary_gpu_profile(hardware_profile)
    compute_capability = get_compute_capability(hardware_profile)
    min_compute = _default_min_compute_capability(service_config)
    if min_compute is not None and compute_capability is not None:
        if compute_capability < min_compute:
            return False
    elif service_config.get("min_compute_capability") is not None:
        return False

    cuda_version = _parse_float(
        primary_gpu.get("cuda_version")
        or (hardware_profile or {}).get("cuda", {}).get("version")
    )
    min_cuda_version = _parse_float(service_config.get("min_cuda_version"))
    if min_cuda_version is not None:
        if cuda_version is None or cuda_version < min_cuda_version:
            return False

    features = primary_gpu.get("features") or _infer_gpu_features(compute_capability)
    for feature in _service_gpu_feature_requirements(service_config):
        if not features.get(feature):
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




def get_service_incompatibility_reason(
    service_config: dict,
    available_vram_mb: Optional[int] = None,
    hardware_profile: Optional[dict] = None,
) -> Optional[str]:
    """
    Returns a human-readable reason why a service can't run, or None if it can run.
    
    Args:
        service_config: Service configuration dict
        available_vram_mb: Optional pre-fetched VRAM in MB
    
    Returns:
        String describing the incompatibility, or None if compatible
    """
    if available_vram_mb is None:
        available_vram_mb = get_available_vram(hardware_profile)
    
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

    primary_gpu = get_primary_gpu_profile(hardware_profile)
    compute_capability = get_compute_capability(hardware_profile)
    min_compute = _default_min_compute_capability(service_config)
    if min_compute is not None and compute_capability is not None:
        if compute_capability < min_compute:
            return (
                f"requires CUDA compute capability {min_compute}+, "
                f"have {compute_capability}"
            )
    elif service_config.get("min_compute_capability") is not None:
        return f"requires CUDA compute capability {min_compute}+ but GPU capability is unknown"

    cuda_version = _parse_float(
        primary_gpu.get("cuda_version")
        or (hardware_profile or {}).get("cuda", {}).get("version")
    )
    min_cuda_version = _parse_float(service_config.get("min_cuda_version"))
    if min_cuda_version is not None:
        if cuda_version is None:
            return f"requires CUDA {min_cuda_version}+ but CUDA driver capability is unknown"
        if cuda_version < min_cuda_version:
            return f"requires CUDA {min_cuda_version}+, have {cuda_version}"

    features = primary_gpu.get("features") or _infer_gpu_features(compute_capability)
    for feature in _service_gpu_feature_requirements(service_config):
        if not features.get(feature):
            return f"requires GPU feature '{feature}'"

    
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
    hardware_profile = get_hardware_profile()
    available_vram = get_available_vram(hardware_profile)
    compatible = []
    incompatible = []
    
    for service_name, config in services_config.items():
        if can_run_service(config, available_vram, hardware_profile):
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

