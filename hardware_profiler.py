import cpuinfo
import GPUtil
import psutil

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
