#!/usr/bin/env python3
"""
GPU Compatibility Checker for OpenFork DGN
Checks if the GPU supports the PyTorch/CUDA versions used in containers
"""

import sys

try:
    import torch
except ImportError:
    print("ERROR: PyTorch not installed. Install with: pip install torch")
    sys.exit(1)

print("=" * 80)
print("GPU COMPATIBILITY CHECK FOR OPENFORK DGN")
print("=" * 80)
print()

# Check CUDA availability
if not torch.cuda.is_available():
    print("❌ CUDA is NOT available")
    print("   Your system cannot run GPU-accelerated DGN jobs")
    print()
    print("   Possible causes:")
    print("   - No NVIDIA GPU installed")
    print("   - NVIDIA driver not installed")
    print("   - CUDA toolkit not installed")
    sys.exit(1)

print("✓ CUDA is available")
print()

# GPU Information
gpu_count = torch.cuda.device_count()
print(f"Number of GPUs: {gpu_count}")
print()

for i in range(gpu_count):
    print(f"GPU {i}:")
    print(f"  Name: {torch.cuda.get_device_name(i)}")
    
    # Get compute capability
    major, minor = torch.cuda.get_device_capability(i)
    compute_cap = f"{major}.{minor}"
    print(f"  Compute Capability: {compute_cap}")
    
    # Memory
    total_mem_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"  Total Memory: {total_mem_gb:.1f} GB")
    
    # Check compatibility
    print()
    print("  COMPATIBILITY WITH OPENFORK CONTAINERS:")
    print("  " + "-" * 60)
    
    # Requirements by container
    containers = {
        "HeartMuLa (Dockerfile.heartmula)": {
            "pytorch": "2.4.0", 
            "cuda": "12.4",
            "min_compute": 5.0,
            "recommended_compute": 6.0,
            "min_vram_gb": 12
        },
        "ComfyUI/LTX-Video": {
            "pytorch": "2.x",
            "cuda": "11.8+",
            "min_compute": 5.0,
            "recommended_compute": 6.0,
            "min_vram_gb": 8
        }
    }
    
    for container_name, reqs in containers.items():
        print(f"  {container_name}:")
        print(f"    Requires: PyTorch {reqs['pytorch']}, CUDA {reqs['cuda']}")
        print(f"    Min Compute Capability: {reqs['min_compute']}")
        
        # Check compute capability
        if float(compute_cap) < reqs['min_compute']:
            print(f"    ❌ INCOMPATIBLE - GPU is too old (compute {compute_cap} < {reqs['min_compute']})")
            print(f"       Your GPU architecture is not supported by PyTorch {reqs['pytorch']} + CUDA {reqs['cuda']}")
            print(f"       You will get: 'no kernel image is available for execution'")
            print()
            print(f"       SOLUTIONS:")
            print(f"       1. Use a newer GPU (GTX 900 series / Compute {reqs['min_compute']}+ or newer)")
            print(f"       2. Rebuild container with PyTorch 1.x + CUDA 11.x (supports compute 3.5+)")
        elif float(compute_cap) < reqs['recommended_compute']:
            print(f"    ⚠️  MINIMAL SUPPORT - Works but not optimal (compute {compute_cap})")
            print(f"       Some optimizations may not be available")
            print(f"       Consider GPU with compute capability {reqs['recommended_compute']}+ for best performance")
        else:
            print(f"    ✓ FULLY COMPATIBLE (compute {compute_cap} >= {reqs['recommended_compute']})")
        
        # Check VRAM
        if total_mem_gb < reqs['min_vram_gb']:
            print(f"    ⚠️  LOW VRAM - {total_mem_gb:.1f}GB < {reqs['min_vram_gb']}GB required")
            print(f"       May need quantization or will run out of memory")
        else:
            print(f"    ✓ Sufficient VRAM ({total_mem_gb:.1f}GB)")
        
        print()

print()
print("=" * 80)
print("PYTORCH ENVIRONMENT")
print("=" * 80)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print()

# Architecture mapping
print("=" * 80)
print("COMMON GPU ARCHITECTURES")
print("=" * 80)
print("Compute 3.0 - 3.7 (Kepler):    GTX 600/700 series - NOT SUPPORTED by PyTorch 2.x + CUDA 12.x")
print("Compute 5.0 - 5.2 (Maxwell):   GTX 900 series - MINIMAL SUPPORT (works but limited)")
print("Compute 6.0 - 6.2 (Pascal):    GTX 10xx series - FULLY SUPPORTED")
print("Compute 7.0 - 7.5 (Volta/Turing): RTX 20xx, GTX 16xx - FULLY SUPPORTED")
print("Compute 8.0 - 8.9 (Ampere):    RTX 30xx series - FULLY SUPPORTED")
print("Compute 9.0 (Hopper):          RTX 40xx, H100 - FULLY SUPPORTED")
print("=" * 80)
