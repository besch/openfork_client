import sys
import os

# Add client directory to path
sys.path.append(os.path.abspath("d:/openfork/client"))

def check_file(path):
    if os.path.exists(path):
        print(f"✅ FOUND: {path}")
    else:
        print(f"❌ MISSING: {path}")

print("--- FILE CHECKS ---")
check_file("d:/openfork/client/comfyui-storage/Dockerfile.qwen-turbo-8gb")
check_file("d:/openfork/client/workflows/qwen-image-t2i-8gb-turbo_api.json")
check_file("d:/openfork/client/services/processors/image/qwen_turbo.py")

print("\n--- IMPORT CHECKS ---")
try:
    from services.job_processors import (
        QwenImageEditTurboProcessor,
        QwenImageInpaintTurboProcessor,
        QwenImageT2ITurboProcessor
    )
    print("✅ Successfully imported Turbo Processors from job_processors")
except ImportError as e:
    print(f"❌ Failed to import processors: {e}")

try:
    from services.processors import (
        QwenImageEditTurboProcessor,
        QwenImageInpaintTurboProcessor,
        QwenImageT2ITurboProcessor
    )
    print("✅ Successfully imported Turbo Processors from processors package")
except ImportError as e:
    print(f"❌ Failed to import processors from package: {e}")
