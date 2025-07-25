
import time
from docker_manager import build_image, run_container
from comfyui_manager import trigger_workflow

def main():
    """Main function to run the DGN client."""
    build_image()
    container = run_container()

    # Wait for ComfyUI to start
    time.sleep(10)

    trigger_workflow()

    print("Processing complete. Stopping container...")
    container.stop()
    print("Container stopped.")

if __name__ == "__main__":
    main()
