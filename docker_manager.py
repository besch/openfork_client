
import docker
import os
from config import DOCKER_IMAGE_NAME, DOCKERFILE_PATH, COMFYUI_PORT, ROOT_DIR

def build_image():
    """Builds the Docker image."""
    client = docker.from_env()
    print("Building Docker image...")
    try:
        client.images.build(
            path=os.path.dirname(DOCKERFILE_PATH),
            dockerfile=os.path.basename(DOCKERFILE_PATH),
            tag=DOCKER_IMAGE_NAME,
            rm=True
        )
        print("Docker image built successfully.")
    except docker.errors.BuildError as e:
        print(f"Error building Docker image: {e}")
        raise

def run_container():
    """Runs the Docker container."""
    client = docker.from_env()
    print("Running Docker container...")
    try:
        container = client.containers.run(
            DOCKER_IMAGE_NAME,
            detach=True,
            ports={f"{COMFYUI_PORT}/tcp": COMFYUI_PORT},
            volumes={os.path.join(ROOT_DIR, 'output'): {'bind': '/opt/ComfyUI/output', 'mode': 'rw'}},
            device_requests=[
                docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
            ]
        )
        print(f"Docker container started with ID: {container.id}")
        return container
    except docker.errors.ContainerError as e:
        print(f"Error running Docker container: {e}")
        raise
