'''
This module contains the production Docker service, which manages pre-built
images from a Docker registry (e.g., Docker Hub).
'''
import docker
import logging
import subprocess
from config import DOCKER_IMAGE_MAP

class DockerProdManager:
    def __init__(self):
        try:
            self.client = docker.from_env()
        except docker.errors.DockerException:
            logging.error("Docker is not running. Please start Docker Desktop.")
            raise

    def get_image_name(self, service_type: str) -> str:
        image = DOCKER_IMAGE_MAP.get(service_type)
        if not image:
            raise ValueError(f"No Docker image configured for service type '{service_type}'")
        return image

    def get_container_name(self, service_type: str) -> str:
        return f"dgn-client-comfyui-{service_type}"

    def pull_image(self, image_name: str):
        try:
            logging.info(f"Checking for Docker image: {image_name}...")
            self.client.images.get(image_name)
            logging.info(f"Image '{image_name}' found locally.")
        except docker.errors.ImageNotFound:
            logging.info(f"Image '{image_name}' not found locally. Pulling from Docker Hub...")
            try:
                self.client.images.pull(image_name)
                logging.info(f"Successfully pulled image: {image_name}")
            except docker.errors.APIError as e:
                logging.error(f"Failed to pull image '{image_name}': {e}")
                raise

    def run_container(self, service_type: str):
        image_name = self.get_image_name(service_type)
        container_name = self.get_container_name(service_type)

        try:
            container = self.client.containers.get(container_name)
            if container.status == 'running':
                logging.info(f"Container '{container_name}' is already running.")
                return
            else:
                logging.info(f"Found a stopped container '{container_name}'. Removing it before starting a new one.")
                container.remove(force=True)
        except docker.errors.NotFound:
            pass

        self.pull_image(image_name)

        logging.info(f"Starting container '{container_name}' from image '{image_name}'...")
        try:
            self.client.containers.run(
                image=image_name,
                detach=True,
                name=container_name,
                ports={'8188/tcp': 8188},
                device_requests=[
                    docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
                ],
                restart_policy={"Name": "no"}
            )
            logging.info(f"Container '{container_name}' started successfully.")
        except docker.errors.APIError as e:
            logging.error(f"Failed to start container '{container_name}': {e}")
            raise

    def copy_file_from_container(self, service_type: str, source_in_container: str, dest_on_host: str):
        container_name = self.get_container_name(service_type)
        source_path = f"{container_name}:{source_in_container}"
        command = ['docker', 'cp', source_path, dest_on_host]
        logging.info(f"Copying file from container: {' '.join(command)}")
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            logging.info(f"Successfully ran command: {' '.join(command)}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running command: {' '.join(command)}")
            logging.error(f"Stderr: {e.stderr}")
            logging.error(f"Stdout: {e.stdout}")
            raise

    def stop_container(self, service_type: str):
        container_name = self.get_container_name(service_type)
        try:
            container = self.client.containers.get(container_name)
            logging.info(f"Stopping container '{container_name}'...")
            container.stop()
            container.remove()
            logging.info(f"Container '{container_name}' stopped and removed.")
        except docker.errors.NotFound:
            logging.warning(f"Attempted to stop container '{container_name}', but it was not found.")
        except docker.errors.APIError as e:
            logging.error(f"Failed to stop or remove container '{container_name}': {e}")
