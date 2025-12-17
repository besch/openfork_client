'''
This module contains the production Docker service, which manages pre-built
images from a Docker registry (e.g., Docker Hub).
'''
import docker
import logging
import threading
from .docker_utils import docker_cp

class DockerProdManager:
    def __init__(self):
        try:
            self.client = docker.from_env()
            self.docker_image_map = {}
            self.services_config = {}
        except docker.errors.DockerException:
            logging.error("Docker is not running. Please start Docker Desktop.")
            raise

    def set_docker_image_map(self, image_map: dict):
        if image_map:
            logging.info("Setting dynamic Docker image map.")
            self.docker_image_map = image_map
        else:
            logging.warning("Dynamic Docker image map is empty. Using fallback static map.")

    def set_services_config(self, services_config: dict):
        if services_config:
            logging.info("Setting services configuration.")
            self.services_config = services_config

    def get_default_ports(self, service_type: str) -> dict:
        config = self.services_config.get(service_type, {})
        port = config.get("port", 8188)  # Default to ComfyUI port
        return {f'{port}/tcp': port}

    def get_image_name(self, service_type: str) -> str:
        image = self.docker_image_map.get(service_type)
        if not image:
            raise ValueError(f"No Docker image configured for service type '{service_type}'")
        return image

    def get_container_name(self, service_type: str) -> str:
        return f"dgn-client-comfyui-{service_type}"

    def pull_image(self, image_name: str):
        try:
            logging.info(f"Checking for Docker image: {image_name}...")
            self.client.images.get(image_name)
            logging.info(f"Image '{image_name}' found locally. Skipping pull.")
            return
        except docker.errors.ImageNotFound:
            logging.info(f"Image '{image_name}' not found locally. Pulling from Docker Hub...")
            try:
                from .docker_progress_logger import stream_pull_with_progress
                stream_pull_with_progress(self.client, image_name, throttle_interval=0.5)
                logging.info(f"Successfully pulled image: {image_name}")
            except docker.errors.APIError as e:
                logging.error(f"Failed to pull image '{image_name}': {e}")
                raise

    def run_container(self, service_type: str, ports: dict = None, force_restart: bool = True):
        image_name = self.get_image_name(service_type)
        container_name = self.get_container_name(service_type)

        # Get ports from configuration if not provided
        if ports is None:
            ports = self.get_default_ports(service_type)

        try:
            container = self.client.containers.get(container_name)
            if force_restart:
                logging.info(f"Found existing container '{container_name}' with status '{container.status}'. Removing it before starting a new one.")
                container.remove(force=True)
            else:
                if container.status == 'running':
                    logging.info(f"Container '{container_name}' is already running and force_restart is False. Skipping start.")
                    return
                else:
                    logging.info(f"Container '{container_name}' exists but is '{container.status}'. Removing and restarting.")
                    container.remove(force=True)

        except docker.errors.NotFound:
            logging.info(f"No existing container named '{container_name}' found. Proceeding to create a new one.")
            pass  # Container does not exist, which is fine.
        except docker.errors.APIError as e:
            logging.error(f"Error checking/removing existing container '{container_name}': {e}. Attempting to continue.")
            # Log the error but try to proceed. The run command will likely fail if removal did, but it's worth a try.

        self.pull_image(image_name)

        logging.info(f"Starting container '{container_name}' from image '{image_name}' with ports {ports}...")
        try:
            self.client.containers.run(
                image=image_name,
                detach=True,
                name=container_name,
                ports=ports,
                device_requests=[
                    docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
                ],
                restart_policy={"Name": "no"}
            )
            logging.info(f"Container '{container_name}' started successfully.")
        except docker.errors.APIError as e:
            logging.error(f"Failed to start container '{container_name}': {e}")
            raise

    def stop_container(self, service_type: str):
        container_name = self.get_container_name(service_type)
        logging.info(f"Attempting to stop and remove container '{container_name}'...")
        try:
            container = self.client.containers.get(container_name)
            logging.info(f"Container '{container_name}' found. Forcefully removing it.")
            container.remove(force=True)
            logging.info(f"Container '{container_name}' removed.")
        except docker.errors.NotFound:
            logging.info(f"Container '{container_name}' not found. Nothing to stop.")
        except docker.errors.APIError as e:
            logging.error(f"Failed to stop or remove container '{container_name}': {e}")

    def copy_file_from_container(self, service_type: str, source_in_container: str, dest_on_host: str, shutdown_event: threading.Event):
        container_name = self.get_container_name(service_type)
        source_path = f"{container_name}:{source_in_container}"
        docker_cp(source_path, dest_on_host, shutdown_event)

    def copy_file_to_container(self, service_type: str, source_on_host: str, dest_in_container: str, shutdown_event: threading.Event):
        container_name = self.get_container_name(service_type)
        dest_path = f"{container_name}:{dest_in_container}"
        docker_cp(source_on_host, dest_path, shutdown_event)
