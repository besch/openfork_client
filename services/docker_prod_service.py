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
            
            # Disk space check before download
            from .disk_space_utils import check_sufficient_space, estimate_image_size_bytes
            import json
            
            estimated_size = estimate_image_size_bytes(image_name)
            has_space, available, required = check_sufficient_space(estimated_size)
            
            if not has_space:
                available_gb = available / (1024**3)
                required_gb = required / (1024**3)
                error_msg = (
                    f"Insufficient disk space to download '{image_name}'. "
                    f"Required: {required_gb:.1f} GB (including 5 GB safety buffer), "
                    f"Available: {available_gb:.1f} GB"
                )
                logging.error(error_msg)
                
                # Emit event for desktop UI notification
                print(json.dumps({
                    "type": "DISK_SPACE_ERROR",
                    "payload": {
                        "image_name": image_name,
                        "required_gb": round(required_gb, 1),
                        "available_gb": round(available_gb, 1),
                        "message": error_msg
                    }
                }), flush=True)
                
                raise OSError(error_msg)
            
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

    def stream_logs(self, service_type: str, shutdown_event: threading.Event):
        """Streams logs from the container to the application logger."""
        container_name = self.get_container_name(service_type)
        logging.info(f"Starting log stream for container '{container_name}'")
        
        try:
            container = self.client.containers.get(container_name)
            # stream=True returns a generator processing the logs
            for line in container.logs(stream=True, follow=True):
                if shutdown_event.is_set():
                    break
                if line:
                    decoded_line = line.decode('utf-8', errors='replace').strip()
                    if decoded_line:
                        logging.info(f"[{service_type}] {decoded_line}")
        except docker.errors.NotFound:
            logging.warning(f"Container '{container_name}' not found for logging.")
        except Exception as e:
            # If the container stops, we might get an APIError or similar, which is expected
            if not shutdown_event.is_set():
                logging.debug(f"Log streaming interrupted for '{container_name}': {e}")
