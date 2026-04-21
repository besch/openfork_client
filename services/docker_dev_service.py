'''
This module contains the development Docker service, which uses docker-compose
to manage local containers for testing and iteration.
'''
import logging
import subprocess
import os
import json
import threading
from config import ROOT_DIR
from .docker_utils import docker_cp, get_subprocess_hidden_kwargs

class DockerDevManager:
    def __init__(self):
        self.compose_dir = os.path.join(ROOT_DIR, 'comfyui-storage')
        self.compose_cmd = self._get_compose_command()
        self._load_service_config()

    def _get_compose_command(self) -> list:
        """Determines whether to use 'docker compose' or 'docker-compose'."""
        # Try 'docker compose' (v2) first
        try:
            subprocess.run(['docker', 'compose', 'version'], capture_output=True, check=True, **get_subprocess_hidden_kwargs())
            return ['docker', 'compose']
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to 'docker-compose' (v1)
            try:
                subprocess.run(['docker-compose', 'version'], capture_output=True, check=True, **get_subprocess_hidden_kwargs())
                return ['docker-compose']
            except (subprocess.CalledProcessError, FileNotFoundError):
                logging.warning("Neither 'docker compose' nor 'docker-compose' found. Dev mode may fail.")
                return ['docker', 'compose'] # Default to v2

    def _load_service_config(self):
        """Loads the service config from the master services.json file."""
        try:
            # Path from dgn-client/services -> dgn-client -> root
            config_path = os.path.join(ROOT_DIR, '..', 'services.json')
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            self.service_config = {s['service_type']: s for s in data['services']}
            logging.info("Successfully loaded services.json for dev manager.")
        except (IOError, json.JSONDecodeError) as e:
            logging.error(f"Failed to load or parse services.json: {e}")
            self.service_config = {}

    def set_docker_image_map(self, image_map: dict):
        # This method is a no-op for the dev manager as it uses docker-compose files,
        # but it's here to maintain a consistent interface with the prod manager.
        logging.debug("set_docker_image_map called on dev manager (no-op).")
        pass

    def pull_image(self, image_name: str, shutdown_event: threading.Event = None, service_type: str = None):
        """No-op for dev manager as it builds images locally via docker-compose."""
        logging.info(f"Dev mode pull_image called for {image_name} (no-op).")
        pass

    def _get_compose_file(self, service_type: str) -> str:
        """Gets the docker-compose file path from the loaded service config."""
        service_info = self.service_config.get(service_type)
        if not service_info or not service_info.get('dev_compose_file'):
            raise ValueError(f"Service '{service_type}' or its dev_compose_file not found in services.json")

        compose_file = os.path.join(self.compose_dir, service_info['dev_compose_file'])
        
        if not os.path.exists(compose_file):
            raise FileNotFoundError(f"Docker compose file '{service_info['dev_compose_file']}' not found for service '{service_type}' at {compose_file}")
        return compose_file

    def _run_command(self, command: list):
        try:
            # Using cwd ensures docker-compose can find relative paths in the .yaml files
            subprocess.run(command, check=True, capture_output=True, text=True, cwd=self.compose_dir, **get_subprocess_hidden_kwargs())
            logging.info(f"Successfully ran command: {' '.join(command)}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running command: {' '.join(command)}")
            logging.error(f"Stderr: {e.stderr}")
            logging.error(f"Stdout: {e.stdout}")
            raise

    def get_container_id(self, service_type: str) -> str:
        compose_file = self._get_compose_file(service_type)
        command = self.compose_cmd + ['-f', compose_file, 'ps', '-q']
        try:
            # Run command to get container ID
            result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=self.compose_dir, **get_subprocess_hidden_kwargs())
            container_id = result.stdout.strip()
            if not container_id:
                raise RuntimeError(f"Could not determine container ID for service '{service_type}'. Is it running?")
            logging.info(f"Found container ID for service '{service_type}': {container_id}")
            return container_id
        except subprocess.CalledProcessError as e:
            logging.error(f"Error getting container ID for service '{service_type}': {e.stderr}")
            raise

    def copy_file_from_container(self, service_type: str, source_in_container: str, dest_on_host: str, shutdown_event: threading.Event):
        container_id = self.get_container_id(service_type)
        source_path = f"{container_id}:{source_in_container}"
        docker_cp(source_path, dest_on_host, shutdown_event)

    def exec_in_container(
        self,
        service_type: str,
        command: list,
        detach: bool = False,
        environment: dict = None,
    ):
        """Run a command inside a running container (dev mode via docker exec)."""
        container_id = self.get_container_id(service_type)
        cmd = ["docker", "exec"]
        if detach:
            cmd.append("-d")
        if environment:
            for k, v in environment.items():
                cmd += ["-e", f"{k}={v}"]
        cmd.append(container_id)
        cmd.extend(command)
        logging.info(f"exec in container: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                **get_subprocess_hidden_kwargs(),
            )
            return result
        except Exception as e:
            logging.error(f"exec_in_container failed for '{service_type}': {e}")
            return None

    def run_container(
        self,
        service_type: str,
        ports: dict = None,
        force_restart: bool = True,
        command: list = None,
        environment: dict = None,
    ):
        compose_file = self._get_compose_file(service_type)
        logging.info(f"Starting container for service '{service_type}' using compose file: {compose_file}")
        # The command is structured to use a specific compose file and bring the service up
        # docker-compose up is idempotent, so force_restart logic is less critical here,
        # but we could add --force-recreate if force_restart is True.
        cmd = self.compose_cmd + ['-f', compose_file, 'up', '--build', '-d']
        if force_restart:
            cmd.append('--force-recreate')

        self._run_command(cmd)

    def stop_container(self, service_type: str):
        compose_file = self._get_compose_file(service_type)
        logging.info(f"Stopping container for service '{service_type}' using compose file: {compose_file}")
        # The 'down' command stops and removes containers, networks, etc.
        command = self.compose_cmd + ['-f', compose_file, 'down']
        self._run_command(command)

    def copy_file_to_container(self, service_type: str, source_on_host: str, dest_in_container: str, shutdown_event: threading.Event):
        if shutdown_event.is_set():
            logging.warning(f"Shutdown event set. Aborting docker cp operation for {source_on_host} to container {service_type}.")
            raise RuntimeError("Docker cp aborted due to shutdown event.")

        container_id = self.get_container_id(service_type)
        dest_path = f"{container_id}:{dest_in_container}"
        command = ['docker', 'cp', source_on_host, dest_path]
        logging.info(f"Copying file to container: {' '.join(command)}")
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=300, **get_subprocess_hidden_kwargs())
            logging.info(f"Successfully ran command: {' '.join(command)}")
        except subprocess.TimeoutExpired:
            logging.error(f"Timeout during docker cp operation for {source_on_host} to container {service_type}")
            raise
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running command: {' '.join(command)}")
            logging.error(f"Stderr: {e.stderr}")
            logging.error(f"Stdout: {e.stdout}")
            raise

    def stream_logs(self, service_type: str, shutdown_event: threading.Event):
        """Streams logs from the container to the application logger."""
        try:
            container_id = self.get_container_id(service_type)
            # Use docker logs -f to follow
            command = ['docker', 'logs', '-f', container_id]
            logging.info(f"Starting log stream for service '{service_type}' (ID: {container_id})")

            # subprocess.Popen to read stdout in real-time
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1  # Line buffered
            )

            # Read lines in a loop
            while not shutdown_event.is_set() and process.poll() is None:
                # We use readline() but we need to handle blocking. 
                # Since we are in a thread, blocking is okay-ish, but if shutdown_event is set
                # we want to exit. However, readline() manages its own blocking.
                # A common valid approach for "interruptible readline" is using selectors or a timeout,
                # but standard readline is simplest. We'll simply check shutdown after each line.
                try:
                    line = process.stdout.readline()
                    if not line:
                        break
                    logging.info(f"[{service_type}] {line.strip()}")
                except Exception:
                    break
            
            # Clean up the subprocess
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

        except Exception as e:
            # It's possible get_container_id fails if container isn't ready yet
            logging.debug(f"Log streaming failed/interrupted for '{service_type}': {e}")
