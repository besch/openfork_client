'''
This module contains the development Docker service, which uses docker-compose
to manage local containers for testing and iteration.
'''
import logging
import subprocess
import os
from config import ROOT_DIR

class DockerDevManager:
    def __init__(self):
        self.compose_dir = os.path.join(ROOT_DIR, 'comfyui-storage')

    def _get_compose_file(self, service_type: str) -> str:
        # Services use a specific compose file, e.g., docker-compose.foley.yaml
        compose_file = os.path.join(self.compose_dir, f'docker-compose.{service_type}.yaml')
        if not os.path.exists(compose_file):
            raise FileNotFoundError(f"Docker compose file not found for service '{service_type}' at {compose_file}")
        return compose_file

    def _run_command(self, command: list):
        try:
            # Using cwd ensures docker-compose can find relative paths in the .yaml files
            subprocess.run(command, check=True, capture_output=True, text=True, cwd=self.compose_dir)
            logging.info(f"Successfully ran command: {' '.join(command)}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running command: {' '.join(command)}")
            logging.error(f"Stderr: {e.stderr}")
            logging.error(f"Stdout: {e.stdout}")
            raise

    def get_container_id(self, service_type: str) -> str:
        compose_file = self._get_compose_file(service_type)
        command = ['docker-compose', '-f', compose_file, 'ps', '-q']
        try:
            # Run command to get container ID
            result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=self.compose_dir)
            container_id = result.stdout.strip()
            if not container_id:
                raise RuntimeError(f"Could not determine container ID for service '{service_type}'. Is it running?")
            logging.info(f"Found container ID for service '{service_type}': {container_id}")
            return container_id
        except subprocess.CalledProcessError as e:
            logging.error(f"Error getting container ID for service '{service_type}': {e.stderr}")
            raise

    def copy_file_from_container(self, service_type: str, source_in_container: str, dest_on_host: str):
        container_id = self.get_container_id(service_type)
        source_path = f"{container_id}:{source_in_container}"
        command = ['docker', 'cp', source_path, dest_on_host]
        logging.info(f"Copying file from container: {' '.join(command)}")
        # We don't use self._run_command here because the cwd for docker-compose is not needed for docker cp.
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            logging.info(f"Successfully ran command: {' '.join(command)}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running command: {' '.join(command)}")
            logging.error(f"Stderr: {e.stderr}")
            logging.error(f"Stdout: {e.stdout}")
            raise

    def run_container(self, service_type: str):
        compose_file = self._get_compose_file(service_type)
        logging.info(f"Starting container for service '{service_type}' using compose file: {compose_file}")
        # The command is structured to use a specific compose file and bring the service up
        command = ['docker-compose', '-f', compose_file, 'up', '--build', '-d']
        self._run_command(command)

    def stop_container(self, service_type: str):
        compose_file = self._get_compose_file(service_type)
        logging.info(f"Stopping container for service '{service_type}' using compose file: {compose_file}")
        # The 'down' command stops and removes containers, networks, etc.
        command = ['docker-compose', '-f', compose_file, 'down']
        self._run_command(command)
