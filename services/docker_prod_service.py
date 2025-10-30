import logging
import subprocess
import os
import tempfile
import sys

from config import Config

class DockerProdManager:
    def __init__(self):
        self.config = Config
        if getattr(sys, 'frozen', False):
            # When frozen, the docker-compose file is likely relative to the executable
            self.compose_file_path = os.path.join(os.path.dirname(sys.executable), 'docker', 'docker-compose.unified.yaml')
        else:
            # When running as a script, it's relative to Config.ROOT_DIR
            self.compose_file_path = os.path.join(Config.ROOT_DIR, 'docker', 'docker-compose.unified.yaml')
        self.service_name = 'comfyui'

    def _run_command(self, command):
        compose_dir = os.path.dirname(self.compose_file_path)
        if not os.path.isdir(compose_dir):
            logging.error(f"Docker compose directory not found: {compose_dir}")
            raise NotADirectoryError(f"Docker compose directory not found: {compose_dir}")

        logging.info(f"Running command: {' '.join(command)}")
        try:
            process = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=compose_dir
            )
            logging.info(process.stdout)
            if process.stderr:
                logging.warning(process.stderr)
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed: {' '.join(command)}")
            logging.error(f"Stderr: {e.stderr}")
            logging.error(f"Stdout: {e.stdout}")
            raise

    def run_container(self, dependencies: dict = None):
        logging.info("Starting unified ComfyUI container...")
        
        # Create a temporary env file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as temp_env_file:
            env_file_path = temp_env_file.name
            if dependencies and dependencies.get('custom_node_urls'):
                urls = ' '.join(dependencies['custom_node_urls'])
                temp_env_file.write(f'CUSTOM_NODES_GIT_URLS="{urls}"\n')
            if dependencies and dependencies.get('model_urls'):
                urls = ' '.join(dependencies['model_urls'])
                temp_env_file.write(f'MODEL_URLS="{urls}"\n')
        
        try:
            command = [
                'docker-compose',
                '-f', self.compose_file_path,
                'up',
                '--build',
                '-d',
                '--env-file', env_file_path
            ]
            self._run_command(command)
            logging.info(f"Container for service '{self.service_name}' started successfully.")
        finally:
            # Clean up the temporary env file
            os.unlink(env_file_path)

    def stop_container(self, service_type: str = None):
        logging.info(f"Stopping unified ComfyUI container...")
        command = [
            'docker-compose',
            '-f', self.compose_file_path,
            'down'
        ]
        self._run_command(command)
        logging.info(f"Container for service '{self.service_name}' stopped successfully.")

    def copy_file_from_container(self, source_in_container: str, dest_on_host: str):
        container_name = f"openfork__{self.service_name}_1" # Default name from docker-compose
        source_path = f"{container_name}:{source_in_container}"
        command = ['docker', 'cp', source_path, dest_on_host]
        self._run_command(command)

    def copy_file_to_container(self, source_on_host: str, dest_in_container: str):
        container_name = f"openfork__{self.service_name}_1" # Default name from docker-compose
        dest_path = f"{container_name}:{dest_in_container}"
        command = ['docker', 'cp', source_on_host, dest_path]
        self._run_command(command)