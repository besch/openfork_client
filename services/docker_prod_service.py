import logging
import subprocess
import os
from config import Config

class DockerProdManager:
    def __init__(self):
        self.config = Config
        self.compose_file_path = os.path.join(Config.ROOT_DIR, 'docker', 'docker-compose.unified.yaml')
        self.service_name = 'comfyui'

    def _run_command(self, command):
        logging.info(f"Running command: {' '.join(command)}")
        try:
            process = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(self.compose_file_path)
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
        
        env_file_path = os.path.join(os.path.dirname(self.compose_file_path), 'dgn_job.env')
        with open(env_file_path, 'w') as f:
            if dependencies and dependencies.get('custom_node_urls'):
                urls = ' '.join(dependencies['custom_node_urls'])
                f.write(f'CUSTOM_NODES_GIT_URLS="{urls}"\n')
            if dependencies and dependencies.get('model_urls'):
                urls = ' '.join(dependencies['model_urls'])
                f.write(f'MODEL_URLS="{urls}"\n')

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