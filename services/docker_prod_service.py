import logging
import subprocess
import os
import sys

from config import Config

def get_root_dir_from_args():
    if '--root-dir' in sys.argv:
        try:
            index = sys.argv.index('--root-dir')
            return sys.argv[index + 1]
        except (ValueError, IndexError):
            return None
    return None

class DockerProdManager:
    def __init__(self):
        self.config = Config
        
        root_dir = get_root_dir_from_args()
        if not root_dir:
            if getattr(sys, 'frozen', False):
                # Fallback for frozen apps if --root-dir is not provided
                root_dir = os.path.dirname(sys.executable)
            else:
                # Fallback for script-based execution
                root_dir = self.config.ROOT_DIR

        self.compose_file_path = os.path.join(root_dir, 'docker', 'docker-compose.unified.yaml')
        self.service_name = 'comfyui'

    def _run_command(self, command, env=None):
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
                cwd=compose_dir,
                env=env
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
        
        env_updates = {}
        if dependencies:
            if dependencies.get('custom_node_urls'):
                urls = ' '.join(dependencies['custom_node_urls'])
                env_updates['CUSTOM_NODES_GIT_URLS'] = urls
            if dependencies.get('model_urls'):
                urls = ' '.join(dependencies['model_urls'])
                env_updates['MODEL_URLS'] = urls
        
        process_env = os.environ.copy()
        process_env.update(env_updates)

        command = [
            'docker-compose',
            '-f', self.compose_file_path,
            'up',
            '--build',
            '-d'
        ]
        self._run_command(command, env=process_env)
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