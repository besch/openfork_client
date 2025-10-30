"""
This module contains the development Docker service.
It now mirrors the production service by using a single, unified docker-compose file.
"""
import logging
import subprocess
import os
import tempfile
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

class DockerDevManager:
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
        self.service_name = 'comfyui' # As defined in the unified compose file

    def _run_command(self, command):
        compose_dir = os.path.dirname(self.compose_file_path)
        if not os.path.isdir(compose_dir):
            logging.error(f"Docker compose directory not found: {compose_dir}")
            raise NotADirectoryError(f"Docker compose directory not found: {compose_dir}")

        logging.info(f"Running dev command: {' '.join(command)}")
        try:
            process = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=compose_dir # Run from the compose file's directory
            )
            logging.info(process.stdout)
            if process.stderr:
                logging.warning(process.stderr)
        except subprocess.CalledProcessError as e:
            logging.error(f"Dev command failed: {' '.join(command)}")
            logging.error(f"Stderr: {e.stderr}")
            logging.error(f"Stdout: {e.stdout}")
            raise

    def run_container(self, dependencies: dict = None):
        """
        Starts the unified ComfyUI container for development.
        Accepts dependencies to write to an environment file, mirroring production behavior.
        """
        logging.info("Starting unified ComfyUI container for development...")
        
        # Create a temporary .env file for the compose command to use
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
                '--build', # Rebuild if Dockerfile has changed
                '-d' # Detached mode
            ]
            self._run_command(command)
            logging.info(f"Dev container for service '{self.service_name}' started successfully.")
        finally:
            # Clean up the temporary env file
            os.unlink(env_file_path)

    def stop_container(self, service_type: str = None):
        """
        Stops the unified ComfyUI container. The service_type argument is ignored
        to maintain interface consistency with the old manager.
        """
        logging.info(f"Stopping unified ComfyUI container for development...")
        command = [
            'docker-compose',
            '-f', self.compose_file_path,
            'down'
        ]
        self._run_command(command)
        logging.info(f"Dev container for service '{self.service_name}' stopped successfully.")

    def copy_file_from_container(self, source_in_container: str, dest_on_host: str):
        """Copies a file from the unified container to the host."""
        # The container name is typically <project>_<service>_1
        container_name = f"openfork_{self.service_name}_1"
        source_path = f"{container_name}:{source_in_container}"
        command = ['docker', 'cp', source_path, dest_on_host]
        self._run_command(command)

    def copy_file_to_container(self, source_on_host: str, dest_in_container: str):
        """Copies a file from the host to the unified container."""
        container_name = f"openfork_{self.service_name}_1"
        dest_path = f"{container_name}:{dest_in_container}"
        command = ['docker', 'cp', source_on_host, dest_path]
        self._run_command(command)
