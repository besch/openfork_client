"""
Complete Docker manager with container lifecycle management.
"""
import logging
import subprocess
import time
import os
from typing import Dict, List, Optional

class DockerManager:
    def __init__(self):
        self.container_name = None
        self.compose_file = None
        self.is_running = False
        self.installed_nodes_cache = set()
        self.cache_timestamp = 0
        self.cache_ttl = 300  # 5 minutes
    
    def set_compose_file(self, compose_file_path: str):
        """Set the docker-compose file path for this manager."""
        self.compose_file = compose_file_path
        logging.info(f"Docker manager configured with compose file: {compose_file_path}")
    
    def get_container_name(self) -> Optional[str]:
        """Get the name of the running ComfyUI container."""
        if self.container_name and self.is_running:
            return self.container_name
        
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=comfyui", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )
            containers = result.stdout.strip().split('\n')
            if containers and containers[0]:
                self.container_name = containers[0]
                self.is_running = True
                logging.debug(f"Found ComfyUI container: {self.container_name}")
                return self.container_name
        except Exception as e:
            logging.debug(f"Could not find ComfyUI container: {e}")
            self.is_running = False
        
        return None
    
    def is_container_running(self) -> bool:
        """Check if the ComfyUI container is currently running."""
        return self.get_container_name() is not None
    
    def invalidate_node_cache(self):
        """Invalidate the cached node information to force a refresh."""
        self.installed_nodes_cache = set()
        self.cache_timestamp = 0
        logging.debug("Node cache invalidated")
    
    def run_container(self, dependencies: Dict = None):
        """
        Start the ComfyUI container with optional dependencies.
        
        Args:
            dependencies: Dict with 'custom_node_urls' and 'model_urls' lists
        """
        if not self.compose_file:
            raise RuntimeError("Cannot start container: compose file not set")
        
        # Check if container is already running
        if self.is_container_running():
            logging.info("ComfyUI container is already running")
            return
        
        logging.info("Starting ComfyUI container...")
        
        # Build environment variables for dependencies
        env = os.environ.copy()
        
        if dependencies:
            custom_nodes = dependencies.get('custom_node_urls', [])
            models = dependencies.get('model_urls', [])
            
            if custom_nodes:
                env['CUSTOM_NODES_GIT_URLS'] = ','.join(custom_nodes)
                logging.info(f"Passing {len(custom_nodes)} custom node URLs to container")
            
            if models:
                env['MODEL_URLS'] = ','.join(models)
                logging.info(f"Passing {len(models)} model URLs to container")
        
        try:
            cmd = [
                "docker-compose",
                "-f", self.compose_file,
                "up", "--build", "-d"
            ]
            
            logging.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=600
            )
            
            if result.stdout:
                logging.debug(result.stdout)
            if result.stderr:
                logging.debug(result.stderr)
            
            # Clear cached container name to force re-detection
            self.container_name = None
            self.is_running = False
            
            # Wait a moment for container to initialize
            time.sleep(3)
            
            # Verify container started
            if self.is_container_running():
                logging.info("Container for service 'comfyui' started successfully.")
            else:
                raise RuntimeError("Container started but could not be detected")
                
        except subprocess.TimeoutExpired:
            logging.error("Timeout while starting container")
            raise RuntimeError("Container startup timed out after 600 seconds")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to start container: {e}")
            if e.stderr:
                logging.error(f"Error output: {e.stderr}")
            raise RuntimeError(f"Failed to start ComfyUI container: {e.stderr}")
    
    def stop_container(self):
        """Stop the ComfyUI container."""
        if not self.compose_file:
            raise RuntimeError("Cannot stop container: compose file not set")
        
        logging.info("Stopping ComfyUI container...")
        
        try:
            cmd = ["docker-compose", "-f", self.compose_file, "stop"]
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
            
            self.container_name = None
            self.is_running = False
            logging.info("Container stopped successfully")
            
        except Exception as e:
            logging.error(f"Failed to stop container: {e}")
            raise
    
    def restart_container(self) -> bool:
        """
        Restart the ComfyUI container to load newly installed dependencies.
        
        Returns:
            True if restart succeeded, False otherwise
        """
        if not self.compose_file:
            logging.error("Cannot restart: compose file not set")
            return False
        
        logging.info("Restarting ComfyUI container to load new dependencies...")
        
        try:
            # Use restart command for faster cycle
            cmd = ["docker-compose", "-f", self.compose_file, "restart"]
            
            logging.debug(f"Restarting container: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=120
            )
            
            logging.info("Container restarted successfully")
            if result.stdout:
                logging.debug(f"Restart output: {result.stdout}")
            
            # Clear caches
            self.container_name = None
            self.is_running = False
            self.invalidate_node_cache()
            
            # Give it a moment to start
            time.sleep(5)
            
            return True
            
        except subprocess.TimeoutExpired:
            logging.error("Timeout while restarting container")
            return False
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to restart container: {e}")
            if e.stderr:
                logging.error(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            logging.error(f"Exception during container restart: {e}", exc_info=True)
            return False
    
    def copy_file_from_container(self, source_in_container: str, dest_on_host: str):
        """Copy a file from the container to the host."""
        container_name = self.get_container_name()
        if not container_name:
            raise RuntimeError("No active container found for file copy")
        
        cmd = [
            "docker", "cp",
            f"{container_name}:{source_in_container}",
            dest_on_host
        ]
        
        logging.debug(f"Copying from container: {source_in_container} -> {dest_on_host}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
            logging.debug(f"Successfully copied file from container")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to copy file from container: {e.stderr}")
            raise RuntimeError(f"Docker cp failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("File copy timed out after 60 seconds")
    
    def execute_in_container(self, command: List[str], timeout: int = 600) -> tuple:
        """
        Execute a command inside the container.
        
        Args:
            command: List of command parts to execute
            timeout: Command timeout in seconds
            
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        container_name = self.get_container_name()
        if not container_name:
            raise RuntimeError("No active container found for command execution")
        
        cmd = ["docker", "exec", container_name] + command
        
        logging.debug(f"Executing in container: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=False,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logging.error(f"Command timed out after {timeout} seconds: {' '.join(command)}")
            return -1, "", f"Command timed out after {timeout} seconds"
    
    def get_container_logs(self, tail: int = 100) -> str:
        """
        Get recent logs from the container.
        
        Args:
            tail: Number of log lines to retrieve
            
        Returns:
            Container logs as string
        """
        if not self.compose_file:
            return "Compose file not set"
        
        try:
            cmd = [
                "docker-compose", "-f", self.compose_file,
                "logs", "--tail", str(tail)
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            return result.stdout
        except Exception as e:
            logging.error(f"Failed to get container logs: {e}")
            return f"Error getting logs: {e}"

# Global singleton instance
docker_manager = DockerManager()