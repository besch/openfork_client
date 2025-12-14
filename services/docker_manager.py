"""
Stub docker_manager module.
Docker-based container management is deprecated in favor of LocalComfyUIManager.
This stub exists only to prevent import errors in legacy code.
"""
import logging


class DockerManager:
    """Stub DockerManager that does nothing - Docker mode is deprecated."""
    
    def run_container(self, service_type=None):
        logging.warning(f"DockerManager.run_container called for '{service_type}' but Docker mode is deprecated. No action taken.")
        pass
    
    def stop_container(self, service_type=None):
        logging.warning(f"DockerManager.stop_container called for '{service_type}' but Docker mode is deprecated. No action taken.")
        pass
    
    def set_docker_image_map(self, image_map):
        pass


docker_manager = DockerManager()
