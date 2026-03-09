'''
This module acts as a factory for the Docker manager.
It inspects the DEV_MODE and HEADLESS_MODE flags from the config and exports 
the appropriate manager (either for production, development, or None for headless).
'''
from config import DEV_MODE, HEADLESS_MODE

_manager = None

def get_docker_manager():
    global _manager
    if _manager is not None:
        return _manager
        
    # In headless/cloud mode (RunPod/Vast.ai), we're already inside the container
    # and there's no Docker daemon to connect to. Set docker_manager to None.
    if HEADLESS_MODE:
        _manager = None
    elif DEV_MODE:
        from .docker_dev_service import DockerDevManager
        _manager = DockerDevManager()
    else:
        from .docker_prod_service import DockerProdManager
        _manager = DockerProdManager()
    
    return _manager

# For backward compatibility with existing imports
class DockerManagerProxy:
    def __getattr__(self, name):
        manager = get_docker_manager()
        if manager is None:
            # For headless mode, some methods might still be called.
            # Handle accordingly or raise informative error.
            if name == 'set_docker_image_map':
                return lambda x: None
            if name == 'set_services_config':
                return lambda x: None
            raise RuntimeError("Docker is not available in headless mode.")
        return getattr(manager, name)
    
    def __bool__(self):
        return get_docker_manager() is not None

docker_manager = DockerManagerProxy()
