'''
This module acts as a factory for the Docker manager.
It inspects the DEV_MODE and HEADLESS_MODE flags from the config and exports 
the appropriate manager (either for production, development, or None for headless).
'''
from config import DEV_MODE, HEADLESS_MODE

# In headless/cloud mode (RunPod/Vast.ai), we're already inside the container
# and there's no Docker daemon to connect to. Set docker_manager to None.
if HEADLESS_MODE:
    docker_manager = None
elif DEV_MODE:
    from .docker_dev_service import DockerDevManager
    docker_manager = DockerDevManager()
else:
    from .docker_prod_service import DockerProdManager
    docker_manager = DockerProdManager()
