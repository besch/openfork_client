'''
This module acts as a factory for the Docker manager.
It inspects the DEV_MODE flag from the config and exports the appropriate
manager (either for production or development) under a unified name.
'''
from config import DEV_MODE

if DEV_MODE:
    from .docker_dev_service import DockerDevManager
    docker_manager = DockerDevManager()
else:
    from .docker_prod_service import DockerProdManager
    docker_manager = DockerProdManager()
