import os
import logging
import subprocess
from config import DOCKER_COMPOSE_DIR

def manage_docker(action: str, service_type: str = 'default'):
    """Starts or stops the ComfyUI Docker container using docker-compose."""
    if service_type == 'foley':
        compose_file = 'docker-compose.foley.yaml'
    elif service_type == 'text_to_image':
        compose_file = 'docker-compose.qwen.yaml'
    elif service_type == 'vibevoice':
        compose_file = 'docker-compose.vibevoice.yaml'
    elif service_type == 'diffrhythm':
        compose_file = 'docker-compose.diffrhythm.yaml'
    else:
        compose_file = 'docker-compose.yaml'
    
    compose_file_path = os.path.join(DOCKER_COMPOSE_DIR, compose_file)
    if not os.path.exists(compose_file_path):
        logging.error(f"{compose_file} not found in {DOCKER_COMPOSE_DIR}")
        return

    command = ["docker-compose", "-f", compose_file_path, action]
    if action == "up":
        command.append("-d")
    
    logging.info(f"Running '{' '.join(command)}' for service '{service_type}'...")
    try:
        result = subprocess.run(
            command,
            cwd=DOCKER_COMPOSE_DIR,
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=False,
            timeout=1800
        )
        if result.returncode == 0:
            logging.info(f"Docker command '{action}' executed successfully.")
        else:
            logging.error(f"Docker command '{action}' failed with exit code {result.returncode}.")
            logging.error(f"Stderr: {result.stderr.strip()}")
    except Exception as e:
        logging.error(f"An exception occurred while running docker-compose: {e}")
