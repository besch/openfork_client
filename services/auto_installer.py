"""
Dynamic dependency installer for ComfyUI nodes running in Docker.
Executes Manager CLI commands inside the container.
"""
import logging
import subprocess
from services.docker_manager import docker_manager

def manager_install_custom_node(repo_url: str) -> bool:
    """
    Installs a custom node inside the running ComfyUI container.
    
    Args:
        repo_url: Git repository URL of the custom node
        
    Returns:
        True if installation succeeded, False otherwise
    """
    container_name = docker_manager.get_container_name()
    
    if not container_name:
        logging.error("No active ComfyUI container found for node installation")
        return False
    
    # Use the Manager's pip package instead of CLI
    # This is more reliable and doesn't depend on file paths
    cmd = [
        "docker", "exec", container_name,
        "python3", "-c",
        f"""
            import sys
            import os
            import subprocess

            # Add ComfyUI to path
            sys.path.insert(0, '/app/ComfyUI')

            # Clone the repository directly
            repo_url = '{repo_url}'
            repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
            target_path = f'/app/ComfyUI/custom_nodes/{{repo_name}}'

            if os.path.exists(target_path):
                print(f'Repository already exists at {{target_path}}')
                sys.exit(0)

            # Clone using git
            result = subprocess.run(
                ['git', 'clone', repo_url, target_path],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print(f'Git clone failed: {{result.stderr}}')
                sys.exit(1)

            # Install requirements if they exist
            req_file = os.path.join(target_path, 'requirements.txt')
            if os.path.exists(req_file):
                print(f'Installing requirements from {{req_file}}')
                result = subprocess.run(
                    ['pip3', 'install', '-r', req_file],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f'Requirements installation failed: {{result.stderr}}')
                    sys.exit(1)

            print(f'Successfully installed {{repo_name}}')
        """
    ]
    
    logging.info(f"Installing custom node in container: {repo_url}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for large repos
            check=False
        )
        
        if result.returncode == 0:
            logging.info(f"Successfully installed custom node: {repo_url}")
            if result.stdout:
                logging.debug(f"Installation output: {result.stdout}")
            return True
        else:
            logging.error(f"Failed to install custom node: {repo_url}")
            if result.stderr:
                logging.error(f"Error output: {result.stderr}")
            if result.stdout:
                logging.error(f"Stdout: {result.stdout}")
            logging.error(f"Return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout installing custom node: {repo_url}")
        return False
    except Exception as e:
        logging.error(f"Exception installing custom node {repo_url}: {e}", exc_info=True)
        return False


def manager_install_model(model_url: str) -> bool:
    """
    Installs a model inside the running ComfyUI container.
    
    Args:
        model_url: URL to download the model from
        
    Returns:
        True if installation succeeded, False otherwise
    """
    container_name = docker_manager.get_container_name()
    
    if not container_name:
        logging.error("No active ComfyUI container found for model installation")
        return False
    
    # Download model directly using wget/aria2
    cmd = [
        "docker", "exec", container_name,
        "python3", "-c",
        f"""
import os
import subprocess
import urllib.parse

model_url = '{model_url}'

# Try to determine target directory from URL
url_path = urllib.parse.urlparse(model_url).path
filename = os.path.basename(url_path)

# Determine model type from filename
if any(x in filename.lower() for x in ['checkpoint', 'ckpt', 'safetensors']):
    if 'lora' in filename.lower():
        target_dir = '/app/ComfyUI/models/loras'
    elif 'vae' in filename.lower():
        target_dir = '/app/ComfyUI/models/vae'
    else:
        target_dir = '/app/ComfyUI/models/checkpoints'
else:
    target_dir = '/app/ComfyUI/models/checkpoints'

os.makedirs(target_dir, exist_ok=True)
target_path = os.path.join(target_dir, filename)

if os.path.exists(target_path):
    print(f'Model already exists at {{target_path}}')
    import sys
    sys.exit(0)

# Download using aria2c (faster) or wget
try:
    result = subprocess.run(
        ['aria2c', '-x', '16', '-s', '16', '-o', target_path, model_url],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f'Successfully downloaded model to {{target_path}}')
        import sys
        sys.exit(0)
except FileNotFoundError:
    pass

# Fallback to wget
result = subprocess.run(
    ['wget', '-O', target_path, model_url],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print(f'Download failed: {{result.stderr}}')
    import sys
    sys.exit(1)

print(f'Successfully downloaded model to {{target_path}}')
"""
    ]
    
    logging.info(f"Installing model in container: {model_url}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout for large models
            check=False
        )
        
        if result.returncode == 0:
            logging.info(f"Successfully installed model: {model_url}")
            if result.stdout:
                logging.debug(f"Installation output: {result.stdout}")
            return True
        else:
            logging.error(f"Failed to install model: {model_url}")
            if result.stderr:
                logging.error(f"Error: {result.stderr}")
            if result.stdout:
                logging.error(f"Stdout: {result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout installing model: {model_url}")
        return False
    except Exception as e:
        logging.error(f"Exception installing model {model_url}: {e}", exc_info=True)
        return False