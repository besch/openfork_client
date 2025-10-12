import subprocess
import logging

def docker_cp(source_path: str, dest_path: str):
    """Executes a 'docker cp' command."""
    command = ['docker', 'cp', source_path, dest_path]
    logging.info(f"Executing command: {' '.join(command)}")
    try:
        # Using cwd is not necessary as docker cp paths should be absolute or relative to the container
        subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"Successfully ran command: {' '.join(command)}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running command: {' '.join(command)}")
        logging.error(f"Stderr: {e.stderr}")
        logging.error(f"Stdout: {e.stdout}")
        raise
