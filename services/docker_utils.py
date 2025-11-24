import subprocess
import logging
import threading

def docker_cp(source_path: str, dest_path: str, shutdown_event: threading.Event):
    """Executes a 'docker cp' command."""
    if shutdown_event.is_set():
        logging.warning(f"Shutdown event set. Aborting docker cp operation for {source_path} to {dest_path}.")
        raise RuntimeError("Docker cp aborted due to shutdown event.")

    command = ['docker', 'cp', source_path, dest_path]
    logging.info(f"Executing command: {' '.join(command)}")
    try:
        # Using cwd is not necessary as docker cp paths should be absolute or relative to the container
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
        logging.info(f"Successfully ran command: {' '.join(command)}")
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout during docker cp operation for {source_path} to {dest_path}")
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running command: {' '.join(command)}")
        logging.error(f"Stderr: {e.stderr}")
        logging.error(f"Stdout: {e.stdout}")
        raise
