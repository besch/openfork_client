import subprocess
import logging
import threading
import time
import os
import tempfile

def docker_cp(source_path: str, dest_path: str, shutdown_event: threading.Event):
    """Executes a 'docker cp' command with Windows-specific workarounds for Docker Desktop hangs."""
    if shutdown_event.is_set():
        logging.warning(f"Shutdown event set. Aborting docker cp operation for {source_path} to {dest_path}.")
        raise RuntimeError("Docker cp aborted due to shutdown event.")

    command = ['docker', 'cp', source_path, dest_path]
    logging.info(f"Executing command: {' '.join(command)}")
    
    try:
        if os.name == 'nt':
            # On Windows, use a wrapper script to avoid pipe inheritance issues
            # that cause docker cp to hang with Docker Desktop
            timeout = 600
            start_time = time.time()
            
            # Run docker cp with shell=True and no pipe redirection
            # This avoids the handle inheritance that causes hangs
            result = subprocess.run(
                ' '.join(command),
                shell=True,
                timeout=timeout,
                # Don't capture any output - this avoids pipe blocking
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                # Detach from console
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
            )
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, command)
            
            logging.info(f"Successfully ran command: {' '.join(command)}")
        else:
            # On non-Windows, use the simple approach
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=600
            )
            logging.info(f"Successfully ran command: {' '.join(command)}")
            
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout during docker cp operation for {source_path} to {dest_path}")
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running command: {' '.join(command)}, return code: {e.returncode}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during docker cp: {e}")
        raise

