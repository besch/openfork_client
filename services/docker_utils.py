import subprocess
import logging
import threading
import time
import os
import tempfile


def get_subprocess_hidden_kwargs():
    """
    Returns kwargs for subprocess.run() that hide the console window on Windows.
    
    On Windows, subprocess calls to console applications (like docker, ffmpeg, ffprobe)
    will briefly flash a terminal window unless properly suppressed. This function
    returns the correct kwargs to prevent this.
    
    Returns:
        dict: Kwargs to pass to subprocess.run() or similar functions.
    """
    kwargs = {}
    if os.name == 'nt':
        # Use STARTUPINFO with STARTF_USESHOWWINDOW to properly hide the console window.
        # CREATE_NO_WINDOW alone doesn't always work, and DETACHED_PROCESS can cause issues.
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs['startupinfo'] = startupinfo
        # CREATE_NO_WINDOW prevents child processes from opening a console
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
    return kwargs


def docker_cp(source_path: str, dest_path: str, shutdown_event: threading.Event):
    """Executes a 'docker cp' command with Windows-specific workarounds for Docker Desktop hangs."""
    if shutdown_event.is_set():
        logging.warning(f"Shutdown event set. Aborting docker cp operation for {source_path} to {dest_path}.")
        raise RuntimeError("Docker cp aborted due to shutdown event.")

    command = ['docker', 'cp', source_path, dest_path]
    logging.info(f"Executing command: {' '.join(command)}")
    
    # Get Windows-specific kwargs to hide console window
    hidden_kwargs = get_subprocess_hidden_kwargs()
    
    try:
        if os.name == 'nt':
            # On Windows, use shell=True and no pipe redirection to avoid
            # handle inheritance issues that cause docker cp to hang with Docker Desktop
            timeout = 600
            
            result = subprocess.run(
                ' '.join(command),
                shell=True,
                timeout=timeout,
                # Don't capture any output - this avoids pipe blocking
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                **hidden_kwargs
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

