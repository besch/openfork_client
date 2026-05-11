import subprocess
import logging
import threading
import time
import os
import socket
import tempfile
import tarfile
import shutil
import posixpath

from requests import exceptions as requests_exceptions
from urllib3 import exceptions as urllib3_exceptions


_DOCKER_COPY_MAX_ATTEMPTS = 4
_DOCKER_COPY_RETRY_BASE_SECONDS = 1.0
_TRANSIENT_SOCKET_WINERRORS = {10053, 10054, 10060, 10061}


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
            # handle inheritance issues that cause docker cp to hang with Docker Desktop.
            # Pass command as a list (not a joined string) so subprocess uses list2cmdline()
            # which correctly quotes arguments containing spaces or special characters.
            timeout = 600

            result = subprocess.run(
                command,
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
                stdin=subprocess.DEVNULL,
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


def should_use_api_file_copy(docker_client) -> bool:
    """
    Returns True when the Windows app is talking to Docker over TCP/HTTP.

    In that mode, local file paths are still Windows paths, but the Docker daemon
    lives in WSL/Linux. Using the Docker API avoids path translation issues and
    removes the dependency on a Windows-side `docker` CLI.
    """
    if os.name != 'nt' or docker_client is None:
        return False

    docker_host = (os.environ.get('DOCKER_HOST') or '').lower()
    if docker_host.startswith(('tcp://', 'ssh://')):
        return True

    base_url = (getattr(getattr(docker_client, 'api', None), 'base_url', '') or '').lower()
    return base_url.startswith(('http://', 'https://', 'ssh://'))


def _raise_if_shutdown(shutdown_event: threading.Event, message: str):
    if shutdown_event.is_set():
        logging.warning(message)
        raise RuntimeError("Docker file transfer aborted due to shutdown event.")


def _is_transient_docker_copy_error(exc: BaseException, seen=None) -> bool:
    if seen is None:
        seen = set()

    if id(exc) in seen:
        return False
    seen.add(id(exc))

    transient_types = (
        ConnectionResetError,
        BrokenPipeError,
        TimeoutError,
        socket.timeout,
        requests_exceptions.ConnectionError,
        requests_exceptions.ChunkedEncodingError,
        requests_exceptions.ReadTimeout,
        requests_exceptions.Timeout,
        urllib3_exceptions.ProtocolError,
        urllib3_exceptions.ReadTimeoutError,
        urllib3_exceptions.NewConnectionError,
        urllib3_exceptions.MaxRetryError,
        urllib3_exceptions.TimeoutError,
        tarfile.TarError,
    )

    if isinstance(exc, transient_types):
        return True

    if isinstance(exc, OSError) and getattr(exc, 'winerror', None) in _TRANSIENT_SOCKET_WINERRORS:
        return True

    nested_errors = [getattr(exc, '__cause__', None), getattr(exc, '__context__', None)]
    nested_errors.extend(arg for arg in getattr(exc, 'args', ()) if isinstance(arg, BaseException))

    return any(
        _is_transient_docker_copy_error(nested, seen)
        for nested in nested_errors
        if nested is not None
    )


def _wait_before_retry(shutdown_event: threading.Event, delay_seconds: float, source_path: str, dest_path: str):
    deadline = time.monotonic() + delay_seconds
    while True:
        _raise_if_shutdown(
            shutdown_event,
            f"Shutdown event set while waiting to retry Docker API file copy from {source_path} to {dest_path}.",
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.25))


def _remove_temp_file(path: str):
    if path and os.path.exists(path):
        os.remove(path)


def _select_archive_member(archive: tarfile.TarFile, source_path: str) -> tarfile.TarInfo:
    safe_name = posixpath.basename(source_path.rstrip('/'))

    for member in archive.getmembers():
        if member.isfile() and posixpath.basename(member.name.rstrip('/')) == safe_name:
            return member

    for member in archive.getmembers():
        if member.isfile():
            return member

    raise FileNotFoundError(f"No regular file found in archive for '{source_path}'")


def _copy_file_from_container_api_once(container, source_path: str, dest_path: str, shutdown_event: threading.Event):
    temp_tar_path = None
    temp_dest_path = None
    dest_abs_path = os.path.abspath(dest_path)
    dest_abs_dir = os.path.dirname(dest_abs_path)
    temp_prefix = f".{os.path.basename(dest_abs_path) or 'docker-copy'}."

    try:
        stream, _ = container.get_archive(source_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.tar') as temp_tar:
            temp_tar_path = temp_tar.name
            for chunk in stream:
                _raise_if_shutdown(
                    shutdown_event,
                    f"Shutdown event set during Docker API file copy from {source_path} to {dest_path}.",
                )
                if chunk:
                    temp_tar.write(chunk)

        with tarfile.open(temp_tar_path, mode='r:*') as archive:
            member = _select_archive_member(archive, source_path)
            extracted_file = archive.extractfile(member)
            if extracted_file is None:
                raise FileNotFoundError(f"Could not extract '{source_path}' from container archive.")

            with tempfile.NamedTemporaryFile(
                delete=False,
                dir=dest_abs_dir,
                prefix=temp_prefix,
                suffix='.part',
            ) as temp_dest:
                temp_dest_path = temp_dest.name
                with extracted_file:
                    shutil.copyfileobj(extracted_file, temp_dest)

        os.replace(temp_dest_path, dest_abs_path)
        temp_dest_path = None
    finally:
        _remove_temp_file(temp_tar_path)
        _remove_temp_file(temp_dest_path)


def copy_file_from_container_api(container, source_path: str, dest_path: str, shutdown_event: threading.Event):
    """Copies a file from a container to the host using the Docker API."""
    _raise_if_shutdown(
        shutdown_event,
        f"Shutdown event set. Aborting Docker API file copy from {source_path} to {dest_path}.",
    )

    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    for attempt in range(1, _DOCKER_COPY_MAX_ATTEMPTS + 1):
        try:
            _copy_file_from_container_api_once(container, source_path, dest_path, shutdown_event)
            logging.info(f"Successfully copied '{source_path}' from container to '{dest_path}' via Docker API.")
            return
        except Exception as e:
            if (
                shutdown_event.is_set()
                or attempt >= _DOCKER_COPY_MAX_ATTEMPTS
                or not _is_transient_docker_copy_error(e)
            ):
                logging.error(
                    f"Failed Docker API file copy from {source_path} to {dest_path} "
                    f"after {attempt} attempt(s): {e}",
                    exc_info=True,
                )
                raise

            retry_delay = _DOCKER_COPY_RETRY_BASE_SECONDS * attempt
            logging.warning(
                f"Transient Docker API file copy failure from {source_path} to {dest_path} "
                f"on attempt {attempt}/{_DOCKER_COPY_MAX_ATTEMPTS}; retrying in {retry_delay:.1f}s: {e}"
            )
            _wait_before_retry(shutdown_event, retry_delay, source_path, dest_path)


def copy_file_to_container_api(container, source_path: str, dest_path: str, shutdown_event: threading.Event):
    """Copies a file from the host into a container using the Docker API."""
    _raise_if_shutdown(
        shutdown_event,
        f"Shutdown event set. Aborting Docker API file copy from {source_path} to {dest_path}.",
    )

    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Source file '{source_path}' does not exist.")

    container_dir = posixpath.dirname(dest_path) or '/'
    container_name = posixpath.basename(dest_path.rstrip('/'))
    if not container_name:
        raise ValueError(f"Destination path '{dest_path}' must include a filename.")

    temp_tar_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tar') as temp_tar:
            temp_tar_path = temp_tar.name

        with tarfile.open(temp_tar_path, mode='w') as archive:
            archive.add(source_path, arcname=container_name)

        _raise_if_shutdown(
            shutdown_event,
            f"Shutdown event set before uploading {source_path} to {dest_path}.",
        )

        with open(temp_tar_path, 'rb') as archive_stream:
            result = container.put_archive(container_dir, archive_stream)

        if not result:
            raise RuntimeError(f"Docker API rejected archive upload to '{dest_path}'.")

        logging.info(f"Successfully copied '{source_path}' to container path '{dest_path}' via Docker API.")
    except Exception as e:
        logging.error(
            f"Failed Docker API file copy from {source_path} to {dest_path}: {e}",
            exc_info=True,
        )
        raise
    finally:
        if temp_tar_path and os.path.exists(temp_tar_path):
            os.remove(temp_tar_path)

