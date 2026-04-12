"""
This module contains the production Docker service, which manages pre-built
images from a Docker registry (e.g., Docker Hub).
"""

import docker
import logging
import os
import threading
import time
import requests
from .docker_utils import (
    docker_cp,
    should_use_api_file_copy,
    copy_file_from_container_api,
    copy_file_to_container_api,
)


class DockerProdManager:
    def _get_wsl_ip(self):
        """Attempts to detect the WSL VM IP address from Windows host."""
        import sys

        if sys.platform != "win32":
            return None
        try:
            import subprocess

            # Use the distro name from OPENFORK_WSL_DISTRO env var if set,
            # otherwise fall back to the default WSL distro (no -d flag).
            distro = os.environ.get("OPENFORK_WSL_DISTRO")
            cmd = (
                ["wsl", "-d", distro, "--", "hostname", "-I"]
                if distro
                else ["wsl", "--", "hostname", "-I"]
            )
            output = subprocess.check_output(cmd, timeout=5, stderr=subprocess.DEVNULL)
            ips = output.decode().strip().split()
            return ips[0] if ips else None
        except:
            return None

    def __init__(self):
        try:
            logging.info("Initializing Docker client...")
            self.client = docker.from_env(timeout=300)
            self.client.ping()
            logging.info("Successfully connected to Docker via from_env()")
        except (docker.errors.DockerException, Exception) as e:
            # On Windows, from_env might fail if DOCKER_HOST isn't perfectly formed
            # or the pipe isn't available. Try explicit fallback connections with retry.
            import os
            import time
            import sys

            error_msg = (
                f"docker.from_env() failed: {e}. Entering fallback retry loop..."
            )
            print(f"DEBUG: {error_msg}", file=sys.stderr, flush=True)
            logging.warning(error_msg)

            wsl_ip = self._get_wsl_ip()
            explicit_host = os.environ.get("DOCKER_HOST")
            if explicit_host:
                # An explicit endpoint was configured (e.g. tcp://127.0.0.1:2375 for WSL
                # Docker or npipe://... for Docker Desktop).  Trust it; only add TCP
                # variants when it is already a TCP URL so we don't accidentally fall
                # through to the WSL daemon when Docker Desktop is the intended engine.
                if explicit_host.startswith("tcp://"):
                    docker_hosts = [
                        explicit_host,
                        "tcp://127.0.0.1:2375",
                        "tcp://localhost:2375",
                        f"tcp://{wsl_ip}:2375" if wsl_ip else None,
                    ]
                else:
                    docker_hosts = [explicit_host, "npipe:////./pipe/docker_engine"]
            else:
                # No explicit endpoint: Docker Desktop native mode.
                # Prefer the named pipe so we never accidentally pick up a WSL Docker
                # daemon that happens to be listening on port 2375.
                docker_hosts = [
                    "npipe:////./pipe/docker_engine",
                    "tcp://127.0.0.1:2375",
                    "tcp://localhost:2375",
                    f"tcp://{wsl_ip}:2375" if wsl_ip else None,
                ]

            # Remove duplicates and None values while preserving order
            docker_hosts = list(dict.fromkeys([h for h in docker_hosts if h]))

            connected = False
            start_time = time.time()
            retry_duration = 60
            iteration = 0

            while True:
                iteration += 1
                current_time = time.time()
                elapsed = current_time - start_time

                if elapsed >= retry_duration:
                    debug_timeout = (
                        f"Docker connection retry timed out after {elapsed:.1f}s"
                    )
                    print(f"DEBUG: {debug_timeout}", file=sys.stderr, flush=True)
                    break

                print(
                    f"DEBUG: Loop iteration {iteration}, elapsed {elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )

                for host in docker_hosts:
                    if not host:
                        continue
                    try:
                        print(f"DEBUG: Testing {host}...", file=sys.stderr, flush=True)
                        self.client = docker.DockerClient(base_url=host, timeout=120)
                        self.client.ping()

                        success_msg = f"Successfully connected to Docker at {host}"
                        print(f"INFO: {success_msg}", flush=True)
                        logging.info(success_msg)
                        connected = True
                        break
                    except Exception as ex:
                        # Log specific error for each host to understand why it fails
                        print(
                            f"DEBUG: {host} failed: {type(ex).__name__}: {ex}",
                            file=sys.stderr,
                            flush=True,
                        )

                if connected:
                    break

                wait_msg = f"Waiting for Docker daemon to become available... ({int(elapsed)}/{retry_duration}s)"
                print(f"INFO: {wait_msg}", flush=True)
                logging.info(wait_msg)
                time.sleep(2)

            if not connected:
                fail_msg = "CRITICAL: Docker daemon not found or not running after all fallback attempts."
                print(f"ERROR: {fail_msg}", file=sys.stderr, flush=True)
                logging.error(fail_msg)
                self.client = None

        self._use_api_file_copy = should_use_api_file_copy(self.client)
        if self._use_api_file_copy:
            base_url = getattr(getattr(self.client, "api", None), "base_url", "unknown")
            logging.info(
                f"Using Docker API for file transfers via remote Docker host: {base_url}"
            )

        self.docker_image_map = {}
        self.services_config = {}

    def _is_transient_transport_error(self, exc: Exception) -> bool:
        if isinstance(
            exc,
            (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                TimeoutError,
            ),
        ):
            return True

        text = str(exc).lower()
        transient_markers = (
            "connection aborted",
            "timed out",
            "timeout",
            "npipe",
            "named pipe",
            "docker daemon",
            "protocolerror",
        )
        return any(marker in text for marker in transient_markers)

    def _refresh_client_connection(self) -> bool:
        """Best-effort reconnect after transient Docker transport failures."""
        candidates = []

        current_base_url = getattr(getattr(self.client, "api", None), "base_url", None)
        if current_base_url:
            candidates.append(current_base_url)

        docker_host = os.environ.get("DOCKER_HOST")
        if docker_host:
            candidates.append(docker_host)

        if os.name == "nt":
            candidates.extend(
                [
                    "npipe:////./pipe/docker_engine",
                    "tcp://127.0.0.1:2375",
                    "tcp://localhost:2375",
                ]
            )

        for base_url in dict.fromkeys(filter(None, candidates)):
            try:
                logging.info(f"Attempting Docker client reconnect via {base_url}...")
                client = docker.DockerClient(base_url=base_url, timeout=300)
                client.ping()
                self.client = client
                self._use_api_file_copy = should_use_api_file_copy(self.client)
                logging.info(f"Reconnected to Docker at {base_url}")
                return True
            except Exception as reconnect_error:
                logging.debug(
                    f"Docker reconnect attempt via {base_url} failed: {reconnect_error}"
                )

        try:
            logging.info("Attempting Docker reconnect via docker.from_env()...")
            client = docker.from_env(timeout=300)
            client.ping()
            self.client = client
            self._use_api_file_copy = should_use_api_file_copy(self.client)
            logging.info("Reconnected to Docker via docker.from_env()")
            return True
        except Exception as reconnect_error:
            logging.error(f"Failed to reconnect to Docker: {reconnect_error}")
            return False

    def _get_existing_container(self, container_name: str):
        try:
            return self.client.containers.get(container_name)
        except docker.errors.NotFound:
            return None
        except Exception as e:
            logging.debug(f"Could not inspect container '{container_name}': {e}")
            return None

    def set_docker_image_map(self, image_map: dict):
        if image_map:
            logging.info("Setting dynamic Docker image map.")
            self.docker_image_map = image_map
        else:
            logging.warning(
                "Dynamic Docker image map is empty. Using fallback static map."
            )

    def set_services_config(self, services_config: dict):
        if services_config:
            logging.info("Setting services configuration.")
            self.services_config = services_config

    def get_default_ports(self, service_type: str) -> dict:
        config = self.services_config.get(service_type, {})
        port = config.get("port", 8188)  # Default to ComfyUI port
        return {f"{port}/tcp": port}

    def get_image_name(self, service_type: str) -> str:
        image = self.docker_image_map.get(service_type)
        if not image:
            raise ValueError(
                f"No Docker image configured for service type '{service_type}'"
            )
        return image

    def get_container_name(self, service_type: str) -> str:
        return f"dgn-client-{service_type}"

    def pull_image(
        self,
        image_name: str,
        shutdown_event: threading.Event = None,
        service_type: str = None,
    ):
        if not self.client:
            logging.error(
                f"Cannot pull image '{image_name}': Docker client not initialized."
            )
            return

        try:
            logging.info(f"Checking for Docker image: {image_name}...")
            self.client.images.get(image_name)
            logging.info(f"Image '{image_name}' found locally. Skipping pull.")
            return
        except docker.errors.ImageNotFound:
            logging.info(
                f"Image '{image_name}' not found locally. Pulling from Docker Hub..."
            )

            # Disk space check before download
            from .disk_space_utils import (
                check_sufficient_space,
                estimate_image_size_bytes,
            )
            import json

            estimated_size = estimate_image_size_bytes(image_name)
            has_space, available, required = check_sufficient_space(estimated_size)

            if not has_space:
                available_gb = available / (1024**3)
                required_gb = required / (1024**3)
                error_msg = (
                    f"Insufficient disk space to download '{image_name}'. "
                    f"Required: {required_gb:.1f} GB (including 5 GB safety buffer), "
                    f"Available: {available_gb:.1f} GB"
                )
                logging.error(error_msg)

                # Emit event for desktop UI notification
                print(
                    json.dumps(
                        {
                            "type": "DISK_SPACE_ERROR",
                            "payload": {
                                "image_name": image_name,
                                "required_gb": round(required_gb, 1),
                                "available_gb": round(available_gb, 1),
                                "message": error_msg,
                            },
                        }
                    ),
                    flush=True,
                )

                raise OSError(error_msg)

            try:
                from .docker_progress_logger import stream_pull_with_progress

                stream_pull_with_progress(
                    self.client,
                    image_name,
                    throttle_interval=0.5,
                    shutdown_event=shutdown_event,
                    service_type=service_type,
                )
                logging.info(f"Successfully pulled image: {image_name}")
            except docker.errors.APIError as e:
                # Case 2: Docker raises APIError mid-pull when the disk fills up.
                # The pre-download check passed but space ran out during the long pull.
                # Emit DISK_SPACE_ERROR so the Electron UI surfaces an actionable alert
                # rather than a silent failure.
                err_str = str(e).lower()
                if "no space left" in err_str or "disk quota exceeded" in err_str:
                    from .disk_space_utils import get_available_disk_space

                    available_gb = get_available_disk_space() / (1024**3)
                    required_gb = estimated_size / (1024**3)  # from pre-check above
                    disk_error_msg = (
                        f"Ran out of disk space while downloading '{image_name}'. "
                        f"Available: {available_gb:.1f} GB"
                    )
                    logging.error(disk_error_msg)
                    print(
                        json.dumps(
                            {
                                "type": "DISK_SPACE_ERROR",
                                "payload": {
                                    "image_name": image_name,
                                    "required_gb": round(required_gb, 1),
                                    "available_gb": round(available_gb, 1),
                                    "message": disk_error_msg,
                                },
                            }
                        ),
                        flush=True,
                    )
                    raise OSError(disk_error_msg)
                logging.error(f"Failed to pull image '{image_name}': {e}")
                raise

    def exec_in_container(
        self,
        service_type: str,
        command: list,
        detach: bool = False,
        environment: dict = None,
    ):
        """Run a command inside a running container."""
        if not self.client:
            logging.error(
                f"Cannot exec in container for '{service_type}': Docker client not initialized."
            )
            return None
        container_name = self.get_container_name(service_type)
        try:
            container = self.client.containers.get(container_name)
            result = container.exec_run(command, detach=detach, environment=environment)
            return result
        except docker.errors.NotFound:
            logging.error(f"Container '{container_name}' not found for exec.")
            return None
        except docker.errors.APIError as e:
            logging.error(f"exec_run failed in '{container_name}': {e}")
            return None

    def run_container(
        self,
        service_type: str,
        ports: dict = None,
        force_restart: bool = True,
        command: list = None,
    ):
        if not self.client:
            logging.error(
                f"Cannot run container for '{service_type}': Docker client not initialized."
            )
            return

        image_name = self.get_image_name(service_type)
        container_name = self.get_container_name(service_type)

        # Get ports from configuration if not provided
        if ports is None:
            ports = self.get_default_ports(service_type)

        try:
            container = self.client.containers.get(container_name)
            if force_restart:
                logging.info(
                    f"Found existing container '{container_name}' with status '{container.status}'. Removing it before starting a new one."
                )
                container.remove(force=True)
            else:
                if container.status == "running":
                    logging.info(
                        f"Container '{container_name}' is already running and force_restart is False. Skipping start."
                    )
                    return
                else:
                    logging.info(
                        f"Container '{container_name}' exists but is '{container.status}'. Removing and restarting."
                    )
                    container.remove(force=True)

        except docker.errors.NotFound:
            logging.info(
                f"No existing container named '{container_name}' found. Proceeding to create a new one."
            )
            pass  # Container does not exist, which is fine.
        except docker.errors.APIError as e:
            logging.error(
                f"Error checking/removing existing container '{container_name}': {e}. Attempting to continue."
            )
            # Log the error but try to proceed. The run command will likely fail if removal did, but it's worth a try.

        self.pull_image(image_name)

        logging.info(
            f"Starting container '{container_name}' from image '{image_name}' with ports {ports}..."
        )

        # Determine device requests (GPU support)
        device_requests = []
        import platform

        if (
            platform.system() != "Darwin"
        ):  # macOS does not support GPU pass-through for Docker
            try:
                from .hardware_profiler import get_available_vram

                if get_available_vram() > 0:
                    device_requests = [
                        docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                    ]
            except Exception as e:
                logging.debug(
                    f"Could not verify GPU availability, skipping GPU request: {e}"
                )

        try:
            run_kwargs = dict(
                image=image_name,
                detach=True,
                name=container_name,
                ports=ports,
                device_requests=device_requests,
                restart_policy={"Name": "no"},
            )
            if command:
                run_kwargs["command"] = command
            max_attempts = 2
            for attempt in range(1, max_attempts + 1):
                try:
                    self.client.containers.run(**run_kwargs)
                    logging.info(f"Container '{container_name}' started successfully.")
                    return
                except Exception as e:
                    if attempt < max_attempts and self._is_transient_transport_error(e):
                        logging.warning(
                            f"Transient Docker transport error while starting "
                            f"'{container_name}' (attempt {attempt}/{max_attempts}): {e}"
                        )
                        self._refresh_client_connection()

                        existing = self._get_existing_container(container_name)
                        if existing is not None:
                            try:
                                existing.reload()
                                status = existing.status
                            except Exception:
                                status = "unknown"

                            if status == "created":
                                try:
                                    existing.start()
                                    existing.reload()
                                    status = existing.status
                                except Exception as start_error:
                                    logging.debug(
                                        f"Could not start existing container "
                                        f"'{container_name}' after reconnect: {start_error}"
                                    )

                            if status in ("running", "restarting"):
                                logging.info(
                                    f"Container '{container_name}' appears to have started "
                                    "despite the transport timeout. Reusing it."
                                )
                                return

                            if status in ("created", "exited", "dead"):
                                try:
                                    existing.remove(force=True)
                                    logging.info(
                                        f"Removed partially created container "
                                        f"'{container_name}' before retry."
                                    )
                                except Exception as cleanup_error:
                                    logging.debug(
                                        f"Could not remove partially created container "
                                        f"'{container_name}': {cleanup_error}"
                                    )

                        time.sleep(2)
                        continue

                    logging.error(f"Failed to start container '{container_name}': {e}")
                    raise
        except docker.errors.APIError as e:
            logging.error(f"Failed to start container '{container_name}': {e}")
            raise

    def stop_container(self, service_type: str):
        if not self.client:
            logging.error(
                f"Cannot stop container for '{service_type}': Docker client not initialized."
            )
            return

        container_name = self.get_container_name(service_type)
        logging.info(f"Attempting to stop and remove container '{container_name}'...")
        try:
            container = self.client.containers.get(container_name)
            logging.info(f"Container '{container_name}' found. Forcefully removing it.")
            container.remove(force=True)
            logging.info(f"Container '{container_name}' removed.")
        except docker.errors.NotFound:
            logging.info(f"Container '{container_name}' not found. Nothing to stop.")
        except docker.errors.APIError as e:
            logging.error(f"Failed to stop or remove container '{container_name}': {e}")

    def copy_file_from_container(
        self,
        service_type: str,
        source_in_container: str,
        dest_on_host: str,
        shutdown_event: threading.Event,
    ):
        container_name = self.get_container_name(service_type)
        if self._use_api_file_copy:
            container = self.client.containers.get(container_name)
            copy_file_from_container_api(
                container, source_in_container, dest_on_host, shutdown_event
            )
            return

        source_path = f"{container_name}:{source_in_container}"
        docker_cp(source_path, dest_on_host, shutdown_event)

    def copy_file_to_container(
        self,
        service_type: str,
        source_on_host: str,
        dest_in_container: str,
        shutdown_event: threading.Event,
    ):
        container_name = self.get_container_name(service_type)
        if self._use_api_file_copy:
            container = self.client.containers.get(container_name)
            copy_file_to_container_api(
                container, source_on_host, dest_in_container, shutdown_event
            )
            return

        dest_path = f"{container_name}:{dest_in_container}"
        docker_cp(source_on_host, dest_path, shutdown_event)

    def stream_logs(self, service_type: str, shutdown_event: threading.Event):
        """Streams logs from the container to the application logger."""
        if not self.client:
            return

        container_name = self.get_container_name(service_type)
        logging.info(f"Starting log stream for container '{container_name}'")

        try:
            container = self.client.containers.get(container_name)
            # stream=True returns a generator processing the logs
            for line in container.logs(stream=True, follow=True):
                if shutdown_event.is_set():
                    break
                if line:
                    # Decode utf-8 with replacement for invalid bytes
                    decoded_line = line.decode("utf-8", errors="replace").strip()
                    # Force encode to ascii to remove characters that crash Windows consoles (like progress bars)
                    decoded_line = decoded_line.encode(
                        "ascii", errors="replace"
                    ).decode("ascii")
                    if decoded_line:
                        logging.info(f"[{service_type}] {decoded_line}")
        except docker.errors.NotFound:
            logging.warning(f"Container '{container_name}' not found for logging.")
        except Exception as e:
            # If the container stops, we might get an APIError or similar, which is expected
            if not shutdown_event.is_set():
                logging.debug(f"Log streaming interrupted for '{container_name}': {e}")
