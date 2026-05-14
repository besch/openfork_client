"""
This module contains the production Docker service, which manages pre-built
images from a Docker registry (e.g., Docker Hub).
"""

import docker
import logging
import os
import re
import subprocess
import threading
import time
import requests
from typing import Optional
from .docker_utils import (
    docker_cp,
    get_subprocess_hidden_kwargs,
    should_use_api_file_copy,
    copy_file_from_container_api,
    copy_file_to_container_api,
)

DOCKER_API_TIMEOUT_SECS = int(os.getenv("OPENFORK_DOCKER_API_TIMEOUT_SECS", "3600"))
DOCKER_CLI_PULL_TIMEOUT_SECS = int(os.getenv("OPENFORK_DOCKER_CLI_PULL_TIMEOUT_SECS", "7200"))


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
            self.client = docker.from_env(timeout=DOCKER_API_TIMEOUT_SECS)
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

            allow_wsl_ip_fallback = (
                os.environ.get("OPENFORK_ALLOW_WSL_DOCKER_IP_FALLBACK") == "1"
            )
            wsl_ip = self._get_wsl_ip() if allow_wsl_ip_fallback else None
            explicit_host = os.environ.get("DOCKER_HOST")
            if explicit_host:
                # An explicit endpoint was configured for the OpenFork Ubuntu Docker daemon.
                # Trust it and add the common local TCP variants to handle WSL IP churn.
                if explicit_host.startswith("tcp://"):
                    docker_hosts = [
                        explicit_host,
                        "tcp://127.0.0.1:2375",
                        "tcp://localhost:2375",
                        f"tcp://{wsl_ip}:2375" if wsl_ip else None,
                    ]
                else:
                    docker_hosts = [explicit_host]
            else:
                # No explicit endpoint: probe the OpenFork Ubuntu Docker daemon directly.
                docker_hosts = [
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
                        self.client = docker.DockerClient(
                            base_url=host,
                            timeout=DOCKER_API_TIMEOUT_SECS,
                        )
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

    @staticmethod
    def _is_layer_extraction_error(exc: Exception) -> bool:
        """True for 500s caused by a transient gRPC/layer-extraction failure.

        These are NOT permanent image errors — Docker Desktop's containerd
        connection to the WSL2 backend dropped mid-extraction. Re-pulling the
        image repairs damaged layers and resolves the error.
        """
        if not (
            isinstance(exc, docker.errors.APIError)
            and getattr(exc, "status_code", None) == 500
        ):
            return False
        text = str(exc).lower()
        return "apply layer error" in text or (
            "extract layer" in text and "canceled" in text
        ) or (
            "grpc" in text
            and (
                "client connection is closing" in text
                or "context canceled" in text
                or "context cancelled" in text
            )
        )

    def _is_transient_transport_error(self, exc: Exception) -> bool:
        if isinstance(exc, docker.errors.APIError):
            if self._is_layer_extraction_error(exc):
                return True
            if getattr(exc, "status_code", None) in (409, 400, 404, 500):
                return False

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
                    "tcp://127.0.0.1:2375",
                    "tcp://localhost:2375",
                ]
            )

        for base_url in dict.fromkeys(filter(None, candidates)):
            try:
                logging.info(f"Attempting Docker client reconnect via {base_url}...")
                client = docker.DockerClient(
                    base_url=base_url,
                    timeout=DOCKER_API_TIMEOUT_SECS,
                )
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
            client = docker.from_env(timeout=DOCKER_API_TIMEOUT_SECS)
            client.ping()
            self.client = client
            self._use_api_file_copy = should_use_api_file_copy(self.client)
            logging.info("Reconnected to Docker via docker.from_env()")
            return True
        except Exception as reconnect_error:
            logging.error(f"Failed to reconnect to Docker: {reconnect_error}")
            return False

    def _docker_cli_env(self) -> dict:
        """Return env for subprocess docker CLI calls, routing to the same daemon as the SDK.

        On Windows the default `docker` CLI connects to Docker Desktop's named pipe,
        not to the OpenFork WSL daemon at TCP 2375. Setting DOCKER_HOST ensures all
        subprocess inspect/rm calls hit the same daemon the SDK is talking to.
        """
        env = os.environ.copy()
        try:
            base_url = getattr(getattr(self.client, "api", None), "base_url", None)
            if base_url and "://" in base_url:
                # SDK stores http:// internally; docker CLI env var expects tcp://
                docker_host = base_url.replace("http://", "tcp://", 1)
                if docker_host.startswith("tcp://"):
                    env["DOCKER_HOST"] = docker_host
        except Exception:
            pass
        return env

    @staticmethod
    def _normalize_docker_arch(arch: Optional[str]) -> Optional[str]:
        if not arch:
            return None
        normalized = arch.lower()
        aliases = {
            "x86_64": "amd64",
            "aarch64": "arm64",
        }
        return aliases.get(normalized, normalized)

    def _get_pull_platform(self) -> Optional[str]:
        """Return an explicit platform for Docker pulls when the daemon reports one.

        OpenFork's WSL daemon runs linux/amd64 on supported Windows installs. Passing
        the platform avoids OCI-index ambiguity for large images that also publish
        non-runnable metadata manifests.
        """
        override = os.environ.get("OPENFORK_DOCKER_PULL_PLATFORM")
        if override:
            return override

        try:
            info = self.client.info()
            os_type = (info.get("OSType") or "").lower()
            arch = self._normalize_docker_arch(info.get("Architecture"))
            if os_type and arch:
                return f"{os_type}/{arch}"
        except Exception as e:
            logging.debug(f"Could not resolve Docker daemon platform: {e}")

        return None

    def _image_is_registered(self, image_name: str) -> bool:
        try:
            self.client.images.get(image_name)
            return True
        except docker.errors.ImageNotFound:
            return False
        except Exception as e:
            if self._is_transient_transport_error(e) and self._refresh_client_connection():
                try:
                    self.client.images.get(image_name)
                    return True
                except docker.errors.ImageNotFound:
                    return False
                except Exception as retry_error:
                    logging.debug(
                        f"Image lookup still failed after reconnect for '{image_name}': {retry_error}"
                    )
            else:
                logging.debug(f"Image lookup failed for '{image_name}': {e}")
            return False

    def _run_docker_cli_pull(
        self,
        image_name: str,
        platform: Optional[str],
        shutdown_event: threading.Event = None,
    ) -> None:
        """Fallback pull path for WSL Docker when the SDK stream ends early."""
        if os.name == "nt" and os.environ.get("OPENFORK_WSL_DISTRO"):
            distro = os.environ["OPENFORK_WSL_DISTRO"]
            command = ["wsl.exe", "-d", distro, "--", "docker", "pull"]
            env = os.environ.copy()
            env.pop("DOCKER_HOST", None)
        else:
            command = ["docker", "pull"]
            env = self._docker_cli_env()

        if platform:
            command.extend(["--platform", platform])
        command.append(image_name)

        logging.info("Running Docker CLI fallback pull for '%s'.", image_name)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            env=env,
            **get_subprocess_hidden_kwargs(),
        )

        deadline = time.monotonic() + DOCKER_CLI_PULL_TIMEOUT_SECS
        output = ""
        while True:
            try:
                stdout, _ = process.communicate(timeout=1.0)
                output += stdout or ""
                break
            except subprocess.TimeoutExpired:
                if shutdown_event and shutdown_event.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise RuntimeError(
                        f"Docker CLI pull for {image_name} cancelled."
                    )
                if time.monotonic() >= deadline:
                    process.kill()
                    raise TimeoutError(
                        f"Docker CLI pull for {image_name} timed out after "
                        f"{DOCKER_CLI_PULL_TIMEOUT_SECS}s."
                    )

        if process.returncode != 0:
            tail = "\n".join(output.splitlines()[-20:])
            raise RuntimeError(
                f"Docker CLI pull failed for {image_name} with exit code "
                f"{process.returncode}: {tail}"
            )

        tail = "\n".join(output.splitlines()[-5:])
        if tail:
            logging.info("Docker CLI fallback pull output tail:\n%s", tail)
        self._refresh_client_connection()

    def _get_existing_container(self, container_name: str):
        try:
            return self.client.containers.get(container_name)
        except docker.errors.NotFound:
            return None
        except Exception as e:
            logging.debug(f"Could not inspect container '{container_name}': {e}")
            return None

    def _list_containers_by_name(self, container_name: str):
        try:
            matches = self.client.containers.list(
                all=True, filters={"name": container_name}
            )
            return [
                c
                for c in matches
                if c.name == container_name or c.name == f"/{container_name}"
            ]
        except Exception as e:
            logging.debug(f"Container list lookup failed for '{container_name}': {e}")
            return []

    def _force_remove_by_name(self, container_name: str, container_id: Optional[str] = None):

        # Collect candidate IDs to remove (by-ID is more reliable than by-name on Windows/WSL2)
        ids_to_remove: list[str] = []
        if container_id:
            ids_to_remove.append(container_id)

        existing = self._get_existing_container(container_name)
        if existing is not None:
            if existing.id not in ids_to_remove:
                ids_to_remove.append(existing.id)
            try:
                existing.remove(force=True)
                logging.info(f"Force-removed conflicting container '{container_name}'.")
                return
            except Exception as e:
                logging.warning(
                    f"containers.get().remove() failed for '{container_name}': {e}"
                )

        try:
            matches = self.client.containers.list(
                all=True, filters={"name": container_name}
            )
            for c in matches:
                if c.name == container_name or c.name == f"/{container_name}":
                    if c.id not in ids_to_remove:
                        ids_to_remove.append(c.id)
                    try:
                        c.remove(force=True)
                        logging.info(
                            f"Force-removed conflicting container '{container_name}' "
                            f"(id={c.short_id}) via list fallback."
                        )
                        return
                    except Exception as e:
                        logging.warning(
                            f"list-based remove failed for '{container_name}': {e}"
                        )
        except Exception as e:
            logging.warning(f"Container list fallback failed for '{container_name}': {e}")

        # Try removing by container ID via SDK API — avoids CLI/daemon routing confusion.
        for cid in ids_to_remove:
            try:
                self.client.api.remove_container(cid, force=True)
                logging.info(
                    f"Force-removed conflicting container '{container_name}' "
                    f"(id={cid[:12]}) via SDK API."
                )
                return
            except docker.errors.NotFound:
                pass  # Already gone
            except Exception as e:
                logging.debug(f"SDK API remove failed for {cid[:12]}: {e}")

        # Try removing by container ID via CLI — fallback for when SDK API fails.
        for cid in ids_to_remove:
            try:
                # Explicitly kill first so the daemon isn't waiting for a graceful stop
                # before it can release GPU/filesystem resources and complete rm.
                subprocess.run(
                    ["docker", "kill", cid],
                    capture_output=True, text=True, timeout=10,
                    env=self._docker_cli_env(),
                )
            except Exception:
                pass
            try:
                result = subprocess.run(
                    ["docker", "rm", "-f", cid],
                    capture_output=True, text=True, timeout=30,
                    env=self._docker_cli_env(),
                )
                if result.returncode == 0:
                    # Verify the name slot is actually freed — CLI exits 0 even for zombie
                    # containers whose name persists in Docker's registry after a containerd crash.
                    try:
                        self.client.api.inspect_container(container_name)
                        logging.debug(
                            f"CLI rm exited 0 for {cid[:12]} but name slot still reserved — zombie state"
                        )
                    except docker.errors.NotFound:
                        logging.info(
                            f"Force-removed conflicting container '{container_name}' "
                            f"(id={cid[:12]}) via docker CLI (by ID)."
                        )
                        return
                    except Exception:
                        logging.info(
                            f"Force-removed conflicting container '{container_name}' "
                            f"(id={cid[:12]}) via docker CLI (by ID)."
                        )
                        return
                else:
                    logging.warning(
                        f"docker rm -f {cid[:12]} exited {result.returncode}: {result.stderr.strip()}"
                    )
            except Exception as e:
                logging.warning(f"docker rm -f by ID subprocess failed for '{container_name}': {e}")

        # Last resort: remove by name via CLI
        try:
            result = subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True, text=True, timeout=30,
                env=self._docker_cli_env(),
            )
            if result.returncode == 0:
                try:
                    self.client.api.inspect_container(container_name)
                    logging.debug("CLI rm by name exited 0 but name slot still reserved — zombie state")
                except docker.errors.NotFound:
                    logging.info(
                        f"Force-removed conflicting container '{container_name}' via docker CLI (by name)."
                    )
                    return
                except Exception:
                    logging.info(
                        f"Force-removed conflicting container '{container_name}' via docker CLI (by name)."
                    )
                    return
            else:
                logging.warning(
                    f"docker rm -f '{container_name}' exited {result.returncode}: {result.stderr.strip()}"
                )
        except Exception as e:
            logging.warning(f"docker rm -f subprocess failed for '{container_name}': {e}")

        # Prune stopped containers to release zombie name slots.
        try:
            pruned = self.client.containers.prune()
            deleted = pruned.get("ContainersDeleted") or []
            logging.info(
                f"Pruned {len(deleted)} stopped container(s) to release zombie name "
                f"slot for '{container_name}'."
            )
            try:
                self.client.api.inspect_container(container_name)
            except docker.errors.NotFound:
                return  # Prune freed the name slot
            except Exception:
                return
        except Exception as e:
            logging.warning(f"Container prune failed for '{container_name}': {e}")

        # Final escalation: restart Docker daemon to flush the name registry.
        # This is necessary when a containerd crash leaves a zombie container whose
        # name slot survives all other removal attempts.
        logging.warning(
            f"Container '{container_name}' is stuck in zombie state. "
            f"Restarting Docker daemon to flush name registry..."
        )
        self._restart_docker_in_wsl()
        # Caller will retry container creation after this returns.

    def _wait_for_container_id_gone(self, container_id: str, timeout: float = 20.0) -> bool:
        """Poll via SDK inspect until the container ID truly disappears.

        Uses the SDK (same daemon as containers.run) rather than CLI subprocess
        to avoid cross-daemon confusion on Windows (Docker Desktop vs WSL daemon).
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                info = self.client.api.inspect_container(container_id)
                state = info.get("State", {}).get("Status", "unknown")
                logging.debug(
                    f"Container {container_id[:12]} still in state '{state}', waiting..."
                )
                time.sleep(0.5)
            except docker.errors.NotFound:
                return True
            except Exception:
                # Transient SDK connection error — keep polling
                time.sleep(1)
        return False

    def _wait_for_container_removal(
        self, container_name: str, container_id: Optional[str] = None, timeout: float = 20.0
    ) -> bool:
        """Poll until the container name is fully released. Returns True if freed.

        When container_id is supplied we poll by ID (most reliable) and fall back
        to name-based checks as a secondary signal.
        """
        if container_id:
            gone = self._wait_for_container_id_gone(container_id, timeout=timeout)
            if not gone:
                return False
            # Name slot may outlast ID removal on Docker Desktop / WSL2.
            # Poll by name via SDK until the slot is confirmed free (up to 10 extra seconds).
            name_deadline = time.monotonic() + 10.0
            while time.monotonic() < name_deadline:
                try:
                    self.client.api.inspect_container(container_name)
                    time.sleep(0.5)
                except docker.errors.NotFound:
                    time.sleep(0.3)  # brief final stabilisation
                    return True
                except Exception:
                    time.sleep(0.5)
            logging.warning(
                f"Container name '{container_name}' still occupied 10 s after ID removal."
            )
            return False

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._get_existing_container(container_name) is None:
                try:
                    matches = self.client.containers.list(
                        all=True, filters={"name": container_name}
                    )
                    still_there = [
                        c for c in matches
                        if c.name == container_name or c.name == f"/{container_name}"
                    ]
                    if not still_there:
                        time.sleep(1.5)
                        return True
                except Exception:
                    pass
            time.sleep(0.5)
        return False

    def _force_pull_image(self, image_name: str) -> None:
        """Pull an image unconditionally, bypassing the 'already present' check.

        Used after a layer-extraction 500 to repair damaged cached layers.
        """
        from .docker_progress_logger import stream_pull_with_progress

        logging.info(f"Force-pulling '{image_name}' to repair layer cache...")
        try:
            stream_pull_with_progress(
                self.client,
                image_name,
                throttle_interval=0.5,
                platform=self._get_pull_platform(),
            )
            logging.info(f"Force-pull of '{image_name}' complete.")
        except Exception as e:
            logging.warning(f"Force-pull of '{image_name}' failed: {e}")

    def _restart_docker_in_wsl(self, wait_timeout: float = 90.0) -> bool:
        """Restart the Docker daemon inside the OpenFork WSL distro.

        After a layer extraction error (grpc: the client connection is closing),
        Docker's containerd backend gets into an inconsistent state — zombie
        containers occupy name slots but are invisible to inspect/rm. Restarting
        the daemon is the only reliable way to flush the name registry and restore
        a clean state.

        Returns True if the daemon is reachable after the restart.
        """
        import sys

        if sys.platform != "win32":
            return False

        distro = os.environ.get("OPENFORK_WSL_DISTRO")
        if not distro:
            logging.warning("Cannot restart Docker in WSL: OPENFORK_WSL_DISTRO not set.")
            return False

        logging.info(f"Restarting Docker daemon in WSL distro '{distro}' to clear inconsistent state...")
        try:
            result = subprocess.run(
                [
                    "wsl", "-d", distro, "--user", "root", "--",
                    "env", "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                    "bash", "-c",
                    "systemctl restart docker 2>/dev/null || service docker restart 2>/dev/null || true",
                ],
                capture_output=True, text=True, timeout=60,
            )
            logging.info(
                f"Docker restart command exited {result.returncode}"
                + (f": {result.stderr.strip()}" if result.stderr.strip() else "")
            )
        except Exception as e:
            logging.warning(f"Docker restart command failed: {e}")

        # Wait for daemon to become reachable again
        logging.info("Waiting for Docker daemon to come back online...")
        deadline = time.monotonic() + wait_timeout
        while time.monotonic() < deadline:
            time.sleep(3)
            try:
                self._refresh_client_connection()
                self.client.ping()
                logging.info("Docker daemon is back online after restart.")
                return True
            except Exception:
                pass

        logging.error(f"Docker daemon did not come back online within {wait_timeout}s after restart.")
        return False

    def set_docker_image_map(self, image_map: dict):
        if image_map:
            logging.info("Setting dynamic Docker image map.")
            safe_image_map = {}
            for service_type, image_name in image_map.items():
                if self._is_allowed_openfork_image(image_name):
                    safe_image_map[service_type] = image_name
                else:
                    logging.warning(
                        "Rejecting Docker image for service '%s': image is not in the OpenFork allowlist",
                        service_type,
                    )
            self.docker_image_map = safe_image_map
        else:
            logging.warning(
                "Dynamic Docker image map is empty. Using fallback static map."
            )

    @staticmethod
    def _is_allowed_openfork_image(image_name: str) -> bool:
        if not isinstance(image_name, str):
            return False
        return bool(
            re.match(
                r"^beschiak/openfork-[a-z0-9._-]+(?::[a-z0-9._-]+|@sha256:[a-f0-9]{64})$",
                image_name,
                re.IGNORECASE,
            )
        )

    def set_services_config(self, services_config: dict):
        if services_config:
            logging.info("Setting services configuration.")
            self.services_config = services_config

    def get_default_ports(self, service_type: str) -> dict:
        config = self.services_config.get(service_type, {})
        port = config.get("port", 8188)  # Default to ComfyUI port
        return {f"{port}/tcp": ("127.0.0.1", port)}

    def get_image_name(self, service_type: str) -> str:
        image = self.docker_image_map.get(service_type)
        if not image:
            raise ValueError(
                f"No Docker image configured for service type '{service_type}'"
            )
        if not self._is_allowed_openfork_image(image):
            raise ValueError(
                f"Refusing unsafe Docker image configured for service type '{service_type}'"
            )
        return image

    def get_container_name(self, service_type: str) -> str:
        return f"dgn-client-{service_type}"

    def pull_image(
        self,
        image_name: str,
        shutdown_event: threading.Event = None,
        service_type: str = None,
        emit_pull_complete: bool = True,
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

                pull_platform = self._get_pull_platform()
                stream_pull_with_progress(
                    self.client,
                    image_name,
                    throttle_interval=0.5,
                    shutdown_event=shutdown_event,
                    service_type=service_type,
                    emit_complete=emit_pull_complete,
                    platform=pull_platform,
                )

                if not self._image_is_registered(image_name):
                    logging.warning(
                        "Docker SDK pull stream ended for '%s' but the image was "
                        "not registered locally. Retrying once via Docker CLI.",
                        image_name,
                    )
                    self._run_docker_cli_pull(
                        image_name,
                        pull_platform,
                        shutdown_event=shutdown_event,
                    )

                if not self._image_is_registered(image_name):
                    raise RuntimeError(
                        f"Docker pull completed for {image_name}, but the image "
                        "was not registered locally."
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

    def _copy_pre_start_files(
        self,
        container,
        pre_start_copies: list[tuple[str, str]],
        shutdown_event: threading.Event = None,
    ) -> None:
        if not pre_start_copies:
            return

        copy_shutdown_event = shutdown_event or threading.Event()
        container_ref = getattr(container, "name", None) or getattr(container, "id", "")

        for source_on_host, dest_in_container in pre_start_copies:
            if self._use_api_file_copy:
                copy_file_to_container_api(
                    container,
                    source_on_host,
                    dest_in_container,
                    copy_shutdown_event,
                )
            else:
                docker_cp(
                    source_on_host,
                    f"{container_ref}:{dest_in_container}",
                    copy_shutdown_event,
                )

    def _create_copy_and_start_container(
        self,
        run_kwargs: dict,
        pre_start_copies: list[tuple[str, str]],
        shutdown_event: threading.Event = None,
    ):
        """Create a container, copy files into it while stopped, then start it.

        This is used for Wan2GP so the HTTP server runs as PID 1 while still
        allowing the client to inject the latest server wrapper before startup.
        If model loading OOMs, Docker marks the container exited/OOMKilled and
        the existing crash monitor can requeue the job.
        """
        create_kwargs = dict(run_kwargs)
        create_kwargs.pop("detach", None)
        container_name = create_kwargs.get("name")

        container = self.client.containers.create(**create_kwargs)
        try:
            self._copy_pre_start_files(container, pre_start_copies, shutdown_event)
        except Exception:
            try:
                container.remove(force=True)
                if container_name:
                    self._wait_for_container_removal(
                        container_name,
                        container_id=getattr(container, "id", None),
                        timeout=30.0,
                    )
            except Exception:
                pass
            raise

        container.start()
        return container

    def run_container(
        self,
        service_type: str,
        ports: dict = None,
        force_restart: bool = True,
        command: list = None,
        entrypoint: list = None,
        environment: dict = None,
        pre_start_copies: list[tuple[str, str]] = None,
        shutdown_event: threading.Event = None,
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
                removed_id = container.id
                container.remove(force=True)
                self._wait_for_container_removal(
                    container_name, container_id=removed_id, timeout=30.0
                )
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
                    removed_id = container.id
                    container.remove(force=True)
                    self._wait_for_container_removal(
                        container_name, container_id=removed_id, timeout=30.0
                    )

        except docker.errors.NotFound:
            matches = self._list_containers_by_name(container_name)
            if matches:
                for c in matches:
                    try:
                        c.remove(force=True)
                        logging.info(
                            f"Removed ghost container '{container_name}' "
                            f"(id={c.short_id}) via list fallback."
                        )
                    except Exception as cleanup_err:
                        logging.debug(
                            f"Could not remove ghost container '{container_name}': {cleanup_err}"
                        )
            else:
                logging.info(
                    f"No existing container named '{container_name}' found. Proceeding to create a new one."
                )
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
            if entrypoint:
                run_kwargs["entrypoint"] = entrypoint
            if environment:
                run_kwargs["environment"] = environment
            max_attempts = 5
            last_missing_conflict_id = None
            missing_conflict_count = 0
            zombie_restart_attempted = False
            for attempt in range(1, max_attempts + 1):
                try:
                    if pre_start_copies:
                        self._create_copy_and_start_container(
                            run_kwargs, pre_start_copies, shutdown_event
                        )
                    else:
                        self.client.containers.run(**run_kwargs)
                    logging.info(f"Container '{container_name}' started successfully.")
                    return
                except Exception as e:
                    is_409 = (
                        isinstance(e, docker.errors.APIError)
                        and getattr(e, "status_code", None) == 409
                    )
                    is_layer_error = self._is_layer_extraction_error(e)

                    if attempt < max_attempts and (
                        self._is_transient_transport_error(e) or is_409 or is_layer_error
                    ):
                        if is_layer_error:
                            logging.warning(
                                f"Layer extraction error creating '{container_name}' "
                                f"(attempt {attempt}/{max_attempts}). "
                                f"Restarting Docker daemon to recover from containerd crash."
                            )
                            # The apply-layer-error crashes Docker's containerd backend,
                            # leaving zombie containers and taking the daemon offline.
                            # Restart is the only reliable recovery; after it completes
                            # we force-pull to repair any damaged layer cache.
                            daemon_back = self._restart_docker_in_wsl()
                            if daemon_back:
                                try:
                                    self.client.containers.prune()
                                    logging.info("Pruned stopped containers after daemon restart.")
                                except Exception:
                                    pass
                                self._force_pull_image(image_name)
                            else:
                                # Daemon didn't come back via WSL restart; try reconnect
                                self._refresh_client_connection()
                                self._force_pull_image(image_name)
                            continue
                        if is_409:
                            _id_match = re.search(
                                r'already in use by container "([0-9a-f]{64})"', str(e)
                            )
                            _conflict_id = _id_match.group(1) if _id_match else None

                            # The transport timeout on the previous attempt may mean Docker
                            # finished creating/starting the container after our connection
                            # dropped. Check its state before deciding to remove it.
                            _conflict_state = None
                            if _conflict_id:
                                try:
                                    _info = self.client.api.inspect_container(_conflict_id)
                                    _conflict_state = _info.get("State", {}).get("Status")
                                except docker.errors.NotFound:
                                    _conflict_state = "missing"
                                except Exception:
                                    pass  # _conflict_state stays None (daemon busy)

                            if _conflict_state in ("running", "restarting"):
                                logging.info(
                                    f"Container '{container_name}' is already running "
                                    f"(id={_conflict_id[:12]}). Reusing it."
                                )
                                return

                            if _conflict_state == "missing" and _conflict_id:
                                if _conflict_id == last_missing_conflict_id:
                                    missing_conflict_count += 1
                                else:
                                    last_missing_conflict_id = _conflict_id
                                    missing_conflict_count = 1

                                if (
                                    missing_conflict_count >= 2
                                    and not zombie_restart_attempted
                                ):
                                    logging.warning(
                                        f"Container name '{container_name}' is still "
                                        f"reserved by invisible container "
                                        f"{_conflict_id[:12]} after removal. "
                                        "Restarting Docker daemon before retrying."
                                    )
                                    zombie_restart_attempted = True
                                    if not self._restart_docker_in_wsl():
                                        self._refresh_client_connection()
                                    try:
                                        self.client.containers.prune()
                                    except Exception:
                                        pass
                                    time.sleep(3)
                                    continue

                            if _conflict_state in ("created", None) and _conflict_id:
                                if _conflict_state == "created" and pre_start_copies:
                                    try:
                                        _created = self.client.containers.get(
                                            _conflict_id
                                        )
                                        self._copy_pre_start_files(
                                            _created,
                                            pre_start_copies,
                                            shutdown_event,
                                        )
                                        _created.start()
                                        logging.info(
                                            f"Started previously created container "
                                            f"'{container_name}' after pre-start copy."
                                        )
                                    except Exception as start_error:
                                        logging.debug(
                                            f"Could not prepare/start created "
                                            f"container '{container_name}': "
                                            f"{start_error}"
                                        )
                                # Either Docker is still initialising the container (overlayfs /
                                # GPU hooks for large images can take many minutes on WSL2), or
                                # the inspect call itself failed because the daemon is overloaded.
                                # In both cases poll via CLI instead of removing and retrying.
                                logging.info(
                                    f"Container '{container_name}' is still initialising "
                                    f"(state={_conflict_state!r}) — polling up to 5 minutes..."
                                )
                                _poll_end = time.monotonic() + 300
                                while time.monotonic() < _poll_end:
                                    time.sleep(5)
                                    try:
                                        _pi = self.client.api.inspect_container(_conflict_id)
                                        _ps = _pi.get("State", {}).get("Status", "")
                                        if _ps in ("running", "restarting", "exited", "dead"):
                                            _conflict_state = _ps
                                            break
                                    except docker.errors.NotFound:
                                        _conflict_state = "missing"
                                        break
                                    except Exception:
                                        continue  # SDK error — keep polling
                                if _conflict_state in ("running", "restarting"):
                                    logging.info(
                                        f"Container '{container_name}' started. Reusing it."
                                    )
                                    return

                            logging.warning(
                                f"Container name conflict for "
                                f"'{container_name}' (attempt {attempt}/{max_attempts}), "
                                f"state={_conflict_state!r}. Force-removing."
                            )
                            self._force_remove_by_name(container_name, container_id=_conflict_id)
                            freed = self._wait_for_container_removal(container_name, container_id=_conflict_id)
                            if not freed:
                                logging.warning(
                                    f"Container '{container_name}' still present after "
                                    f"removal — proceeding anyway."
                                )
                            # Docker Desktop / WSL2: the TCP endpoint can still refuse
                            # name creation for a few seconds after inspect says it's
                            # gone. Wait before the next attempt to avoid burning all
                            # retries in <1 s.
                            time.sleep(3)
                            continue

                        logging.warning(
                            f"Transient Docker transport error while starting "
                            f"'{container_name}' (attempt {attempt}/{max_attempts}): {e}"
                        )
                        self._refresh_client_connection()

                        existing = self._get_existing_container(container_name)
                        if existing is None:
                            # SDK lookup failed; retry via API directly before giving up.
                            try:
                                _info = self.client.api.inspect_container(container_name)
                                _api_status = _info.get("State", {}).get("Status", "")
                                if _api_status in ("running", "restarting"):
                                    logging.info(
                                        f"Container '{container_name}' is running "
                                        f"(SDK API check after transport timeout). Reusing it."
                                    )
                                    return
                                if _api_status in ("created", ""):
                                    logging.info(
                                        f"Container '{container_name}' is initialising "
                                        f"(SDK API, state={_api_status!r}) — polling up to 5 minutes..."
                                    )
                                    _poll_end = time.monotonic() + 300
                                    _poll_status = _api_status or "created"
                                    while time.monotonic() < _poll_end:
                                        time.sleep(5)
                                        try:
                                            _ri = self.client.api.inspect_container(container_name)
                                            _ps = _ri.get("State", {}).get("Status", "")
                                            if _ps in ("running", "restarting", "exited", "dead"):
                                                _poll_status = _ps
                                                break
                                        except docker.errors.NotFound:
                                            _poll_status = "missing"
                                            break
                                        except Exception:
                                            continue
                                    if _poll_status in ("running", "restarting"):
                                        logging.info(
                                            f"Container '{container_name}' started. Reusing it."
                                        )
                                        return
                            except docker.errors.NotFound:
                                pass  # Container gone — retry containers.run below
                            except Exception:
                                pass  # SDK error — fall through to retry
                        if existing is not None:
                            # Reload with one retry — the fresh client may need a moment
                            for _reload_try in range(2):
                                try:
                                    existing.reload()
                                    status = existing.status
                                    break
                                except Exception:
                                    if _reload_try == 0:
                                        time.sleep(1)
                                    else:
                                        status = "unknown"

                            if status == "created":
                                if pre_start_copies:
                                    try:
                                        self._copy_pre_start_files(
                                            existing,
                                            pre_start_copies,
                                            shutdown_event,
                                        )
                                    except Exception as copy_error:
                                        logging.warning(
                                            f"Pre-start copy failed for "
                                            f"'{container_name}': {copy_error}. "
                                            "Removing partial container before retry."
                                        )
                                        try:
                                            existing.remove(force=True)
                                            self._wait_for_container_removal(
                                                container_name,
                                                container_id=existing.id,
                                                timeout=30.0,
                                            )
                                        except Exception:
                                            pass
                                        time.sleep(2)
                                        continue
                                # Try to kick off start in case the earlier call never reached
                                # the daemon; idempotent if it's already starting.
                                try:
                                    existing.start()
                                except Exception as start_error:
                                    logging.debug(
                                        f"Could not start existing container "
                                        f"'{container_name}' after reconnect: {start_error}"
                                    )

                                # Large GPU containers (Wan2GP etc.) can take many minutes to
                                # complete Docker initialisation after the HTTP timeout fires.
                                # Poll via CLI (avoids SDK timeout) for up to 5 minutes.
                                logging.info(
                                    f"Container '{container_name}' is still initialising "
                                    f"after transport timeout — polling up to 5 minutes..."
                                )
                                _poll_end = time.monotonic() + 300
                                while time.monotonic() < _poll_end:
                                    time.sleep(5)
                                    try:
                                        _ri = self.client.api.inspect_container(container_name)
                                        _ps = _ri.get("State", {}).get("Status", "")
                                        if _ps in ("running", "restarting", "exited", "dead"):
                                            status = _ps
                                            break
                                    except docker.errors.NotFound:
                                        status = "missing"
                                        break
                                    except Exception:
                                        continue  # SDK error — keep polling

                            if status in ("running", "restarting"):
                                logging.info(
                                    f"Container '{container_name}' appears to have started "
                                    "despite the transport timeout. Reusing it."
                                )
                                return

                            if status in ("created", "exited", "dead", "unknown", "missing"):
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
            removed_id = container.id
            container.remove(force=True)
            # Wait for the name slot to be fully released so the next run_container()
            # call doesn't hit a 409 conflict on the same container name.
            # 15 s is generous but bounded; the slot is normally free within 1-2 s.
            self._wait_for_container_removal(
                container_name, container_id=removed_id, timeout=15.0
            )
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
