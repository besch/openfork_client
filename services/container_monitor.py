"""
Container Monitor

Monitors Docker container health and detects unexpected crashes (OOM, etc.).
When a container crash is detected, it marks the job as failed and allows
the DGN client to continue processing new jobs.
"""

import threading
import logging
import time
from typing import Optional, Callable


class ContainerMonitor:
    """Monitors a Docker container for unexpected crashes or OOM kills."""
    
    def __init__(
        self,
        docker_client,
        container_name: str,
        job_id: str,
        on_container_crash: Callable[[str, str, Optional[str]], None],
        shutdown_event: threading.Event
    ):
        """
        Initialize container monitor.
        
        Args:
            docker_client: Docker client instance
            container_name: Name of container to monitor
            job_id: ID of job being processed
            on_container_crash: Callback(job_id, reason) called when crash detected
            shutdown_event: Event to stop monitoring
        """
        self.docker_client = docker_client
        self.container_name = container_name
        self.job_id = job_id
        self.on_container_crash = on_container_crash
        self.shutdown_event = shutdown_event
        self.monitor_thread = None
        self._stop_monitoring = threading.Event()
        
    def start(self):
        """Start monitoring the container in a background thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            logging.warning(f"Container monitor for {self.container_name} already running")
            return
            
        self._stop_monitoring.clear()
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name=f"ContainerMonitor-{self.container_name}"
        )
        self.monitor_thread.start()
        logging.info(f"Started container monitor for {self.container_name} (job {self.job_id})")
    
    def stop(self):
        """Stop monitoring the container."""
        self._stop_monitoring.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            logging.debug(f"Stopped container monitor for {self.container_name}")
    
    def _monitor_loop(self):
        """Main monitoring loop - checks container status every 2 seconds."""
        try:
            while not self._stop_monitoring.is_set() and not self.shutdown_event.is_set():
                try:
                    container = self.docker_client.containers.get(self.container_name)
                    status = container.status
                    
                    # Check if container has exited unexpectedly
                    if status in ('exited', 'dead'):
                        # Get exit code and OOM status
                        container.reload()  # Refresh container state
                        exit_code = container.attrs.get('State', {}).get('ExitCode', -1)
                        oom_killed = container.attrs.get('State', {}).get('OOMKilled', False)
                        
                        # Determine crash reason
                        if oom_killed:
                            reason = "Container killed by OOM (Out of Memory)"
                            logging.error(
                                f"Container {self.container_name} was OOM killed! "
                                f"Job {self.job_id} failed. Exit code: {exit_code}"
                            )
                        elif exit_code != 0:
                            reason = f"Container exited unexpectedly with code {exit_code}"
                            logging.error(
                                f"Container {self.container_name} crashed! "
                                f"Job {self.job_id} failed. Exit code: {exit_code}"
                            )
                        else:
                            # Exit code 0 means clean shutdown - not a crash
                            logging.debug(f"Container {self.container_name} exited cleanly (code 0)")
                            return
                        
                        logs_tail = None
                        # Get last 100 lines of logs for debugging and failure persistence
                        try:
                            logs_tail = container.logs(tail=100).decode(
                                "utf-8",
                                errors="replace",
                            )
                            logging.error(
                                f"Last logs from {self.container_name}:\n{logs_tail}"
                            )
                        except Exception as log_err:
                            logging.debug(f"Could not retrieve container logs: {log_err}")
                        
                        # Notify caller of crash
                        if self.on_container_crash:
                            self.on_container_crash(self.job_id, reason, logs_tail)
                        
                        return  # Exit monitoring loop
                        
                except Exception as e:
                    if "No such container" in str(e):
                        # Container was removed - this might be normal cleanup
                        logging.debug(f"Container {self.container_name} no longer exists")
                        return
                    else:
                        # Log but don't crash the monitor
                        logging.debug(f"Error monitoring container {self.container_name}: {e}")
                
                # Check every 2 seconds
                self._stop_monitoring.wait(2.0)
                
        except Exception as e:
            logging.error(f"Container monitor crashed: {e}", exc_info=True)
