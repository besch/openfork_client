"""
Docker pull progress logger with throttling.

This module provides utilities to parse Docker pull progress events
and emit structured, throttled progress updates to avoid log spam.
"""
import json
import time
import sys
import threading


class DownloadInterruptedError(Exception):
    """Exception raised when a download is explicitly cancelled by the user."""
    pass


class DockerPullProgressLogger:
    """
    Parses Docker pull progress stream and emits throttled progress updates.
    
    Docker pull progress comes as a stream of JSON objects, one per layer.
    This class aggregates layer progress into an overall percentage and
    emits updates at a throttled rate.
    """
    
    def __init__(self, image_name: str, throttle_interval: float = 0.5, service_type: str = None):
        self.image_name = image_name
        self.throttle_interval = throttle_interval
        self.service_type = service_type
        self.last_emit_time = 0
        self.layers = {}  # layer_id -> {"current": bytes, "total": bytes}
        self.last_progress = -1
        
    def parse_progress_event(self, event: dict) -> bool:
        """Parse a single progress event from Docker pull stream.
        
        Returns:
            True if progress should be emitted immediately (layer completed)
        """
        status = event.get("status", "")
        layer_id = event.get("id", "")
        progress_detail = event.get("progressDetail", {})
        
        if not layer_id:
            return False

        # Initialize layer tracking if not already present
        if layer_id not in self.layers:
            self.layers[layer_id] = {"current": 0, "total": 0, "status": status, "complete": False}
            
        # Track layer download progress
        if status in ("Downloading", "Extracting"):
            current = progress_detail.get("current", 0)
            total = progress_detail.get("total", 0)
            if total > 0:
                self.layers[layer_id].update({"current": current, "total": total, "status": status})
                return False  # Normal progress, use throttle
        elif status in ("Download complete", "Pull complete", "Already exists") or "complete" in status.lower():
            # Ensure we mark it as complete
            if self.layers[layer_id]["total"] == 0:
                 self.layers[layer_id]["total"] = 1
                 self.layers[layer_id]["current"] = 1
            else:
                 self.layers[layer_id]["current"] = self.layers[layer_id]["total"]
            self.layers[layer_id]["complete"] = True
            self.layers[layer_id]["status"] = status
            return True  # Force emit when layer completes
        
        return False
                
    def calculate_overall_progress(self) -> int:
        """Calculate overall progress as a percentage (0-100)."""
        if not self.layers:
            return 0
            
        total_bytes = sum(layer["total"] for layer in self.layers.values())
        current_bytes = sum(layer["current"] for layer in self.layers.values())
        unknown_incomplete_layers = [
            layer
            for layer in self.layers.values()
            if not layer.get("complete") and layer.get("total", 0) <= 0
        ]
        
        # If we have any bytes info (at least one layer has a known total size > 0),
        # use byte-based progress.
        if total_bytes > 0:
            progress = int((current_bytes / total_bytes) * 100)
            if unknown_incomplete_layers:
                known_layer_count = len(self.layers) - len(unknown_incomplete_layers)
                layer_cap = int((known_layer_count / len(self.layers)) * 100)
                progress = min(progress, layer_cap)
            return min(progress, 100)
            
        # Fallback to counting layers if NO layers have reported size yet
        # (happens with 'Already exists' layers or initial 'Pulling' state)
        completed_layers = sum(1 for layer in self.layers.values() if layer.get("complete"))
        
        # SENSITIVE: If we have layers that are NOT complete and haven't reported size yet,
        # don't cap at 100% based on layers alone. This avoids the 100% spike at startup.
        if completed_layers < len(self.layers):
            return int((completed_layers / len(self.layers)) * 100)
        elif len(self.layers) > 0:
            # If all known layers are complete but total_bytes is 0,
            # it might be that Docker hasn't reported ALL layers yet.
            # We return 99% instead of 100% to avoid premature completion UI spikes
            # unless we're actually done (handled by emit_complete).
            return 99 if any(l.get("status") in ("Downloading", "Extracting") for l in self.layers.values()) else 100
            
        return 0
    
    def get_current_status(self) -> str:
        """Get the most relevant current status."""
        downloading = sum(1 for l in self.layers.values() if l.get("status") == "Downloading")
        extracting = sum(1 for l in self.layers.values() if l.get("status") == "Extracting")
        
        # If all known layers completed, Docker is likely finalizing the image locally.
        if self.layers and all(l.get("complete") for l in self.layers.values()):
            return "Processing"
            
        if extracting > 0:
            return "Extracting"
        elif downloading > 0:
            return "Downloading"
        return "Preparing"
        
    def should_emit(self) -> bool:
        """Check if enough time has passed for a new emit."""
        now = time.time()
        return (now - self.last_emit_time) >= self.throttle_interval
        
    def emit_progress(self, force: bool = False) -> None:
        """Emit a progress update if throttle allows or if forced."""
        progress = self.calculate_overall_progress()
        
        # Force emit at 10% milestones (10%, 20%, 30%, etc.) to show progress on fast downloads
        progress_milestone = (progress // 10) * 10
        last_milestone = (self.last_progress // 10) * 10
        crossed_milestone = progress_milestone > last_milestone and progress_milestone > 0
        
        # Emit if:
        # 1. Forced
        # 2. Crossed a 10% milestone
        # 3. Enough time passed AND progress changed
        should_emit_now = force or crossed_milestone or (self.should_emit() and progress != self.last_progress)
        
        if not should_emit_now:
            return
            
        self.last_progress = progress
        self.last_emit_time = time.time()
        
        message = {
            "type": "DOCKER_PULL_PROGRESS",
            "payload": {
                "image": self.image_name,
                "service_type": self.service_type,
                "progress": progress,
                "status": self.get_current_status()
            }
        }
        print(json.dumps(message), flush=True)
        
    def emit_complete(self) -> None:
        """Emit a completion message."""
        message = {
            "type": "DOCKER_PULL_COMPLETE",
            "payload": {
                "image": self.image_name,
                "service_type": self.service_type
            }
        }
        print(json.dumps(message), flush=True)
        
    def emit_start(self) -> None:
        """Emit a start message."""
        message = {
            "type": "DOCKER_PULL_START",
            "payload": {
                "image": self.image_name,
                "service_type": self.service_type
            }
        }
        print(json.dumps(message), flush=True)

    def emit_failed(self, error: str = None) -> None:
        """Emit a failure message."""
        message = {
            "type": "DOCKER_PULL_FAILED",
            "payload": {
                "image": self.image_name,
                "service_type": self.service_type,
                "error": error
            }
        }
        print(json.dumps(message), flush=True)


def stream_pull_with_progress(
    docker_client,
    image_name: str,
    throttle_interval: float = 0.5,
    shutdown_event: threading.Event = None,
    service_type: str = None,
    emit_complete: bool = True,
):
    """
    Pull a Docker image using the Docker Python SDK with streaming progress.
    
    This uses the low-level API's stream=True parameter to get real-time
    progress updates that work reliably on all platforms including Windows.
    """
    import logging
    import json
    
    logger = DockerPullProgressLogger(image_name, throttle_interval, service_type=service_type)
    logger.emit_start()
    
    logging.info(f"Starting Docker pull via SDK for {image_name}")
    
    try:
        # Parse image name into repository and tag
        if ':' in image_name and '/' in image_name.split(':')[-1] is False:
            # Has a tag like "repo/image:tag"
            repository, tag = image_name.rsplit(':', 1)
        elif '@' in image_name:
            # Has a digest like "repo/image@sha256:..."
            repository = image_name
            tag = None
        else:
            # No tag, use 'latest'
            repository = image_name
            tag = 'latest' if ':' not in image_name else image_name.split(':')[-1]
            if ':' in image_name:
                repository = image_name.rsplit(':', 1)[0]
        
        logging.info(f"Pulling repository={repository}, tag={tag}")
        
        # Use the low-level API to stream progress
        # This returns a generator of JSON objects
        lines_received = 0
        logging.info(f"Iteration through Docker pull stream for {image_name} started.")
        
        # We manually iterate to ensure we can catch the shutdown signal as fast as possible
        pull_stream = docker_client.api.pull(repository, tag=tag, stream=True, decode=True)
        
        for chunk in pull_stream:
            if shutdown_event and shutdown_event.is_set():
                logging.info(f"Docker pull for {image_name} cancellation detected in stream loop.")
                # We try to close the stream if possible to release resources
                try:
                    if hasattr(pull_stream, 'close'):
                        pull_stream.close()
                except:
                    pass
                raise DownloadInterruptedError(f"Docker pull for {image_name} cancelled.")

            lines_received += 1
            
            # Log first few chunks for debugging
            if lines_received <= 5:
                logging.info(f"Docker SDK progress chunk {lines_received}: {str(chunk)[:100]}")
            
            # Parse the progress event
            if isinstance(chunk, dict):
                should_force_emit = logger.parse_progress_event(chunk)
                # Force emit when a layer completes to show progress on fast downloads
                logger.emit_progress(force=should_force_emit)
            
        logging.info(f"Docker SDK pull completed. Total chunks: {lines_received}")
        
        # Emit final 100% progress
        logger.emit_progress(force=True)
        
        if emit_complete:
            logger.emit_complete()
            
    except DownloadInterruptedError:
        logging.info(f"Docker pull for {image_name} interrupted by user.")
        logger.emit_failed("cancelled")
        raise
    except Exception as e:
        # Check if it was actually a cancellation that manifested as a different error
        if shutdown_event and shutdown_event.is_set():
            logging.info(f"Docker pull for {image_name} stopped during shutdown/cancellation.")
            logger.emit_failed("cancelled")
        else:
            logging.error(f"Error during Docker SDK pull: {e}", exc_info=True)
            logger.emit_failed(str(e))
        raise

