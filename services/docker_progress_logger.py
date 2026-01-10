"""
Docker pull progress logger with throttling.

This module provides utilities to parse Docker pull progress events
and emit structured, throttled progress updates to avoid log spam.
"""
import json
import time
import sys


class DockerPullProgressLogger:
    """
    Parses Docker pull progress stream and emits throttled progress updates.
    
    Docker pull progress comes as a stream of JSON objects, one per layer.
    This class aggregates layer progress into an overall percentage and
    emits updates at a throttled rate.
    """
    
    def __init__(self, image_name: str, throttle_interval: float = 0.5):
        self.image_name = image_name
        self.throttle_interval = throttle_interval
        self.last_emit_time = 0
        self.layers = {}  # layer_id -> {"current": bytes, "total": bytes}
        self.last_progress = -1
        
    def parse_progress_event(self, event: dict) -> None:
        """Parse a single progress event from Docker pull stream."""
        status = event.get("status", "")
        layer_id = event.get("id", "")
        progress_detail = event.get("progressDetail", {})
        
        if not layer_id:
            return
            
        # Track layer download progress
        if status in ("Downloading", "Extracting"):
            current = progress_detail.get("current", 0)
            total = progress_detail.get("total", 0)
            if total > 0:
                self.layers[layer_id] = {"current": current, "total": total, "status": status}
        elif status == "Download complete" or status == "Pull complete":
            if layer_id in self.layers:
                self.layers[layer_id]["current"] = self.layers[layer_id]["total"]
                
    def calculate_overall_progress(self) -> int:
        """Calculate overall progress as a percentage (0-100)."""
        if not self.layers:
            return 0
            
        total_bytes = sum(layer["total"] for layer in self.layers.values())
        current_bytes = sum(layer["current"] for layer in self.layers.values())
        
        if total_bytes == 0:
            return 0
            
        return int((current_bytes / total_bytes) * 100)
    
    def get_current_status(self) -> str:
        """Get the most relevant current status."""
        downloading = sum(1 for l in self.layers.values() if l.get("status") == "Downloading")
        extracting = sum(1 for l in self.layers.values() if l.get("status") == "Extracting")
        
        # If we are at 100%, we are likely waiting for docker to finish internal processing
        if self.calculate_overall_progress() >= 100:
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
        if not force and not self.should_emit():
            return
            
        progress = self.calculate_overall_progress()
        
        # Only emit if progress actually changed (avoid duplicate messages)
        if progress == self.last_progress and not force:
            return
            
        self.last_progress = progress
        self.last_emit_time = time.time()
        
        message = {
            "type": "DOCKER_PULL_PROGRESS",
            "payload": {
                "image": self.image_name,
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
                "image": self.image_name
            }
        }
        print(json.dumps(message), flush=True)
        
    def emit_start(self) -> None:
        """Emit a start message."""
        message = {
            "type": "DOCKER_PULL_START",
            "payload": {
                "image": self.image_name
            }
        }
        print(json.dumps(message), flush=True)


def stream_pull_with_progress(docker_client, image_name: str, throttle_interval: float = 0.5):
    """
    Pull a Docker image with progress streaming.
    
    Args:
        docker_client: Docker client instance (docker.from_env())
        image_name: Full image name (e.g., "camenduru/wan21:latest")
        throttle_interval: Minimum seconds between progress updates
        
    Returns:
        The pulled image object
    """
    logger = DockerPullProgressLogger(image_name, throttle_interval)
    logger.emit_start()
    
    # Use the low-level API to get streaming progress
    api = docker_client.api
    
    # Parse repository and tag
    if ":" in image_name:
        repository, tag = image_name.rsplit(":", 1)
    else:
        repository = image_name
        tag = "latest"
    
    for event in api.pull(repository, tag=tag, stream=True, decode=True):
        logger.parse_progress_event(event)
        logger.emit_progress()
        
    # Emit final 100% progress which will now show "Processing"
    # This covers the delay between pull stream finishing and images.get() returning
    logger.emit_progress(force=True)
    
    # This call can take some time as Docker finalizes the image layers
    image = docker_client.images.get(image_name)
    
    # Now we are truly done
    logger.emit_complete()
    
    return image
