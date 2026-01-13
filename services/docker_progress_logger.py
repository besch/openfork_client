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
                self.layers[layer_id] = {"current": current, "total": total, "status": status, "complete": False}
        elif status in ("Download complete", "Pull complete", "Already exists"):
            # If it already exists or finished, ensure we count it as 100%
            if layer_id in self.layers:
                self.layers[layer_id]["current"] = self.layers[layer_id]["total"]
                self.layers[layer_id]["complete"] = True
            else:
                # If we didn't see the download start (e.g. Already exists),
                # we don't know the size, but we mark it as a 'completed layer'
                self.layers[layer_id] = {"current": 1, "total": 1, "status": status, "complete": True}
                
    def calculate_overall_progress(self) -> int:
        """Calculate overall progress as a percentage (0-100)."""
        if not self.layers:
            return 0
            
        total_bytes = sum(layer["total"] for layer in self.layers.values())
        current_bytes = sum(layer["current"] for layer in self.layers.values())
        
        # If we have no bytes info (only 'Already exists' layers), 
        # count by number of layers
        if total_bytes == 0:
            completed_layers = sum(1 for layer in self.layers.values() if layer.get("complete"))
            return int((completed_layers / len(self.layers)) * 100) if self.layers else 0
            
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
    Pull a Docker image using the native Docker CLI via subprocess.
    This bypasses docker-py API hangs on Windows.
    """
    import logging
    import json
    import time
    import subprocess
    import sys
    import re
    
    logger = DockerPullProgressLogger(image_name, throttle_interval)
    logger.emit_start()
    
    logging.info(f"Starting Docker pull via CLI for {image_name}")
    
    # Regex to parse CLI output lines like:
    # "8e3ba11ec2a2: Downloading [=>      ]  10.5MB/200MB" or just "Downloading 10.5MB/200MB"
    # capturing id, current, total
    # This regex looks for patterns like: "123abcde: Downloading 12.3MB/45.6MB"
    progress_pattern = re.compile(r'([a-f0-9]+):\s+Downloading\s+.*?((?:\d+\.?\d*)(?:[KMGT]?B))\s*/\s*((?:\d+\.?\d*)(?:[KMGT]?B))', re.IGNORECASE)
    
    # Helper to parse sizes like "10MB", "5.5GB" to bytes
    def parse_size(size_str):
        units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4}
        match = re.search(r'([\d.]+)([KMGT]?B)', size_str.upper())
        if not match:
            return 0
        value = float(match.group(1))
        unit = match.group(2)
        return int(value * units.get(unit, 1))

    try:
        # Run docker pull command
        # unbuffer output if possible
        # We process stdout and stderr together
        process = subprocess.Popen(
            ["docker", "pull", image_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            bufsize=0, # Unbuffered
            shell=False 
        )
        
        logging.info(f"Subprocess started with PID {process.pid}")
        
        # We need to read character by character to handle \r updates
        # or use a loop that can handle partial lines
        buffer = ""
        
        while True:
            # Read character by character to ensure we capture everything 
            # immediately without waiting for buffer fills
            char = process.stdout.read(1)
            if not char and process.poll() is not None:
                break
                
            if char:
                # Decode bytes to string
                try:
                    text = char.decode('utf-8', errors='replace')
                except:
                    continue
                    
                buffer += text
                
                # split by newline or carriage return
                # We need to handle \r because docker updates lines in place
                while '\n' in buffer or '\r' in buffer:
                    if '\n' in buffer and ('\r' not in buffer or buffer.find('\n') < buffer.find('\r')):
                        line, buffer = buffer.split('\n', 1)
                    elif '\r' in buffer:
                        line, buffer = buffer.split('\r', 1)
                    else:
                        break # should not happen given while condition
                        
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Log raw line for debugging (verbose, but necessary now)
                    # logging.debug(f"CLI RAW: {line}")
                    
                    # Check for "Already exists" or "Pull complete"
                    if "Pull complete" in line or "Already exists" in line:
                        parts = line.split(":")
                        if len(parts) > 1:
                            layer_id = parts[0].strip()
                            event = {
                                "status": "Pull complete" if "Pull complete" in line else "Already exists",
                                "id": layer_id,
                                "progressDetail": {}
                            }
                            logger.parse_progress_event(event)
                            logger.emit_progress()
                            
                    # Check for Downloading progress
                    match = progress_pattern.search(line)
                    if match:
                        layer_id = match.group(1)
                        current_str = match.group(2)
                        total_str = match.group(3)
                        
                        current_bytes = parse_size(current_str)
                        total_bytes = parse_size(total_str)
                        
                        event = {
                            "status": "Downloading",
                            "id": layer_id,
                            "progressDetail": {
                                "current": current_bytes,
                                "total": total_bytes
                            }
                        }
                        logger.parse_progress_event(event)
                        logger.emit_progress()
        
        return_code = process.poll()
        
        if return_code != 0:
            err_msg = f"Docker CLI pull failed with code {return_code}"
            logging.error(err_msg)
            raise Exception(err_msg)
            
        logging.info("Docker CLI pull completed successfully")
        
        # Emit final 100% progress
        logger.emit_progress(force=True)
        
        # Verify image exists via client
        logging.info(f"Verifying image {image_name}...")
        image = docker_client.images.get(image_name)
        
        logger.emit_complete()
        return image
            
    except Exception as e:
        logging.error(f"Error during CLI docker pull: {e}", exc_info=True)
        try:
            process.kill()
        except:
            pass
        raise
