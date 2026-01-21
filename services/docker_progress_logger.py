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
    
    # Regex patterns to parse CLI output lines like:
    # "8e3ba11ec2a2: Downloading [=>      ]  10.5MB/200MB" 
    # "8e3ba11ec2a2: Downloading  1.234kB/5.678kB"
    # "8e3ba11ec2a2: Downloading [==>                                                ]  123.4MB/1.234GB"
    # The key is to capture the layer ID, current size, and total size
    # 
    # Pattern breakdown:
    # - ([a-f0-9]+) - layer ID (hex string)
    # - :\s+ - colon followed by whitespace
    # - Downloading\s+ - "Downloading" keyword followed by whitespace
    # - (?:\[.*?\]\s*)? - optional progress bar in brackets (non-capturing)
    # - (\d+\.?\d*)\s*([KMGT]?B) - current size with unit
    # - \s*/\s* - slash separator with optional whitespace
    # - (\d+\.?\d*)\s*([KMGT]?B) - total size with unit
    progress_pattern = re.compile(
        r'([a-f0-9]+):\s+Downloading\s+(?:\[.*?\]\s*)?(\d+\.?\d*)\s*([KMGT]?B)\s*/\s*(\d+\.?\d*)\s*([KMGT]?B)',
        re.IGNORECASE
    )
    
    # Alternative pattern for when there's no progress bar brackets
    # "8e3ba11ec2a2: Pulling fs layer" etc - these don't have size info
    alt_progress_pattern = re.compile(
        r'([a-f0-9]+):\s+Downloading\s+(\d+\.?\d*)\s*([KMGT]?B)\s*/\s*(\d+\.?\d*)\s*([KMGT]?B)',
        re.IGNORECASE
    )
    
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
        import threading
        import queue
        
        # Run docker pull command with --progress=plain for parseable text output
        # Modern Docker uses a fancy terminal UI by default that we can't parse
        cmd = ["docker", "pull", "--progress=plain", image_name]
        logging.info(f"Running command: {' '.join(cmd)}")
        
        # Use shell=True on Windows for proper output handling
        import platform
        use_shell = platform.system() == "Windows"
        cmd_str = ' '.join(cmd) if use_shell else cmd
        
        process = subprocess.Popen(
            cmd_str if use_shell else cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            shell=use_shell,
        )
        
        logging.info(f"Subprocess started with PID {process.pid}")
        
        lines_received = 0
        output_queue = queue.Queue()
        
        def reader_thread():
            """Thread to read stdout and put lines in queue."""
            try:
                for line in process.stdout:
                    output_queue.put(line)
            except Exception as e:
                logging.error(f"Reader thread error: {e}")
            finally:
                output_queue.put(None)  # Signal end of output
        
        # Start reader thread
        reader = threading.Thread(target=reader_thread, daemon=True)
        reader.start()
        
        # Process lines from queue
        while True:
            try:
                raw_line = output_queue.get(timeout=0.5)
                if raw_line is None:
                    break
                    
                # Handle carriage returns (Docker progress updates)
                for part in raw_line.split('\r'):
                    line = part.strip()
                    if not line:
                        continue
                    
                    # Log first few lines to diagnose if output is being captured
                    lines_received += 1
                    if lines_received <= 5:
                        logging.info(f"Docker CLI line {lines_received}: {repr(line)[:100]}")
                    
                    # Log raw line for debugging - ENABLED to diagnose progress issues
                    # Look for "Downloading" in the line to reduce noise (show first few only)
                    if "Downloading" in line and len(logger.layers) < 3:
                        logging.info(f"Docker progress line sample: {repr(line)[:100]}")
                    
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
                    if not match:
                        # Try alternative pattern without brackets
                        match = alt_progress_pattern.search(line)
                    
                    if match:
                        layer_id = match.group(1)
                        # New pattern: group 2 = current number, group 3 = current unit
                        # group 4 = total number, group 5 = total unit
                        current_str = f"{match.group(2)}{match.group(3)}"
                        total_str = f"{match.group(4)}{match.group(5)}"
                        
                        current_bytes = parse_size(current_str)
                        total_bytes = parse_size(total_str)
                        
                        # Log when we successfully parse a progress update for the first time
                        if len(logger.layers) == 1 and total_bytes > 0:
                            logging.info(f"Docker download progress parsing started: layer={layer_id}, total={total_str}")
                        
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
                    elif "Downloading" in line:
                        # Regex didn't match a "Downloading" line - log for debugging
                        logging.warning(f"Failed to parse Downloading line: {repr(line)}")
            except queue.Empty:
                # Check if process has ended
                if process.poll() is not None:
                    break
                continue
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
