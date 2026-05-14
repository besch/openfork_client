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
        self.layers = {}
        self.last_progress = -1
        self.synthetic_interval = max(throttle_interval * 4, 3.0)
        
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

        # Initialize layer tracking if not already present. Docker reports
        # downloading and extracting with independent byte counters for the
        # same layer, so keep those phases separate to avoid UI regressions.
        if layer_id not in self.layers:
            self.layers[layer_id] = {
                "download_current": 0,
                "download_total": 0,
                "download_total_known": False,
                "download_complete": False,
                "extract_current": 0,
                "extract_total": 0,
                "extract_complete": False,
                "status": status,
                "complete": False,
            }
            
        # Track layer download progress
        if status == "Downloading":
            current = progress_detail.get("current", 0)
            total = progress_detail.get("total", 0)
            if total > 0:
                self.layers[layer_id].update(
                    {
                        "download_current": max(
                            current,
                            self.layers[layer_id].get("download_current", 0),
                        ),
                        "download_total": max(
                            total,
                            self.layers[layer_id].get("download_total", 0),
                        ),
                        "download_total_known": True,
                        "status": status,
                    }
                )
                return False  # Normal progress, use throttle
        elif status == "Extracting":
            current = progress_detail.get("current", 0)
            total = progress_detail.get("total", 0)
            if total > 0:
                updates = {
                    "extract_current": max(
                        current,
                        self.layers[layer_id].get("extract_current", 0),
                    ),
                    "extract_total": max(
                        total,
                        self.layers[layer_id].get("extract_total", 0),
                    ),
                    "status": status,
                    "complete": False,
                }
                if self.layers[layer_id].get("download_total", 0) == 0:
                    updates.update(
                        {
                            "download_current": total,
                            "download_total": total,
                            "download_total_known": True,
                            "download_complete": True,
                        }
                    )
                self.layers[layer_id].update(updates)
                return False
        elif status == "Download complete":
            if self.layers[layer_id]["download_total"] == 0:
                self.layers[layer_id]["download_total"] = 1
            self.layers[layer_id]["download_current"] = self.layers[layer_id][
                "download_total"
            ]
            self.layers[layer_id]["download_complete"] = True
            self.layers[layer_id]["status"] = status
            return True
        elif status in ("Pull complete", "Already exists") or "complete" in status.lower():
            # Pull complete means the layer is ready locally. Ensure both phase
            # counters are fully satisfied even if Docker omitted byte totals.
            if self.layers[layer_id]["download_total"] == 0:
                self.layers[layer_id]["download_total"] = 1
            if self.layers[layer_id]["extract_total"] == 0:
                self.layers[layer_id]["extract_total"] = self.layers[layer_id][
                    "download_total"
                ]
            self.layers[layer_id]["download_current"] = self.layers[layer_id][
                "download_total"
            ]
            self.layers[layer_id]["extract_current"] = self.layers[layer_id][
                "extract_total"
            ]
            self.layers[layer_id]["download_complete"] = True
            self.layers[layer_id]["extract_complete"] = True
            self.layers[layer_id]["complete"] = True
            self.layers[layer_id]["status"] = status
            return True  # Force emit when layer completes
        
        return False

    def _is_pull_active(self) -> bool:
        has_active_layer = any(
            layer.get("status") in ("Downloading", "Extracting")
            for layer in self.layers.values()
        )
        if has_active_layer:
            return True

        # Docker may spend a while verifying/applying layers after all byte
        # download events have stopped. Treat that as active only after real
        # progress has started, so the UI does not look frozen near the end.
        return self.last_progress >= 85 and any(
            not layer.get("complete") for layer in self.layers.values()
        )

    def _synthetic_progress_cap(self) -> int:
        if any(layer.get("status") == "Extracting" for layer in self.layers.values()):
            return 97
        if self.last_progress >= 85 and any(
            not layer.get("complete") for layer in self.layers.values()
        ):
            return 97
        return 90

    def _calculate_layer_phase_progress(self) -> int:
        if not self.layers:
            return 0

        total_score = 0.0
        for layer in self.layers.values():
            if layer.get("complete"):
                total_score += 1.0
                continue

            if layer.get("extract_total", 0) > 0:
                extract_total = layer.get("extract_total", 0)
                extract_fraction = min(
                    layer.get("extract_current", 0) / extract_total,
                    1,
                )
                total_score += 0.70 + extract_fraction * 0.25
                continue

            if layer.get("download_complete"):
                total_score += 0.65
                continue

            if layer.get("download_total_known") and layer.get("download_total", 0) > 0:
                download_total = layer.get("download_total", 0)
                download_fraction = min(
                    layer.get("download_current", 0) / download_total,
                    1,
                )
                total_score += 0.05 + download_fraction * 0.60
                continue

            if layer.get("status") == "Downloading":
                total_score += 0.05
            elif layer.get("status") == "Extracting":
                total_score += 0.70

        return min(int((total_score / len(self.layers)) * 95), 99)
                
    def calculate_overall_progress(self) -> int:
        """Calculate overall progress as a percentage (0-100)."""
        if not self.layers:
            return 0

        known_download_layers = [
            layer
            for layer in self.layers.values()
            if layer.get("download_total_known") and layer.get("download_total", 0) > 0
        ]
        total_bytes = sum(
            layer.get("download_total", 0) for layer in known_download_layers
        )
        current_bytes = sum(
            min(layer.get("download_current", 0), layer.get("download_total", 0))
            for layer in known_download_layers
        )
        unknown_incomplete_layers = [
            layer
            for layer in self.layers.values()
            if not layer.get("download_complete")
            and not layer.get("complete")
            and not layer.get("download_total_known")
        ]
        
        # If we have any bytes info (at least one layer has a known total size > 0),
        # use byte-based progress.
        if total_bytes > 0:
            download_fraction = min(current_bytes / total_bytes, 1)
            if unknown_incomplete_layers:
                known_layer_count = len(self.layers) - len(unknown_incomplete_layers)
                layer_cap = known_layer_count / len(self.layers)
                download_fraction = min(download_fraction, layer_cap)

            extraction_seen = any(
                layer.get("status") == "Extracting"
                or (layer.get("extract_total", 0) > 0 and not layer.get("complete"))
                for layer in self.layers.values()
            )
            all_complete = all(layer.get("complete") for layer in self.layers.values())

            if all_complete:
                return 100

            layer_progress = self._calculate_layer_phase_progress()

            if extraction_seen:
                extract_total = 0
                extract_current = 0
                for layer in self.layers.values():
                    layer_extract_total = layer.get("extract_total", 0) or layer.get(
                        "download_total",
                        0,
                    )
                    if layer_extract_total <= 0:
                        continue
                    extract_total += layer_extract_total
                    if layer.get("extract_complete") or layer.get("complete"):
                        extract_current += layer_extract_total
                    else:
                        extract_current += min(
                            layer.get("extract_current", 0),
                            layer_extract_total,
                        )
                extract_fraction = (
                    min(extract_current / extract_total, 1)
                    if extract_total > 0
                    else 0
                )
                progress = int((download_fraction * 0.85 + extract_fraction * 0.14) * 100)
                layer_progress = min(layer_progress, progress + 8)
                return min(max(progress, layer_progress), 99)

            # Keep pre-extraction progress below complete so the UI does not
            # show 100% while Docker is still applying layers locally.
            progress = int(download_fraction * 85)
            layer_progress = min(layer_progress, progress + 5)
            return min(max(progress, layer_progress), 85)
            
        # Before Docker reports real byte totals, layer counts are only a weak
        # readiness signal. Cached "Already exists" layers can arrive first and
        # otherwise make the UI jump to 20%+ even though the real pull has not
        # started. Keep this phase near the beginning until byte progress exists.
        layer_progress = self._calculate_layer_phase_progress()
        if layer_progress > 0:
            return min(layer_progress, 5)
            
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
        elif self.last_progress >= 85:
            return "Finalizing"
        return "Preparing"
        
    def should_emit(self) -> bool:
        """Check if enough time has passed for a new emit."""
        now = time.time()
        return (now - self.last_emit_time) >= self.throttle_interval
        
    def emit_progress(self, force: bool = False) -> None:
        """Emit a progress update if throttle allows or if forced."""
        progress = self.calculate_overall_progress()
        if self.last_progress >= 0:
            progress = max(progress, self.last_progress)
            if (
                progress == self.last_progress
                and self._is_pull_active()
                and (time.time() - self.last_emit_time) >= self.synthetic_interval
            ):
                progress = min(progress + 1, self._synthetic_progress_cap())
        
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
    platform: str = None,
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
        from docker.utils import parse_repository_tag

        repository, tag = parse_repository_tag(image_name)
        tag = tag or "latest"
        
        platform_msg = f", platform={platform}" if platform else ""
        logging.info(f"Pulling repository={repository}, tag={tag}{platform_msg}")
        
        # Use the low-level API to stream progress
        # This returns a generator of JSON objects
        lines_received = 0
        logging.info(f"Iteration through Docker pull stream for {image_name} started.")
        
        # We manually iterate to ensure we can catch the shutdown signal as fast as possible
        pull_kwargs = {
            "tag": tag,
            "stream": True,
            "decode": True,
        }
        if platform:
            pull_kwargs["platform"] = platform

        pull_stream = docker_client.api.pull(repository, **pull_kwargs)
        
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

