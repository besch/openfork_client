import time
import os
import logging
import threading


def get_headless_log_paths(service_type: str):
    normalized = (service_type or "").lower()
    if (
        "davinci" in normalized
        or "scail" in normalized
        or "vista4d" in normalized
        or (
            "ltx23" in normalized and "comfyui" not in normalized
        )
    ):
        return ["/tmp/wan2gp_server.log"]
    return ["/tmp/comfyui.log"]


class LogTailer:
    def __init__(self, file_path: str, service_type: str = "headless"):
        self.file_path = file_path
        self.service_type = service_type

    def tail(self, shutdown_event: threading.Event):
        """Tails the log file and logs new lines to the application logger."""
        logging.info(f"Starting log tailer for {self.file_path} ([{self.service_type}])")
        
        # Wait for file to exist
        while not os.path.exists(self.file_path):
            if shutdown_event.is_set():
                return
            time.sleep(1)

        try:
            with open(self.file_path, "r") as f:
                # Go to the end of the file
                f.seek(0, 2)
                
                while not shutdown_event.is_set():
                    line = f.readline()
                    if not line:
                        time.sleep(0.1)
                        continue
                    
                    clean_line = line.strip()
                    if clean_line:
                        logging.info(f"[{self.service_type}] {clean_line}")
        except Exception as e:
            logging.error(f"Error tailing log file {self.file_path}: {e}")
