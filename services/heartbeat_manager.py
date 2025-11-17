import threading
import logging
import json
from services.orchestrator_service import TokenExpiredError

class HeartbeatManager:
    def __init__(self, orchestrator_service, provider_id, shutdown_event):
        self.orchestrator_service = orchestrator_service
        self.provider_id = provider_id
        self.shutdown_event = shutdown_event
        self.thread = None

    def start(self):
        """Starts a background thread to send heartbeats."""
        self.thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.thread.start()
        logging.info("Heartbeat thread started.")

    def _heartbeat_loop(self):
        """The loop that sends heartbeats periodically."""
        while not self.shutdown_event.is_set():
            try:
                self.orchestrator_service.send_heartbeat(self.provider_id)
            except TokenExpiredError:
                # Notify the main process that the token has expired
                print(json.dumps({"status": "AUTH_EXPIRED"}), flush=True)
                logging.warning("Heartbeat failed due to expired token. Notified main process.")
            except Exception as e:
                logging.error(f"An error occurred in the heartbeat loop: {e}")
            # Wait for 60 seconds or until shutdown event is set
            self.shutdown_event.wait(60)
