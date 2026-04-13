import threading
import logging
import json
from services.orchestrator_service import TokenExpiredError, ProviderNotFoundError

class HeartbeatManager:
    def __init__(self, orchestrator_service, provider_id, shutdown_event, client=None):
        self.orchestrator_service = orchestrator_service
        self.provider_id = provider_id
        self.shutdown_event = shutdown_event
        self.client = client  # Optional DGNClient reference for routing config hot-reload
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
                routing_config = self.orchestrator_service.send_heartbeat(self.provider_id)
                # Apply routing config hot-reload if client reference is available
                if routing_config and self.client and hasattr(self.client, "apply_routing_config"):
                    self.client.apply_routing_config(routing_config)
            except TokenExpiredError:
                self.orchestrator_service.signal_auth_expired()
                logging.warning("Heartbeat failed due to expired token.")
                if self.orchestrator_service.is_auth_failed_permanently():
                    logging.error("Auth permanently failed. Stopping heartbeat loop.")
                    break
            except ProviderNotFoundError:
                logging.warning("Provider registration expired (detected in heartbeat). Signaling main process for restart.")
                print(json.dumps({"status": "PROVIDER_EXPIRED"}), flush=True)
                self.shutdown_event.set()  # Trigger shutdown
                break
            except Exception as e:
                logging.error(f"An error occurred in the heartbeat loop: {e}")
            # Wait for 30 seconds or until shutdown event is set
            self.shutdown_event.wait(30)

