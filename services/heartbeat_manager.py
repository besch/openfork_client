import threading
import logging
import json
from services.orchestrator_service import (
    ProviderNotFoundError,
    TokenExpiredError,
    UpgradeRequiredError,
)

# Number of consecutive failed heartbeats before we escalate. At 30s/heartbeat,
# 4 misses means the orchestrator has been unreachable for ~2 minutes.
HEARTBEAT_DEGRADED_THRESHOLD = 4


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
        consecutive_failures = 0
        degraded_emitted = False
        while not self.shutdown_event.is_set():
            try:
                routing_config = self.orchestrator_service.send_heartbeat(
                    self.provider_id,
                    compaction_pending=bool(
                        getattr(self.client, "compaction_pending", False)
                    ),
                )
                # Apply routing config hot-reload if client reference is available
                if routing_config and self.client and hasattr(self.client, "apply_routing_config"):
                    self.client.apply_routing_config(routing_config)
                if consecutive_failures > 0:
                    logging.info(f"Heartbeat recovered after {consecutive_failures} failure(s).")
                    if degraded_emitted:
                        print(json.dumps({"status": "PROVIDER_RECOVERED"}), flush=True)
                consecutive_failures = 0
                degraded_emitted = False
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
            except UpgradeRequiredError as e:
                logging.error("Required OpenFork update detected. Stopping heartbeat loop.")
                if not getattr(self.orchestrator_service, "_upgrade_required_emitted", False):
                    print(
                        json.dumps(
                            {
                                "status": "UPGRADE_REQUIRED",
                                "payload": getattr(e, "payload", {}),
                            }
                        ),
                        flush=True,
                    )
                self.shutdown_event.set()
                break
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures >= HEARTBEAT_DEGRADED_THRESHOLD:
                    if not degraded_emitted:
                        logging.error(
                            f"Heartbeat has failed {consecutive_failures} times in a row "
                            f"(~{consecutive_failures * 30}s). Marking provider as degraded."
                        )
                        print(
                            json.dumps({
                                "status": "PROVIDER_DEGRADED",
                                "payload": {
                                    "consecutive_failures": consecutive_failures,
                                    "last_error": str(e),
                                },
                            }),
                            flush=True,
                        )
                        degraded_emitted = True
                    else:
                        logging.warning(
                            f"Heartbeat still failing ({consecutive_failures} consecutive): {e}"
                        )
                else:
                    logging.error(f"An error occurred in the heartbeat loop: {e}")
            # Wait for 30 seconds or until shutdown event is set
            self.shutdown_event.wait(30)

