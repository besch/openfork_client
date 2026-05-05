import websocket
import uuid
import json
import os
import time
import urllib.request
import urllib.error
import http.client
import threading
from queue import Queue, Empty
import logging
from typing import Union, Dict, Any, Optional, List
import requests

from config import TimeoutConfig
from exceptions import AuthError
from services.orchestrator_service import TokenExpiredError

REMOTE_WORKFLOW_STOP_STATUSES = ("cancelled", "deleted", "lease_lost")


class ComfyUIClient:
    def __init__(self, comfyui_ws_url: str, access_token: str = None):
        self.comfyui_ws_url = comfyui_ws_url
        self.http_base = self._http_base_from_ws(comfyui_ws_url)
        self.access_token = access_token

    def _http_base_from_ws(self, ws_url: str) -> str:
        # ws://host:port/ws?... -> http://host:port
        # wss://host:port/ws?... -> https://host:port
        base = ws_url.split("/ws")[0]
        if base.startswith("wss://"):
            return base.replace("wss://", "https://")
        return base.replace("ws://", "http://")

    def wait_for_ready(
        self,
        shutdown_event: threading.Event,
        timeout: Optional[int] = None,
        abort_event: Optional[threading.Event] = None,
    ) -> bool:
        """Waits for the ComfyUI server to be available.
        
        Args:
            shutdown_event: Event to signal shutdown
            timeout: Maximum seconds to wait (default: TimeoutConfig.COMFYUI_READY_TIMEOUT)
            abort_event: Optional event for provider-local aborts, such as a
                monitored container crash during startup.
            
        Returns:
            True if server became ready, False otherwise
        """
        if timeout is None:
            timeout = TimeoutConfig.COMFYUI_READY_TIMEOUT
        logging.info(f"Waiting for ComfyUI server to be ready (timeout: {timeout}s)...")
        start_time = time.time()
        url = f"{self.http_base}/object_info"
        while time.time() - start_time < timeout:
            if shutdown_event.is_set():
                logging.warning("Shutdown requested while waiting for ComfyUI.")
                return False
            if abort_event and abort_event.is_set():
                logging.warning("ComfyUI readiness wait aborted by local container failure.")
                return False
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logging.info("ComfyUI server is ready.")
                    return True
                else:
                    logging.debug(f"ComfyUI not ready yet (status {response.status_code}). Retrying...")
            except requests.exceptions.RequestException as e:
                logging.debug(f"ComfyUI not ready yet (connection error: {e}). Retrying...")

            wait_until = time.time() + 5
            while time.time() < wait_until:
                if shutdown_event.is_set():
                    logging.warning("Shutdown requested while waiting for ComfyUI.")
                    return False
                if abort_event and abort_event.is_set():
                    logging.warning("ComfyUI readiness wait aborted by local container failure.")
                    return False
                shutdown_event.wait(0.5)
        logging.error(f"ComfyUI server did not become ready in {timeout} seconds.")
        return False

    def trigger_workflow(self, workflow_json: dict) -> str:
        """Trigger a workflow in ComfyUI by HTTP POST /prompt and return the prompt ID."""
        client_id = str(uuid.uuid4())

        if not (isinstance(workflow_json, dict) and "prompt" in workflow_json and isinstance(workflow_json["prompt"], dict)):
            raise ValueError("Invalid workflow payload; expected a dict with a 'prompt' key containing the API graph.")
        payload_prompt = workflow_json["prompt"]

        # Validate API graph structure: dict of nodes with class_type and inputs
        if not isinstance(payload_prompt, dict) or not payload_prompt:
            raise ValueError("Invalid workflow payload; 'prompt' must be a non-empty dict of nodes.")
        for k, node in payload_prompt.items():
            if not isinstance(node, dict):
                raise ValueError(f"Invalid node for id {k}: expected dict.")
            if not node.get("class_type"):
                raise ValueError(f"Invalid node for id {k}: missing 'class_type'.")

        # Probe with retry to ensure ComfyUI is fully ready. This acts as a mini
        # "wait_for_ready" for modes where the full wait is not performed.
        max_retries = 60
        retry_delay = 5 # seconds
        for i in range(max_retries):
            try:
                probe_req = urllib.request.Request(f"{self.http_base}/object_info")
                with urllib.request.urlopen(probe_req, timeout=5):
                    pass # Success
                break # Exit loop
            except Exception as e:
                if i < max_retries - 1:
                    logging.warning(f"ComfyUI probe failed (attempt {i+1}/{max_retries}). Retrying in {retry_delay}s... Error: {e}")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"ComfyUI probe failed after {max_retries} attempts.")
                    raise RuntimeError(f"Cannot reach ComfyUI at {self.http_base} (/object_info): {e}. "
                                       f"Check COMFYUI_WS_URL and that ComfyUI is running and port 8188 is published.")

        body = json.dumps({"prompt": payload_prompt, "client_id": client_id}).encode("utf-8")
        req = urllib.request.Request(f"{self.http_base}/prompt", data=body, headers={"Content-Type": "application/json"})

        prompt_id = None
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                resp_body = resp.read().decode("utf-8")
                try:
                    resp_json = json.loads(resp_body)
                except Exception:
                    resp_json = {}
                prompt_id = resp_json.get("prompt_id") or resp_json.get("data", {}).get("prompt_id")
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                detail = str(e)
            raise RuntimeError(f"ComfyUI /prompt returned HTTP {e.code}: {detail}")
        except (urllib.error.URLError, http.client.RemoteDisconnected) as e:
            reason = e.reason if hasattr(e, 'reason') else str(e)
            error_message = f"Cannot reach ComfyUI at {self.http_base} (/prompt): {reason}. Check COMFYUI_WS_URL and that ComfyUI is running and the port is published."
            logging.error(error_message)
            raise RuntimeError(error_message)

        return prompt_id

    def interrupt_workflow(self):
        """Interrupts the currently running workflow in ComfyUI."""
        try:
            req = urllib.request.Request(f"{self.http_base}/interrupt", method='POST', data=b'')
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    logging.info("Successfully sent interrupt request to ComfyUI.")
                else:
                    logging.warning(f"ComfyUI interrupt request returned status {resp.status}")
        except Exception as e:
            logging.error(f"Failed to send interrupt request to ComfyUI: {e}")

    def _ws_reader_thread(self, ws, q, shutdown_event):
        while not shutdown_event.is_set():
            try:
                out = ws.recv()
                q.put(out)
            except websocket.WebSocketConnectionClosedException:
                logging.info("WebSocket connection closed.")
                break
            except websocket.WebSocketTimeoutException:
                logging.debug("WebSocket recv timed out, continuing to listen.")
                continue
            except Exception as e:
                # Only log error if it's not a direct result of a clean shutdown
                if not shutdown_event.is_set():
                    logging.error(f"Exception in WebSocket reader thread: {e}")
                    q.put(e) # Signal error to the main thread
                break

    def get_workflow_output(
        self,
        prompt_id: str,
        job_id: str,
        orchestrator_service,
        terminal_node_ids: Optional[List[str]] = None,
        timeout_sec: Optional[int] = None,
        shutdown_event: Optional[threading.Event] = None
    ) -> Union[Dict[str, Any], None, str]:
        """Get the output of a completed workflow by listening on WS in a separate thread.
        
        Args:
            prompt_id: The ComfyUI prompt ID
            job_id: The DGN job ID for cancellation polling
            orchestrator_service: Service for checking job status
            terminal_node_ids: Optional list of node IDs that must complete
            timeout_sec: Maximum seconds to wait (default: TimeoutConfig.WORKFLOW_TIMEOUT)
            shutdown_event: Optional event to signal shutdown
            
        Returns:
            Dict of outputs, 'interrupted' string, or None on failure
        """
        if timeout_sec is None:
            timeout_sec = TimeoutConfig.WORKFLOW_TIMEOUT
        if not prompt_id:
            logging.warning("get_workflow_output called with empty prompt_id.")
            return None

        client_id = str(uuid.uuid4())
        ws = websocket.WebSocket()
        internal_shutdown_event = threading.Event()

        try:
            ws.connect(
                self.comfyui_ws_url.format(client_id),
                timeout=600,
                ping_interval=20,
                ping_timeout=10
            )
            ws.settimeout(10) # Set a shorter timeout for recv operations
            logging.info(f"Successfully connected to ComfyUI WebSocket at {self.comfyui_ws_url.format(client_id)}")
        except Exception as e:
            try:
                ws.close()
            except Exception:
                pass
            logging.error(f"Failed to connect to ComfyUI WebSocket at {self.comfyui_ws_url.format(client_id)}: {e}")
            raise RuntimeError(f"Failed to connect to ComfyUI WebSocket at {self.comfyui_ws_url}: {e}")

        q = Queue()
        reader_thread = threading.Thread(target=self._ws_reader_thread, args=(ws, q, internal_shutdown_event), daemon=True)
        reader_thread.start()
        logging.info("WebSocket reader thread started.")

        start_ts = time.time()
        last_poll_ts = start_ts
        all_node_outputs = {}
        executed_nodes = set()

        try:
            while True:
                if shutdown_event and shutdown_event.is_set():
                    logging.warning("Shutdown event received, interrupting workflow output wait.")
                    self.interrupt_workflow()
                    return "interrupted"

                if (time.time() - start_ts) > timeout_sec:
                    logging.warning(f"Workflow output timed out after {timeout_sec} seconds for prompt_id: {prompt_id}. Breaking loop to fetch history.")
                    self.interrupt_workflow()
                    break

                # Polling mechanism for cancellation
                if time.time() - last_poll_ts > 5:
                    last_poll_ts = time.time()
                    try:
                        job_details = orchestrator_service.get_job(job_id)
                        status = job_details.get("status") if isinstance(job_details, dict) else None
                        if status in REMOTE_WORKFLOW_STOP_STATUSES:
                            logging.warning(
                                f"Remote stop requested for job {job_id} "
                                f"(polled status: {status}). Interrupting workflow "
                                "so the worker can recover and accept other jobs."
                            )
                            self.interrupt_workflow()
                            return "interrupted"
                    except TokenExpiredError:
                        orchestrator_service.signal_auth_expired()
                        logging.warning(f"Auth expired during cancellation polling for job {job_id}.")
                        # If auth is permanently failed, stop polling
                        if orchestrator_service.is_auth_failed_permanently():
                            logging.error("Auth permanently failed. Stopping workflow.")
                            self.interrupt_workflow()
                            return "auth_failed"
                    except Exception as e:
                        logging.error(f"Error checking for job cancellation (polling fallback): {e}")

                try:
                    out = q.get(timeout=1)
                    if isinstance(out, Exception):
                        logging.error(f"Exception received from WebSocket reader thread: {out}")
                        raise out
                except Empty:
                    logging.debug("Queue empty, continuing to wait for messages.")
                    # Fallback check if it's been a while without activity
                    if time.time() - last_poll_ts > 30:
                        try:
                            # 1. Log Queue Status
                            queue_blocks = requests.get(f"{self.http_base}/queue", timeout=5).json()
                            running_len = len(queue_blocks.get('queue_running', []))
                            pending_len = len(queue_blocks.get('queue_pending', []))
                            logging.info(f"ComfyUI Status Check: Running={running_len}, Pending={pending_len}")
                            
                            if running_len > 0:
                                # What is currently running?
                                current = queue_blocks['queue_running'][0]
                                logging.info(f"Currently executing: {current[1]} (prompt_id: {current[1]})")

                            # 2. Tail ComfyUI Log in Headless Mode (Debug)
                            from config import HEADLESS_MODE
                            if HEADLESS_MODE and (time.time() - start_ts > 60) and running_len == 0 and pending_len > 0:
                                logging.warning("Job seems stuck pending. Tailing /tmp/comfyui.log:")
                                try:
                                    with open("/tmp/comfyui.log", "r") as f:
                                        # Read last 2KB
                                        f.seek(0, 2)
                                        fsize = f.tell()
                                        f.seek(max(fsize - 2048, 0), 0)
                                        print(f"--- COMFYUI LOG TAIL ---\n{f.read()}\n------------------------")
                                except Exception as log_e:
                                    logging.warning(f"Could not read comfyui.log: {log_e}")

                        except Exception as e:
                            logging.warning(f"Status check failed: {e}")
                            
                    continue

                if isinstance(out, str):
                    try:
                        message = json.loads(out)
                    except Exception as e:
                        logging.error(f"Failed to parse WebSocket message as JSON: {out}. Error: {e}")
                        continue

                    mtype = message.get("type")
                    data = message.get("data", {}) if isinstance(message, dict) else {}
                    if not isinstance(data, dict):
                        data = {}

                    if mtype == "executed" and data.get("prompt_id") == prompt_id:
                        node_id = data.get("node")
                        if node_id is not None:
                            executed_nodes.add(str(node_id))
                            logging.info(f"Executed message for prompt_id {prompt_id}, node_id: {node_id}. Executed nodes count: {len(executed_nodes)}")

                        if "output" in data and node_id is not None:
                            all_node_outputs[str(node_id)] = data["output"]

                        if terminal_node_ids:
                            need = {str(n) for n in terminal_node_ids}
                            if need and need.issubset(executed_nodes):
                                logging.info(f"All terminal nodes {terminal_node_ids} executed for prompt_id {prompt_id}. Breaking loop to fetch history.")
                                break

                    elif mtype == "status":
                        status = data.get("status")
                        if isinstance(status, dict):
                            exec_info = status.get("exec_info", {})
                            if isinstance(exec_info, dict) and exec_info.get("queue_remaining") == 0:
                                logging.info(f"ComfyUI queue is empty for prompt_id {prompt_id}. Breaking loop to fetch history.")
                                break
                    else:
                        logging.debug(f"Received WebSocket message of type: {mtype}.")
        finally:
            internal_shutdown_event.set()
            try:
                ws.close()
                logging.info("WebSocket connection closed.")
            except Exception as e:
                logging.error(f"Error closing WebSocket connection: {e}")
            reader_thread.join(timeout=5)

        logging.info(f"Exiting get_workflow_output for prompt_id {prompt_id} due to loop completion. Fetching history for outputs.")
        history_outputs = self.fetch_history_outputs(prompt_id)
        if history_outputs is not None:
            return history_outputs
        else:
            logging.warning(f"Failed to fetch history outputs for prompt_id {prompt_id}. Returning accumulated outputs (which might be empty).")
            return all_node_outputs

    def fetch_history_outputs(self, prompt_id: str) -> Union[dict, None]:
        """Fetches workflow outputs from ComfyUI's /history endpoint for a given prompt_id."""
        try:
            history_url = f"{self.http_base}/history?prompt_id={prompt_id}"
            logging.info(f"Fetching history from: {history_url}")
            req = urllib.request.Request(history_url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_body = resp.read().decode("utf-8")
                history_data = json.loads(resp_body)

                if prompt_id in history_data:
                    prompt_history = history_data[prompt_id]
                    if "outputs" in prompt_history:
                        logging.info(f"Successfully fetched outputs from history for prompt_id {prompt_id}.")
                        return prompt_history["outputs"]
                    else:
                        logging.warning(f"No 'outputs' found in history for prompt_id {prompt_id}.")
                else:
                    logging.warning(f"Prompt ID {prompt_id} not found in ComfyUI history.")
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                detail = str(e)
            logging.error(f"ComfyUI /history returned HTTP {e.code}: {detail}")
        except Exception as e:
            logging.error(f"Failed to fetch history from ComfyUI: {e}")
        return None # type: ignore
