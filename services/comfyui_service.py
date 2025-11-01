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
from typing import Union, Dict, List
import requests
from supabase import create_client, Client

class ComfyUIClient:
    def __init__(self, comfyui_ws_url: str):
        self.comfyui_ws_url = comfyui_ws_url
        self.http_base = self._http_base_from_ws(comfyui_ws_url)
        self.supabase_client: Union[Client, None] = self._init_supabase_client()
        self._health_check_cache = {'is_healthy': False, 'timestamp': 0}
        self._health_check_ttl = 10  # Cache health status for 10 seconds

    def _init_supabase_client(self) -> Union[Client, None]:
        supabase_url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not supabase_url or not supabase_key:
            logging.warning("Supabase URL/key not set. Real-time cancellation will not work.")
            return None
        try:
            return create_client(supabase_url, supabase_key)
        except Exception as e:
            logging.error(f"Failed to create Supabase client: {e}")
            return None

    def _http_base_from_ws(self, ws_url: str) -> str:
        base = ws_url.split("/ws")[0]
        if base.startswith("wss://"):
            return base.replace("wss://", "https://")
        return base.replace("ws://", "http://")

    def check_health(self, use_cache: bool = True) -> bool:
        """
        Check if ComfyUI server is healthy and responding.
        
        Args:
            use_cache: If True, use cached health status if available
            
        Returns:
            True if server is healthy, False otherwise
        """
        # Check cache first
        if use_cache:
            cache_age = time.time() - self._health_check_cache['timestamp']
            if cache_age < self._health_check_ttl:
                return self._health_check_cache['is_healthy']
        
        try:
            # Try to get object_info as a health check
            response = requests.get(f"{self.http_base}/object_info", timeout=5)
            is_healthy = response.status_code == 200
            
            # Update cache
            self._health_check_cache = {
                'is_healthy': is_healthy,
                'timestamp': time.time()
            }
            
            return is_healthy
            
        except Exception as e:
            logging.debug(f"Health check failed: {e}")
            self._health_check_cache = {
                'is_healthy': False,
                'timestamp': time.time()
            }
            return False

    def get_object_info(self) -> Union[Dict, None]:
        """Fetches the raw object_info from the ComfyUI server."""
        url = f"{self.http_base}/object_info"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to get object_info from ComfyUI: {e}")
            return None

    def get_installed_nodes(self) -> List[str]:
        """Returns a list of all installed node class_types."""
        object_info = self.get_object_info()
        if not object_info:
            return []
        
        return list(object_info.keys())

    def refresh_nodes(self):
        """Refresh ComfyUI's node cache."""
        try:
            requests.post(f"{self.http_base}/refresh", timeout=10)
            time.sleep(3)
            logging.info("Nodes refreshed")
            # Invalidate health cache after refresh
            self._health_check_cache['timestamp'] = 0
        except Exception as e:
            logging.warning(f"Could not request /refresh: {e}")

    def wait_for_ready(self, shutdown_event: threading.Event, timeout=180) -> bool:
        """
        Waits for the ComfyUI server to be available and healthy.
        
        Args:
            shutdown_event: Event to check for shutdown signal
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if server became ready, False otherwise
        """
        logging.info("Waiting for ComfyUI server to be ready...")
        start_time = time.time()
        attempt = 0
        
        while time.time() - start_time < timeout:
            if shutdown_event.is_set():
                logging.warning("Shutdown requested while waiting for ComfyUI.")
                return False
            
            attempt += 1
            
            # Check health
            if self.check_health(use_cache=False):
                logging.info(f"ComfyUI server is ready (took {int(time.time() - start_time)}s)")
                return True
            
            # Log progress every 10 attempts
            if attempt % 10 == 0:
                elapsed = int(time.time() - start_time)
                logging.info(f"Still waiting for ComfyUI... ({elapsed}s elapsed)")
            
            # Wait before retry
            shutdown_event.wait(2)
        
        logging.error(f"ComfyUI server did not become ready in {timeout} seconds.")
        return False

    def trigger_workflow(self, workflow_json: dict) -> str:
        """Trigger a workflow in ComfyUI by HTTP POST /prompt and return the prompt ID."""
        client_id = str(uuid.uuid4())

        if not (isinstance(workflow_json, dict) and "prompt" in workflow_json and isinstance(workflow_json["prompt"], dict)):
            raise ValueError("Invalid workflow payload; expected a dict with a 'prompt' key containing the API graph.")
        payload_prompt = workflow_json["prompt"]

        # Validate API graph structure
        if not isinstance(payload_prompt, dict) or not payload_prompt:
            raise ValueError("Invalid workflow payload; 'prompt' must be a non-empty dict of nodes.")
        for k, node in payload_prompt.items():
            if not isinstance(node, dict):
                raise ValueError(f"Invalid node for id {k}: expected dict.")
            if not node.get("class_type"):
                raise ValueError(f"Invalid node for id {k}: missing 'class_type'.")

        # Probe with retry to ensure ComfyUI is fully ready
        max_retries = 12
        retry_delay = 5
        for i in range(max_retries):
            if self.check_health(use_cache=False):
                break
            
            if i < max_retries - 1:
                logging.warning(f"ComfyUI not fully ready (attempt {i+1}/{max_retries}). Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Cannot reach ComfyUI at {self.http_base}. Check that ComfyUI is running.")

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
            error_message = f"Cannot reach ComfyUI at {self.http_base} (/prompt): {reason}"
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

    def _ws_reader_thread(self, ws, q):
        while True:
            try:
                out = ws.recv()
                q.put(out)
            except Exception as e:
                q.put(e)
                break

    def get_workflow_output(self, prompt_id: str, job_id: str, orchestrator_service, terminal_node_ids: Union[list[str], None] = None, timeout_sec: int = 7200, shutdown_event: threading.Event = None) -> Union[dict, None, str]:
        """Get the output of a completed workflow by listening on WS in a separate thread."""
        if not prompt_id:
            logging.warning("get_workflow_output called with empty prompt_id.")
            return None

        client_id = str(uuid.uuid4())
        ws = websocket.WebSocket()
        try:
            ws.connect(self.comfyui_ws_url.format(client_id), timeout=600)
            logging.info(f"Successfully connected to ComfyUI WebSocket")
        except Exception as e:
            logging.error(f"Failed to connect to ComfyUI WebSocket: {e}")
            raise RuntimeError(f"Failed to connect to ComfyUI WebSocket: {e}")

        q = Queue()
        reader_thread = threading.Thread(target=self._ws_reader_thread, args=(ws, q), daemon=True)
        reader_thread.start()
        logging.debug("WebSocket reader thread started.")

        start_ts = time.time()
        last_poll_ts = start_ts
        all_node_outputs = {}
        executed_nodes = set()

        job_cancellation_event = threading.Event()
        subscription = None
        if self.supabase_client:
            try:
                def on_update(payload):
                    if payload.get('new', {}).get('status') == 'cancelled':
                        logging.warning(f"Cancellation requested for job {job_id} via real-time.")
                        job_cancellation_event.set()

                subscription = self.supabase_client.table("dgn_jobs").on("UPDATE", on_update).filter("id", "eq", job_id).subscribe()
            except Exception as e:
                logging.error(f"Failed to subscribe to real-time job updates: {e}")

        try:
            while True:
                if shutdown_event and shutdown_event.is_set():
                    logging.warning("Shutdown event received, interrupting workflow.")
                    return "interrupted"

                if job_cancellation_event.is_set():
                    self.interrupt_workflow()
                    return "interrupted"

                if (time.time() - start_ts) > timeout_sec:
                    logging.warning(f"Workflow timed out after {timeout_sec}s for prompt_id: {prompt_id}")
                    break

                # Fallback polling
                if time.time() - last_poll_ts > 5:
                    last_poll_ts = time.time()
                    try:
                        job_details = orchestrator_service.get_job(job_id)
                        if job_details and job_details.get('status') == 'cancelled':
                            logging.warning(f"Job {job_id} cancelled (polled).")
                            self.interrupt_workflow()
                            return "interrupted"
                    except Exception as e:
                        logging.error(f"Error checking for cancellation: {e}")

                try:
                    out = q.get(timeout=1)
                    if isinstance(out, Exception):
                        logging.error(f"Exception in WebSocket reader: {out}")
                        raise out
                except Empty:
                    continue

                if isinstance(out, str):
                    try:
                        message = json.loads(out)
                    except Exception:
                        continue

                    mtype = message.get("type")
                    data = message.get("data", {}) if isinstance(message, dict) else {}
                    if not isinstance(data, dict):
                        data = {}

                    if mtype == "executed" and data.get("prompt_id") == prompt_id:
                        node_id = data.get("node") or data.get("node_id") or data.get("node_id_name")
                        if node_id is not None:
                            executed_nodes.add(str(node_id))

                        if "output" in data and node_id is not None:
                            all_node_outputs[str(node_id)] = data["output"]

                        if terminal_node_ids:
                            need = {str(n) for n in terminal_node_ids}
                            if need and need.issubset(executed_nodes):
                                break

                    elif mtype == "status":
                        status = data.get("status") if isinstance(data, dict) else None
                        if isinstance(status, dict):
                            exec_info = status.get("exec_info", {})
                            if isinstance(exec_info, dict) and exec_info.get("queue_remaining") == 0:
                                break
        finally:
            if subscription and self.supabase_client:
                try:
                    self.supabase_client.realtime.remove_channel(subscription)
                except Exception as e:
                    logging.error(f"Error unsubscribing: {e}")
            try:
                ws.close()
            except Exception as e:
                logging.error(f"Error closing WebSocket: {e}")

        history_outputs = self.fetch_history_outputs(prompt_id)
        if history_outputs is not None:
            return history_outputs
        else:
            logging.warning(f"Failed to fetch history for prompt_id {prompt_id}. Returning accumulated outputs.")
            return all_node_outputs

    def fetch_history_outputs(self, prompt_id: str) -> Union[dict, None]:
        """Fetches workflow outputs from ComfyUI's /history endpoint."""
        try:
            history_url = f"{self.http_base}/history?prompt_id={prompt_id}"
            req = urllib.request.Request(history_url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_body = resp.read().decode("utf-8")
                history_data = json.loads(resp_body)

                if prompt_id in history_data:
                    prompt_history = history_data[prompt_id]
                    if "outputs" in prompt_history:
                        return prompt_history["outputs"]
        except Exception as e:
            logging.error(f"Failed to fetch history: {e}")
        return None