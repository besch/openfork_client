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
        self.supabase_client: Union[Client, None] = None
        self._health_check_cache = {'is_healthy': False, 'timestamp': 0}
        self._health_check_ttl = 10
        self._node_cache = {'nodes': [], 'timestamp': 0}
        self._node_cache_ttl = 60  # Cache nodes for 60 seconds

    def _init_supabase_client(self) -> Union[Client, None]:
        """
        NOTE: Real-time subscriptions are disabled in sync client.
        We rely on polling for job cancellation checks instead.
        """
        supabase_url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not supabase_url or not supabase_key:
            logging.warning("Supabase URL/key not set. Job cancellation will use polling only.")
            return None
        try:
            # Create client but don't use it for real-time
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
        IMPROVED: More robust health check with retries.
        """
        if use_cache:
            cache_age = time.time() - self._health_check_cache['timestamp']
            if cache_age < self._health_check_ttl:
                return self._health_check_cache['is_healthy']
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"{self.http_base}/object_info",
                    timeout=5
                )
                is_healthy = response.status_code == 200
                
                if is_healthy:
                    self._health_check_cache = {
                        'is_healthy': True,
                        'timestamp': time.time()
                    }
                    return True
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.debug(f"Health check attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(2)
                else:
                    logging.debug(f"Health check failed after {max_retries} attempts: {e}")
        
        self._health_check_cache = {
            'is_healthy': False,
            'timestamp': time.time()
        }
        return False

    def get_object_info(self) -> Union[Dict, None]:
        """
        IMPROVED: Fetches object_info with retry logic.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"{self.http_base}/object_info",
                    timeout=10
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Failed to get object_info (attempt {attempt + 1}): {e}")
                    time.sleep(2)
                else:
                    logging.error(f"Failed to get object_info after {max_retries} attempts: {e}")
        return None

    def get_installed_nodes(self, use_cache: bool = True) -> List[str]:
        """
        IMPROVED: Returns list of installed node class_types with caching.
        """
        if use_cache:
            cache_age = time.time() - self._node_cache['timestamp']
            if cache_age < self._node_cache_ttl and self._node_cache['nodes']:
                logging.debug(f"Using cached node list ({len(self._node_cache['nodes'])} nodes)")
                return self._node_cache['nodes']
        
        object_info = self.get_object_info()
        if not object_info:
            logging.warning("Could not fetch object_info, returning empty node list")
            return []
        
        nodes = list(object_info.keys())
        
        # Update cache
        self._node_cache = {
            'nodes': nodes,
            'timestamp': time.time()
        }
        
        logging.debug(f"Fetched {len(nodes)} installed nodes")
        return nodes

    def invalidate_node_cache(self):
        """Invalidate the cached node information."""
        self._node_cache = {'nodes': [], 'timestamp': 0}
        logging.debug("Node cache invalidated")

    def refresh_nodes(self):
        """
        IMPROVED: Refresh ComfyUI's node cache and invalidate local cache.
        """
        try:
            # Invalidate local cache first
            self.invalidate_node_cache()
            
            # Trigger ComfyUI's internal refresh
            response = requests.post(f"{self.http_base}/refresh", timeout=10)
            
            # Wait for refresh to complete
            time.sleep(5)
            
            # Verify refresh worked by fetching nodes
            nodes = self.get_installed_nodes(use_cache=False)
            logging.info(f"Nodes refreshed successfully. {len(nodes)} nodes available.")
            
            # Invalidate health cache too
            self._health_check_cache['timestamp'] = 0
            
        except Exception as e:
            logging.warning(f"Could not request /refresh: {e}")

    def wait_for_ready(self, shutdown_event: threading.Event, timeout=180) -> bool:
        """
        IMPROVED: More robust ready check with better logging.
        """
        logging.info("Waiting for ComfyUI server to be ready...")
        start_time = time.time()
        attempt = 0
        last_error = None
        
        while time.time() - start_time < timeout:
            if shutdown_event.is_set():
                logging.warning("Shutdown requested while waiting for ComfyUI.")
                return False
            
            attempt += 1
            
            # Check health
            if self.check_health(use_cache=False):
                # Additional verification: try to get nodes
                try:
                    nodes = self.get_installed_nodes(use_cache=False)
                    if len(nodes) > 0:
                        elapsed = int(time.time() - start_time)
                        logging.info(f"ComfyUI server is ready! {len(nodes)} nodes available (took {elapsed}s)")
                        return True
                    else:
                        last_error = "Server responded but no nodes available"
                except Exception as e:
                    last_error = f"Error fetching nodes: {e}"
            
            # Log progress every 10 attempts
            if attempt % 10 == 0:
                elapsed = int(time.time() - start_time)
                if last_error:
                    logging.info(f"Still waiting for ComfyUI... ({elapsed}s elapsed) - Last error: {last_error}")
                else:
                    logging.info(f"Still waiting for ComfyUI... ({elapsed}s elapsed)")
            
            # Wait before retry
            shutdown_event.wait(2)
        
        logging.error(f"ComfyUI server did not become ready in {timeout} seconds.")
        if last_error:
            logging.error(f"Last error: {last_error}")
        return False

    def trigger_workflow(self, workflow_json: dict) -> str:
        """Trigger a workflow in ComfyUI by HTTP POST /prompt and return the prompt ID."""
        client_id = str(uuid.uuid4())

        if not (isinstance(workflow_json, dict) and "prompt" in workflow_json):
            raise ValueError("Invalid workflow payload; expected a dict with a 'prompt' key.")
        
        payload_prompt = workflow_json["prompt"]
        
        # === CRITICAL FIX: ComfyUI /prompt endpoint ONLY accepts API format ===
        # LiteGraph format must be converted first
        
        if isinstance(payload_prompt, dict) and 'nodes' in payload_prompt:
            # This is LiteGraph format - ComfyUI CANNOT handle this!
            raise ValueError(
                "LiteGraph format workflows are not supported by ComfyUI's /prompt endpoint. "
                "The workflow must be converted to API format first using the workflow converter service. "
                "Please ensure workflows are converted during sync, not at execution time."
            )
        
        elif isinstance(payload_prompt, dict):
            # API format - validate structure
            if not payload_prompt:
                raise ValueError("Invalid workflow payload; 'prompt' must be a non-empty dict.")
            
            # Validate API graph structure
            for k, node in payload_prompt.items():
                if not isinstance(node, dict):
                    raise ValueError(f"Invalid node for id {k}: expected dict, got {type(node).__name__}.")
                if not node.get("class_type"):
                    raise ValueError(f"Invalid node for id {k}: missing 'class_type'.")
                
                # CRITICAL: Check for UUID nodes (subgraphs not flattened)
                class_type = node.get("class_type")
                try:
                    import uuid
                    uuid.UUID(class_type)  # This will succeed if class_type is a UUID
                    raise ValueError(
                        f"Node {k} has UUID class_type '{class_type}'. "
                        "This indicates the workflow contains unflattened subgraphs. "
                        "The workflow must be properly converted to API format."
                    )
                except ValueError as e:
                    if "UUID" in str(e) or "subgraph" in str(e):
                        raise
                    # Not a UUID, continue validation
                    pass
        else:
            raise ValueError("Invalid workflow payload; 'prompt' must be a dict.")

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
                
                # Check for validation errors
                if "error" in resp_json:
                    error_info = resp_json["error"]
                    logging.error("=" * 60)
                    logging.error("COMFYUI VALIDATION ERROR")
                    logging.error("=" * 60)
                    logging.error(f"Error: {error_info}")
                    logging.error("=" * 60)
                    
                    # Log the full workflow for debugging
                    logging.error("WORKFLOW THAT FAILED:")
                    logging.error(json.dumps(payload_prompt, indent=2))
                    logging.error("=" * 60)
                    
                    raise RuntimeError(f"ComfyUI rejected workflow: {error_info}")
                    
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8")
                try:
                    error_json = json.loads(detail)
                    if "error" in error_json:
                        logging.error("=" * 60)
                        logging.error("COMFYUI HTTP ERROR")
                        logging.error("=" * 60)
                        logging.error(f"Status: {e.code}")
                        logging.error(f"Error: {error_json['error']}")
                        
                        if "node_errors" in error_json:
                            logging.error("\nNode Errors:")
                            for node_id, node_error in error_json["node_errors"].items():
                                logging.error(f"  Node {node_id}:")
                                logging.error(f"    {node_error}")
                        
                        logging.error("=" * 60)
                        detail = json.dumps(error_json, indent=2)
                except:
                    pass
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
        """
        Get the output of a completed workflow by listening on WS in a separate thread.
        Uses polling for cancellation checks (no real-time subscriptions).
        """
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

        try:
            while True:
                if shutdown_event and shutdown_event.is_set():
                    logging.warning("Shutdown event received, interrupting workflow.")
                    return "interrupted"

                if (time.time() - start_ts) > timeout_sec:
                    logging.warning(f"Workflow timed out after {timeout_sec}s for prompt_id: {prompt_id}")
                    break

                # Polling for cancellation (every 5 seconds)
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