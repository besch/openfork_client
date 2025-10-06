import websocket
import uuid
import json
import os
import time
import urllib.request
import urllib.error
import threading
from queue import Queue, Empty
import logging
from typing import Union
import requests

class ComfyUIClient:
    def __init__(self, comfyui_ws_url: str):
        self.comfyui_ws_url = comfyui_ws_url
        self.http_base = self._http_base_from_ws(comfyui_ws_url)

    def _http_base_from_ws(self, ws_url: str) -> str:
        # ws://host:port/ws?... -> http://host:port
        # wss://host:port/ws?... -> https://host:port
        base = ws_url.split("/ws")[0]
        if base.startswith("wss://"):
            return base.replace("wss://", "https://")
        return base.replace("ws://", "http://")

    def wait_for_ready(self, shutdown_event: threading.Event, timeout=180):
        """Waits for the ComfyUI server to be available."""
        logging.info("Waiting for ComfyUI server to be ready...")
        start_time = time.time()
        url = f"{self.http_base}/object_info"
        while time.time() - start_time < timeout:
            if shutdown_event.is_set():
                logging.warning("Shutdown requested while waiting for ComfyUI.")
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

            shutdown_event.wait(5)
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
            error_message = f"Cannot reach ComfyUI at {self.server_address} (/prompt): {reason}. Check COMFYUI_WS_URL and that ComfyUI is running and the port is published."
            logging.error(error_message)
            raise RuntimeError(error_message)

        return prompt_id

    def _ws_reader_thread(self, ws, q):
        while True:
            try:
                out = ws.recv()
                q.put(out)
            except Exception as e:
                q.put(e) # Signal error to the main thread
                break

    def get_workflow_output(self, prompt_id: str, terminal_node_ids: Union[list[str], None] = None, timeout_sec: int = 7200, shutdown_event: threading.Event = None) -> Union[dict, None, str]:
        """Get the output of a completed workflow by listening on WS in a separate thread."""
        if not prompt_id:
            logging.warning("get_workflow_output called with empty prompt_id.")
            return None

        client_id = str(uuid.uuid4())
        ws = websocket.WebSocket()
        try:
            ws.connect(self.comfyui_ws_url.format(client_id), timeout=600)
            logging.info(f"Successfully connected to ComfyUI WebSocket at {self.comfyui_ws_url.format(client_id)}")
        except Exception as e:
            logging.error(f"Failed to connect to ComfyUI WebSocket at {self.comfyui_ws_url.format(client_id)}: {e}")
            raise RuntimeError(f"Failed to connect to ComfyUI WebSocket at {self.comfyui_ws_url}: {e}")

        q = Queue()
        reader_thread = threading.Thread(target=self._ws_reader_thread, args=(ws, q), daemon=True)
        reader_thread.start()
        logging.info("WebSocket reader thread started.")

        start_ts = time.time()
        all_node_outputs = {}
        executed_nodes = set()

        try:
            while True:
                if shutdown_event and shutdown_event.is_set():
                    logging.warning("Shutdown event received, interrupting workflow output wait.")
                    return "interrupted"

                if (time.time() - start_ts) > timeout_sec:
                    logging.warning(f"Workflow output timed out after {timeout_sec} seconds for prompt_id: {prompt_id}. Breaking loop to fetch history.")
                    break

                try:
                    out = q.get(timeout=2)  # Use a short timeout to allow checking the shutdown event
                    logging.debug(f"Received raw WebSocket message: {out}")
                    if isinstance(out, Exception):
                        logging.error(f"Exception in WebSocket reader thread: {out}")
                        raise out
                except Empty:
                    logging.debug("Queue empty, continuing to wait for messages.")
                    continue

                if isinstance(out, str):
                    try:
                        message = json.loads(out)
                        logging.debug(f"Parsed WebSocket message: {json.dumps(message, indent=2)}")
                    except Exception as e:
                        logging.error(f"Failed to parse WebSocket message as JSON: {out}. Error: {e}")
                        continue

                    mtype = message.get("type")
                    data = message.get("data", {}) if isinstance(message, dict) else {}
                    if not isinstance(data, dict):
                        data = {}

                    if mtype == "executed" and data.get("prompt_id") == prompt_id:
                        node_id = data.get("node") or data.get("node_id") or data.get("node_id_name")
                        if node_id is not None:
                            executed_nodes.add(str(node_id))
                            logging.info(f"Executed message for prompt_id {prompt_id}, node_id: {node_id}. Executed nodes count: {len(executed_nodes)}")

                        if "output" in data and node_id is not None:
                            all_node_outputs[str(node_id)] = data["output"]
                            logging.info(f"Stored output for node_id {node_id}. Current all_node_outputs keys: {all_node_outputs.keys()}")

                        if terminal_node_ids:
                            need = {str(n) for n in terminal_node_ids}
                            if need and need.issubset(executed_nodes):
                                logging.info(f"All terminal nodes {terminal_node_ids} executed for prompt_id {prompt_id}. Breaking loop to fetch history.")
                                break

                    elif mtype == "status":
                        status = data.get("status") if isinstance(data, dict) else None
                        if isinstance(status, dict):
                            exec_info = status.get("exec_info", {})
                            if isinstance(exec_info, dict) and exec_info.get("queue_remaining") == 0:
                                logging.info(f"ComfyUI queue is empty for prompt_id {prompt_id}. Breaking loop to fetch history.")
                                break
                    else:
                        logging.info(f"Received WebSocket message of type: {mtype}. Data keys: {data.keys()}")
        finally:
            try:
                ws.close()
                logging.info("WebSocket connection closed.")
            except Exception as e:
                logging.error(f"Error closing WebSocket connection: {e}")

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