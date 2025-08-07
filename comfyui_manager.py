import websocket
import uuid
import json
import os
import time
import urllib.request
import urllib.error

# Allow override by env var COMFYUI_WS_URL, default to local
COMFYUI_URL = os.environ.get("COMFYUI_WS_URL", "ws://127.0.0.1:8188/ws?clientId={}")
# Derived HTTP base for REST calls
HTTP_BASE = None

def _http_base_from_ws(ws_url: str) -> str:
    # ws://host:port/ws?... -> http://host:port
    # wss://host:port/ws?... -> https://host:port
    base = ws_url.split("/ws")[0]
    if base.startswith("wss://"):
        return base.replace("wss://", "https://")
    return base.replace("ws://", "http://")

def trigger_workflow(workflow_json):
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
        # Normalize id keys to strings as ComfyUI typically uses string keys
        # (No transform needed here; we just validate)


    # Resolve HTTP base once and cache
    global HTTP_BASE
    if not HTTP_BASE:
        HTTP_BASE = _http_base_from_ws(COMFYUI_URL)
    http_base = HTTP_BASE

    # Quick connectivity probe to aid debugging
    try:
        probe_req = urllib.request.Request(f"{http_base}/object_info")
        with urllib.request.urlopen(probe_req, timeout=5) as resp:
            # just ensure we can connect; content not strictly required
            pass
    except Exception as e:
        raise RuntimeError(f"Cannot reach ComfyUI at {http_base} (/object_info): {e}. "
                           f"Check COMFYUI_WS_URL and that ComfyUI is running and port 8188 is published.")

    body = json.dumps({"prompt": payload_prompt, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(f"{http_base}/prompt", data=body, headers={"Content-Type": "application/json"})

    prompt_id = None
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp_body = resp.read().decode("utf-8")
            try:
                resp_json = json.loads(resp_body)
            except Exception:
                resp_json = {}
            prompt_id = resp_json.get("prompt_id") or resp_json.get("data", {}).get("prompt_id")
    except urllib.error.HTTPError as e:
        # Return HTTP body to help diagnose server-side errors
        try:
            detail = e.read().decode("utf-8")
        except Exception:
            detail = str(e)
        raise RuntimeError(f"ComfyUI /prompt returned HTTP {e.code}: {detail}")
    except Exception as e:
        raise RuntimeError(f"Failed to enqueue workflow via {http_base}/prompt: {e}")

    return prompt_id

def get_workflow_output(prompt_id, terminal_node_ids=None, timeout_sec=7200):
    """
    Get the output of a completed workflow by listening on WS.

    Improvements:
    - Track all executed nodes for our prompt_id.
    - If terminal_node_ids provided, only return after all of them executed.
    - Otherwise, return the last executed output for our prompt_id.
    - Long overall timeout with shorter per-recv timeout to avoid premature failure.
    """
    if not prompt_id:
        return None

    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    try:
        ws.connect(COMFYUI_URL.format(client_id), timeout=10)
    except Exception as e:
        raise RuntimeError(f"Failed to connect to ComfyUI WebSocket at {COMFYUI_URL}: {e}")

    # Per receive timeout; we loop until wall clock timeout
    ws.settimeout(60)

    start_ts = time.time()
    executed_nodes = set()
    last_output = None

    try:
        while True:
            # Wall clock timeout guard
            if (time.time() - start_ts) > timeout_sec:
                # Return whatever we have to avoid hanging forever
                return last_output

            try:
                out = ws.recv()
            except Exception:
                # On transient timeout, continue waiting within wall clock limit
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

                # Track prompt-specific execution
                if mtype == "executed" and data.get("prompt_id") == prompt_id:
                    # Record node_id if provided
                    node_id = data.get("node") or data.get("node_id") or data.get("node_id_name")
                    if node_id is not None:
                        executed_nodes.add(str(node_id))
                    # Save last output seen
                    last_output = data.get("output")

                    # If we know the terminal nodes, check completion
                    if terminal_node_ids:
                        # Normalize to str for comparison
                        need = {str(n) for n in terminal_node_ids}
                        if need.issubset(executed_nodes):
                            return last_output
                    else:
                        # If not specified, heuristic: Comfy often sends a 'status' done message later,
                        # but if this is the only executed we get for this prompt for long time, we still return last_output.
                        # Keep waiting for a 'status' event but also allow return if no terminal set and we receive another
                        # executed with no new nodes for a while. That is handled by wall-clock timeout.
                        pass

                # Some ComfyUI builds also send a 'status' message with an overall status for the prompt
                if mtype == "status":
                    # Example: {"type":"status","data":{"status":{"exec_info":{"queue_remaining":0}},"sid":"..."}}
                    # Not all payloads include prompt_id here; so we still rely on executed nodes above.
                    # If queue_remaining is 0 and we have at least one executed for our prompt, return last output.
                    status = data.get("status") if isinstance(data, dict) else None
                    if isinstance(status, dict):
                        exec_info = status.get("exec_info", {})
                        if isinstance(exec_info, dict) and exec_info.get("queue_remaining") == 0 and last_output is not None:
                            return last_output
    finally:
        try:
            ws.close()
        except Exception:
            pass
