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

    # Expect DGN client to provide ComfyUI API graph dict under 'prompt'.
    # Accept either {'prompt': {...}} or a bare API dict (ids -> node dicts).
    payload_prompt = None
    if isinstance(workflow_json, dict) and "prompt" in workflow_json and isinstance(workflow_json["prompt"], dict):
        payload_prompt = workflow_json["prompt"]
    elif isinstance(workflow_json, dict) and all(isinstance(k, (str, int)) and isinstance(v, dict) for k, v in workflow_json.items()):
        payload_prompt = workflow_json
    else:
        raise ValueError("Invalid workflow payload; expected ComfyUI API graph dict under 'prompt' or a bare API graph dict.")

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

def get_workflow_output(prompt_id):
    """Get the output of a completed workflow by listening on WS."""
    if not prompt_id:
        return None

    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    try:
        ws.connect(COMFYUI_URL.format(client_id), timeout=10)
    except Exception as e:
        raise RuntimeError(f"Failed to connect to ComfyUI WebSocket at {COMFYUI_URL}: {e}")

    # Allow longer job runtimes before timing out
    ws.settimeout(300)
    try:
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                # ComfyUI streams multiple messages; wait for the one matching our prompt_id
                if message.get("type") == "executed":
                    data = message.get("data", {})
                    if data.get("prompt_id") == prompt_id:
                        # data.output contains node outputs
                        return data.get("output")
    finally:
        try:
            ws.close()
        except Exception:
            pass
