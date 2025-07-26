import websocket
import uuid
import json

COMFYUI_URL = "ws://127.0.0.1:8188/ws?clientId={}"

def trigger_workflow(workflow_json):
    """Trigger a workflow in ComfyUI and return the prompt ID."""
    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.connect(COMFYUI_URL.format(client_id))

    prompt = {"prompt": workflow_json, "client_id": client_id}
    ws.send(json.dumps(prompt))

    # The first message from the server should be the prompt ID
    response = ws.recv()
    response_data = json.loads(response)
    prompt_id = response_data.get('data', {}).get('prompt_id')

    ws.close()
    return prompt_id

def get_workflow_output(prompt_id):
    """Get the output of a completed workflow."""
    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.connect(COMFYUI_URL.format(client_id))

    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executed':
                data = message['data']
                if data['prompt_id'] == prompt_id:
                    ws.close()
                    return data['output']