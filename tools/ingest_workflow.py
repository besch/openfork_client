import sys
import os

# Add the parent directory (client) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import re
import requests
from typing import Dict, List, Any, Union
from dotenv import load_dotenv
from config import Config

# --- Globals ---
CACHE_FILE = os.path.join(os.path.dirname(__file__), 'dependency_cache.json')
STANDARD_NODES_FILE = os.path.join(os.path.dirname(__file__), 'standard_nodes.json')

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')) # Load from client/.env

# --- Helper Functions ---

def get_input(prompt: str, default: Any = None) -> str:
    """Helper to get user input with an optional default value."""
    user_input = input(f"{prompt} [{default}]: ")
    if user_input == "":
        return str(default) if default is not None else "" # Return default as string, or empty string if default is None
    return user_input

def load_json_file(path: str) -> Union[Dict, List, None]:
    """Loads a JSON file from the given path."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found at {path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {path}")
        return None

def save_json_file(path: str, data: Any):
    """Saves data to a JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def is_url(s: str) -> bool:
    # Simple regex to check if string looks like a URL
    return re.match(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', s) is not None

def load_cache() -> Dict:
    """Loads the dependency cache file."""
    cache = load_json_file(CACHE_FILE)
    if cache is None:
        return {"models": {}, "custom_nodes": {}}
    return cache

def save_cache(cache: Dict):
    """Saves the dependency cache file."""
    save_json_file(CACHE_FILE, cache)

def get_standard_nodes() -> List[str]:
    """Loads the list of standard ComfyUI nodes."""
    nodes = load_json_file(STANDARD_NODES_FILE)
    if nodes is None:
        print("Warning: standard_nodes.json not found. Custom node detection may be inaccurate.")
        return []
    return nodes

# --- Workflow Analysis Functions ---

def analyze_workflow(workflow: Dict) -> Dict:
    """Analyzes a workflow to extract models, custom nodes, and potential inputs."""
    models = set()
    custom_nodes = set()
    inputs = {}
    
    standard_nodes = get_standard_nodes()
    
    # --- Node Iteration ---
    nodes = workflow.get('nodes', []) # LiteGraph format
    if not nodes and isinstance(workflow, dict): # API format
        nodes = list(workflow.values())

    for node in nodes:
        node_type = node.get('type') or node.get('class_type')
        if not node_type:
            continue

        # --- Custom Node Detection ---
        if node_type not in standard_nodes:
            custom_nodes.add(node_type)

        # --- Model Detection ---
        widgets = node.get('widgets_values', [])
        if widgets:
            for widget_val in widgets:
                if isinstance(widget_val, str):
                    # Check if it's a direct URL to a model
                    if is_url(widget_val) and any(widget_val.endswith(ext) for ext in ['.safetensors', '.pth', '.ckpt', '.bin']):
                        # Store the URL directly
                        models.add(widget_val) # Store the URL, not just basename
                    # If not a URL, but a model filename
                    elif any(widget_val.endswith(ext) for ext in ['.safetensors', '.pth', '.ckpt', '.bin']):
                        # Store the basename, and the main loop will ask for URL if not cached
                        models.add(os.path.basename(widget_val))

        # --- Improved Input Detection ---
        input_widgets = []
        if 'inputs' in node:
            for input_def in node['inputs']:
                if input_def.get('link') is None and 'name' in input_def: # Check if it's an unconnected widget input
                    input_widgets.append(input_def)

        if input_widgets:
            widget_values = node.get('widgets_values', [])
            for i, input_def in enumerate(input_widgets):
                input_name = input_def['name']
                comfyui_type = input_def['type']

                json_schema_type = 'string'
                if comfyui_type in ['INT', 'FLOAT']:
                    json_schema_type = 'number'
                elif comfyui_type == 'BOOLEAN':
                    json_schema_type = 'boolean'
                elif comfyui_type == 'IMAGE':
                    json_schema_type = 'image'

                # Generate a unique key, prioritizing common names
                key_name_base = input_name # Use a base name for uniqueness check
                
                # Apply heuristics for common names
                if key_name_base.lower() == 'text' and node_type == 'CLIPTextEncode':
                    if 'positive' in node.get('title', '').lower() or 'prompt' in node.get('title', '').lower():
                        key_name_base = 'positive_prompt'
                    elif 'negative' in node.get('title', '').lower():
                        key_name_base = 'negative_prompt'
                    else:
                        key_name_base = 'text_input'
                elif key_name_base.lower() == 'image' and node_type == 'LoadImage':
                    key_name_base = 'input_image'
                elif key_name_base.lower() == 'seed':
                    key_name_base = 'seed'
                elif key_name_base.lower() == 'ckpt_name' or key_name_base.lower() == 'model_name':
                    key_name_base = 'model_name'
                elif key_name_base.lower() == 'lora_name':
                    key_name_base = 'lora_name'
                elif key_name_base.lower() == 'vae_name':
                    key_name_base = 'vae_name'
                elif key_name_base.lower() == 'clip_name':
                    key_name_base = 'clip_name'
                
                # Ensure uniqueness for the final key_name
                final_key_name = key_name_base
                counter = 1
                while final_key_name in inputs:
                    final_key_name = f"{key_name_base}_{counter}"
                    counter += 1

                # Construct the input definition dictionary
                input_definition = {'type': json_schema_type}
                # Safely get default value if widget_values is a list and index is valid
                if isinstance(widget_values, list) and i < len(widget_values):
                    input_definition['default'] = widget_values[i]
                
                # Assign the complete definition to the inputs dictionary
                inputs[final_key_name] = input_definition
            
    return {
        "models": list(models),
        "custom_nodes": list(custom_nodes),
        "inputs": inputs
    }

# --- Main Ingestion Logic ---

def main():
    parser = argparse.ArgumentParser(description="Ingest a ComfyUI workflow into the database.")
    parser.add_argument("file_path", help="Path to the ComfyUI API format JSON file.")
    parser.add_argument("--orchestrator-url", type=str, help="URL of the orchestrator (overrides .env)")
    parser.add_argument("--service-role-key", type=str, help="Supabase Service Role Key (overrides .env)")
    args = parser.parse_args()

    # Determine orchestrator URL based on DEV_MODE
    if Config.DEV_MODE:
        determined_orchestrator_url = Config.ORCHESTRATOR_URL_DEV
        if determined_orchestrator_url is None:
            determined_orchestrator_url = "http://localhost:3000" # Fallback if config.py's default failed
    else:
        determined_orchestrator_url = Config.ORCHESTRATOR_URL_PROD
        if determined_orchestrator_url is None:
            determined_orchestrator_url = "https://www.openfork.video" # Fallback if config.py's default failed

    # Get values from config.py and environment variables or command-line arguments
    orchestrator_url = args.orchestrator_url
    if orchestrator_url is None:
        orchestrator_url = determined_orchestrator_url

    service_role_key = args.service_role_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not orchestrator_url:
        raise ValueError("Orchestrator URL not provided. Ensure ORCHESTRATOR_URL_PROD/DEV is set in config.py or .env, or use --orchestrator-url.")
    if not service_role_key:
        raise ValueError("Service Role Key not provided. Set SUPABASE_SERVICE_ROLE_KEY in .env or use --service-role-key.")

    workflow_data = load_json_file(args.file_path)
    if not workflow_data:
        return

    # In API format, the actual graph is the value. We'll just use the whole thing.
    analysis_result = analyze_workflow(workflow_data)
    cache = load_cache()

    print("--- Workflow Analysis Results ---")
    print(f"Detected Models: {analysis_result['models']}")
    print(f"Detected Custom Nodes: {analysis_result['custom_nodes']}")
    print(f"Detected Inputs: {list(analysis_result['inputs'].keys())}")
    print("---------------------------------\n")

    # --- Gather Metadata ---
    print("--- Enter Workflow Metadata ---")
    name = get_input("Workflow Name")
    description = get_input("Description")
    workflow_type = get_input("Workflow Type (e.g., text_to_image, image_to_video)")
    target_entity = get_input("Target Entity (scene, audio_clip, character, project)", default='scene')
    is_public = get_input("Is Public? (true/false)", default='true').lower() == 'true'

    print("\n--- Define Hardware Requirements ---")
    gpu_vram = int(get_input("Minimum GPU VRAM (GB, 0 for none)", default='0'))
    hardware_requirements = {"gpu_vram": gpu_vram} if gpu_vram > 0 else {}

    # --- Gather Dependency URLs ---
    print("\n--- Define Dependencies ---")
    model_urls_to_collect = []
    for item in analysis_result['models']:
        if is_url(item): # If it's already a URL, just add it
            model_urls_to_collect.append(item)
        elif item not in cache['models']: # If it's a model name and not cached, ask for URL
            cache['models'][item] = get_input(f"Enter download URL for model '{item}'")
            model_urls_to_collect.append(cache['models'][item])
        else: # It's a model name and is cached
            model_urls_to_collect.append(cache['models'][item])
    model_urls = model_urls_to_collect

    custom_node_urls = []
    for node_type in analysis_result['custom_nodes']:
        if node_type not in cache['custom_nodes']:
            cache['custom_nodes'][node_type] = get_input(f"Enter Git URL for custom node '{node_type}'")
        custom_node_urls.append(cache['custom_nodes'][node_type])
    
    save_cache(cache)

    # --- Define Input Schema ---
    print("\n--- Review and Define Input Schema ---")
    input_schema = {
        "type": "object",
        "properties": analysis_result['inputs'],
        "required": []
    }
    print("Auto-detected inputs:")
    print(json.dumps(input_schema['properties'], indent=2))
    
    while True:
        add_more = get_input("Add or modify inputs? (y/n)", default='n').lower()
        if add_more != 'y':
            break
        
        input_name = get_input("Input Name (e.g., positive_prompt)")
        input_type = get_input("Input Type (string, number, boolean, image)", default='string')
        is_required = get_input("Is this input required? (y/n)", default='y').lower() == 'y'
        default_value_str = get_input("Default value (optional, press enter to skip)")

        input_definition = {"type": input_type}
        if default_value_str:
            if input_type == 'number':
                input_definition['default'] = float(default_value_str)
            elif input_type == 'boolean':
                input_definition['default'] = default_value_str.lower() == 'true'
            else:
                input_definition['default'] = default_value_str
        
        input_schema['properties'][input_name] = input_definition
        if is_required:
            if input_name not in input_schema['required']:
                input_schema['required'].append(input_name)

    # --- Final Template and Submission ---
    template = {
        "name": name,
        "description": description,
        "workflow_json": workflow_data,
        "input_schema": input_schema,
        "workflow_type": workflow_type,
        "target_entity": target_entity,
        "hardware_requirements": hardware_requirements,
        "custom_node_urls": custom_node_urls,
        "model_urls": model_urls,
        "is_public": is_public,
    }

    print("\n--- Review Workflow Template ---")
    # Create a serializable copy for printing
    template_to_print = template.copy()
    template_to_print['workflow_json'] = "...omitted..."
    print(json.dumps(template_to_print, indent=2))

    confirm = get_input("\nInsert this template into the database? (y/n)", default='y').lower()
    if confirm == 'y':
        ingest_url = f"https://www.openfork.video/api/workflows/ingest"
        headers = {
            "Content-Type": "application/json",
            "apikey": args.service_role_key,
            "Authorization": f"Bearer {args.service_role_key}"
        }
        try:
            response = requests.post(ingest_url, headers=headers, json=template)
            response.raise_for_status()
            
            data = response.json()
            print("\nSuccessfully inserted workflow template!")
            print(f"ID: {data['id']}")
        except requests.exceptions.HTTPError as e:
            print(f"\nError inserting workflow template: {e}\nResponse: {e.response.text}")
        except requests.exceptions.RequestException as e:
            print(f"\nAn error occurred during the request: {e}")

if __name__ == "__main__":
    # A simple list of standard nodes to help differentiate custom ones.
    # This could be expanded or loaded from a more comprehensive source.
    standard_nodes_list = [
        "KSampler", "KSamplerAdvanced", "CheckpointLoaderSimple", "CLIPTextEncode",
        "VAEDecode", "VAEEncode", "SaveImage", "LoadImage", "EmptyLatentImage",
        "LoraLoader", "CLIPSetLastLayer", "ControlNetApplyAdvanced", "ControlNetLoader",
        "VAELoader", "HypernetworkLoader", "Note", "PrimitiveNode"
    ]
    if not os.path.exists(STANDARD_NODES_FILE):
        save_json_file(STANDARD_NODES_FILE, standard_nodes_list)
    
    main()