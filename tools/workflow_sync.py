import json
import os
import sys
import subprocess
import logging
import re
import requests
import uuid
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime

# Add the parent directory (client) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
import asyncio
from supabase import create_client, Client


# --- Configuration ---
REPO_URL = "https://github.com/Comfy-Org/workflow_templates.git"
REPO_DIR_NAME = "workflow_templates"
LOCAL_REPO_PATH = Path(Config.ROOT_DIR) / REPO_DIR_NAME
TEMPLATES_DIR = LOCAL_REPO_PATH / "templates"
SCRIPTS_DIR = LOCAL_REPO_PATH / "scripts"
CUSTOM_NODE_LIST_URL = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose logging from httpx and h2
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("h2").setLevel(logging.WARNING)

# --- Helper Functions (adapted from ingest_workflow.py) ---
STANDARD_NODES_FILE = Path(__file__).parent / 'standard_nodes.json'

def load_json_file(path: Path) -> Union[Dict, List, None]:
    """Loads a JSON file from the given path."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"File not found at {path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in {path}")
        return None

def get_standard_nodes() -> List[str]:
    """Loads the list of standard ComfyUI nodes."""
    nodes = load_json_file(STANDARD_NODES_FILE)
    if nodes is None:
        logger.warning("standard_nodes.json not found. Custom node detection may be inaccurate.")
        return []
    return nodes

def analyze_workflow_json(workflow_json: Dict, custom_node_registry: Dict[str, Dict]) -> Dict:
    """
    Analyzes a workflow JSON to extract models, custom nodes (with git URLs and node class_types),
    and a generated input schema.
    """
    model_urls = set()
    custom_node_map: Dict[str, set] = {}  # Maps git_url -> set of class_types
    inputs = {}
    
    standard_nodes = get_standard_nodes()
    
    # Determine format: LiteGraph has a 'nodes' list, API format has a dict of nodes.
    is_api_format = 'nodes' not in workflow_json and isinstance(workflow_json, dict)
    
    nodes_to_process = []
    if is_api_format:
        # Convert API format to a list of nodes with their IDs
        for node_id, node_data in workflow_json.items():
            node_with_id = node_data.copy()
            node_with_id['id'] = node_id
            # API format uses 'class_type', LiteGraph uses 'type'. Standardize to 'type' for processing.
            if 'class_type' in node_with_id:
                node_with_id['type'] = node_with_id.pop('class_type')
            nodes_to_process.append(node_with_id)
    else:
        # LiteGraph format already has a 'nodes' list
        nodes_to_process = workflow_json.get('nodes', [])

    for node in nodes_to_process:
        node_type = node.get('type')
        node_id = node.get('id')
        if not node_type:
            continue

        # --- Custom Node Detection ---
        # Use a more robust detection that checks against the full custom_node_registry
        if node_type not in standard_nodes:
            found_node = False
            # The registry maps a custom node's title/folder to its details including git url
            for reg_key, reg_value in custom_node_registry.items():
                # The registry's `title` is often the node's folder name or a close variant
                # and `files` contains the python files where nodes are defined.
                # A simple heuristic: if the node_type is in the list of nodes provided by a custom_node entry.
                if 'nodes' in reg_value and node_type in reg_value['nodes']:
                    git_url = reg_value.get('reference')
                    if git_url:
                        if git_url not in custom_node_map:
                            custom_node_map[git_url] = set()
                        custom_node_map[git_url].add(node_type)
                        found_node = True
                        break # Found the right registry entry for this node
            if not found_node:
                 logger.warning(f"Custom node '{node_type}' found in workflow but could not be resolved to a git repository. It might be a built-in or subgraph node.")


        # --- Model URL Detection ---
        # This can be found in various places, widgets_values is a common one for LiteGraph
        widgets = node.get('widgets_values', [])
        if widgets:
            for widget_val in widgets:
                if isinstance(widget_val, str):
                    # Regex for markdown links: [filename.safetensors](url)
                    for match in re.finditer(r'\[(?:[^\]]+?\.safetensors)\]\(([^)]+)\)', widget_val):
                        model_urls.add(match.group(1))
        
        # Also check 'inputs' for API format which might contain model names in markdown
        if 'inputs' in node and isinstance(node['inputs'], dict):
            for input_name, input_value in node['inputs'].items():
                if isinstance(input_value, str):
                    for match in re.finditer(r'\[(?:[^\]]+?\.safetensors)\]\(([^)]+)\)', input_value):
                        model_urls.add(match.group(1))


        # --- Input Schema Detection (Heuristics) ---
        # This part remains complex, the existing logic is a good starting point.
        # We'll refine it slightly for clarity.
        
        # Heuristic for LoadImage
        if node_type == 'LoadImage':
            node_title = node.get('title', f'Input Image {node_id}')
            key_name_base = f'input_image_{node_id}'
            if key_name_base not in inputs:
                inputs[key_name_base] = {
                    'type': 'image', # Use a specific type for images
                    'description': node_title,
                    'node_type': 'LoadImage',
                    'field_name': 'image' # The field in the node to update
                }
            continue

        # General input detection
        widget_values = node.get('widgets_values', [])
        widget_names = []

        if 'inputs' in node:
            # This works for both API and LiteGraph formats if 'inputs' is a list of dicts
            if isinstance(node['inputs'], list):
                 widget_names = [i['name'] for i in node['inputs'] if isinstance(i, dict) and i.get('link') is None and 'name' in i]
            # For API format, inputs is a dict, we look at non-linked values
            elif isinstance(node['inputs'], dict):
                 for name, val in node['inputs'].items():
                     if not isinstance(val, list): # Links are lists [node_id, slot_index]
                         widget_names.append(name)
                         widget_values.append(val)


        if not widget_names or not widget_values:
            continue

        for i, input_name in enumerate(widget_names):
            if i >= len(widget_values):
                break
            
            input_value = widget_values[i]

            # Filter out non-user-configurable settings
            if input_name.lower().endswith(('_name', '.name')) or input_name.lower() in ['model', 'clip', 'vae', 'latent', 'image', 'pixels', 'control_after_generate', 'sampler_name', 'scheduler']:
                continue

            json_schema_type = 'string'
            if isinstance(input_value, int):
                json_schema_type = 'integer'
            elif isinstance(input_value, float):
                json_schema_type = 'number'
            elif isinstance(input_value, bool):
                json_schema_type = 'boolean'

            key_name_base = input_name
            if 'prompt' in key_name_base.lower() or 'text' in key_name_base.lower():
                title = node.get('title', '').lower()
                if 'negative' in title:
                    key_name_base = 'negative_prompt'
                else:
                    key_name_base = 'prompt'
            elif key_name_base.lower() == 'seed':
                key_name_base = 'seed'
            
            final_key_name = key_name_base
            counter = 1
            while final_key_name in inputs:
                final_key_name = f"{key_name_base}_{counter}"
                counter += 1
            
            inputs[final_key_name] = {
                'type': json_schema_type,
                'default': input_value,
                'description': f'{input_name} for {node_type}',
                'node_type': node_type,
                'field_name': input_name
            }

    # Convert the custom_node_map to the desired JSONB format
    custom_node_dependencies = [
        {"url": url, "nodes": list(nodes)} for url, nodes in custom_node_map.items()
    ]

    return {
        "model_urls": list(model_urls),
        "custom_node_dependencies": custom_node_dependencies,
        "input_schema_properties": inputs
    }


class WorkflowSynchronizer:
    def _unique_key(self, props: Dict, base: str, counter: Dict) -> str:
        key = base
        c = counter.get(base, 0)
        while key in props:
            c += 1
            key = f"{base}_{c}"
        counter[base] = c
        return key

    def extract_inputs_from_litegraph(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts user-editable inputs → JSON Schema for dynamic form."""
        schema = {'type': 'object', 'properties': {}}
        props: Dict[str, Any] = schema['properties']
        counter: Dict[str, int] = {}
        
        defs = workflow.get('definitions', {}).get('subgraphs', [])
        for node in workflow['nodes']:
            ntype = node['type']
            widgets = node.get('widgets_values', [])
            title = node.get('title', '').lower()
            
            # 1. SUBGRAPHS (90% win: wan2, qwen, ace, video)
            try:
                uuid.UUID(ntype)  # Is UUID?
                subgraph = next((s for s in defs if s['id'] == ntype), None)
                if subgraph:
                    for i, inp in enumerate(subgraph.get('inputs', [])):
                        name = inp['name'].lower().replace(' ', '_')
                        itype = inp['type'].lower()
                        default = widgets[i] if i < len(widgets) else None
                        
                        key = self._unique_key(props, name, counter)
                        
                        if itype == 'image':
                            props[key] = {'type': 'string', 'format': 'uri', 'default': '', 'description': f'Upload {name.title()}'}
                        elif itype in ['string', 'text'] or 'prompt' in name or 'tags' in name or 'lyrics' in name:
                            props[key] = {'type': 'string', 'default': str(default) if default else '', 'description': name.title()}
                        elif itype in ['int', 'number', 'width', 'height', 'steps', 'seed', 'length', 'seconds']:
                            props[key] = {'type': 'number', 'default': float(default) if default is not None else 512, 'description': name.title()}
                        elif itype == 'boolean':
                            props[key] = {'type': 'boolean', 'default': bool(default) if default is not None else False, 'description': name.title()}
                    continue
            except ValueError:
                pass
            
            # 2. REGULAR NODES
            if ntype == 'CLIPTextEncode' and widgets:
                key = 'positive_prompt' if 'pos' in title or 'positive' in title else 'negative_prompt'
                key = self._unique_key(props, key, counter)
                props[key] = {'type': 'string', 'default': str(widgets[0]), 'description': key.replace('_', ' ').title()}
            elif ntype == 'EmptyLatentImage' and len(widgets) >= 2:
                props['width'] = {'type': 'number', 'default': float(widgets[0]), 'description': 'Width'}
                props['height'] = {'type': 'number', 'default': float(widgets[1]), 'description': 'Height'}
            elif ntype == 'LoadImage' and widgets:
                key = self._unique_key(props, 'input_image', counter)
                props[key] = {'type': 'string', 'format': 'uri', 'default': '', 'description': f'Upload Image ({node.get("title", "Input")})'}
            elif 'TextToImage' in ntype or 't2i' in ntype.lower():  # API nodes like WanTextToImageApi
                if len(widgets) > 1:
                    props['prompt'] = {'type': 'string', 'default': str(widgets[1]), 'description': 'Prompt'}
                    if len(widgets) > 2 and widgets[2]:
                        props['negative_prompt'] = {'type': 'string', 'default': str(widgets[2]), 'description': 'Negative Prompt'}
        
        return schema

    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.repo_path = LOCAL_REPO_PATH
        self.current_commit_hash: Optional[str] = None
        self.workflow_previews_bucket = "workflow-previews"
        self.custom_node_registry: Dict[str, Dict] = {}

    def _run_git_command(self, command: List[str]) -> str:
        """Runs a git command in the local repository directory."""
        try:
            result = subprocess.run(
                ["git"] + command,
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {' '.join(command)}\nStdout: {e.stdout}\nStderr: {e.stderr}")
            raise

    def clone_or_pull_repository(self):
        """Clones the repository if it doesn't exist, otherwise pulls the latest changes."""
        if not self.repo_path.exists():
            logger.info(f"Cloning {REPO_URL} into {self.repo_path}...")
            subprocess.run(["git", "clone", REPO_URL, self.repo_path], check=True)
            logger.info("Repository cloned successfully.")
        else:
            logger.info(f"Pulling latest changes for {self.repo_path}...")
            self._run_git_command(["pull", "origin", "main"])
            logger.info("Repository updated successfully.")
        
        # Get current commit hash
        self.current_commit_hash = self._run_git_command(["rev-parse", "HEAD"])
        logger.info(f"Current repository commit hash: {self.current_commit_hash}")

    def _get_custom_node_registry(self):
        """Fetches and parses the custom-node-list.json from ComfyUI-Manager's GitHub repo."""
        logger.info(f"Fetching custom node registry from {CUSTOM_NODE_LIST_URL}...")
        try:
            response = requests.get(CUSTOM_NODE_LIST_URL)
            response.raise_for_status() # Raise an exception for HTTP errors
            custom_node_list = response.json()
            
            registry = {}
            # The json is a list of custom node entries
            for node_entry in custom_node_list.get("custom_nodes", []):
                # We need to know the nodes provided by each git repo.
                # The 'title' is often the folder name and a good key.
                title = node_entry.get('title')
                git_url = node_entry.get('reference')
                author = node_entry.get('author')
                
                if not title or not git_url:
                    continue

                # The 'files' array lists python files that define the nodes.
                # We need to extract the node class names from them.
                # The `node_list` in the JSON is exactly what we need.
                provided_nodes = node_entry.get('nodes', [])
                
                if provided_nodes:
                    registry[title] = {
                        "reference": git_url,
                        "author": author,
                        "nodes": provided_nodes
                    }

            self.custom_node_registry = registry
            logger.info(f"Loaded {len(self.custom_node_registry)} custom node entries from registry.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch custom node list: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse custom node list JSON: {e}")

    def _upload_preview_asset(self, file_path: Path, destination_name: str) -> Optional[str]:
        """Uploads a preview asset to Supabase Storage and returns its public URL."""
        if not file_path or not file_path.exists():
            logger.warning(f"Preview asset file not found: {file_path}")
            return None
            
        logger.info(f"Uploading preview asset: {file_path} -> {destination_name}")

        try:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            
            # Determine content type more accurately
            content_type = "application/octet-stream"
            if file_path.suffix.lower() == ".webp":
                content_type = "image/webp"
            elif file_path.suffix.lower() == ".mp3":
                content_type = "audio/mpeg"
            elif file_path.suffix.lower() == ".mp4":
                content_type = "video/mp4"
            elif file_path.suffix.lower() == ".png":
                content_type = "image/png"
            elif file_path.suffix.lower() == ".jpg" or file_path.suffix.lower() == ".jpeg":
                content_type = "image/jpeg"

            try:
                self.supabase.storage.from_(self.workflow_previews_bucket).upload(
                    path=destination_name,
                    file=file_bytes,
                    file_options={"contentType": content_type}
                )
            except Exception as e:
                if "The resource already exists" in str(e):
                    logger.info(f"Asset {destination_name} already exists")
                else:
                    logger.error(f"Upload failed for {file_path.name}: {str(e)}")
                    return None

            # Get public URL after successful upload or if file already exists
            try:
                public_url = self.supabase.storage.from_(self.workflow_previews_bucket).get_public_url(destination_name)
                logger.info(f"Public URL for {file_path.name}: {public_url}")
                return public_url
            except Exception as e:
                logger.error(f"Failed to get public URL for {destination_name}: {str(e)}")
                return None
                return None
        except Exception as e:
            logger.error(f"Error uploading preview asset {file_path.name}: {e}")
            return None

    def parse_repository(self) -> List[Dict[str, Any]]:
        """Parses the cloned repository to extract workflow metadata and details."""
        workflows_data = []
        index_file = TEMPLATES_DIR / "index.json"
        if not index_file.exists():
            logger.error(f"Index file not found at {index_file}")
            return []

        index_data = load_json_file(index_file)
        if not isinstance(index_data, list):
            logger.error(f"Invalid format for {index_file}: Expected a list.")
            return []

        # Fetch custom node registry once
        self._get_custom_node_registry()

        for category_entry in index_data:
            category_name = category_entry.get("category", "General")
            workflow_type_from_category = category_entry.get("type", "unknown")
            
            # Filter out workflows from 'CLOSED SOURCE MODELS' category
            if category_name == "CLOSED SOURCE MODELS":
                logger.info(f"Skipping category '{category_name}' as it contains closed-source models.")
                continue

            for template_entry in category_entry.get("templates", []):
                workflow_name = template_entry.get("name")
                if not workflow_name:
                    logger.warning(f"Skipping template with no name in category {category_name}.")
                    continue

                workflow_json_file = TEMPLATES_DIR / f"{workflow_name}.json"
                if not workflow_json_file.exists():
                    logger.warning(f"Workflow JSON file not found for {workflow_name}: {workflow_json_file}. Skipping.")
                    continue

                workflow_json = load_json_file(workflow_json_file)
                if not workflow_json:
                    logger.warning(f"Could not load workflow JSON for {workflow_name}. Skipping.")
                    continue

                # Analyze workflow for models and custom nodes, but ignore its input schema
                analysis_result = analyze_workflow_json(workflow_json, self.custom_node_registry)

                # Generate a proper input schema using the new parser
                input_schema = self.extract_inputs_from_litegraph(workflow_json)

                # Determine target_entity based on workflow_type_from_category or tags
                target_entity = "scene"  # Default
                if workflow_type_from_category == "audio":
                    target_entity = "audio_clip"
                elif workflow_type_from_category == "3d":
                    target_entity = "character"  # Assuming 3D models are for characters

                # Convert LiteGraph to a basic node dictionary if necessary
                api_formatted_workflow = workflow_json
                if 'nodes' in workflow_json and isinstance(workflow_json.get('nodes'), list):
                    logger.info(f"Converting LiteGraph format to node dictionary for {workflow_name}")
                    converted_workflow = {}
                    for node in workflow_json['nodes']:
                        node_copy = node.copy()
                        # The API format uses 'class_type' but LiteGraph uses 'type'.
                        if 'type' in node_copy:
                            node_copy['class_type'] = node_copy.pop('type')
                        # The node ID is the key in the API format, not a field in the object.
                        if 'id' in node_copy:
                            node_id = str(node_copy.pop('id'))
                            converted_workflow[node_id] = node_copy
                    api_formatted_workflow = converted_workflow

                # Determine preview_image_url (local path for now, will be uploaded later)
                uploaded_preview_url = None
                preview_asset_suffix = template_entry.get("mediaSubtype", "webp")
                
                # Check for workflow_name-1.suffix
                preview_asset_path_1 = TEMPLATES_DIR / f"{workflow_name}-1.{preview_asset_suffix}"
                if preview_asset_path_1.exists():
                    uploaded_preview_url = self._upload_preview_asset(preview_asset_path_1, f"{workflow_name}-1.{preview_asset_suffix}")
                
                # If -1 doesn't exist or failed, check for workflow_name.suffix
                if not uploaded_preview_url:
                    preview_asset_path_no_num = TEMPLATES_DIR / f"{workflow_name}.{preview_asset_suffix}"
                    if preview_asset_path_no_num.exists():
                        uploaded_preview_url = self._upload_preview_asset(preview_asset_path_no_num, f"{workflow_name}.{preview_asset_suffix}")

                workflows_data.append({
                    "source_repo_identifier": workflow_name,
                    "source_repo_commit_hash": self.current_commit_hash,
                    "name": template_entry.get("title", workflow_name),
                    "description": template_entry.get("description", ""),
                    "category": category_name, # Use category from index.json
                    "preview_image_url": uploaded_preview_url, # Now it's the public URL
                    "workflow_json": api_formatted_workflow,
                    "input_schema": input_schema,
                    "workflow_type": workflow_type_from_category, # Use type from index.json
                    "target_entity": target_entity,
                    "hardware_requirements": {"gpu_vram": round(template_entry.get("vram", 0) / (1024**3))} if template_entry.get("vram") else {},
                    "custom_node_dependencies": analysis_result["custom_node_dependencies"], # Now these are Git URLs
                    "model_urls": analysis_result["model_urls"],
                    "is_public": True, # Assuming all templates from this repo are public
                })
        return workflows_data

    def _sync_to_database(self, parsed_workflows: List[Dict[str, Any]]):
        """Synchronizes the parsed workflows to the Supabase database."""
        logger.info("Starting database synchronization...")
        
        # Fetch existing workflows from the database
        try:
            response = self.supabase.table("workflow_templates").select("id, source_repo_identifier, source_repo_commit_hash").execute()
            existing_workflows = {wf["source_repo_identifier"]: wf for wf in response.data if wf["source_repo_identifier"]}
        except Exception as e:
            logger.error(f"Failed to fetch existing workflows from database: {str(e)}")
            return
        
        for workflow_data in parsed_workflows:
            identifier = workflow_data["source_repo_identifier"]
            commit_hash = workflow_data["source_repo_commit_hash"]
            
            # The data to be inserted or updated.
            # custom_node_dependencies is now a JSONB field.
            db_payload = {
                "source_repo_identifier": identifier,
                "source_repo_commit_hash": commit_hash,
                "name": workflow_data["name"],
                "description": workflow_data["description"],
                "category": workflow_data["category"],
                "preview_image_url": workflow_data["preview_image_url"],
                "workflow_json": workflow_data["workflow_json"],
                "input_schema": workflow_data["input_schema"],
                "workflow_type": workflow_data["workflow_type"],
                "target_entity": workflow_data["target_entity"],
                "hardware_requirements": workflow_data["hardware_requirements"],
                "custom_node_dependencies": workflow_data["custom_node_dependencies"], # Changed from custom_node_urls
                "model_urls": workflow_data["model_urls"],
                "is_public": workflow_data["is_public"],
            }

            if identifier in existing_workflows:
                # For now, we will always update if the workflow exists.
                # A more sophisticated check could compare commit_hash.
                try:
                    logger.info(f"Updating workflow: {identifier}")
                    self.supabase.table("workflow_templates").update(db_payload).eq("source_repo_identifier", identifier).execute()
                except Exception as e:
                    logger.error(f"Failed to update workflow {identifier}: {str(e)}")
            else:
                # Insert new workflow
                try:
                    logger.info(f"Inserting new workflow: {identifier}")
                    self.supabase.table("workflow_templates").insert(db_payload).execute()
                except Exception as e:
                    logger.error(f"Failed to insert workflow {identifier}: {str(e)}")
        
        logger.info("Database synchronization completed.")

    def sync_workflows(self):
        """Main synchronization logic will go here."""
        logger.info("Starting workflow synchronization...")
        self.clone_or_pull_repository()
        
        parsed_workflows = self.parse_repository()
        logger.info(f"Parsed and processed {len(parsed_workflows)} workflows from the repository.")

        self._sync_to_database(parsed_workflows)
        
        logger.info("Workflow synchronization completed.")

if __name__ == "__main__":
    # A simple list of standard nodes to help differentiate custom ones.
    # This could be expanded or loaded from a more comprehensive source.
    standard_nodes_list = [
        "KSampler", "KSamplerAdvanced", "CheckpointLoaderSimple", "CLIPTextEncode",
        "VAEDecode", "VAEEncode", "SaveImage", "LoadImage", "EmptyLatentImage",
        "LoraLoader", "CLIPSetLastLayer", "ControlNetApplyAdvanced", "ControlNetLoader",
        "VAELoader", "HypernetworkLoader", "Note", "PrimitiveNode",
        "ImageOnlyCheckpointLoader", "ControlNetLoader", "ControlNetApply",
        "CLIPVisionEncode", "ImageScale", "LatentUpscale", "LatentFromImage",
        "ImageToLatent", "LatentToImage", "VAEEncode", "VAEDecode",
        "SetNodeInput", "GetNodeInput", "Reroute", "Primitive", "AnythingEverywhere" # Common nodes
    ]
    if not STANDARD_NODES_FILE.exists():
        with open(STANDARD_NODES_FILE, 'w', encoding='utf-8') as f:
            json.dump(standard_nodes_list, f, indent=2)

    supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        logger.error("Supabase URL or Service Role Key not found. Please set NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables.")
        sys.exit(1)

    synchronizer = WorkflowSynchronizer(supabase_url, supabase_key)
    synchronizer.sync_workflows()
