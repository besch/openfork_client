import json
import os
import sys
import subprocess
import logging
import re
import requests
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime

# Add the parent directory (client) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
import asyncio
from supabase.client import create_client, AsyncClient


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

def analyze_workflow_json(workflow_json: Dict, custom_node_registry: Dict) -> Dict:
    """Analyzes a workflow JSON to extract models, custom nodes (cnr_ids), and potential inputs."""
    model_urls = set()
    custom_node_git_urls = set()
    inputs = {}
    
    standard_nodes = get_standard_nodes()
    
    # Determine format: LiteGraph has a 'nodes' list, API format has a dict of nodes.
    is_api_format = 'nodes' not in workflow_json and isinstance(workflow_json, dict)

    if is_api_format:
        # --- API FORMAT PARSING ---
        node_items = workflow_json.items()

        for node_id, node in node_items:
            node_class_type = node.get("class_type")
            if not node_class_type:
                continue

            # Custom Node Detection (by cnr_id in properties)
            properties = node.get('properties', {})
            cnr_id = properties.get('cnr_id')
            if cnr_id and cnr_id != 'comfy-core': # 'comfy-core' is whitelisted/standard
                git_url = custom_node_registry.get(cnr_id)
                if git_url:
                    custom_node_git_urls.add(git_url)
                else:
                    logger.warning(f"Custom node with cnr_id '{cnr_id}' found but no Git URL in registry. Skipping.")
            elif node_class_type not in standard_nodes: # Fallback to class_type if no cnr_id
                # This might be a custom node without a cnr_id, or a subgraph. For now, collect cnr_ids.
                pass

            # Input and Model Detection
            if 'inputs' in node and isinstance(node['inputs'], dict):
                for input_name, input_value in node['inputs'].items():
                    # Model Detection from markdown links
                    if isinstance(input_value, str):
                        # Markdown link: [filename.safetensors](url)
                        for match in re.finditer(r'\\\[([^\\]+?\\.safetensors)]\(([^)]+)\)', input_value):
                            model_urls.add(match.group(2))

                    is_link = (isinstance(input_value, list) and
                               len(input_value) == 2 and
                               isinstance(input_value[0], str) and
                               isinstance(input_value[1], int))

                    if not is_link:
                        # This is a widget input or a model name
                        
                        # Model Detection (from input_name heuristics, if not already found by markdown link)
                        if input_name.lower() in ['ckpt_name', 'model_name', 'lora_name', 'vae_name', 'control_net_name'] and isinstance(input_value, str):
                            # This is a model name, but we need its URL. We'll rely on markdown links for URLs.
                            pass # We already handle model_urls via markdown links

                        # Input Schema Detection (heuristic)
                        json_schema_type = 'string'
                        if isinstance(input_value, (int, float)):
                            json_schema_type = 'number'
                        elif isinstance(input_value, bool):
                            json_schema_type = 'boolean'

                        key_name_base = input_name
                        if key_name_base.lower() == 'text' and node_class_type == 'CLIPTextEncode':
                            key_name_base = f'text_prompt_{node_id}'
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
                            'description': f'{input_name} (node {node_id})'
                        }
    else:
        # --- LITEGRAPH FORMAT PARSING (Not expected for Comfy-Org templates, but kept for robustness) ---
        nodes = workflow_json.get('nodes', [])
        for node in nodes:
            node_type = node.get('type')
            if not node_type:
                continue

            # Custom Node Detection (by cnr_id in properties)
            properties = node.get('properties', {})
            cnr_id = properties.get('cnr_id')
            if cnr_id and cnr_id != 'comfy-core':
                git_url = custom_node_registry.get(cnr_id)
                if git_url:
                    custom_node_git_urls.add(git_url)
                else:
                    logger.warning(f"Custom node with cnr_id '{cnr_id}' found but no Git URL in registry. Skipping.")
            elif node_type not in standard_nodes:
                pass

            widgets = node.get('widgets_values', [])
            if widgets:
                for widget_val in widgets:
                    if isinstance(widget_val, str):
                        # Markdown link: [filename.safetensors](url)
                        for match in re.finditer(r'\\\[([^\\]+?\\.safetensors)]\(([^)]+)\)', widget_val):
                            model_urls.add(match.group(2))

                        # Model Detection (from widget_val heuristics, if not already found by markdown link)
                        if any(widget_val.endswith(ext) for ext in ['.safetensors', '.pth', '.ckpt', '.bin']):
                            pass # We already handle model_urls via markdown links

            input_widgets = []
            if 'inputs' in node:
                for input_def in node.get('inputs', []):
                    if isinstance(input_def, dict) and input_def.get('link') is None and 'name' in input_def:
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

                    key_name_base = input_name
                    if key_name_base.lower() == 'text' and node_type == 'CLIPTextEncode':
                        title = node.get('title', '').lower()
                        if 'positive' in title or 'prompt' in title:
                            key_name_base = 'positive_prompt'
                        elif 'negative' in title:
                            key_name_base = 'negative_prompt'
                    elif key_name_base.lower() == 'seed':
                        key_name_base = 'seed'
                    
                    final_key_name = key_name_base
                    counter = 1
                    while final_key_name in inputs:
                        final_key_name = f"{key_name_base}_{counter}"
                        counter += 1

                    input_definition = {'type': json_schema_type, 'description': f'{input_name} (node {node.get("id")})'}
                    if isinstance(widget_values, list) and i < len(widget_values):
                        input_definition['default'] = widget_values[i]
                    
                    inputs[final_key_name] = input_definition

    return {
        "model_urls": list(model_urls),
        "custom_node_git_urls": list(custom_node_git_urls),
        "input_schema_properties": inputs
    }


class WorkflowSynchronizer:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase: AsyncClient = create_client(supabase_url, supabase_key)
        self.repo_path = LOCAL_REPO_PATH
        self.current_commit_hash: Optional[str] = None
        self.workflow_previews_bucket = "workflow-previews"
        self.custom_node_registry: Dict[str, str] = {}

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

    async def _get_custom_node_registry(self):
        """Fetches and parses the custom-node-list.json from ComfyUI-Manager's GitHub repo."""
        logger.info(f"Fetching custom node registry from {CUSTOM_NODE_LIST_URL}...")
        try:
            response = requests.get(CUSTOM_NODE_LIST_URL)
            response.raise_for_status() # Raise an exception for HTTP errors
            logger.debug(f"Raw custom node list response: {response.text[:500]}...") # Log first 500 chars
            custom_node_list = response.json()
            
            registry = {}
            for node_entry in custom_node_list.get("custom_nodes", []):
                if "id" in node_entry and "reference" in node_entry:
                    registry[node_entry["id"]] = node_entry["reference"]
            self.custom_node_registry = registry
            logger.info(f"Loaded {len(self.custom_node_registry)} custom node entries from registry.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch custom node list: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse custom node list JSON: {e}")

    async def _upload_preview_asset(self, file_path: Path, destination_name: str) -> Optional[str]:
        """Uploads a preview asset to Supabase Storage and returns its public URL."""
        if not file_path or not file_path.exists():
            logger.warning(f"Preview asset file not found: {file_path}")
            return None

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

            response = await self.supabase.storage.from_(self.workflow_previews_bucket).upload(
                destination_name,  # path argument first
                file_bytes,
                {"content-type": content_type}
            )

            if response.error is None:
                public_url = self.supabase.storage.from_(self.workflow_previews_bucket).get_public_url(destination_name)
                logger.info(f"Uploaded {file_path.name} to {public_url}")
                return public_url
            elif "The resource already exists" in str(response.error):
                public_url = self.supabase.storage.from_(self.workflow_previews_bucket).get_public_url(destination_name)
                logger.info(f"Asset already exists, returning existing URL {public_url}")
                return public_url
            else:
                logger.error(f"Upload failed for {file_path.name}: {response.error}")
                return None
        except Exception as e:
            logger.error(f"Error uploading preview asset {file_path.name}: {e}")
            return None

    async def parse_repository(self) -> List[Dict[str, Any]]:
        """Parses the cloned repository to extract workflow metadata and details."""
        workflows_data = []
        index_file = TEMPLATES_DIR / "index.json"
        if not index_file.exists():
            logger.error(f"Master index file not found: {index_file}")
            return []

        index_data = load_json_file(index_file)
        if not isinstance(index_data, list):
            logger.error(f"Invalid format for {index_file}: Expected a list.")
            return []

        # Fetch custom node registry once
        await self._get_custom_node_registry()

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

                # Analyze workflow JSON for inputs and dependencies
                analysis_result = analyze_workflow_json(workflow_json, self.custom_node_registry)

                # Determine target_entity based on workflow_type_from_category or tags
                target_entity = "scene" # Default
                if workflow_type_from_category == "audio":
                    target_entity = "audio_clip"
                elif workflow_type_from_category == "3d":
                    target_entity = "character" # Assuming 3D models are for characters
                # Further refinement could be done based on tags or specific workflow names

                # Construct input_schema
                input_schema = {
                    "type": "object",
                    "properties": analysis_result["input_schema_properties"],
                    "required": [] # We don't have explicit required info from Comfy-Org templates
                }

                # Determine preview_image_url (local path for now, will be uploaded later)
                uploaded_preview_url = None
                preview_asset_suffix = template_entry.get("mediaSubtype", "webp")
                
                # Check for workflow_name-1.suffix
                preview_asset_path_1 = TEMPLATES_DIR / f"{workflow_name}-1.{preview_asset_suffix}"
                if preview_asset_path_1.exists():
                    uploaded_preview_url = await self._upload_preview_asset(preview_asset_path_1, f"{workflow_name}-1.{preview_asset_suffix}")
                
                # If -1 doesn't exist or failed, check for workflow_name.suffix
                if not uploaded_preview_url:
                    preview_asset_path_no_num = TEMPLATES_DIR / f"{workflow_name}.{preview_asset_suffix}"
                    if preview_asset_path_no_num.exists():
                        uploaded_preview_url = await self._upload_preview_asset(preview_asset_path_no_num, f"{workflow_name}.{preview_asset_suffix}")

                workflows_data.append({
                    "source_repo_identifier": workflow_name,
                    "source_repo_commit_hash": self.current_commit_hash,
                    "name": template_entry.get("title", workflow_name),
                    "description": template_entry.get("description", ""),
                    "category": category_name, # Use category from index.json
                    "preview_image_url": uploaded_preview_url, # Now it's the public URL
                    "workflow_json": workflow_json,
                    "input_schema": input_schema,
                    "workflow_type": workflow_type_from_category, # Use type from index.json
                    "target_entity": target_entity,
                    "hardware_requirements": {"gpu_vram": round(template_entry.get("vram", 0) / (1024**3))} if template_entry.get("vram") else {},
                    "custom_node_urls": analysis_result["custom_node_git_urls"], # Now these are Git URLs
                    "model_urls": analysis_result["model_urls"],
                    "is_public": True, # Assuming all templates from this repo are public
                })
        return workflows_data

    async def _sync_to_database(self, parsed_workflows: List[Dict[str, Any]]):
        """Synchronizes the parsed workflows to the Supabase database."""
        logger.info("Starting database synchronization...")
        
        # Fetch existing workflows from the database
        response = await self.supabase.from_("workflow_templates").select("id, source_repo_identifier, source_repo_commit_hash").execute()
        if response.data is None:
            logger.error(f"Failed to fetch existing workflows: {response.error}")
            return
        
        existing_workflows = {wf["source_repo_identifier"]: wf for wf in response.data if wf["source_repo_identifier"]}
        
        for workflow_data in parsed_workflows:
            identifier = workflow_data["source_repo_identifier"]
            commit_hash = workflow_data["source_repo_commit_hash"]
            
            if identifier in existing_workflows:
                # Check if update is needed
                if existing_workflows[identifier]["source_repo_commit_hash"] != commit_hash:
                    logger.info(f"Updating workflow: {identifier}")
                    response = await self.supabase.from_("workflow_templates").update(workflow_data).eq("source_repo_identifier", identifier).execute()
                    if response.data is None:
                        logger.error(f"Failed to update workflow {identifier}: {response.error}")
                else:
                    logger.info(f"Workflow {identifier} is up-to-date. Skipping.")
            else:
                # Insert new workflow
                logger.info(f"Inserting new workflow: {identifier}")
                response = await self.supabase.from_("workflow_templates").insert(workflow_data).execute()
                if response.data is None:
                    logger.error(f"Failed to insert workflow {identifier}: {response.error}")
        
        # Optional: Deactivate workflows no longer in the repository
        # For now, we won't delete, but could set is_public = False or add a 'deleted_in_repo' flag
        # This would require comparing all existing_workflows with parsed_workflows
        # and marking those not found in parsed_workflows.
        
        logger.info("Database synchronization completed.")

    async def sync_workflows(self):
        """Main synchronization logic will go here."""
        logger.info("Starting workflow synchronization...")
        self.clone_or_pull_repository()
        
        parsed_workflows = await self.parse_repository()
        logger.info(f"Parsed and processed {len(parsed_workflows)} workflows from the repository.")

        await self._sync_to_database(parsed_workflows)
        
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
    asyncio.run(synchronizer.sync_workflows())
