import os
import logging
import subprocess
import time
import requests
import glob
import json
from typing import List, Dict, Any, Union
from dataclasses import dataclass, field

try:
    from services.workflow_analyzer import WorkflowAnalyzer
except ImportError:
    WorkflowAnalyzer = None


@dataclass
class InstalledNodeInfo:
    """Information about an installed ComfyUI node type."""
    name: str
    category: str
    input_types: Dict[str, Any] = field(default_factory=dict)
    output_types: list = field(default_factory=list)
    description: str = ""

class LocalComfyUIManager:
    def __init__(self, comfyui_install_dir: str = None, comfyui_url: str = "http://127.0.0.1:8188"):
        self.comfyui_install_dir = comfyui_install_dir or self._detect_install_dir()
        if self.comfyui_install_dir:
            logging.info(f"Using ComfyUI installation at: {self.comfyui_install_dir}")
        else:
             logging.warning("No ComfyUI installation detected. Please specify --comfyui-install-dir if you wish to manage it.")

        self.comfyui_url = comfyui_url.strip("/")
        self.process = None
        self._installed_nodes_cache = None
        self._workflow_analyzer = WorkflowAnalyzer() if WorkflowAnalyzer else None

    def _detect_install_dir(self) -> Union[str, None]:
        """Attempts to detect ComfyUI installation in common locations."""
        home = os.path.expanduser("~")
        common_paths = []

        if os.name == 'nt': # Windows
            common_paths = [
                "C:\\ComfyUI_windows_portable",
                "D:\\ComfyUI_windows_portable",
                os.path.join(home, "ComfyUI_windows_portable"),
                os.path.join(home, "ComfyUI"),
                "C:\\ComfyUI",
                "D:\\ComfyUI",
            ]
        else: # Mac/Linux
            common_paths = [
                os.path.join(home, "ComfyUI"),
                "/Applications/ComfyUI",
                "/opt/ComfyUI",
            ]

        for path in common_paths:
            if os.path.exists(os.path.join(path, "main.py")):
                return path
            # Check for ComfyUI folder inside the portable folder
            nested_path = os.path.join(path, "ComfyUI")
            if os.path.exists(os.path.join(nested_path, "main.py")):
                return nested_path
        
        return None

    def is_running(self) -> bool:
        """Check if ComfyUI is reachable at the configured URL."""
        try:
            response = requests.get(f"{self.comfyui_url}/object_info", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def start(self):
        """Attempts to start ComfyUI if a local installation directory is provided."""
        if self.is_running():
            logging.info(f"ComfyUI is already running at {self.comfyui_url}")
            return

        if not self.comfyui_install_dir or not os.path.exists(self.comfyui_install_dir):
            logging.warning("ComfyUI is not running and no valid installation directory provided. Cannot start it.")
            return

        logging.info(f"Attempting to start ComfyUI from {self.comfyui_install_dir}...")
        
        main_py = os.path.join(self.comfyui_install_dir, "main.py")
        if not os.path.exists(main_py):
            logging.error(f"Could not find main.py in {self.comfyui_install_dir}")
            return

        # Attempt to find python executable.
        python_exec = "python"
        
        # Check for python_embed (portable version common in Windows)
        if os.path.exists(os.path.join(self.comfyui_install_dir, "python_embed")):
             potential_python = os.path.join(self.comfyui_install_dir, "python_embed", "python.exe")
             if os.path.exists(potential_python):
                 python_exec = potential_python

        try:
             # Use CREATE_NEW_CONSOLE on Windows to avoid killing it if the client dies, 
             # or maybe keep it attached. For now, separate console is safer for debugging.
             creationheaders = 0
             if os.name == 'nt':
                 creationheaders = subprocess.CREATE_NEW_CONSOLE

             self.process = subprocess.Popen(
                 [python_exec, main_py], 
                 cwd=self.comfyui_install_dir, 
                 creationflags=creationheaders
             )
             logging.info("Started ComfyUI process.")
             
             # Wait for it to come up
             for _ in range(30):
                 if self.is_running():
                     logging.info("ComfyUI started and reachable.")
                     return
                 time.sleep(1)
             
             logging.warning("ComfyUI process started but API is not reachable yet after 30s.")
        except Exception as e:
            logging.error(f"Failed to start ComfyUI: {e}")

    def get_installed_nodes(self) -> Dict[str, InstalledNodeInfo]:
        """Query /object_info to get all installed node types with their inputs/outputs."""
        if self._installed_nodes_cache is not None:
            return self._installed_nodes_cache
            
        nodes = {}
        
        try:
            response = requests.get(f"{self.comfyui_url}/object_info", timeout=10)
            if response.status_code == 200:
                object_info = response.json()
                
                for node_name, node_data in object_info.items():
                    nodes[node_name] = InstalledNodeInfo(
                        name=node_name,
                        category=node_data.get("category", "unknown"),
                        input_types=node_data.get("input", {}),
                        output_types=node_data.get("output", []),
                        description=node_data.get("description", "")
                    )
                
                self._installed_nodes_cache = nodes
                logging.info(f"Discovered {len(nodes)} installed ComfyUI node types")
            else:
                logging.warning(f"Failed to query /object_info: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logging.warning(f"Could not query ComfyUI /object_info: {e}")
        
        return nodes
    
    def validate_workflow(self, workflow_data: dict) -> tuple[bool, list[str]]:
        """Check if all nodes in workflow are installed.
        
        Returns:
            Tuple of (is_valid, list_of_missing_nodes)
        """
        installed = self.get_installed_nodes()
        missing = []
        
        graph = workflow_data.get("prompt", workflow_data)
        
        for node in graph.values():
            if isinstance(node, dict) and "class_type" in node:
                class_type = node["class_type"]
                if class_type not in installed:
                    if class_type not in missing:
                        missing.append(class_type)
        
        return len(missing) == 0, missing

    def invalidate_cache(self):
        """Clear the installed nodes cache to force re-fetching from /object_info."""
        self._installed_nodes_cache = None
        logging.info("Invalidated installed nodes cache")

    def get_node_to_package_map(self) -> Dict[str, str]:
        """
        Load ComfyUI-Manager's extension-node-map.json to map class_type -> package git URL.
        
        Returns:
            Dict mapping class_type names to git repository URLs.
            Example: {"MarkdownNote": "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"}
        """
        node_map = {}
        
        # Try local ComfyUI-Manager first
        if self.comfyui_install_dir:
            manager_path = os.path.join(
                self.comfyui_install_dir, "custom_nodes", "ComfyUI-Manager"
            )
            map_file = os.path.join(manager_path, "extension-node-map.json")
            
            if os.path.exists(map_file):
                try:
                    with open(map_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Format: {"git_url": ["node_class_type1", "node_class_type2", ...], ...}
                    for git_url, node_types in data.items():
                        if isinstance(node_types, list):
                            for node_type in node_types:
                                node_map[node_type] = git_url
                    
                    logging.info(f"Loaded {len(node_map)} node-to-package mappings from local ComfyUI-Manager")
                    return node_map
                except Exception as e:
                    logging.warning(f"Error reading extension-node-map.json: {e}")
        
        # Fall back to fetching from GitHub
        GITHUB_URL = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json"
        try:
            logging.info("Fetching extension-node-map.json from GitHub...")
            response = requests.get(GITHUB_URL, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for git_url, node_types in data.items():
                if isinstance(node_types, list):
                    for node_type in node_types:
                        node_map[node_type] = git_url
            
            logging.info(f"Loaded {len(node_map)} node-to-package mappings from GitHub")
        except Exception as e:
            logging.warning(f"Failed to fetch extension-node-map.json from GitHub: {e}")
        
        return node_map

    def restart(self, timeout_seconds: int = 60) -> bool:
        """
        Stop ComfyUI, wait briefly, then start it again.
        
        Args:
            timeout_seconds: Maximum time to wait for ComfyUI to become ready after restart.
            
        Returns:
            True if successfully restarted and ready, False otherwise.
        """
        if not self.comfyui_install_dir:
            logging.warning("Cannot restart ComfyUI: no installation directory configured")
            return False
        
        logging.info("Restarting ComfyUI to load newly installed nodes...")
        
        # Stop if we have a managed process
        if self.process:
            self.stop()
            time.sleep(2)  # Brief pause after stopping
        
        # Start ComfyUI
        self.start()
        
        # Wait for it to become ready
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if self.is_running():
                # Also invalidate the node cache since we restarted
                self.invalidate_cache()
                logging.info(f"ComfyUI restarted and ready after {int(time.time() - start_time)}s")
                return True
            time.sleep(2)
        
        logging.error(f"ComfyUI failed to become ready within {timeout_seconds}s after restart")
        return False

    def fetch_comfyui_templates(self) -> List[Dict[str, Any]]:
        """Fetch templates list from ComfyUI's /templates API endpoint if available."""
        templates = []
        
        try:
            # ComfyUI may expose templates via /templates endpoint
            response = requests.get(f"{self.comfyui_url}/templates", timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Parse the response - format varies by ComfyUI version
                if isinstance(data, list):
                    for template in data:
                        if isinstance(template, dict):
                            templates.append({
                                "name": template.get("name", "Unknown"),
                                "filename": template.get("file", ""),
                                "path": "",  # API-based, no local path
                                "category": template.get("category", "builtin"),
                                "source": "builtin_api",
                                "source_name": "ComfyUI",
                                "input_schema": [],  # Would need to fetch workflow to analyze
                                "metadata": {
                                    "description": template.get("description", ""),
                                    "from_api": True
                                }
                            })
                logging.info(f"Fetched {len(templates)} templates from ComfyUI API")
        except requests.exceptions.RequestException as e:
            logging.debug(f"Could not fetch templates from ComfyUI API: {e}")
        except Exception as e:
            logging.debug(f"Error parsing templates: {e}")
        
        return templates

    def fetch_github_templates(self) -> List[Dict[str, Any]]:
        """Fetch built-in workflow templates from Comfy-Org's GitHub repository."""
        templates = []
        GITHUB_API_URL = "https://api.github.com/repos/Comfy-Org/workflow_templates/contents"
        
        try:
            # Fetch the list of template categories/folders
            response = requests.get(GITHUB_API_URL, timeout=10, headers={
                "Accept": "application/vnd.github.v3+json"
            })
            
            if response.status_code != 200:
                logging.debug(f"GitHub API returned {response.status_code}")
                return templates
            
            items = response.json()
            
            for item in items:
                if item.get("type") == "dir" and not item.get("name", "").startswith("."):
                    category_name = item["name"]
                    category_url = item.get("url", "")
                    
                    # Fetch contents of each category folder
                    try:
                        category_response = requests.get(category_url, timeout=10, headers={
                            "Accept": "application/vnd.github.v3+json"
                        })
                        
                        if category_response.status_code == 200:
                            category_items = category_response.json()
                            
                            for file_item in category_items:
                                if file_item.get("name", "").endswith(".json"):
                                    workflow_name = file_item["name"].replace(".json", "")
                                    download_url = file_item.get("download_url", "")
                                    
                                    # Download and analyze workflow for input_schema
                                    input_schema = []
                                    estimated_vram = 0
                                    if download_url and self._workflow_analyzer:
                                        try:
                                            wf_response = requests.get(download_url, timeout=15)
                                            if wf_response.status_code == 200:
                                                workflow_data = wf_response.json()
                                                input_schema = self._workflow_analyzer.to_input_schema(
                                                    workflow_data, workflow_name
                                                )
                                                metadata = self._workflow_analyzer.analyze(
                                                    workflow_data, workflow_name
                                                )
                                                estimated_vram = metadata.estimated_vram_mb
                                        except Exception as e:
                                            logging.debug(f"Could not analyze {workflow_name}: {e}")
                                    
                                    templates.append({
                                        "name": workflow_name,
                                        "filename": file_item["name"],
                                        "path": "",
                                        "category": category_name,
                                        "source": "github_builtin",
                                        "source_name": "ComfyUI Official",
                                        "download_url": download_url,
                                        "input_schema": input_schema,
                                        "metadata": {
                                            "description": f"Official ComfyUI template from {category_name}",
                                            "github_url": file_item.get("html_url", ""),
                                            "from_github": True,
                                            "vram": estimated_vram
                                        }
                                    })
                    except Exception as e:
                        logging.debug(f"Error fetching category {category_name}: {e}")
                        
            logging.info(f"Fetched {len(templates)} templates from ComfyUI GitHub repository")
            
        except requests.exceptions.RequestException as e:
            logging.warning(f"Could not fetch templates from GitHub: {e}")
        except Exception as e:
            logging.warning(f"Error processing GitHub templates: {e}")
        
        return templates

    def scan_workflows(self) -> List[Dict[str, Any]]:
        """Scans the local ComfyUI installation for saved workflows and templates."""
        workflows = []
        if not self.comfyui_install_dir:
            return workflows

        # Define patterns to search - expanded locations
        search_patterns = [
            # User workflows
            os.path.join(self.comfyui_install_dir, "user", "default", "workflows", "**", "*.json"),
            # OpenFork dedicated folder
            os.path.join(self.comfyui_install_dir, "user", "default", "workflows", "openfork", "**", "*.json"),
            # Custom node example workflows
            os.path.join(self.comfyui_install_dir, "custom_nodes", "**", "workflows", "*.json"),
            os.path.join(self.comfyui_install_dir, "custom_nodes", "**", "example_workflows", "*.json"),
            os.path.join(self.comfyui_install_dir, "custom_nodes", "**", "examples", "*.json"),
            os.path.join(self.comfyui_install_dir, "custom_nodes", "**", "example", "*.json"),
            # ComfyUI built-in web templates
            os.path.join(self.comfyui_install_dir, "web", "templates", "**", "*.json"),
            os.path.join(self.comfyui_install_dir, "web_custom_versions", "**", "templates", "**", "*.json"),
            # Root examples folder
            os.path.join(self.comfyui_install_dir, "examples", "**", "*.json"),
        ]

        found_paths = set()

        for pattern in search_patterns:
            for filepath in glob.glob(pattern, recursive=True):
                 if filepath in found_paths:
                     continue
                 found_paths.add(filepath)
                 
                 try:
                     with open(filepath, 'r', encoding='utf-8') as f:
                         content = f.read()
                         if not content: 
                             continue
                     
                     # Re-open to parse JSON (file pointer was moved)
                     with open(filepath, 'r', encoding='utf-8') as f:
                         data = json.load(f)
                     
                     # Skip UI-format workflows (they have 'nodes' and 'links' at root)
                     if "nodes" in data and "links" in data and "prompt" not in data:
                         logging.debug(f"Skipping UI-format workflow: {filepath}")
                         continue
                     
                     # Use WorkflowAnalyzer if available for better input detection
                     workflow_name = os.path.splitext(os.path.basename(filepath))[0]
                     
                     if self._workflow_analyzer:
                         metadata = self._workflow_analyzer.analyze(data, workflow_name)
                         input_schema = self._workflow_analyzer.to_input_schema(data, workflow_name)
                         category = metadata.category
                         vram = metadata.estimated_vram_mb
                     else:
                         input_schema = self._infer_inputs_from_workflow(data)
                         category = "local"
                         vram = 0
                     
                     # Determine source from file path
                     source = "user"
                     source_name = ""
                     if "custom_nodes" in filepath:
                         source = "custom_node"
                         # Extract custom node name from path
                         parts = filepath.split(os.sep)
                         if "custom_nodes" in parts:
                             idx = parts.index("custom_nodes")
                             if idx + 1 < len(parts):
                                 source_name = parts[idx + 1]
                     elif "web" in filepath and "templates" in filepath:
                         source = "builtin"
                         source_name = "ComfyUI"
                     elif "examples" in filepath:
                         source = "example"
                         source_name = "ComfyUI"
                     elif "openfork" in filepath.lower():
                         source = "openfork"
                         source_name = "OpenFork"
                     
                     workflows.append({
                         "name": workflow_name,
                         "filename": os.path.basename(filepath),
                         "path": filepath,
                         "category": category,
                         "source": source,
                         "source_name": source_name,
                         "input_schema": input_schema,
                         "metadata": {
                             "vram": vram,
                             "description": data.get("extra", {}).get("description", ""),
                             **data.get("extra", {})
                         }
                     })
                 except json.JSONDecodeError as e:
                     logging.debug(f"Invalid JSON in workflow file {filepath}: {e}")
                 except Exception as e:
                     logging.warning(f"Error reading workflow {filepath}: {e}")
        
        # Also try to fetch templates from ComfyUI API
        api_templates = self.fetch_comfyui_templates()
        for template in api_templates:
            # Avoid duplicates by checking name
            if not any(w["name"] == template["name"] for w in workflows):
                workflows.append(template)
        
        # Fetch built-in templates from GitHub
        github_templates = self.fetch_github_templates()
        for template in github_templates:
            # Avoid duplicates by checking name
            if not any(w["name"] == template["name"] for w in workflows):
                workflows.append(template)
        
        # Cache the workflows for get_workflow_content to access
        self._cached_workflows = workflows
        
        logging.info(f"Found {len(workflows)} total workflows/templates")
        return workflows

    def get_workflow_content(self, workflow_name: str) -> Union[Dict[str, Any], None]:
        """Retrieves workflow content by name - from local files or by downloading from GitHub."""
        
        # First, check if we have this workflow in our cached scan results
        # This handles GitHub templates that have download_url
        if hasattr(self, '_cached_workflows') and self._cached_workflows:
            for wf in self._cached_workflows:
                if wf.get('name') == workflow_name:
                    # If it has a local path, try to load from there
                    if wf.get('path') and os.path.exists(wf['path']):
                        try:
                            with open(wf['path'], 'r', encoding='utf-8') as f:
                                logging.info(f"Loaded workflow '{workflow_name}' from cached path: {wf['path']}")
                                return json.load(f)
                        except Exception as e:
                            logging.error(f"Error reading workflow from cached path: {e}")
                    
                    # If it has a download_url (GitHub template), download it
                    if wf.get('download_url'):
                        try:
                            logging.info(f"Downloading workflow '{workflow_name}' from GitHub: {wf['download_url']}")
                            response = requests.get(wf['download_url'], timeout=30)
                            response.raise_for_status()
                            workflow_data = response.json()
                            logging.info(f"Successfully downloaded workflow '{workflow_name}' from GitHub")
                            return workflow_data
                        except Exception as e:
                            logging.error(f"Failed to download workflow from GitHub: {e}")
        
        # Fall back to searching local files if not found in cache
        if not self.comfyui_install_dir:
            return None
        
        search_patterns = [
             os.path.join(self.comfyui_install_dir, "user", "default", "workflows", "**", f"{workflow_name}.json"),
             os.path.join(self.comfyui_install_dir, "user", "default", "workflows", "**", f"{workflow_name}.api.json"),
             os.path.join(self.comfyui_install_dir, "custom_nodes", "**", "workflows", f"{workflow_name}.json"),
             os.path.join(self.comfyui_install_dir, "custom_nodes", "**", "examples", f"{workflow_name}.json"),
        ]

        for pattern in search_patterns:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                 filepath = matches[0]
                 try:
                     with open(filepath, 'r', encoding='utf-8') as f:
                         logging.info(f"Loaded workflow '{workflow_name}' from local path: {filepath}")
                         return json.load(f)
                 except Exception as e:
                     logging.error(f"Error reading workflow {filepath}: {e}")
                     return None
        
        logging.warning(f"Workflow '{workflow_name}' not found in cache or local files")
        return None

    def _infer_inputs_from_workflow(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Heuristically determines input fields from ComfyUI API JSON structure.
        Looks for known node types like CLIPTextEncode, KSampler, etc.
        """
        inputs = []
        
        # Normalize: if "prompt" key exists (API format wrapper), use it. Otherwise assume data is the graph.
        # But wait, sometimes 'nodes' exists (UI format). UI format is harder to parse for values because 
        # values are in 'widgets_values'. 
        # For now, let's target API format (which is what .api.json usually is).
        
        graph = data
        if "prompt" in data and isinstance(data["prompt"], dict):
            graph = data["prompt"]
        elif "nodes" in data:
            # UI format. This is trickier. Let's return empty for now or try basics.
            # In UI format, direct node map isn't the root. 
            return [] 

        # Map of node_id -> node_data
        nodes = graph
        
        for node_id, node in nodes.items():
            if not isinstance(node, dict): continue
            
            class_type = node.get("class_type")
            node_inputs = node.get("inputs", {})
            
            if class_type == "CLIPTextEncode":
                # Check directly provided text
                text_val = node_inputs.get("text")
                if isinstance(text_val, str):
                    # Guess label based on content or standard defaults
                    label = "Prompt"
                    if "negative" in text_val.lower() or "bad" in text_val.lower() or "nsfw" in text_val.lower():
                        label = "Negative Prompt"
                    
                    # Refine label if we have multiple generic prompts
                    # (Could be improved by checking incoming links to Sampler, but simple is good for now)
                    
                    inputs.append({
                        "name": label if not any(i["name"] == label for i in inputs) else f"{label} ({node_id})",
                        "type": "text",
                        "default": text_val,
                        "node_id": node_id,
                        "widget_name": "text"
                    })

            elif class_type in ["KSampler", "KSamplerAdvanced", "SamplerCustom", "A1111_KSampler"]:
                # Seed
                seed = node_inputs.get("seed") or node_inputs.get("noise_seed")
                if isinstance(seed, (int, float)):
                    inputs.append({
                        "name": "Seed",
                        "type": "number",
                        "default": seed,
                        "node_id": node_id,
                        "widget_name": "seed" if "seed" in node_inputs else "noise_seed"
                    })
                
                # Steps
                steps = node_inputs.get("steps")
                if isinstance(steps, int):
                    inputs.append({
                        "name": "Steps",
                        "type": "number",
                        "default": steps,
                        "node_id": node_id,
                        "widget_name": "steps"
                    })
                
                # CFG
                cfg = node_inputs.get("cfg")
                if isinstance(cfg, (int, float)):
                     inputs.append({
                        "name": "CFG Scale",
                        "type": "number",
                        "default": cfg,
                        "node_id": node_id,
                        "widget_name": "cfg"
                    })

            elif class_type == "EmptyLatentImage":
                width = node_inputs.get("width")
                height = node_inputs.get("height")
                batch = node_inputs.get("batch_size")
                
                if isinstance(width, int):
                    inputs.append({"name": "Width", "type": "number", "default": width, "node_id": node_id, "widget_name": "width"})
                if isinstance(height, int):
                    inputs.append({"name": "Height", "type": "number", "default": height, "node_id": node_id, "widget_name": "height"})
            
            elif class_type in ["WanImageToVideo", "WanVideoToVideo"]: 
                 # WAN specific nodes
                 width = node_inputs.get("width")
                 height = node_inputs.get("height")
                 length = node_inputs.get("length")
                 if isinstance(width, int):
                    inputs.append({"name": "Width", "type": "number", "default": width, "node_id": node_id, "widget_name": "width"})
                 if isinstance(height, int):
                    inputs.append({"name": "Height", "type": "number", "default": height, "node_id": node_id, "widget_name": "height"})
                 if isinstance(length, int):
                    inputs.append({"name": "Frames", "type": "number", "default": length, "node_id": node_id, "widget_name": "length"})

            elif class_type == "LoadImage":
                image = node_inputs.get("image")
                if isinstance(image, str):
                    inputs.append({
                        "name": "Input Image",
                        "type": "image",
                        "default": image,
                        "node_id": node_id,
                        "widget_name": "image"
                    })
            
            elif class_type in ["VHS_LoadVideo", "LoadVideo"]:
                video = node_inputs.get("video") or node_inputs.get("video_path") # Some nodes use different keys
                if isinstance(video, str):
                    inputs.append({
                        "name": "Input Video",
                        "type": "input_video",
                        "default": video,
                        "node_id": node_id,
                        "widget_name": "video" if "video" in node_inputs else "video_path"
                    })
            
            elif class_type == "LoadAudio":
                audio = node_inputs.get("audio")
                if isinstance(audio, str):
                    inputs.append({
                        "name": "Input Audio",
                        "type": "input_audio",
                        "default": audio,
                        "node_id": node_id,
                        "widget_name": "audio"
                    })

            elif class_type == "PrimitiveNode":
                 # Primitives can be anything. We check the value type.
                 # Usually they have a 'value' input which holds the data.
                 # But in API format, it might be nested or direct.
                 # Actually, PrimitiveNode usually just has 'value'.
                 value = node_inputs.get("value")
                 
                 # Boolean Switch
                 if isinstance(value, bool):
                      # Try to infer name from what it connects to? Too complex for now.
                      # Just list it.
                      inputs.append({
                          "name": f"Switch ({node_id})",
                          "type": "boolean",
                          "default": value,
                          "node_id": node_id,
                          "widget_name": "value"
                      })
                 elif isinstance(value, str):
                      inputs.append({
                          "name": f"Text Input ({node_id})",
                          "type": "text",
                          "default": value,
                          "node_id": node_id,
                          "widget_name": "value"
                      })

        return inputs

    def get_input_directory(self) -> Union[str, None]:
        if not self.comfyui_install_dir: return None
        return os.path.join(self.comfyui_install_dir, "input")

    def get_output_directory(self) -> Union[str, None]:
        if not self.comfyui_install_dir: return None
        return os.path.join(self.comfyui_install_dir, "output")

    def stop(self):
        """Stops the ComfyUI process if it was started by this manager."""
        if self.process:
            logging.info("Stopping ComfyUI process...")
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                logging.info("ComfyUI process stopped.")
            except Exception as e:
                logging.error(f"Error stopping ComfyUI: {e}")
            finally:
                self.process = None
