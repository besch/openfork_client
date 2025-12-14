"""
WorkflowImporter - Import ComfyUI workflows from various sources.

Enables:
- Importing workflows from local files
- Downloading workflows from URLs (Civitai, OpenArt, etc.)
- Auto-installing missing custom nodes via comfy-cli
- Validating workflow compatibility
"""

import os
import logging
import json
import shutil
from typing import Union
from dataclasses import dataclass
import requests


@dataclass
class ImportResult:
    """Result of a workflow import operation."""
    success: bool
    workflow_name: str
    workflow_path: str = ""
    message: str = ""
    missing_nodes: list[str] = None
    
    def __post_init__(self):
        if self.missing_nodes is None:
            self.missing_nodes = []


class WorkflowImporter:
    """Import ComfyUI workflows from various sources with auto-dependency installation."""
    
    def __init__(self, comfyui_install_dir: str, comfy_cli_manager=None):
        self.comfyui_install_dir = comfyui_install_dir
        self.comfy_cli_manager = comfy_cli_manager
        
        # Default save location for imported workflows
        self.workflows_dir = os.path.join(
            comfyui_install_dir, "user", "default", "workflows", "openfork"
        ) if comfyui_install_dir else None
    
    def import_from_file(self, file_path: str, auto_install_deps: bool = True) -> ImportResult:
        """
        Import a workflow from a local file.
        
        Args:
            file_path: Path to the workflow JSON file
            auto_install_deps: Whether to auto-install missing custom nodes
            
        Returns:
            ImportResult with success status and details
        """
        if not os.path.exists(file_path):
            return ImportResult(
                success=False,
                workflow_name="",
                message=f"File not found: {file_path}"
            )
        
        workflow_name = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
        except json.JSONDecodeError as e:
            return ImportResult(
                success=False,
                workflow_name=workflow_name,
                message=f"Invalid JSON: {e}"
            )
        except Exception as e:
            return ImportResult(
                success=False,
                workflow_name=workflow_name,
                message=f"Error reading file: {e}"
            )
        
        return self._process_workflow(workflow_data, workflow_name, auto_install_deps)
    
    def import_from_url(self, url: str, auto_install_deps: bool = True) -> ImportResult:
        """
        Download and import a workflow from a URL.
        
        Args:
            url: URL to the workflow JSON file
            auto_install_deps: Whether to auto-install missing custom nodes
            
        Returns:
            ImportResult with success status and details
        """
        try:
            logging.info(f"Downloading workflow from: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Try to get filename from URL or Content-Disposition header
            workflow_name = self._extract_filename_from_url(url, response)
            
            workflow_data = response.json()
            return self._process_workflow(workflow_data, workflow_name, auto_install_deps)
            
        except requests.exceptions.RequestException as e:
            return ImportResult(
                success=False,
                workflow_name="",
                message=f"Failed to download: {e}"
            )
        except json.JSONDecodeError as e:
            return ImportResult(
                success=False,
                workflow_name="",
                message=f"Invalid JSON at URL: {e}"
            )
    
    def _extract_filename_from_url(self, url: str, response: requests.Response) -> str:
        """Extract a reasonable filename from URL or response headers."""
        # Try Content-Disposition header
        cd = response.headers.get('Content-Disposition', '')
        if 'filename=' in cd:
            filename = cd.split('filename=')[1].strip('"\'')
            return os.path.splitext(filename)[0]
        
        # Fall back to URL path
        from urllib.parse import urlparse
        path = urlparse(url).path
        filename = os.path.basename(path)
        if filename.endswith('.json'):
            return os.path.splitext(filename)[0]
        
        # Generate a name if nothing else works
        import hashlib
        return f"imported_workflow_{hashlib.md5(url.encode()).hexdigest()[:8]}"
    
    def _process_workflow(self, workflow_data: dict, workflow_name: str, 
                          auto_install_deps: bool) -> ImportResult:
        """Process an imported workflow - validate, install deps, save."""
        
        # Check if it's API format (has 'prompt' with nodes) or needs conversion
        if not self._is_api_format(workflow_data):
            # It might be UI format, which we can't easily process
            if "nodes" in workflow_data and "links" in workflow_data:
                return ImportResult(
                    success=False,
                    workflow_name=workflow_name,
                    message="Workflow is in UI format. Please export it in API format from ComfyUI."
                )
        
        # Check for missing nodes
        missing_nodes = self._find_missing_nodes(workflow_data)
        
        if missing_nodes and auto_install_deps and self.comfy_cli_manager:
            logging.info(f"Auto-installing missing nodes: {missing_nodes}")
            results = self.comfy_cli_manager.install_nodes(missing_nodes)
            
            # Update missing list with any that failed to install
            still_missing = [r.node_name for r in results if not r.success]
            if still_missing:
                logging.warning(f"Some nodes could not be installed: {still_missing}")
                missing_nodes = still_missing
            else:
                missing_nodes = []
        
        # Save to workflows directory
        if self.workflows_dir:
            os.makedirs(self.workflows_dir, exist_ok=True)
            
            # Ensure unique filename
            dest_path = os.path.join(self.workflows_dir, f"{workflow_name}.json")
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(self.workflows_dir, f"{workflow_name}_{counter}.json")
                counter += 1
            
            try:
                with open(dest_path, 'w', encoding='utf-8') as f:
                    json.dump(workflow_data, f, indent=2)
                logging.info(f"Saved workflow to: {dest_path}")
                
                return ImportResult(
                    success=True,
                    workflow_name=workflow_name,
                    workflow_path=dest_path,
                    message="Workflow imported successfully",
                    missing_nodes=missing_nodes
                )
            except Exception as e:
                return ImportResult(
                    success=False,
                    workflow_name=workflow_name,
                    message=f"Failed to save workflow: {e}",
                    missing_nodes=missing_nodes
                )
        else:
            return ImportResult(
                success=False,
                workflow_name=workflow_name,
                message="No workflows directory configured",
                missing_nodes=missing_nodes
            )
    
    def _is_api_format(self, workflow_data: dict) -> bool:
        """Check if workflow is in API format."""
        # API format has 'prompt' key with dict of nodes, OR is a direct dict of nodes
        if "prompt" in workflow_data and isinstance(workflow_data["prompt"], dict):
            return True
        
        # Check if it's a direct graph (dict of nodes with class_type)
        for key, value in workflow_data.items():
            if isinstance(value, dict) and "class_type" in value:
                return True
        
        return False
    
    def _find_missing_nodes(self, workflow_data: dict) -> list[str]:
        """Find custom nodes that are required but might be missing."""
        # Get all class_types from workflow
        graph = workflow_data.get("prompt", workflow_data)
        class_types = set()
        
        for node in graph.values():
            if isinstance(node, dict) and "class_type" in node:
                class_types.add(node["class_type"])
        
        # Map class_type prefixes to known custom node packages
        # This is heuristic - full accuracy requires querying /object_info
        prefix_to_package = {
            "VHS_": "ComfyUI-VideoHelperSuite",
            "KJ": "ComfyUI-KJNodes",
            "Impact": "ComfyUI-Impact-Pack",
            "BRIA_": "ComfyUI-BRIA",
            "WAN": "ComfyUI-Wan",
            "LTX": "ComfyUI-LTXVideo",
            "Hunyuan": "ComfyUI-HunyuanVideo",
            "DiffRhythm": "ComfyUI_DiffRhythm",
            "VibeVoice": "ComfyUI_VibeVoice",
            "FluxNoise": "ComfyUI-FluxNoise",
            "RealESRGAN": "ComfyUI-ESRGAN",
        }
        
        missing = []
        installed = self._get_installed_custom_nodes() if self.comfyui_install_dir else []
        
        for class_type in class_types:
            for prefix, package in prefix_to_package.items():
                if class_type.startswith(prefix):
                    if package not in installed and package not in missing:
                        missing.append(package)
                    break
        
        return missing
    
    def _get_installed_custom_nodes(self) -> list[str]:
        """Get list of installed custom node directory names."""
        if not self.comfyui_install_dir:
            return []
        
        custom_nodes_dir = os.path.join(self.comfyui_install_dir, "custom_nodes")
        if not os.path.exists(custom_nodes_dir):
            return []
        
        return [d for d in os.listdir(custom_nodes_dir) 
                if os.path.isdir(os.path.join(custom_nodes_dir, d)) and not d.startswith('.')]
    
    def list_imported_workflows(self) -> list[str]:
        """List all workflows in the OpenFork imports directory."""
        if not self.workflows_dir or not os.path.exists(self.workflows_dir):
            return []
        
        workflows = []
        for f in os.listdir(self.workflows_dir):
            if f.endswith('.json'):
                workflows.append(os.path.splitext(f)[0])
        
        return workflows
    
    def delete_workflow(self, workflow_name: str) -> bool:
        """Delete an imported workflow."""
        if not self.workflows_dir:
            return False
        
        path = os.path.join(self.workflows_dir, f"{workflow_name}.json")
        if os.path.exists(path):
            os.remove(path)
            logging.info(f"Deleted workflow: {workflow_name}")
            return True
        
        return False
