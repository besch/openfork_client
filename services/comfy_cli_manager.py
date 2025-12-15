"""
ComfyCliManager - Wrapper around comfy-cli for programmatic node and dependency management.

This module provides automation for:
- Installing custom nodes
- Installing workflow dependencies
- Checking installed nodes
- Managing model downloads
"""

import subprocess
import logging
import os
import json
import shutil
from typing import Union
from dataclasses import dataclass


@dataclass
class NodeInstallResult:
    success: bool
    node_name: str
    message: str


@dataclass
class DependencyCheckResult:
    all_satisfied: bool
    missing_nodes: list[str]
    missing_models: list[dict]


class ComfyCliManager:
    """Wrapper around comfy-cli for programmatic node management."""
    
    def __init__(self, comfyui_install_dir: str = None):
        self.comfyui_install_dir = comfyui_install_dir
        self._comfy_cli_available = None
        self._comfy_cmd = None  # Will store the path to comfy command
    
    def _find_comfy_command(self) -> Union[str, None]:
        """Find the comfy command, checking multiple locations."""
        # First check if we've already found it
        if self._comfy_cmd:
            return self._comfy_cmd
        
        # Locations to check
        candidates = ["comfy"]  # Try system PATH first
        
        # Try common Python Scripts directories
        import sys
        home = os.path.expanduser("~")
        
        # Add Python Scripts directory from the current interpreter
        python_scripts = os.path.join(os.path.dirname(sys.executable), "Scripts")
        if os.name == 'nt':  # Windows
            candidates.extend([
                os.path.join(python_scripts, "comfy.exe"),
                os.path.join(home, ".pyenv", "pyenv-win", "shims", "comfy"),
                os.path.join(home, ".local", "bin", "comfy"),
                os.path.join(home, "AppData", "Local", "Programs", "Python", "Python39", "Scripts", "comfy.exe"),
                os.path.join(home, "AppData", "Local", "Programs", "Python", "Python310", "Scripts", "comfy.exe"),
                os.path.join(home, "AppData", "Local", "Programs", "Python", "Python311", "Scripts", "comfy.exe"),
            ])
            # Also check pyenv versions
            pyenv_base = os.path.join(home, ".pyenv", "pyenv-win", "versions")
            if os.path.exists(pyenv_base):
                for version_dir in os.listdir(pyenv_base):
                    candidates.append(os.path.join(pyenv_base, version_dir, "Scripts", "comfy.exe"))
        else:  # Unix-like
            candidates.extend([
                os.path.join(python_scripts, "comfy"),
                os.path.join(home, ".local", "bin", "comfy"),
                "/usr/local/bin/comfy",
            ])
        
        # Try shutil.which first for system PATH
        cmd = shutil.which("comfy")
        if cmd:
            self._comfy_cmd = cmd
            logging.info(f"Found comfy-cli at: {cmd}")
            return cmd
        
        # Check all candidates
        for cmd in candidates:
            if os.path.isfile(cmd):
                self._comfy_cmd = cmd
                logging.info(f"Found comfy-cli at: {cmd}")
                return cmd
        
        return None
    
    def _get_comfy_cmd(self) -> list[str]:
        """Get the comfy command as a list for subprocess."""
        cmd = self._find_comfy_command()
        if cmd:
            return [cmd]
        return ["comfy"]  # Fallback to system PATH
        
    def is_available(self) -> bool:
        """Check if comfy-cli is installed and available."""
        if self._comfy_cli_available is not None:
            return self._comfy_cli_available
        
        cmd = self._find_comfy_command()
        if not cmd:
            self._comfy_cli_available = False
            logging.warning("comfy-cli is not installed. Run: pip install comfy-cli")
            return False
            
        try:
            result = subprocess.run(
                [cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            self._comfy_cli_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._comfy_cli_available = False
            
        if not self._comfy_cli_available:
            logging.warning("comfy-cli is not installed. Run: pip install comfy-cli")
            
        return self._comfy_cli_available
    
    def _get_workspace_args(self) -> list[str]:
        """Get workspace arguments for comfy-cli commands."""
        if self.comfyui_install_dir and os.path.exists(self.comfyui_install_dir):
            return ["--workspace", self.comfyui_install_dir]
        return []
    
    def install_node(self, node_name: str) -> NodeInstallResult:
        """
        Install a single custom node by name.
        
        Args:
            node_name: The name of the custom node (e.g., 'ComfyUI-VideoHelperSuite')
            
        Returns:
            NodeInstallResult with success status and message
        """
        if not self.is_available():
            return NodeInstallResult(
                success=False,
                node_name=node_name,
                message="comfy-cli is not available"
            )
        
        try:
            cmd = self._get_comfy_cmd() + self._get_workspace_args() + ["node", "install", node_name]
            logging.info(f"Installing node: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for installation
            )
            
            if result.returncode == 0:
                logging.info(f"Successfully installed node: {node_name}")
                return NodeInstallResult(
                    success=True,
                    node_name=node_name,
                    message=result.stdout
                )
            else:
                logging.error(f"Failed to install node {node_name}: {result.stderr}")
                return NodeInstallResult(
                    success=False,
                    node_name=node_name,
                    message=result.stderr
                )
                
        except subprocess.TimeoutExpired:
            return NodeInstallResult(
                success=False,
                node_name=node_name,
                message="Installation timed out after 5 minutes"
            )
        except Exception as e:
            return NodeInstallResult(
                success=False,
                node_name=node_name,
                message=str(e)
            )
    
    def install_node_by_url(self, git_url: str) -> NodeInstallResult:
        """
        Install a custom node from a git URL using comfy-cli.
        
        Args:
            git_url: The git repository URL (e.g., 'https://github.com/pythongosssss/ComfyUI-Custom-Scripts')
            
        Returns:
            NodeInstallResult with success status and message
        """
        if not self.is_available():
            return NodeInstallResult(
                success=False,
                node_name=git_url,
                message="comfy-cli is not available"
            )
        
        # Extract package name from URL for logging
        package_name = git_url.rstrip('/').split('/')[-1].replace('.git', '')
        
        try:
            cmd = self._get_comfy_cmd() + self._get_workspace_args() + ["node", "install", git_url]
            logging.info(f"Installing node from URL: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for installation (may need to clone + install deps)
            )
            
            if result.returncode == 0:
                logging.info(f"Successfully installed node: {package_name}")
                return NodeInstallResult(
                    success=True,
                    node_name=package_name,
                    message=result.stdout
                )
            else:
                logging.error(f"Failed to install node {package_name}: {result.stderr}")
                return NodeInstallResult(
                    success=False,
                    node_name=package_name,
                    message=result.stderr
                )
                
        except subprocess.TimeoutExpired:
            return NodeInstallResult(
                success=False,
                node_name=package_name,
                message="Installation timed out after 10 minutes"
            )
        except Exception as e:
            return NodeInstallResult(
                success=False,
                node_name=package_name,
                message=str(e)
            )
    
    def install_nodes(self, node_names: list[str]) -> list[NodeInstallResult]:
        """Install multiple custom nodes."""
        results = []
        for node_name in node_names:
            result = self.install_node(node_name)
            results.append(result)
        return results
    
    def install_workflow_deps(self, workflow_path: str) -> bool:
        """
        Install all dependencies for a workflow using comfy-cli.
        
        Args:
            workflow_path: Path to the workflow JSON file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logging.warning("comfy-cli not available, cannot install workflow dependencies")
            return False
            
        if not os.path.exists(workflow_path):
            logging.error(f"Workflow file not found: {workflow_path}")
            return False
        
        try:
            cmd = ["comfy"] + self._get_workspace_args() + [
                "node", "install-deps", 
                "--workflow", workflow_path
            ]
            logging.info(f"Installing workflow dependencies: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                logging.info(f"Successfully installed dependencies for: {workflow_path}")
                return True
            else:
                logging.warning(f"Failed to install dependencies: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"Error installing workflow dependencies: {e}")
            return False
    
    def get_installed_nodes(self) -> list[str]:
        """
        Get list of installed custom nodes.
        
        Returns:
            List of installed node names
        """
        if not self.comfyui_install_dir:
            return []
            
        custom_nodes_dir = os.path.join(self.comfyui_install_dir, "custom_nodes")
        if not os.path.exists(custom_nodes_dir):
            return []
        
        installed = []
        for item in os.listdir(custom_nodes_dir):
            item_path = os.path.join(custom_nodes_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if it's a valid custom node (has __init__.py or similar)
                if (os.path.exists(os.path.join(item_path, "__init__.py")) or
                    os.path.exists(os.path.join(item_path, "nodes.py")) or
                    os.path.exists(os.path.join(item_path, "pyproject.toml"))):
                    installed.append(item)
                    
        return installed
    
    def update_all_nodes(self) -> bool:
        """Update all installed custom nodes to latest versions."""
        if not self.is_available():
            return False
            
        try:
            cmd = ["comfy"] + self._get_workspace_args() + ["node", "update", "all"]
            logging.info(f"Updating all nodes: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            return result.returncode == 0
            
        except Exception as e:
            logging.error(f"Error updating nodes: {e}")
            return False
    
    def check_workflow_dependencies(self, workflow_data: dict) -> DependencyCheckResult:
        """
        Analyze a workflow and check what dependencies are missing.
        
        Args:
            workflow_data: The parsed workflow JSON
            
        Returns:
            DependencyCheckResult with lists of missing nodes and models
        """
        missing_nodes = []
        missing_models = []
        
        # Get installed node class types by querying ComfyUI /object_info
        # For now, we'll do a simpler check based on known node patterns
        installed_nodes = self.get_installed_nodes()
        
        # Extract all class_types from workflow
        graph = workflow_data.get("prompt", workflow_data)
        required_class_types = set()
        
        for node_id, node in graph.items():
            if isinstance(node, dict) and "class_type" in node:
                required_class_types.add(node["class_type"])
        
        # Map common class_types to their custom node packages
        # This is a heuristic - the full mapping would come from /object_info
        node_package_hints = {
            "VHS_": "ComfyUI-VideoHelperSuite",
            "KJ": "ComfyUI-KJNodes",
            "Impact": "ComfyUI-Impact-Pack",
            "WAN": "ComfyUI-Wan",
            "LTX": "ComfyUI-LTXVideo",
            "Hunyuan": "ComfyUI-HunyuanVideo",
            "DiffRhythm": "ComfyUI_DiffRhythm",
        }
        
        for class_type in required_class_types:
            for prefix, package in node_package_hints.items():
                if class_type.startswith(prefix) and package not in installed_nodes:
                    if package not in missing_nodes:
                        missing_nodes.append(package)
        
        return DependencyCheckResult(
            all_satisfied=len(missing_nodes) == 0 and len(missing_models) == 0,
            missing_nodes=missing_nodes,
            missing_models=missing_models
        )


def install_comfy_cli() -> bool:
    """Attempt to install comfy-cli if not present."""
    try:
        logging.info("Attempting to install comfy-cli...")
        result = subprocess.run(
            ["pip", "install", "comfy-cli"],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            logging.info("Successfully installed comfy-cli")
            return True
        else:
            logging.error(f"Failed to install comfy-cli: {result.stderr}")
            return False
    except Exception as e:
        logging.error(f"Error installing comfy-cli: {e}")
        return False
