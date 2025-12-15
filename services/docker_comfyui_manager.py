"""
Docker-based ComfyUI Manager for full programmatic control.
Manages container lifecycle, model downloads, node installation, and file I/O.
"""

import subprocess
import logging
import time
import os
import json
import requests
import shutil
from typing import Union, Dict, List, Any
from dataclasses import dataclass


@dataclass
class NodeInstallResult:
    success: bool
    package_name: str
    message: str


@dataclass
class ModelDownloadResult:
    success: bool
    filename: str
    message: str


class DockerComfyUIManager:
    """Manages a Docker-based ComfyUI instance for full programmatic control.
    
    Features:
    - Container lifecycle management
    - Volume mounts for persistent storage
    - Model downloading (wget, huggingface-cli)
    - Custom node installation via git clone
    - Workflow scanning and validation
    - Input/output file handling via docker cp
    """
    
    CONTAINER_NAME = "openfork-comfyui"
    DEFAULT_IMAGE = "ghcr.io/ai-dock/comfyui:latest-cuda"
    INTERNAL_PORT = 8188
    
    # Container paths
    COMFYUI_PATH = "/opt/ComfyUI"
    CUSTOM_NODES_PATH = "/opt/ComfyUI/custom_nodes"
    MODELS_PATH = "/opt/ComfyUI/models"
    INPUT_PATH = "/opt/ComfyUI/input"
    OUTPUT_PATH = "/opt/ComfyUI/output"
    WORKFLOWS_PATH = "/opt/ComfyUI/user/default/workflows"
    
    def __init__(
        self,
        container_name: str = None,
        image: str = None,
        host_port: int = 8188,
        gpu_enabled: bool = True,
        data_dir: str = None
    ):
        self.container_name = container_name or self.CONTAINER_NAME
        self.image = image or self.DEFAULT_IMAGE
        self.host_port = host_port
        self.gpu_enabled = gpu_enabled
        self.comfyui_url = f"http://127.0.0.1:{host_port}"
        
        # Setup host data directory for volume mounts
        self.data_dir = data_dir or os.path.join(os.path.expanduser("~"), ".openfork")
        self.comfyui_data_dir = os.path.join(self.data_dir, "comfyui")
        
        # Host paths for volume mounts
        self.host_models_dir = os.path.join(self.comfyui_data_dir, "models")
        self.host_custom_nodes_dir = os.path.join(self.comfyui_data_dir, "custom_nodes")
        self.host_input_dir = os.path.join(self.comfyui_data_dir, "input")
        self.host_output_dir = os.path.join(self.comfyui_data_dir, "output")
        self.host_workflows_dir = os.path.join(self.comfyui_data_dir, "workflows")
        
        self._installed_nodes_cache = None
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create host directories for volume mounts."""
        dirs = [
            self.host_models_dir,
            self.host_custom_nodes_dir,
            self.host_input_dir,
            self.host_output_dir,
            self.host_workflows_dir,
            # Model subdirectories
            os.path.join(self.host_models_dir, "checkpoints"),
            os.path.join(self.host_models_dir, "loras"),
            os.path.join(self.host_models_dir, "vae"),
            os.path.join(self.host_models_dir, "clip"),
            os.path.join(self.host_models_dir, "clip_vision"),
            os.path.join(self.host_models_dir, "unet"),
            os.path.join(self.host_models_dir, "controlnet"),
            os.path.join(self.host_models_dir, "diffusion_models"),
            os.path.join(self.host_models_dir, "text_encoders"),
            os.path.join(self.host_models_dir, "upscale_models"),
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        
        # Provision bundled workflows to host directory
        self._provision_bundled_workflows()
    
    def _normalize_path_for_docker(self, path: str) -> str:
        """Convert Windows paths to Docker-compatible format.
        
        On Windows with Docker Desktop using WSL2 backend:
        - C:\Users\... becomes /c/Users/...
        - Backslashes become forward slashes
        """
        if os.name == 'nt':  # Windows
            # Convert backslashes to forward slashes
            path = path.replace('\\', '/')
            # Convert drive letter: C:/... -> /c/...
            if len(path) >= 2 and path[1] == ':':
                path = '/' + path[0].lower() + path[2:]
        return path
    
    def _provision_bundled_workflows(self):
        """Copy bundled workflows from client/workflows to host workflows directory."""
        # Find the bundled workflows directory (relative to this file)
        bundled_dir = os.path.join(os.path.dirname(__file__), '..', 'workflows')
        bundled_dir = os.path.abspath(bundled_dir)
        
        if not os.path.exists(bundled_dir):
            logging.debug(f"No bundled workflows directory found at {bundled_dir}")
            return
        
        copied_count = 0
        for filename in os.listdir(bundled_dir):
            if filename.endswith('.json') and filename != 'manifest.json':
                src = os.path.join(bundled_dir, filename)
                dst = os.path.join(self.host_workflows_dir, filename)
                
                # Only copy if not already present (don't overwrite user modifications)
                if not os.path.exists(dst):
                    try:
                        shutil.copy2(src, dst)
                        copied_count += 1
                        logging.debug(f"Copied workflow: {filename}")
                    except Exception as e:
                        logging.warning(f"Failed to copy workflow {filename}: {e}")
        
        if copied_count > 0:
            logging.info(f"Provisioned {copied_count} bundled workflow(s) to {self.host_workflows_dir}")
    
    def _run_docker_cmd(self, args: list, timeout: int = 60) -> subprocess.CompletedProcess:
        """Execute a docker command with timeout."""
        cmd = ["docker"] + args
        logging.debug(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result
        except subprocess.TimeoutExpired as e:
            logging.error(f"Docker command timed out: {' '.join(cmd)}")
            raise
    
    def _run_docker_cmd_streaming(self, args: list, timeout: int = 600) -> tuple[int, str, str]:
        """Execute a docker command with streaming output for long-running commands."""
        cmd = ["docker"] + args
        logging.info(f"Running (streaming): {' '.join(cmd[:10])}...")
        
        import threading
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            stdout_lines = []
            stderr_lines = []
            
            def read_output(pipe, lines, name):
                for line in iter(pipe.readline, ''):
                    if line:
                        lines.append(line.rstrip())
                        if name == "stderr" or "error" in line.lower():
                            logging.warning(f"[docker {name}] {line.rstrip()}")
                        else:
                            logging.debug(f"[docker {name}] {line.rstrip()}")
                pipe.close()
            
            stdout_thread = threading.Thread(target=read_output, args=(process.stdout, stdout_lines, "stdout"))
            stderr_thread = threading.Thread(target=read_output, args=(process.stderr, stderr_lines, "stderr"))
            
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for completion with timeout
            start_time = time.time()
            while process.poll() is None:
                if time.time() - start_time > timeout:
                    process.kill()
                    return -1, "", "Command timed out"
                time.sleep(0.5)
            
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            
            return process.returncode, '\n'.join(stdout_lines), '\n'.join(stderr_lines)
            
        except Exception as e:
            logging.error(f"Error running docker command: {e}")
            return -1, "", str(e)
    
    # ========== Container Lifecycle ==========
    
    def is_container_running(self) -> bool:
        """Check if the ComfyUI container is running."""
        result = self._run_docker_cmd(
            ["ps", "-q", "-f", f"name=^/{self.container_name}$"],
            timeout=10
        )
        return bool(result.stdout.strip())
    
    def is_container_exists(self) -> bool:
        """Check if the container exists (running or stopped)."""
        result = self._run_docker_cmd(
            ["ps", "-aq", "-f", f"name=^/{self.container_name}$"],
            timeout=10
        )
        return bool(result.stdout.strip())
    
    def is_comfyui_ready(self) -> bool:
        """Check if ComfyUI API is responding."""
        try:
            response = requests.get(f"{self.comfyui_url}/object_info", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def pull_image(self) -> bool:
        """Pull the ComfyUI Docker image."""
        logging.info(f"Pulling Docker image: {self.image}")
        returncode, stdout, stderr = self._run_docker_cmd_streaming(["pull", self.image], timeout=900)
        
        if returncode == 0:
            logging.info(f"Successfully pulled {self.image}")
            return True
        else:
            logging.error(f"Failed to pull image: {stderr}")
            return False
    
    def start_container(self) -> bool:
        """Start the ComfyUI container with volume mounts."""
        if self.is_container_running():
            logging.info(f"Container {self.container_name} is already running")
            return self._wait_for_ready()
        
        # Check if container exists but is stopped
        if self.is_container_exists():
            logging.info(f"Starting existing container {self.container_name}...")
            result = self._run_docker_cmd(["start", self.container_name], timeout=30)
            if result.returncode == 0:
                return self._wait_for_ready()
            else:
                logging.warning(f"Failed to start existing container: {result.stderr}")
                # Try removing and recreating
                self.remove_container()
        
        # Create and run new container with volume mounts
        logging.info(f"Creating new container {self.container_name}...")
        
        # Normalize paths for Docker on Windows
        models_mount = self._normalize_path_for_docker(self.host_models_dir)
        nodes_mount = self._normalize_path_for_docker(self.host_custom_nodes_dir)
        input_mount = self._normalize_path_for_docker(self.host_input_dir)
        output_mount = self._normalize_path_for_docker(self.host_output_dir)
        workflows_mount = self._normalize_path_for_docker(self.host_workflows_dir)
        
        run_args = [
            "run", "-d",
            "--name", self.container_name,
            "-p", f"{self.host_port}:{self.INTERNAL_PORT}",
            "-e", "COMFYUI_PORT=8188",
            # Volume mounts for persistent storage (using normalized paths)
            "-v", f"{models_mount}:{self.MODELS_PATH}",
            "-v", f"{nodes_mount}:{self.CUSTOM_NODES_PATH}",
            "-v", f"{input_mount}:{self.INPUT_PATH}",
            "-v", f"{output_mount}:{self.OUTPUT_PATH}",
            "-v", f"{workflows_mount}:{self.WORKFLOWS_PATH}",
        ]
        
        if self.gpu_enabled:
            run_args.extend(["--gpus", "all"])
        
        run_args.append(self.image)
        
        result = self._run_docker_cmd(run_args, timeout=60)
        
        if result.returncode == 0:
            logging.info(f"Container {self.container_name} created with volume mounts")
            return self._wait_for_ready()
        else:
            logging.error(f"Failed to create container: {result.stderr}")
            return False
    
    def stop_container(self) -> bool:
        """Stop the ComfyUI container."""
        if not self.is_container_running():
            logging.info(f"Container {self.container_name} is not running")
            return True
        
        logging.info(f"Stopping container {self.container_name}...")
        result = self._run_docker_cmd(["stop", self.container_name], timeout=30)
        
        if result.returncode == 0:
            logging.info(f"Container {self.container_name} stopped")
            return True
        else:
            logging.error(f"Failed to stop container: {result.stderr}")
            return False
    
    def restart_container(self) -> bool:
        """Restart the ComfyUI container to reload nodes."""
        logging.info(f"Restarting container {self.container_name}...")
        result = self._run_docker_cmd(["restart", self.container_name], timeout=60)
        
        if result.returncode == 0:
            self._installed_nodes_cache = None
            return self._wait_for_ready()
        else:
            logging.error(f"Failed to restart container: {result.stderr}")
            return False
    
    def remove_container(self) -> bool:
        """Remove the container completely (for cleanup/reset)."""
        self.stop_container()
        
        result = self._run_docker_cmd(["rm", "-f", self.container_name], timeout=30)
        
        if result.returncode == 0:
            logging.info(f"Container {self.container_name} removed")
            return True
        else:
            logging.warning(f"Failed to remove container: {result.stderr}")
            return False
    
    def _wait_for_ready(self, timeout: int = 180) -> bool:
        """Wait for ComfyUI to become ready."""
        logging.info("Waiting for ComfyUI to become ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_comfyui_ready():
                elapsed = int(time.time() - start_time)
                logging.info(f"ComfyUI is ready (took {elapsed}s)")
                return True
            time.sleep(2)
        
        logging.error(f"ComfyUI did not become ready within {timeout}s")
        return False
    
    # ========== Model Management ==========
    
    def download_model(self, url: str, dest_filename: str, model_type: str = "checkpoints") -> ModelDownloadResult:
        """Download a model file via wget inside the container.
        
        Args:
            url: Direct download URL for the model
            dest_filename: Filename to save as (e.g., 'wan2.1_fp8.safetensors')
            model_type: Model type/subfolder (checkpoints, loras, vae, clip, unet, controlnet)
        """
        target_path = f"{self.MODELS_PATH}/{model_type}/{dest_filename}"
        
        # Check if already exists (on host since we're using volumes)
        host_path = os.path.join(self.host_models_dir, model_type, dest_filename)
        if os.path.exists(host_path):
            logging.info(f"Model {dest_filename} already exists at {host_path}")
            return ModelDownloadResult(True, dest_filename, "Already exists")
        
        logging.info(f"Downloading model {dest_filename} to {target_path}...")
        
        # Use wget inside container
        returncode, stdout, stderr = self._run_docker_cmd_streaming([
            "exec", self.container_name,
            "wget", "-q", "--show-progress", "-O", target_path, url
        ], timeout=1800)  # 30 min timeout for large models
        
        if returncode == 0:
            logging.info(f"Successfully downloaded {dest_filename}")
            return ModelDownloadResult(True, dest_filename, "Downloaded successfully")
        else:
            logging.error(f"Failed to download model: {stderr}")
            return ModelDownloadResult(False, dest_filename, stderr)
    
    def download_hf_model(self, repo_id: str, filename: str, model_type: str = "checkpoints", revision: str = "main") -> ModelDownloadResult:
        """Download a model from Hugging Face.
        
        Args:
            repo_id: HuggingFace repo ID (e.g., 'Kijai/WanVideo_comfy')
            filename: File to download from the repo
            model_type: Model type/subfolder
            revision: Git revision (branch, tag, or commit)
        """
        target_dir = f"{self.MODELS_PATH}/{model_type}"
        
        # Check if already exists
        host_path = os.path.join(self.host_models_dir, model_type, filename)
        if os.path.exists(host_path):
            logging.info(f"Model {filename} already exists")
            return ModelDownloadResult(True, filename, "Already exists")
        
        logging.info(f"Downloading {filename} from HuggingFace {repo_id}...")
        
        # Construct HuggingFace download URL
        hf_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
        
        returncode, stdout, stderr = self._run_docker_cmd_streaming([
            "exec", self.container_name,
            "wget", "-q", "-O", f"{target_dir}/{filename}", hf_url
        ], timeout=1800)
        
        if returncode == 0:
            logging.info(f"Successfully downloaded {filename} from HuggingFace")
            return ModelDownloadResult(True, filename, "Downloaded successfully")
        else:
            logging.error(f"Failed to download from HuggingFace: {stderr}")
            return ModelDownloadResult(False, filename, stderr)
    
    def list_models(self, model_type: str = "checkpoints") -> List[str]:
        """List available models of a given type."""
        host_path = os.path.join(self.host_models_dir, model_type)
        if not os.path.exists(host_path):
            return []
        
        models = []
        for f in os.listdir(host_path):
            if f.endswith(('.safetensors', '.ckpt', '.pt', '.pth', '.bin')):
                models.append(f)
        return models
    
    def load_workflow_manifest(self) -> Dict[str, Any]:
        """Load the workflow manifest that maps workflows to required models."""
        manifest_path = os.path.join(os.path.dirname(__file__), '..', 'workflows', 'manifest.json')
        manifest_path = os.path.abspath(manifest_path)
        
        if not os.path.exists(manifest_path):
            logging.debug(f"No workflow manifest found at {manifest_path}")
            return {}
        
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load workflow manifest: {e}")
            return {}
    
    def download_models_for_workflow(self, workflow_name: str) -> bool:
        """Download all models required for a specific workflow.
        
        Args:
            workflow_name: The workflow key in manifest.json (e.g., 'text-to-video-wan21')
            
        Returns:
            True if all models are available (existing or downloaded), False otherwise.
        """
        manifest = self.load_workflow_manifest()
        
        if workflow_name not in manifest:
            logging.debug(f"Workflow '{workflow_name}' not found in manifest")
            return True  # No known requirements, assume OK
        
        workflow_config = manifest[workflow_name]
        models = workflow_config.get('models', [])
        
        if not models:
            return True
        
        logging.info(f"Checking {len(models)} model(s) for workflow '{workflow_name}'...")
        
        all_success = True
        for model in models:
            model_type = model.get('type', 'checkpoints')
            filename = model.get('filename')
            url = model.get('url')
            
            if not filename or not url:
                continue
            
            # Check if model already exists
            host_path = os.path.join(self.host_models_dir, model_type, filename)
            if os.path.exists(host_path):
                size_gb = os.path.getsize(host_path) / (1024**3)
                logging.info(f"Model {filename} already exists ({size_gb:.1f} GB)")
                continue
            
            # Download the model
            result = self.download_model(url, filename, model_type)
            if not result.success:
                logging.error(f"Failed to download {filename}: {result.message}")
                all_success = False
            else:
                # Verify download completed
                if os.path.exists(host_path):
                    size_gb = os.path.getsize(host_path) / (1024**3)
                    logging.info(f"Downloaded {filename} ({size_gb:.1f} GB)")
                else:
                    logging.error(f"Download reported success but file not found: {host_path}")
                    all_success = False
        
        return all_success
    
    def get_workflow_requirements(self, workflow_name: str) -> Dict[str, Any]:
        """Get the model and node requirements for a workflow.
        
        Returns dict with 'models' list and 'custom_nodes' list.
        """
        manifest = self.load_workflow_manifest()
        
        if workflow_name not in manifest:
            return {'models': [], 'custom_nodes': []}
        
        config = manifest[workflow_name]
        return {
            'models': config.get('models', []),
            'custom_nodes': config.get('custom_nodes', [])
        }
    
    def get_gpu_vram_mb(self) -> int:
        """Get the total VRAM of the primary GPU in MB.
        
        Returns 0 if no GPU is detected.
        """
        try:
            from services.hardware_profiler import get_hardware_profile
            profile = get_hardware_profile()
            gpus = profile.get('gpus', [])
            if gpus:
                # Return the VRAM of the primary (first) GPU
                return int(gpus[0].get('vram', 0))
            return 0
        except Exception as e:
            logging.warning(f"Failed to detect GPU VRAM: {e}")
            return 0
    
    def can_run_workflow(self, workflow_name: str) -> tuple[bool, str]:
        """Check if the user's GPU can run a specific workflow.
        
        Args:
            workflow_name: The workflow key in manifest.json
            
        Returns:
            Tuple of (can_run, reason_message)
        """
        manifest = self.load_workflow_manifest()
        
        if workflow_name not in manifest:
            # Unknown workflow, assume it can run
            return True, "Unknown workflow, no VRAM requirement specified"
        
        config = manifest[workflow_name]
        required_vram = config.get('vram_required_mb', 0)
        
        if required_vram == 0:
            return True, "No VRAM requirement specified"
        
        available_vram = self.get_gpu_vram_mb()
        
        if available_vram == 0:
            return False, "No GPU detected"
        
        if available_vram >= required_vram:
            return True, f"GPU has {available_vram}MB VRAM (requires {required_vram}MB)"
        else:
            return False, f"GPU has {available_vram}MB VRAM but workflow requires {required_vram}MB"
    
    def get_runnable_workflows(self) -> List[Dict[str, Any]]:
        """Get list of workflows that can run on the user's GPU.
        
        Returns list of workflow configs that the GPU can handle.
        """
        manifest = self.load_workflow_manifest()
        available_vram = self.get_gpu_vram_mb()
        
        runnable = []
        for workflow_key, config in manifest.items():
            required_vram = config.get('vram_required_mb', 0)
            
            # Include if no VRAM requirement or GPU meets requirement
            if required_vram == 0 or available_vram >= required_vram:
                runnable.append({
                    'key': workflow_key,
                    'name': config.get('name', workflow_key),
                    'category': config.get('category', 'other'),
                    'vram_required_mb': required_vram,
                    **config
                })
        
        logging.info(f"GPU VRAM: {available_vram}MB. {len(runnable)}/{len(manifest)} workflows are runnable.")
        return runnable
    
    def ensure_workflow_ready(self, workflow_name: str) -> tuple[bool, str]:
        """Ensure a workflow is ready to run: check GPU capability and download models.
        
        This is the main entry point for automatic model downloading.
        
        Args:
            workflow_name: The workflow key in manifest.json (e.g., 'text-to-image-sdxl')
            
        Returns:
            Tuple of (is_ready, message)
        """
        # First check GPU capability
        can_run, reason = self.can_run_workflow(workflow_name)
        if not can_run:
            return False, f"Cannot run workflow: {reason}"
        
        # Then ensure models are downloaded
        logging.info(f"Ensuring models are ready for workflow '{workflow_name}'...")
        if not self.download_models_for_workflow(workflow_name):
            return False, "Failed to download required models"
        
        return True, "Workflow is ready to run"
    
    # ========== Node Management ==========
    
    def install_node(self, git_url: str) -> NodeInstallResult:
        """Install a custom node via git clone inside the container."""
        package_name = git_url.rstrip('/').split('/')[-1].replace('.git', '')
        
        # Check if already exists on host (since we're using volume mounts)
        host_path = os.path.join(self.host_custom_nodes_dir, package_name)
        if os.path.exists(host_path):
            logging.info(f"Node {package_name} already installed at {host_path}")
            return NodeInstallResult(True, package_name, "Already installed")
        
        logging.info(f"Installing node {package_name} from {git_url}...")
        
        target_path = f"{self.CUSTOM_NODES_PATH}/{package_name}"
        
        # Clone the repository
        returncode, stdout, stderr = self._run_docker_cmd_streaming([
            "exec", self.container_name,
            "git", "clone", "--depth", "1", git_url, target_path
        ], timeout=300)
        
        if returncode != 0:
            return NodeInstallResult(False, package_name, stderr)
        
        # Install requirements if present
        req_path = f"{target_path}/requirements.txt"
        check_result = self._run_docker_cmd(
            ["exec", self.container_name, "test", "-f", req_path],
            timeout=10
        )
        
        if check_result.returncode == 0:
            logging.info(f"Installing requirements for {package_name}...")
            returncode, stdout, stderr = self._run_docker_cmd_streaming([
                "exec", self.container_name,
                "pip", "install", "-r", req_path
            ], timeout=300)
            
            if returncode != 0:
                logging.warning(f"Requirements install had issues: {stderr}")
        
        logging.info(f"Successfully installed {package_name}")
        self._installed_nodes_cache = None  # Invalidate cache
        return NodeInstallResult(True, package_name, "Installed successfully")
    
    def get_installed_nodes(self) -> Dict[str, dict]:
        """Query ComfyUI for installed node types via /object_info API."""
        if self._installed_nodes_cache is not None:
            return self._installed_nodes_cache
        
        try:
            response = requests.get(f"{self.comfyui_url}/object_info", timeout=10)
            if response.status_code == 200:
                self._installed_nodes_cache = response.json()
                logging.info(f"Discovered {len(self._installed_nodes_cache)} installed nodes")
                return self._installed_nodes_cache
        except requests.exceptions.RequestException as e:
            logging.warning(f"Failed to get installed nodes: {e}")
        
        return {}
    
    def invalidate_cache(self):
        """Clear the installed nodes cache."""
        self._installed_nodes_cache = None
        logging.debug("Invalidated installed nodes cache")
    
    def validate_workflow(self, workflow_data: dict) -> tuple[bool, List[str]]:
        """Check if all nodes in a workflow are installed.
        
        Returns:
            Tuple of (is_valid, list_of_missing_class_types)
        """
        installed = set(self.get_installed_nodes().keys())
        
        graph = workflow_data.get("prompt", workflow_data)
        required = set()
        
        for node in graph.values():
            if isinstance(node, dict) and "class_type" in node:
                required.add(node["class_type"])
        
        missing = list(required - installed)
        return len(missing) == 0, missing
    
    def get_node_to_package_map(self) -> Dict[str, str]:
        """Fetch node-to-package mapping from ComfyUI-Manager's GitHub.
        
        Returns:
            Dict mapping class_type names to git repository URLs.
        """
        node_map = {}
        
        GITHUB_URL = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json"
        try:
            logging.info("Fetching extension-node-map.json from GitHub...")
            response = requests.get(GITHUB_URL, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for git_url, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    node_types = value[0]
                    if isinstance(node_types, list):
                        for node_type in node_types:
                            if isinstance(node_type, str):
                                node_map[node_type] = git_url
            
            logging.info(f"Loaded {len(node_map)} node-to-package mappings")
        except Exception as e:
            logging.warning(f"Failed to fetch extension-node-map.json: {e}")
        
        return node_map
    
    # ========== Workflow Management ==========
    
    def scan_workflows(self) -> List[Dict[str, Any]]:
        """Scan for available workflows from the host workflows directory."""
        workflows = []
        
        # Scan host workflows directory (mounted as volume)
        if os.path.exists(self.host_workflows_dir):
            for root, dirs, files in os.walk(self.host_workflows_dir):
                for f in files:
                    if f.endswith('.json'):
                        filepath = os.path.join(root, f)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as file:
                                data = json.load(file)
                            
                            # Skip UI-format workflows
                            if "nodes" in data and "links" in data and "prompt" not in data:
                                continue
                            
                            workflow_name = os.path.splitext(f)[0]
                            rel_path = os.path.relpath(filepath, self.host_workflows_dir)
                            category = os.path.dirname(rel_path) or "default"
                            
                            workflows.append({
                                "name": workflow_name,
                                "filename": f,
                                "path": filepath,
                                "category": category,
                                "source": "local",
                                "source_name": "OpenFork",
                                "input_schema": self._infer_inputs_from_workflow(data),
                                "metadata": {}
                            })
                        except Exception as e:
                            logging.warning(f"Error reading workflow {filepath}: {e}")
        
        logging.info(f"Found {len(workflows)} locally stored workflows")
        return workflows
    
    def get_workflow_content(self, workflow_name: str) -> Union[Dict[str, Any], None]:
        """Get workflow content by name."""
        # Search in host workflows directory
        for root, dirs, files in os.walk(self.host_workflows_dir):
            for f in files:
                name = os.path.splitext(f)[0]
                if name == workflow_name or name == f"{workflow_name}.api":
                    filepath = os.path.join(root, f)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as file:
                            return json.load(file)
                    except Exception as e:
                        logging.error(f"Error reading workflow {filepath}: {e}")
        
        return None
    
    def _infer_inputs_from_workflow(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Heuristically determine input fields from workflow structure."""
        inputs = []
        graph = data.get("prompt", data)
        
        if not isinstance(graph, dict):
            return inputs
        
        for node_id, node in graph.items():
            if not isinstance(node, dict):
                continue
            
            class_type = node.get("class_type")
            node_inputs = node.get("inputs", {})
            
            if class_type == "CLIPTextEncode":
                text_val = node_inputs.get("text")
                if isinstance(text_val, str):
                    label = "negative_prompt" if any(x in text_val.lower() for x in ["negative", "bad", "nsfw"]) else "prompt"
                    inputs.append({
                        "name": label,
                        "type": "text",
                        "default": text_val,
                        "node_id": node_id,
                        "widget_name": "text"
                    })
            
            elif class_type in ["KSampler", "KSamplerAdvanced"]:
                seed = node_inputs.get("seed") or node_inputs.get("noise_seed")
                if isinstance(seed, (int, float)):
                    inputs.append({
                        "name": "seed",
                        "type": "number",
                        "default": seed,
                        "node_id": node_id,
                        "widget_name": "seed"
                    })
        
        return inputs
    
    # ========== File I/O ==========
    
    def get_input_directory(self) -> str:
        """Get the host input directory path (for copying files into)."""
        return self.host_input_dir
    
    def get_output_directory(self) -> str:
        """Get the host output directory path (for reading outputs)."""
        return self.host_output_dir
    
    def copy_file_to_input(self, source_path: str) -> Union[str, None]:
        """Copy a file to the container's input directory.
        
        Returns the filename (not full path) that can be used in workflows.
        """
        if not os.path.exists(source_path):
            logging.error(f"Source file not found: {source_path}")
            return None
        
        filename = os.path.basename(source_path)
        dest_path = os.path.join(self.host_input_dir, filename)
        
        try:
            shutil.copy2(source_path, dest_path)
            logging.info(f"Copied {filename} to input directory")
            return filename
        except Exception as e:
            logging.error(f"Failed to copy file to input: {e}")
            return None
    
    def get_output_file(self, filename: str, subfolder: str = "") -> Union[str, None]:
        """Get the full path to an output file.
        
        Returns the host path if file exists, None otherwise.
        """
        if subfolder:
            path = os.path.join(self.host_output_dir, subfolder, filename)
        else:
            path = os.path.join(self.host_output_dir, filename)
        
        if os.path.exists(path):
            return path
        
        logging.warning(f"Output file not found: {path}")
        return None
    
    def list_output_files(self, subfolder: str = "") -> List[str]:
        """List files in the output directory."""
        if subfolder:
            path = os.path.join(self.host_output_dir, subfolder)
        else:
            path = self.host_output_dir
        
        if not os.path.exists(path):
            return []
        
        files = []
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path, f)):
                files.append(f)
        return files
    
    def stop(self):
        """Alias for stop_container for compatibility."""
        return self.stop_container()
    
    def start(self):
        """Alias for start_container for compatibility."""
        return self.start_container()
    
    # ========== Dynamic Workflow Templates ==========
    
    def fetch_workflow_templates(self) -> Dict[str, Any]:
        """Fetch workflow templates from the running ComfyUI instance.
        
        Queries the /api/workflow_templates endpoint which returns built-in
        and custom node workflow templates.
        
        Returns:
            Dict mapping template categories/names to workflow data.
        """
        if not self.is_container_running():
            logging.warning("Cannot fetch workflow templates: container not running")
            return {}
        
        try:
            response = requests.get(f"{self.comfyui_url}/api/workflow_templates", timeout=30)
            if response.status_code == 200:
                templates = response.json()
                logging.info(f"Fetched {len(templates)} workflow template categories from ComfyUI")
                return templates
            else:
                logging.warning(f"Failed to fetch workflow templates: HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            logging.warning(f"Failed to fetch workflow templates: {e}")
        
        return {}
    
    def get_dynamic_workflows(self) -> List[Dict[str, Any]]:
        """Get all available workflows dynamically from ComfyUI.
        
        Combines templates from /api/workflow_templates with WorkflowAnalyzer
        to extract metadata like inputs, outputs, and VRAM requirements.
        
        Returns:
            List of workflow metadata dicts with 'name', 'category', 'input_schema',
            'vram_required_mb', 'workflow_data', etc.
        """
        workflows = []
        templates = self.fetch_workflow_templates()
        
        if not templates:
            logging.info("No dynamic templates available, falling back to local workflows")
            return self.scan_workflows()
        
        # Import WorkflowAnalyzer
        try:
            from services.workflow_analyzer import WorkflowAnalyzer
            analyzer = WorkflowAnalyzer()
        except ImportError:
            logging.warning("WorkflowAnalyzer not available, cannot analyze templates")
            analyzer = None
        
        # Templates are organized by source (custom node name or "default")
        for source_name, source_data in templates.items():
            if not isinstance(source_data, dict):
                continue
            
            template_list = source_data.get("templates", [])
            for template in template_list:
                if not isinstance(template, dict):
                    continue
                
                name = template.get("name", "Unknown")
                workflow_data = template.get("workflow", {})
                
                if not workflow_data:
                    # Try to load from file path if provided
                    file_path = template.get("file")
                    if file_path:
                        workflow_data = self._load_template_file(file_path)
                
                if not workflow_data:
                    continue
                
                # Analyze workflow for inputs and metadata
                input_schema = []
                vram_required = 4000  # Default 4GB
                category = "general"
                
                if analyzer:
                    try:
                        metadata = analyzer.analyze(workflow_data, name)
                        input_schema = [
                            {
                                "name": inp.name,
                                "type": inp.type,
                                "default": inp.default,
                                "node_id": inp.node_id,
                                "widget_name": inp.widget_name
                            }
                            for inp in metadata.inputs
                        ]
                        vram_required = metadata.estimated_vram_mb
                        category = metadata.category
                    except Exception as e:
                        logging.warning(f"Failed to analyze workflow '{name}': {e}")
                        input_schema = self._infer_inputs_from_workflow(workflow_data)
                        vram_required = self._estimate_vram_from_workflow(workflow_data)
                else:
                    input_schema = self._infer_inputs_from_workflow(workflow_data)
                    vram_required = self._estimate_vram_from_workflow(workflow_data)
                
                workflows.append({
                    "name": name,
                    "filename": f"{name}.json",
                    "path": None,  # Dynamic, not file-based
                    "category": category,
                    "source": "comfyui",
                    "source_name": source_name,
                    "input_schema": input_schema,
                    "vram_required_mb": vram_required,
                    "workflow_data": workflow_data,
                    "metadata": {
                        "dynamic": True,
                        "template_source": source_name
                    }
                })
        
        logging.info(f"Loaded {len(workflows)} dynamic workflow templates from ComfyUI")
        return workflows
    
    def _load_template_file(self, file_path: str) -> Dict[str, Any]:
        """Load a workflow template from a file path inside the container."""
        try:
            # Templates may reference paths inside the container
            # Try to access via the host mount if it maps
            if file_path.startswith(self.WORKFLOWS_PATH):
                # Map container path to host path
                relative = file_path[len(self.WORKFLOWS_PATH):].lstrip("/")
                host_path = os.path.join(self.host_workflows_dir, relative)
                if os.path.exists(host_path):
                    with open(host_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            
            # Try direct host path
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
                    
        except Exception as e:
            logging.warning(f"Failed to load template file {file_path}: {e}")
        
        return {}
    
    def _estimate_vram_from_workflow(self, workflow_data: Dict[str, Any]) -> int:
        """Estimate VRAM requirement based on model loader nodes in workflow."""
        # Known model nodes and their approximate VRAM usage in MB
        MODEL_VRAM = {
            "CheckpointLoader": 4000,
            "CheckpointLoaderSimple": 4000,
            "UNETLoader": 4000,
            "VAELoader": 500,
            "CLIPLoader": 1000,
            "LoraLoader": 500,
            "ControlNetLoader": 2000,
            # Video models need more VRAM
            "WanVideoSampler": 24000,
            "WanImageToVideo": 24000,
            "HunyuanVideoSampler": 24000,
            "LTXVSampler": 16000,
            "DiffRhythmRun": 8000,
        }
        
        graph = workflow_data.get("prompt", workflow_data)
        if not isinstance(graph, dict):
            return 4000
        
        total_vram = 0
        for node in graph.values():
            if isinstance(node, dict):
                class_type = node.get("class_type", "")
                if class_type in MODEL_VRAM:
                    total_vram = max(total_vram, MODEL_VRAM[class_type])
        
        return total_vram if total_vram > 0 else 4000
    
    def get_all_workflows(self) -> List[Dict[str, Any]]:
        """Get all available workflows from both dynamic and local sources.
        
        This is the main entry point for workflow discovery.
        Combines:
        1. Dynamic templates from running ComfyUI (/api/workflow_templates)
        2. Local workflows from host directory
        
        Returns filtered list based on GPU capability.
        """
        all_workflows = []
        seen_names = set()
        
        # First, try to get dynamic templates from running ComfyUI
        if self.is_container_running():
            dynamic = self.get_dynamic_workflows()
            for wf in dynamic:
                if wf["name"] not in seen_names:
                    all_workflows.append(wf)
                    seen_names.add(wf["name"])
        
        # Then add local workflows (won't duplicate if name already seen)
        local = self.scan_workflows()
        for wf in local:
            if wf["name"] not in seen_names:
                # Add VRAM estimate if not present
                if "vram_required_mb" not in wf:
                    workflow_data = self.get_workflow_content(wf["name"])
                    if workflow_data:
                        wf["vram_required_mb"] = self._estimate_vram_from_workflow(workflow_data)
                    else:
                        wf["vram_required_mb"] = 4000
                all_workflows.append(wf)
                seen_names.add(wf["name"])
        
        # Filter by GPU capability
        available_vram = self.get_gpu_vram_mb()
        if available_vram > 0:
            runnable = [
                wf for wf in all_workflows 
                if wf.get("vram_required_mb", 0) <= available_vram
            ]
            logging.info(
                f"GPU: {available_vram}MB VRAM. "
                f"{len(runnable)}/{len(all_workflows)} workflows can run on this GPU"
            )
            return runnable
        
        return all_workflows

