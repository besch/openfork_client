import os
import json
import logging
import random
import shutil
import time
import requests
from typing import Union, Dict
from abc import ABC, abstractmethod
from services.orchestrator_service import TokenExpiredError
from config import DEV_MODE
from services.local_comfyui_manager import LocalComfyUIManager
import subprocess
from utils.media_utils import (
    get_audio_duration, find_audio_in_output, find_audio_file_in_directory,
    find_image_in_output, find_video_in_output, generate_thumbnail,
    get_video_duration, get_video_dimensions, get_video_framerate, extract_last_frame
)


class BaseJobProcessor(ABC):
    def __init__(self, client, job, shutdown_event):
        self.client = client
        self.orchestrator_service = client.orchestrator_service
        self.comfyui_client = client.comfyui_client
        self.job = job
        self.job_id = job['id']
        self.shutdown_event = shutdown_event
        self.root_dir = client.root_dir
        self.input_dir = client.input_dir
        self.cache_dir = client.cache_dir
        self.positive_prompt = job.get('prompt') or ""
        self.negative_prompt = job.get('negative_prompt') or ""
        self.workflow_type = job.get('workflow_type')

    def _check_interruption(self, outputs):
        if outputs == "interrupted":
            logging.warning(f"Processing of job {self.job_id} was interrupted.")
            return True
        return False

    def _retrieve_output_file(self, filename: str, subfolder: str) -> Union[str, None]:
        """Copies a file from ComfyUI output directory to a temporary location on the client."""
        safe_filename = os.path.basename(filename)
        
        comfy_output_dir = self.client.comfyui_manager.get_output_directory()
        if not comfy_output_dir or not os.path.exists(comfy_output_dir):
             logging.error("ComfyUI output directory not found.")
             return None

        source_path = os.path.join(comfy_output_dir, subfolder, safe_filename)
        
        if not os.path.exists(source_path):
             logging.error(f"Output file not found at {source_path}")
             return None

        os.makedirs(self.cache_dir, exist_ok=True)

        temp_filename = f"{self.job_id}_{safe_filename}"
        dest_path = os.path.join(self.cache_dir, temp_filename)

        try:
            shutil.copy2(source_path, dest_path)
            logging.info(f"Successfully retrieved output file to: {dest_path}")
            return dest_path
        except Exception as e:
            logging.error(f"Failed to retrieve output file: {e}", exc_info=True)
            return None

    def _trigger_and_get_output(self, payload):
        prompt_id = self.comfyui_client.trigger_workflow(payload)
        if not prompt_id:
            logging.error(f"Failed to trigger workflow for job {self.job_id}.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return None

        outputs = self.comfyui_client.get_workflow_output(
            prompt_id,
            job_id=self.job_id,
            orchestrator_service=self.orchestrator_service,
            timeout_sec=7200,
            shutdown_event=self.shutdown_event
        )
        if self._check_interruption(outputs):
            return None
        
        if not outputs:
            logging.error(f"Workflow for job {self.job_id} failed to produce outputs.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return None
            
        return outputs

    @abstractmethod
    def process(self):
        pass


class TextGenerationJobProcessor(BaseJobProcessor):
    """Processor for text generation using Ollama API (not ComfyUI)."""
    
    def process(self):
        if not self.job:
            logging.error(f"Job object is None for TextGenerationJobProcessor. Cannot proceed.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        api_base = os.getenv("LLM_API_BASE", "http://localhost:11434")
        
        logging.info(f"Waiting for Ollama service to be ready at {api_base}...")
        time.sleep(3)
        
        ready = False
        for attempt in range(60):
            if self.shutdown_event.is_set():
                logging.info("Shutdown event received during Ollama readiness check.")
                return
            try:
                response = requests.get(f"{api_base}/api/tags", timeout=2)
                response.raise_for_status()
                ready = True
                logging.info(f"Ollama service is ready! (attempt {attempt + 1}/60)")
                break
            except requests.exceptions.RequestException as e:
                if attempt % 10 == 0:
                    logging.debug(f"Ollama not ready yet (attempt {attempt + 1}/60): {e}")
                time.sleep(1)
        
        if not ready:
            logging.error(f"Ollama service failed to become ready within 60 seconds at {api_base}")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        inputs = self.job.get('inputs', {})
        model_name = inputs.get('model', 'llama3.1:8b')
        system_prompt = inputs.get('system_prompt', "You are a helpful assistant.")
        temperature = inputs.get('temperature', 0.7)
        max_tokens = inputs.get('max_tokens', 2000)
        seed = inputs.get('seed') or random.randint(0, 2**31 - 1)

        logging.info(f"Generating text with model {model_name}...")
        
        try:
            tags_response = requests.get(f"{api_base}/api/tags", timeout=5)
            tags_response.raise_for_status()
            tags_data = tags_response.json()
            
            available_models = [m.get('name', '') for m in tags_data.get('models', [])]
            
            if model_name not in available_models:
                logging.info(f"Model {model_name} not found. Pulling it now...")
                pull_payload = {"name": model_name, "stream": False}
                pull_response = requests.post(f"{api_base}/api/pull", json=pull_payload, timeout=600)
                pull_response.raise_for_status()
                logging.info(f"Successfully pulled model {model_name}")
        except Exception as e:
            logging.error(f"Failed to verify/pull model {model_name}: {e}")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        try:
            full_prompt = f"{system_prompt}\n\nUser: {self.positive_prompt}\n\nAssistant:"
            
            payload = {
                "model": model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "seed": seed
                }
            }
            
            logging.info(f"Calling Ollama API at {api_base}/api/generate with model {model_name}")
            response = requests.post(f"{api_base}/api/generate", json=payload, timeout=1200)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get('response', '')
            
            if not generated_text:
                logging.error(f"Ollama returned empty response. Full result: {result}")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                return
            
            logging.info(f"Generation complete. Length: {len(generated_text)} chars.")

            output_filename = f"{self.job_id}_script.txt"
            output_path = os.path.join(self.cache_dir, output_filename)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(generated_text)

            storage_path = self.orchestrator_service.upload_output(output_path, self.job_id, "text/plain")
            
            if storage_path:
                self.orchestrator_service.update_job_status(
                    self.job_id, 
                    'completed', 
                    storage_path=storage_path,
                    completion_metadata={"model": model_name}
                )
            else:
                logging.error("Failed to upload generated script.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')

            if os.path.exists(output_path):
                os.remove(output_path)

        except Exception as e:
            logging.error(f"Text generation failed: {e}", exc_info=True)
            self.orchestrator_service.update_job_status(self.job_id, 'failed')


class GenericComfyWorkflowProcessor(BaseJobProcessor):
    """
    Generic processor that can execute ANY ComfyUI workflow without custom code.
    
    This processor:
    1. Loads workflow by name from local ComfyUI or by path
    2. Automatically detects inputs using WorkflowAnalyzer
    3. Injects user-provided input values
    4. Executes the workflow
    5. Automatically detects and uploads outputs (video, audio, image)
    """
    
    def __init__(self, dgn_client, job_data, shutdown_event, local_comfyui_manager=None):
        super().__init__(dgn_client, job_data, shutdown_event)
        self.local_comfyui_manager = local_comfyui_manager or dgn_client.comfyui_manager
        
        inputs = job_data.get("inputs", {})
        self.workflow_name = inputs.get("workflowName") or inputs.get("workflow_name")
        self.workflow_path = inputs.get("workflowPath") or inputs.get("workflow_path")
        
        if not self.workflow_name and not self.workflow_path:
            logging.warning("GenericComfyWorkflowProcessor: No workflow_name or workflow_path provided")
        
        try:
            from services.workflow_analyzer import WorkflowAnalyzer
            self._analyzer = WorkflowAnalyzer()
        except ImportError:
            self._analyzer = None

    def _is_ui_format(self, workflow_data: dict) -> bool:
        """Check if workflow is in UI format (with nodes/links arrays) vs API format."""
        return "nodes" in workflow_data and "links" in workflow_data and "prompt" not in workflow_data

    def _convert_ui_to_api_format(self, ui_workflow: dict) -> dict:
        """Convert ComfyUI UI format workflow to API format.
        
        UI format has 'nodes' array with 'id', 'type', 'widgets_values' etc.
        API format has numbered dict keys with 'class_type' and 'inputs'.
        """
        nodes = ui_workflow.get("nodes", [])
        links = ui_workflow.get("links", [])
        
        # Build link map: link_id -> (from_node_id, from_slot, to_node_id, to_slot, type)
        link_map = {}
        for link in links:
            if len(link) >= 6:
                link_id, from_node, from_slot, to_node, to_slot, link_type = link[:6]
                link_map[link_id] = {
                    "from_node": from_node,
                    "from_slot": from_slot,
                    "to_node": to_node,
                    "to_slot": to_slot,
                    "type": link_type
                }
        
        api_prompt = {}
        
        for node in nodes:
            node_id = str(node.get("id"))
            class_type = node.get("type")
            
            if not class_type:
                continue
            
            inputs = {}
            
            # Process widget values - these come from the UI
            widgets_values = node.get("widgets_values", [])
            widget_names = node.get("widgets_names", [])
            
            # If no explicit names, try to assign based on common patterns
            if widgets_values and not widget_names:
                # Use generic naming
                for i, val in enumerate(widgets_values):
                    if val is not None:
                        inputs[f"widget_{i}"] = val
            else:
                for i, val in enumerate(widgets_values):
                    if i < len(widget_names):
                        inputs[widget_names[i]] = val
            
            # Process inputs from links
            node_inputs = node.get("inputs", [])
            for inp in node_inputs:
                if isinstance(inp, dict):
                    inp_name = inp.get("name")
                    link_id = inp.get("link")
                    if link_id and link_id in link_map:
                        link_info = link_map[link_id]
                        # Reference format: [from_node_id, from_slot]
                        inputs[inp_name] = [str(link_info["from_node"]), link_info["from_slot"]]
            
            api_prompt[node_id] = {
                "class_type": class_type,
                "inputs": inputs
            }
        
        logging.info(f"Converted UI workflow with {len(nodes)} nodes to API format")
        return {"prompt": api_prompt}

    def _get_workflow(self):
        """Load workflow from local ComfyUI or by absolute path."""
        workflow = None
        
        if self.workflow_name and self.local_comfyui_manager:
            workflow = self.local_comfyui_manager.get_workflow_content(self.workflow_name)
            if workflow:
                logging.info(f"Loaded workflow '{self.workflow_name}' from local ComfyUI")
        
        if not workflow and self.workflow_path and os.path.exists(self.workflow_path):
            try:
                with open(self.workflow_path, 'r', encoding='utf-8') as f:
                    workflow = json.load(f)
                logging.info(f"Loaded workflow from path: {self.workflow_path}")
            except Exception as e:
                logging.error(f"Failed to load workflow from {self.workflow_path}: {e}")
        
        if not workflow:
            return None
        
        # Check if it's UI format and convert if needed
        if self._is_ui_format(workflow):
            logging.info("Detected UI format workflow, converting to API format...")
            workflow = self._convert_ui_to_api_format(workflow)
        
        return workflow

    def _inject_inputs(self, workflow_data: dict, user_inputs: dict) -> dict:
        """Inject user inputs into workflow using automatically detected schema."""
        graph = workflow_data
        if "prompt" in workflow_data and isinstance(workflow_data["prompt"], dict):
            graph = workflow_data["prompt"]
        
        if self._analyzer:
            schema = self._analyzer.to_input_schema(workflow_data, self.workflow_name or "")
        elif self.local_comfyui_manager:
            schema = self.local_comfyui_manager._infer_inputs_from_workflow(workflow_data)
        else:
            schema = []
        
        schema_by_name = {s['name']: s for s in schema}
        
        for field_name, value in user_inputs.items():
            if field_name in ['workflowName', 'workflow_name', 'workflowPath', 'workflow_path']:
                continue
                
            if field_name not in schema_by_name:
                logging.debug(f"Input '{field_name}' not found in schema, skipping")
                continue
            
            schema_entry = schema_by_name[field_name]
            node_id = schema_entry['node_id']
            widget_name = schema_entry['widget_name']
            input_type = schema_entry.get('type', 'text')
            
            if input_type in ['image', 'video', 'input_video', 'input_audio', 'audio']:
                if isinstance(value, str) and (value.startswith("http") or "supabase" in value.lower()):
                    input_dir = self.local_comfyui_manager.get_input_directory() if self.local_comfyui_manager else self.input_dir
                    local_path = self.orchestrator_service.download_asset_by_url(value, input_dir)
                    if local_path:
                        value = os.path.basename(local_path)
                        logging.info(f"Downloaded {input_type} for '{field_name}' -> {value}")
            
            if node_id in graph and "inputs" in graph[node_id]:
                graph[node_id]["inputs"][widget_name] = value
                logging.info(f"Injected input '{field_name}' -> Node {node_id}.{widget_name}")
        
        return {"prompt": graph}

    def process(self):
        """Execute the workflow and handle outputs."""
        if not self.job:
            logging.error(f"Job object is None for GenericComfyWorkflowProcessor")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return
        
        workflow_data = self._get_workflow()
        if not workflow_data:
            logging.error(f"Could not load workflow for job {self.job_id}")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return
        
        if self.local_comfyui_manager:
            is_valid, missing_nodes = self.local_comfyui_manager.validate_workflow(workflow_data)
            if not is_valid:
                logging.warning(f"Workflow has missing nodes: {missing_nodes}")
                # Attempt automatic installation
                if not self._auto_install_missing_nodes(missing_nodes):
                    logging.error(f"Failed to auto-install missing nodes for job {self.job_id}")
                    self.orchestrator_service.update_job_status(
                        self.job_id, 'failed',
                        completion_metadata={"error": "missing_nodes", "missing": missing_nodes}
                    )
                    return
        
        user_inputs = self.job.get('inputs', {})
        payload = self._inject_inputs(workflow_data, user_inputs)
        
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return
        
        self._handle_outputs(outputs)

    def _auto_install_missing_nodes(self, missing_class_types: list[str]) -> bool:
        """
        Attempt to automatically install missing custom nodes.
        
        Args:
            missing_class_types: List of missing node class_type names
            
        Returns:
            True if all nodes were successfully installed and ComfyUI restarted.
        """
        if not missing_class_types:
            return True
        
        logging.info(f"Attempting to auto-install {len(missing_class_types)} missing node(s): {missing_class_types}")
        
        # Get the node-to-package mapping
        node_to_package = self.local_comfyui_manager.get_node_to_package_map()
        if not node_to_package:
            logging.error("Could not load node-to-package mapping. Cannot auto-install nodes.")
            return False
        
        # Resolve class_types to git URLs
        packages_to_install = {}  # git_url -> list of class_types it provides
        unresolved = []
        
        # Fallback mappings for common nodes not in the registry
        fallback_mappings = {
            # pythongosssss ComfyUI-Custom-Scripts nodes
            "MarkdownNote": "https://github.com/pythongosssss/ComfyUI-Custom-Scripts",
            "Note": "https://github.com/pythongosssss/ComfyUI-Custom-Scripts",
            "ShowText": "https://github.com/pythongosssss/ComfyUI-Custom-Scripts",
            "StringFunction": "https://github.com/pythongosssss/ComfyUI-Custom-Scripts",
            # Prefix-based fallbacks
        }
        
        # Prefix-based fallback mappings
        prefix_fallbacks = {
            "VHS_": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
            "KJ": "https://github.com/kijai/ComfyUI-KJNodes",
            "WAS ": "https://github.com/WASasquatch/was-node-suite-comfyui",
            "Impact": "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
        }
        
        for class_type in missing_class_types:
            if class_type in node_to_package:
                git_url = node_to_package[class_type]
                if git_url not in packages_to_install:
                    packages_to_install[git_url] = []
                packages_to_install[git_url].append(class_type)
            elif class_type in fallback_mappings:
                git_url = fallback_mappings[class_type]
                if git_url not in packages_to_install:
                    packages_to_install[git_url] = []
                packages_to_install[git_url].append(class_type)
                logging.info(f"Using fallback mapping for {class_type} -> {git_url}")
            else:
                # Try prefix-based matching
                matched = False
                for prefix, git_url in prefix_fallbacks.items():
                    if class_type.startswith(prefix):
                        if git_url not in packages_to_install:
                            packages_to_install[git_url] = []
                        packages_to_install[git_url].append(class_type)
                        logging.info(f"Using prefix fallback for {class_type} -> {git_url}")
                        matched = True
                        break
                if not matched:
                    unresolved.append(class_type)
        
        if unresolved:
            logging.warning(f"Could not find packages for these nodes (not in ComfyUI-Manager registry): {unresolved}")
        
        if not packages_to_install:
            logging.error("No packages to install - all missing nodes are unresolved")
            return False
        
        # Find custom_nodes directory (handles split installations)
        custom_nodes_dir = self.local_comfyui_manager.get_custom_nodes_dir()
        if not custom_nodes_dir:
            logging.error("Could not find custom_nodes directory. Cannot auto-install nodes.")
            return False
        
        logging.info(f"Installing nodes to: {custom_nodes_dir}")
        
        # Install each package via git clone
        all_success = True
        for git_url, class_types in packages_to_install.items():
            package_name = git_url.rstrip('/').split('/')[-1].replace('.git', '')
            target_dir = os.path.join(custom_nodes_dir, package_name)
            
            if os.path.exists(target_dir):
                logging.info(f"Package {package_name} already exists at {target_dir}")
                continue
            
            logging.info(f"Cloning {package_name} (provides: {class_types})...")
            
            try:
                # Prevent git from waiting for credentials
                env = os.environ.copy()
                env["GIT_TERMINAL_PROMPT"] = "0"
                
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", "--progress", git_url, target_dir],
                    capture_output=True,
                    text=True,
                    timeout=120,  # 2 minute timeout
                    env=env
                )
                
                if result.returncode == 0:
                    logging.info(f"Successfully cloned {package_name}")
                    
                    # Install requirements if present
                    requirements_file = os.path.join(target_dir, "requirements.txt")
                    if os.path.exists(requirements_file):
                        logging.info(f"Installing requirements for {package_name}...")
                        subprocess.run(
                            ["pip", "install", "-r", requirements_file],
                            capture_output=True,
                            timeout=300
                        )
                else:
                    logging.error(f"Failed to clone {package_name}: {result.stderr}")
                    all_success = False
                    
            except subprocess.TimeoutExpired:
                logging.error(f"Git clone timed out for {package_name}")
                all_success = False
            except FileNotFoundError:
                logging.error("Git is not installed. Cannot auto-install nodes.")
                return False
            except Exception as e:
                logging.error(f"Error cloning {package_name}: {e}")
                all_success = False
        
        # Restart ComfyUI to load the new nodes
        if not self.local_comfyui_manager.restart(timeout_seconds=90):
            logging.error("Failed to restart ComfyUI after node installation")
            return False
        
        # Re-validate the workflow
        # Need to re-get workflow since we need fresh validation
        workflow_data = self._get_workflow()
        if not workflow_data:
            return False
        
        is_valid, still_missing = self.local_comfyui_manager.validate_workflow(workflow_data)
        
        if not is_valid:
            logging.error(f"Still missing nodes after installation: {still_missing}")
            return False
        
        logging.info("All missing nodes installed and verified successfully!")
        return True

    def _handle_outputs(self, outputs: dict):
        """Detect output type and upload appropriately."""
        try:
            video_info = find_video_in_output(outputs)
            if video_info:
                self._handle_video_output(video_info)
                return
            
            audio_info = find_audio_in_output(outputs)
            if audio_info:
                self._handle_audio_output(audio_info)
                return
            
            image_info = find_image_in_output(outputs)
            if image_info:
                self._handle_image_output(image_info)
                return
            
            logging.warning(f"No recognizable output found for job {self.job_id}")
            self.orchestrator_service.update_job_status(
                self.job_id, 'completed',
                completion_metadata={"status": "completed_no_output"}
            )
            
        except Exception as e:
            logging.error(f"Error handling outputs for job {self.job_id}: {e}", exc_info=True)
            self.orchestrator_service.update_job_status(self.job_id, 'failed')

    def _handle_video_output(self, video_info: tuple):
        """Handle video output upload."""
        video_filename, subfolder = video_info
        temp_path = self._retrieve_output_file(video_filename, subfolder)
        if not temp_path:
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return
        
        try:
            storage_path = self.orchestrator_service.upload_output(temp_path, self.job_id, 'video/mp4')
            
            thumb_path = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
            thumb_storage = None
            if generate_thumbnail(temp_path, thumb_path, width=100):
                thumb_storage = self.orchestrator_service.upload_thumbnail(thumb_path, self.job_id)
                if os.path.exists(thumb_path):
                    os.remove(thumb_path)
            
            duration = get_video_duration(temp_path)
            
            self.orchestrator_service.update_job_status(
                self.job_id, 'completed',
                storage_path=storage_path,
                thumbnail_storage_path=thumb_storage,
                duration_seconds=duration,
                media_type='video/mp4'
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _handle_audio_output(self, audio_info: tuple):
        """Handle audio output upload."""
        audio_filename, subfolder = audio_info
        temp_path = self._retrieve_output_file(audio_filename, subfolder)
        if not temp_path:
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return
        
        try:
            storage_path = self.orchestrator_service.upload_audio_output(temp_path, self.job_id)
            duration = get_audio_duration(temp_path)
            
            self.orchestrator_service.update_job_status(
                self.job_id, 'completed',
                storage_path=storage_path,
                duration_seconds=duration,
                media_type='audio/flac'
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _handle_image_output(self, image_info: tuple):
        """Handle image output upload."""
        image_filename, subfolder = image_info
        temp_path = self._retrieve_output_file(image_filename, subfolder)
        if not temp_path:
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return
        
        try:
            storage_path = self.orchestrator_service.upload_image_output(temp_path, self.job_id)
            
            self.orchestrator_service.update_job_status(
                self.job_id, 'completed',
                storage_path=storage_path,
                thumbnail_storage_path=storage_path,
                media_type='image/png'
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
