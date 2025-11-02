import os
import logging
from typing import Union
from services.docker_manager import docker_manager
from utils.media_utils import get_audio_duration, find_audio_in_output, find_image_in_output, find_video_in_output, generate_thumbnail, get_video_duration
from utils.comfyui_workflow_utils import find_node_by_type_and_field, set_node_property, materialize_image_input
from services.auto_installer import manager_install_custom_node
import time

class MissingDependenciesError(Exception):
    """Custom exception for missing ComfyUI custom node dependencies."""
    def __init__(self, missing_repos: list[str]):
        self.missing_repos = missing_repos
        message = (
            f"Execution failed: {len(missing_repos)} required custom node repositories are not installed.\n"
            f"Please install them and restart the service.\n"
            f"Missing repositories: {', '.join(missing_repos)}"
        )
        super().__init__(message)

class DynamicJobProcessor:
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
        
        self.workflow_template = self._get_workflow_template()
        if not self.workflow_template:
            raise ValueError(f"Failed to load workflow template for job {self.job_id}")
        
        self.workflow_json = self.workflow_template['workflow_json']
        self.input_schema = self.workflow_template.get('input_schema', {})
        self.job_inputs = self.job.get('inputs', {})
        self.workflow_type = self.workflow_template.get('workflow_type')
        self.target_entity = self.workflow_template.get('target_entity')

    def _get_workflow_template(self):
        workflow_template_id = self.job.get('workflow_template_id')
        if not workflow_template_id:
            raise ValueError(f"Job {self.job_id} is missing workflow_template_id.")
        return self.orchestrator_service.get_workflow_template(workflow_template_id)

    def _check_interruption(self, outputs):
        if outputs == "interrupted":
            logging.warning(f"Processing of job {self.job_id} was interrupted.")
            return True
        return False

    def _copy_file_from_container(self, filename: str, subfolder: str) -> Union[str, None]:
        """Copies a file from the active container to a temporary location on the host."""
        safe_filename = os.path.basename(filename)
        source_in_container = os.path.join("/app/ComfyUI/output", subfolder, safe_filename).replace('\\', '/')
        
        os.makedirs(self.cache_dir, exist_ok=True)
        temp_filename = f"{self.job_id}_{safe_filename}"
        dest_on_host = os.path.join(self.cache_dir, temp_filename)

        try:
            docker_manager.copy_file_from_container(
                source_in_container=source_in_container,
                dest_on_host=dest_on_host
            )
            if os.path.exists(dest_on_host):
                logging.info(f"Successfully copied file to temporary host path: {dest_on_host}")
                return dest_on_host
            else:
                raise RuntimeError("docker cp command finished but destination file does not exist.")
        except Exception as e:
            logging.error(f"Failed to copy file from container: {e}", exc_info=True)
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

    def _get_input_value(self, input_name, input_definition):
        value = self.job_inputs.get(input_name)
        if value is None and 'default' in input_definition:
            value = input_definition['default']
        return value

    def _inject_dynamic_inputs(self):
        if not self.input_schema or not self.input_schema.get('properties'):
            logging.info(f"No input schema defined for workflow {self.workflow_template.get('name')}. Skipping dynamic injection.")
            return

        for input_name, input_definition in self.input_schema['properties'].items():
            value = self._get_input_value(input_name, input_definition)
            if value is None:
                if input_name in self.input_schema.get('required', []):
                    logging.warning(f"Required input '{input_name}' missing for job {self.job_id}. Workflow might fail.")
                continue

            node_type = input_definition.get('node_type')
            field_name = input_definition.get('field_name')

            if not node_type or not field_name:
                logging.warning(f"Input '{input_name}' in schema is missing 'node_type' or 'field_name'. Skipping.")
                continue

            if input_definition['type'] == 'image':
                materialized_path = materialize_image_input(value, self.input_dir)
                if materialized_path:
                    value = os.path.basename(materialized_path)
                else:
                    logging.error(f"Failed to materialize image for input '{input_name}'. Skipping injection.")
                    continue

            node = find_node_by_type_and_field(self.workflow_json, node_type, field_name)
            if node:
                set_node_property(node, field_name, value)
                logging.info(f"Injected input '{input_name}' (value: {value}) into node {node_type} field {field_name}.")
            else:
                logging.warning(f"Could not find node {node_type} with field {field_name} for input '{input_name}'. Skipping injection.")

    def _ensure_dependencies(self):
        """
        Checks for and installs missing custom node dependencies.
        Uses fuzzy matching to verify installation success.
        Handles false positives from misidentified core nodes gracefully.
        """
        required_deps = self.workflow_template.get('custom_node_dependencies', [])
        if not required_deps:
            logging.info("No custom node dependencies specified.")
            return

        logging.info(f"Checking {len(required_deps)} custom node dependencies...")
        
        # Get currently installed nodes
        installed_nodes = self.comfyui_client.get_installed_nodes()
        installed_nodes_set = set(installed_nodes)
        
        logging.debug(f"Currently installed nodes: {len(installed_nodes_set)}")

        repos_to_install = []
        verified_deps = []
        
        for dep in required_deps:
            repo_url = dep.get('url')
            expected_nodes = dep.get('nodes', [])
            
            if not expected_nodes:
                logging.warning(f"Dependency {repo_url} has no node list, skipping")
                continue
            
            # Check if ANY of the expected nodes are installed (exact match)
            found_nodes = [n for n in expected_nodes if n in installed_nodes_set]
            
            if found_nodes:
                logging.info(f"✓ Dependency satisfied: {repo_url}")
                logging.debug(f"  Found nodes: {found_nodes}")
                verified_deps.append(repo_url)
                continue
            
            # Try fuzzy matching - maybe node names are slightly different
            fuzzy_matches = []
            for expected_node in expected_nodes:
                for installed_node in installed_nodes:
                    # Case-insensitive substring matching
                    if (expected_node.lower() in installed_node.lower() or 
                        installed_node.lower() in expected_node.lower()):
                        fuzzy_matches.append((expected_node, installed_node))
            
            if fuzzy_matches:
                logging.info(f"✓ Dependency likely satisfied (fuzzy): {repo_url}")
                logging.debug(f"  Fuzzy matches: {fuzzy_matches}")
                verified_deps.append(repo_url)
                continue
            
            # Not found - needs installation
            logging.warning(f"✗ Dependency missing: {repo_url}")
            logging.debug(f"  Expected nodes: {expected_nodes}")
            repos_to_install.append((repo_url, expected_nodes))

        if not repos_to_install:
            logging.info("All dependencies are satisfied!")
            return

        # Install missing repos
        logging.info(f"Installing {len(repos_to_install)} missing repositories...")
        
        failed_installs = []
        successful_installs = []
        
        for repo_url, expected_nodes in repos_to_install:
            logging.info(f"Installing: {repo_url}")
            success = manager_install_custom_node(repo_url)
            
            if success:
                successful_installs.append(repo_url)
                logging.info(f"✓ Successfully installed {repo_url}")
            else:
                failed_installs.append((repo_url, expected_nodes))
                logging.error(f"✗ Failed to install {repo_url}")

        # Only restart if we installed something successfully
        if successful_installs:
            logging.info("Restarting ComfyUI to load new nodes...")
            if not docker_manager.restart_container():
                raise RuntimeError("Container restart failed after dependency installation")
            
            if not self.comfyui_client.wait_for_ready(self.shutdown_event, timeout=180):
                raise RuntimeError("ComfyUI did not become ready after restart")
            
            # Re-verify installations with improved detection
            logging.info("Verifying newly installed dependencies...")
            installed_nodes = self.comfyui_client.get_installed_nodes()
            installed_nodes_set = set(installed_nodes)
            
            # Log a sample of installed nodes for debugging
            logging.debug(f"Sample of installed nodes after restart: {list(installed_nodes_set)[:20]}")
            
            still_missing = []
            possibly_working = []
            
            for repo_url, expected_nodes in repos_to_install:
                if repo_url in [url for url, _ in failed_installs]:
                    continue  # Already know it failed to install
                
                # Check if ANY expected node is now present (exact match)
                found = any(n in installed_nodes_set for n in expected_nodes)
                
                # Also try fuzzy matching again
                fuzzy_found = False
                if not found:
                    for expected_node in expected_nodes:
                        for installed_node in installed_nodes:
                            if (expected_node.lower() in installed_node.lower() or
                                installed_node.lower() in expected_node.lower()):
                                found = True
                                fuzzy_found = True
                                logging.info(f"  ✓ Found {expected_node} as {installed_node} (fuzzy)")
                                break
                        if found:
                            break
                
                if found:
                    logging.info(f"✓ Verified: {repo_url}")
                else:
                    # Check if the repository directory exists (install succeeded but nodes not detected)
                    # This might be a false positive from workflow_sync misidentifying core nodes
                    still_missing.append(repo_url)
                    logging.warning(f"⚠ Could not verify: {repo_url}")
                    logging.warning(f"  Expected: {expected_nodes}")
                    logging.warning(f"  This may be a false positive (core node misidentified as custom)")
                    possibly_working.append(repo_url)
            
            if still_missing:
                logging.warning("=" * 60)
                logging.warning("DEPENDENCY VERIFICATION WARNING")
                logging.warning("=" * 60)
                logging.warning(f"Installed {len(successful_installs)} repos but could not verify {len(still_missing)} dependencies")
                logging.warning(f"Unverified: {still_missing}")
                logging.warning("")
                logging.warning("This could mean:")
                logging.warning("1. The node names in the registry are incorrect")
                logging.warning("2. The nodes are installed but under different names")
                logging.warning("3. These were core nodes misidentified as custom (false positive)")
                logging.warning("4. The repository installation succeeded but nodes aren't loading")
                logging.warning("")
                logging.warning("Attempting to continue anyway - workflow may still work...")
                logging.warning("=" * 60)
                
                # Don't raise MissingDependenciesError - try to run the workflow
                # It might work despite our detection failing
        
        if failed_installs:
            failed_repos = [url for url, _ in failed_installs]
            logging.error("=" * 60)
            logging.error("INSTALLATION FAILURES")
            logging.error("=" * 60)
            logging.error(f"Failed to install {len(failed_installs)} repositories:")
            for url, nodes in failed_installs:
                logging.error(f"  - {url}")
                logging.error(f"    Expected nodes: {nodes}")
            logging.error("")
            logging.error("These installations genuinely failed (not verification issues).")
            logging.error("The workflow will likely fail without these dependencies.")
            logging.error("=" * 60)
            
            # Only raise if we have actual installation failures
            # Don't raise for verification failures (might be false positives)
            raise MissingDependenciesError(failed_repos)

    def process(self):
        # Ensure all dependencies are met before processing
        self._ensure_dependencies()

        self._inject_dynamic_inputs()
        payload = {"prompt": self.workflow_json}
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        output_path = None
        thumbnail_path = None
        duration = None

        if self.target_entity == 'scene' or self.workflow_type in ['wan-2.2-text-to-video', 'wan-2.2-image-to-video']:
            video_info = find_video_in_output(outputs)
            if not video_info:
                logging.error(f"Workflow for job {self.job_id} completed, but no video file found.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                return

            video_filename, subfolder = video_info
            temp_host_path = self._copy_file_from_container(video_filename, subfolder)
            if not temp_host_path:
                logging.error(f"Failed to copy output file from container for job {self.job_id}.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                return

            try:
                output_path = self.orchestrator_service.upload_output(temp_host_path, self.job_id, 'video/mp4')
                if output_path:
                    thumbnail_local_path = os.path.join(self.cache_dir, f"{self.job_id}_thumb.jpg")
                    if generate_thumbnail(temp_host_path, thumbnail_local_path):
                        thumbnail_path = self.orchestrator_service.upload_thumbnail(thumbnail_local_path, self.job_id)
                        os.remove(thumbnail_local_path)
                    duration = get_video_duration(temp_host_path)
                else:
                    logging.error(f"Video upload failed for job {self.job_id}.")
                    self.orchestrator_service.update_job_status(self.job_id, 'failed')
                    return
            finally:
                logging.info(f"Cleaning up temporary file: {temp_host_path}")
                os.remove(temp_host_path)

        elif self.target_entity == 'audio_clip' or self.workflow_type in ['hunyuan_video_foley', 'vibevoice', 'vibevoice_multi_clone', 'diffrhythm']:
            audio_info = find_audio_in_output(outputs)
            if not audio_info:
                logging.error(f"Workflow for job {self.job_id} completed, but no audio file found.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                return

            audio_filename, subfolder = audio_info
            temp_host_path = self._copy_file_from_container(audio_filename, subfolder)
            if not temp_host_path:
                logging.error(f"Failed to copy output file from container for job {self.job_id}.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                return

            try:
                output_path = self.orchestrator_service.upload_audio_output(temp_host_path, self.job_id)
                if output_path:
                    duration = get_audio_duration(temp_host_path)
                else:
                    logging.error(f"Audio upload failed for job {self.job_id}.")
                    self.orchestrator_service.update_job_status(self.job_id, 'failed')
                    return
            finally:
                logging.info(f"Cleaning up temporary file: {temp_host_path}")
                os.remove(temp_host_path)

        elif self.target_entity == 'character' or self.workflow_type == 'qwen':
            image_info = find_image_in_output(outputs)
            if not image_info:
                logging.error(f"Workflow for job {self.job_id} completed, but no image file found.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                return

            image_filename, subfolder = image_info
            temp_host_path = self._copy_file_from_container(image_filename, subfolder)
            if not temp_host_path:
                logging.error(f"Failed to copy output file from container for job {self.job_id}.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                return

            try:
                output_path = self.orchestrator_service.upload_image_output(temp_host_path, self.job_id)
                if output_path:
                    thumbnail_path = output_path
                else:
                    logging.error(f"Image upload failed for job {self.job_id}.")
                    self.orchestrator_service.update_job_status(self.job_id, 'failed')
                    return
            finally:
                logging.info(f"Cleaning up temporary file: {temp_host_path}")
                os.remove(temp_host_path)

        else:
            logging.warning(f"Job {self.job_id} completed, but no specific output handling for workflow_type: {self.workflow_type} and target_entity: {self.target_entity}. Marking as completed without asset upload.")

        self.orchestrator_service.update_job_status(
            self.job_id, 
            'completed', 
            output_path=output_path, 
            thumbnail_path=thumbnail_path, 
            duration_seconds=duration, 
            completion_metadata=self.job.get('completion_metadata')
        )