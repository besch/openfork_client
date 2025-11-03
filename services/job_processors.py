import os
import logging
import uuid
from typing import Union
from services.docker_manager import docker_manager
from utils.media_utils import get_audio_duration, find_audio_in_output, find_image_in_output, find_video_in_output, generate_thumbnail, get_video_duration
from utils.comfyui_workflow_utils import find_node_by_type_and_field, set_node_property, materialize_image_input
from services.auto_installer import (
    manager_install_custom_node_via_cli,
    fix_all_custom_node_dependencies,
    get_node_name_from_url
)
import time

# UI-only nodes that should be removed before execution
UI_ONLY_NODES = {
    "Note",  # Core UI node
    "MarkdownNote",  # pythongosssss
    "ShowText",  # pythongosssss (sometimes)
    "PreviewBridge",  # Core preview
}

def flatten_litegraph(wf: dict) -> dict:
    nodes = wf.get('nodes', [])
    links = wf.get('links', [])
    definitions = wf.get('definitions', {})
    subgraph_map = {sg['id']: sg for sg in definitions.get('subgraphs', []) if isinstance(sg, dict) and 'id' in sg}

    flat_nodes = []
    flat_links = []
    node_map = {}
    subgraph_instances = {}
    max_node_id = max([n.get('id', 0) for n in nodes] if nodes else [0])
    max_link_id = max([l[0] for l in links] if links else [0])
    id_counter = max(max_node_id, max_link_id) + 1

    for node in nodes:
        if not isinstance(node, dict):
            continue
        old_id = node.get('id')
        if old_id is None:
            continue
        node_type = node.get('type', 'Unknown')
        is_subgraph = False
        sg_def = None
        try:
            uuid.UUID(node_type)
            is_subgraph = True
            sg_def = subgraph_map.get(node_type)
        except (ValueError, AttributeError):
            pass

        if not is_subgraph:
            new_id = old_id
            node_map[old_id] = new_id
            flat_nodes.append(node.copy())
            flat_nodes[-1]['id'] = new_id
        else:
            if not sg_def:
                logging.warning(f"No definition for subgraph {node_type}")
                continue

            internal_wf = {
                'nodes': sg_def.get('nodes', []),
                'links': sg_def.get('links', []),
                'definitions': definitions
            }
            internal_flat = flatten_litegraph(internal_wf)
            internal_nodes = internal_flat['nodes']
            internal_links = internal_flat['links']

            internal_map = {}
            for int_node in internal_nodes:
                old_int_id = int_node['id']
                new_int_id = id_counter
                id_counter += 1
                internal_map[old_int_id] = new_int_id
                int_node['id'] = new_int_id

            link_map = {}
            for link in internal_links:
                old_link_id = link[0]
                new_link_id = id_counter
                id_counter += 1
                link_map[old_link_id] = new_link_id
                link[0] = new_link_id
                link[1] = internal_map[link[1]]
                link[3] = internal_map[link[3]]

            flat_nodes.extend(internal_nodes)
            flat_links.extend(internal_links)

            subgraph_instances[old_id] = {
                'internal_map': internal_map,
                'internal_links': internal_links,
                'link_map': link_map,
                'sg_def': sg_def,
                'node': node
            }

    for link in links:
        if not isinstance(link, list) or len(link) < 4:
            continue
        source_node = link[1]
        target_node = link[3]
        if source_node not in subgraph_instances and target_node not in subgraph_instances:
            new_link = link.copy()
            new_link[1] = node_map.get(source_node, source_node)
            new_link[3] = node_map.get(target_node, target_node)
            flat_links.append(new_link)

    for link in links:
        if not isinstance(link, list) or len(link) < 5:
            continue
        source_node = link[1]
        source_slot = link[2]
        target_node = link[3]
        target_slot = link[4]

        if target_node in subgraph_instances:
            sg_inst = subgraph_instances[target_node]
            sg_def = sg_inst['sg_def']
            sg_inputs = sg_def.get('inputs', [])
            if target_slot >= len(sg_inputs):
                continue
            sg_input = sg_inputs[target_slot]
            internal_link_id = sg_input.get('link')
            if internal_link_id is None:
                continue
            internal_link = next((l for l in sg_def.get('links', []) if l[0] == internal_link_id), None)
            if internal_link:
                internal_target_node_old = internal_link[3]
                internal_target_slot = internal_link[4]
                new_internal_target = sg_inst['internal_map'].get(internal_target_node_old)
                if new_internal_target is None:
                    continue
                new_source = node_map.get(source_node, source_node)
                new_link_id = id_counter
                id_counter += 1
                new_link = [new_link_id, new_source, source_slot, new_internal_target, internal_target_slot]
                if len(link) > 5:
                    new_link.append(link[5])
                flat_links.append(new_link)

        if source_node in subgraph_instances:
            sg_inst = subgraph_instances[source_node]
            sg_def = sg_inst['sg_def']
            sg_outputs = sg_def.get('outputs', [])
            if source_slot >= len(sg_outputs):
                continue
            sg_output = sg_outputs[source_slot]
            internal_link_id = sg_output.get('link')
            if internal_link_id is None:
                continue
            internal_link = next((l for l in sg_def.get('links', []) if l[0] == internal_link_id), None)
            if internal_link:
                internal_source_node_old = internal_link[1]
                internal_source_slot = internal_link[2]
                new_internal_source = sg_inst['internal_map'].get(internal_source_node_old)
                if new_internal_source is None:
                    continue
                new_target = node_map.get(target_node, target_node)
                new_link_id = id_counter
                id_counter += 1
                new_link = [new_link_id, new_internal_source, internal_source_slot, new_target, target_slot]
                if len(link) > 5:
                    new_link.append(link[5])
                flat_links.append(new_link)

    for sg_node_id, sg_inst in subgraph_instances.items():
        node = sg_inst['node']
        widgets = node.get('widgets_values', [])
        sg_inputs = sg_inst['sg_def'].get('inputs', [])
        for i, sg_input in enumerate(sg_inputs):
            input_name = sg_input.get('name')
            if input_name is None:
                continue

            is_connected = False
            if 'inputs' in node and isinstance(node['inputs'], list):
                for node_inp in node['inputs']:
                    if isinstance(node_inp, dict) and node_inp.get('name') == input_name and node_inp.get('link') is not None:
                        is_connected = True
                        break

            if not is_connected and i < len(widgets):
                widget_value = widgets[i]
                internal_link_id = sg_input.get('link')
                if internal_link_id is None:
                    continue

                new_internal_link_id = sg_inst['link_map'].get(internal_link_id)
                if new_internal_link_id:
                    flat_links = [l for l in flat_links if l[0] != new_internal_link_id]

                internal_link = next((l for l in sg_inst['sg_def'].get('links', []) if l[0] == internal_link_id), None)
                if internal_link:
                    internal_target_node_old = internal_link[3]
                    internal_target_slot = internal_link[4]
                    new_internal_target = sg_inst['internal_map'].get(internal_target_node_old)
                    if new_internal_target is None:
                        continue

                    internal_node = next((n for n in flat_nodes if n['id'] == new_internal_target), None)
                    if internal_node is None:
                        continue

                    if 'inputs' in internal_node and len(internal_node['inputs']) > internal_target_slot:
                        internal_node['inputs'][internal_target_slot]['link'] = None

                    current_widgets = internal_node.get('widgets_values', [])

                    current_values = {}
                    k = 0
                    for s, inp in enumerate(internal_node.get('inputs', [])):
                        if inp.get('link') is None:
                            if k < len(current_widgets):
                                current_values[s] = current_widgets[k]
                            k += 1

                    current_values[internal_target_slot] = widget_value

                    new_unconnected = [s for s, inp in enumerate(internal_node.get('inputs', [])) if inp.get('link') is None]

                    new_widgets = [current_values.get(s) for s in new_unconnected if s in current_values]

                    internal_node['widgets_values'] = new_widgets

    return {'nodes': flat_nodes, 'links': flat_links}

def convert_litegraph_to_api(workflow_json: dict) -> dict:
    if 'nodes' not in workflow_json or not isinstance(workflow_json.get('nodes'), list):
        return workflow_json

    logging.info("Converting LiteGraph format to API format...")

    flat = flatten_litegraph(workflow_json)
    nodes_list = flat['nodes']
    links_list = flat['links']

    link_map = {}
    for link in links_list:
        if isinstance(link, list) and len(link) >= 5:
            link_id = link[0]
            source_node_id = str(link[1])
            source_slot = link[2]
            target_node_id = str(link[3])
            target_slot = link[4]
            link_map[str(link_id)] = (source_node_id, source_slot, target_node_id, target_slot)

    api_workflow = {}

    for node in nodes_list:
        if not isinstance(node, dict):
            continue

        node_id = str(node.get('id', ''))
        if not node_id:
            continue

        node_type = node.get('type', 'Unknown')
        widgets = node.get('widgets_values', [])

        api_inputs = {}

        all_input_names = []
        connected_input_names = set()

        if 'inputs' in node and isinstance(node['inputs'], list):
            for inp in node['inputs']:
                if not isinstance(inp, dict):
                    continue

                input_name = inp.get('name')
                if not input_name:
                    continue

                all_input_names.append(input_name)

                link_id = inp.get('link')
                if link_id is not None:
                    link_id_str = str(link_id)
                    if link_id_str in link_map:
                        source_node_id, source_slot, _, _ = link_map[link_id_str]
                        api_inputs[input_name] = [source_node_id, source_slot]
                        connected_input_names.add(input_name)

        unconnected_inputs = [name for name in all_input_names if name not in connected_input_names]

        for i, widget_value in enumerate(widgets):
            if i >= len(unconnected_inputs):
                break
            input_name = unconnected_inputs[i]
            if isinstance(widget_value, dict):
                if 'name' in widget_value:
                    api_inputs[input_name] = widget_value['name']
                else:
                    api_inputs[input_name] = widget_value
            else:
                api_inputs[input_name] = widget_value

        if 'properties' in node and isinstance(node['properties'], dict):
            for prop_name, prop_value in node['properties'].items():
                if prop_name not in api_inputs:
                    api_inputs[prop_name] = prop_value

        api_workflow[node_id] = {
            'class_type': node_type,
            'inputs': api_inputs,
            '_meta': {
                'title': node.get('title', '')
            }
        }

    logging.info(f"Converted {len(api_workflow)} nodes from LiteGraph to API format")
    return api_workflow

def clean_workflow_for_execution(workflow_json: dict) -> dict:
    """
    Removes UI-only nodes from a workflow before execution.
    These nodes are for documentation/display only and don't affect the workflow logic.
    
    IMPORTANT: Only removes nodes that have NO outputs (not connected to anything).
    If a UI node is connected to other nodes, we keep it to avoid breaking the graph.
    
    Also handles format conversion from LiteGraph to API format if needed.
    """
    if not isinstance(workflow_json, dict):
        return workflow_json
    
    # Convert LiteGraph to API format if needed
    if 'nodes' in workflow_json and isinstance(workflow_json.get('nodes'), list):
        workflow_json = convert_litegraph_to_api(workflow_json)
    
    # First, find all nodes that are used as inputs to other nodes
    referenced_nodes = set()
    for node_id, node_data in workflow_json.items():
        if not isinstance(node_data, dict):
            continue
        
        inputs = node_data.get('inputs', {})
        if isinstance(inputs, dict):
            for input_name, input_value in inputs.items():
                # Check if this input references another node
                # Format is usually [node_id, slot_index] or just node_id
                if isinstance(input_value, list) and len(input_value) >= 1:
                    referenced_nodes.add(str(input_value[0]))
                elif isinstance(input_value, str):
                    # Sometimes it's just a string node_id
                    referenced_nodes.add(input_value)
    
    cleaned = {}
    removed_nodes = []
    kept_connected_ui_nodes = []
    
    for node_id, node_data in workflow_json.items():
        if not isinstance(node_data, dict):
            cleaned[node_id] = node_data
            continue
        
        class_type = node_data.get('class_type', '')
        
        # Only remove UI-only nodes if they're NOT referenced by other nodes
        if class_type in UI_ONLY_NODES:
            if str(node_id) in referenced_nodes:
                # Keep it - something else needs it
                cleaned[node_id] = node_data
                kept_connected_ui_nodes.append((node_id, class_type))
            else:
                # Safe to remove - nothing uses it
                removed_nodes.append((node_id, class_type))
            continue
        
        cleaned[node_id] = node_data
    
    if removed_nodes:
        logging.info(f"Removed {len(removed_nodes)} UI-only nodes before execution:")
        for node_id, class_type in removed_nodes:
            logging.debug(f"  - Node {node_id}: {class_type}")
    
    if kept_connected_ui_nodes:
        logging.debug(f"Kept {len(kept_connected_ui_nodes)} UI nodes that are connected to the workflow:")
        for node_id, class_type in kept_connected_ui_nodes:
            logging.debug(f"  - Node {node_id}: {class_type} (has connections)")
    
    return cleaned

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

    def _clean_ui_nodes(self):
        """Remove UI-only nodes before execution."""
        self.workflow_json = clean_workflow_for_execution(self.workflow_json)

    def _inject_dynamic_inputs(self):
        if not self.input_schema or not self.input_schema.get('properties'):
            logging.info(f"No input schema defined for workflow {self.workflow_template.get('name')}. Skipping dynamic injection.")
            return

        injection_count = 0
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

            if input_definition.get('type') == 'image':
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
                injection_count += 1
            else:
                logging.debug(f"Could not find node {node_type} with field {field_name} for input '{input_name}'. This may be normal if the workflow doesn't use this input.")
        
        if injection_count > 0:
            logging.info(f"Successfully injected {injection_count} dynamic inputs into workflow.")

    def _ensure_dependencies(self):
        """
        IMPROVED: Checks for and installs missing custom node dependencies using cm-cli.
        This is more reliable than manual git cloning.
        """
        required_deps = self.workflow_template.get('custom_node_dependencies', [])
        if not required_deps:
            logging.info("No custom node dependencies specified.")
            return

        logging.info(f"Checking {len(required_deps)} custom node dependencies...")
        
        # Get currently installed nodes
        installed_nodes = self.comfyui_client.get_installed_nodes()
        installed_nodes_set = set(installed_nodes)
        
        logging.debug(f"Currently {len(installed_nodes_set)} nodes installed")

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
                logging.info(f"[OK] Dependency satisfied: {repo_url}")
                logging.debug(f"  Found nodes: {found_nodes}")
                verified_deps.append(repo_url)
                continue
            
            # Try fuzzy matching
            fuzzy_matches = []
            for expected_node in expected_nodes:
                for installed_node in installed_nodes:
                    if (expected_node.lower() in installed_node.lower() or 
                        installed_node.lower() in expected_node.lower()):
                        fuzzy_matches.append((expected_node, installed_node))
            
            if fuzzy_matches:
                logging.info(f"[OK] Dependency likely satisfied (fuzzy): {repo_url}")
                logging.debug(f"  Fuzzy matches: {fuzzy_matches}")
                verified_deps.append(repo_url)
                continue
            
            # Not found - needs installation
            logging.warning(f"[MISSING] Dependency missing: {repo_url}")
            logging.debug(f"  Expected nodes: {expected_nodes}")
            repos_to_install.append((repo_url, expected_nodes))

        if not repos_to_install:
            logging.info("All dependencies are satisfied!")
            return

        # Install missing repos using cm-cli
        logging.info(f"Installing {len(repos_to_install)} missing repositories using cm-cli...")
        
        failed_installs = []
        successful_installs = []
        
        for repo_url, expected_nodes in repos_to_install:
            logging.info(f"Installing: {repo_url}")
            
            # Use the improved cm-cli based installer
            success = manager_install_custom_node_via_cli(repo_url)
            
            if success:
                successful_installs.append(repo_url)
                logging.info(f"[OK] Successfully installed {repo_url}")
            else:
                failed_installs.append((repo_url, expected_nodes))
                logging.error(f"[ERROR] Failed to install {repo_url}")

        # Only restart if we installed something successfully
        if successful_installs:
            logging.info("=" * 60)
            logging.info(f"Installed {len(successful_installs)} new custom nodes")
            logging.info("Restarting ComfyUI to load new nodes...")
            logging.info("=" * 60)
            
            if not docker_manager.restart_container():
                raise RuntimeError("Container restart failed after dependency installation")
            
            # Wait for ComfyUI to be ready
            if not self.comfyui_client.wait_for_ready(self.shutdown_event, timeout=180):
                raise RuntimeError("ComfyUI did not become ready after restart")
            
            # CRITICAL: Run cm-cli fix to ensure all dependencies are installed
            logging.info("Running cm-cli fix to install all node dependencies...")
            fix_all_custom_node_dependencies()
            
            # Give nodes time to initialize
            logging.info("Waiting for nodes to initialize...")
            time.sleep(10)
            
            # Trigger explicit node refresh
            logging.info("Refreshing node cache...")
            self.comfyui_client.refresh_nodes()
            docker_manager.invalidate_node_cache()
            
            # Wait a bit more after refresh
            time.sleep(5)
            
            # Re-verify installations
            logging.info("Verifying newly installed dependencies...")
            installed_nodes = self.comfyui_client.get_installed_nodes()
            installed_nodes_set = set(installed_nodes)
            
            still_missing = []
            
            for repo_url, expected_nodes in repos_to_install:
                if repo_url in [url for url, _ in failed_installs]:
                    continue
                
                repo_name = get_node_name_from_url(repo_url)
                
                # Check if ANY expected node is now present
                found = any(n in installed_nodes_set for n in expected_nodes)
                
                # Also try fuzzy matching
                if not found:
                    for expected_node in expected_nodes:
                        for installed_node in installed_nodes:
                            if (expected_node.lower() in installed_node.lower() or
                                installed_node.lower() in expected_node.lower()):
                                found = True
                                logging.info(f"  [FUZZY] Found {expected_node} as {installed_node}")
                                break
                        if found:
                            break
                
                if found:
                    logging.info(f"[OK] Verified: {repo_url}")
                else:
                    still_missing.append(repo_url)
                    logging.warning(f"[WARNING] Could not verify: {repo_url}")
                    logging.warning(f"  Expected: {expected_nodes}")
            
            if still_missing:
                logging.warning("=" * 60)
                logging.warning("DEPENDENCY VERIFICATION WARNING")
                logging.warning("=" * 60)
                logging.warning(f"Could not verify {len(still_missing)} dependencies")
                logging.warning(f"Unverified: {still_missing}")
                logging.warning("")
                logging.warning("This could mean:")
                logging.warning("1. Node names in registry are incorrect")
                logging.warning("2. Nodes installed under different names")
                logging.warning("3. Core nodes misidentified as custom (false positive)")
                logging.warning("4. Repository installed but nodes aren't loading")
                logging.warning("")
                logging.warning("Attempting to continue anyway - workflow may still work...")
                logging.warning("=" * 60)
        
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
            logging.error("The workflow will likely fail without these dependencies.")
            logging.error("=" * 60)
            
            raise MissingDependenciesError(failed_repos)

    def _validate_workflow(self):
        """
        IMPROVED: Validates the workflow before execution with better error messages.
        """
        issues = []
        warnings = []
        
        # Check if workflow is empty
        if not self.workflow_json:
            issues.append("Workflow is empty")
            return False
        
        for node_id, node_data in self.workflow_json.items():
            if not isinstance(node_data, dict):
                warnings.append(f"Node {node_id} is not a dictionary")
                continue
            
            class_type = node_data.get('class_type', '')
            inputs = node_data.get('inputs', {})
            
            if not class_type:
                issues.append(f"Node {node_id} is missing class_type")
                continue
            
            # Check for nodes with missing required inputs
            if not isinstance(inputs, dict):
                warnings.append(f"Node {node_id} ({class_type}) has invalid inputs")
                continue
            
            # Special checks for critical nodes
            if class_type == 'SaveImage':
                if 'images' not in inputs or inputs.get('images') is None:
                    issues.append(f"Node {node_id} (SaveImage) is missing required 'images' input")
            
            # Check for broken connections
            for input_name, input_value in inputs.items():
                if isinstance(input_value, list) and len(input_value) >= 1:
                    referenced_node = str(input_value[0])
                    if referenced_node not in self.workflow_json:
                        issues.append(
                            f"Node {node_id} ({class_type}) input '{input_name}' "
                            f"references non-existent node {referenced_node}"
                        )
        
        if issues:
            logging.error("=" * 60)
            logging.error("WORKFLOW VALIDATION FAILED")
            logging.error("=" * 60)
            for issue in issues:
                logging.error(f"  [ERROR] {issue}")
            logging.error("=" * 60)
            return False
        
        if warnings:
            logging.warning("Workflow validation warnings:")
            for warning in warnings:
                logging.warning(f"  [WARNING] {warning}")
        
        logging.info("Workflow validation passed")
        return True

    def process(self):
        # Ensure all dependencies are met before processing
        self._ensure_dependencies()

        # Clean UI-only nodes before processing
        self._clean_ui_nodes()
        
        # Validate workflow structure
        if not self._validate_workflow():
            logging.error("Workflow validation failed. Aborting job.")
            self.orchestrator_service.update_job_status(self.job_id, 'failed')
            return

        self._inject_dynamic_inputs()
        
        # Log the workflow for debugging
        logging.debug(f"Final workflow for job {self.job_id}:")
        logging.debug(f"Number of nodes: {len(self.workflow_json)}")
        logging.debug(f"Node types: {[n.get('class_type', 'UNKNOWN') for n in self.workflow_json.values()]}")
        
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