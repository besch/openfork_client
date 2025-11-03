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

# Import logger for use in job processor
logger = logging.getLogger(__name__)

# UI-only nodes that should be removed before execution
UI_ONLY_NODES = {
    "Note",  # Core UI node
    "MarkdownNote",  # pythongosssss
    "ShowText",  # pythongosssss (sometimes)
    "PreviewBridge",  # Core preview
}

def track_node_references(workflow_json: dict) -> dict:
    """
    Build a mapping of which nodes reference which other nodes.
    Returns a dict: {node_id: [list_of_nodes_that_reference_this_node]}
    """
    references = {}
    for node_id in workflow_json.keys():
        references[node_id] = []
    
    for node_id, node_data in workflow_json.items():
        if not isinstance(node_data, dict):
            continue
        
        inputs = node_data.get('inputs', {})
        if not isinstance(inputs, dict):
            continue
        
        for input_name, input_value in inputs.items():
            # Check if this is a node reference [node_id, slot]
            if isinstance(input_value, list) and len(input_value) >= 2:
                ref_node_id = str(input_value[0])
                if ref_node_id in references:
                    references[ref_node_id].append(node_id)
    
    return references

def normalize_node_id(node_id) -> str:
    """Normalize node ID by removing # prefix if present."""
    node_id_str = str(node_id)
    return node_id_str.lstrip('#')

def fix_node_id_references(workflow_json: dict) -> dict:
    """
    CRITICAL FIX: Normalize all node IDs and references to remove # prefixes.
    This fixes workflows that were incorrectly converted from LiteGraph format.
    
    Changes:
    1. Renames all node IDs (keys) to remove # prefix
    2. Updates all references in inputs to use normalized IDs
    """
    if not isinstance(workflow_json, dict):
        return workflow_json
    
    logging.info("Fixing node ID references...")
    
    # Step 1: Build mapping of old IDs to normalized IDs
    id_mapping = {}
    for old_id in workflow_json.keys():
        normalized_id = normalize_node_id(old_id)
        if old_id != normalized_id:
            id_mapping[old_id] = normalized_id
            logging.debug(f"Will rename node: {old_id} -> {normalized_id}")
    
    # Step 2: Create new workflow with normalized node IDs
    fixed_workflow = {}
    
    for old_id, node_data in workflow_json.items():
        if not isinstance(node_data, dict):
            continue
        
        # Use normalized ID as the key
        new_id = normalize_node_id(old_id)
        node_copy = node_data.copy()
        
        # Step 3: Fix all references in inputs
        inputs = node_copy.get('inputs', {})
        if isinstance(inputs, dict):
            fixed_inputs = {}
            
            for input_name, input_value in inputs.items():
                # Check if this is a node reference [node_id, slot]
                if isinstance(input_value, list) and len(input_value) >= 1:
                    ref_node_id = input_value[0]
                    normalized_ref = normalize_node_id(ref_node_id)
                    
                    # Create fixed reference
                    fixed_value = [normalized_ref] + input_value[1:]
                    fixed_inputs[input_name] = fixed_value
                    
                    if str(ref_node_id) != normalized_ref:
                        logging.debug(
                            f"Fixed reference in {new_id}.{input_name}: {ref_node_id} -> {normalized_ref}"
                        )
                else:
                    # Not a reference, keep as-is
                    fixed_inputs[input_name] = input_value
            
            node_copy['inputs'] = fixed_inputs
        
        fixed_workflow[new_id] = node_copy
    
    if id_mapping:
        logging.info(f"Fixed {len(id_mapping)} node IDs with # prefix")
    
    return fixed_workflow

def clean_workflow_references(workflow_json: dict) -> dict:
    """
    Clean up node references in API format workflow to ensure consistency.
    Removes # prefixes and ensures all references use the same format.
    """
    if not isinstance(workflow_json, dict):
        return workflow_json
    
    # First pass: normalize all node IDs (keys)
    normalized_workflow = {}
    old_to_new_id = {}
    
    for node_id, node_data in workflow_json.items():
        normalized_id = normalize_node_id(node_id)
        old_to_new_id[node_id] = normalized_id
        old_to_new_id[normalize_node_id(node_id)] = normalized_id  # Handle both forms
        normalized_workflow[normalized_id] = node_data
    
    # Second pass: normalize all references in inputs
    for node_id, node_data in normalized_workflow.items():
        if not isinstance(node_data, dict):
            continue
        
        inputs = node_data.get('inputs', {})
        if not isinstance(inputs, dict):
            continue
        
        for input_name, input_value in inputs.items():
            # Check if this is a node reference [node_id, slot]
            if isinstance(input_value, list) and len(input_value) >= 1:
                ref_node_id = input_value[0]
                normalized_ref = normalize_node_id(ref_node_id)
                
                # Update the reference to use normalized ID
                if normalized_ref != ref_node_id:
                    input_value[0] = normalized_ref
                    logging.debug(f"Normalized reference: {ref_node_id} -> {normalized_ref}")
    
    return normalized_workflow

# Replace the validate_workflow_structure function in job_processors.py with this:

def validate_workflow_structure(workflow_json: dict) -> tuple[bool, list[str]]:
    """
    Validates workflow structure and returns (is_valid, list_of_errors).
    
    CRITICAL FIX: Handle incomplete workflows (only SaveImage node) gracefully.
    - A node reference is a list like [node_id, slot_index] where slot_index is an integer
    - Everything else (strings, numbers, dicts, bools, other lists) is a VALUE
    """
    if not isinstance(workflow_json, dict) or not workflow_json:
        return False, ["Workflow is empty or not a dict"]
    
    errors = []
    warnings = []
    
    # Build set of valid node IDs (all as strings)
    valid_node_ids = set(str(node_id) for node_id in workflow_json.keys())
    
    # CRITICAL: Check for incomplete workflow (only SaveImage node)
    if len(workflow_json) == 1:
        only_node_id = list(workflow_json.keys())[0]
        only_node_data = workflow_json[only_node_id]
        if isinstance(only_node_data, dict):
            class_type = only_node_data.get('class_type', '')
            if class_type == 'SaveImage':
                warnings.append(
                    f"Incomplete workflow detected: Only SaveImage node {only_node_id} found with no source nodes. "
                    f"Workflow will likely fail execution as SaveImage requires an 'images' input from upstream nodes."
                )
                # Don't fail validation for this case - it will be handled by the broken reference fixer
                # and we want to let execution attempt proceed (it will fail gracefully)
    
    for node_id, node_data in workflow_json.items():
        if not isinstance(node_data, dict):
            errors.append(f"Node {node_id} is not a dictionary")
            continue
        
        class_type = node_data.get('class_type', '')
        if not class_type:
            errors.append(f"Node {node_id} is missing class_type")
            continue
        
        # Skip UI-only nodes in validation (they should be removed but check anyway)
        if class_type in UI_ONLY_NODES:
            warnings.append(f"Found UI-only node {node_id} ({class_type}) - should have been removed")
            continue
        
        inputs = node_data.get('inputs', {})
        if not isinstance(inputs, dict):
            warnings.append(f"Node {node_id} ({class_type}) has invalid inputs")
            continue
        
        # Check for broken node references
        for input_name, input_value in inputs.items():
            # CRITICAL: A node reference in ComfyUI API format is ALWAYS: [node_id, slot_index]
            # where node_id is a string/int and slot_index is an integer
            # Examples:
            #   ["123", 0] - valid reference to node 123, slot 0
            #   [123, 0] - valid reference to node 123, slot 0
            #   "model.safetensors" - NOT a reference, it's a string value
            #   {"name": "model.safetensors"} - NOT a reference, it's a dict value
            #   [1, 2, 3] - NOT a reference (slot_index would be 2, which is valid, but this is a list of values)
            
            if isinstance(input_value, list) and len(input_value) >= 2:
                potential_node_id = str(input_value[0])
                potential_slot = input_value[1]
                
                # Validate: second element MUST be an integer for this to be a reference
                if isinstance(potential_slot, int):
                    # This is a proper node reference format: [node_id, slot_index]
                    # Verify the referenced node exists
                    if potential_node_id not in valid_node_ids:
                        # Special handling for SaveImage nodes - don't fail validation for broken images references
                        # as these will be handled by fix_broken_node_references
                        if class_type == 'SaveImage' and input_name == 'images':
                            warnings.append(
                                f"Node {node_id} ({class_type}) input '{input_name}' references non-existent node '{potential_node_id}' "
                                f"- this will be handled by broken reference fixer"
                            )
                        else:
                            errors.append(
                                f"Node {node_id} ({class_type}) input '{input_name}' "
                                f"references non-existent node '{potential_node_id}'"
                            )
                # else: second element is not an integer, so this is just a list of values
            
            # For all other types (strings, numbers, dicts, bools, single-element lists):
            # These are VALUES, not node references. No validation needed.
            # Examples:
            #   "qwen_image_fp8.safetensors" - model filename (string value)
            #   512 - width parameter (number value)
            #   {"name": "model.safetensors", "url": "..."} - model config (dict value)
            #   True - boolean parameter
            #   ["option1", "option2"] - list of options (list value, not a reference)
    
    if warnings:
        for warning in warnings:
            logging.warning(f"[VALIDATION WARNING] {warning}")
    
    return len(errors) == 0, errors

def remove_ui_only_nodes(workflow_json: dict) -> dict:
    """
    Remove UI-only nodes from API format workflow.
    Also removes any connections to/from these nodes and updates references.
    CRITICAL FIX: Properly handle node reference updates when UI-only nodes are removed.
    """
    if not isinstance(workflow_json, dict):
        return workflow_json
    
    # Identify UI-only node IDs
    ui_node_ids = set()
    for node_id, node_data in workflow_json.items():
        if isinstance(node_data, dict):
            class_type = node_data.get('class_type', '')
            if class_type in UI_ONLY_NODES:
                ui_node_ids.add(node_id)
                logging.debug(f"Marking UI-only node for removal: {node_id} ({class_type})")
    
    if not ui_node_ids:
        return workflow_json
    
    logging.info(f"Removing {len(ui_node_ids)} UI-only nodes: {ui_node_ids}")
    
    # CRITICAL: Build reference mapping BEFORE removing nodes
    node_references = track_node_references(workflow_json)
    
    # Create new workflow without UI-only nodes
    cleaned_workflow = {}
    connections_removed = 0
    
    for node_id, node_data in workflow_json.items():
        if node_id in ui_node_ids:
            continue
        
        # Keep node, but clean its inputs
        node_copy = node_data.copy()
        inputs = node_copy.get('inputs', {})
        
        if isinstance(inputs, dict):
            cleaned_inputs = {}
            for input_name, input_value in inputs.items():
                # Check if this is a reference to a UI-only node
                if isinstance(input_value, list) and len(input_value) >= 1:
                    ref_node_id = str(input_value[0])
                    if ref_node_id in ui_node_ids:
                        logging.debug(
                            f"Removing connection from {node_id}.{input_name} to UI-only node {ref_node_id}"
                        )
                        connections_removed += 1
                        continue  # Skip this connection
                
                cleaned_inputs[input_name] = input_value
            
            node_copy['inputs'] = cleaned_inputs
        
        cleaned_workflow[node_id] = node_copy
    
    if connections_removed > 0:
        logging.info(f"Removed {connections_removed} broken connections to UI-only nodes")
    
    return cleaned_workflow

def normalize_model_references(workflow_json: dict) -> dict:
    """
    CRITICAL FIX: Clean up workflow for execution.
    
    1. Convert model dictionaries to simple filename strings
    2. Remove UI-only metadata fields like widget_ue_connectable
    
    ComfyUI expects model inputs as strings (filenames), not dictionaries.
    """
    if not isinstance(workflow_json, dict):
        return workflow_json
    
    logging.info("Normalizing workflow for execution...")
    model_fixes = 0
    metadata_removals = 0
    
    # UI-only input fields that should be removed
    UI_METADATA_FIELDS = {
        'widget_ue_connectable',  # Unreal Engine integration metadata
        'widget_control_after_generate',  # UI control metadata
        'widget_control_filter_list',  # UI control metadata
        '_meta',  # Sometimes incorrectly placed in inputs
    }
    
    for node_id, node_data in workflow_json.items():
        if not isinstance(node_data, dict):
            continue
        
        inputs = node_data.get('inputs', {})
        if not isinstance(inputs, dict):
            continue
        
        class_type = node_data.get('class_type', '')
        
        # Remove UI-only metadata fields
        for ui_field in UI_METADATA_FIELDS:
            if ui_field in inputs:
                del inputs[ui_field]
                metadata_removals += 1
                logging.debug(f"Removed UI metadata {ui_field} from {node_id} ({class_type})")
        
        # Common model loader nodes and their model input names
        MODEL_INPUTS = {
            'CheckpointLoaderSimple': ['ckpt_name'],
            'CheckpointLoader': ['ckpt_name'],
            'UNETLoader': ['unet_name'],
            'CLIPLoader': ['clip_name'],
            'DualCLIPLoader': ['clip_name1', 'clip_name2'],
            'TripleCLIPLoader': ['clip_name1', 'clip_name2', 'clip_name3'],
            'VAELoader': ['vae_name'],
            'LoraLoader': ['lora_name'],
            'LoraLoaderModelOnly': ['lora_name'],
            'ControlNetLoader': ['control_net_name'],
            'UpscaleModelLoader': ['model_name'],
            'StyleModelLoader': ['style_model_name'],
        }
        
        # Check ALL remaining inputs for dict values (model references)
        for input_name, value in list(inputs.items()):
            # Skip node references (they're lists like ["123", 0])
            if isinstance(value, list) and len(value) >= 2 and isinstance(value[1], int):
                continue
            
            # Check if this is a model dictionary
            if isinstance(value, dict) and 'name' in value:
                # Extract just the filename
                model_filename = value['name']
                inputs[input_name] = model_filename
                model_fixes += 1
                logging.debug(
                    f"Fixed model reference in {node_id} ({class_type}).{input_name}: "
                    f"{value} -> {model_filename}"
                )
    
    if model_fixes > 0:
        logging.info(f"Fixed {model_fixes} model references")
    if metadata_removals > 0:
        logging.info(f"Removed {metadata_removals} UI metadata fields")
    if model_fixes == 0 and metadata_removals == 0:
        logging.debug("Workflow already clean")
    
    return workflow_json

def find_image_source_node(workflow_json: dict) -> str:
    """
    Find a node that produces images that can be connected to SaveImage.
    Returns the node ID of the first image-producing node found.
    CRITICAL FIX: Be more specific about image sources and handle edge cases.
    """
    # List of node types that produce images (excluding SaveImage itself)
    image_producing_nodes = {
        'VAEDecode', 'PreviewImage', 'ImageUpscaleWithModel',
        'CLIPVisionEncode', 'MaskToImage', 'LatentComposite', 'LatentBlend',
        'ImageCompositeMasked', 'ImageBlend', 'ImageInvert', 'ImageQuantize',
        'ImageSharpen', 'ImageBlur', 'Canny', 'ImageColorToMask',
        # Add more image-producing nodes
        'KSampler', 'KSamplerAdvanced', 'BasicScheduler', 'CFGGuider',
        'ConditioningAverage', 'ConditioningCombine', 'ConditioningConcat',
        'ImageScale', 'ImageScaleBy', 'ImageScaleToTotalPixels',
        'ImageCrop', 'ImageBatch', 'ImageFromBatch', 'ImageFromBatch',
        'ImageBlur', 'ImageQuantize', 'ImageSharpen', 'ImageCrop'
    }
    
    # First pass: look for specific image-producing nodes (excluding SaveImage)
    for node_id, node_data in workflow_json.items():
        if isinstance(node_data, dict):
            class_type = node_data.get('class_type', '')
            if class_type in image_producing_nodes:
                logging.debug(f"Found image-producing node: {node_id} ({class_type})")
                return str(node_id)
    
    # Second pass: look for any node that isn't UI-only or SaveImage itself
    for node_id, node_data in workflow_json.items():
        if isinstance(node_data, dict):
            class_type = node_data.get('class_type', '')
            # Skip SaveImage itself and UI-only nodes
            if class_type not in UI_ONLY_NODES and class_type != 'SaveImage':
                logging.debug(f"Fallback: using node {node_id} ({class_type}) as image source")
                return str(node_id)
    
    logging.warning("No suitable image source node found in workflow")
    return None

def fix_broken_node_references(workflow_json: dict) -> dict:
    """
    Fix broken node references instead of removing them completely.
    This handles cases where nodes reference non-existent nodes.
    CRITICAL FIX: Handle cases where workflow is incomplete (only SaveImage node).
    ENHANCED: More aggressive handling of broken references.
    """
    if not isinstance(workflow_json, dict):
        return workflow_json
    
    valid_node_ids = set(str(node_id) for node_id in workflow_json.keys())
    broken_references_fixed = 0
    broken_references_removed = 0
    
    # CRITICAL: Check if we have an incomplete workflow (only SaveImage node)
    if len(workflow_json) == 1:
        only_node_id = list(workflow_json.keys())[0]
        only_node_data = workflow_json[only_node_id]
        if isinstance(only_node_data, dict):
            class_type = only_node_data.get('class_type', '')
            if class_type == 'SaveImage':
                logging.error("=" * 60)
                logging.error("CRITICAL: Incomplete workflow detected!")
                logging.error(f"Workflow contains only a SaveImage node ({only_node_id}) with no source nodes")
                logging.error("This indicates the workflow template is corrupted or incomplete")
                logging.error("=" * 60)
                
                # Check if SaveImage has inputs (likely from the conversion process)
                inputs = only_node_data.get('inputs', {})
                if 'images' in inputs and isinstance(inputs['images'], list):
                    logging.warning(f"Removing broken 'images' reference from incomplete SaveImage node")
                    del inputs['images']
                    broken_references_removed += 1
                
                return workflow_json
    
    # ENHANCED: First pass - identify all broken references
    broken_refs = []
    for node_id, node_data in workflow_json.items():
        if not isinstance(node_data, dict):
            continue
        
        inputs = node_data.get('inputs', {})
        if not isinstance(inputs, dict):
            continue
        
        for input_name, input_value in inputs.items():
            if isinstance(input_value, list) and len(input_value) >= 2:
                potential_node_id = str(input_value[0])
                potential_slot = input_value[1]
                
                if potential_node_id not in valid_node_ids:
                    broken_refs.append((node_id, input_name, input_value, node_data.get('class_type', 'UNKNOWN')))
    
    # ENHANCED: Process broken references with better logic
    for node_id, input_name, input_value, class_type in broken_refs:
        logging.warning(f"Found broken reference in {node_id} ({class_type}).{input_name}: [{input_value[0]}, {input_value[1]}]")
        
        # For SaveImage nodes with broken images references
        if class_type == 'SaveImage' and input_name == 'images':
            # Find a proper image source node to connect to
            image_source_id = find_image_source_node(workflow_json)
            if image_source_id:
                # Fix the reference by connecting to the found image source
                inputs = workflow_json[node_id]['inputs']
                inputs[input_name] = [image_source_id, 0]
                broken_references_fixed += 1
                logging.info(f"Fixed broken 'images' reference in SaveImage node {node_id}: connected to node {image_source_id}")
            else:
                # For incomplete workflows, remove the broken reference
                inputs = workflow_json[node_id]['inputs']
                if input_name in inputs:
                    del inputs[input_name]
                    broken_references_removed += 1
                    logging.warning(f"Removed broken 'images' reference from SaveImage node {node_id} - no image source found")
        else:
            # For other cases, remove the broken reference
            inputs = workflow_json[node_id]['inputs']
            if input_name in inputs:
                del inputs[input_name]
                broken_references_removed += 1
                logging.info(f"Removed broken reference in {node_id} ({class_type}).{input_name}")
    
    if broken_references_fixed > 0:
        logging.info(f"Fixed {broken_references_fixed} broken node references")
    if broken_references_removed > 0:
        logging.info(f"Removed {broken_references_removed} broken node references")
    
    return workflow_json

def clean_workflow_for_execution(workflow_json: dict) -> dict:
    """
    Prepares workflow for execution:
    0. Fixes node ID references (removes # prefixes) - CRITICAL FIX!
    1. Normalizes node IDs (removes # prefixes)
    2. Normalizes model references (dict -> string) - NEW FIX!
    3. Removes UI-only nodes
    4. Validates structure (first pass to catch issues)
    5. Fix broken node references
    6. Validates structure (second pass to ensure fixes worked)
    """
    if not isinstance(workflow_json, dict):
        return workflow_json
    
    logging.info("Cleaning workflow for execution...")
    
    # Step 0: Fix node ID references FIRST (CRITICAL!)
    workflow_json = fix_node_id_references(workflow_json)
    logging.debug(f"Step 0: Fixed node ID references")
    
    # Step 1: Normalize node IDs and references
    workflow_json = clean_workflow_references(workflow_json)
    logging.debug(f"Step 1: Normalized {len(workflow_json)} node IDs")
    
    # Step 2: Normalize model references (NEW!)
    workflow_json = normalize_model_references(workflow_json)
    logging.debug(f"Step 2: Normalized model references")
    
    # Step 3: Remove UI-only nodes
    original_count = len(workflow_json)
    workflow_json = remove_ui_only_nodes(workflow_json)
    removed_count = original_count - len(workflow_json)
    if removed_count > 0:
        logging.info(f"Step 3: Removed {removed_count} UI-only nodes")
    
    # Step 4: CRITICAL FIX - Fix broken node references immediately after UI removal
    workflow_json = fix_broken_node_references(workflow_json)
    logging.debug(f"Step 4: Fixed broken node references after UI removal")
    
    # Step 5: Validation pass to ensure workflow is clean
    is_valid, errors = validate_workflow_structure(workflow_json)
    if not is_valid:
        logging.warning(f"Validation found {len(errors)} issues - attempting final fixes")
        for i, error in enumerate(errors[:5], 1):  # Log first 5 errors
            logging.warning(f"  Issue {i}: {error}")
        if len(errors) > 5:
            logging.warning(f"  ... and {len(errors) - 5} more issues")
        
        # Step 5.5: Emergency broken reference fix attempt
        workflow_json = fix_broken_node_references(workflow_json)
        
        # Re-validate after emergency fix
        is_valid, errors = validate_workflow_structure(workflow_json)
    
    # Step 6: Final validation - if still not valid, fail
    if not is_valid:
        logging.error("=" * 60)
        logging.error("WORKFLOW VALIDATION FAILED")
        logging.error("=" * 60)
        logging.error(f"Found {len(errors)} validation errors after all fixes:")
        for i, error in enumerate(errors, 1):
            logging.error(f"  [{i}] {error}")
        logging.error("=" * 60)
        logging.error(f"Workflow has {len(workflow_json)} nodes:")
        for node_id, node_data in workflow_json.items():
            class_type = node_data.get('class_type', 'UNKNOWN') if isinstance(node_data, dict) else 'INVALID'
            logging.error(f"  - Node {node_id}: {class_type}")
            
            # Log inputs for each node
            if isinstance(node_data, dict):
                inputs = node_data.get('inputs', {})
                if isinstance(inputs, dict):
                    for input_name, input_value in inputs.items():
                        value_str = str(input_value)
                        if len(value_str) > 50:
                            value_str = value_str[:50] + "..."
                        logging.error(f"      {input_name}: {value_str}")
        logging.error("=" * 60)
        raise ValueError(f"Workflow validation failed with {len(errors)} errors")
    
    logging.info(f"Workflow ready for execution: {len(workflow_json)} nodes")
    return workflow_json

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
        self.workflow_format = self.workflow_template.get('workflow_format', 'api')  # NEW FIELD!

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
        """Clean and prepare workflow for execution."""
        self.workflow_json = clean_workflow_for_execution(self.workflow_json)
        
        # CRITICAL: Check for incomplete workflows after cleaning
        if len(self.workflow_json) <= 1:
            only_node_id = list(self.workflow_json.keys())[0] if self.workflow_json else "unknown"
            only_node_data = self.workflow_json.get(only_node_id, {})
            class_type = only_node_data.get('class_type', 'UNKNOWN') if isinstance(only_node_data, dict) else 'INVALID'
            
            if class_type == 'SaveImage':
                logging.error("=" * 60)
                logging.error("CRITICAL WORKFLOW ERROR:")
                logging.error(f"Workflow template '{self.workflow_template.get('name', 'unknown')}' appears to be incomplete")
                logging.error(f"After cleaning, only {len(self.workflow_json)} node(s) remain: {class_type}")
                logging.error("This workflow template likely failed to convert properly or is corrupted")
                logging.error(f"Expected: A complete workflow with image generation/saving nodes")
                logging.error(f"Got: Only a SaveImage node with no source nodes")
                logging.error("=" * 60)
                
                # Update job status to failed with clear error message
                error_msg = (
                    f"Workflow template is incomplete or corrupted. "
                    f"Expected a complete workflow with image generation nodes, but only found '{class_type}' node. "
                    f"This may indicate conversion failure or corrupted workflow template."
                )
                
                self.orchestrator_service.update_job_status(self.job_id, 'failed', completion_metadata={
                    'error_message': error_msg,
                    'workflow_template_name': self.workflow_template.get('name'),
                    'node_count': len(self.workflow_json),
                    'nodes_found': [f"{node_id}: {data.get('class_type', 'INVALID')}" for node_id, data in self.workflow_json.items()]
                })
                raise ValueError(f"Workflow validation failed: {error_msg}")

    def _inject_dynamic_inputs(self):
        if not self.input_schema or not self.input_schema.get('properties'):
            logging.info(f"No input schema defined for workflow {self.workflow_template.get('name')}. Skipping dynamic injection.")
            return

        injection_count = 0
        new_nodes_created = []  # Track LoadImage nodes for potential cleanup
        
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
                # CRITICAL FIX: Handle image inputs properly by creating LoadImage nodes
                materialized_path = materialize_image_input(value, self.input_dir)
                if not materialized_path:
                    logging.error(f"Failed to materialize image for input '{input_name}'. Skipping injection.")
                    continue
                
                # Get filename without path for ComfyUI
                image_filename = os.path.basename(materialized_path)
                
                # Find target node and check if it expects an image
                target_node = find_node_by_type_and_field(self.workflow_json, node_type, field_name)
                if not target_node:
                    logging.debug(f"Could not find node {node_type} with field {field_name} for input '{input_name}'. This may be normal if the workflow doesn't use this input.")
                    continue
                
                # Create a unique ID for the new LoadImage node
                image_node_id = f"load_image_{len(new_nodes_created)}"
                
                # Create the LoadImage node
                load_image_node = {
                    "class_type": "LoadImage",
                    "inputs": {
                        "image": image_filename
                    },
                    "_meta": {
                        "title": f"Load Image for {input_name}"
                    }
                }
                
                # Add the LoadImage node to the workflow
                self.workflow_json[image_node_id] = load_image_node
                new_nodes_created.append(image_node_id)
                
                # Set the target node to reference the new LoadImage node
                # Images typically connect to input slot 0, but let's check the node type
                input_slot = 0  # Default for most image inputs
                set_node_property(target_node, field_name, [image_node_id, input_slot])
                
                logging.info(f"Created LoadImage node '{image_node_id}' for input '{input_name}' and connected to {node_type}.{field_name}")
                injection_count += 1
                
            else:
                # Handle non-image inputs (text, numbers, etc.) - use existing logic
                node = find_node_by_type_and_field(self.workflow_json, node_type, field_name)
                if node:
                    set_node_property(node, field_name, value)
                    logging.info(f"Injected input '{input_name}' (value: {value}) into node {node_type} field {field_name}.")
                    injection_count += 1
                else:
                    logging.debug(f"Could not find node {node_type} with field {field_name} for input '{input_name}'. This may be normal if the workflow doesn't use this input.")
        
        if injection_count > 0:
            logging.info(f"Successfully injected {injection_count} dynamic inputs into workflow.")
            if new_nodes_created:
                logging.info(f"Created {len(new_nodes_created)} LoadImage nodes: {new_nodes_created}")

    def _inject_dynamic_inputs_litegraph(self):
        """Inject inputs into LiteGraph format workflow."""
        if not self.input_schema or not self.input_schema.get('properties'):
            logging.info(f"No input schema defined for workflow {self.workflow_template.get('name')}. Skipping dynamic injection.")
            return

        injection_count = 0
        new_nodes_created = []  # Track LoadImage nodes for potential cleanup
        
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
                # CRITICAL FIX: Handle image inputs properly by creating LoadImage nodes
                materialized_path = materialize_image_input(value, self.input_dir)
                if not materialized_path:
                    logging.error(f"Failed to materialize image for input '{input_name}'. Skipping injection.")
                    continue
                
                # Get filename without path for ComfyUI
                image_filename = os.path.basename(materialized_path)
                
                # Find target node in LiteGraph nodes array
                target_node = None
                for node in self.workflow_json.get('nodes', []):
                    if node.get('type') == node_type:
                        # Check if this node has the target field
                        inputs = node.get('inputs', [])
                        if isinstance(inputs, list):
                            for inp in inputs:
                                if isinstance(inp, dict) and inp.get('name') == field_name:
                                    target_node = node
                                    break
                        if target_node:
                            break
                
                if not target_node:
                    logging.debug(f"Could not find node {node_type} with field {field_name} for input '{input_name}'. This may be normal if the workflow doesn't use this input.")
                    continue
                
                # Create a unique ID for the new LoadImage node
                image_node_id = f"load_image_{len(new_nodes_created)}"
                
                # Create the LoadImage node in LiteGraph format
                load_image_node = {
                    "id": image_node_id,
                    "type": "LoadImage",
                    "pos": [100, 100],  # Default position
                    "size": [248, 46],
                    "flags": {},
                    "order": len(self.workflow_json.get('nodes', [])),
                    "mode": 0,
                    "inputs": [],
                    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1]}],
                    "widgets_values": [image_filename],
                    "properties": {"Node name for S&R": "LoadImage"},
                    "widgets_order": ["image"]
                }
                
                # Add the LoadImage node to the workflow
                if 'nodes' not in self.workflow_json:
                    self.workflow_json['nodes'] = []
                self.workflow_json['nodes'].append(load_image_node)
                new_nodes_created.append(image_node_id)
                
                # Set the target node to reference the new LoadImage node
                # Find the input slot index for the target field
                input_slot = 0  # Default for most image inputs
                inputs = target_node.get('inputs', [])
                if isinstance(inputs, list):
                    for i, inp in enumerate(inputs):
                        if isinstance(inp, dict) and inp.get('name') == field_name:
                            input_slot = i
                            break
                
                # Create a link between the new LoadImage node and the target node
                link_id = len(self.workflow_json.get('links', []))
                if 'links' not in self.workflow_json:
                    self.workflow_json['links'] = []
                
                # Add the link
                self.workflow_json['links'].append([
                    link_id,          # link_id
                    image_node_id,    # from_node_id
                    0,                # from_slot
                    target_node.get('id'),  # to_node_id
                    input_slot,       # to_slot
                    "IMAGE"           # type
                ])
                
                logging.info(f"Created LoadImage node '{image_node_id}' for input '{input_name}' and connected to {node_type}.{field_name}")
                injection_count += 1
                
            else:
                # Handle non-image inputs (text, numbers, etc.)
                # Find node in LiteGraph nodes array
                target_node = None
                for node in self.workflow_json.get('nodes', []):
                    if node.get('type') == node_type:
                        # Check if this node has the target field
                        inputs = node.get('inputs', [])
                        if isinstance(inputs, list):
                            for inp in inputs:
                                if isinstance(inp, dict) and inp.get('name') == field_name:
                                    target_node = node
                                    break
                        if target_node:
                            break
                
                if target_node:
                    # Update widgets_values for the target node
                    # Find the index of the field in widgets_values
                    widgets_values = target_node.get('widgets_values', [])
                    inputs = target_node.get('inputs', [])
                    
                    field_index = None
                    for i, inp in enumerate(inputs):
                        if isinstance(inp, dict) and inp.get('name') == field_name:
                            field_index = i
                            break
                    
                    if field_index is not None and field_index < len(widgets_values):
                        widgets_values[field_index] = value
                        target_node['widgets_values'] = widgets_values
                        logging.info(f"Injected input '{input_name}' (value: {value}) into node {node_type} field {field_name}.")
                        injection_count += 1
                    else:
                        logging.debug(f"Could not find widgets_values index for field {field_name} in node {node_type}")
                else:
                    logging.debug(f"Could not find node {node_type} with field {field_name} for input '{input_name}'. This may be normal if the workflow doesn't use this input.")
        
        if injection_count > 0:
            logging.info(f"Successfully injected {injection_count} dynamic inputs into LiteGraph workflow.")
            if new_nodes_created:
                logging.info(f"Created {len(new_nodes_created)} LoadImage nodes: {new_nodes_created}")

    def _ensure_dependencies(self):
        """Check for and install missing custom node dependencies."""
        required_deps = self.workflow_template.get('custom_node_dependencies', [])
        if not required_deps:
            logging.info("No custom node dependencies specified.")
            return

        logging.info(f"Checking {len(required_deps)} custom node dependencies...")
        
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
            
            found_nodes = [n for n in expected_nodes if n in installed_nodes_set]
            
            if found_nodes:
                logging.info(f"[OK] Dependency satisfied: {repo_url}")
                logging.debug(f"  Found nodes: {found_nodes}")
                verified_deps.append(repo_url)
                continue
            
            fuzzy_matches = []
            for expected_node in expected_nodes:
                for installed_node in installed_nodes:
                    if (
                        expected_node.lower() in installed_node.lower() or 
                        installed_node.lower() in expected_node.lower()
                    ):
                        fuzzy_matches.append((expected_node, installed_node))
            
            if fuzzy_matches:
                logging.info(f"[OK] Dependency likely satisfied (fuzzy): {repo_url}")
                logging.debug(f"  Fuzzy matches: {fuzzy_matches}")
                verified_deps.append(repo_url)
                continue
            
            logging.warning(f"[MISSING] Dependency missing: {repo_url}")
            logging.debug(f"  Expected nodes: {expected_nodes}")
            repos_to_install.append((repo_url, expected_nodes))

        if not repos_to_install:
            logging.info("All dependencies are satisfied!")
            return

        logging.info(f"Installing {len(repos_to_install)} missing repositories...")
        
        failed_installs = []
        successful_installs = []
        
        for repo_url, expected_nodes in repos_to_install:
            logging.info(f"Installing: {repo_url}")
            success = manager_install_custom_node_via_cli(repo_url)
            
            if success:
                successful_installs.append(repo_url)
                logging.info(f"[OK] Successfully installed {repo_url}")
            else:
                failed_installs.append((repo_url, expected_nodes))
                logging.error(f"[ERROR] Failed to install {repo_url}")

        if successful_installs:
            logging.info("=" * 60)
            logging.info(f"Installed {len(successful_installs)} new custom nodes")
            logging.info("Restarting ComfyUI to load new nodes...")
            logging.info("=" * 60)
            
            if not docker_manager.restart_container():
                raise RuntimeError("Container restart failed")
            
            if not self.comfyui_client.wait_for_ready(self.shutdown_event, timeout=180):
                raise RuntimeError("ComfyUI did not become ready after restart")
            
            logging.info("Running cm-cli fix...")
            fix_all_custom_node_dependencies()
            
            time.sleep(10)
            
            self.comfyui_client.refresh_nodes()
            docker_manager.invalidate_node_cache()
            
            time.sleep(5)
        
        if failed_installs:
            failed_repos = [url for url, _ in failed_installs]
            logging.error(f"Failed to install {len(failed_installs)} repositories")
            raise MissingDependenciesError(failed_repos)

    def _validate_workflow(self):
        """Validate workflow structure."""
        is_valid, errors = validate_workflow_structure(self.workflow_json)
        if not is_valid:
            for error in errors:
                logging.error(f"[VALIDATION] {error}")
            return False
        
        logging.info("Workflow validation passed")
        return True

    def process(self):
        self._ensure_dependencies()
        
        # === HANDLE FORMAT ===
        if self.workflow_format == 'litegraph':
            logging.info("=" * 60)
            logging.info("LITEGRAPH FORMAT WORKFLOW (has subgraphs)")
            logging.info("Skipping cleaning/validation - ComfyUI will handle")
            logging.info("=" * 60)
            
            # Inject inputs into LiteGraph format
            self._inject_dynamic_inputs_litegraph()
            
            # Send directly
            payload = {"prompt": self.workflow_json}
            
        else:  # API format
            logging.info("API FORMAT WORKFLOW (no subgraphs)")
            
            # Clean and validate as before
            self._clean_ui_nodes()
            if not self._validate_workflow():
                logging.error("Workflow validation failed")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                return
            
            self._inject_dynamic_inputs()
            payload = {"prompt": self.workflow_json}
        
        # === COMPREHENSIVE DEBUG: Check for invalid data types BEFORE sending ===
        logging.info("=" * 60)
        logging.info("PRE-FLIGHT WORKFLOW CHECK:")
        
        # UI-only fields that should be removed
        UI_METADATA_FIELDS = {
            'widget_ue_connectable',
            'widget_control_after_generate',
            'widget_control_filter_list',
            '_meta',
        }
        
        # Fields that should NEVER have strings (these expect tensors/connections)
        IMAGE_EXPECTED_FIELDS = {
            'images', 'image', 'pixels', 'input_image', 'start_image', 'init_image',
            'image1', 'image2', 'image3', 'reference_image', 'style_image',
            'control_image', 'mask_image', 'background_image', 'foreground_image'
        }
        
        issues_found = False
        for node_id, node_data in self.workflow_json.items():
            if isinstance(node_data, dict):
                class_type = node_data.get('class_type', 'UNKNOWN')
                inputs = node_data.get('inputs', {})
                
                if isinstance(inputs, dict):
                    for input_name, input_value in inputs.items():
                        # Skip node references
                        if isinstance(input_value, list) and len(input_value) >= 2 and isinstance(input_value[1], int):
                            continue
                        
                        # CRITICAL: Check for strings in image-expected fields
                        if (input_name.lower() in IMAGE_EXPECTED_FIELDS or
                            any(field in input_name.lower() for field in IMAGE_EXPECTED_FIELDS)):
                            if isinstance(input_value, str):
                                logging.error(f"  [CRITICAL ERROR] String value in image field {node_id} ({class_type}).{input_name}: '{input_value}'")
                                issues_found = True
                            elif isinstance(input_value, dict):
                                logging.error(f"  [CRITICAL ERROR] Dict value in image field {node_id} ({class_type}).{input_name}: {input_value}")
                                issues_found = True
                        
                        # Check for dict values (should NOT exist!)
                        if isinstance(input_value, dict):
                            # Check if it's a UI metadata field (can be safely removed)
                            if input_name in UI_METADATA_FIELDS:
                                logging.warning(f"  [WARN] UI metadata in {node_id} ({class_type}).{input_name}: {input_value}")
                            else:
                                # This is a real problem (like model dicts)
                                logging.error(f"  [ERROR] Found dict in {node_id} ({class_type}).{input_name}: {input_value}")
                                issues_found = True
                        
                        # Also check for strings in general (might be problematic)
                        if isinstance(input_value, str) and len(input_value) > 100:
                            logging.warning(f"  [WARN] Very long string in {node_id} ({class_type}).{input_name}: '{input_value[:100]}...'")
                        
                        # Check for numbers where strings might be expected
                        if isinstance(input_value, (int, float)) and input_name in ['steps', 'width', 'height']:
                            if not isinstance(input_value, (int, float)):
                                logging.warning(f"  [WARN] Non-numeric value in numeric field {node_id} ({class_type}).{input_name}: {input_value}")
        
        if issues_found:
            logging.error("=" * 60)
            logging.error("CRITICAL ERROR: Invalid dictionaries found in workflow!")
            logging.error("This will cause ComfyUI to reject the workflow.")
            logging.error("Running emergency normalization...")
            logging.error("=" * 60)
            
            # Emergency fix
            self.workflow_json = normalize_model_references(self.workflow_json)
            payload = {"prompt": self.workflow_json}
            
            # Re-check
            logging.info("Re-checking after emergency normalization...")
            issues_found = False
            for node_id, node_data in self.workflow_json.items():
                if isinstance(node_data, dict):
                    inputs = node_data.get('inputs', {})
                    if isinstance(inputs, dict):
                        for input_name, input_value in inputs.items():
                            if isinstance(input_value, list) and len(input_value) >= 2 and isinstance(input_value[1], int):
                                continue
                            if isinstance(input_value, dict):
                                if input_name not in UI_METADATA_FIELDS:
                                    logging.error(f"  [STILL BROKEN] {node_id}.{input_name}: {input_value}")
                                    issues_found = True
            
            if issues_found:
                logging.error("Emergency normalization FAILED. Aborting job.")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                return
            else:
                logging.info("Emergency normalization SUCCESS!")
        else:
            logging.info("  [OK] Workflow is clean")
        
        logging.info("=" * 60)
        
        # COMPREHENSIVE WORKFLOW STRUCTURE LOGGING
        logging.info("FINAL WORKFLOW STRUCTURE:")
        logging.info(f"Total nodes: {len(self.workflow_json)}")
        
        save_image_issues = []  # Track SaveImage issues
        
        for node_id, node_data in self.workflow_json.items():
            if isinstance(node_data, dict):
                class_type = node_data.get('class_type', 'UNKNOWN')
                logging.info(f"  Node {node_id}: {class_type}")
                
                # Log inputs to see references
                inputs = node_data.get('inputs', {})
                if isinstance(inputs, dict):
                    for input_name, input_value in inputs.items():
                        if isinstance(input_value, list) and len(input_value) >= 1:
                            logging.info(f"    {input_name} -> Node {input_value[0]}")
                        else:
                            # Log value (truncated if long)
                            value_str = str(input_value)
                            if len(value_str) > 50:
                                value_str = value_str[:50] + "..."
                            logging.info(f"    {input_name} = {value_str}")
                            
                            # CRITICAL: Check for problematic assignments
                            if (class_type == 'SaveImage' and input_name == 'images' and isinstance(input_value, str)):
                                save_image_issues.append(f"Node {node_id}: '{input_value}'")
                                logging.error(f"  [CRITICAL ERROR] SaveImage has string in images field: {input_value}")
        
        # Summary of SaveImage issues
        if save_image_issues:
            logging.error("=" * 60)
            logging.error("CRITICAL: SaveImage nodes with string images:")
            for issue in save_image_issues:
                logging.error(f"  - {issue}")
            logging.error("=" * 60)
        
        logging.info("=" * 60)
        
        outputs = self._trigger_and_get_output(payload)
        if not outputs:
            return

        output_path = None
        thumbnail_path = None
        duration = None

        if self.target_entity == 'scene' or self.workflow_type in ['wan-2.2-text-to-video', 'wan-2.2-image-to-video']:
            video_info = find_video_in_output(outputs)
            if not video_info:
                logging.error(f"No video output found for job {self.job_id}")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                return

            video_filename, subfolder = video_info
            temp_host_path = self._copy_file_from_container(video_filename, subfolder)
            if not temp_host_path:
                logging.error(f"Failed to copy video from container")
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
                    logging.error(f"Video upload failed")
                    self.orchestrator_service.update_job_status(self.job_id, 'failed')
                    return
            finally:
                os.remove(temp_host_path)

        elif self.target_entity == 'audio_clip' or self.workflow_type in ['hunyuan_video_foley', 'vibevoice', 'vibevoice_multi_clone', 'diffrhythm']:
            audio_info = find_audio_in_output(outputs)
            if not audio_info:
                logging.error(f"No audio output found for job {self.job_id}")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                return

            audio_filename, subfolder = audio_info
            temp_host_path = self._copy_file_from_container(audio_filename, subfolder)
            if not temp_host_path:
                logging.error(f"Failed to copy audio from container")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                return

            try:
                output_path = self.orchestrator_service.upload_audio_output(temp_host_path, self.job_id)
                if output_path:
                    duration = get_audio_duration(temp_host_path)
                else:
                    logging.error(f"Audio upload failed")
                    self.orchestrator_service.update_job_status(self.job_id, 'failed')
                    return
            finally:
                os.remove(temp_host_path)

        elif self.target_entity == 'character' or self.workflow_type == 'qwen':
            image_info = find_image_in_output(outputs)
            if not image_info:
                logging.error(f"No image output found for job {self.job_id}")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                return

            image_filename, subfolder = image_info
            temp_host_path = self._copy_file_from_container(image_filename, subfolder)
            if not temp_host_path:
                logging.error(f"Failed to copy image from container")
                self.orchestrator_service.update_job_status(self.job_id, 'failed')
                return

            try:
                output_path = self.orchestrator_service.upload_image_output(temp_host_path, self.job_id)
                if output_path:
                    thumbnail_path = output_path
                else:
                    logging.error(f"Image upload failed")
                    self.orchestrator_service.update_job_status(self.job_id, 'failed')
                    return
            finally:
                os.remove(temp_host_path)

        self.orchestrator_service.update_job_status(
            self.job_id, 
            'completed', 
            output_path=output_path, 
            thumbnail_path=thumbnail_path, 
            duration_seconds=duration, 
            completion_metadata=self.job.get('completion_metadata')
        )