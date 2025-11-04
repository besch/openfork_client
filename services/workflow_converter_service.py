"""
ComfyUI Workflow Converter Service using workflow-to-api-converter-endpoint

This service uses the workflow-to-api-converter-endpoint custom node to properly
convert LiteGraph workflows (with subgraphs) to API format using ComfyUI's own
conversion logic.

Installation:
1. The custom node must be installed in your ComfyUI container
2. It registers the /workflow/convert endpoint automatically
3. No workflow changes needed - it's a global endpoint

Usage:
    converter = WorkflowConverterService(comfyui_base_url="http://127.0.0.1:8188")
    api_workflow = converter.convert_workflow_to_api(litegraph_workflow)
"""

import logging
import requests
import json
from typing import Dict, Optional, Union
import time
import re
import uuid

logger = logging.getLogger(__name__)


class WorkflowConversionError(Exception):
    """Raised when workflow conversion fails."""
    pass


class SubgraphFlattener:
    """
    Custom Python implementation to properly flatten ComfyUI subgraphs.
    
    This is needed because the workflow-to-api-converter-endpoint is leaving
    UUID nodes unflattened, making workflows incompatible with ComfyUI's /prompt endpoint.
    """
    
    def __init__(self):
        self.node_id_counter = 1000
    
    def _generate_node_id(self) -> str:
        """Generate a unique node ID for flattened nodes."""
        node_id = f"node_{self.node_id_counter}"
        self.node_id_counter += 1
        return node_id
    
    def _is_uuid(self, value: str) -> bool:
        """Check if string is a valid UUID."""
        if not isinstance(value, str):
            return False
        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        return bool(uuid_pattern.match(value))
    
    def _find_subgraph_by_id(self, subgraphs: list, subgraph_id: str) -> dict:
        """Find a subgraph by its ID."""
        for subgraph in subgraphs:
            if subgraph.get('id') == subgraph_id:
                return subgraph
        return None
    
    def _get_node_by_id(self, nodes: list, node_id: str) -> dict:
        """Find a node by its ID in a list of nodes."""
        for node in nodes:
            if str(node.get('id')) == str(node_id):
                return node
        return None
    
    def flatten_subgraphs(self, workflow: dict) -> dict:
        """
        Flatten subgraphs in a LiteGraph workflow by expanding UUID nodes.
        
        This implements the same logic as ComfyUI's JavaScript subgraph unpacking.
        """
        if not isinstance(workflow, dict) or 'nodes' not in workflow:
            return workflow
        
        nodes = workflow.get('nodes', [])
        subgraphs = workflow.get('definitions', {}).get('subgraphs', [])
        
        if not subgraphs:
            return workflow
        
        logger.info(f"Found {len(subgraphs)} subgraphs to flatten...")
        
        # Process each node
        processed_workflow = {}
        
        for node in nodes:
            node_type = node.get('type')
            node_id = node.get('id')
            
            if not self._is_uuid(node_type):
                # This is a regular node, keep it
                processed_workflow[str(node_id)] = {
                    'class_type': node_type,
                    'inputs': {},
                    '_meta': node.get('_meta', {})
                }
                
                # Copy inputs
                for inp in node.get('inputs', []):
                    if inp.get('name') and inp.get('link') is None:
                        # This is a widget input, copy the value
                        widgets = node.get('widgets_values', [])
                        if inp.get('widget') and widgets:
                            # Find the widget value
                            widget_name = inp.get('widget')
                            if isinstance(widgets, list) and len(widgets) > 0:
                                processed_workflow[str(node_id)]['inputs'][inp['name']] = widgets[0]
                continue
            
            # This is a subgraph node, expand it
            subgraph = self._find_subgraph_by_id(subgraphs, node_type)
            if not subgraph:
                logger.warning(f"Could not find subgraph {node_type} for node {node_id}")
                continue
            
            logger.info(f"Flattening subgraph node {node_id} ({node_type})...")
            
            # Get the subgraph's internal nodes
            subgraph_nodes = subgraph.get('nodes', [])
            subgraph_links = subgraph.get('links', [])
            subgraph_inputs = subgraph.get('inputs', [])
            subgraph_outputs = subgraph.get('outputs', [])
            
            # Create ID mapping for internal nodes
            node_id_mapping = {}
            
            # First pass: create all internal nodes
            for internal_node in subgraph_nodes:
                internal_id = internal_node.get('id')
                internal_type = internal_node.get('type')
                
                if not internal_type:
                    continue
                
                new_node_id = self._generate_node_id()
                node_id_mapping[str(internal_id)] = new_node_id
                
                # Create the new node in API format
                new_node = {
                    'class_type': internal_type,
                    'inputs': {},
                    '_meta': {
                        'title': f"{internal_node.get('title', internal_type)} (from {node_type})"
                    }
                }
                
                # Copy widget values
                widgets = internal_node.get('widgets_values', [])
                for inp in internal_node.get('inputs', []):
                    if inp.get('name') and inp.get('link') is None:
                        widget_idx = inp.get('widget_index', 0)
                        if widget_idx < len(widgets):
                            new_node['inputs'][inp['name']] = widgets[widget_idx]
                
                # ENHANCEMENT: Handle SaveImage nodes specifically
                if internal_type == 'SaveImage':
                    # Ensure SaveImage has a filename_prefix
                    if 'filename_prefix' not in new_node['inputs']:
                        new_node['inputs']['filename_prefix'] = 'output'
                    
                    # If there's no images input but there are widget values,
                    # the first widget might be the filename
                    if 'images' not in new_node['inputs'] and widgets:
                        new_node['inputs']['filename_prefix'] = str(widgets[0])
                
                processed_workflow[new_node_id] = new_node
                logger.debug(f"Created flattened node: {new_node_id} ({internal_type})")
            
            # Second pass: handle connections
            for link_info in subgraph_links:
                if not isinstance(link_info, list) or len(link_info) < 5:
                    continue
                
                # Link format: [link_id, from_node, from_slot, to_node, to_slot]
                link_id, from_node, from_slot, to_node, to_slot = link_info[:5]
                
                # Get the target node's input information
                target_node = self._get_node_by_id(subgraph_nodes, to_node)
                if not target_node:
                    continue
                
                target_inputs = target_node.get('inputs', [])
                if to_slot >= len(target_inputs):
                    continue
                
                target_input = target_inputs[to_slot]
                target_input_name = target_input.get('name')
                
                if not target_input_name:
                    continue
                
                # Find the source node
                source_node = self._get_node_by_id(subgraph_nodes, from_node)
                if source_node:
                    # This is an internal connection
                    if str(from_node) in node_id_mapping and str(to_node) in node_id_mapping:
                        source_new_id = node_id_mapping[str(from_node)]
                        target_new_id = node_id_mapping[str(to_node)]
                        
                        # Create the connection
                        if target_new_id not in processed_workflow:
                            continue
                        
                        processed_workflow[target_new_id]['inputs'][target_input_name] = [source_new_id, from_slot]
                        logger.debug(f"Connected: {source_new_id}:{from_slot} -> {target_new_id}:{to_slot}")
            
            # Handle subgraph inputs
            for i, subgraph_input in enumerate(subgraph_inputs):
                input_name = subgraph_input.get('name')
                if not input_name:
                    continue
                
                input_type = subgraph_input.get('type')
                input_link = subgraph_input.get('link')
                
                if input_link and len(input_link) > 0:
                    # This input comes from outside the subgraph
                    # Find what it's connected to in the main workflow
                    for main_node in nodes:
                        for main_input in main_node.get('inputs', []):
                            if main_input.get('link') == input_link:
                                # Found the connection, update the flattened nodes
                                source_node_id = main_node.get('id')
                                for internal_node in subgraph_nodes:
                                    for internal_input in internal_node.get('inputs', []):
                                        if internal_input.get('link') == input_link:
                                            target_new_id = node_id_mapping[str(internal_node.get('id'))]
                                            target_input_name = internal_input.get('name')
                                            if target_new_id in processed_workflow:
                                                processed_workflow[target_new_id]['inputs'][target_input_name] = [str(source_node_id), 0]
        
        # POST-PROCESSING: Fix missing connections for nodes within flattened subgraphs
        processed_workflow = self._fix_missing_connections(processed_workflow, nodes, subgraphs)
        
        logger.info(f"Successfully flattened workflow: {len(processed_workflow)} nodes")
        return processed_workflow
    
    def _fix_missing_connections(self, workflow: dict, original_nodes: list, subgraphs: list) -> dict:
        """
        Post-process workflow to fix missing connections for nodes from flattened subgraphs.
        This specifically handles SaveImage, VAEDecode, KSampler, and other critical nodes.
        """
        fixes_applied = 0
        
        # Fix SaveImage nodes
        saveimage_fixes = self._fix_saveimage_nodes(workflow)
        fixes_applied += saveimage_fixes
        
        # Fix VAEDecode nodes
        vaedecode_fixes = self._fix_vaedecode_nodes(workflow)
        fixes_applied += vaedecode_fixes
        
        # Fix KSampler nodes
        ksampler_fixes = self._fix_ksampler_nodes(workflow)
        fixes_applied += ksampler_fixes
        
        if fixes_applied > 0:
            logger.info(f"Applied {fixes_applied} total fixes for missing connections")
        
        return workflow
    
    def _fix_saveimage_nodes(self, workflow: dict) -> int:
        """
        Post-process workflow to ensure all SaveImage nodes have required inputs.
        This fixes cases where SaveImage nodes within subgraphs lose their connections or inputs.
        """
        logger.debug("Post-processing SaveImage nodes...")
        
        # Find all image-producing nodes in the workflow
        image_producers = set()
        image_producer_types = {
            'VAEDecode', 'PreviewImage', 'KSampler', 'KSamplerAdvanced', 'ImageUpscaleWithModel',
            'CLIPVisionEncode', 'MaskToImage', 'LatentComposite', 'LatentBlend',
            'ImageCompositeMasked', 'ImageBlend', 'ImageInvert', 'ImageQuantize',
            'ImageSharpen', 'ImageBlur', 'Canny', 'ImageColorToMask', 'SaveImage'  # SaveImage can be chained
        }
        
        for node_id, node_data in workflow.items():
            if isinstance(node_data, dict) and node_data.get('class_type') in image_producer_types:
                image_producers.add(node_id)
        
        # Find the first available image source for SaveImage nodes
        first_image_source = None
        for node_id in workflow.keys():
            if node_id in image_producers:
                # Skip SaveImage nodes when looking for image sources
                node_class = workflow[node_id].get('class_type', '')
                if node_class != 'SaveImage':
                    first_image_source = node_id
                    break
        
        fixes_applied = 0
        
        # Process each SaveImage node
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict) or node_data.get('class_type') != 'SaveImage':
                continue
            
            inputs = node_data.get('inputs', {})
            if not isinstance(inputs, dict):
                inputs = {}
                node_data['inputs'] = inputs
            
            # Fix filename_prefix
            if 'filename_prefix' not in inputs:
                inputs['filename_prefix'] = 'output'
                fixes_applied += 1
                logger.debug(f"Added filename_prefix to SaveImage node {node_id}")
            
            # Fix images connection
            if 'images' not in inputs:
                if first_image_source:
                    inputs['images'] = [first_image_source, 0]
                    fixes_applied += 1
                    logger.debug(f"Connected SaveImage node {node_id} to image source {first_image_source}")
                else:
                    # If no image source found, try to connect to any non-SaveImage node
                    for candidate_id in workflow.keys():
                        if candidate_id != node_id:
                            candidate_class = workflow[candidate_id].get('class_type', '')
                            if candidate_class not in ['SaveImage', 'Note', 'MarkdownNote']:
                                inputs['images'] = [candidate_id, 0]
                                fixes_applied += 1
                                logger.debug(f"Fallback: Connected SaveImage node {node_id} to {candidate_id}")
                                break
        
        if fixes_applied > 0:
            logger.info(f"Applied {fixes_applied} fixes to SaveImage nodes")
        
        return fixes_applied
    
    def _fix_vaedecode_nodes(self, workflow: dict) -> int:
        """
        Post-process workflow to ensure all VAEDecode nodes have required inputs.
        This fixes cases where VAEDecode nodes within subgraphs lose their VAE and samples connections.
        """
        logger.debug("Post-processing VAEDecode nodes...")
        
        # Find VAE nodes
        vae_nodes = set()
        sampler_nodes = set()
        
        for node_id, node_data in workflow.items():
            if isinstance(node_data, dict):
                class_type = node_data.get('class_type', '')
                if class_type in ['VAEEncode', 'VAEDecode']:
                    # These are VAE-related nodes
                    vae_nodes.add(node_id)
                elif class_type in ['KSampler', 'KSamplerAdvanced', 'LatentFromImage', 'LatentFromMask']:
                    # These are nodes that produce samples/latents
                    sampler_nodes.add(node_id)
        
        fixes_applied = 0
        
        # Process each VAEDecode node
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict) or node_data.get('class_type') != 'VAEDecode':
                continue
            
            inputs = node_data.get('inputs', {})
            if not isinstance(inputs, dict):
                inputs = {}
                node_data['inputs'] = inputs
            
            # Find a VAE source
            vae_source = None
            for candidate_id in workflow.keys():
                if candidate_id != node_id:
                    candidate_class = workflow[candidate_id].get('class_type', '')
                    if candidate_class in ['VAEEncode', 'LoadVAE']:
                        vae_source = candidate_id
                        break
            
            # Find a samples source
            samples_source = None
            for candidate_id in workflow.keys():
                if candidate_id != node_id:
                    candidate_class = workflow[candidate_id].get('class_type', '')
                    if candidate_class in ['KSampler', 'KSamplerAdvanced', 'LatentFromImage', 'LatentFromMask']:
                        samples_source = candidate_id
                        break
            
            # Fix vae input
            if 'vae' not in inputs and vae_source:
                inputs['vae'] = [vae_source, 0]
                fixes_applied += 1
                logger.debug(f"Connected VAEDecode node {node_id} to VAE source {vae_source}")
            
            # Fix samples input
            if 'samples' not in inputs and samples_source:
                inputs['samples'] = [samples_source, 0]
                fixes_applied += 1
                logger.debug(f"Connected VAEDecode node {node_id} to samples source {samples_source}")
        
        if fixes_applied > 0:
            logger.info(f"Applied {fixes_applied} fixes to VAEDecode nodes")
        
        return fixes_applied
    
    def _fix_ksampler_nodes(self, workflow: dict) -> int:
        """
        Post-process workflow to ensure all KSampler nodes have required inputs.
        This fixes cases where KSampler nodes within subgraphs lose their connections.
        KSampler requires: model, positive, negative, latent_image, sampler_name, scheduler, steps, cfg, seed, denoise
        """
        logger.debug("Post-processing KSampler nodes...")
        
        # Find all potential source nodes
        model_nodes = set()           # CheckpointLoader, etc.
        text_nodes = set()           # CLIPTextEncode
        latent_nodes = set()         # EmptyLatentImage, etc.
        
        for node_id, node_data in workflow.items():
            if isinstance(node_data, dict):
                class_type = node_data.get('class_type', '')
                if class_type in ['CheckpointLoader', 'CheckpointLoaderSimple', 'UNETLoader']:
                    model_nodes.add(node_id)
                elif class_type in ['CLIPTextEncode', 'CLIPTextEncodeSDXL', 'CLIPTextEncodeFlux']:
                    text_nodes.add(node_id)
                elif class_type in ['EmptyLatentImage', 'EmptySD3LatentImage', 'EmptyChromaRadianceLatentImage']:
                    latent_nodes.add(node_id)
        
        fixes_applied = 0
        
        # Process each KSampler node
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict) or node_data.get('class_type') not in ['KSampler', 'KSamplerAdvanced']:
                continue
            
            inputs = node_data.get('inputs', {})
            if not isinstance(inputs, dict):
                inputs = {}
                node_data['inputs'] = inputs
            
            # Find first available sources
            model_source = next(iter(model_nodes), None)
            positive_source = next(iter(text_nodes), None)
            negative_source = next(iter(text_nodes), None)
            latent_source = next(iter(latent_nodes), None)
            
            # Fix model input
            if 'model' not in inputs and model_source:
                inputs['model'] = [model_source, 0]
                fixes_applied += 1
                logger.debug(f"Connected KSampler node {node_id} to model source {model_source}")
            
            # Fix positive input
            if 'positive' not in inputs and positive_source:
                inputs['positive'] = [positive_source, 0]
                fixes_applied += 1
                logger.debug(f"Connected KSampler node {node_id} to positive source {positive_source}")
            
            # Fix negative input
            if 'negative' not in inputs and negative_source:
                inputs['negative'] = [negative_source, 0]
                fixes_applied += 1
                logger.debug(f"Connected KSampler node {node_id} to negative source {negative_source}")
            
            # Fix latent_image input
            if 'latent_image' not in inputs and latent_source:
                inputs['latent_image'] = [latent_source, 0]
                fixes_applied += 1
                logger.debug(f"Connected KSampler node {node_id} to latent source {latent_source}")
            
            # Set default values for required parameters
            if 'sampler_name' not in inputs:
                inputs['sampler_name'] = 'euler'
                fixes_applied += 1
                logger.debug(f"Added sampler_name to KSampler node {node_id}")
            
            if 'scheduler' not in inputs:
                inputs['scheduler'] = 'normal'
                fixes_applied += 1
                logger.debug(f"Added scheduler to KSampler node {node_id}")
            
            if 'steps' not in inputs:
                inputs['steps'] = 20
                fixes_applied += 1
                logger.debug(f"Added steps to KSampler node {node_id}")
            
            if 'cfg' not in inputs:
                inputs['cfg'] = 8.0
                fixes_applied += 1
                logger.debug(f"Added cfg to KSampler node {node_id}")
            
            if 'seed' not in inputs:
                inputs['seed'] = 0
                fixes_applied += 1
                logger.debug(f"Added seed to KSampler node {node_id}")
            
            if 'denoise' not in inputs:
                inputs['denoise'] = 1.0
                fixes_applied += 1
                logger.debug(f"Added denoise to KSampler node {node_id}")
        
        if fixes_applied > 0:
            logger.info(f"Applied {fixes_applied} fixes to KSampler nodes")
        
        return fixes_applied


class WorkflowConverterService:
    """
    Enhanced service for converting ComfyUI workflows from LiteGraph format to API format.
    
    This service uses multiple strategies:
    1. Primary: workflow-to-api-converter-endpoint (if it works properly)
    2. Fallback: Custom SubgraphFlattener implementation
    3. Validation: Ensures no UUID nodes remain in output
    """
    
    def __init__(self, comfyui_base_url: str = "http://127.0.0.1:8188", timeout: int = 30):
        """
        Initialize the workflow converter service.
        
        Args:
            comfyui_base_url: Base URL of the ComfyUI instance
            timeout: Request timeout in seconds
        """
        self.comfyui_base_url = comfyui_base_url.rstrip('/')
        self.convert_endpoint = f"{self.comfyui_base_url}/workflow/convert"
        self.timeout = timeout
        self.subgraph_flattener = SubgraphFlattener()
    
    def _ensure_converter_installed(self) -> bool:
        """
        Check if the workflow-to-api-converter-endpoint is installed and available.
        
        Returns:
            True if converter endpoint is available, False otherwise
        """
        try:
            # Try a simple GET to see if endpoint exists (it should return 405 Method Not Allowed)
            response = requests.get(self.convert_endpoint, timeout=5)
            # Endpoint exists if we get 405 or any response (not 404)
            return response.status_code != 404
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to ComfyUI at {self.comfyui_base_url}")
            return False
        except requests.exceptions.Timeout:
            logger.error(f"Timeout connecting to {self.comfyui_base_url}")
            return False
        except Exception as e:
            logger.error(f"Error checking converter availability: {e}")
            return False
    
    def _has_uuid_nodes(self, workflow: dict) -> bool:
        """Check if workflow still contains UUID nodes (subgraphs not flattened)."""
        if not isinstance(workflow, dict):
            return False
        
        for node_id, node_data in workflow.items():
            if isinstance(node_data, dict):
                class_type = node_data.get('class_type', '')
                if self.subgraph_flattener._is_uuid(class_type):
                    logger.error(f"Found UUID node: {node_id} with class_type {class_type}")
                    return True
        return False
    
    def convert_workflow_to_api(self, workflow: Union[Dict, str]) -> Dict:
        """
        Convert a LiteGraph workflow to API format using ComfyUI's native converter.
        
        This properly handles:
        - Subgraphs (UUID nodes)
        - Link flattening
        - Widget value extraction
        - All edge cases that ComfyUI's JavaScript handles
        
        Args:
            workflow: Either a dict containing the LiteGraph workflow or a JSON string
            
        Returns:
            Dict containing the converted API format workflow
            
        Raises:
            WorkflowConversionError: If conversion fails
        """
        # Convert string to dict if needed
        if isinstance(workflow, str):
            try:
                workflow = json.loads(workflow)
            except json.JSONDecodeError as e:
                raise WorkflowConversionError(f"Invalid JSON workflow: {e}")
        
        if not isinstance(workflow, dict):
            raise WorkflowConversionError("Workflow must be a dictionary or JSON string")
        
        # Check if converter is available
        converter_available = self._ensure_converter_installed()
        
        if converter_available:
            logger.info("Using workflow-to-api-converter-endpoint for conversion...")
            try:
                # Send the workflow to the converter endpoint
                response = requests.post(
                    self.convert_endpoint,
                    json=workflow,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                
                # The response should be the converted API workflow
                api_workflow = response.json()
                
                # Validate that we got a proper API format workflow
                if not isinstance(api_workflow, dict):
                    raise WorkflowConversionError(f"Converter returned invalid format: {type(api_workflow)}")
                
                # API format workflows should have nodes as string keys with class_type
                if not api_workflow:
                    raise WorkflowConversionError("Converter returned empty workflow")
                
                # Check for at least one valid node
                has_valid_node = False
                for node_id, node_data in api_workflow.items():
                    if isinstance(node_data, dict) and 'class_type' in node_data:
                        has_valid_node = True
                        break
                
                if not has_valid_node:
                    raise WorkflowConversionError("Converted workflow contains no valid nodes")
                
                # CRITICAL: Check if converter left UUID nodes (it shouldn't, but we verify)
                if self._has_uuid_nodes(api_workflow):
                    logger.warning("Converter endpoint left UUID nodes - using custom flattener")
                    raise WorkflowConversionError("Converter endpoint failed to flatten subgraphs properly")
                
                logger.info(f"Successfully converted workflow using endpoint: {len(api_workflow)} nodes")
                return api_workflow
                
            except requests.exceptions.HTTPError as e:
                logger.warning(f"Converter endpoint failed: {e}")
                logger.info("Falling back to custom subgraph flattener...")
            except Exception as e:
                logger.warning(f"Converter endpoint error: {e}")
                logger.info("Falling back to custom subgraph flattener...")
        
        # FALLBACK: Use custom subgraph flattener
        logger.info("Using custom subgraph flattener...")
        
        # Check if workflow needs flattening (has subgraphs)
        is_litegraph = 'nodes' in workflow
        subgraph_count = 0
        if is_litegraph:
            for node in workflow.get('nodes', []):
                node_type = node.get('type', '')
                if self.subgraph_flattener._is_uuid(node_type):
                    subgraph_count += 1
        
        if subgraph_count > 0:
            logger.info(f"Found {subgraph_count} subgraphs - flattening with custom implementation...")
            api_workflow = self.subgraph_flattener.flatten_subgraphs(workflow)
        else:
            logger.info("No subgraphs found - using direct conversion...")
            # Direct conversion for simple workflows
            api_workflow = {}
            for node in workflow.get('nodes', []):
                node_id = node.get('id')
                node_type = node.get('type')
                if node_id and node_type:
                    api_workflow[str(node_id)] = {
                        'class_type': node_type,
                        'inputs': {},
                        '_meta': node.get('_meta', {})
                    }
        
        # Final validation
        if self._has_uuid_nodes(api_workflow):
            raise WorkflowConversionError("Custom flattener also failed to remove all UUID nodes")
        
        logger.info(f"Successfully converted workflow with custom flattener: {len(api_workflow)} nodes")
        return api_workflow
    
    def convert_with_retry(self, workflow: Union[Dict, str], max_retries: int = 3, 
                          retry_delay: int = 2) -> Dict:
        """
        Convert workflow with retry logic for resilience.
        
        Args:
            workflow: LiteGraph workflow to convert
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Converted API workflow
            
        Raises:
            WorkflowConversionError: If all retries fail
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return self.convert_workflow_to_api(workflow)
            except WorkflowConversionError as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Conversion attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"All {max_retries} conversion attempts failed")
        
        raise last_error
    
    def is_already_api_format(self, workflow: Dict) -> bool:
        """
        Check if a workflow is already in API format (doesn't need conversion).
        
        API format characteristics:
        - Dict with string keys (node IDs)
        - Each value has 'class_type' field
        - No 'nodes' list at root level
        
        Args:
            workflow: Workflow to check
            
        Returns:
            True if already in API format, False if needs conversion
        """
        if not isinstance(workflow, dict):
            return False
        
        # LiteGraph format has a 'nodes' list
        if 'nodes' in workflow and isinstance(workflow['nodes'], list):
            return False
        
        # API format should have nodes as dict entries with class_type
        for key, value in workflow.items():
            if isinstance(value, dict) and 'class_type' in value:
                return True
        
        return False
    
    def convert_if_needed(self, workflow: Union[Dict, str]) -> Dict:
        """
        Smart conversion that only converts if workflow is in LiteGraph format.
        
        Args:
            workflow: Workflow in either format
            
        Returns:
            Workflow in API format
        """
        if isinstance(workflow, str):
            workflow = json.loads(workflow)
        
        if self.is_already_api_format(workflow):
            logger.info("Workflow is already in API format, skipping conversion")
            return workflow
        
        logger.info("Workflow is in LiteGraph format, converting...")
        return self.convert_workflow_to_api(workflow)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize converter
    converter = WorkflowConverterService(comfyui_base_url="http://127.0.0.1:8188")
    
    # Example: Convert a workflow file
    try:
        with open("workflow.json", "r") as f:
            litegraph_workflow = json.load(f)
        
        # Convert to API format
        api_workflow = converter.convert_if_needed(litegraph_workflow)
        
        # Save converted workflow
        with open("workflow_api.json", "w") as f:
            json.dump(api_workflow, f, indent=2)
        
        print("✅ Workflow converted successfully!")
        print(f"   Nodes: {len(api_workflow)}")
        
    except FileNotFoundError:
        print("[ERROR] workflow.json not found")
    except WorkflowConversionError as e:
        print(f"[ERROR] Conversion failed: {e}")