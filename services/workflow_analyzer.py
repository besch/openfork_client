"""
WorkflowAnalyzer - Analyzes ComfyUI workflow JSON to extract metadata automatically.

This module provides:
- Automatic input/output detection from workflow structure
- Node type analysis for dependency checking  
- Output type detection (video, audio, image)
- VRAM estimation based on model nodes
"""

import logging
from typing import Union, Any
from dataclasses import dataclass, field


@dataclass
class WorkflowInput:
    """Represents an input field that can be customized in a workflow."""
    name: str
    type: str  # 'text', 'number', 'image', 'video', 'audio', 'boolean', 'model', 'select'
    node_id: str
    widget_name: str
    default: Any = None
    options: list[str] = field(default_factory=list)  # For select types
    min_value: float = None
    max_value: float = None


@dataclass
class WorkflowOutput:
    """Represents an output from the workflow."""
    node_id: str
    output_type: str  # 'image', 'video', 'audio', 'text'
    class_type: str
    filename_prefix: str = None


@dataclass
class ModelRequirement:
    """Represents a model file required by the workflow."""
    filename: str
    model_type: str  # 'checkpoint', 'lora', 'vae', 'clip', 'controlnet', etc.
    node_id: str
    widget_name: str


@dataclass 
class WorkflowMetadata:
    """Complete metadata extracted from a workflow."""
    name: str
    inputs: list[WorkflowInput]
    outputs: list[WorkflowOutput]
    required_nodes: list[str]
    required_models: list[ModelRequirement] = field(default_factory=list)
    estimated_vram_mb: int = 0
    description: str = ""
    category: str = "general"


class WorkflowAnalyzer:
    """Analyzes ComfyUI workflow JSON to extract metadata automatically."""
    
    # Known output node types
    OUTPUT_NODES = {
        # Image output nodes
        "SaveImage": "image",
        "PreviewImage": "image",
        "SaveImageWebsocket": "image",
        
        # Video output nodes
        "SaveVideo": "video",
        "VHS_VideoCombine": "video",
        "SaveAnimatedWEBP": "video",
        "SaveAnimatedPNG": "video",
        
        # Audio output nodes
        "SaveAudio": "audio",
        "VHS_AudioCombine": "audio",
        "SaveAudioTensor": "audio",
        
        # Text output nodes
        "SaveText": "text",
        "ShowText": "text",
    }
    
    # Known input node patterns and their types
    INPUT_PATTERNS = {
        "CLIPTextEncode": {"inputs": {"text": "text"}, "names": ["Prompt", "Negative Prompt"]},
        "KSampler": {"inputs": {"seed": "number", "steps": "number", "cfg": "number"}},
        "KSamplerAdvanced": {"inputs": {"seed": "number", "steps": "number", "cfg": "number"}},
        "EmptyLatentImage": {"inputs": {"width": "number", "height": "number", "batch_size": "number"}},
        "LoadImage": {"inputs": {"image": "image"}},
        "VHS_LoadVideo": {"inputs": {"video": "video"}},
        "LoadVideo": {"inputs": {"video_path": "video"}},
        "LoadAudio": {"inputs": {"audio": "audio"}},
        "DiffRhythmRun": {"inputs": {"style_prompt": "text", "lyrics_or_edit_lyrics": "text", "seed": "number"}},
        "WanImageToVideo": {"inputs": {"width": "number", "height": "number", "length": "number"}},
        "HunyuanVideoSampler": {"inputs": {"seed": "number", "steps": "number", "cfg": "number"}},
        "LTXVSampler": {"inputs": {"seed": "number", "steps": "number", "cfg": "number"}},
    }
    
    # Model nodes that indicate VRAM usage
    MODEL_NODES = {
        "CheckpointLoader": 4000,  # ~4GB
        "CheckpointLoaderSimple": 4000,
        "UNETLoader": 4000,
        "VAELoader": 500,
        "CLIPLoader": 1000,
        "LoraLoader": 500,
        "ControlNetLoader": 2000,
        "DiffRhythmRun": 6000,  # Music generation
        "WanVideoSampler": 12000,  # Video generation
        "HunyuanVideoSampler": 12000,
        "LTXVSampler": 8000,
    }
    
    # Widget names that contain model filenames, mapped to model type
    # This is extensible - just add new entries as ComfyUI evolves
    MODEL_LOADER_WIDGETS = {
        "ckpt_name": "checkpoint",
        "unet_name": "unet",
        "vae_name": "vae",
        "clip_name": "clip",
        "lora_name": "lora",
        "control_net_name": "controlnet",
        "model_name": "checkpoint",  # Generic
        "style_model_name": "style_models",
        "gligen_name": "gligen",
        "ipadapter_file": "ipadapter",
        "upscale_model": "upscale_models",
        "model_path": "checkpoint",  # Used by some loaders
    }
    
    def analyze(self, workflow_data: dict, workflow_name: str = "Untitled") -> WorkflowMetadata:
        """
        Extract complete metadata from a workflow.
        
        Args:
            workflow_data: The parsed workflow JSON (API format)
            workflow_name: Name for the workflow
            
        Returns:
            WorkflowMetadata with inputs, outputs, and other info
        """
        # Normalize workflow format
        graph = self._normalize_graph(workflow_data)
        
        inputs = self._extract_inputs(graph)
        outputs = self._extract_outputs(graph)
        required_nodes = self._extract_required_nodes(graph)
        required_models = self._extract_model_requirements(graph)
        vram = self._estimate_vram(graph)
        category = self._infer_category(graph, outputs)
        
        return WorkflowMetadata(
            name=workflow_name,
            inputs=inputs,
            outputs=outputs,
            required_nodes=required_nodes,
            required_models=required_models,
            estimated_vram_mb=vram,
            category=category
        )
    
    def _normalize_graph(self, workflow_data: dict) -> dict:
        """Normalize workflow to get the node graph, converting UI format if needed."""
        # Check if already in API format with prompt wrapper
        if "prompt" in workflow_data and isinstance(workflow_data["prompt"], dict):
            return workflow_data["prompt"]
        
        # Check if it's UI format (has 'nodes' and 'links' arrays)
        if self._is_ui_format(workflow_data):
            return self._convert_ui_to_api(workflow_data)
        
        # Check if it's already the graph (dict of nodes with class_type)
        if all(isinstance(v, dict) and "class_type" in v for k, v in workflow_data.items() if isinstance(k, str) and k.isdigit()):
            return workflow_data
        
        return workflow_data
    
    def _is_ui_format(self, workflow_data: dict) -> bool:
        """Check if workflow is in UI format (with nodes/links arrays)."""
        return ("nodes" in workflow_data and 
                "links" in workflow_data and 
                isinstance(workflow_data.get("nodes"), list))
    
    def _convert_ui_to_api(self, ui_workflow: dict) -> dict:
        """
        Convert ComfyUI UI format workflow to API format.
        
        UI format has 'nodes' array with 'id', 'type', 'widgets_values' etc.
        API format has numbered dict keys with 'class_type' and 'inputs'.
        """
        nodes = ui_workflow.get("nodes", [])
        links = ui_workflow.get("links", [])
        
        # Build link map: link_id -> (from_node_id, from_slot)
        link_map = {}
        for link in links:
            if len(link) >= 6:
                link_id, from_node, from_slot = link[0], link[1], link[2]
                link_map[link_id] = {"from_node": from_node, "from_slot": from_slot}
        
        # Get widget names from node configuration if available
        # ComfyUI stores extra info sometimes in 'extra' -> 'groupNodes' or similar
        
        api_prompt = {}
        
        for node in nodes:
            node_id = str(node.get("id"))
            class_type = node.get("type")
            
            if not class_type:
                continue
            
            inputs = {}
            
            # UI nodes have 'widgets_values' array containing the values in order
            widgets_values = node.get("widgets_values", [])
            
            # Get widget order from the node's known input structure
            # We'll use common patterns to map widget_values to named inputs
            widget_names = self._get_widget_names_for_class(class_type, len(widgets_values))
            
            for i, val in enumerate(widgets_values):
                if val is not None and i < len(widget_names):
                    inputs[widget_names[i]] = val
            
            # Process inputs from links (these are connections to other nodes)
            node_inputs_config = node.get("inputs", [])
            for inp_config in node_inputs_config:
                if isinstance(inp_config, dict):
                    inp_name = inp_config.get("name")
                    link_id = inp_config.get("link")
                    if link_id and link_id in link_map:
                        link_info = link_map[link_id]
                        # In API format, connected inputs are [node_id, slot_index]
                        inputs[inp_name] = [str(link_info["from_node"]), link_info["from_slot"]]
            
            api_prompt[node_id] = {
                "class_type": class_type,
                "inputs": inputs
            }
        
        return api_prompt
    
    def _get_widget_names_for_class(self, class_type: str, num_widgets: int) -> list[str]:
        """Get ordered widget names for a class type based on known patterns."""
        # Common widget name mappings for known node types
        widget_mappings = {
            "CLIPTextEncode": ["text"],
            "KSampler": ["seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
            "KSamplerAdvanced": ["add_noise", "noise_seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "start_at_step", "end_at_step", "return_with_leftover_noise"],
            "EmptyLatentImage": ["width", "height", "batch_size"],
            "LoadImage": ["image", "upload"],
            "VHS_LoadVideo": ["video", "force_rate", "force_size", "custom_width", "custom_height", "frame_load_cap", "skip_first_frames", "select_every_nth"],
            "CheckpointLoaderSimple": ["ckpt_name"],
            "LoraLoader": ["lora_name", "strength_model", "strength_clip"],
            "SaveImage": ["filename_prefix"],
            "VHS_VideoCombine": ["frame_rate", "loop_count", "filename_prefix", "format", "pingpong", "save_output"],
            # Add more as needed - this is heuristic based
        }
        
        if class_type in widget_mappings:
            names = widget_mappings[class_type]
            # Pad with generic names if needed
            while len(names) < num_widgets:
                names.append(f"widget_{len(names)}")
            return names[:num_widgets]
        
        # Fallback: generate generic names
        return [f"widget_{i}" for i in range(num_widgets)]
    
    def _extract_inputs(self, graph: dict) -> list[WorkflowInput]:
        """Extract customizable inputs from the workflow."""
        inputs = []
        seen_input_keys = set()  # Prevent duplicates
        
        for node_id, node in graph.items():
            if not isinstance(node, dict):
                continue
                
            class_type = node.get("class_type", "")
            node_inputs = node.get("inputs", {})
            
            # Check known input patterns
            if class_type in self.INPUT_PATTERNS:
                pattern = self.INPUT_PATTERNS[class_type]
                for widget_name, input_type in pattern.get("inputs", {}).items():
                    if widget_name in node_inputs:
                        value = node_inputs[widget_name]
                        # Skip if it's a link reference like ["5", 0]
                        if isinstance(value, list):
                            continue
                            
                        # Create a readable name
                        name = self._create_input_name(class_type, widget_name, pattern)
                        
                        # Create unique key
                        input_key = f"{node_id}:{widget_name}"
                        if input_key in seen_input_keys:
                            continue
                        seen_input_keys.add(input_key)
                        
                        inputs.append(WorkflowInput(
                            name=name,
                            type=input_type,
                            node_id=node_id,
                            widget_name=widget_name,
                            default=value
                        ))
            else:
                # Generic detection for unknown node types
                self._extract_generic_inputs(node_id, class_type, node_inputs, inputs, seen_input_keys)
        
        return inputs
    
    def _extract_generic_inputs(self, node_id: str, class_type: str, 
                                 node_inputs: dict, inputs: list[WorkflowInput],
                                 seen_keys: set):
        """Extract inputs from unknown node types using heuristics."""
        for widget_name, value in node_inputs.items():
            # Skip link references
            if isinstance(value, list):
                continue
                
            input_key = f"{node_id}:{widget_name}"
            if input_key in seen_keys:
                continue
                
            # Infer type from value
            input_type = self._infer_input_type(widget_name, value)
            if input_type is None:
                continue  # Skip unknown types
                
            # Only include "interesting" inputs
            if widget_name.lower() in ['prompt', 'text', 'seed', 'steps', 'cfg', 
                                        'width', 'height', 'image', 'video', 'audio',
                                        'style_prompt', 'lyrics', 'positive', 'negative']:
                seen_keys.add(input_key)
                inputs.append(WorkflowInput(
                    name=f"{class_type} - {widget_name}",
                    type=input_type,
                    node_id=node_id,
                    widget_name=widget_name,
                    default=value
                ))
    
    def _infer_input_type(self, widget_name: str, value: Any) -> Union[str, None]:
        """Infer the input type from widget name and value."""
        name_lower = widget_name.lower()
        
        if 'prompt' in name_lower or 'text' in name_lower or 'lyrics' in name_lower:
            return 'text'
        elif 'seed' in name_lower:
            return 'number'
        elif name_lower in ['steps', 'cfg', 'width', 'height', 'length', 'frames']:
            return 'number'
        elif 'image' in name_lower:
            return 'image'
        elif 'video' in name_lower:
            return 'video'
        elif 'audio' in name_lower:
            return 'audio'
        elif isinstance(value, bool):
            return 'boolean'
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            return 'number'
        elif isinstance(value, str) and len(value) > 20:
            return 'text'
            
        return None
    
    def _create_input_name(self, class_type: str, widget_name: str, pattern: dict) -> str:
        """Create a user-friendly name for an input."""
        # Use predefined names if available
        names = pattern.get("names", [])
        
        if class_type == "CLIPTextEncode":
            if widget_name == "text":
                return "Prompt"  # Will be disambiguated by caller if needed
                
        # Default: clean up the widget name
        name = widget_name.replace("_", " ").title()
        return name
    
    def _extract_outputs(self, graph: dict) -> list[WorkflowOutput]:
        """Find output nodes in the workflow."""
        outputs = []
        
        for node_id, node in graph.items():
            if not isinstance(node, dict):
                continue
                
            class_type = node.get("class_type", "")
            
            if class_type in self.OUTPUT_NODES:
                output_type = self.OUTPUT_NODES[class_type]
                node_inputs = node.get("inputs", {})
                filename_prefix = node_inputs.get("filename_prefix", "")
                
                outputs.append(WorkflowOutput(
                    node_id=node_id,
                    output_type=output_type,
                    class_type=class_type,
                    filename_prefix=filename_prefix
                ))
        
        return outputs
    
    def _extract_required_nodes(self, graph: dict) -> list[str]:
        """Extract unique class_types used in the workflow."""
        class_types = set()
        
        for node in graph.values():
            if isinstance(node, dict) and "class_type" in node:
                class_types.add(node["class_type"])
        
        return list(class_types)
    
    def _extract_model_requirements(self, graph: dict) -> list[ModelRequirement]:
        """Extract model file requirements from workflow nodes.
        
        Uses MODEL_LOADER_WIDGETS mapping to detect model filenames
        in any node that has a matching input widget.
        """
        models = []
        seen = set()  # Prevent duplicates
        
        for node_id, node in graph.items():
            if not isinstance(node, dict):
                continue
            
            node_inputs = node.get("inputs", {})
            
            for widget_name, model_type in self.MODEL_LOADER_WIDGETS.items():
                if widget_name in node_inputs:
                    value = node_inputs[widget_name]
                    
                    # Only process string values (actual filenames)
                    # Skip link references like ["5", 0]
                    if isinstance(value, str) and value:
                        # Skip placeholder/empty values
                        if value.lower() in ["none", "", "default"]:
                            continue
                        
                        # Skip if we've already seen this model
                        if value.lower() in seen:
                            continue
                        seen.add(value.lower())
                        
                        models.append(ModelRequirement(
                            filename=value,
                            model_type=model_type,
                            node_id=node_id,
                            widget_name=widget_name
                        ))
        
        return models
    
    def extract_model_filenames(self, workflow_data: dict) -> list[str]:
        """Convenience method to get just the list of model filenames."""
        graph = self._normalize_graph(workflow_data)
        models = self._extract_model_requirements(graph)
        return [m.filename for m in models]

    def _estimate_vram(self, graph: dict) -> int:
        """Estimate required VRAM based on model nodes."""
        total_vram = 0
        
        for node in graph.values():
            if isinstance(node, dict):
                class_type = node.get("class_type", "")
                if class_type in self.MODEL_NODES:
                    total_vram += self.MODEL_NODES[class_type]
        
        return total_vram if total_vram > 0 else 4000  # Default 4GB
    
    def _infer_category(self, graph: dict, outputs: list[WorkflowOutput]) -> str:
        """Infer the workflow category based on outputs and nodes."""
        # Check outputs first
        output_types = [o.output_type for o in outputs]
        
        if "video" in output_types:
            return "video"
        elif "audio" in output_types:
            return "audio"
        elif "image" in output_types:
            return "image"
        
        # Check node types for hints
        for node in graph.values():
            if isinstance(node, dict):
                class_type = node.get("class_type", "").lower()
                if "video" in class_type or "wan" in class_type or "hunyuan" in class_type or "ltx" in class_type:
                    return "video"
                elif "audio" in class_type or "diffrhythm" in class_type or "music" in class_type:
                    return "audio"
        
        return "image"  # Default
    
    def get_primary_output_type(self, workflow_data: dict) -> str:
        """Get the primary output type of a workflow."""
        graph = self._normalize_graph(workflow_data)
        outputs = self._extract_outputs(graph)
        
        if not outputs:
            return "unknown"
            
        # Priority: video > audio > image > text
        for output_type in ["video", "audio", "image", "text"]:
            if any(o.output_type == output_type for o in outputs):
                return output_type
                
        return outputs[0].output_type
    
    def to_input_schema(self, workflow_data: dict, workflow_name: str = "Untitled") -> list[dict]:
        """
        Convert workflow to input schema format compatible with GenericWorkflowForm.
        
        Returns:
            List of input schema dicts with name, type, default, node_id, widget_name
        """
        metadata = self.analyze(workflow_data, workflow_name)
        
        schema = []
        for inp in metadata.inputs:
            schema.append({
                "name": inp.name,
                "type": inp.type,
                "default": inp.default,
                "node_id": inp.node_id,
                "widget_name": inp.widget_name
            })
        
        return schema
