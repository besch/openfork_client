import json
import os
import sys
import subprocess
import logging
import re
import requests
import uuid
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Add the parent directory (client) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from supabase import create_client, Client
from config import Config
from services.workflow_converter_service import WorkflowConverterService, WorkflowConversionError
from services.docker_manager import docker_manager


# --- Configuration ---
REPO_URL = "https://github.com/Comfy-Org/workflow_templates.git"
REPO_DIR_NAME = "workflow_templates"
LOCAL_REPO_PATH = Path(Config.ROOT_DIR) / REPO_DIR_NAME
TEMPLATES_DIR = LOCAL_REPO_PATH / "templates"
SCRIPTS_DIR = LOCAL_REPO_PATH / "scripts"
CUSTOM_NODE_LIST_URL = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json"
COMFYUI_BASE_URL = os.environ.get("COMFYUI_BASE_URL", "http://127.0.0.1:8188")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose logging from httpx and h2
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("h2").setLevel(logging.WARNING)

# --- Helper Functions ---
def load_json_file(path: Path) -> Union[Dict, List, None]:
    """Loads a JSON file from the given path."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"File not found at {path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in {path}")
        return None

def ensure_comfyui_running(max_wait: int = 60) -> bool:
    """
    Ensure ComfyUI container is running and ready before attempting conversion.
    
    Args:
        max_wait: Maximum seconds to wait for ComfyUI to be ready
        
    Returns:
        True if ComfyUI is ready, False otherwise
    """
    logger.info("Checking if ComfyUI container is running...")
    
    # Check if container is running
    try:
        if not docker_manager.is_container_running():
            logger.info("ComfyUI container not running. Starting it...")
            docker_manager.run_container(dependencies={'custom_node_urls': [], 'model_urls': []})
    except Exception as e:
        logger.error(f"Failed to start ComfyUI container: {e}")
        return False
    
    # Wait for ComfyUI to be ready
    import time
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{COMFYUI_BASE_URL}/system_stats", timeout=5)
            if response.status_code == 200:
                logger.info("ComfyUI is ready!")
                return True
        except:
            pass
        
        time.sleep(2)
    
    logger.error(f"ComfyUI did not become ready within {max_wait} seconds")
    return False


def ensure_converter_installed() -> bool:
    """
    Ensure workflow-to-api-converter-endpoint is installed in ComfyUI.
    
    Returns:
        True if installed, False otherwise
    """
    logger.info("Checking if workflow-to-api-converter-endpoint is installed...")
    
    container_name = docker_manager.get_container_name()
    if not container_name:
        logger.error("No running ComfyUI container found")
        return False
    
    # Check if the custom node directory exists
    check_cmd = [
        "bash", "-c",
        "[ -d /app/ComfyUI/custom_nodes/comfyui-workflow-to-api-converter-endpoint ] && echo 'EXISTS' || echo 'NOT_FOUND'"
    ]
    
    returncode, stdout, stderr = docker_manager.execute_in_container(check_cmd, timeout=10)
    
    if "EXISTS" in stdout:
        logger.info("✅ workflow-to-api-converter-endpoint is already installed")
        return True
    
    # Install the custom node
    logger.info("Installing workflow-to-api-converter-endpoint...")
    install_cmd = [
        "bash", "-c",
        "cd /app/ComfyUI/custom_nodes && "
        "git clone https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint"
    ]
    
    returncode, stdout, stderr = docker_manager.execute_in_container(install_cmd, timeout=60)
    
    if returncode == 0:
        logger.info("✅ Successfully installed workflow-to-api-converter-endpoint")
        
        # Restart container to load the custom node
        logger.info("Restarting ComfyUI to load the converter endpoint...")
        docker_manager.restart_container()
        
        # Wait for ComfyUI to be ready
        import time
        time.sleep(10)
        
        if not ensure_comfyui_running(max_wait=60):
            logger.error("ComfyUI did not restart successfully")
            return False
        
        return True
    else:
        logger.error(f"Failed to install converter: {stderr}")
        return False

def normalize_node_id(node_id) -> str:
    """Normalize node ID by converting to string and removing # prefix."""
    return str(node_id).lstrip('#')

def get_standard_nodes() -> set:
    """
    Returns a comprehensive list of core ComfyUI nodes.
    This list covers ComfyUI 0.3.x+ (as of October 2025).
    """
    
    # COMPREHENSIVE CORE NODE LIST (500+ nodes)
    CORE_NODES = {
        # === SAMPLERS & SCHEDULING ===
        "KSampler", "KSamplerAdvanced", "KSamplerSelect",
        "SamplerCustom", "SamplerCustomAdvanced",
        "BasicScheduler", "SDTurboScheduler", "BetaSamplingScheduler",
        "AlignYourStepsScheduler", "VPScheduler",
        
        # === GUIDERS (NEW IN 0.3.x) ===
        "CFGGuider", "DualCFGGuider", "CFGNorm",
        
        # === LOADERS ===
        "CheckpointLoaderSimple", "CheckpointLoader",
        "UNETLoader", "CLIPLoader", "DualCLIPLoader", "TripleCLIPLoader", "QuadrupleCLIPLoader",
        "VAELoader", "CLIPVisionLoader", "StyleModelLoader",
        "LoraLoader", "LoraLoaderModelOnly",
        "ControlNetLoader", "ControlNetApply", "ControlNetApplyAdvanced", "ControlNetApplySD3",
        "unCLIPCheckpointLoader", "GLIGENLoader",
        "ImageOnlyCheckpointLoader",
        
        # === CONDITIONING ===
        "CLIPTextEncode", "CLIPTextEncodeFlux", "CLIPTextEncodeSDXL", "CLIPTextEncodeSDXLRefiner",
        "CLIPSetLastLayer", "CLIPVisionEncode",
        "ConditioningAverage", "ConditioningCombine", "ConditioningConcat",
        "ConditioningSetArea", "ConditioningSetAreaPercentage", "ConditioningSetAreaStrength",
        "ConditioningSetMask", "ConditioningSetTimestepRange",
        "ConditioningZeroOut",
        "unCLIPConditioning",
        "GLIGENTextBoxApply",
        "InpaintModelConditioning", "InstructPixToPixConditioning",
        "T5TokenizerOptions",
        
        # === LATENTS ===
        "EmptyLatentImage", "EmptySD3LatentImage", "EmptyChromaRadianceLatentImage",
        "LatentUpscale", "LatentUpscaleBy", "LatentRotate", "LatentFlip",
        "LatentCrop", "LatentComposite", "LatentBlend", "LatentBatch", "LatentBatchSeedBehavior",
        "LatentFromBatch", "RepeatLatentBatch",
        "SetLatentNoiseMask",
        "VAEEncode", "VAEEncodeForInpaint", "VAEDecode", "VAEDecodeTiled",
        
        # === VIDEO LATENTS ===
        "EmptyLatentVideo", "EmptyLTXVLatentVideo", "EmptyMochiLatentVideo",
        "EmptyAceStepLatentAudio", "EmptyLatentAudio",
        "EmptyLatentHunyuan3Dv2",
        
        # === IMAGE OPERATIONS ===
        "LoadImage", "LoadImageMask", "SaveImage", "PreviewImage",
        "ImageScale", "ImageScaleBy", "ImageScaleToTotalPixels",
        "ImageUpscaleWithModel", "ImageInvert", "ImagePadForOutpaint",
        "ImageBatch", "ImageFromBatch", "RepeatImageBatch",
        "ImageBlur", "ImageQuantize", "ImageSharpen",
        "ImageCrop", "ImageCompositeMasked",
        "ImageBlend", "ImageColorToMask",
        "Canny", "GetImageSize",
        
        # === VIDEO OPERATIONS ===
        "LTXVScheduler", "LTXVConditioning", "LTXVImgToVideo",
        "GetVideoComponents", "TrimVideoLatent",
        
        # === MASKS ===
        "MaskToImage", "ImageToMask", "SolidMask",
        "InvertMask", "CropMask", "MaskComposite",
        "FeatherMask", "GrowMask", "MaskPreview",
        "SetLatentNoiseMask", "DrawMaskOnImage", "BlockifyMask",
        
        # === NOISE ===
        "RandomNoise", "DisableNoise", "SetFirstSigma",
        
        # === STYLE & EFFECTS ===
        "StyleModelApply", "DifferentialDiffusion",
        "FreeU", "FreeU_V2",
        "PerturbedAttentionGuidance",
        "SelfAttentionGuidance",
        
        # === UPSCALING ===
        "UpscaleModelLoader", "ImageUpscaleWithModel",
        
        # === HYPERNETWORKS ===
        "HypernetworkLoader",
        
        # === MODEL PATCHES ===
        "ModelSamplingDiscrete", "ModelSamplingContinuousEDM", "ModelSamplingContinuousV",
        "ModelSamplingStableCascade", "ModelSamplingSD3", "ModelSamplingFlux",
        "RescaleCFG",
        "PatchModelAddDownscale",
        "UNetTemporalAttentionMultiply",
        
        # === REFERENCE ===
        "ReferenceLatent",
        
        # === FLUX SPECIFIC ===
        "FluxGuidance",
        
        # === AUDIO ===
        "RecordAudio",
        
        # === 3D ===
        "VoxelToMesh",
        
        # === UTILITY ===
        "Note", "Reroute", "ReroutePrimitive",
        "PrimitiveNode", "PrimitiveInt", "PrimitiveFloat", "PrimitiveString", "PrimitiveBoolean",
        "ImageOnlyCheckpointSave", "ImageOnlyCheckpointLoader",
        "PointsEditor",
        "PreviewBridge",
        
        # === LATENT OPERATIONS ===
        "LatentAdd", "LatentSubtract", "LatentMultiply",
        "LatentInterpolate",
        
        # === MODEL MERGING ===
        "ModelMergeSimple", "ModelMergeBlocks", "ModelMergeAdd", "ModelMergeSubtract",
        
        # === CLIP OPERATIONS ===
        "CLIPMergeSimple", "CLIPMergeAdd", "CLIPMergeSubtract",
        "CLIPSave",
        
        # === CHECKPOINT OPERATIONS ===
        "CheckpointSave",
        
        # === GLIGEN ===
        "GLIGENTextBoxApply",
        
        # === PHOTOMAKER ===
        "PhotoMakerLoader", "PhotoMakerEncode",
        
        # === IPADAPTER ===
        "IPAdapterModelLoader", "IPAdapterApply", "IPAdapterApplyFaceID",
        
        # === ANIMATE DIFF ===
        "AnimateDiffLoader", "AnimateDiffCombine",
        
        # === STABLE CASCADE ===
        "StableCascade_EmptyLatentImage", "StableCascade_StageB_Conditioning", "StableCascade_StageC_VAEEncode",
        
        # === STABLE ZERO123 ===
        "StableZero123_Conditioning", "StableZero123_Conditioning_Batched",
        
        # === SV3D ===
        "SV3D_Conditioning",
        
        # === STABLE VIDEO DIFFUSION ===
        "SVD_img2vid_Conditioning",
        
        # === DEPRECATED BUT STILL PRESENT ===
        "SaveAnimatedWEBP", "SaveAnimatedPNG",
        
        # === EXEC CONTROL ===
        "ExecutionSwitch", "ExecutionBlocker",
        
        # === MATH ===
        "IntConstant", "FloatConstant", "StringConstant",
        
        # === IMAGE FILTERS ===
        "ImageFilterGaussianBlur", "ImageFilterEdgeEnhance", "ImageFilterSmooth",
        
        # === UNCLIP ===
        "unCLIPConditioning", "unCLIPCheckpointLoader",
        
        # === TOMESD ===
        "TomePatchModel",
        
        # === HYPERTILE ===
        "HyperTile",
        
        # === T2I ADAPTER ===
        "T2IAdapterLoader",
        
        # === LORA BLOCK WEIGHT ===
        "LoraLoaderBlockWeight",
    }
    
    return CORE_NODES


def get_standard_nodes_dynamic() -> set:
    """
    Attempts to fetch nodes from a running ComfyUI instance.
    Falls back to comprehensive static list if unavailable.
    """
    try:
        response = requests.get("http://127.0.0.1:8188/object_info", timeout=5)
        if response.status_code == 200:
            object_info = response.json()
            logger.info(f"[OK] Loaded {len(object_info)} nodes from running ComfyUI instance")
            return set(object_info.keys())
    except Exception as e:
        logger.debug(f"Could not fetch live node list: {e}")
    
    # Fallback to comprehensive static list
    core_nodes = get_standard_nodes()
    logger.info(f"Using comprehensive static core node list ({len(core_nodes)} nodes)")
    return core_nodes


def analyze_workflow_json(workflow_json: Dict, custom_node_registry: Dict[str, Dict]) -> Dict:
    """
    Analyzes a workflow JSON to extract models, custom nodes (with git URLs and node class_types),
    and a generated input schema.
    """
    # Hard mappings for nodes that are ACTUALLY custom nodes but hard to detect
    # NOTE: Do NOT include core nodes here (like StyleModelLoader, StyleModelApply which are now core)
    # NOTE: UI-only nodes (MarkdownNote, ShowText, Note) should be skipped during execution, not mapped
    HARD_NODE_MAPPINGS = {
        # Wan Video
        "WanCameraImageToVideo": "https://github.com/kijai/ComfyUI-WanVideoWrapper",
        "WanCameraEmbedding": "https://github.com/kijai/ComfyUI-WanVideoWrapper",
        "CreateVideo": "https://github.com/kijai/ComfyUI-WanVideoWrapper",
    }

    model_urls = set()
    custom_node_map: Dict[str, set] = {}  # Maps git_url -> set of class_types
    inputs = {}
    
    # Determine format: LiteGraph has a 'nodes' list, API format has a dict of nodes.
    is_api_format = 'nodes' not in workflow_json and isinstance(workflow_json, dict)
    
    nodes_to_process = []
    if is_api_format:
        # Convert API format to a list of nodes with their IDs
        for node_id, node_data in workflow_json.items():
            node_with_id = node_data.copy()
            node_with_id['id'] = node_id
            # API format uses 'class_type', LiteGraph uses 'type'. Standardize to 'type' for processing.
            if 'class_type' in node_with_id:
                node_with_id['type'] = node_with_id.pop('class_type')
            nodes_to_process.append(node_with_id)
    else:
        # For LiteGraph, use original format for analysis (no complex flattening needed)
        nodes_to_process = workflow_json.get('nodes', [])
        logger.info(f"Processing {len(nodes_to_process)} nodes from LiteGraph workflow.")

    # Use dynamic detection (tries live ComfyUI, falls back to comprehensive static list)
    standard_nodes = get_standard_nodes_dynamic()
    
    # Track unidentified nodes for debugging
    unidentified_nodes = set()
    
    # UI-only nodes that don't need dependencies
    UI_ONLY_NODES = {"Note", "MarkdownNote", "ShowText", "PreviewBridge"}

    for node in nodes_to_process:
        node_type = node.get('type')
        node_id = node.get('id')
        if not node_type:
            continue
        
        # Skip UI-only nodes - they don't execute and don't need dependencies
        if node_type in UI_ONLY_NODES:
            logger.debug(f"Skipping UI-only node: {node_type}")
            continue

        # Skip CORE nodes early
        if node_type in standard_nodes:
            continue

        # --- Custom Node Detection (Heuristic-based) ---
        found_repo_url = None
        
        # 1. Check hard mappings FIRST (highest confidence)
        if node_type in HARD_NODE_MAPPINGS:
            found_repo_url = HARD_NODE_MAPPINGS[node_type]
            logger.debug(f"Hard-mapped {node_type} -> {found_repo_url}")
        
        # 2. Check explicit node lists in registry (exact match)
        if not found_repo_url:
            for reg_title, reg_value in custom_node_registry.items():
                if node_type in reg_value.get('nodes', []) or node_type in reg_value.get('preemptions', []):
                    found_repo_url = reg_value.get('reference')
                    logger.debug(f"Registry exact match: {node_type} -> {found_repo_url}")
                    break

        # 3. Fuzzy matching on title (be conservative - only distinctive prefixes)
        if not found_repo_url:
            for reg_title, reg_value in custom_node_registry.items():
                # Extract a key from the title, e.g., "WAS" from "WAS Node Suite"
                title_key = reg_title.split(' ')[0].replace('ComfyUI-', '').replace('_', ' ').split(' ')[0]
                # Only match if the prefix is distinctive (3+ chars) and matches start of node name
                if len(title_key) >= 3 and node_type.startswith(title_key):
                    found_repo_url = reg_value.get('reference')
                    logger.debug(f"Prefix matched {node_type} -> {found_repo_url} (via {title_key})")
                    break
        
        if found_repo_url:
            if found_repo_url not in custom_node_map:
                custom_node_map[found_repo_url] = set()
            custom_node_map[found_repo_url].add(node_type)
        else:
            # Track but don't fail - might be a core node we don't know about
            unidentified_nodes.add(node_type)
            logger.debug(f"Could not identify repository for node: {node_type}")


        # --- Model URL Detection ---
        # This can be found in various places, widgets_values is a common one for LiteGraph
        widgets = node.get('widgets_values', [])
        if widgets:
            for widget_val in widgets:
                if isinstance(widget_val, str):
                    # Regex for markdown links: [filename.safetensors](url)
                    for match in re.finditer(r'\[(?:[^\]]+?\.safetensors)\]\(([^)]+)\)', widget_val):
                        model_urls.add(match.group(1))
        
        # Also check 'inputs' for API format which might contain model names in markdown
        if 'inputs' in node and isinstance(node['inputs'], dict):
            for input_name, input_value in node['inputs'].items():
                if isinstance(input_value, str):
                    for match in re.finditer(r'\[(?:[^\]]+?\.safetensors)\]\(([^)]+)\)', input_value):
                        model_urls.add(match.group(1))


        # --- Input Schema Detection (Heuristics) ---
        
        # Heuristic for LoadImage
        if node_type == 'LoadImage':
            node_title = node.get('title', f'Input Image {node_id}')
            key_name_base = f'input_image_{node_id}'
            if key_name_base not in inputs:
                inputs[key_name_base] = {
                    'type': 'image',
                    'description': node_title,
                    'node_type': 'LoadImage',
                    'field_name': 'image'
                }
            continue

        # General input detection
        widget_values = node.get('widgets_values', [])
        widget_names = []

        if 'inputs' in node:
            # This works for both API and LiteGraph formats if 'inputs' is a list of dicts
            if isinstance(node['inputs'], list):
                 widget_names = [i['name'] for i in node['inputs'] if isinstance(i, dict) and i.get('link') is None and 'name' in i]
            # For API format, inputs is a dict, we look at non-linked values
            elif isinstance(node['inputs'], dict):
                 for name, val in node['inputs'].items():
                     if not isinstance(val, list): # Links are lists [node_id, slot_index]
                         widget_names.append(name)
                         widget_values.append(val)


        if not widget_names or not widget_values:
            continue

        for i, input_name in enumerate(widget_names):
            if i >= len(widget_values):
                break
            
            input_value = widget_values[i]

            # Filter out non-user-configurable settings
            if input_name.lower().endswith(('_name', '.name')) or input_name.lower() in ['model', 'clip', 'vae', 'latent', 'image', 'pixels', 'control_after_generate', 'sampler_name', 'scheduler']:
                continue

            json_schema_type = 'string'
            if isinstance(input_value, int):
                json_schema_type = 'integer'
            elif isinstance(input_value, float):
                json_schema_type = 'number'
            elif isinstance(input_value, bool):
                json_schema_type = 'boolean'

            key_name_base = input_name
            if 'prompt' in key_name_base.lower() or 'text' in key_name_base.lower():
                title = node.get('title', '').lower()
                if 'negative' in title:
                    key_name_base = 'negative_prompt'
                else:
                    key_name_base = 'prompt'
            elif key_name_base.lower() == 'seed':
                key_name_base = 'seed'
            
            final_key_name = key_name_base
            counter = 1
            while final_key_name in inputs:
                final_key_name = f"{key_name_base}_{counter}"
                counter += 1
            
            inputs[final_key_name] = {
                'type': json_schema_type,
                'default': input_value,
                'description': f'{input_name} for {node_type}',
                'node_type': node_type,
                'field_name': input_name
            }

    # Log unidentified nodes for debugging (these might be new core nodes or need manual mapping)
    if unidentified_nodes:
        logger.warning(f"Could not identify {len(unidentified_nodes)} nodes: {unidentified_nodes}")
        logger.warning("These may be:")
        logger.warning("  1. New core nodes not in our static list (safe to ignore)")
        logger.warning("  2. Custom nodes needing manual hard mapping")
        logger.warning("  3. Deprecated/removed nodes")

    # Convert the custom_node_map to the desired JSONB format
    custom_node_dependencies = [
        {"url": url, "nodes": list(nodes)} for url, nodes in custom_node_map.items()
    ]

    return {
        "model_urls": list(model_urls),
        "custom_node_dependencies": custom_node_dependencies,
        "input_schema_properties": inputs
    }


class WorkflowSynchronizer:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.repo_path = LOCAL_REPO_PATH
        self.current_commit_hash: Optional[str] = None
        self.workflow_previews_bucket = "workflow-previews"
        self.custom_node_registry: Dict[str, Dict] = {}
        
        # Initialize workflow converter
        self.converter = WorkflowConverterService(comfyui_base_url=COMFYUI_BASE_URL)
    
    def sync_workflows(self):
        """Main synchronization logic with proper conversion."""
        logger.info("Starting workflow synchronization...")
        
        # Step 1: Clone/pull repository first (always needed)
        self.clone_or_pull_repository()
        
        # Step 2: Try to ensure ComfyUI is running (for API conversion)
        comfyui_available = False
        if ensure_comfyui_running():
            if ensure_converter_installed():
                comfyui_available = True
                logger.info("✅ ComfyUI is available for API conversion")
            else:
                logger.warning("⚠️ ComfyUI is running but converter is not installed. Will skip API conversion.")
        else:
            logger.warning("⚠️ ComfyUI is not available. Will skip API conversion for simple workflows.")
        
        # Step 3: Parse and process workflows
        parsed_workflows = self.parse_repository(comfyui_available)
        logger.info(f"Parsed and processed {len(parsed_workflows)} workflows")
        
        # Step 4: Sync to database
        self._sync_to_database(parsed_workflows)
        
        logger.info("Workflow synchronization completed.")
    
    def _count_subgraphs(self, workflow: Dict) -> int:
        """Count subgraph nodes (UUID types) in workflow."""
        if 'nodes' not in workflow:
            return 0
        
        count = 0
        for node in workflow.get('nodes', []):
            node_type = node.get('type', '')
            if self._is_uuid(node_type):
                count += 1
        
        return count
    
    def _is_uuid(self, value: str) -> bool:
        """Check if string is a valid UUID."""
        try:
            uuid.UUID(value)
            return True
        except (ValueError, AttributeError, TypeError):
            return False
    
    def _has_uuid_nodes_api_format(self, workflow: Dict) -> bool:
        """Check if API format workflow still has UUID nodes (shouldn't happen)."""
        if not isinstance(workflow, dict):
            return False
        
        for node_id, node_data in workflow.items():
            if isinstance(node_data, dict):
                class_type = node_data.get('class_type', '')
                if self._is_uuid(class_type):
                    return True
        
        return False

    def parse_repository(self, comfyui_available: bool = True) -> List[Dict[str, Any]]:
        """
        Parse repository and convert ALL workflows to API format.
        
        CRITICAL: ComfyUI's /prompt endpoint requires API format.
        LiteGraph format (with subgraphs) cannot be executed directly.
        The conversion flattens subgraphs into regular nodes.
        """
        workflows_data = []
        index_file = TEMPLATES_DIR / "index.json"
        
        if not index_file.exists():
            logger.error(f"Index file not found at {index_file}")
            return []
        
        index_data = load_json_file(index_file)
        if not isinstance(index_data, list):
            logger.error(f"Invalid format for {index_file}")
            return []
        
        self._get_custom_node_registry()
        
        # Track conversion statistics
        total_workflows = 0
        converted_workflows = 0
        skipped_workflows = 0
        failed_workflows = 0
        
        for category_entry in index_data:
            category_name = category_entry.get("category", "General")
            workflow_type_from_category = category_entry.get("type", "unknown")
            
            if category_name == "CLOSED SOURCE MODELS":
                logger.info(f"Skipping closed-source category: {category_name}")
                continue
            
            for template_entry in category_entry.get("templates", []):
                workflow_name = template_entry.get("name")
                if not workflow_name:
                    logger.warning(f"Skipping template with no name in category {category_name}")
                    continue
                
                total_workflows += 1
                
                workflow_json_file = TEMPLATES_DIR / f"{workflow_name}.json"
                if not workflow_json_file.exists():
                    logger.warning(f"Workflow JSON not found: {workflow_name}")
                    skipped_workflows += 1
                    continue
                
                workflow_json = load_json_file(workflow_json_file)
                if not workflow_json:
                    logger.warning(f"Could not load workflow: {workflow_name}")
                    skipped_workflows += 1
                    continue
                
                # === CHECK IF COMFYUI IS AVAILABLE ===
                if not comfyui_available:
                    logger.error(f"[ERROR] {workflow_name}: ComfyUI not available - cannot convert")
                    logger.error("   All workflows must be converted to API format")
                    logger.error("   Please ensure ComfyUI is running before syncing workflows")
                    skipped_workflows += 1
                    continue
                
                # === DETECT WORKFLOW FORMAT ===
                subgraph_count = self._count_subgraphs(workflow_json)
                is_litegraph = 'nodes' in workflow_json
                
                if subgraph_count > 0:
                    logger.info(f"📦 {workflow_name}: {subgraph_count} subgraphs detected - MUST convert")
                elif is_litegraph:
                    logger.info(f"📄 {workflow_name}: LiteGraph format - converting to API")
                else:
                    logger.info(f"✅ {workflow_name}: Already in API format - verifying")
                
                # === ALWAYS CONVERT (even if already API format, for consistency) ===
                logger.info(f"🔄 {workflow_name}: Converting to API format...")
                
                try:
                    # Use ComfyUI's native converter with retry
                    stored_workflow = self.converter.convert_with_retry(
                        workflow_json,
                        max_retries=3,
                        retry_delay=2
                    )
                    
                    # === VALIDATE CONVERSION ===
                    if not stored_workflow:
                        logger.error(f"[ERROR] {workflow_name}: Conversion produced empty workflow")
                        failed_workflows += 1
                        continue
                    
                    if not isinstance(stored_workflow, dict):
                        logger.error(f"[ERROR] {workflow_name}: Conversion produced invalid format: {type(stored_workflow)}")
                        failed_workflows += 1
                        continue
                    
                    # Check for suspiciously small workflows
                    if len(stored_workflow) <= 1:
                        logger.error(f"[ERROR] {workflow_name}: Workflow has only {len(stored_workflow)} nodes after conversion")
                        logger.error(f"   This usually means conversion failed or workflow is corrupt")
                        failed_workflows += 1
                        continue
                    
                    # CRITICAL: Check for UUID nodes (subgraphs not flattened)
                    has_uuid_nodes = False
                    for node_id, node_data in stored_workflow.items():
                        if isinstance(node_data, dict):
                            class_type = node_data.get('class_type', '')
                            try:
                                uuid.UUID(class_type)  # Will succeed if class_type is UUID
                                has_uuid_nodes = True
                                logger.error(f"   Found UUID node: {node_id} with class_type {class_type}")
                            except ValueError:
                                pass  # Not a UUID, this is good
                    
                    if has_uuid_nodes:
                        logger.error(f"[ERROR] {workflow_name}: UUID nodes found after conversion!")
                        logger.error(f"   Subgraphs were not properly flattened")
                        logger.error(f"   This workflow cannot be executed by ComfyUI's /prompt endpoint")
                        failed_workflows += 1
                        continue
                    
                    # === CONVERSION SUCCESSFUL ===
                    logger.info(f"✅ {workflow_name}: Successfully converted to API format ({len(stored_workflow)} nodes)")
                    converted_workflows += 1
                    
                    # Always use API format
                    workflow_format = "api"
                    
                except WorkflowConversionError as e:
                    logger.error(f"[ERROR] {workflow_name}: Conversion failed - {e}")
                    failed_workflows += 1
                    continue
                except Exception as e:
                    logger.error(f"[ERROR] {workflow_name}: Unexpected error - {e}", exc_info=True)
                    failed_workflows += 1
                    continue
                
                # === ANALYZE WORKFLOW (use original LiteGraph for analysis) ===
                analysis_result = analyze_workflow_json(workflow_json, self.custom_node_registry)
                
                # === GENERATE INPUT SCHEMA (use original LiteGraph) ===
                input_schema = self.extract_inputs_from_litegraph(workflow_json)
                
                # === DETERMINE TARGET ENTITY ===
                target_entity = "scene"
                if workflow_type_from_category == "audio":
                    target_entity = "audio_clip"
                elif workflow_type_from_category == "3d":
                    target_entity = "character"
                
                # === UPLOAD PREVIEW ===
                uploaded_preview_url = self._upload_preview_asset_logic(template_entry, workflow_name)
                
                # === BUILD WORKFLOW DATA ===
                workflows_data.append({
                    "source_repo_identifier": workflow_name,
                    "source_repo_commit_hash": self.current_commit_hash,
                    "name": template_entry.get("title", workflow_name),
                    "description": template_entry.get("description", ""),
                    "category": category_name,
                    "preview_image_url": uploaded_preview_url,
                    "workflow_json": stored_workflow,  # ALWAYS API FORMAT
                    "workflow_format": workflow_format,  # ALWAYS "api"
                    "input_schema": input_schema,
                    "workflow_type": workflow_type_from_category,
                    "target_entity": target_entity,
                    "hardware_requirements": {"gpu_vram": round(template_entry.get("vram", 0) / (1024**3))} if template_entry.get("vram") else {},
                    "custom_node_dependencies": analysis_result["custom_node_dependencies"],
                    "model_urls": analysis_result["model_urls"],
                    "is_public": True,
                })
        
        # === SUMMARY ===
        logger.info("=" * 60)
        logger.info("WORKFLOW SYNC SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total workflows found: {total_workflows}")
        logger.info(f"Successfully converted: {converted_workflows}")
        logger.info(f"Failed conversions: {failed_workflows}")
        logger.info(f"Skipped: {skipped_workflows}")
        logger.info(f"Ready for database: {len(workflows_data)}")
        logger.info("=" * 60)
        
        if failed_workflows > 0:
            logger.warning(f"⚠️  {failed_workflows} workflows failed to convert")
            logger.warning("   These workflows will NOT be synced to the database")
            logger.warning("   Check the logs above for specific error messages")
        
        return workflows_data

    
    
    def _unique_key(self, props: Dict, base: str, counter: Dict) -> str:
        key = base
        c = counter.get(base, 0)
        while key in props:
            c += 1
            key = f"{base}_{c}"
        counter[base] = c
        return key

    def extract_inputs_from_litegraph(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts user-editable inputs -> JSON Schema for dynamic form with proper node_type and field_name."""
        schema = {'type': 'object', 'properties': {}}
        props: Dict[str, Any] = schema['properties']
        counter: Dict[str, int] = {}
        
        defs = workflow.get('definitions', {}).get('subgraphs', [])
        
        # Track which nodes we've processed to avoid duplicates
        processed_nodes = set()
        
        # Build a map of all nodes by ID for quick lookup
        all_nodes = {}
        for node in workflow['nodes']:
            node_id = node.get('id')
            if node_id is not None:
                all_nodes[node_id] = node
        
        for node in workflow['nodes']:
            node_id = node.get('id')
            ntype = node['type']
            widgets = node.get('widgets_values', [])
            title = node.get('title', '').lower()
            
            # 1. SUBGRAPHS (90% win: wan2, qwen, ace, video)
            try:
                uuid.UUID(ntype)  # Is UUID?
                subgraph = next((s for s in defs if s['id'] == ntype), None)
                if subgraph:
                    # For subgraphs, we need to map inputs to the internal nodes
                    # Get the subgraph's input definitions
                    for i, inp in enumerate(subgraph.get('inputs', [])):
                        name = inp['name'].lower().replace(' ', '_')
                        itype = inp['type'].lower()
                        default = widgets[i] if i < len(widgets) else None
                        
                        key = self._unique_key(props, name, counter)
                        
                        # Try to find which internal node this connects to
                        internal_link_id = inp.get('link')
                        target_node_type = None
                        target_field_name = None
                        
                        if internal_link_id:
                            # Find the link in the subgraph
                            for link in subgraph.get('links', []):
                                if isinstance(link, list) and len(link) > 4 and link[0] == internal_link_id:
                                    target_node_id = link[3]
                                    target_slot = link[4]
                                    
                                    # Find the target node
                                    for internal_node in subgraph.get('nodes', []):
                                        if internal_node.get('id') == target_node_id:
                                            target_node_type = internal_node.get('type')
                                            
                                            # Get the input name from the target node
                                            node_inputs = internal_node.get('inputs', [])
                                            if target_slot < len(node_inputs):
                                                target_field_name = node_inputs[target_slot].get('name')
                                            break
                                    break
                        
                        # Fallback logic for when we can't determine the target node
                        if not target_node_type or not target_field_name:
                            if itype == 'image':
                                target_node_type = 'LoadImage'
                                target_field_name = 'image'
                            elif itype in ['string', 'text'] or any(kw in name for kw in ['prompt', 'text', 'tags', 'lyrics']):
                                target_node_type = 'CLIPTextEncode'
                                target_field_name = 'text'
                            elif name in ['width', 'height']:
                                target_node_type = 'EmptySD3LatentImage'
                                target_field_name = name
                            elif name == 'batch_size':
                                target_node_type = 'EmptySD3LatentImage'
                                target_field_name = 'batch_size'
                            elif name == 'steps':
                                target_node_type = 'KSampler'
                                target_field_name = 'steps'
                            elif name == 'seed':
                                target_node_type = 'KSampler'
                                target_field_name = 'seed'
                            else:
                                # Generic fallback
                                target_node_type = 'PrimitiveNode'
                                target_field_name = name
                        
                        if itype == 'image':
                            props[key] = {
                                'type': 'string', 
                                'format': 'uri', 
                                'default': '', 
                                'description': f'Upload {name.title()}',
                                'node_type': target_node_type,
                                'field_name': target_field_name
                            }
                        elif itype in ['string', 'text'] or any(kw in name for kw in ['prompt', 'text', 'tags', 'lyrics']):
                            props[key] = {
                                'type': 'string', 
                                'default': str(default) if default else '', 
                                'description': name.title(),
                                'node_type': target_node_type,
                                'field_name': target_field_name
                            }
                        elif itype in ['int', 'number'] or any(kw in name for kw in ['width', 'height', 'steps', 'seed', 'length', 'seconds', 'batch']):
                            # Safe numeric conversion with fallback
                            try:
                                numeric_default = float(default) if default is not None and isinstance(default, (int, float)) else None
                            except (ValueError, TypeError):
                                numeric_default = None
                            
                            if numeric_default is None:
                                # Set sensible defaults based on field name
                                if 'width' in name or 'height' in name:
                                    numeric_default = 512.0
                                elif 'steps' in name:
                                    numeric_default = 20.0
                                elif 'seed' in name:
                                    numeric_default = 0.0
                                else:
                                    numeric_default = 0.0
                            
                            props[key] = {
                                'type': 'number', 
                                'default': numeric_default, 
                                'description': name.title(),
                                'node_type': target_node_type,
                                'field_name': target_field_name
                            }
                        elif itype == 'boolean':
                            props[key] = {
                                'type': 'boolean', 
                                'default': bool(default) if default is not None else False, 
                                'description': name.title(),
                                'node_type': target_node_type,
                                'field_name': target_field_name
                            }
                    processed_nodes.add(node_id)
                    continue
            except (ValueError, AttributeError):
                pass
            
            # Skip if already processed
            if node_id in processed_nodes:
                continue
            
            # 2. REGULAR NODES
            if ntype == 'CLIPTextEncode' and widgets:
                key = 'positive_prompt' if 'pos' in title or 'positive' in title else 'negative_prompt'
                key = self._unique_key(props, key, counter)
                props[key] = {
                    'type': 'string', 
                    'default': str(widgets[0]), 
                    'description': key.replace('_', ' ').title(),
                    'node_type': 'CLIPTextEncode',
                    'field_name': 'text'
                }
                processed_nodes.add(node_id)
            elif ntype in ['EmptyLatentImage', 'EmptySD3LatentImage'] and len(widgets) >= 2:
                # Safe numeric conversion for dimensions
                try:
                    width_val = float(widgets[0]) if isinstance(widgets[0], (int, float)) else 512
                except (ValueError, TypeError):
                    width_val = 512
                
                try:
                    height_val = float(widgets[1]) if isinstance(widgets[1], (int, float)) else 512
                except (ValueError, TypeError):
                    height_val = 512
                
                if 'width' not in props:
                    props['width'] = {
                        'type': 'number', 
                        'default': width_val, 
                        'description': 'Width',
                        'node_type': ntype,
                        'field_name': 'width'
                    }
                if 'height' not in props:
                    props['height'] = {
                        'type': 'number', 
                        'default': height_val, 
                        'description': 'Height',
                        'node_type': ntype,
                        'field_name': 'height'
                    }
                if len(widgets) >= 3 and 'batch_size' not in props:
                    try:
                        batch_val = float(widgets[2]) if isinstance(widgets[2], (int, float)) else 1
                    except (ValueError, TypeError):
                        batch_val = 1
                    props['batch_size'] = {
                        'type': 'number',
                        'default': batch_val,
                        'description': 'Batch Size',
                        'node_type': ntype,
                        'field_name': 'batch_size'
                    }
                processed_nodes.add(node_id)
            elif ntype == 'KSampler' and len(widgets) >= 3:
                # KSampler typically has: seed, steps, cfg, sampler_name, scheduler, denoise
                if 'seed' not in props and len(widgets) > 0:
                    # Seed can be a number or "randomize"/"fixed"
                    try:
                        seed_val = float(widgets[0]) if isinstance(widgets[0], (int, float)) else 0
                    except (ValueError, TypeError):
                        seed_val = 0
                    props['seed'] = {
                        'type': 'number',
                        'default': seed_val,
                        'description': 'Seed',
                        'node_type': 'KSampler',
                        'field_name': 'seed'
                    }
                if 'steps' not in props and len(widgets) > 1:
                    # Steps should be numeric, but validate
                    try:
                        steps_val = float(widgets[1]) if isinstance(widgets[1], (int, float)) else 20
                    except (ValueError, TypeError):
                        steps_val = 20
                    props['steps'] = {
                        'type': 'number',
                        'default': steps_val,
                        'description': 'Steps',
                        'node_type': 'KSampler',
                        'field_name': 'steps'
                    }
                processed_nodes.add(node_id)
            elif ntype == 'LoadImage' and widgets:
                key = self._unique_key(props, 'input_image', counter)
                props[key] = {
                    'type': 'string', 
                    'format': 'uri', 
                    'default': '', 
                    'description': f'Upload Image ({node.get("title", "Input")})',
                    'node_type': 'LoadImage',
                    'field_name': 'image'
                }
                processed_nodes.add(node_id)
            elif 'TextToImage' in ntype or 't2i' in ntype.lower():
                if len(widgets) > 1:
                    props['prompt'] = {
                        'type': 'string', 
                        'default': str(widgets[1]), 
                        'description': 'Prompt',
                        'node_type': ntype,
                        'field_name': 'prompt'
                    }
                    if len(widgets) > 2 and widgets[2]:
                        props['negative_prompt'] = {
                            'type': 'string', 
                            'default': str(widgets[2]), 
                            'description': 'Negative Prompt',
                            'node_type': ntype,
                            'field_name': 'negative_prompt'
                        }
                processed_nodes.add(node_id)
        
        return schema


    def _run_git_command(self, command: List[str]) -> str:
        """Runs a git command in the local repository directory."""
        try:
            result = subprocess.run(
                ["git"] + command,
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {' '.join(command)}\nStdout: {e.stdout}\nStderr: {e.stderr}")
            raise

    def clone_or_pull_repository(self):
        """Clones the repository if it doesn't exist, otherwise pulls the latest changes."""
        if not self.repo_path.exists():
            logger.info(f"Cloning {REPO_URL} into {self.repo_path}...")
            subprocess.run(["git", "clone", REPO_URL, self.repo_path], check=True)
            logger.info("Repository cloned successfully.")
        else:
            logger.info(f"Pulling latest changes for {self.repo_path}...")
            self._run_git_command(["pull", "origin", "main"])
            logger.info("Repository updated successfully.")
        
        # Get current commit hash
        self.current_commit_hash = self._run_git_command(["rev-parse", "HEAD"])
        logger.info(f"Current repository commit hash: {self.current_commit_hash}")

    def _get_custom_node_registry(self):
        """Fetches and parses the custom-node-list.json from ComfyUI-Manager's GitHub repo."""
        logger.info(f"Fetching custom node registry from {CUSTOM_NODE_LIST_URL}...")
        try:
            response = requests.get(CUSTOM_NODE_LIST_URL)
            response.raise_for_status()
            custom_node_list = response.json()
            
            registry = {}
            for node_entry in custom_node_list.get("custom_nodes", []):
                title = node_entry.get('title')
                git_url = node_entry.get('reference')
                
                if not title or not git_url:
                    continue

                registry[title] = {
                    "reference": git_url,
                    "author": node_entry.get('author'),
                    "nodes": node_entry.get('nodes', []),
                    "preemptions": node_entry.get('preemptions', [])
                }

            self.custom_node_registry = registry
            logger.info(f"Loaded {len(self.custom_node_registry)} custom node entries from registry.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch custom node list: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse custom node list JSON: {e}")

        if not self.custom_node_registry:
            logger.critical("Custom node registry is empty. Check your internet connection.")
            sys.exit(1)

    def _upload_preview_asset(self, file_path: Path, destination_name: str) -> Optional[str]:
        """Uploads a preview asset to Supabase Storage and returns its public URL."""
        if not file_path or not file_path.exists():
            logger.warning(f"Preview asset file not found: {file_path}")
            return None
            
        logger.info(f"Uploading preview asset: {file_path} -> {destination_name}")

        try:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            
            # Determine content type
            content_type = "application/octet-stream"
            if file_path.suffix.lower() == ".webp":
                content_type = "image/webp"
            elif file_path.suffix.lower() == ".mp3":
                content_type = "audio/mpeg"
            elif file_path.suffix.lower() == ".mp4":
                content_type = "video/mp4"
            elif file_path.suffix.lower() == ".png":
                content_type = "image/png"
            elif file_path.suffix.lower() == ".jpg" or file_path.suffix.lower() == ".jpeg":
                content_type = "image/jpeg"

            try:
                self.supabase.storage.from_(self.workflow_previews_bucket).upload(
                    path=destination_name,
                    file=file_bytes,
                    file_options={"contentType": content_type}
                )
            except Exception as e:
                if "The resource already exists" in str(e):
                    logger.info(f"Asset {destination_name} already exists")
                else:
                    logger.error(f"Upload failed for {file_path.name}: {str(e)}")
                    return None

            # Get public URL after successful upload or if file already exists
            try:
                public_url = self.supabase.storage.from_(self.workflow_previews_bucket).get_public_url(destination_name)
                logger.info(f"Public URL for {file_path.name}: {public_url}")
                return public_url
            except Exception as e:
                logger.error(f"Failed to get public URL for {destination_name}: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error uploading preview asset {file_path.name}: {e}")
            return None

    def _upload_preview_asset_logic(self, template_entry: Dict, workflow_name: str) -> Optional[str]:
        """Helper method to upload preview assets with common logic."""
        uploaded_preview_url = None
        preview_asset_suffix = template_entry.get("mediaSubtype", "webp")

        preview_asset_path_1 = TEMPLATES_DIR / f"{workflow_name}-1.{preview_asset_suffix}"
        if preview_asset_path_1.exists():
            uploaded_preview_url = self._upload_preview_asset(preview_asset_path_1, f"{workflow_name}-1.{preview_asset_suffix}")

        if not uploaded_preview_url:
            preview_asset_path_no_num = TEMPLATES_DIR / f"{workflow_name}.{preview_asset_suffix}"
            if preview_asset_path_no_num.exists():
                uploaded_preview_url = self._upload_preview_asset(preview_asset_path_no_num, f"{workflow_name}.{preview_asset_suffix}")

        return uploaded_preview_url

    def _sync_to_database(self, parsed_workflows: List[Dict[str, Any]]):
        """Synchronizes the parsed workflows to the Supabase database."""
        logger.info("Starting database synchronization...")
        
        # Fetch existing workflows from the database
        try:
            response = self.supabase.table("workflow_templates").select("id, source_repo_identifier, source_repo_commit_hash").execute()
            existing_workflows = {wf["source_repo_identifier"]: wf for wf in response.data if wf["source_repo_identifier"]}
        except Exception as e:
            logger.error(f"Failed to fetch existing workflows from database: {str(e)}")
            return
        
        for workflow_data in parsed_workflows:
            identifier = workflow_data["source_repo_identifier"]
            commit_hash = workflow_data["source_repo_commit_hash"]
            
            db_payload = {
                "source_repo_identifier": identifier,
                "source_repo_commit_hash": commit_hash,
                "name": workflow_data["name"],
                "description": workflow_data["description"],
                "category": workflow_data["category"],
                "preview_image_url": workflow_data["preview_image_url"],
                "workflow_json": workflow_data["workflow_json"],
                "workflow_format": workflow_data.get("workflow_format", "api"),  # NEW FIELD!
                "input_schema": workflow_data["input_schema"],
                "workflow_type": workflow_data["workflow_type"],
                "target_entity": workflow_data["target_entity"],
                "hardware_requirements": workflow_data["hardware_requirements"],
                "custom_node_dependencies": workflow_data["custom_node_dependencies"],
                "model_urls": workflow_data["model_urls"],
                "is_public": workflow_data["is_public"],
            }

            if identifier in existing_workflows:
                try:
                    logger.info(f"Updating workflow: {identifier}")
                    self.supabase.table("workflow_templates").update(db_payload).eq("source_repo_identifier", identifier).execute()
                except Exception as e:
                    logger.error(f"Failed to update workflow {identifier}: {str(e)}")
            else:
                try:
                    logger.info(f"Inserting new workflow: {identifier}")
                    self.supabase.table("workflow_templates").insert(db_payload).execute()
                except Exception as e:
                    logger.error(f"Failed to insert workflow {identifier}: {str(e)}")
        
        logger.info("Database synchronization completed.")



if __name__ == "__main__":
    supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not supabase_key:
        logger.error("Supabase URL or Service Role Key not found. Please set NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables.")
        sys.exit(1)

    synchronizer = WorkflowSynchronizer(supabase_url, supabase_key)
    synchronizer.sync_workflows()