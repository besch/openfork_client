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

logger = logging.getLogger(__name__)


class WorkflowConversionError(Exception):
    """Raised when workflow conversion fails."""
    pass


class WorkflowConverterService:
    """
    Service for converting ComfyUI workflows from LiteGraph format to API format
    using the workflow-to-api-converter-endpoint custom node.
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
        if not self._ensure_converter_installed():
            raise WorkflowConversionError(
                "workflow-to-api-converter-endpoint is not installed or ComfyUI is not running. "
                "Install it with: cd ComfyUI/custom_nodes && "
                "git clone https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint"
            )
        
        logger.info("Converting workflow using ComfyUI's native converter...")
        
        try:
            # Send the workflow to the converter endpoint
            # The endpoint expects the full workflow JSON in the body
            response = requests.post(
                self.convert_endpoint,
                json=workflow,  # Send the entire workflow as JSON body
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
            
            logger.info(f"Successfully converted workflow: {len(api_workflow)} nodes")
            return api_workflow
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error during conversion: {e}"
            if e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg += f"\nDetail: {error_detail}"
                except:
                    error_msg += f"\nResponse: {e.response.text}"
            raise WorkflowConversionError(error_msg)
            
        except requests.exceptions.Timeout:
            raise WorkflowConversionError(f"Conversion timed out after {self.timeout} seconds")
            
        except requests.exceptions.ConnectionError:
            raise WorkflowConversionError(f"Cannot connect to ComfyUI at {self.comfyui_base_url}")
            
        except Exception as e:
            raise WorkflowConversionError(f"Unexpected error during conversion: {e}")
    
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