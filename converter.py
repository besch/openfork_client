#!/usr/bin/env python3

"""
Workflow Converter using Python LiteGraph Logic
===============================================

This script converts ComfyUI workflows from LiteGraph format to API format
using the same logic as the JavaScript version, but implemented in Python.
"""

import json
import sys
import os
from pathlib import Path

def load_workflow(file_path):
    """Load workflow from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load workflow {file_path}: {e}")
        return None

def save_workflow(workflow, file_path):
    """Save workflow to JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(workflow, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Failed to save workflow {file_path}: {e}")
        return False

def convert_litegraph_to_api(workflow):
    """
    Convert LiteGraph format to API format using Python logic.
    Based on the JavaScript version but implemented in Python.
    """
    if not workflow or not isinstance(workflow, dict):
        print("Invalid workflow data")
        return None
    
    # Check if already in API format
    if "nodes" not in workflow or not isinstance(workflow["nodes"], list):
        print("Workflow appears to be in API format already")
        return workflow
    
    print("Processing LiteGraph workflow...")
    
    # Build link map
    link_map = {}
    if "links" in workflow and isinstance(workflow["links"], list):
        for link in workflow["links"]:
            if isinstance(link, list) and len(link) >= 5:
                link_id = link[0]
                source_node = str(link[1]).replace("#", "")
                source_slot = link[2]
                target_node = str(link[3]).replace("#", "")
                target_slot = link[4]
                link_map[link_id] = {
                    "source_node": source_node,
                    "source_slot": source_slot,
                    "target_node": target_node,
                    "target_slot": target_slot,
                }
    
    print(f"Found {len(link_map)} links to process")
    
    # Convert nodes
    converted = {}
    converted_count = 0
    
    for node in workflow["nodes"]:
        if not isinstance(node, dict):
            continue
        
        node_id = str(node.get("id", "")).replace("#", "")
        if not node_id:
            continue
        
        node_type = node.get("type")
        if not node_type:
            continue
        
        # Skip subgraphs (UUIDs)
        if len(node_type) == 36 and "-" in node_type:
            print(f"Skipping subgraph: {node_id} ({node_type})")
            continue
        
        api_node = {
            "class_type": node_type,
            "inputs": {},
        }
        
        widgets = node.get("widgets_values", [])
        inputs = node.get("inputs", [])
        
        # Map inputs
        widget_idx = 0
        for inp in inputs:
            if not isinstance(inp, dict):
                continue
            
            input_name = inp.get("name")
            if not input_name:
                continue
            
            link_id = inp.get("link")
            
            # Connected input
            if link_id is not None and link_id in link_map:
                link_info = link_map[link_id]
                source_node = link_info["source_node"]
                source_slot = link_info["source_slot"]
                api_node["inputs"][input_name] = [source_node, source_slot]
                print(f"  {node_id} ({node_type}): {input_name} -> [{source_node}, {source_slot}]")
            
            # Unconnected input (use widget)
            elif widget_idx < len(widgets):
                widget_value = widgets[widget_idx]
                
                # Special handling for SaveImage 'images' input
                if node_type == "SaveImage" and input_name == "images":
                    print(f"  WARNING: SaveImage {node_id} - skipping widget assignment to 'images' input")
                    widget_idx += 1
                    continue
                
                # Normalize widget value
                if isinstance(widget_value, dict) and "name" in widget_value:
                    api_node["inputs"][input_name] = widget_value["name"]
                    print(f"  {node_id} ({node_type}): {input_name} = {widget_value['name']} (normalized)")
                else:
                    api_node["inputs"][input_name] = widget_value
                    print(f"  {node_id} ({node_type}): {input_name} = {widget_value}")
                widget_idx += 1
        
        # Add filename_prefix for SaveImage
        if node_type == "SaveImage" and "filename_prefix" not in api_node["inputs"]:
            api_node["inputs"]["filename_prefix"] = widgets[0] if widgets else "ComfyUI"
            print(f"  {node_id} (SaveImage): Added filename_prefix = {api_node['inputs']['filename_prefix']}")
        
        converted[node_id] = api_node
        converted_count += 1
    
    print(f"Converted {converted_count} nodes")
    return converted

def convert_workflow(input_path, output_path):
    """Main conversion function."""
    print(f"Converting: {input_path} -> {output_path}")
    
    # Load input workflow
    workflow = load_workflow(input_path)
    if not workflow:
        return False
    
    # Convert workflow
    converted = convert_litegraph_to_api(workflow)
    if not converted:
        return False
    
    # Save converted workflow
    if save_workflow(converted, output_path):
        print("✓ Successfully converted workflow")
        return True
    else:
        return False

def main():
    """Main function."""
    args = sys.argv[1:]
    
    if len(args) != 2:
        print("Usage: python convert_with_litegraph_python.py <input.json> <output.json>")
        print()
        print("Example:")
        print("  python convert_with_litegraph_python.py workflow.json converted.json")
        sys.exit(1)
    
    input_path, output_path = args
    
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        sys.exit(1)
    
    success = convert_workflow(input_path, output_path)
    
    if success:
        print("\n✓ Conversion completed successfully!")
    else:
        print("\n✗ Conversion failed")
        print("\nAlternative solutions:")
        print("1. Use ComfyUI web interface to manually export as API format")
        print("2. Try different conversion approaches")
        print("3. Check workflow structure for compatibility")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()