import os

def generate_manifest():
    # Directories to include
    include_dirs = ["services", "utils", "comfyui-storage", "workflows"]
    # Files to include in root
    include_files = ["cli.py", "dgn_client.py", "config.py", "exceptions.py", "requirements.txt", "start_cloud.sh"]
    
    manifest = []
    
    # Root files
    for f in include_files:
        if os.path.exists(f):
            manifest.append(f)
            
    # Subdirectories
    for d in include_dirs:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if (f.endswith(".py") or f.endswith(".json")) and "__pycache__" not in root:
                        rel_path = os.path.relpath(os.path.join(root, f), ".")
                        manifest.append(rel_path.replace("\\", "/"))
    
    # Sort for consistency
    manifest.sort()
    
    with open("manifest.txt", "w") as f:
        f.write("\n".join(manifest))
    
    print(f"Generated manifest.txt with {len(manifest)} files.")

if __name__ == "__main__":
    generate_manifest()
