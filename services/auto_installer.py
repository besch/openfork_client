

# -------------------------------------------------------
# Install via ComfyUI-Manager CLI
# -------------------------------------------------------
def manager_install_custom_node(repo_url: str):
    if not os.path.exists(MANAGER_CLI):
        logging.warning("ComfyUI-Manager CLI not present")
        return False
    try:
        subprocess.run(["python3", MANAGER_CLI, "--install-custom-node", repo_url], check=True)
        # give ComfyUI a moment to load new node files
        time.sleep(2)
        comfy_refresh_nodes()
        return True
    except subprocess.CalledProcessError as e:
        logging.warning(f"Manager CLI install failed for {repo_url}: {e}")
        return False

def manager_install_model(url: str):
    if not os.path.exists(MANAGER_CLI):
        return False
    try:
        subprocess.run(["python3", MANAGER_CLI, "--install-model", url], check=True)
        time.sleep(1)
        comfy_refresh_models()
        return True
    except subprocess.CalledProcessError as e:
        logging.warning(f"Manager CLI model install failed for {url}: {e}")
        return False

# -------------------------------------------------------
# Top-level orchestration: given a workflow template dict,
# install nodes & models (safe, idempotent).
# -------------------------------------------------------
def auto_install_all(template: dict):
    installed_nodes = []
    installed_models = []

    # 1. NODES: EXPLICIT URLs (from sync!)
    for dep in template.get('custom_node_dependencies', []):
        url = dep.get('url')
        if url and manager_install_custom_node(url):
            installed_nodes.append(url)

    # 2. HARD-MAPS (backup for undocumented)
    # This part requires a get_missing_nodes function, which is not defined yet.
    # For now, I will skip this part and assume all nodes are explicitly defined or handled by manager_install_custom_node.
    # If get_missing_nodes is needed, it would involve querying ComfyUI for installed nodes and comparing.
    # The user's prompt implies that the custom_node_dependencies from workflow_sync should be sufficient.

    # 3. MODELS: MANAGER CLI (handles ALL!)
    for url in template.get('model_urls', []):
        if manager_install_model(url):  # CivitAI/HF/GH/direct → MAGIC!
            installed_models.append(url)

    comfy_refresh_nodes()
    comfy_refresh_models()
    return installed_nodes, installed_models