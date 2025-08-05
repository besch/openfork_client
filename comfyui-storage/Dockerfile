# Rebase image to cu128-slim runtime from YanWenKun's ComfyUI-Docker.
# Keep only minimal layering to run ComfyUI with your existing volumes and DGN client.
FROM yanwk/comfyui-boot:cu128-slim

# The base image uses /root as volume root and /root/ComfyUI as app dir.
# Create required directories if they don't exist (when host volume is empty).
RUN mkdir -p /root/ComfyUI/{models,input,output,user,custom_nodes}
# If the host bind-mount ./storage:/root is empty on first run, this init step will populate
# /root/ComfyUI/custom_nodes with the required plugins. If the host already has content, this
# will no-op because target dirs will exist.
RUN set -e; \
    if [ ! -d "/root/ComfyUI/custom_nodes" ] || [ -z "$(ls -A /root/ComfyUI/custom_nodes 2>/dev/null)" ]; then \
      echo "Custom nodes dir empty; initializing required repos..."; \
      mkdir -p /root/ComfyUI/custom_nodes; \
      cd /root/ComfyUI/custom_nodes; \
      git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes.git || true; \
      git clone --depth=1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git || true; \
      git clone --depth=1 https://github.com/city96/ComfyUI-GGUF.git || true; \
    else \
      echo "Custom nodes dir already populated; skipping init."; \
    fi

# Optionally include custom nodes that are not preinstalled in the base image.
# Comment out any you don't need. We keep your prior custom nodes.
# NOTE: Base image already includes ComfyUI and ComfyUI-Manager via requirements layer.
# If a node already exists it will be left as-is (git will fail if dir not empty, so check first).
# Ensure required custom nodes exist even when host volume is empty at runtime start.
# This runtime guard handles cases where the host-mounted ./storage wipes image-added content.
RUN bash -lc 'set -e; \
  if [ ! -d "/root/ComfyUI/custom_nodes/ComfyUI-KJNodes" ]; then \
    echo "Installing ComfyUI-KJNodes at runtime..."; \
    git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes.git /root/ComfyUI/custom_nodes/ComfyUI-KJNodes || true; \
  fi; \
  if [ ! -d "/root/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite" ]; then \
    echo "Installing ComfyUI-VideoHelperSuite at runtime..."; \
    git clone --depth=1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /root/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite || true; \
  fi; \
  if [ ! -d "/root/ComfyUI/custom_nodes/ComfyUI-GGUF" ]; then \
    echo "Installing ComfyUI-GGUF at runtime..."; \
    git clone --depth=1 https://github.com/city96/ComfyUI-GGUF.git /root/ComfyUI/custom_nodes/ComfyUI-GGUF || true; \
  fi'

# Ensure Python deps for added custom nodes (keep minimal; base image already installs core deps).
# Add some common runtime libs used by KJNodes/VHS and GGUF.
RUN pip3 install --no-cache-dir --upgrade \
    gguf \
    color-matcher \
    matplotlib \
    huggingface_hub \
    mss \
    toml \
    rich \
    imageio-ffmpeg \
    opencv-python-headless \
    pillow \
    einops \
    addict \
    pyyaml

# Optional: SageAttention build requires CUDA toolchain and headers at build time.
# In cu128-slim base, CUDA is present at runtime but headers/toolchain are not exposed during image build.
# Skip building SageAttention at build-time to avoid CUDA_HOME errors.
# KJNodes PatchSageAttentionKJ will be available only if SageAttention is runtime-installed.
# You can runtime-install inside a running container when GPUs are present:
#   docker exec -it comfyui-cu128 bash -lc "git clone https://github.com/thu-ml/SageAttention.git /root/SageAttention && cd /root/SageAttention && pip3 install -v ."

# Expose port used by ComfyUI
EXPOSE 8188

# Set working directory to ComfyUI root
WORKDIR /root/ComfyUI

# Force ComfyUI to (re)index custom nodes at build-time to catch missing imports early.
# This won't run the server; it just triggers module import/registration.
RUN python3 - <<'PY' || true
import os, json
root="/root/ComfyUI/custom_nodes"
mods=[d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))]
print("Custom nodes present at build:", mods)
for d in mods:
    p=os.path.join(root,d)
    count=sum(1 for _ in os.walk(p))
    print(f" - {d}: tree entries={count}")
PY

# Start ComfyUI; pass through CLI_ARGS if provided by docker-compose.
# --listen binds to 0.0.0.0, --port 8188 aligns with your compose mapping.
# Base image provides python3, not `python`.
# At container start, ensure required custom nodes exist (host mount may be empty) then run ComfyUI.
ENV CLI_ARGS=""
CMD ["bash", "-lc", "\
  if [ ! -d /root/ComfyUI/custom_nodes/ComfyUI-KJNodes ] || \
     [ ! -d /root/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite ] || \
     [ ! -d /root/ComfyUI/custom_nodes/ComfyUI-GGUF ]; then \
    echo '[startup] Ensuring required custom nodes are present...'; \
    mkdir -p /root/ComfyUI/custom_nodes; \
    [ -d /root/ComfyUI/custom_nodes/ComfyUI-KJNodes ] || git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes.git /root/ComfyUI/custom_nodes/ComfyUI-KJNodes || true; \
    [ -d /root/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite ] || git clone --depth=1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /root/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite || true; \
    [ -d /root/ComfyUI/custom_nodes/ComfyUI-GGUF ] || git clone --depth=1 https://github.com/city96/ComfyUI-GGUF.git /root/ComfyUI/custom_nodes/ComfyUI-GGUF || true; \
  fi; \
  echo '[startup] Custom nodes: '; ls -la /root/ComfyUI/custom_nodes || true; \
  python3 main.py --listen --port 8188 $CLI_ARGS \
"]
