FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122
RUN pip3 install --no-cache-dir opencv-python-headless numpy

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git
WORKDIR /app/ComfyUI
RUN pip3 install --no-cache-dir -r requirements.txt

# Install custom nodes (comfyui-kjnodes, comfyui-videohelpersuite)
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git custom_nodes/ComfyUI-KJNodes
RUN git clone https://github.com/kosinkadink/ComfyUI-VideoHelperSuite.git custom_nodes/ComfyUI-VideoHelperSuite
RUN pip3 install --no-cache-dir -r custom_nodes/ComfyUI-KJNodes/requirements.txt
RUN pip3 install --no-cache-dir -r custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt

# Download models and LoRAs (from workflow JSON)
RUN mkdir -p models/checkpoints models/vae models/loras
RUN wget -O models/vae/wan_2.1_vae.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors
RUN wget -O models/checkpoints/wan2.1-i2v-14b-480p-Q5_K_S.gguf https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/unet/wan2.1-i2v-14b-480p-Q5_K_S.gguf
RUN wget -O models/loras/wan21_lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/loras/wan21_lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors
RUN wget -O models/loras/Wan21_AccVid_I2V_480P_14B_lora_rank32_fp16.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/loras/Wan21_AccVid_I2V_480P_14B_lora_rank32_fp16.safetensors
RUN wget -O models/loras/Wan2.1-Fun-14B-InP-MPS.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/loras/Wan2.1-Fun-14B-InP-MPS.safetensors
RUN wget -O models/loras/wan21_pusa_v1.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/loras/wan21_pusa_v1.safetensors
RUN wget -O models/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors

# Copy workflow JSON (assumed to be provided locally for now)
COPY workflow.json /app/ComfyUI/workflow.json

# Set entrypoint to run ComfyUI with the workflow
CMD ["python3", "main.py", "--input-workflow", "/app/ComfyUI/workflow.json", "--output", "/app/ComfyUI/output"]