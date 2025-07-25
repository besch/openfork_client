

# Use the official NVIDIA CUDA image as a base
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI

# Install ComfyUI dependencies
RUN pip install --no-cache-dir -r /opt/ComfyUI/requirements.txt

# Install custom nodes
RUN cd /opt/ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    git clone https://github.com/jtydhr88/ComfyUI-GGUF-Loader.git

# Download models
RUN mkdir -p /opt/ComfyUI/models/vae && \
    wget -O /opt/ComfyUI/models/vae/wan_2.1_vae.safetensors https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors && \
    mkdir -p /opt/ComfyUI/models/gguf && \
    wget -O /opt/ComfyUI/models/gguf/wan2.1-i2v-14b-480p-Q5_K_S.gguf https://huggingface.co/ltdrdata/wan-i2v/resolve/main/wan2.1-i2v-14b-480p-Q5_K_S.gguf && \
    mkdir -p /opt/ComfyUI/models/clip && \
    wget -O /opt/ComfyUI/models/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors https://huggingface.co/ltdrdata/umt5-xxl/resolve/main/umt5_xxl_fp8_e4m3fn_scaled.safetensors && \
    mkdir -p /opt/ComfyUI/models/loras && \
    wget -O /opt/ComfyUI/models/loras/wan21_lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors https://huggingface.co/ltdrdata/wan-i2v/resolve/main/wan21_lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank128_bf16.safetensors && \
    wget -O /opt/ComfyUI/models/loras/wan21_pusa_v1.safetensors https://huggingface.co/ltdrdata/wan-i2v/resolve/main/wan21_pusa_v1.safetensors && \
    wget -O /opt/ComfyUI/models/loras/Wan21_AccVid_I2V_480P_14B_lora_rank32_fp16.safetensors https://huggingface.co/ltdrdata/wan-i2v/resolve/main/Wan21_AccVid_I2V_480P_14B_lora_rank32_fp16.safetensors && \
    wget -O /opt/ComfyUI/models/loras/Wan2.1-Fun-14B-InP-MPS.safetensors https://huggingface.co/ltdrdata/wan-i2v/resolve/main/Wan2.1-Fun-14B-InP-MPS.safetensors

# Set the working directory
WORKDIR /opt/ComfyUI

# Expose the ComfyUI port
EXPOSE 8188

# Set the entrypoint
CMD ["python3", "main.py", "--listen", "0.0.0.0"]
