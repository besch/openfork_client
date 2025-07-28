

# Use the official NVIDIA CUDA image as a base
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

# Install dependencies
RUN apt-get update && apt-get install -y     git     python3     python3-pip     wget     unzip     libgl1-mesa-glx     ffmpeg     && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI

# Install ComfyUI dependencies
RUN pip install --no-cache-dir -r /opt/ComfyUI/requirements.txt

# Install custom nodes
RUN set -e && \
    cd /opt/ComfyUI/custom_nodes && \
    mkdir -p ComfyUI-KJNodes && \
    wget https://github.com/kijai/ComfyUI-KJNodes/archive/refs/heads/main.zip -O ComfyUI-KJNodes.zip && \
    unzip ComfyUI-KJNodes.zip -d ComfyUI-KJNodes-temp && \
    mv ComfyUI-KJNodes-temp/ComfyUI-KJNodes-main/* ComfyUI-KJNodes/ && \
    rm -rf ComfyUI-KJNodes-temp ComfyUI-KJNodes.zip && \
    mkdir -p ComfyUI-VideoHelperSuite && \
    wget https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/archive/refs/heads/main.zip -O ComfyUI-VideoHelperSuite.zip && \
    unzip ComfyUI-VideoHelperSuite.zip -d ComfyUI-VideoHelperSuite-temp && \
    mv ComfyUI-VideoHelperSuite-temp/ComfyUI-VideoHelperSuite-main/* ComfyUI-VideoHelperSuite/ && \
    rm -rf ComfyUI-VideoHelperSuite-temp ComfyUI-VideoHelperSuite.zip && \
    mkdir -p ComfyUI-GGUF-Loader && \
    wget https://github.com/city96/ComfyUI-GGUF/archive/refs/heads/main.zip -O ComfyUI-GGUF-Loader.zip &&     unzip ComfyUI-GGUF-Loader.zip -d ComfyUI-GGUF-Loader-temp &&     mv ComfyUI-GGUF-Loader-temp/ComfyUI-GGUF-main/* ComfyUI-GGUF-Loader/ &&     rm -rf ComfyUI-GGUF-Loader-temp ComfyUI-GGUF-Loader.zip &&     pip install gguf opencv-python

# Download models

# Copy workflow.json to ComfyUI's workflows directory
COPY workflows/pusa.json /opt/ComfyUI/workflows/pusa.json

# Copy models to ComfyUI's model directories
COPY models/text_encoders/. /opt/ComfyUI/models/text_encoders/
COPY models/unet/. /opt/ComfyUI/models/unet/
COPY models/loras/. /opt/ComfyUI/models/loras/
COPY models/vae/. /opt/ComfyUI/models/vae/

# Set the working directory
WORKDIR /opt/ComfyUI

# Expose the ComfyUI port
EXPOSE 8188

# Set the entrypoint
CMD ["python3", "main.py", "--listen", "0.0.0.0"]
