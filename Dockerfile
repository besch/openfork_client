# Use the official NVIDIA CUDA image as a base
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI

# Install ComfyUI dependencies
RUN pip install --no-cache-dir -r /opt/ComfyUI/requirements.txt

# Install custom nodes
RUN set -e && \
    cd /opt/ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    git clone https://github.com/city96/ComfyUI-GGUF.git ComfyUI-GGUF-Loader && \
    pip install gguf opencv-python

# Download models (commented out as in original)
# COPY workflow.json to ComfyUI's workflows directory
# COPY workflows/pusa.json /opt/ComfyUI/user/default/workflows/pusa.json

# Copy models to ComfyUI's model directories
# COPY models/text_encoders/. /opt/ComfyUI/models/text_encoders/
# COPY models/unet/. /opt/ComfyUI/models/unet/
# COPY models/loras/. /opt/ComfyUI/models/loras/
# COPY models/vae/. /opt/ComfyUI/models/vae/

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH

# Install custom libraries (SageAttention from source and Triton)
RUN pip install triton

# Install SageAttention
RUN git clone https://github.com/thu-ml/SageAttention.git /opt/SageAttention/source && \
    cd /opt/SageAttention/source && \
    EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 pip install -e .

# Set the working directory
WORKDIR /opt/ComfyUI

# Expose the ComfyUI port
EXPOSE 8188

# Set the entrypoint
CMD ["python3", "main.py", "--listen", "0.0.0.0"]