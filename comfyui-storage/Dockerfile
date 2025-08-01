# Use CUDA 12.8 base image
FROM runpod/base:0.6.2-cuda12.6.2

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-distutils python3.11-dev curl git \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install PyTorch & Triton with python3.11 pip
RUN python -m pip install --upgrade pip \
 && python -m pip install torch>=2.3.0 triton>=3.0.0

# Clone ComfyUI and install requirements + PyYAML
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI
WORKDIR /opt/ComfyUI
RUN python -m pip install -r requirements.txt \
 && python -m pip install PyYAML \
 && python -m pip install opencv-python \
 && python -m pip install imageio-ffmpeg

# Install SageAttention precompiled wheel for Python 3.11
RUN python -m pip install \
    https://huggingface.co/ModelsLab/Sage_2_plus_plus_build/resolve/main/sageattention-2.2.0-cp311-cp311-linux_x86_64.whl

# Install custom nodes (example)
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git /opt/ComfyUI/custom_nodes/comfyui-kjnodes
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /opt/ComfyUI/custom_nodes/VideoHelperSuite
RUN git clone https://github.com/city96/ComfyUI-GGUF.git /opt/ComfyUI/custom_nodes/ComfyUI-GGUF
RUN python -m pip install --upgrade gguf

# # Copy models to ComfyUI's model directories and workflows
# COPY workflows/ /opt/ComfyUI/user/default/workflows/
# COPY models/text_encoders/ /opt/ComfyUI/models/text_encoders/
# COPY models/unet/ /opt/ComfyUI/models/unet/
# COPY models/loras/ /opt/ComfyUI/models/loras/
# COPY models/vae/ /opt/ComfyUI/models/vae/

# Set final working directory and command
WORKDIR /opt/ComfyUI
CMD ["python", "main.py", "--listen"]

