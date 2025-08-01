FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-distutils python3.11-dev curl git \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.8 wheels explicitly
RUN python -m pip install torch==2.7.1+cu128 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128

# Install other dependencies as before, including gitpython to fix 'git' module error
RUN python -m pip install gitpython PyYAML opencv-python imageio-ffmpeg

# Clone SageAttention for installation at runtime (when GPU is available)
RUN python -m pip install ninja
RUN git clone https://github.com/thu-ml/SageAttention.git /opt/sageattention

# Set working directory back to /opt/ComfyUI
WORKDIR /opt/ComfyUI

# Clone ComfyUI and custom nodes as before
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI
RUN python -m pip install -r requirements.txt

# Clone ComfyUI Manager into custom_nodes
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git /opt/ComfyUI/custom_nodes/ComfyUI-Manager
# Install ComfyUI Manager Python dependencies
RUN python -m pip install GitPython PyGithub matrix-client==0.4.0 transformers huggingface-hub>0.20

RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git /opt/ComfyUI/custom_nodes/comfyui-kjnodes
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /opt/ComfyUI/custom_nodes/VideoHelperSuite
RUN git clone https://github.com/city96/ComfyUI-GGUF.git /opt/ComfyUI/custom_nodes/ComfyUI-GGUF
RUN python -m pip install --upgrade gguf

# Install requirements for custom nodes
RUN python -m pip install color-matcher matplotlib huggingface_hub mss toml rich

# Install additional system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

# Install uv for ComfyUI-Manager
RUN python -m pip install uv


# # Copy models to ComfyUI's model directories and workflows
# COPY workflows/ /opt/ComfyUI/user/default/workflows/
# COPY models/text_encoders/ /opt/ComfyUI/models/text_encoders/
# COPY models/unet/ /opt/ComfyUI/models/unet/
# COPY models/loras/ /opt/ComfyUI/models/loras/
# COPY models/vae/ /opt/ComfyUI/models/vae/

# Set final working directory and command
WORKDIR /opt/ComfyUI
# Install SageAttention at runtime when GPU is available, then start ComfyUI
CMD ["sh", "-c", "cd /opt/sageattention && python -m pip install -v . && cd /opt/ComfyUI && python main.py --listen"]
