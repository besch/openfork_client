FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-distutils python3.11-dev curl git wget \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.8 wheels explicitly
RUN python -m pip install torch==2.7.1+cu128 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128

# Install base dependencies
RUN python -m pip install gitpython PyYAML opencv-python imageio-ffmpeg

# Set working directory
WORKDIR /opt/ComfyUI

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI
RUN python -m pip install -r requirements.txt

# Install ComfyUI Manager
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git /opt/ComfyUI/custom_nodes/ComfyUI-Manager
RUN python -m pip install -r /opt/ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt

# Download FLUX models
# Create directories
RUN mkdir -p /opt/ComfyUI/models/diffusion_models             /opt/ComfyUI/models/text_encoders             /opt/ComfyUI/models/vae

# Download models using curl
RUN curl -L -o /opt/ComfyUI/models/diffusion_models/flux1-schnell.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors
RUN curl -L -o /opt/ComfyUI/models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors
RUN curl -L -o /opt/ComfyUI/models/text_encoders/clip_l.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/text_eng_encoder/clip_l.safetensors
RUN curl -L -o /opt/ComfyUI/models/text_encoders/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/text_eng_encoder/t5xxl_fp8_e4m3fn.safetensors




# Install additional system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

# Install uv for ComfyUI-Manager
RUN python -m pip install uv

# Set final working directory and command
WORKDIR /opt/ComfyUI
CMD ["python", "main.py", "--listen"]
