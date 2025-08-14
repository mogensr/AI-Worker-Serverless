# Use CUDA-enabled base image for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# Upgrade pip and install basic packages first
RUN python -m pip install --upgrade pip setuptools wheel

# Copy and install requirements
COPY requirements.txt .

# Install PyTorch with CUDA support first
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies first (without SAM2)
RUN pip install fastapi uvicorn[standard] python-multipart transformers pillow numpy opencv-python

# Install MatAnyone dependencies (FIXED LINE)
RUN pip install git+https://github.com/SHI-Labs/matanyone.git omegaconf==2.3.0 hydra-core==1.3.2 easydict==0.1.10 imageio==2.25.0 huggingface-hub>=0.16.0 safetensors>=0.3.0 einops>=0.6.0 scipy>=1.10.0 av>=10.0.0

# Install SAM2 separately to avoid naming conflicts
RUN pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
