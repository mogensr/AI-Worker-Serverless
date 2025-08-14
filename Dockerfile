# GPU image (CUDA 12.1)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git build-essential wget curl ffmpeg unzip \
    libgl1 libglib2.0-0 cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# Upgrade pip toolchain
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements first (cache layer)
COPY requirements.txt /app/requirements.txt

# --- Install GPU Torch first (match CUDA 12.1) ---
# Pin versions for stability (anbefalet)
RUN python -m pip install --no-cache-dir \
    torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# --- Install remaining deps (incl. RunPod SDK) ---
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# --- Install MatAnyone + friends ---
# Brug officiel repo (eller behold din fork, hvis tilsigtet)
RUN git clone https://github.com/pq-yang/MatAnyone.git /tmp/MatAnyone && \
    python -m pip install --no-cache-dir /tmp/MatAnyone && \
    rm -rf /tmp/MatAnyone

# Ekstra afhængigheder (VIGTIGT: citér >= kravene)
RUN python -m pip install --no-cache-dir \
    "omegaconf==2.3.0" "hydra-core==1.3.2" "easydict==0.1.10" \
    "imageio==2.25.0" "huggingface-hub>=0.16.0" "safetensors>=0.3.0" \
    "einops>=0.6.0" "scipy>=1.10.0" "av>=10.0.0"

# SAM2 separat (kræver build tools; vi installerede cmake/ninja)
RUN python -m pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything-2.git

# Copy worker code
COPY handler.py /app/handler.py
COPY test_input.json /app/test_input.json

# Runtime env
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Serverless entrypoint
CMD ["python", "-u", "/app/handler.py"]
