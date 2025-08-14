# GPU image (CUDA 12.1)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git build-essential wget curl ffmpeg unzip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# Speed up pip and ensure wheels
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements first (layer cache)
COPY requirements.txt /app/requirements.txt

# --- Install GPU Torch first (match CUDA 12.1) ---
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# --- Install remaining deps (incl. RunPod SDK) ---
RUN pip install --no-cache-dir -r /app/requirements.txt

# --- Install MatAnyone + friends ---
# MatAnyone (git clone) + exact libs
RUN git clone https://github.com/SHI-Labs/matanyone.git /tmp/matanyone && \
    cd /tmp/matanyone && \
    pip install --no-cache-dir . && \
    cd / && \
    rm -rf /tmp/matanyone && \
    pip install --no-cache-dir \
    omegaconf==2.3.0 hydra-core==1.3.2 easydict==0.1.10 \
    imageio==2.25.0 huggingface-hub>=0.16.0 safetensors>=0.3.0 \
    einops>=0.6.0 scipy>=1.10.0 av>=10.0.0

# SAM2 separat
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything-2.git

# Copy worker code
COPY handler.py /app/handler.py
COPY test_input.json /app/test_input.json

# Runtime env
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Serverless entrypoint (vigtigt!)
CMD ["python", "-u", "/app/handler.py"]
