# ==== GPU base (CUDA 12.1, Ubuntu 22.04) ====
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ---- System packages ----
# ffmpeg til video I/O, libgl/libglib for OpenCV, cmake+ninja til evt. build af extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git build-essential cmake ninja-build \
    ffmpeg unzip wget curl \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Gør py310 til default "python"/"python3"
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# ---- Env for hurtigere, renere builds og caches ----
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
    TORCH_HOME=/root/.cache/torch

# Opdatér toolchain
RUN python -m pip install --upgrade pip setuptools wheel

# ---- Torch (CUDA 12.1) først for at matche basen ----
# Pin for stabilitet
RUN python -m pip install \
    torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121

# ---- Python dependencies ----
# Indeholder også MatAnyone + SAM2 via git+ i requirements.txt
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

# ---- Worker code ----
COPY handler.py /app/handler.py
COPY test_input.json /app/test_input.json

# ---- Serverless entrypoint ----
CMD ["python", "-u", "/app/handler.py"]
