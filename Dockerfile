# ==== GPU base (CUDA 12.1, Ubuntu 22.04) ====
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ---- System packages ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git build-essential cmake ninja-build \
    ffmpeg unzip wget curl \
    libgl1 libglib2.0-0 libsm6 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Gør py310 til default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# ---- Env (cache + CUDA) ----
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    HF_HOME=/models/hf-cache \
    TRANSFORMERS_CACHE=/models/hf-cache \
    TORCH_HOME=/models/torch-cache

# ---- Opdatér pip toolchain ----
RUN python -m pip install --upgrade pip setuptools wheel

# ---- Torch (CUDA 12.1) først ----
RUN python -m pip install \
    torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121

# ---- Python dependencies ----
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

# ---- Installer SAM2 & MatAnyone (git) ----
RUN python -m pip install \
    git+https://github.com/facebookresearch/segment-anything-2.git#egg=segment-anything-2 \
 && python -m pip install \
    git+https://github.com/pq-yang/MatAnyone.git#egg=matanyone

# ---- (Valgfrit) Hugging Face token til private repos ----
ARG HF_TOKEN=""

# ---- Prefetch model weights ind i image ----
RUN python - << 'PY'
import os
from huggingface_hub import snapshot_download, login

token = os.environ.get("HF_TOKEN", "")
if token:
    try:
        login(token=token)
    except Exception:
        pass

snapshot_download(
    repo_id="facebook/sam2-hiera-large",
    local_dir="/models/sam2-hiera-large",
    local_dir_use_symlinks=False
)

snapshot_download(
    repo_id="PeiqingYang/MatAnyone",
    local_dir="/models/matanyone",
    local_dir_use_symlinks=False
)
PY

# ---- Worker code ----
COPY handler.py /app/handler.py
COPY test_input.json /app/test_input.json

# ---- Entrypoint (RunPod Serverless) ----
CMD ["python", "-u", "/app/handler.py"]
