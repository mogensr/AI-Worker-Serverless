# ==== GPU base (CUDA 12.1, Ubuntu 22.04) ====
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ---- System packages ----
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

# ---- Env for hurtigere builds, lavere RAM og stabil GPU-perf ----
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    # GPU/driver
    CUDA_VISIBLE_DEVICES=0 \
    # Hugging Face / Torch caches (hurtigere cold starts)
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
    TORCH_HOME=/root/.cache/torch \
    # Reducér CPU-tråde/overhead i serverless
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    UV_THREADPOOL_SIZE=2 \
    # Undgå ekstra tråde fra tokenizers (hvis transformers ender med at blive brugt)
    TOKENIZERS_PARALLELISM=false \
    # PyTorch allocator: færre fragmenteringsproblemer ved store tensors
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Opdatér toolchain
RUN python -m pip install --upgrade pip setuptools wheel

# ---- Torch (CUDA 12.1) først for at matche basen ----
RUN python -m pip install \
    torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121

# ---- Python dependencies ----
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

# ---- Worker code ----
COPY handler.py /app/handler.py
COPY test_input.json /app/test_input.json

# ---- Serverless entrypoint ----
CMD ["python", "-u", "/app/handler.py"]
