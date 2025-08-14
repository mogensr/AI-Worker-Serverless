# ==== GPU base (CUDA 12.1, Ubuntu 22.04) ====
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ---- System packages ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git build-essential cmake ninja-build \
    ffmpeg unzip wget curl \
    libgl1 libglib2.0-0 libsm6 libxext6 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Make Python 3.10 default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# ---- Env (cache + CUDA) ----
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    HF_HOME=/models/hf-cache \
    TRANSFORMERS_CACHE=/models/hf-cache \
    TORCH_HOME=/models/torch-cache \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

# ---- Upgrade pip toolchain ----
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ---- Torch (CUDA 12.1) first ----
RUN python -m pip install --no-cache-dir \
    torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121

# ---- Python dependencies ----
COPY requirements.txt /app/requirements.txt

# Install requirements ONE BY ONE for clear error reporting
RUN set -eux; \
    grep -v '^\s*#' /app/requirements.txt | sed '/^\s*$/d' > /tmp/reqs.clean; \
    while read -r PKG; do \
        echo "=== Installing: ${PKG} ==="; \
        python -m pip install --no-cache-dir "${PKG}"; \
    done < /tmp/reqs.clean

# ---- Install SAM2 & MatAnyone (from git, pinned) ----
# Pin SAM2 to a pre-SAM2.1 commit (2024-08-07) compatible with Torch 2.3.x
RUN python -m pip install --no-cache-dir \
    git+https://github.com/facebookresearch/sam2.git@6186d15#egg=sam2
# Pin MatAnyone to v1.0.0 release
RUN python -m pip install --no-cache-dir \
    git+https://github.com/pq-yang/MatAnyone.git@v1.0.0#egg=matanyone

# ---- Optional: Hugging Face token for private repos ----
# Build with: --build-arg HF_TOKEN=hf_xxx (if needed)
ARG HF_TOKEN=""

# ---- Prefetch model weights into the image (no token persisted) ----
RUN HF_TOKEN=$HF_TOKEN python - <<'PY'
import os
from huggingface_hub import snapshot_download, login

token = os.environ.get("HF_TOKEN") or ""
if token:
    try:
        login(token=token)
    except Exception:
        pass

# SAM2 (will be found by .from_pretrained via HF cache)
snapshot_download(
    repo_id="facebook/sam2-hiera-large",
    local_dir="/models/sam2-hiera-large",
    local_dir_use_symlinks=False
)

# MatAnyone (HF assets used by InferenceCore)
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
