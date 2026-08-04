# ============================================================
# UVIP-AI — image produksi berbasis CUDA (Step 10)
# GPU inference untuk YOLOv8 / SegFormer / DINOv2 + FastAPI
# ============================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/models/hf_cache \
    PIP_NO_CACHE_DIR=1

# Python 3.10 + dependency sistem untuk opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv \
        libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch build CUDA 12.1 dulu (layer cache terpisah)
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src \
    UVIP_DEVICE=auto \
    UVIP_USE_FP16=true

EXPOSE 8001

# API pipeline AI (Step 8). Sesuaikan path app bila berubah.
CMD ["uvicorn", "uvip_ai.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
