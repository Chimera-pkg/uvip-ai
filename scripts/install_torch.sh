#!/usr/bin/env bash
# Instal PyTorch build CUDA 12.1 (cocok untuk RTX 3060 / 30-40 series).
# Jalankan SEBELUM: pip install -r requirements.txt
#
# Windows (Git Bash / PowerShell):  bash scripts/install_torch.sh
set -e
echo ">> Menginstal PyTorch (CUDA 12.1)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo ">> Selesai. Verifikasi dengan: python scripts/verify_gpu.py"
