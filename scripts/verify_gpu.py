#!/usr/bin/env python
"""
Step 1 — Verifikasi environment & GPU untuk UVIP-AI.

Jalankan:  python scripts/verify_gpu.py

Mengecek: versi Python, PyTorch, ketersediaan CUDA, VRAM, dan
melakukan uji komputasi kecil di GPU. Juga cek library inti terpasang.
"""
from __future__ import annotations

import importlib
import platform
import sys

GREEN, RED, YEL, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}[ OK ]{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{YEL}[WARN]{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}[FAIL]{RESET} {msg}")


def check_python() -> bool:
    v = sys.version_info
    print(f"\n== Python ==")
    print(f"  versi: {platform.python_version()}  ({platform.system()} {platform.machine()})")
    if v >= (3, 10):
        ok("Python 3.10+ terpenuhi")
        return True
    fail("Butuh Python 3.10 atau lebih baru")
    return False


def check_torch() -> bool:
    print("\n== PyTorch & CUDA ==")
    try:
        import torch
    except ImportError:
        fail("PyTorch belum terpasang. Lihat README bagian instalasi torch.")
        return False

    print(f"  torch: {torch.__version__}")
    print(f"  build CUDA: {torch.version.cuda}")

    if not torch.cuda.is_available():
        warn("torch.cuda.is_available() == False → akan jalan di CPU (lambat).")
        warn("Pastikan install torch build CUDA, bukan CPU-only.")
        return False

    n = torch.cuda.device_count()
    ok(f"CUDA tersedia — {n} GPU terdeteksi")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        vram = p.total_memory / 1024**3
        print(f"  GPU {i}: {p.name}  |  VRAM {vram:.1f} GB  |  CC {p.major}.{p.minor}")
        if vram < 7:
            warn(f"  VRAM {vram:.1f}GB kecil — aktifkan UVIP_USE_FP16 & UVIP_LOW_VRAM_MODE.")

    # uji komputasi kecil di GPU
    try:
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x).sum().item()
        torch.cuda.synchronize()
        ok(f"Uji matmul di GPU sukses (checksum={y:.1f})")
    except Exception as e:  # noqa: BLE001
        fail(f"Uji komputasi GPU gagal: {e}")
        return False
    return True


def check_libraries() -> bool:
    print("\n== Library inti ==")
    libs = {
        "ultralytics": "YOLOv8 (privacy guard)",
        "transformers": "SegFormer / DINOv2",
        "cv2": "opencv (blur)",
        "xgboost": "regresi persepsi",
        "sklearn": "K-Fold / metrik",
        "shap": "explainability",
        "pandas": "dataset",
        "fastapi": "API",
    }
    all_ok = True
    for mod, desc in libs.items():
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            ok(f"{mod:14s} {ver:12s} — {desc}")
        except ImportError:
            warn(f"{mod:14s} {'-':12s} — belum terpasang ({desc})")
            all_ok = False
    return all_ok


def main() -> int:
    print("=" * 60)
    print(" UVIP-AI — Verifikasi Environment (Step 1)")
    print("=" * 60)
    py = check_python()
    torch_ok = check_torch()
    libs_ok = check_libraries()

    print("\n" + "=" * 60)
    if py and torch_ok and libs_ok:
        ok("Environment SIAP untuk semua step.")
        return 0
    if py and torch_ok:
        warn("GPU siap, tapi sebagian library belum terpasang.")
        warn("Jalankan: pip install -r requirements.txt")
        return 0
    fail("Environment BELUM siap — ikuti README untuk melengkapi.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
