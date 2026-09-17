"""Helper device & memori GPU — dipakai semua modul model.

Menyesuaikan otomatis dengan RTX 3060 Laptop (6GB): fp16 + low-VRAM mode.
"""
from __future__ import annotations

import contextlib
import gc

from uvip_ai.config import settings


def get_device() -> str:
    """'cuda' atau 'cpu' sesuai config & ketersediaan."""
    return settings.resolve_device()


def get_dtype():
    """torch.float16 bila fp16 aktif & GPU tersedia, selain itu float32."""
    import torch

    if settings.uvip_use_fp16 and get_device() == "cuda":
        return torch.float16
    return torch.float32


def free_gpu_memory() -> None:
    """Bersihkan cache VRAM — penting untuk LOW_VRAM_MODE (load->infer->unload)."""
    gc.collect()
    with contextlib.suppress(ImportError):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def vram_summary() -> str:
    """Ringkasan pemakaian VRAM saat ini (untuk debugging/profiling)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "GPU tidak tersedia (CPU mode)"
        used = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"VRAM: {used:.2f}GB terpakai / {reserved:.2f}GB reserved / {total:.1f}GB total"
    except ImportError:
        return "torch belum terpasang"
