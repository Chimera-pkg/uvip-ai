"""Konfigurasi terpusat untuk UVIP-AI. Dibaca dari environment / .env."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Backend
    uvip_api_base_url: str = "http://127.0.0.1:8000"
    uvip_api_token: str = ""

    # Device
    uvip_device: str = "auto"          # auto | cuda | cpu
    uvip_use_fp16: bool = True
    uvip_low_vram_mode: bool = True

    # Models
    hf_home: str = str(PROJECT_ROOT / "models" / "hf_cache")
    segformer_model: str = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"  # b5 (akurasi lebih tinggi)
    dinov2_model: str = "facebook/dinov2-small"  # small untuk CPU, large untuk GPU
    yolo_model: str = "yolov8n.pt"
    xgboost_model_path: str = str(PROJECT_ROOT / "models" / "perception" / "beauty_xgb.pkl")

    # Paths
    data_raw_dir: str = str(PROJECT_ROOT / "data" / "raw")
    data_extracted_dir: str = str(PROJECT_ROOT / "data" / "extracted")
    masks_dir: str = str(PROJECT_ROOT / "data" / "masks")
    datasets_dir: str = str(PROJECT_ROOT / "data" / "datasets")
    models_dir: str = str(PROJECT_ROOT / "models")

    # Performance
    latency_target_ms: int = 700

    # LLM gateway - dipakai oleh router /llm/* (lihat uvip_ai/llm/router.py).
    # Upstream ini OpenAI-compatible tapi HANYA menerima POST; router /llm/*
    # menambah varian GET supaya prompt bisa dipanggil dari browser/<iframe>/curl.
    nine_router_url: str = "https://api.deepseek.com/v1"
    nine_router_key: str = ""
    # Model default /llm/ask & /llm/chat. "deepseek-flash" (DeepSeek-V4.1-Flash)
    # = model "flash" resmi: teks + gambar, reasoning. Id lain: deepseek-v4-pro
    # (teks saja, BUKAN flash - jangan dipakai sesuai permintaan "flash aja").
    nine_router_model: str = "deepseek-flash"
    # Model untuk /llm/vision & /llm/advise (harus vision-capable).
    nine_router_vision_model: str = "deepseek-flash"
    # Model khusus POST /llm/advise (analisa foto -> rekomendasi + prompt gambar).
    nine_router_advise_model: str = "deepseek-flash"
    # Rantai fallback (dipisah koma) - diarahkan ke `nine_router_fallback_url`
    # bila diisi. DIPAKAI HANYA jalur TEKS (/llm/ask & /llm/chat); jalur vision
    # memakai deepseek-flash saja (fallback teks tidak bisa baca gambar).
    nine_router_fallbacks: str = "x/deepseek-v4.1-flash"
    # Endpoint fallback (opsional). Kosong = pakai `nine_router_url`.
    nine_router_fallback_url: str = ""
    nine_router_fallback_key: str = ""
    nine_router_timeout_s: int = 120

    # OpenAI Images API - dipakai endpoint POST /llm/generate.
    # Key `sk-proj-...` disimpan di .env (gitignored). Model gambar resmi OpenAI
    # adalah gpt-image-1.5 (gpt-image-1 sudah dihapus); size valid:
    # 1024x1024 | 1536x1024 | 1024x1536.
    openai_api_key: str = ""
    openai_image_model: str = "gpt-image-1.5"
    openai_image_size: str = "1024x1024"
    openai_image_timeout_s: int = 120

    def resolve_device(self) -> str:
        """Kembalikan 'cuda' atau 'cpu' berdasarkan setting & ketersediaan."""
        if self.uvip_device == "cpu":
            return "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"


settings = Settings()
