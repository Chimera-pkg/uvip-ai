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
    segformer_model: str = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
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
