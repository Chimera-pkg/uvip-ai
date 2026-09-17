"""
Model Registry — simpan metadata semua model (Step 10).

Struktur JSON:
{
  "models": [
    {
      "name": "UVIP Privacy Guard YOLOv8n",
      "type": "privacy_masking",
      "version": "v1.0.0",
      "framework": "ultralytics/YOLOv8",
      "model_path": "models/privacy/yolov8n.pt",
      "created_at": "2026-08-04T00:00:00+07:00",
      "hardware_used": "NVIDIA RTX 3060 Laptop GPU",
      "notes": "Blur wajah & plat nomor"
    },
    ...
  ],
  "perception_models": [
    {
      "target": "beauty",
      "model_type": "perception_prediction",
      "version": "v1.0.0",
      "r2_score": 0.75,
      "mae_score": 0.45,
      "rmse_score": 0.62,
      "training_dataset": "Malang Street View 2025",
      "created_at": "...",
      "file": "models/perception/beauty_xgb.pkl"
    }
  ]
}
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ModelInfo:
    name: str
    type: str  # privacy_masking | segmentation | feature_extraction | perception_prediction
    version: str
    framework: str
    model_path: str
    created_at: str = ""
    hardware_used: str = ""
    notes: str = ""


@dataclass
class PerceptionModel:
    target: str  # beauty | safety | comfort | uvi
    model_type: str = "perception_prediction"
    version: str = "v1.0.0"
    r2_score: float = 0.0
    mae_score: float = 0.0
    rmse_score: float = 0.0
    training_dataset: str = ""
    created_at: str = ""
    file: str = ""


class ModelRegistry:
    """Register & load model metadata."""

    DEFAULT_FILE = Path("models/model_registry.json")

    def __init__(self):
        self.models: list[ModelInfo] = []
        self.perception_models: list[PerceptionModel] = []
        self._registry_file = None

    def register(self, info: ModelInfo) -> None:
        if not info.created_at:
            info.created_at = datetime.utcnow().isoformat() + "+07:00"
        self.models.append(info)

    def register_perception(self, pm: PerceptionModel) -> None:
        if not pm.created_at:
            pm.created_at = datetime.utcnow().isoformat() + "+07:00"
        self.perception_models.append(pm)

    def save(self, path: Path | None = None) -> None:
        path = path or self.DEFAULT_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "models": [asdict(m) for m in self.models],
            "perception_models": [asdict(m) for m in self.perception_models],
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path | None = None) -> "ModelRegistry":
        instance = cls()
        path = path or cls.DEFAULT_FILE
        if not path.exists():
            return instance
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        for m in data.get("models", []):
            instance.models.append(ModelInfo(**m))
        for m in data.get("perception_models", []):
            instance.perception_models.append(PerceptionModel(**m))
        return instance


# Helper untuk init default models
def create_default_registry() -> ModelRegistry:
    reg = ModelRegistry()
    reg.register(ModelInfo(
        name="UVIP Privacy Guard YOLOv8n",
        type="privacy_masking",
        version="v1.0.0",
        framework="ultralytics/YOLOv8",
        model_path="models/privacy/yolov8n.pt",
        hardware_used="NVIDIA GeForce RTX 3060 Laptop GPU",
        notes="Deteksi & blur wajah/person + car proxy plate"
    ))
    reg.register(ModelInfo(
        name="SegFormer-B5 Cityscapes Urban",
        type="segmentation",
        version="v1.0.0",
        framework="transformers/segformer",
        model_path="models/segmentation/segformer-b5.pt",
        hardware_used="NVIDIA GeForce RTX 3060 Laptop GPU",
        notes="Segmentasi pixel + 5 metrik urban"
    ))
    reg.register(ModelInfo(
        name="DINOv2 Large Feature Extractor",
        type="feature_extraction",
        version="v1.0.0",
        framework="transformers/dinov2",
        model_path="models/features/dinov2-large.pt",
        hardware_used="NVIDIA GeForce RTX 3060 Laptop GPU",
        notes="Embedding 1024-d untuk input XGBoost"
    ))
    return reg


if __name__ == "__main__":
    reg = create_default_registry()
    reg.save()
    print("Created default registry:", ModelRegistry.DEFAULT_FILE)
