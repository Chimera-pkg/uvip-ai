"""
XGBoost Perception Model Wrapper (Step 5).

Wrapper untuk model XGBoost yang memprediksi 4 skor persepsi urban:
- beauty_score (0-10)
- safety_score (0-10)
- comfort_score (0-10)
- uvi_score (0-10)

Input: 5 metrik segmentasi + DINOv2 embedding (384-d untuk small)
Output: dict dengan 4 skor persepsi

Jika model .pkl belum ada, gunakan dummy prediction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class XGBoostPerceptionModel:
    """Wrapper untuk XGBoost perception model."""

    def __init__(self, model_path: str = "models/perception/beauty_xgb.pkl"):
        self.model_path = Path(model_path)
        self._model = None
        self._loaded = False

        # Feature names: 5 metrik segmentasi
        self.feature_names = [
            "green_coverage_pct",
            "building_coverage_pct",
            "walkability_ratio",
            "visual_clutter_index",
            "sky_visibility_pct",
        ]

    def _load_model(self):
        """Load model XGBoost dari file .pkl."""
        if self._loaded:
            return

        if not self.model_path.exists():
            logger.warning("Model XGBoost tidak ditemukan: %s — gunakan dummy prediction", self.model_path)
            self._model = None
            self._loaded = True
            return

        try:
            import joblib
            self._model = joblib.load(self.model_path)
            self._loaded = True
            logger.info("✅ Model XGBoost loaded: %s", self.model_path)
        except Exception as e:
            logger.error("Gagal load model XGBoost: %s", e)
            self._model = None
            self._loaded = True

    def predict(self, metrics: dict, embedding: np.ndarray | None = None) -> dict:
        """
        Predict 4 skor persepsi dari metrik segmentasi + embedding.

        Args:
            metrics: dict dengan 5 metrik segmentasi
            embedding: DINOv2 embedding (384-d untuk small), optional

        Returns:
            dict dengan beauty_score, safety_score, comfort_score, uvi_score
        """
        self._load_model()

        # Extract 5 metrik utama
        features = np.array([
            metrics.get("green_coverage_pct", 0.0),
            metrics.get("building_coverage_pct", 0.0),
            metrics.get("walkability_ratio", 0.0),
            metrics.get("visual_clutter_index", 0.0),
            metrics.get("sky_visibility_pct", 0.0),
        ]).reshape(1, -1)

        # Jika ada embedding, gabungkan dengan features
        if embedding is not None and self._model is not None:
            # Flatten embedding jika perlu
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            features = np.hstack([features, embedding.reshape(1, -1)])

        # Jika model tidak ada, gunakan dummy prediction
        if self._model is None:
            return self._dummy_prediction(metrics)

        try:
            # Predict dengan model XGBoost
            predictions = self._model.predict(features)

            # XGBoost bisa return 1 value atau 4 values (multi-output)
            if predictions.shape[1] == 4:
                beauty, safety, comfort, uvi = predictions[0]
            else:
                # Single output — gunakan untuk semua (fallback)
                score = float(predictions[0])
                beauty = safety = comfort = uvi = score

            # Clamp ke range 0-10
            beauty = float(np.clip(beauty, 0, 10))
            safety = float(np.clip(safety, 0, 10))
            comfort = float(np.clip(comfort, 0, 10))
            uvi = float(np.clip(uvi, 0, 10))

            return {
                "beauty_score": round(beauty, 2),
                "safety_score": round(safety, 2),
                "comfort_score": round(comfort, 2),
                "uvi_score": round(uvi, 2),
            }

        except Exception as e:
            logger.error("Error saat predict dengan XGBoost: %s", e)
            return self._dummy_prediction(metrics)

    def _dummy_prediction(self, metrics: dict) -> dict:
        """
        Dummy prediction berdasarkan heuristik sederhana.
        Digunakan jika model XGBoost belum dilatih.
        """
        green = metrics.get("green_coverage_pct", 0.0)
        building = metrics.get("building_coverage_pct", 0.0)
        walkability = metrics.get("walkability_ratio", 0.0)
        clutter = metrics.get("visual_clutter_index", 0.0)
        sky = metrics.get("sky_visibility_pct", 0.0)

        # Heuristik sederhana
        beauty = min(10, green * 0.3 + sky * 0.2 + (10 - clutter) * 0.2 + 3)
        safety = min(10, walkability * 5 + sky * 0.1 + 4)
        comfort = min(10, green * 0.2 + sky * 0.3 + walkability * 2 + 3)
        uvi = min(10, (10 - green) * 0.3 + (10 - sky) * 0.2 + 5)

        return {
            "beauty_score": round(beauty, 2),
            "safety_score": round(safety, 2),
            "comfort_score": round(comfort, 2),
            "uvi_score": round(uvi, 2),
        }

    def free_memory(self):
        """Release model dari memory."""
        self._model = None
        self._loaded = False


if __name__ == "__main__":
    # Test
    model = XGBoostPerceptionModel()
    metrics = {
        "green_coverage_pct": 25.5,
        "building_coverage_pct": 40.2,
        "walkability_ratio": 0.35,
        "visual_clutter_index": 1.2,
        "sky_visibility_pct": 20.0,
    }
    predictions = model.predict(metrics)
    print("Predictions:", predictions)
