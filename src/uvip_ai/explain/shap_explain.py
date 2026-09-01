"""
SHAP Explainability Module (Step 6).

Menggunakan SHAP (SHapley Additive exPlanations) untuk menjelaskan
faktor pendorong di balik prediksi XGBoost.

Output: list of {"feature": str, "impact": float, "direction": "positive"/"negative"}
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ShapExplainer:
    """SHAP explainer untuk XGBoost perception model."""

    def __init__(self, model_path: str = "models/perception/beauty_xgb.pkl"):
        self.model_path = Path(model_path)
        self._model = None
        self._explainer = None
        self._loaded = False

    def _load_model(self):
        """Load model dan buat SHAP explainer."""
        if self._loaded:
            return

        if not self.model_path.exists():
            logger.warning("Model XGBoost tidak ditemukan: %s — SHAP tidak tersedia", self.model_path)
            self._model = None
            self._explainer = None
            self._loaded = True
            return

        try:
            import joblib
            import shap

            self._model = joblib.load(self.model_path)
            self._explainer = shap.TreeExplainer(self._model)
            self._loaded = True
            logger.info("✅ SHAP explainer loaded untuk: %s", self.model_path)
        except ImportError as e:
            logger.warning("SHAP tidak terinstall: %s — explainability tidak tersedia", e)
            self._model = None
            self._explainer = None
            self._loaded = True
        except Exception as e:
            logger.error("Gagal load SHAP explainer: %s", e)
            self._model = None
            self._explainer = None
            self._loaded = True

    def explain(self, features: np.ndarray, feature_names: list[str] | None = None) -> list[dict]:
        """
        Hitung SHAP values untuk satu sample.

        Args:
            features: array 1D atau 2D (1 sample, N features)
            feature_names: list nama fitur (optional)

        Returns:
            list of dict: [{"feature": str, "impact": float, "direction": str}]
        """
        self._load_model()

        if self._explainer is None:
            return []

        try:
            # Reshape jika 1D
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # Hitung SHAP values
            shap_values = self._explainer.shap_values(features)

            # Ambil SHAP values untuk sample pertama
            if isinstance(shap_values, list):
                # Multi-output model
                shap_vals = shap_values[0][0]
            else:
                # Single output
                shap_vals = shap_values[0]

            # Default feature names
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(shap_vals))]

            # Format output
            explanations = []
            for i, (name, value) in enumerate(zip(feature_names, shap_vals)):
                explanations.append({
                    "feature": name,
                    "impact": round(float(abs(value)), 4),
                    "direction": "positive" if value > 0 else "negative",
                    "shap_value": round(float(value), 4),
                })

            # Sort by impact (descending)
            explanations.sort(key=lambda x: x["impact"], reverse=True)

            return explanations

        except Exception as e:
            logger.error("Error saat hitung SHAP values: %s", e)
            return []

    def free_memory(self):
        """Release model dan explainer dari memory."""
        self._model = None
        self._explainer = None
        self._loaded = False


if __name__ == "__main__":
    # Test
    explainer = ShapExplainer()
    features = np.array([25.5, 40.2, 0.35, 1.2, 20.0])
    feature_names = [
        "green_coverage_pct",
        "building_coverage_pct",
        "walkability_ratio",
        "visual_clutter_index",
        "sky_visibility_pct",
    ]
    explanations = explainer.explain(features, feature_names)
    print("Explanations:", explanations)
