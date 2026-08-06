"""
SHAP Explainability — faktor pendorong skor persepsi (Step 7).

Input: model XGBoost + feature vector → output SHAP values per fitur.
Output untuk setiap target: daftar [feature_name, display_label, shap_value, is_positive, rank_order].

Display labels dalam Bahasa Indonesia:
  - green_coverage_pct        → "Tutupan Hijau"
  - building_coverage_pct     → "Tutupan Bangunan"
  - walkability_ratio         → "Rasio Trotoar"
  - visual_clutter_index      → "Indeks Kerumitan Visual"
  - sky_visibility_pct        → "Visibilitas Langit"
  - emb_*                     → embedding features (aggregasi top k)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from transformers import Pipeline


class ShapExplainer:
    """Explainability XGBoost dengan SHAP."""

    LABEL_MAP = {
        "seg_green_coverage_pct": "Tutupan Hijau",
        "seg_building_coverage_pct": "Tutupan Bangunan",
        "seg_walkability_ratio": "Rasio Trotoar",
        "seg_visual_clutter_index": "Indeks Kerumitan Visual",
        "seg_sky_visibility_pct": "Visibilitas Langit",
    }

    def __init__(self, model: object, feature_names: list[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None

    def _load_explainer(self) -> None:
        if self.explainer is not None:
            return
        # Use TreeExplainer for XGBoost
        self.explainer = shap.TreeExplainer(self.model)
        print("[SHAP] Explainer loaded.")

    @property
    def explainer(self):
        self._load_explainer()
        return self._explainer

    @explainer.setter
    def explainer(self, val):
        self._explainer = val

    def explain_single(self, x: np.ndarray, target: str = None) -> list[dict]:
        """Explain satu prediksi → list of shap values ordered by magnitude."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        shap_values = self.explainer.shap_values(x)[0]  # [n_features, ]

        results = []
        for i, name in enumerate(self.feature_names):
            sv = float(shap_values[i])
            label = self.LABEL_MAP.get(name, f"Feature {i}")
            is_pos = sv > 0
            results.append({
                "feature_name": name,
                "display_label": label,
                "shap_value": round(sv, 6),
                "is_positive": is_pos,
            })

        # Sort descending by absolute value
        results.sort(key=lambda r: abs(r["shap_value"]), reverse=True)
        for idx, r in enumerate(results):
            r["rank_order"] = idx + 1
        return results

    def explain_batch(self, x: np.ndarray, batch_size: int = 50) -> list[list[dict]]:
        """Explain batch gambar → list of explanations."""
        all_results = []
        for i in range(0, len(x), batch_size):
            chunk = x[i:i+batch_size]
            chunk_sv = self.explainer.shap_values(chunk)
            for j, sv in enumerate(chunk_sv):
                exps = self._sv_to_exps(sv, batch_size)
                all_results.append(exps)
        return all_results

    def _sv_to_exps(self, sv: np.ndarray, max_n: int = 20) -> list[dict]:
        """Convert SHAP array ke list ranked dicts."""
        indexed = list(zip(self.feature_names, sv))
        indexed.sort(key=lambda t: abs(t[1]), reverse=True)
        top = indexed[:max_n]
        results = []
        for rank, (name, val) in enumerate(top):
            label = self.LABEL_MAP.get(name, name)
            results.append({
                "feature_name": name,
                "display_label": label,
                "shap_value": round(float(val), 6),
                "is_positive": val > 0,
                "rank_order": rank + 1,
            })
        return results


if __name__ == "__main__":
    # Stub example
    print("Usage after training: load model + call ShapExplainer.explain_single(embedding)")
