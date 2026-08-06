"""
Training XGBoost — 4 model regresi terpisah: Beauty, Safety, Comfort, UVI (Step 6).

Input features = [5 metrik segmentasi] + [1024-d embedding DINOv2] → total 1029 dims.
Label: skor persepsi dari kuesioner responden (Beauty/Safety/Comfort/UVI).
Metrik evaluasi: R² (dapat K-Fold CV), MAE, RMSE. Target: R² >= 0.7.

Struktur data: DataFrame CSV dengan kolom:
  - seg_green_coverage_pct, seg_building_coverage_pct, seg_walkability_ratio,
    seg_visual_clutter_index, seg_sky_visibility_pct
  - emb_0 .. emb_1023 (embedding)
  - label_beauty, label_safety, label_comfort, label_uvi

Output: model.pkl + metrics.json + shap_values/*.pkl (Step 7).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


class XGBoostPerceptionTrainer:
    """Train 4 XGBoost models untuk Beauty/Safety/Comfort/UVI."""

    TARGETS = ["beauty", "safety", "comfort", "uvi"]

    # Feature columns
    SEG_COLS = [
        "seg_green_coverage_pct", "seg_building_coverage_pct", "seg_walkability_ratio",
        "seg_visual_clutter_index", "seg_sky_visibility_pct"
    ]

    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.models = {t: None for t in self.TARGETS}
        self.metrics = {}

    def _build_feature_cols(self, df: pd.DataFrame) -> list[str]:
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        return self.SEG_COLS + sorted(emb_cols)

    def train_model(self, train_df: pd.DataFrame, target: str) -> xgb.XGBRegressor:
        """Train satu model XGBoost untuk target tertentu."""
        feat_cols = self._build_feature_cols(train_df)
        X_train = train_df[feat_cols].fillna(0)
        y_train = train_df[target].fillna(np.nan).dropna()
        mask = ~y_train.isna()
        X_train = X_train[mask]
        y_train = y_train[mask]

        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=self.random_state,
            tree_method="hist",
        )
        model.fit(X_train, y_train, early_stopping_rounds=50, verbose=False)
        return model

    def cross_validate(self, df: pd.DataFrame, target: str) -> dict:
        """K-Fold cross-validation untuk target."""
        feat_cols = self._build_feature_cols(df)
        X = df[feat_cols].fillna(0).values
        y = df[target].to_numpy()

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        r2_scores = []
        mae_scores = []
        rmse_scores = []
        predictions = np.zeros_like(y, dtype=float)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = self.train_model(pd.DataFrame(X_tr, columns=feat_cols), target)
            y_pred = model.predict(X_val)

            r2_scores.append(r2_score(y_val, y_pred))
            mae_scores.append(mean_absolute_error(y_val, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
            predictions[val_idx] = y_pred

        avg_r2 = np.mean(r2_scores)
        is_achieved = avg_r2 >= 0.7
        return {
            "target": target,
            "r2_mean": round(avg_r2, 4),
            "r2_std": round(np.std(r2_scores), 4),
            "mae_mean": round(np.mean(mae_scores), 4),
            "rmse_mean": round(np.mean(rmse_scores), 4),
            "n_folds": self.n_folds,
            "is_accepted": is_achieved,
            "predictions_sample": predictions[:5].tolist(),
        }

    def train_all(self, df: pd.DataFrame) -> dict:
        """Train semua 4 model & validasi."""
        results = {}
        for tgt in self.TARGETS:
            print(f"[XGBoost] Training '{tgt}' ...")
            res = self.cross_validate(df, tgt)
            results[tgt] = res
            status = "OK" if res["is_accepted"] else f"LOW ({res['r2_mean']:.2f})"
            print(f"  {tgt}: R²={res['r2_mean']:.4f} ±{res['r2_std']:.4f} → {status}")
        self.results = results
        return results

    def save(self, out_dir: Path) -> None:
        """Simpan model ke file dan metrics JSON."""
        import pickle

        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = out_dir / "metrics.json"
        with metrics_file.open("w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

        for tgt in self.TARGETS:
            model_path = out_dir / f"{tgt}_xgb.pkl"
            with model_path.open("wb") as f:
                pickle.dump(self.models[tgt], f)
            print(f"[XGBoost] Model '{tgt}' saved to {model_path}")

        print(f"[XGBoost] Metrics saved to {metrics_file}")


if __name__ == "__main__":
    # Example stub: user needs to prepare training data first
    print("Usage: python scripts/train_xgboost.py --data data/datasets/training.csv")
