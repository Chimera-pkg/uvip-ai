#!/usr/bin/env python3
"""
Script untuk training XGBoost model dari dataset yang sudah disiapkan.

Usage:
    python scripts/train_xgboost.py --data data/datasets/training.csv --output models/perception

Dataset harus berisi kolom:
- filename, area, point_id (metadata)
- seg_green_coverage_pct, seg_building_coverage_pct, seg_walkability_ratio,
  seg_visual_clutter_index, seg_sky_visibility_pct (5 metrik segmentasi)
- emb_0 sampai emb_1023 (1024-d embedding DINOv2)
- label_beauty, label_safety, label_comfort, label_uvi (target dari kuesioner)

Output:
- models/perception/{beauty,safety,comfort,uvi}_xgb.pkl
- models/perception/metrics.json (R², MAE, RMSE per target)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uvip_ai.training.xgboost_model import XGBoostPerceptionTrainer


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost perception models")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training CSV (with labels)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/perception",
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"[Train] Loading dataset from {args.data}")
    df = pd.read_csv(args.data)
    print(f"[Train] Dataset shape: {df.shape}")

    # Validate required columns
    required_cols = [
        "seg_green_coverage_pct",
        "seg_building_coverage_pct",
        "seg_walkability_ratio",
        "seg_visual_clutter_index",
        "seg_sky_visibility_pct",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing required columns: {missing}")
        sys.exit(1)

    # Check for embedding columns
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if len(emb_cols) < 100:
        print(f"[WARNING] Only {len(emb_cols)} embedding columns found (expected ~1024)")

    # Check for label columns
    label_cols = ["label_beauty", "label_safety", "label_comfort", "label_uvi"]
    missing_labels = [c for c in label_cols if c not in df.columns]
    if missing_labels:
        print(f"[ERROR] Missing label columns: {missing_labels}")
        print("[HINT] You need to prepare survey labels first. See data/templates/labels_template.csv")
        sys.exit(1)

    # Train models
    print(f"\n[Train] Training {len(XGBoostPerceptionTrainer.TARGETS)} models with {args.n_folds}-fold CV")
    trainer = XGBoostPerceptionTrainer(n_folds=args.n_folds, random_state=args.random_state)
    results = trainer.train_all(df)

    # Save models
    output_dir = Path(args.output)
    trainer.save(output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for target, metrics in results.items():
        r2 = metrics["r2_mean"]
        status = "✓ PASS" if r2 >= 0.7 else "✗ FAIL"
        print(f"{target:12s}: R² = {r2:.4f} ± {metrics['r2_std']:.4f}  {status}")

    print(f"\nModels saved to: {output_dir.resolve()}")
    print(f"Metrics saved to: {(output_dir / 'metrics.json').resolve()}")

    # Check if all targets passed
    all_passed = all(m["r2_mean"] >= 0.7 for m in results.values())
    if not all_passed:
        print("\n[WARNING] Some targets did not reach R² ≥ 0.7")
        print("[HINT] Consider:")
        print("  - Collecting more survey data (increase dataset size)")
        print("  - Feature engineering (add more segmentation metrics)")
        print("  - Hyperparameter tuning (adjust n_estimators, max_depth, learning_rate)")
        sys.exit(1)
    else:
        print("\n✓ All targets reached R² ≥ 0.7 threshold")


if __name__ == "__main__":
    main()
