#!/usr/bin/env python
"""
Generate synthetic labels untuk training model.
Labels ini berdasarkan heuristik sederhana dari segmentation features.

Usage:
    python scripts/generate_synthetic_labels.py
"""
import pandas as pd
import numpy as np
from pathlib import Path


def calculate_beauty_score(row):
    """Beauty: kombinasi vegetation, sky, dan low clutter."""
    score = (
        row['vegetation_pct'] * 0.4 +
        row['sky_pct'] * 0.3 +
        (100 - row['visual_clutter_index']) * 0.3
    )
    return np.clip(score / 10, 1, 10)


def calculate_safety_score(row):
    """Safety: sidewalk, road quality, low vehicle."""
    score = (
        row['sidewalk_pct'] * 0.5 +
        row['road_pct'] * 0.3 +
        (100 - row['vehicle_pct']) * 0.2
    )
    return np.clip(score / 10, 1, 10)


def calculate_comfort_score(row):
    """Comfort: vegetation, sky, walkability."""
    score = (
        row['vegetation_pct'] * 0.4 +
        row['sky_pct'] * 0.3 +
        row['walkability_ratio'] * 100 * 0.3
    )
    return np.clip(score / 10, 1, 10)


def calculate_uvi_score(row):
    """UVI (Urban Vegetation Index): vegetation dominance."""
    score = (
        row['green_coverage_pct'] * 0.7 +
        row['vegetation_pct'] * 0.3
    )
    return np.clip(score / 10, 1, 10)


def main():
    # Load features
    features_path = Path('data/training/features_segmentation.csv')

    if not features_path.exists():
        print(f"❌ Features file tidak ditemukan: {features_path}")
        print("   Run: python scripts/retrain_model_for_backend.py")
        return

    features_df = pd.read_csv(features_path)
    print(f"✓ Loaded {len(features_df)} photos")

    # Generate synthetic labels
    print("\n🔍 Generating synthetic labels...")

    labels_df = features_df[['filename', 'area', 'point_id']].copy()
    labels_df['label_beauty'] = features_df.apply(calculate_beauty_score, axis=1)
    labels_df['label_safety'] = features_df.apply(calculate_safety_score, axis=1)
    labels_df['label_comfort'] = features_df.apply(calculate_comfort_score, axis=1)
    labels_df['label_uvi'] = features_df.apply(calculate_uvi_score, axis=1)

    # Save labels
    labels_path = Path('data/training/labels_synthetic.csv')
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(labels_path, index=False)

    print(f"✓ Synthetic labels generated: {len(labels_df)} photos")
    print(f"✓ Saved to: {labels_path}")

    # Show statistics
    print("\n📊 Label statistics:")
    for col in ['label_beauty', 'label_safety', 'label_comfort', 'label_uvi']:
        print(f"   {col:15s}: mean={labels_df[col].mean():.2f}, "
              f"std={labels_df[col].std():.2f}, "
              f"min={labels_df[col].min():.2f}, max={labels_df[col].max():.2f}")

    print("\n⚠️  Labels ini SYNTHETIC (dari heuristik, bukan survey manusia)")
    print("   Cocok untuk testing pipeline. Untuk production, pakai data survey asli.")
    print("\n✅ Sekarang bisa run: python scripts/retrain_model_for_backend.py")


if __name__ == "__main__":
    main()
