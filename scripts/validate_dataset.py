#!/usr/bin/env python3
"""
Validate dataset quality before training.

Checks:
- Missing values
- Label distribution
- Feature statistics
- Outliers
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def validate_dataset(dataset_csv: Path):
    """Validate dataset quality."""
    print(f"[Validate] Loading dataset from {dataset_csv}")
    df = pd.read_csv(dataset_csv)

    print(f"[Validate] Dataset shape: {df.shape}")
    print(f"[Validate] Columns: {len(df.columns)}")

    # Check for missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        print(f"\n[WARNING] Found {total_missing} missing values:")
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count} missing ({count/len(df)*100:.1f}%)")
    else:
        print("\n[OK] No missing values")

    # Check label columns
    label_cols = ["label_beauty", "label_safety", "label_comfort", "label_uvi"]
    print("\n[Validate] Label statistics:")
    for col in label_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"  {col:20s}: mean={mean:.2f}, std={std:.2f}, range=[{min_val:.1f}, {max_val:.1f}]")
        else:
            print(f"  {col:20s}: MISSING")

    # Check segmentation metrics
    seg_cols = [
        "green_coverage_pct", "building_coverage_pct", "walkability_ratio",
        "visual_clutter_index", "sky_visibility_pct"
    ]
    print("\n[Validate] Segmentation metrics:")
    for col in seg_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            print(f"  {col:25s}: mean={mean:.2f}, std={std:.2f}")

    # Check embedding columns
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    print(f"\n[Validate] Embedding dimensions: {len(emb_cols)}")
    if len(emb_cols) < 1000:
        print(f"[WARNING] Expected ~1024 embedding dimensions, got {len(emb_cols)}")

    # Check for outliers in labels
    print("\n[Validate] Checking for outliers in labels:")
    for col in label_cols:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers > 0:
                print(f"  {col}: {outliers} outliers ({outliers/len(df)*100:.1f}%)")
            else:
                print(f"  {col}: no outliers")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(df.columns) - len(label_cols) - 3}")  # exclude filename, area, point_id
    print(f"Labels: {len([c for c in label_cols if c in df.columns])}/4")

    if total_missing > 0:
        print("\n[WARNING] Dataset has missing values")
        print("[HINT] Consider: df.dropna() or df.fillna()")

    if len(df) < 100:
        print("\n[WARNING] Small dataset (< 100 samples)")
        print("[HINT] Collect more survey data for better model performance")

    print("\n[OK] Dataset validation complete")


def main():
    parser = argparse.ArgumentParser(description="Validate dataset quality")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to dataset CSV",
    )

    args = parser.parse_args()

    dataset_csv = Path(args.input)
    if not dataset_csv.exists():
        print(f"[ERROR] Dataset file not found: {dataset_csv}")
        sys.exit(1)

    validate_dataset(dataset_csv)


if __name__ == "__main__":
    main()
