#!/usr/bin/env python3
"""
Merge extracted features with survey labels.

Input:
- features.csv: hasil extract_features.py
- labels.csv: survey labels dari kuesioner

Output:
- dataset.csv: merged data siap training
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def merge_features_and_labels(features_csv: Path, labels_csv: Path, output_csv: Path):
    """Merge features dengan labels berdasarkan filename."""
    print(f"[Merge] Loading features from {features_csv}")
    features = pd.read_csv(features_csv)
    print(f"[Merge] Features shape: {features.shape}")

    print(f"[Merge] Loading labels from {labels_csv}")
    labels = pd.read_csv(labels_csv)
    print(f"[Merge] Labels shape: {labels.shape}")

    # Validate label columns
    required_label_cols = ["filename", "label_beauty", "label_safety", "label_comfort", "label_uvi"]
    missing = [c for c in required_label_cols if c not in labels.columns]
    if missing:
        print(f"[ERROR] Missing label columns: {missing}")
        sys.exit(1)

    # Merge on filename
    print("[Merge] Merging on 'filename'...")
    merged = pd.merge(features, labels, on="filename", how="inner")

    # Check for missing labels
    n_features = len(features)
    n_merged = len(merged)
    n_missing = n_features - n_merged

    if n_missing > 0:
        print(f"[WARNING] {n_missing} photos have no labels (will be excluded)")
        missing_files = set(features["filename"]) - set(merged["filename"])
        print(f"[WARNING] Missing labels for: {list(missing_files)[:5]}...")

    if n_merged == 0:
        print("[ERROR] No matching files found between features and labels")
        print("[HINT] Make sure 'filename' column matches exactly")
        sys.exit(1)

    # Save merged dataset
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    print(f"\n[Merge] Saved {n_merged} rows to {output_csv}")
    print(f"[Merge] Final shape: {merged.shape}")
    print(f"[Merge] Label columns: {required_label_cols}")


def main():
    parser = argparse.ArgumentParser(description="Merge features with labels")
    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Path to features CSV",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to labels CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training/dataset.csv",
        help="Output merged CSV",
    )

    args = parser.parse_args()

    features_csv = Path(args.features)
    labels_csv = Path(args.labels)
    output_csv = Path(args.output)

    if not features_csv.exists():
        print(f"[ERROR] Features file not found: {features_csv}")
        sys.exit(1)

    if not labels_csv.exists():
        print(f"[ERROR] Labels file not found: {labels_csv}")
        sys.exit(1)

    merge_features_and_labels(features_csv, labels_csv, output_csv)


if __name__ == "__main__":
    main()
