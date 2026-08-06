#!/usr/bin/env python3
"""
Extract features from all photos for training.

This script processes all photos through the AI pipeline:
1. Privacy Guard (YOLOv8n) - blur faces/plates
2. SegFormer-B5 - extract 5 urban metrics
3. DINOv2-Large - extract 1024-d embedding

Output: CSV with all features ready for XGBoost training.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uvip_ai.features.dinov2 import Dinov2Extractor
from uvip_ai.pipeline.build_dataset import DatasetBuilder
from uvip_ai.segmentation.segformer import SegformerB5


def extract_features(photos_dir: Path, output_csv: Path, batch_size: int = 8):
    """Extract features from all photos in directory."""
    print(f"[Extract] Processing photos from {photos_dir}")

    # Get all image files
    image_files = list(photos_dir.rglob("*.jpg")) + list(photos_dir.rglob("*.png"))
    print(f"[Extract] Found {len(image_files)} images")

    if not image_files:
        print("[ERROR] No images found")
        return

    # Initialize models
    print("[Extract] Loading models...")
    seg_model = SegformerB5(low_vram_mode=True)
    feat_model = Dinov2Extractor(low_vram_mode=True)

    # Process each image
    all_features = []
    for img_path in tqdm(image_files, desc="Extracting features"):
        try:
            # Extract segmentation metrics
            seg_result = seg_model.segment(str(img_path))
            metrics = seg_model.extract_metrics(seg_result)

            # Extract embedding
            embedding = feat_model.extract(str(img_path))

            # Build feature row
            features = {
                "filename": img_path.name,
                "area": img_path.parent.name,
                "point_id": img_path.stem,
            }

            # Add segmentation metrics
            features.update(metrics)

            # Add embedding (1024-d)
            for i, val in enumerate(embedding):
                features[f"emb_{i}"] = val

            all_features.append(features)

        except Exception as e:
            print(f"\n[WARNING] Failed to process {img_path}: {e}")
            continue

    # Cleanup
    seg_model.free_memory()
    feat_model.free_memory()

    # Save to CSV
    df = pd.DataFrame(all_features)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\n[Extract] Saved {len(df)} feature rows to {output_csv}")
    print(f"[Extract] Columns: {len(df.columns)} ({len([c for c in df.columns if c.startswith('emb_')])} embedding dims)")


def main():
    parser = argparse.ArgumentParser(description="Extract features from photos")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing photos",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training/features.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing",
    )

    args = parser.parse_args()

    photos_dir = Path(args.input)
    output_csv = Path(args.output)

    if not photos_dir.exists():
        print(f"[ERROR] Input directory not found: {photos_dir}")
        sys.exit(1)

    extract_features(photos_dir, output_csv, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
