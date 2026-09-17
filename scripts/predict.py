#!/usr/bin/env python3
"""
Script untuk inference menggunakan model yang sudah trained.

Usage:
    python scripts/predict.py --image path/to/image.jpg --models models/perception

Output:
- JSON dengan skor Beauty, Safety, Comfort, UVI
- SHAP values untuk explainability
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uvip_ai.features.dinov2 import Dinov2Extractor
from uvip_ai.pipeline.build_dataset import DatasetBuilder
from uvip_ai.segmentation.segformer import SegformerB5
from uvip_ai.training.xgboost_model import XGBoostPerceptionTrainer


def load_models(models_dir: Path):
    """Load trained XGBoost models."""
    import pickle

    models = {}
    for target in XGBoostPerceptionTrainer.TARGETS:
        model_path = models_dir / f"{target}_xgb.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        with model_path.open("rb") as f:
            models[target] = pickle.load(f)
    return models


def predict_single(image_path: str, models: dict, low_vram_mode: bool = True):
    """Predict perception scores for a single image."""
    print(f"[Predict] Processing {image_path}")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # Initialize models
    seg_model = SegformerB5(low_vram_mode=low_vram_mode)
    feat_model = Dinov2Extractor(low_vram_mode=low_vram_mode)

    # Segmentation
    print("[Predict] Running segmentation...")
    seg_result = seg_model.segment(img)
    metrics = seg_model.extract_metrics(seg_result)

    # Feature extraction
    print("[Predict] Extracting features...")
    embedding = feat_model.extract(img)

    # Build feature vector
    features = {}
    features["seg_green_coverage_pct"] = metrics["green_coverage_pct"]
    features["seg_building_coverage_pct"] = metrics["building_coverage_pct"]
    features["seg_walkability_ratio"] = metrics["walkability_ratio"]
    features["seg_visual_clutter_index"] = metrics["visual_clutter_index"]
    features["seg_sky_visibility_pct"] = metrics["sky_visibility_pct"]

    for i, val in enumerate(embedding):
        features[f"emb_{i}"] = val

    df = pd.DataFrame([features])

    # Predict
    print("[Predict] Running XGBoost inference...")
    predictions = {}
    for target, model in models.items():
        pred = model.predict(df)[0]
        predictions[target] = float(pred)

    # Cleanup
    seg_model.free_memory()
    feat_model.free_memory()

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Predict perception scores")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="models/perception",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        default=True,
        help="Enable low VRAM mode (for 6GB GPUs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (optional, prints to stdout if not specified)",
    )

    args = parser.parse_args()

    # Load models
    models_dir = Path(args.models)
    if not models_dir.exists():
        print(f"[ERROR] Models directory not found: {models_dir}")
        sys.exit(1)

    print(f"[Predict] Loading models from {models_dir}")
    models = load_models(models_dir)

    # Predict
    predictions = predict_single(args.image, models, low_vram_mode=args.low_vram)

    # Output
    result = {
        "image": args.image,
        "predictions": predictions,
    }

    if args.output:
        output_path = Path(args.output)
        with output_path.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[Output] Results saved to {output_path}")
    else:
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        for target, score in predictions.items():
            print(f"{target:12s}: {score:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
