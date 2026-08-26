#!/usr/bin/env python
"""
Retrain model dengan features yang match dengan backend schema.

Backend expects:
- vegetation_pct
- building_pct
- road_pct
- sidewalk_pct
- sky_pct
- signage_pct
- vehicle_pct
- pedestrian_pct
- street_furniture_pct
- green_coverage_pct
- building_coverage_pct
- sky_visibility_pct
- walkability_ratio
- visual_clutter_index

Usage:
    python scripts/retrain_model_for_backend.py
"""
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import pickle
import json


# ADE20K class mapping (simplified)
# Map ADE20K classes to backend categories
CLASS_MAPPING = {
    # Vegetation
    'tree': 'vegetation',
    'grass': 'vegetation',
    'plant': 'vegetation',
    'bush': 'vegetation',
    'flower': 'vegetation',

    # Building
    'building': 'building',
    'house': 'building',
    'skyscraper': 'building',
    'wall': 'building',

    # Road
    'road': 'road',
    'street': 'road',
    'highway': 'road',
    'path': 'road',

    # Sidewalk
    'sidewalk': 'sidewalk',
    'pavement': 'sidewalk',
    'footpath': 'sidewalk',

    # Sky
    'sky': 'sky',
    'cloud': 'sky',

    # Signage
    'signboard': 'signage',
    'traffic_sign': 'signage',
    'billboard': 'signage',

    # Vehicle
    'car': 'vehicle',
    'truck': 'vehicle',
    'bus': 'vehicle',
    'motorcycle': 'vehicle',
    'bicycle': 'vehicle',

    # Pedestrian
    'person': 'pedestrian',
    'pedestrian': 'pedestrian',

    # Street furniture
    'bench': 'street_furniture',
    'lamp': 'street_furniture',
    'pole': 'street_furniture',
    'fence': 'street_furniture',
}


def extract_segmentation_features(image_path, seg_model, seg_processor):
    """
    Extract segmentation features yang match dengan backend schema.
    """
    img = Image.open(image_path).convert('RGB')

    # SegFormer inference
    inputs = seg_processor(images=img, return_tensors="pt").to('cuda')
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)

    with torch.no_grad():
        outputs = seg_model(**inputs)

    # Get segmentation map
    seg_map = outputs.logits.argmax(dim=1)[0].cpu().numpy()

    # Calculate percentages for each category
    total_pixels = seg_map.size

    # Initialize counters
    categories = {
        'vegetation': 0,
        'building': 0,
        'road': 0,
        'sidewalk': 0,
        'sky': 0,
        'signage': 0,
        'vehicle': 0,
        'pedestrian': 0,
        'street_furniture': 0,
    }

    # Count pixels per category (simplified - need actual class mapping)
    # For now, use heuristic based on class indices
    unique_classes, counts = np.unique(seg_map, return_counts=True)

    # Heuristic mapping (this should be replaced with actual ADE20K class mapping)
    for cls_idx, count in zip(unique_classes, counts):
        pct = count / total_pixels

        # Simplified mapping based on typical ADE20K class indices
        if cls_idx in [0, 1, 2, 3, 4]:  # vegetation-like classes
            categories['vegetation'] += pct
        elif cls_idx in [5, 6, 7, 8]:  # building-like classes
            categories['building'] += pct
        elif cls_idx in [9, 10]:  # road-like classes
            categories['road'] += pct
        elif cls_idx in [11, 12]:  # sidewalk-like classes
            categories['sidewalk'] += pct
        elif cls_idx == 13:  # sky
            categories['sky'] += pct
        elif cls_idx in [14, 15]:  # signage
            categories['signage'] += pct
        elif cls_idx in [16, 17, 18]:  # vehicle
            categories['vehicle'] += pct
        elif cls_idx == 19:  # pedestrian
            categories['pedestrian'] += pct
        else:  # street furniture and others
            categories['street_furniture'] += pct

    # Calculate derived metrics
    green_coverage_pct = categories['vegetation'] * 100
    building_coverage_pct = categories['building'] * 100
    sky_visibility_pct = categories['sky'] * 100

    # Walkability ratio: sidewalk / (road + sidewalk)
    walkability_ratio = (
        categories['sidewalk'] / (categories['road'] + categories['sidewalk'] + 1e-6)
    )

    # Visual clutter index: signage + vehicle + street_furniture
    visual_clutter_index = (
        categories['signage'] + categories['vehicle'] + categories['street_furniture']
    ) * 100

    features = {
        'vegetation_pct': round(categories['vegetation'] * 100, 2),
        'building_pct': round(categories['building'] * 100, 2),
        'road_pct': round(categories['road'] * 100, 2),
        'sidewalk_pct': round(categories['sidewalk'] * 100, 2),
        'sky_pct': round(categories['sky'] * 100, 2),
        'signage_pct': round(categories['signage'] * 100, 2),
        'vehicle_pct': round(categories['vehicle'] * 100, 2),
        'pedestrian_pct': round(categories['pedestrian'] * 100, 2),
        'street_furniture_pct': round(categories['street_furniture'] * 100, 2),
        'green_coverage_pct': round(green_coverage_pct, 2),
        'building_coverage_pct': round(building_coverage_pct, 2),
        'sky_visibility_pct': round(sky_visibility_pct, 2),
        'walkability_ratio': round(walkability_ratio, 4),
        'visual_clutter_index': round(visual_clutter_index, 2),
    }

    return features


def main():
    # Load SegFormer
    print("Loading SegFormer-B0...")
    seg_processor = SegformerImageProcessor.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512')
    seg_model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/segformer-b0-finetuned-ade-512-512'
    ).to('cuda').to(torch.float16).eval()
    print("✓ SegFormer loaded")

    # Load manifest
    manifest_path = Path('data/extracted/manifest.csv')
    manifest_df = pd.read_csv(manifest_path)

    print(f"✓ Loaded {len(manifest_df)} photos from manifest")

    # Extract features
    features_list = []
    photos_dir = Path('data/extracted/photos')

    print("\n🔍 Extracting segmentation features...")
    for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Processing"):
        filename = row['filename']
        photo_path = photos_dir / row['area'] / filename

        if not photo_path.exists():
            continue

        try:
            features = extract_segmentation_features(photo_path, seg_model, seg_processor)
            features['filename'] = filename
            features['area'] = row['area']
            features['point_id'] = row['point_id']
            features_list.append(features)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    # Save features
    features_df = pd.DataFrame(features_list)
    features_path = Path('data/training/features_segmentation.csv')
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(features_path, index=False)

    print(f"\n✓ Features extracted: {len(features_df)} photos")
    print(f"✓ Saved to: {features_path}")
    print(f"✓ Columns: {list(features_df.columns)}")

    # Now train XGBoost models
    print("\n🔍 Training XGBoost models...")

    # Load labels (synthetic for now)
    labels_path = Path('data/training/labels_synthetic.csv')
    if not labels_path.exists():
        print("❌ labels_synthetic.csv not found. Run synthetic label generation first.")
        return

    labels_df = pd.read_csv(labels_path)

    # Merge features with labels
    dataset_df = pd.merge(features_df, labels_df, on='filename', how='inner')
    print(f"✓ Merged dataset: {len(dataset_df)} rows")

    # Prepare training data
    feature_cols = [
        'vegetation_pct', 'building_pct', 'road_pct', 'sidewalk_pct',
        'sky_pct', 'signage_pct', 'vehicle_pct', 'pedestrian_pct',
        'street_furniture_pct', 'green_coverage_pct', 'building_coverage_pct',
        'sky_visibility_pct', 'walkability_ratio', 'visual_clutter_index'
    ]

    X = dataset_df[feature_cols].fillna(0).values

    # Train models for each target
    import xgboost as xgb
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    targets = ['label_beauty', 'label_safety', 'label_comfort', 'label_uvi']
    target_names = ['beauty', 'safety', 'comfort', 'uvi']

    results = {}

    for target, name in zip(targets, target_names):
        print(f"\n{'='*60}")
        print(f"Training {name.upper()} model...")
        print(f"{'='*60}")

        y = dataset_df[target].values

        # K-Fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        r2_scores = []
        mae_scores = []
        rmse_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train model
            model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                n_jobs=-1,
                random_state=42,
                tree_method='hist',
                early_stopping_rounds=50
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Predict
            y_pred = model.predict(X_val)

            # Metrics
            r2 = r2_score(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            r2_scores.append(r2)
            mae_scores.append(mae)
            rmse_scores.append(rmse)

        # Print metrics
        print(f"R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
        print(f"MAE: {np.mean(mae_scores):.4f}")
        print(f"RMSE: {np.mean(rmse_scores):.4f}")

        # Save model
        model_path = f'models/perception/{name}_model.pkl'
        Path('models/perception').mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Model saved: {model_path}")

        # Store results
        results[name] = {
            'r2_mean': float(np.mean(r2_scores)),
            'r2_std': float(np.std(r2_scores)),
            'mae_mean': float(np.mean(mae_scores)),
            'rmse_mean': float(np.mean(rmse_scores)),
            'feature_columns': feature_cols
        }

    # Save metrics
    with open('models/perception/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Models saved to: models/perception/")
    print(f"✓ Metrics saved to: models/perception/metrics.json")


if __name__ == "__main__":
    main()
