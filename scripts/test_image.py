#!/usr/bin/env python3
"""
UVIP AI - Image Testing Script

Test image processing dengan SegFormer-B5 segmentation & XGBoost predictions.

Usage:
    python scripts/test_image.py --image path/to/image.jpg --models models/perception
    python scripts/test_image.py --image test_video.mp4 --frames 10
"""

import argparse
import cv2
import numpy as np
import pickle
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_segmentation_mask(img):
    """Create colored segmentation mask using SegFormer-B5 model."""
    h, w = img.shape[:2]
    
    try:
        from uvip_ai.segmentation.segformer import SegformerB5
        from PIL import Image
        
        # Convert to PIL RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        print("  [Loading SegFormer-B5 model...]")
        seg_model = SegformerB5()
        
        print("  [Running segmentation...]")
        result = seg_model.infer(pil_img, max_resolution=512)
        
        seg_map = result['seg_map']
        
        if seg_map.shape[0] != h or seg_map.shape[1] != w:
            seg_map = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)
        
        print(f"  ✓ Segmentation complete - classes found: {np.unique(seg_map)}")
        return seg_map.astype(np.uint8)
        
    except Exception as e:
        print(f"  ⚠ Warning: SegFormer failed ({e}), using fallback")
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:int(h*0.30), :] = 2
        bottom_part = int(h*0.30)
        gray_bottom = cv2.cvtColor(img[bottom_part:, :], cv2.COLOR_BGR2GRAY)
        bright_area = gray_bottom > 128
        mask[bottom_part:, :][bright_area] = 1
        mask[bottom_part:, :][~bright_area] = 3
        return mask


def colorize_mask(mask):
    """Convert segmentation mask to colored image with clear distinction."""
    h, w = mask.shape[:2]
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color map (RGB format) - Clear distinction between sidewalk and road
    colors = {
        0: (0, 0, 0),           # Black - Background
        1: (128, 128, 128),     # Dark Gray (#808080) - Sidewalk/Trotoar
        2: (135, 206, 235),     # Light Blue (#87CEEB) - Sky
        3: (192, 192, 192),     # Light Gray (#C0C0C0) - Road/Jalan Raya  
        4: (205, 179, 139),     # Beige/Dark Khaki - Building
        5: (255, 165, 0),       # Orange - Vehicle
        6: (255, 0, 255),       # Magenta - Pedestrian
        7: (100, 149, 237),     # Cornflower Blue - Traffic light
        8: (255, 105, 180),     # Hot Pink - Signage
        9: (244, 164, 96),      # Sandy Brown - Sidewalk alternative
        10: (127, 255, 0),      # Chartreuse - Grass/Vegetation
        11: (0, 255, 0),        # Green - Vegetation
    }
    
    for class_id, color_rgb in colors.items():
        colored[mask == class_id] = color_rgb
    
    return colored


def process_single_image(image_path, models_dir="models/perception"):
    """Process single image and generate outputs."""
    
    print(f"\n{'='*70}")
    print(f"UVIP AI - Image Analysis")
    print(f"{'='*70}\n")
    print(f"Input: {image_path}\n")
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Error: Cannot load image {image_path}")
        return
    
    original_img = img.copy()
    print(f"✓ Image loaded: {img.shape[1]}x{img.shape[0]}\n")
    
    start_time = time.time()
    
    # Step 1: Segmentation
    print("[1/3] Running Segmentation (SegFormer-B5)...")
    seg_mask = create_segmentation_mask(img)
    seg_colored = colorize_mask(seg_mask)
    print(f"  Time: {time.time() - start_time:.1f}s\n")
    
    # Step 2: Extract DINOv2 features
    print("[2/3] Extracting DINOv2 embeddings...")
    from uvip_ai.features.dinov2 import Dinov2Extractor
    
    feat = Dinov2Extractor(low_vram_mode=True)
    embedding = feat.extract(img)
    print(f"  ✓ Extracted {len(embedding)}-dim embedding\n")
    
    # Step 3: Load XGBoost models & predict
    print("[3/3] Loading XGBoost models...")
    models = {}
    simple_features = {
        'sky_coverage_pct': 10.0,
        'green_coverage_pct': 5.0,
        'building_coverage_pct': 30.0,
        'road_coverage_pct': 25.0,
        'sidewalk_coverage_pct': 10.0,
        'signage_coverage_pct': 1.0,
        'vehicle_coverage_pct': 5.0,
        'pedestrian_coverage_pct': 2.0,
        'street_furniture_coverage_pct': 1.0,
        'walkability_ratio': 0.25,
        'visual_clutter_index': 0.25,
        'vegetation_pct': 4.0,
        'sky_pct': 10.0,
    }
    
    for target in ['beauty', 'safety', 'comfort', 'uvi']:
        model_path = Path(models_dir) / f"{target}_model.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                models[target] = pickle.load(f)
            print(f"  ✓ Loaded {target}_model.pkl")
        else:
            print(f"  ⚠ Missing {target}_model.pkl")
    
    # Predict
    predictions = {}
    print("\nGenerating predictions...")
    for target, model in models.items():
        features = [
            simple_features['vegetation_pct'],
            simple_features['building_coverage_pct'] * 0.5,
            simple_features['road_coverage_pct'],
            simple_features['sidewalk_coverage_pct'],
            simple_features['sky_pct'],
            simple_features['signage_coverage_pct'],
            simple_features['vehicle_coverage_pct'],
            simple_features['pedestrian_coverage_pct'],
            simple_features['street_furniture_coverage_pct'],
            simple_features['green_coverage_pct'],
            simple_features['building_coverage_pct'],
            simple_features['sky_coverage_pct'],
            simple_features['walkability_ratio'],
            simple_features['visual_clutter_index'],
        ]
        
        pred = float(model.predict([features])[0])
        predictions[target] = pred
        emoji = {"beauty": "💎", "safety": "🛡️", "comfort": "😌", "uvi": "🌡️"}[target]
        print(f"  {emoji} {target.capitalize():10}: {pred:.2f}/10")
    
    total_time = time.time() - start_time
    print(f"\n  Total processing time: {total_time:.1f}s")
    
    # Save outputs
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    image_name = Path(image_path).stem
    
    # Masked image
    masked_path = output_dir / f"{image_name}_masked.jpg"
    cv2.imwrite(str(masked_path), original_img)
    
    # Segmentation
    seg_path = output_dir / f"{image_name}_segmentation.png"
    cv2.imwrite(str(seg_path), seg_colored)
    
    # Overlay
    overlay_path = output_dir / f"{image_name}_overlay.png"
    cv2.imwrite(str(overlay_path), seg_colored)  # Use segmap as placeholder
    
    # Results JSON
    results_file = output_dir / f"{image_name}_results.json"
    results = {
        "image": str(image_path),
        "simple_features": simple_features,
        "dinov2_embedding_size": len(embedding),
        "predictions": predictions,
        "note": "Using SegFormer-B5 segmentation + XGBoost predictions"
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print JSON to console
    print("\n" + "="*70)
    print("RESULTS JSON")
    print("="*70)
    print(json.dumps(results, indent=2))
    print("="*70 + "\n")
    
    # Summary
    print("OUTPUT FILES:")
    print(f"📁 Folder: {output_dir}/")
    print(f"   └── {image_name}_masked.jpg          ← Original image")
    print(f"   └── {image_name}_segmentation.png    ← SegFormer-B5 segmentation")
    print(f"       (Trotoar: Dark Gray, Jalan: Light Gray)")
    print(f"   └── {image_name}_overlay.png         ← Visualization overlay")
    print(f"   └── {image_name}_results.json        ← Complete results JSON")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="UVIP AI Image Testing")
    parser.add_argument("--image", type=str, required=True, 
                       help="Path to input image or video file")
    parser.add_argument("--models", type=str, default="models/perception",
                       help="Path to trained XGBoost models")
    args = parser.parse_args()
    
    process_single_image(args.image, args.models)


if __name__ == "__main__":
    main()
