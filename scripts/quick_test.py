#!/usr/bin/env python3
"""
Quick test script untuk melihat hasil model UVIP AI.
Script ini lebih simple dan langsung menampilkan hasil.
"""

import cv2
import numpy as np
import pickle
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_single_image(image_path, models_dir="models/perception"):
    """Test single image dan tampilkan hasil."""
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print(f"{'='*60}\n")

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Error: Cannot load image {image_path}")
        return

    print(f"✓ Image loaded: {img.shape[1]}x{img.shape[0]}")

    # Step 1: Privacy Guard
    print("\n[1/4] Running Privacy Guard (YOLOv8n)...")
    from uvip_ai.privacy.guard import PrivacyGuard
    guard = PrivacyGuard(low_vram_mode=True)
    masked_img, detections = guard.process(img)
    print(f"  ✓ Found {len(detections)} objects to blur")

    # Step 2: Segmentation
    print("\n[2/4] Running Segmentation (SegFormer)...")
    from uvip_ai.segmentation.segformer import SegformerB5
    seg = SegformerB5(low_vram_mode=True)
    seg_result = seg.segment(masked_img)
    metrics = seg.extract_metrics(seg_result)

    print("  ✓ Urban Metrics:")
    for key, value in metrics.items():
        print(f"    - {key}: {value:.2f}%")

    # Step 3: Feature Extraction
    print("\n[3/4] Extracting Features (DINOv2)...")
    from uvip_ai.features.dinov2 import Dinov2Extractor
    feat = Dinov2Extractor(low_vram_mode=True)
    embedding = feat.extract(masked_img)
    print(f"  ✓ Extracted {len(embedding)}-dimensional embedding")

    # Step 4: Perception Prediction
    print("\n[4/4] Running Perception Prediction (XGBoost)...")
    models_path = Path(models_dir)

    if not models_path.exists():
        print(f"  ⚠ Models directory not found: {models_path}")
        print("  ⚠ Skipping perception prediction")
        predictions = {}
    else:
        # Build feature vector
        features = list(metrics.values()) + list(embedding)
        feature_names = list(metrics.keys()) + [f'emb_{i}' for i in range(len(embedding))]

        predictions = {}
        for target in ['beauty', 'safety', 'comfort', 'uvi']:
            model_path = models_path / f"{target}_xgb.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                pred = model.predict([features])[0]
                predictions[target] = float(pred)
                print(f"  ✓ {target.capitalize()}: {pred:.2f}/10")
            else:
                print(f"  ⚠ Model not found: {model_path}")

    # Save results
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)

    image_name = Path(image_path).stem

    # Save masked image
    masked_path = output_dir / f"{image_name}_masked.jpg"
    cv2.imwrite(str(masked_path), masked_img)
    print(f"\n✓ Saved masked image: {masked_path}")

    # Save segmentation mask
    mask = seg_result['mask']
    mask_colored = colorize_mask(mask)
    mask_path = output_dir / f"{image_name}_segmentation.png"
    cv2.imwrite(str(mask_path), mask_colored)
    print(f"✓ Saved segmentation mask: {mask_path}")

    # Save metrics JSON
    results = {
        'image': str(image_path),
        'metrics': metrics,
        'predictions': predictions,
        'detections': len(detections)
    }

    json_path = output_dir / f"{image_name}_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results: {json_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Privacy: {len(detections)} objects blurred")
    print(f"\nUrban Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}%")

    if predictions:
        print(f"\nPerception Scores:")
        for target, score in predictions.items():
            bar = "█" * int(score)
            print(f"  {target.capitalize():10s}: {score:5.2f}/10 {bar}")

    print(f"\n✓ Results saved to: {output_dir}/")
    print(f"{'='*60}\n")

    return results


def test_video(video_path, models_dir="models/perception", sample_frames=5):
    """Test video dengan sample beberapa frame."""
    print(f"\n{'='*60}")
    print(f"Testing Video: {video_path}")
    print(f"{'='*60}\n")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video Info:")
    print(f"  - Total frames: {total_frames}")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Duration: {duration:.2f}s")
    print(f"  - Sampling: {sample_frames} frames\n")

    # Calculate frame indices to sample
    frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)

    all_results = []

    for idx, frame_idx in enumerate(frame_indices):
        print(f"\n[{idx+1}/{sample_frames}] Processing frame {frame_idx}...")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"  ⚠ Cannot read frame {frame_idx}")
            continue

        # Save frame temporarily
        temp_path = Path("test_outputs") / f"temp_frame_{frame_idx}.jpg"
        temp_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(temp_path), frame)

        # Process frame
        try:
            results = test_single_image(str(temp_path), models_dir)
            if results:
                results['frame_idx'] = frame_idx
                results['timestamp'] = frame_idx / fps if fps > 0 else 0
                all_results.append(results)
        except Exception as e:
            print(f"  ❌ Error processing frame: {e}")
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

    cap.release()

    # Print video summary
    if all_results:
        print(f"\n{'='*60}")
        print("VIDEO SUMMARY")
        print(f"{'='*60}")
        print(f"Video: {video_path}")
        print(f"Frames processed: {len(all_results)}")

        # Calculate averages
        avg_predictions = {}
        for target in ['beauty', 'safety', 'comfort', 'uvi']:
            scores = [r['predictions'].get(target, 0) for r in all_results if target in r['predictions']]
            if scores:
                avg_predictions[target] = np.mean(scores)

        if avg_predictions:
            print(f"\nAverage Perception Scores:")
            for target, score in avg_predictions.items():
                bar = "█" * int(score)
                print(f"  {target.capitalize():10s}: {score:5.2f}/10 {bar}")

        # Save video summary
        output_dir = Path("test_outputs")
        video_name = Path(video_path).stem
        summary_path = output_dir / f"{video_name}_summary.json"

        summary = {
            'video': str(video_path),
            'frames_processed': len(all_results),
            'average_predictions': avg_predictions,
            'timeline': [
                {
                    'frame': r['frame_idx'],
                    'timestamp': r['timestamp'],
                    'predictions': r['predictions']
                }
                for r in all_results
            ]
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✓ Video summary saved: {summary_path}")

    print(f"\n{'='*60}\n")


def colorize_mask(mask):
    """Convert segmentation mask ke colored image."""
    colors = {
        0: [128, 64, 128],    # road
        1: [244, 35, 232],    # sidewalk
        2: [70, 70, 70],      # building
        3: [102, 102, 156],   # wall
        4: [190, 153, 153],   # fence
        5: [153, 153, 153],   # pole
        6: [250, 170, 30],    # traffic light
        7: [220, 220, 0],     # traffic sign
        8: [107, 142, 35],    # vegetation
        9: [152, 251, 152],   # terrain
        10: [70, 130, 180],   # sky
        11: [220, 20, 60],    # person
        12: [255, 0, 0],      # rider
        13: [0, 0, 142],      # car
        14: [0, 0, 70],       # truck
        15: [0, 60, 100],     # bus
        16: [0, 80, 100],     # train
        17: [0, 0, 230],      # motorcycle
        18: [119, 11, 32],    # bicycle
    }

    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in colors.items():
        colored[mask == class_id] = color

    return colored


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Quick test UVIP AI model")
    parser.add_argument("input", help="Path to image or video file")
    parser.add_argument("--models", default="models/perception",
                       help="Path to trained models (default: models/perception)")
    parser.add_argument("--video-frames", type=int, default=5,
                       help="Number of frames to sample from video (default: 5)")

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"❌ Error: File not found: {input_path}")
        sys.exit(1)

    # Check if video or image
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

    if input_path.suffix.lower() in video_extensions:
        test_video(input_path, args.models, args.video_frames)
    else:
        test_single_image(input_path, args.models)


if __name__ == "__main__":
    main()
