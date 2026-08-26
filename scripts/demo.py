#!/usr/bin/env python3
"""
Demo script untuk melihat hasil UVIP AI pipeline tanpa perlu trained models.
Script ini menggunakan mock predictions untuk demonstrate pipeline.
"""

import cv2
import numpy as np
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def demo_image(image_path):
    """Demo pipeline pada single image."""
    print(f"\n{'='*70}")
    print(f"UVIP AI DEMO - Image Analysis")
    print(f"{'='*70}")
    print(f"Input: {image_path}\n")

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Cannot load image {image_path}")
        return

    print(f"[OK] Image loaded: {img.shape[1]}x{img.shape[0]} pixels")

    # Step 1: Privacy Guard
    print("\n[1/4] Privacy Guard (YOLOv8n)")
    print("  Detecting faces and license plates...")
    from uvip_ai.privacy.guard import PrivacyGuard
    guard = PrivacyGuard()
    result = guard.process_image(img)
    masked_img = result['blurred_image']
    detections = result['detections']
    print(f"  ✓ Found {len(detections)} objects to blur")
    for i, det in enumerate(detections, 1):
        print(f"    {i}. {det['class']} (confidence: {det['confidence']:.2f})")

    # Step 2: Segmentation
    print("\n[2/4] 🏙️  Urban Segmentation (SegFormer-B0)")
    print("  Analyzing urban elements...")
    from uvip_ai.segmentation.segformer import SegformerB5
    seg = SegformerB5(low_vram_mode=True)
    result = seg.infer(masked_img)
    seg_map = result['seg_map']
    metrics = result['metrics']

    print("  ✓ Urban Metrics:")
    print(f"    Green Coverage:      {metrics['green_coverage_pct']:5.2f}%")
    print(f"    Building Coverage:   {metrics['building_coverage_pct']:5.2f}%")
    print(f"    Walkability Ratio:   {metrics['walkability_ratio']:5.2f}")
    print(f"    Visual Clutter:      {metrics['visual_clutter_index']:5.2f}")
    print(f"    Sky Visibility:      {metrics['sky_visibility_pct']:5.2f}%")

    # Step 3: Feature Extraction
    print("\n[3/4] 🧠 Feature Extraction (DINOv2-Base)")
    print("  Extracting visual features...")
    from uvip_ai.features.dinov2 import Dinov2Extractor
    feat = Dinov2Extractor(low_vram_mode=True)
    embedding = feat.extract(masked_img)
    print(f"  ✓ Extracted {len(embedding)}-dimensional embedding")
    print(f"    Embedding norm: {np.linalg.norm(embedding):.4f}")

    # Step 4: Mock Perception Prediction
    print("\n[4/4] 🎯 Perception Prediction (Mock)")
    print("  Generating perception scores...")

    # Generate mock predictions based on metrics
    # This simulates what the trained XGBoost models would do
    beauty = min(10, max(0, metrics['green_coverage_pct'] * 0.15 + metrics['sky_visibility_pct'] * 0.1 + 3))
    safety = min(10, max(0, metrics['walkability_ratio'] * 5 + 4))
    comfort = min(10, max(0, metrics['green_coverage_pct'] * 0.12 - metrics['visual_clutter_index'] * 2 + 5))
    uvi = min(10, max(0, (beauty + safety + comfort) / 3))

    predictions = {
        'beauty': beauty,
        'safety': safety,
        'comfort': comfort,
        'uvi': uvi
    }

    for target, score in predictions.items():
        bar = "█" * int(score)
        emoji = "💎" if target == "beauty" else "🛡️" if target == "safety" else "😌" if target == "comfort" else "🌡️"
        print(f"  {emoji} {target.capitalize():10s}: {score:5.2f}/10 {bar}")

    # Save results
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)

    image_name = Path(image_path).stem

    # Save masked image
    masked_path = output_dir / f"{image_name}_masked.jpg"
    cv2.imwrite(str(masked_path), masked_img)

    # Save segmentation mask
    mask_colored = colorize_mask(seg_map)
    mask_path = output_dir / f"{image_name}_segmentation.png"
    cv2.imwrite(str(mask_path), mask_colored)

    # Save overlay (original + mask blended)
    overlay = cv2.addWeighted(img, 0.6, mask_colored, 0.4, 0)
    overlay_path = output_dir / f"{image_name}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    # Save results JSON
    results = {
        'image': str(image_path),
        'image_size': f"{img.shape[1]}x{img.shape[0]}",
        'privacy': {
            'detections_count': len(detections),
            'detections': detections
        },
        'segmentation_metrics': metrics,
        'embedding_dimension': len(embedding),
        'predictions': predictions,
        'note': 'Predictions are MOCK (not from trained model)'
    }

    json_path = output_dir / f"{image_name}_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print("📊 RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Image: {image_path}")
    print(f"Size: {img.shape[1]}x{img.shape[0]} pixels")
    print(f"\n🔒 Privacy: {len(detections)} objects blurred")
    print(f"\n🏙️  Urban Metrics:")
    print(f"   🌳 Green Coverage:    {metrics['green_coverage_pct']:5.2f}%")
    print(f"   🏢 Building Coverage: {metrics['building_coverage_pct']:5.2f}%")
    print(f"   🚶 Walkability:       {metrics['walkability_ratio']:5.2f}")
    print(f"   📊 Visual Clutter:    {metrics['visual_clutter_index']:5.2f}")
    print(f"   ☁️  Sky Visibility:    {metrics['sky_visibility_pct']:5.2f}%")
    print(f"\n🎯 Perception Scores (MOCK):")
    for target, score in predictions.items():
        emoji = "💎" if target == "beauty" else "🛡️" if target == "safety" else "😌" if target == "comfort" else "🌡️"
        print(f"   {emoji} {target.capitalize():10s}: {score:5.2f}/10")

    print(f"\n💾 Output Files:")
    print(f"   📸 {masked_path}")
    print(f"   🎨 {mask_path}")
    print(f"   🖼️  {overlay_path}")
    print(f"   📄 {json_path}")
    print(f"\n⚠️  Note: Predictions are MOCK (demo mode)")
    print(f"   Train models to get real predictions!")
    print(f"{'='*70}\n")

    return results


def demo_video(video_path, sample_frames=5):
    """Demo pipeline pada video."""
    print(f"\n{'='*70}")
    print(f"🌆 UVIP AI DEMO - Video Analysis")
    print(f"{'='*70}")
    print(f"Input: {video_path}\n")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"📹 Video Info:")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Sampling: {sample_frames} frames\n")

    # Calculate frame indices to sample
    frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)

    all_results = []

    for idx, frame_idx in enumerate(frame_indices):
        print(f"\n{'─'*70}")
        print(f"[{idx+1}/{sample_frames}] Processing frame {frame_idx} (t={frame_idx/fps:.2f}s)")
        print(f"{'─'*70}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"  ⚠ Cannot read frame {frame_idx}")
            continue

        # Save frame temporarily
        temp_path = Path("demo_outputs") / f"temp_frame_{frame_idx}.jpg"
        temp_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(temp_path), frame)

        # Process frame
        try:
            results = demo_image(str(temp_path))
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
        print(f"\n{'='*70}")
        print("📊 VIDEO SUMMARY")
        print(f"{'='*70}")
        print(f"Video: {video_path}")
        print(f"Frames processed: {len(all_results)}")

        # Calculate averages
        avg_metrics = {}
        for key in all_results[0]['segmentation_metrics'].keys():
            values = [r['segmentation_metrics'][key] for r in all_results]
            avg_metrics[key] = np.mean(values)

        avg_predictions = {}
        for target in ['beauty', 'safety', 'comfort', 'uvi']:
            scores = [r['predictions'][target] for r in all_results]
            avg_predictions[target] = np.mean(scores)

        print(f"\n🏙️  Average Urban Metrics:")
        print(f"   🌳 Green Coverage:    {avg_metrics['green_coverage_pct']:5.2f}%")
        print(f"   🏢 Building Coverage: {avg_metrics['building_coverage_pct']:5.2f}%")
        print(f"   🚶 Walkability:       {avg_metrics['walkability_ratio']:5.2f}")
        print(f"   📊 Visual Clutter:    {avg_metrics['visual_clutter_index']:5.2f}")
        print(f"   ☁️  Sky Visibility:    {avg_metrics['sky_visibility_pct']:5.2f}%")

        print(f"\n🎯 Average Perception Scores (MOCK):")
        for target, score in avg_predictions.items():
            emoji = "💎" if target == "beauty" else "🛡️" if target == "safety" else "😌" if target == "comfort" else "🌡️"
            bar = "█" * int(score)
            print(f"   {emoji} {target.capitalize():10s}: {score:5.2f}/10 {bar}")

        # Save video summary
        output_dir = Path("demo_outputs")
        video_name = Path(video_path).stem
        summary_path = output_dir / f"{video_name}_summary.json"

        summary = {
            'video': str(video_path),
            'frames_processed': len(all_results),
            'average_metrics': avg_metrics,
            'average_predictions': avg_predictions,
            'timeline': [
                {
                    'frame': r['frame_idx'],
                    'timestamp': r['timestamp'],
                    'metrics': r['segmentation_metrics'],
                    'predictions': r['predictions']
                }
                for r in all_results
            ]
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 Output Files:")
        print(f"   📄 {summary_path}")
        print(f"   📁 {output_dir}/ (contains frame-by-frame results)")

    print(f"\n⚠️  Note: Predictions are MOCK (demo mode)")
    print(f"   Train models to get real predictions!")
    print(f"{'='*70}\n")


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

    parser = argparse.ArgumentParser(
        description="🌆 UVIP AI Demo - Test model pada gambar/video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single image
  python scripts/demo.py data/extracted/photos/KAYUTANGAN/KAYUTANGAN_ST-01.jpg

  # Test video (sample 5 frames)
  python scripts/demo.py test_video.mp4

  # Test video (sample 10 frames)
  python scripts/demo.py test_video.mp4 --frames 10

Output:
  Results akan disimpan di demo_outputs/ directory:
  - *_masked.jpg         : Image dengan privacy masking
  - *_segmentation.png   : Segmentation mask (colored)
  - *_overlay.png        : Original + mask blended
  - *_results.json       : Detailed results (metrics, predictions)
        """
    )

    parser.add_argument("input", help="Path to image or video file")
    parser.add_argument("--frames", type=int, default=5,
                       help="Number of frames to sample from video (default: 5)")

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"❌ Error: File not found: {input_path}")
        sys.exit(1)

    # Check if video or image
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

    if input_path.suffix.lower() in video_extensions:
        demo_video(input_path, args.frames)
    else:
        demo_image(input_path)


if __name__ == "__main__":
    main()
