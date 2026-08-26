#!/usr/bin/env python3
"""
Test inference script untuk UVIP AI.
Process gambar/video dan tampilkan hasil:
- Privacy masking (blur wajah/plat)
- Segmentation mask (5 metrik urban)
- Perception scores (Beauty, Safety, Comfort, UVI)
- SHAP values (feature importance)
"""

import argparse
import cv2
import numpy as np
import pickle
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uvip_ai.privacy.guard import PrivacyGuard
from uvip_ai.segmentation.segformer import SegformerB5
from uvip_ai.features.dinov2 import Dinov2Extractor
from uvip_ai.explain.shap_explain import ShapExplainer


class UVIPTester:
    """Test UVIP AI pipeline pada gambar/video."""

    def __init__(self, models_dir="models/perception", low_vram_mode=True):
        self.models_dir = Path(models_dir)
        self.low_vram_mode = low_vram_mode

        # Load models
        print("Loading models...")
        self.privacy_guard = PrivacyGuard(low_vram_mode=low_vram_mode)
        self.seg_model = SegformerB5(low_vram_mode=low_vram_mode)
        self.feat_model = Dinov2Extractor(low_vram_mode=low_vram_mode)

        # Load XGBoost models
        self.perception_models = {}
        for target in ['beauty', 'safety', 'comfort', 'uvi']:
            model_path = self.models_dir / f"{target}_xgb.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.perception_models[target] = pickle.load(f)
                print(f"  ✓ Loaded {target} model")
            else:
                print(f"  ⚠ {target} model not found")

        # Load SHAP explainer (optional)
        self.shap_explainer = None
        if self.perception_models:
            # Use first model for SHAP
            first_model = list(self.perception_models.values())[0]
            feature_names = ['green_coverage_pct', 'building_coverage_pct',
                           'walkability_ratio', 'visual_clutter_index',
                           'sky_visibility_pct'] + [f'emb_{i}' for i in range(768)]
            self.shap_explainer = ShapExplainer(first_model, feature_names)

        print("✓ All models loaded\n")

    def process_image(self, image_path, save_results=True, output_dir="outputs"):
        """Process single image dan return semua hasil."""
        print(f"Processing: {image_path}")

        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")

        original_h, original_w = img.shape[:2]

        # Step 1: Privacy masking
        print("  [1/5] Privacy masking...")
        masked_img, detections = self.privacy_guard.process(img)

        # Step 2: Segmentation
        print("  [2/5] Segmentation...")
        seg_result = self.seg_model.segment(masked_img)
        seg_mask = seg_result['mask']
        metrics = self.seg_model.extract_metrics(seg_result)

        # Step 3: Feature extraction
        print("  [3/5] Feature extraction...")
        embedding = self.feat_model.extract(masked_img)

        # Step 4: Perception prediction
        print("  [4/5] Perception prediction...")
        features = {}
        features.update(metrics)
        for i, val in enumerate(embedding):
            features[f'emb_{i}'] = val

        predictions = {}
        for target, model in self.perception_models.items():
            X = np.array([[features.get(f'emb_{i}', 0) for i in range(768)] +
                         [metrics['green_coverage_pct'], metrics['building_coverage_pct'],
                          metrics['walkability_ratio'], metrics['visual_clutter_index'],
                          metrics['sky_visibility_pct']]])
            pred = model.predict(X)[0]
            predictions[target] = float(pred)

        # Step 5: SHAP explainability (optional)
        print("  [5/5] SHAP explainability...")
        shap_values = None
        if self.shap_explainer:
            try:
                X = np.array([[metrics['green_coverage_pct'], metrics['building_coverage_pct'],
                              metrics['walkability_ratio'], metrics['visual_clutter_index'],
                              metrics['sky_visibility_pct']] + list(embedding)])
                shap_values = self.shap_explainer.explain(X)
            except Exception as e:
                print(f"    ⚠ SHAP failed: {e}")

        # Compile results
        results = {
            'image_path': str(image_path),
            'image_size': (original_w, original_h),
            'privacy': {
                'detections': detections,
                'masked_image': masked_img
            },
            'segmentation': {
                'mask': seg_mask,
                'metrics': metrics
            },
            'embedding': embedding,
            'predictions': predictions,
            'shap_values': shap_values
        }

        # Save results
        if save_results:
            self._save_results(results, output_dir)

        return results

    def process_video(self, video_path, save_results=True, output_dir="outputs",
                     process_every_n_frames=30):
        """Process video frame by frame."""
        print(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"  Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")

        results_list = []
        frame_idx = 0
        processed_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every N frames
            if frame_idx % process_every_n_frames == 0:
                print(f"\n  Processing frame {frame_idx}/{total_frames}...")

                # Save frame temporarily
                temp_path = Path(output_dir) / f"temp_frame_{frame_idx}.jpg"
                cv2.imwrite(str(temp_path), frame)

                # Process frame
                try:
                    results = self.process_image(temp_path, save_results=False)
                    results['frame_idx'] = frame_idx
                    results['timestamp'] = frame_idx / fps if fps > 0 else 0
                    results_list.append(results)
                    processed_count += 1

                    # Print predictions
                    print(f"    Timestamp: {results['timestamp']:.2f}s")
                    for target, score in results['predictions'].items():
                        print(f"      {target}: {score:.2f}")
                except Exception as e:
                    print(f"    ⚠ Frame {frame_idx} failed: {e}")
                finally:
                    # Clean up temp file
                    if temp_path.exists():
                        temp_path.unlink()

            frame_idx += 1

        cap.release()

        print(f"\n✓ Processed {processed_count} frames from video")

        # Save video results
        if save_results and results_list:
            self._save_video_results(results_list, video_path, output_dir)

        return results_list

    def _save_results(self, results, output_dir):
        """Save results ke files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_name = Path(results['image_path']).stem

        # Save masked image
        masked_path = output_dir / f"{image_name}_masked.jpg"
        cv2.imwrite(str(masked_path), results['privacy']['masked_image'])

        # Save segmentation mask (colored)
        mask_path = output_dir / f"{image_name}_segmentation.png"
        mask_colored = self._colorize_mask(results['segmentation']['mask'])
        cv2.imwrite(str(masked_path), mask_colored)

        # Save metrics JSON
        metrics_path = output_dir / f"{image_name}_metrics.json"
        metrics_data = {
            'image': results['image_path'],
            'segmentation_metrics': results['segmentation']['metrics'],
            'predictions': results['predictions'],
            'privacy_detections': len(results['privacy']['detections'])
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        # Save SHAP values if available
        if results['shap_values'] is not None:
            shap_path = output_dir / f"{image_name}_shap.json"
            with open(shap_path, 'w') as f:
                json.dump(results['shap_values'], f, indent=2)

        print(f"  ✓ Results saved to {output_dir}/")

    def _save_video_results(self, results_list, video_path, output_dir):
        """Save video results ke files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_name = Path(video_path).stem

        # Save summary JSON
        summary_path = output_dir / f"{video_name}_summary.json"
        summary = {
            'video': str(video_path),
            'frames_processed': len(results_list),
            'timeline': []
        }

        for results in results_list:
            timeline_entry = {
                'frame': results['frame_idx'],
                'timestamp': results['timestamp'],
                'predictions': results['predictions'],
                'metrics': results['segmentation']['metrics']
            }
            summary['timeline'].append(timeline_entry)

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save average predictions
        avg_predictions = {}
        for target in ['beauty', 'safety', 'comfort', 'uvi']:
            scores = [r['predictions'][target] for r in results_list if target in r['predictions']]
            if scores:
                avg_predictions[target] = np.mean(scores)

        print(f"\n✓ Video summary saved to {summary_path}")
        print(f"\nAverage predictions:")
        for target, score in avg_predictions.items():
            print(f"  {target}: {score:.2f}")

    def _colorize_mask(self, mask):
        """Convert segmentation mask ke colored image untuk visualisasi."""
        # Cityscapes color map (simplified)
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

    def visualize_results(self, results, show=True, save_path=None):
        """Visualize results dengan matplotlib."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Original image
        img = cv2.imread(results['image_path'])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Masked image
        masked_rgb = cv2.cvtColor(results['privacy']['masked_image'], cv2.COLOR_BGR2RGB)
        axes[0, 1].imshow(masked_rgb)
        axes[0, 1].set_title(f"Privacy Masked ({len(results['privacy']['detections'])} detections)")
        axes[0, 1].axis('off')

        # Segmentation mask
        mask_colored = self._colorize_mask(results['segmentation']['mask'])
        mask_rgb = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
        axes[1, 0].imshow(mask_rgb)
        axes[1, 0].set_title('Segmentation Mask')
        axes[1, 0].axis('off')

        # Predictions
        ax = axes[1, 1]
        targets = list(results['predictions'].keys())
        scores = list(results['predictions'].values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

        bars = ax.barh(targets, scores, color=colors)
        ax.set_xlim(0, 10)
        ax.set_xlabel('Score (0-10)')
        ax.set_title('Perception Predictions')

        # Add score labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{score:.2f}', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test UVIP AI inference")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to image or video file")
    parser.add_argument("--models", type=str, default="models/perception",
                       help="Path to trained models directory")
    parser.add_argument("--output", type=str, default="outputs",
                       help="Output directory for results")
    parser.add_argument("--video-frames", type=int, default=30,
                       help="Process every N frames for video (default: 30)")
    parser.add_argument("--visualize", action="store_true",
                       help="Show visualization with matplotlib")
    parser.add_argument("--low-vram", action="store_true", default=True,
                       help="Enable low VRAM mode")

    args = parser.parse_args()

    # Initialize tester
    tester = UVIPTester(models_dir=args.models, low_vram_mode=args.low_vram)

    # Determine if input is image or video
    input_path = Path(args.input)
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

    if input_path.suffix.lower() in video_extensions:
        # Process video
        results = tester.process_video(
            input_path,
            save_results=True,
            output_dir=args.output,
            process_every_n_frames=args.video_frames
        )
    else:
        # Process image
        results = tester.process_image(
            input_path,
            save_results=True,
            output_dir=args.output
        )

        # Visualize if requested
        if args.visualize:
            viz_path = Path(args.output) / f"{input_path.stem}_visualization.png"
            tester.visualize_results(results, show=True, save_path=viz_path)

    print("\n✓ Testing complete!")


if __name__ == "__main__":
    main()
