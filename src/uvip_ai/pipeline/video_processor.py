"""
Video Processing Pipeline untuk UVIP-AI.

Input: Video file (mp4, avi, mov)
Output: Video dengan segmentation overlay (color-coded classes)

Flow:
1. Extract frames dari video
2. Process setiap frame (segmentation)
3. Generate overlay visualization
4. Combine frames jadi video output
5. Save ke uploads/videos/ (permanent, bukan /tmp)
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Cityscapes color palette (19 classes)
CITYSCAPES_COLORS = {
    0: (128, 64, 128),    # road - purple
    1: (244, 35, 232),    # sidewalk - pink
    2: (70, 70, 70),      # building - gray
    3: (102, 102, 156),   # wall - dark blue
    4: (190, 153, 153),   # fence - light pink
    5: (153, 153, 153),   # pole - gray
    6: (250, 170, 30),    # traffic light - orange
    7: (220, 220, 0),     # traffic sign - yellow
    8: (107, 142, 35),    # vegetation - green
    9: (152, 251, 152),   # terrain - light green
    10: (70, 130, 180),   # sky - blue
    11: (220, 20, 60),    # person - red
    12: (255, 0, 0),      # rider - bright red
    13: (0, 0, 142),      # car - dark blue
    14: (0, 0, 70),       # truck - darker blue
    15: (0, 60, 100),     # bus - navy
    16: (0, 80, 100),     # train - teal
    17: (0, 0, 230),      # motorcycle - blue
    18: (119, 11, 32),    # bicycle - brown
}


class VideoProcessor:
    """Process video frames dengan segmentation overlay."""

    def __init__(self, output_dir: str = "uploads/videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_frames_dir = self.output_dir / "temp_frames"
        self.temp_frames_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames(self, video_path: str, fps: float = None) -> list[Path]:
        """Extract frames dari video. Jika fps=None, extract semua frame."""
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_paths = []
        frame_idx = 0
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        target_fps = fps if fps else original_fps

        # Calculate frame skip ratio
        skip_ratio = max(1, int(original_fps / target_fps)) if target_fps > 0 else 1

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % skip_ratio == 0:
                    frame_path = self.temp_frames_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_paths.append(frame_path)

                frame_idx += 1
        finally:
            cap.release()

        logger.info(f"Extracted {len(frame_paths)} frames from {video_path.name}")
        return frame_paths

    def create_overlay(self, frame: np.ndarray, segmentation_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Create color overlay dari segmentation mask."""
        overlay = frame.copy()

        # Apply color untuk setiap class
        for class_id, color in CITYSCAPES_COLORS.items():
            mask = segmentation_mask == class_id
            if np.any(mask):
                overlay[mask] = color

        # Blend dengan original
        result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        return result

    def combine_frames_to_video(
        self,
        frame_paths: list[Path],
        output_path: str,
        fps: float = 30.0,
        codec: str = "mp4v"
    ) -> Path:
        """Combine frames jadi video."""
        if not frame_paths:
            raise ValueError("No frames to combine")

        # Read first frame untuk get dimensions
        first_frame = cv2.imread(str(frame_paths[0]))
        if first_frame is None:
            raise ValueError(f"Cannot read frame: {frame_paths[0]}")

        height, width = first_frame.shape[:2]
        output_path = Path(output_path)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        try:
            for frame_path in frame_paths:
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    out.write(frame)
        finally:
            out.release()

        logger.info(f"Created video: {output_path} ({len(frame_paths)} frames @ {fps} fps)")
        return output_path

    def cleanup_temp_frames(self):
        """Cleanup temporary frames."""
        if self.temp_frames_dir.exists():
            shutil.rmtree(self.temp_frames_dir)
            self.temp_frames_dir.mkdir(parents=True, exist_ok=True)

    def get_video_info(self, video_path: str) -> dict[str, Any]:
        """Get video metadata."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            return {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration_seconds": duration,
            }
        finally:
            cap.release()

    def process_video(
        self,
        video_path: str,
        segmentation_model,
        output_fps: float = None,
        overlay_alpha: float = 0.5
    ) -> dict[str, Any]:
        """
        Process video dengan segmentation overlay.

        Args:
            video_path: Path ke video input
            segmentation_model: Instance SegformerB5
            output_fps: FPS output (default: sama dengan input)
            overlay_alpha: Transparency overlay (0.0-1.0)

        Returns:
            dict dengan video_url, video_info, dan aggregated metrics
        """
        import time
        start_time = time.time()

        video_path = Path(video_path)
        video_info = self.get_video_info(str(video_path))

        # Extract frames
        logger.info(f"Processing video: {video_path.name}")
        frame_paths = self.extract_frames(str(video_path), fps=output_fps)

        if not frame_paths:
            raise ValueError("No frames extracted from video")

        # Process each frame
        processed_frames = []
        all_metrics = []

        for i, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            # Run segmentation
            seg_result = segmentation_model.infer(frame)
            seg_map = seg_result["seg_map"]
            metrics = seg_result["metrics"]
            all_metrics.append(metrics)

            # Create overlay
            overlay_frame = self.create_overlay(frame, seg_map, alpha=overlay_alpha)

            # Save processed frame
            processed_path = self.temp_frames_dir / f"processed_{i:06d}.jpg"
            cv2.imwrite(str(processed_path), overlay_frame)
            processed_frames.append(processed_path)

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(frame_paths)} frames")

        # Combine into video
        output_video_name = f"segmented_{video_path.stem}_{int(time.time())}.mp4"
        output_video_path = self.output_dir / output_video_name

        final_fps = output_fps if output_fps else video_info["fps"]
        self.combine_frames_to_video(processed_frames, str(output_video_path), fps=final_fps)

        # Cleanup temp frames
        self.cleanup_temp_frames()

        # Aggregate metrics (average across all frames)
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m]
                avg_metrics[key] = float(np.mean(values)) if values else 0.0

        elapsed = time.time() - start_time
        logger.info(f"Video processing completed in {elapsed:.2f}s")

        return {
            "video_url": str(output_video_path),
            "video_info": video_info,
            "segmentation_results": avg_metrics,
            "frames_processed": len(processed_frames),
            "processing_time_seconds": elapsed,
        }
