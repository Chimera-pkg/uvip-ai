#!/usr/bin/env python3
"""
Extract features from all photos for training.

Pipeline per foto (dijalankan oleh ``DatasetBuilder`` — satu implementasi
untuk skrip ini dan training penuh):
1. Privacy Guard (YOLOv8n) - blur wajah/plat nomor
2. SegFormer-B5 (excel_mode) - 8 indeks PVI penelitian
3. DINOv2-Large - extract 1024-d embedding

Output: CSV dengan kolom ``seg_<nama_indeks>`` (8 kolom) + ``emb_0..emb_1023``.
Embedding butuh dua model besar sekaligus (~2 GB VRAM); jalankan di GPU.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uvip_ai.pipeline.build_dataset import DatasetBuilder

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def extract_features(photos_dir: Path, output_csv: Path) -> None:
    """Ekstrak fitur semua foto di ``photos_dir`` ke ``output_csv``."""
    images = [p for p in photos_dir.rglob("*")
              if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    print(f"[Extract] Processing photos from {photos_dir}")
    print(f"[Extract] Found {len(images)} images")
    if not images:
        print("[ERROR] No images found")
        return

    print("[Extract] Loading models...")
    builder = DatasetBuilder(low_vram_mode=True)
    try:
        builder.build(photos_dir, output_csv, None)
    finally:
        builder.free_all()

    print(f"\n[Extract] Selesai: {output_csv}")


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

    args = parser.parse_args()

    photos_dir = Path(args.input)
    output_csv = Path(args.output)

    if not photos_dir.exists():
        print(f"[ERROR] Input directory not found: {photos_dir}")
        sys.exit(1)

    extract_features(photos_dir, output_csv)


if __name__ == "__main__":
    main()
