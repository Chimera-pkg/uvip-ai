#!/usr/bin/env python
"""
Data augmentation untuk UVIP-AI.
Augment 431 foto -> 2155 foto (5x lipat).

Augmentations:
- Original
- Horizontal flip
- Rotate 90°
- Brightness +20%
- Contrast +20%

Usage:
    python scripts/augment_data.py
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv


def augment_image(img):
    """Generate 4 augmented versions dari 1 image."""
    augmented = []

    # 1. Horizontal flip
    flipped = cv2.flip(img, 1)
    augmented.append(('flip', flipped))

    # 2. Rotate 90°
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), 90, 1)
    rotated = cv2.warpAffine(img, M, (w, h))
    augmented.append(('rot90', rotated))

    # 3. Brightness +20%
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255)
    bright = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    augmented.append(('bright', bright))

    # 4. Contrast +20%
    contrast = cv2.convertScaleAbs(img, alpha=1.2, beta=0)
    augmented.append(('contrast', contrast))

    return augmented


def main():
    # Paths
    src_dir = Path('data/extracted/photos')
    dst_dir = Path('data/extracted/photos_augmented')
    manifest_path = Path('data/extracted/manifest.csv')

    if not src_dir.exists():
        print(f"Source directory tidak ditemukan: {src_dir}")
        return

    # Create output directory
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = list(src_dir.rglob('*.jpg')) + list(src_dir.rglob('*.png'))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

    print(f"Found {len(image_files)} original images")
    print(f"Augmenting to {len(image_files) * 5} total images...")

    # Load manifest for reference
    manifest_rows = []
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            manifest_rows = list(reader)

    # Create manifest for augmented images
    augmented_manifest = []

    # Augment each image
    for img_path in tqdm(image_files, desc="Augmenting"):
        # Read original
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Get metadata from original manifest
        original_meta = next((r for r in manifest_rows if r['filename'] == img_path.name), None)
        area = original_meta['area'] if original_meta else img_path.parent.name
        point_id = original_meta['point_id'] if original_meta else img_path.stem

        # Save original (copy to augmented folder)
        orig_dst = dst_dir / area / img_path.name
        orig_dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(orig_dst), img)
        augmented_manifest.append({
            'filename': img_path.name,
            'area': area,
            'point_id': point_id,
            'augmentation': 'original'
        })

        # Save augmented versions
        for aug_type, aug_img in augment_image(img):
            aug_name = f"{img_path.stem}_{aug_type}{img_path.suffix}"
            aug_dst = dst_dir / area / aug_name
            cv2.imwrite(str(aug_dst), aug_img)
            augmented_manifest.append({
                'filename': aug_name,
                'area': area,
                'point_id': point_id,
                'augmentation': aug_type
            })

    # Save augmented manifest
    aug_manifest_path = dst_dir / 'manifest_augmented.csv'
    with open(aug_manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'area', 'point_id', 'augmentation'])
        writer.writeheader()
        writer.writerows(augmented_manifest)

    print(f"\n✓ Augmented {len(image_files)} -> {len(augmented_manifest)} images")
    print(f"✓ Saved to: {dst_dir}")
    print(f"✓ Manifest: {aug_manifest_path}")
    print(f"\n📊 Augmentation breakdown:")
    print(f"   Original:    {len(image_files)}")
    print(f"   Flipped:     {len(image_files)}")
    print(f"   Rotated:     {len(image_files)}")
    print(f"   Brightness:  {len(image_files)}")
    print(f"   Contrast:    {len(image_files)}")
    print(f"   Total:       {len(augmented_manifest)}")


if __name__ == "__main__":
    main()
