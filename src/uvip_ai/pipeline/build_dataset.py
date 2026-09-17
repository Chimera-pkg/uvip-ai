"""
Build training dataset (Step 5) + label template.

Workflow:
1. Load foto dari data/extracted/photos/<AREA>/
2. Run privacy guard → segmentation → feature extraction
3. Merge dengan label kuesioner (CSV external) → output training.csv

Template CSV kolom:
  - filename, area, point_id, lat, long (metadata)
  - seg_green_coverage_pct ... seg_sky_visibility_pct (segmentation results)
  - emb_0 .. emb_1023 (embedding)
  - label_beauty, label_safety, label_comfort, label_uvi (kuesioner label — MISSING!)

Run ini hanya yang punya embedding & metrik segmentasi; baris tanpa label diskip.
User perlu menyiapkan file labels/kuesioner.csv terpisah sebelum run full training.
"""
from __future__ import annotations

import csv
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from uvip_ai.config import settings
from uvip_ai.privacy.guard import PrivacyGuard
from uvip_ai.segmentation.segformer import SegformerB5
from uvip_ai.features.dinov2 import Dinov2Extractor


class DatasetBuilder:
    """Orkestrasi pipeline untuk build dataset training."""

    def __init__(self, low_vram_mode: bool = True):
        self.low_vram_mode = low_vram_mode
        self.guard = PrivacyGuard(low_vram_mode=low_vram_mode)
        self.seg = SegformerB5(low_vram_mode=low_vram_mode)
        self.feature = Dinov2Extractor(low_vram_mode=low_vram_mode)

    def _process_photo(self, photo_path: Path) -> dict | None:
        """Process satu foto → return features dict atau None jika error."""
        try:
            # Step A: Privacy masking (optional)
            masked_img, _ = self.guard.process(photo_path)

            # Step B: Segmentation
            seg_result = self.seg.infer(masked_img if hasattr(masked_img, 'convert') else photo_path)

            # Step C: Feature extraction
            emb = self.feature.extract(masked_img if hasattr(masked_img, 'convert') else photo_path)

            metrics = {f"seg_{k}": v for k, v in seg_result["metrics"].items()}
            feat = {"filename": str(photo_path), "embedding": emb, **metrics}
            return feat
        except Exception as e:  # noqa: BLE001
            print(f"[Dataset] Error processing {photo_path}: {e}")
            return None
        finally:
            if self.low_vram_mode:
                self.seg.free_memory()
                self.feature.free_memory()

    def build(self, photos_dir: Path, out_csv: Path, label_csv: Path | None = None) -> pd.DataFrame:
        """Build semua fitur + merge label → return DataFrame."""
        rows = []
        for p in tqdm(photos_dir.rglob("*"), desc="Build dataset"):
            if not p.is_file():
                continue
            feat = self._process_photo(p)
            if feat is None:
                continue
            # Parse metadata from path/filename
            parts = p.relative_to(photos_dir).parts
            area = parts[0]
            fname = p.name
            point_id = fname.split(".")[0]
            row = {
                "filename": p.name,
                "area": area,
                "point_id": point_id,
                "emb_" + str(i): v for i, v in enumerate(feat["embedding"])
            }
            row.update({k: v for k, v in feat.items() if k != "embedding"})
            rows.append(row)

        df = pd.DataFrame(rows)

        # Merge label jika ada
        if label_csv and label_csv.exists():
            labels = pd.read_csv(label_csv)
            df = pd.merge(df, labels, on="filename", how="inner")
            print(f"[Dataset] Merged {len(labels)} labels, kept {len(df)} rows.")
        else:
            print("[Dataset] No labels provided; add label columns manually later.")

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[Dataset] Saved {len(df)} rows to {out_csv}")
        return df

    def free_all(self) -> None:
        self.guard.free_memory()
        self.seg.free_memory()
        self.feature.free_memory()


def main():
    parser = argparse.ArgumentParser(description="Build dataset UVIP AI")
    parser.add_argument("--photos-dir", type=str, default="data/extracted/photos/KAYUTANGAN")
    parser.add_argument("--out-csv", type=str, default="data/datasets/features_partial.csv")
    parser.add_argument("--labels", type=str, default=None, help="CSV label kuesioner")
    args = parser.parse_args()

    builder = DatasetBuilder(low_vram_mode=settings.uvip_low_vram_mode)
    builder.build(Path(args.photos_dir), Path(args.out_csv), Path(args.labels) if args.labels else None)
    builder.free_all()


if __name__ == "__main__":
    main()
