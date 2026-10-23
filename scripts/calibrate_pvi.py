"""Kalibrasi anchor PVI dari segmentasi live dataset referensi.

Menjalankan SegFormer (mode ``excel_mode``) pada seluruh foto di
``data/extracted/photos`` lalu mengambil persentil 2 dan 98 dari:

    pvi          -> anchor ``beauty_score``
    safety_raw    -> anchor ``safety_score``   (0.6*GAI + 0.4*TII)
    comfort_raw   -> anchor ``comfort_score``  (0.5*VCI + 0.5*SOI)

Persentil - bukan min/max - supaya satu foto outlier tidak menggeser seluruh
skala. Tanpa langkah ini skor akan menumpuk di ujung atas/bawah kembali.

Pemakaian
---------
    python scripts/calibrate_pvi.py                      # tampilkan saja
    python scripts/calibrate_pvi.py --write              # tulis ke pvi_scorer.py
    python scripts/calibrate_pvi.py --limit 40           # uji cepat

Butuh GPU (atau CPU yang sabar). 431 foto ~2-4 menit di RTX 3060.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uvip_ai.scoring.pvi_scorer import (  # noqa: E402
    SCORE_MAX,
    SCORE_MIN,
    comfort_index,
    compute_pvi,
    compute_scores_from_metrics,
    safety_index,
)
from uvip_ai.segmentation.segformer import SegformerB5  # noqa: E402

SCORER_PATH = ROOT / "src" / "uvip_ai" / "scoring" / "pvi_scorer.py"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def collect_images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise SystemExit(f"Folder foto tidak ditemukan: {root}")
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise SystemExit(f"Tidak ada gambar di {root}")
    return files


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def rewrite_calibration(values: dict[str, dict[str, float]]) -> None:
    text = SCORER_PATH.read_text(encoding="utf-8")
    block = (
        "CALIBRATION: Dict[str, Dict[str, float]] = {\n"
        f'    "pvi": {{"min": {values["pvi"]["min"]:.4f}, '
        f'"max": {values["pvi"]["max"]:.4f}}},\n'
        f'    "safety_raw": {{"min": {values["safety_raw"]["min"]:.4f}, '
        f'"max": {values["safety_raw"]["max"]:.4f}}},\n'
        f'    "comfort_raw": {{"min": {values["comfort_raw"]["min"]:.4f}, '
        f'"max": {values["comfort_raw"]["max"]:.4f}}},\n'
        "}"
    )
    pattern = re.compile(
        r"^CALIBRATION: Dict\[str, Dict\[str, float\]\] = \{.*?^\}", re.DOTALL | re.MULTILINE
    )
    if not pattern.search(text):
        raise SystemExit("Blok CALIBRATION tidak ditemukan di pvi_scorer.py")
    SCORER_PATH.write_text(pattern.sub(block, text, count=1), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", default=str(ROOT / "data" / "extracted" / "photos"))
    parser.add_argument("--limit", type=int, default=0, help="batasi jumlah foto")
    parser.add_argument("--low", type=float, default=2.0, help="persentil bawah")
    parser.add_argument("--high", type=float, default=98.0, help="persentil atas")
    parser.add_argument("--write", action="store_true", help="tulis hasil ke pvi_scorer.py")
    parser.add_argument("--out", default=str(ROOT / "data" / "scoring" / "calibration.json"))
    parser.add_argument("--dump", default=str(ROOT / "data" / "scoring" / "pvi_per_photo.csv"),
                        help="CSV metrik per foto ('-' = jangan tulis)")
    args = parser.parse_args()

    files = collect_images(Path(args.photos))
    if args.limit:
        files = files[: args.limit]
    print(f"Kalibrasi dari {len(files)} foto (persentil {args.low}/{args.high})")

    seg = SegformerB5()
    pvi_values: list[float] = []
    safety_values: list[float] = []
    comfort_values: list[float] = []
    per_photo: list[dict] = []

    for idx, path in enumerate(files, 1):
        image = cv2.imread(str(path))
        if image is None:
            print(f"  lewati (tidak terbaca): {path.name}")
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        metrics = seg.infer(rgb, excel_mode=True)["metrics"]
        pvi = compute_pvi(metrics)
        safety_raw = safety_index(metrics)
        comfort_raw = comfort_index(metrics)
        pvi_values.append(pvi)
        safety_values.append(safety_raw)
        comfort_values.append(comfort_raw)
        per_photo.append({
            "filename": path.name,
            "area": path.parent.name,
            "pvi": round(pvi, 6),
            "safety_raw": round(safety_raw, 6),
            "comfort_raw": round(comfort_raw, 6),
            **{k: round(v, 6) for k, v in metrics.items()
               if isinstance(v, (int, float))},
        })
        if idx % 25 == 0 or idx == len(files):
            print(
                f"  {idx}/{len(files)}  pvi~{statistics.fmean(pvi_values):.4f} "
                f"safety~{statistics.fmean(safety_values):.4f} "
                f"comfort~{statistics.fmean(comfort_values):.4f}"
            )

    if len(pvi_values) < 10:
        raise SystemExit("Butuh minimal 10 foto valid untuk kalibrasi.")

    values = {
        "pvi": {
            "min": percentile(pvi_values, args.low),
            "max": percentile(pvi_values, args.high),
        },
        "safety_raw": {
            "min": percentile(safety_values, args.low),
            "max": percentile(safety_values, args.high),
        },
        "comfort_raw": {
            "min": percentile(comfort_values, args.low),
            "max": percentile(comfort_values, args.high),
        },
    }
    # Sebaran skor kontrak (1..10) dengan anchor hasil kalibrasi ini. Inilah
    # angka yang menjawab "masih mentok 10 atau tidak".
    scores = [compute_scores_from_metrics(row, values) for row in per_photo]

    def dist(key: str, lo: float = SCORE_MIN, hi: float = SCORE_MAX) -> dict:
        vals = [s[key] for s in scores]
        return {
            "min": min(vals),
            "p2": percentile(vals, 2),
            "mean": statistics.fmean(vals),
            "p98": percentile(vals, 98),
            "max": max(vals),
            "std": statistics.pstdev(vals),
            "at_floor_pct": round(100 * sum(1 for v in vals if v <= lo) / len(vals), 2),
            "at_ceiling_pct": round(100 * sum(1 for v in vals if v >= hi) / len(vals), 2),
        }

    report = {
        "n": len(pvi_values),
        "percentiles": {"low": args.low, "high": args.high},
        "calibration": values,
        "observed": {
            "pvi": {
                "min": min(pvi_values),
                "max": max(pvi_values),
                "mean": statistics.fmean(pvi_values),
                "std": statistics.pstdev(pvi_values),
            },
            "safety_raw": {
                "min": min(safety_values),
                "max": max(safety_values),
                "mean": statistics.fmean(safety_values),
                "std": statistics.pstdev(safety_values),
            },
            "comfort_raw": {
                "min": min(comfort_values),
                "max": max(comfort_values),
                "mean": statistics.fmean(comfort_values),
                "std": statistics.pstdev(comfort_values),
            },
        },
        "scores": {
            **{key: dist(key) for key in
               ("beauty_score", "safety_score", "comfort_score")},
            # UVI memakai skala 0..1, bukan 1..10 seperti skor persepsi.
            "uvi_score": dist("uvi_score", lo=0.0, hi=1.0),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nRingkasan:")
    print(json.dumps(report, indent=2))

    if args.dump and args.dump != "-":
        dump_path = Path(args.dump)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_photo).to_csv(dump_path, index=False)
        print(f"\nMetrik per foto: {dump_path} ({len(per_photo)} baris)")

    if args.write:
        rewrite_calibration(values)
        print(f"\nCALIBRATION ditulis ke {SCORER_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
