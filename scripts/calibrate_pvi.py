#!/usr/bin/env python
"""Hitung ulang anchor kalibrasi PVI dari dataset foto nyata.

Anchor di ``src/uvip_ai/scoring/pvi_scorer.py`` (CALIBRATION) adalah
min/max terukur. Script ini menghitungnya ulang dari hasil segmentasi
sehingga anchor bisa diperbarui saat dataset bertambah.

Input: CSV atau JSON berisi satu baris per foto dengan kolom 8 indeks
(dari ``SegformerB5.infer(excel_mode=True)['metrics']``).

Contoh:
    python scripts/calibrate_pvi.py data/training/pvi_metrics.csv
    python scripts/calibrate_pvi.py _cmp.json --json
    python scripts/calibrate_pvi.py data/training/pvi_metrics.csv --write

Tanpa --write, script hanya MENAMPILKAN blok CALIBRATION untuk disalin.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from uvip_ai.scoring.pvi_scorer import PVI_WEIGHTS, compute_pvi  # noqa: E402

REQUIRED = [
    "building_visibility_index",
    "vegetation_coverage_index",
    "sky_openness_index",
    "ground_accessibility_index",
    "vehicle_intensity_index",
    "traffic_infrastructure_index",
]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def load_rows(path: Path, is_json: bool) -> list[dict]:
    if is_json:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("rows") or data.get("data") or []
        return [dict(r) for r in data]

    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", help="CSV atau JSON berisi kolom indeks")
    ap.add_argument("--json", action="store_true", help="input berformat JSON")
    ap.add_argument("--write", action="store_true",
                    help="tulis langsung ke pvi_scorer.py (default: tampilkan saja)")
    args = ap.parse_args()

    path = Path(args.dataset)
    if not path.is_file():
        print(f"[X] Tidak ditemukan: {path}", file=sys.stderr)
        return 1

    rows = load_rows(path, args.json)
    if not rows:
        print("[X] Dataset kosong", file=sys.stderr)
        return 1

    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        print(f"[X] Kolom tidak ada di dataset: {missing}", file=sys.stderr)
        print(f"    Kolom tersedia: {sorted(rows[0])}", file=sys.stderr)
        return 1

    def num(row: dict, key: str) -> float:
        try:
            return float(row.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    pvi, safety, comfort = [], [], []
    for row in rows:
        pvi.append(compute_pvi(row))
        safety.append(
            0.6 * _clamp(num(row, "ground_accessibility_index"))
            + 0.4 * _clamp(num(row, "traffic_infrastructure_index"))
        )
        comfort.append(
            0.5 * _clamp(num(row, "vegetation_coverage_index"))
            + 0.5 * _clamp(num(row, "sky_openness_index"))
        )

    calibration = {
        "pvi": {"min": min(pvi), "max": max(pvi)},
        "safety_raw": {"min": min(safety), "max": max(safety)},
        "comfort_raw": {"min": min(comfort), "max": max(comfort)},
    }

    print(f"[i] Foto dianalisis : {len(rows)}")
    print(f"[i] Sumber          : {path}")
    print(f"[i] Bobot PVI       : {PVI_WEIGHTS}")
    print()
    print("CALIBRATION = {")
    for key, anchor in calibration.items():
        print(f'    "{key}": {{"min": {anchor["min"]!r}, '
              f'"max": {anchor["max"]!r}}},')
    print("}")

    if not args.write:
        print()
        print("[i] Mode tampil saja. Tambahkan --write untuk menulis ke "
              "src/uvip_ai/scoring/pvi_scorer.py")
        return 0

    target = REPO_ROOT / "src" / "uvip_ai" / "scoring" / "pvi_scorer.py"
    text = target.read_text(encoding="utf-8")
    start = text.index("CALIBRATION: Dict[str, Dict[str, float]] = {")
    end = text.index("}\n", start) + 2
    block = "CALIBRATION: Dict[str, Dict[str, float]] = {\n"
    for key, anchor in calibration.items():
        block += f'    "{key}": {{"min": {anchor["min"]!r}, "max": {anchor["max"]!r}}},\n'
    block += "}\n"
    target.write_text(text[:start] + block + text[end:], encoding="utf-8")
    print()
    print(f"[OK] CALIBRATION diperbarui di {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
