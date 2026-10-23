#!/usr/bin/env python3
"""Buat foto uji dengan EXIF GPS + waktu pengambilan.

Foto dataset UVIP tidak menyimpan GPS (diekspor tanpa metadata), jadi jalur
EXIF di ``/ai/process`` dan ``/ai/advise`` tidak bisa diuji dengan data asli.
Skrip ini menempelkan GPS IFD + DateTimeOriginal ke sembarang JPEG.

Contoh:
    python scripts/make_test_photo.py \
        --source data/extracted/photos/ALUN_ALUN_MERDEKA/ALUN_ALUN_MERDEKA_BU-03.jpg \
        --out data/testing/test_gps_exif.jpg \
        --lat -7.97685 --lon 112.6306111 --time "2026-10-22 08:31:09"

Hasilnya bisa langsung dipakai:
    curl -X POST http://127.0.0.1:8000/ai/process \
        -F "file=@data/testing/test_gps_exif.jpg" \
        -F "latitude=-6.2" -F "longitude=106.8166"
Koordinat klien (Jakarta) akan DIABAIKAN karena EXIF menang, dan
``capture_location.source`` bernilai ``exif_gps``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


def _dms(value: float) -> tuple[float, float, float]:
    """Derajat desimal -> (derajat, menit, detik) positif."""
    value = abs(value)
    degrees = int(value)
    minutes_f = (value - degrees) * 60
    minutes = int(minutes_f)
    seconds = round((minutes_f - minutes) * 60, 4)
    return float(degrees), float(minutes), seconds


def write_gps_exif(source: Path, out: Path, *, lat: float, lon: float,
                   captured_at: str) -> Path:
    """Simpan ``source`` ke ``out`` dengan EXIF GPS + waktu pengambilan."""
    img = Image.open(source).convert("RGB")
    exif = Image.Exif()
    exif[0x010F] = "UVIP"                      # Make
    exif[0x0110] = "Test GPS"                  # Model
    exif[0x0132] = captured_at                 # DateTime
    exif[0x8769] = {0x9003: captured_at}       # ExifIFD.DateTimeOriginal
    exif[0x8825] = {                           # GPS IFD
        1: "S" if lat < 0 else "N", 2: _dms(lat),
        3: "W" if lon < 0 else "E", 4: _dms(lon),
        5: 0, 6: 505.2,                        # 0 = di atas permukaan laut
        16: "WGS-84", 17: 187.5,               # datum + arah kamera (derajat)
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "jpeg", quality=92, exif=exif)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="JPEG sumber")
    p.add_argument("--out", default="data/testing/test_gps_exif.jpg", help="JPEG hasil")
    p.add_argument("--lat", type=float, default=-7.97685, help="Lintang (derajat desimal)")
    p.add_argument("--lon", type=float, default=112.6306111, help="Bujur (derajat desimal)")
    p.add_argument("--time", default="2026-10-22 08:31:09",
                   help='Waktu ambil "YYYY-MM-DD HH:MM:SS"')
    args = p.parse_args()

    src = Path(args.source)
    if not src.is_file():
        print(f"[ERROR] Sumber tidak ditemukan: {src}")
        return 1
    out = write_gps_exif(src, Path(args.out), lat=args.lat, lon=args.lon,
                         captured_at=args.time)
    print(f"[OK] {out} (lat={args.lat}, lon={args.lon}, waktu={args.time})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
