"""Pembacaan metadata EXIF: lokasi pengambilan foto dan waktu pengambilan.

Latar belakang
==============
Seluruh pipeline sebelumnya tidak pernah membaca EXIF. Akibatnya foto yang
diunggah dicap dengan posisi GPS perangkat SAAT UPLOAD, bukan posisi saat
foto diambil. Untuk foto arsip/survei (diambil pagi, diunggah sore dari
kantor), titik peta jadi salah tempat.

Modul ini menyediakan satu sumber kebenaran untuk data EXIF:

    read_capture_location(path) -> dict | None

Prioritas nilai yang dikembalikan (``source``):

    "exif_gps"   GPS tertanam di file foto       -> paling akurat
    "sidecar"    file .json di sebelah foto      -> hasil ekspor tool survei
    None         tidak ada data                  -> pemanggil memakai fallback

Modul ini TIDAK menebak. Bila EXIF tidak ada, ia mengembalikan ``None`` dan
pemanggil yang memutuskan fallback (mis. koordinat kiriman klien) supaya
asal-usul setiap titik tetap bisa diaudit.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from PIL.ExifTags import Base as ExifBase
from PIL.ExifTags import GPS as ExifGps

__all__ = ["read_capture_location", "read_capture_time", "read_exif_metadata"]

GPS_IFD = 0x8825

# Tag waktu pengambilan, berurutan sesuai prioritas.
_TIME_TAGS = (
    "DateTimeOriginal",
    "DateTimeDigitized",
    "DateTime",
)


def _to_float(value: Any) -> Optional[float]:
    """Konversi rasional EXIF (IFDRational / tuple / angka) ke float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:  # tuple rasional (num, den)
        num, den = value
        return float(num) / float(den) if float(den) else None
    except Exception:  # noqa: BLE001 - EXIF bisa berisi tipe aneh
        return None


def _dms_to_decimal(dms: Any, ref: Optional[str]) -> Optional[float]:
    """(derajat, menit, detik) + 'S'/'W' -> derajat desimal."""
    try:
        degrees, minutes, seconds = (float(part) for part in dms)
    except (TypeError, ValueError):
        return None
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref and str(ref).strip().upper() in {"S", "W"}:
        decimal = -decimal
    return decimal


def _exif_datetime(raw: Any) -> Optional[str]:
    """'2024:05:17 08:31:09' -> '2024-05-17T08:31:09' (ISO 8601)."""
    if not raw:
        return None
    text = str(raw).strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M"):
        try:
            return datetime.strptime(text, fmt).isoformat()
        except ValueError:
            continue
    return None


def read_exif_metadata(path: Path | str) -> dict:
    """Baca EXIF mentah yang relevan. Tidak pernah melempar exception.

    Returns
    -------
    dict
        ``{"gps": {...} | None, "captured_at": str | None, "make": str | None,
        "model": str | None, "orientation": int | None}``
    """
    result: dict[str, Any] = {
        "gps": None,
        "captured_at": None,
        "make": None,
        "model": None,
        "orientation": None,
    }
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            if not exif:
                return result
            for tag_name in _TIME_TAGS:
                tag = getattr(ExifBase, tag_name, None)
                if tag is None:
                    continue
                captured = _exif_datetime(exif.get(tag))
                if captured:
                    result["captured_at"] = captured
                    break
            for key, name in (("Make", "make"), ("Model", "model"),
                              ("Orientation", "orientation")):
                tag = getattr(ExifBase, key, None)
                if tag is not None and exif.get(tag) is not None:
                    value = exif.get(tag)
                    result[name] = int(value) if name == "orientation" else str(value).strip()
            gps_ifd = exif.get_ifd(GPS_IFD)
            if gps_ifd:
                lat = _dms_to_decimal(
                    gps_ifd.get(ExifGps.GPSLatitude), gps_ifd.get(ExifGps.GPSLatitudeRef)
                )
                lon = _dms_to_decimal(
                    gps_ifd.get(ExifGps.GPSLongitude), gps_ifd.get(ExifGps.GPSLongitudeRef)
                )
                if lat is not None and lon is not None and (lat or lon):
                    altitude = _to_float(gps_ifd.get(ExifGps.GPSAltitude))
                    if altitude is not None and gps_ifd.get(ExifGps.GPSAltitudeRef) == 1:
                        altitude = -altitude
                    result["gps"] = {
                        "latitude": round(lat, 7),
                        "longitude": round(lon, 7),
                        "altitude_m": round(altitude, 2) if altitude is not None else None,
                        "direction_deg": _to_float(gps_ifd.get(ExifGps.GPSImgDirection)),
                        "map_datum": gps_ifd.get(ExifGps.GPSMapDatum),
                    }
    except Exception:  # noqa: BLE001 - file rusak / bukan gambar
        return result
    return result


def read_capture_location(path: Path | str) -> Optional[dict]:
    """Lokasi pengambilan foto dalam bentuk ``capture_location``.

    Urutan sumber: GPS EXIF, lalu sidecar ``<nama>.json`` dengan field
    ``latitude``/``longitude``. ``None`` bila tidak ada satu pun.

    Format hasil::

        {"latitude": -7.9786, "longitude": 112.6306, "source": "exif_gps",
         "captured_at": "2024-05-17T08:31:09", "altitude_m": 505.2,
         "direction_deg": 187.5}
    """
    path = Path(path)
    meta = read_exif_metadata(path)
    gps = meta.get("gps")
    if gps:
        return {
            "latitude": gps["latitude"],
            "longitude": gps["longitude"],
            "source": "exif_gps",
            "captured_at": meta.get("captured_at"),
            "altitude_m": gps.get("altitude_m"),
            "direction_deg": gps.get("direction_deg"),
        }

    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            lat = _to_float(data.get("latitude") or data.get("lat"))
            lon = _to_float(data.get("longitude") or data.get("lon") or data.get("lng"))
            if lat is not None and lon is not None and (lat or lon):
                return {
                    "latitude": round(lat, 7),
                    "longitude": round(lon, 7),
                    "source": "sidecar",
                    "captured_at": data.get("captured_at") or meta.get("captured_at"),
                    "altitude_m": _to_float(data.get("altitude_m")),
                    "direction_deg": _to_float(data.get("direction_deg")),
                }
        except (json.JSONDecodeError, OSError):
            pass
    return None


def read_capture_time(path: Path | str) -> Optional[str]:
    """Waktu pengambilan dari EXIF (ISO 8601), ``None`` bila tidak ada."""
    return read_exif_metadata(path).get("captured_at")
