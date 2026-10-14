"""PVI scorer - 8 indeks penelitian -> 4 skor `perception_prediction`.

KONTEKS
=======
Modul ini menggantikan rumus hardcoded di ``uvip_ai/api/main.py``
(``build_unified_metrics``) yang menghasilkan skor selalu 10 karena
``min(10, ...)`` jenuh.

Struktur JSON output TIDAK berubah. Yang berubah hanya ISI 4 angka:
``beauty_score``, ``safety_score``, ``comfort_score``, ``uvi_score``.

SUMBER DATA
===========
Input adalah ``metrics`` dari ``SegformerB5.infer(excel_mode=True)`` -
8 indeks yang SUDAH dihitung runtime saat ini tetapi dibuang:

    building_visibility_index    BVI
    vegetation_coverage_index    VCI
    sky_openness_index           SOI
    ground_accessibility_index   GAI
    human_activity_index         HAI   (selalu 0 - tidak dipakai)
    vehicle_intensity_index      VDI
    traffic_infrastructure_index TII
    heritage_dominance_index     HDI   (selalu 0 - tidak dipakai)

RUMUS
=====
PVI = 0.06*BVI + 0.10*VCI + 0.56*SOI + 0.10*GAI + 0.18*VDI

Pemetaan ke 4 field (normalisasi min-max terhadap ``CALIBRATION``):

    beauty_score  <- PVI
    safety_score  <- 0.6*GAI + 0.4*TII
    comfort_score <- 0.5*VCI + 0.5*SOI
    uvi_score     <- 0.4*beauty + 0.3*safety + 0.3*comfort   (rumus lama)

BUG YANG DITANGANI DI SISI KONSUMEN (tanpa mengubah segformer.py)
================================================================
``vehicle_intensity_index`` tidak di-clamp di sumbernya, sehingga bisa
melebihi 1.0 (terukur: 1.4273 pada ALUN_ALUN_MERDEKA_BS-03). Modul ini
meng-clamp VDI ke [0, 1] sebelum dipakai.

KETERBATASAN (jujur)
====================
Skor ini BUKAN persepsi manusia. Masih murni turunan piksel - hanya
formulasinya masuk akal dan tidak jenuh. Validasi terhadap penilaian
panel manusia belum ada.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

__all__ = [
    "CALIBRATION",
    "PVI_WEIGHTS",
    "SCORE_MIN",
    "SCORE_MAX",
    "compute_pvi",
    "compute_scores",
    "compute_scores_from_metrics",
    "indices_from_pct_by_class",
]


# --------------------------------------------------------------------------
# Bobot PVI (sesuai dokumen penelitian)
# --------------------------------------------------------------------------
PVI_WEIGHTS: Dict[str, float] = {
    "building_visibility_index": 0.06,
    "vegetation_coverage_index": 0.10,
    "sky_openness_index": 0.56,
    "ground_accessibility_index": 0.10,
    "vehicle_intensity_index": 0.18,
}

# --------------------------------------------------------------------------
# Anchor kalibrasi: min/max terukur dari 60 foto data nyata.
# Dipakai untuk normalisasi min-max -> skor 1..10.
# Hitung ulang dengan: python scripts/calibrate_pvi.py
# --------------------------------------------------------------------------
CALIBRATION: Dict[str, Dict[str, float]] = {
    "pvi": {"min": 0.08729200000000001, "max": 0.406776},
    "safety_raw": {"min": 0.05046, "max": 0.25554},
    "comfort_raw": {"min": 0.06775, "max": 0.3413},
}

SCORE_MIN = 1.0
SCORE_MAX = 10.0


def _get(metrics: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Ambil nilai numerik dari metrics, aman terhadap None/str."""
    value = metrics.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _normalize(value: float, anchor: Mapping[str, float]) -> float:
    """Normalisasi min-max ke [SCORE_MIN, SCORE_MAX], lalu clamp.

    Anchor ``max == min`` (degenerate) -> kembalikan titik tengah.
    """
    span = anchor["max"] - anchor["min"]
    if span <= 0:
        return (SCORE_MIN + SCORE_MAX) / 2.0
    ratio = (value - anchor["min"]) / span
    score = SCORE_MIN + (SCORE_MAX - SCORE_MIN) * ratio
    return _clamp(score, SCORE_MIN, SCORE_MAX)


def compute_pvi(metrics: Mapping[str, Any]) -> float:
    """Perceptual Vegetation Index (PVI) - indeks komposit 5 dimensi.

    ``vehicle_intensity_index`` di-clamp ke [0, 1] lebih dulu (fix bug).
    """
    vdi = _clamp(_get(metrics, "vehicle_intensity_index"))
    return (
        PVI_WEIGHTS["building_visibility_index"]
        * _clamp(_get(metrics, "building_visibility_index"))
        + PVI_WEIGHTS["vegetation_coverage_index"]
        * _clamp(_get(metrics, "vegetation_coverage_index"))
        + PVI_WEIGHTS["sky_openness_index"]
        * _clamp(_get(metrics, "sky_openness_index"))
        + PVI_WEIGHTS["ground_accessibility_index"]
        * _clamp(_get(metrics, "ground_accessibility_index"))
        + PVI_WEIGHTS["vehicle_intensity_index"] * vdi
    )


def compute_scores(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Ubah 8 indeks penelitian menjadi 4 skor siap-kontrak API.

    Parameters
    ----------
    metrics:
        Dict hasil ``seg.infer(excel_mode=True)["metrics"]``.

    Returns
    -------
    dict
        ``beauty_score`` / ``safety_score`` / ``comfort_score`` /
        ``uvi_score`` (float, 2 desimal) plus ``_raw`` untuk audit.

    Catatan: nilai mentah dihitung penuh (belum dibulatkan) sebelum
    dipakai menghitung ``uvi_score``, lalu semuanya dibulatkan.
    """
    if not metrics:
        metrics = {}

    pvi = compute_pvi(metrics)
    safety_raw = 0.6 * _clamp(
        _get(metrics, "ground_accessibility_index")
    ) + 0.4 * _clamp(_get(metrics, "traffic_infrastructure_index"))
    comfort_raw = 0.5 * _clamp(
        _get(metrics, "vegetation_coverage_index")
    ) + 0.5 * _clamp(_get(metrics, "sky_openness_index"))

    beauty = _normalize(pvi, CALIBRATION["pvi"])
    safety = _normalize(safety_raw, CALIBRATION["safety_raw"])
    comfort = _normalize(comfort_raw, CALIBRATION["comfort_raw"])

    # Rumus UVI dipertahankan dari implementasi lama.
    uvi = (beauty / 10.0) * 0.4 + (safety / 10.0) * 0.3 + (comfort / 10.0) * 0.3

    return {
        "beauty_score": round(beauty, 2),
        "safety_score": round(safety, 2),
        "comfort_score": round(comfort, 2),
        "uvi_score": round(_clamp(uvi, 0.0, 1.0), 2),
        "_raw": {
            "pvi": round(pvi, 6),
            "safety_raw": round(safety_raw, 6),
            "comfort_raw": round(comfort_raw, 6),
        },
    }


def compute_scores_from_metrics(
    metrics: Mapping[str, Any],
) -> Dict[str, float]:
    """Varian ringkas: hanya 4 field kontrak, tanpa ``_raw``.

    Cocok untuk langsung di-``**`` ke payload ``perception_prediction``.
    """
    scores = compute_scores(metrics)
    scores.pop("_raw", None)
    return scores


def indices_from_pct_by_class(
    pct_by_class: Mapping[str, Any],
) -> Dict[str, float]:
    """Hitung 8 indeks dari ``pct_by_class`` (fallback).

    Dipakai saat ``metrics`` dari ``infer(excel_mode=True)`` tidak tersedia -
    terutama jalur VIDEO, yang memanggil ``infer()`` tanpa ``excel_mode``
    sehingga ``metrics`` tidak pernah ada.

    Meniru ``SegformerB5._compute_excel_metrics`` semantik-untuk-semantik,
    termasuk re-normalisasi yang membuang objek bergerak (pedestrian,
    kendaraan) dari komposisi statis. Re-normalisasi inilah yang membuat
    indeks berbeda dari sekadar ``pct/100`` mentah.

    Tidak mengubah ``segformer.py`` - murni di sisi konsumen.
    """
    get = lambda key: _get(pct_by_class, key)  # noqa: E731

    static_categories = [
        "building",
        "vegetation",
        "sky",
        "road",
        "sidewalk",
        "signage",
        "street_furniture",
        "heritage",
    ]
    static_sum = sum(get(cat) for cat in static_categories)
    scale_factor = 100.0 / static_sum if static_sum > 0 else 1.0
    # Persentase statis ternormalisasi dalam rentang [0, 1]
    norm = {cat: get(cat) * scale_factor / 100.0 for cat in static_categories}

    building_pct = get("building")
    vegetation_pct = get("vegetation")
    sky_pct = get("sky")
    road_pct = get("road")
    sidewalk_pct = get("sidewalk")
    signage_pct = get("signage")
    street_furniture_pct = get("street_furniture")
    heritage_pct = get("heritage")
    vehicle_pct = get("vehicle")
    pedestrian_pct = get("pedestrian")

    # VDI: mengikuti perilaku sumber apa adanya (BELUM di-clamp di sini).
    # Clamp [0, 1] dilakukan di compute_pvi().
    vehicle_intensity = vehicle_pct * 0.51

    total_building = building_pct + heritage_pct
    heritage_dominance = (
        heritage_pct / total_building if total_building > 0 else 0.0
    )

    # HAI: sumber membaca dari dict yang HANYA berisi kelas statis, sehingga
    # "pedestrian" tidak pernah ada di dalamnya dan nilainya SELALU 0.
    # Disamakan di sini agar kedua jalur konsisten. Tidak berdampak pada
    # skor karena HAI tidak dipakai di PVI.
    human_activity = 0.0

    return {
        "building_visibility_index": min(building_pct / 100.0, 1.0),
        "vegetation_coverage_index": min(vegetation_pct / 100.0, 1.0),
        "sky_openness_index": min(sky_pct / 100.0, 1.0),
        "ground_accessibility_index": min(
            (road_pct + sidewalk_pct) / 100.0, 1.0
        ),
        "human_activity_index": human_activity,
        "vehicle_intensity_index": vehicle_intensity,
        "traffic_infrastructure_index": min(
            (signage_pct + street_furniture_pct) / 100.0, 1.0
        ),
        "heritage_dominance_index": heritage_dominance,
        "_static_normalization_factor": scale_factor,
        "_normalized_static": norm,
    }
