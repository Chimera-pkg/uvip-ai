"""PVI scorer - 8 indeks penelitian -> 4 skor ``perception_prediction``.

KONTEKS
=======
Modul ini menggantikan rumus hardcoded di ``build_unified_metrics``
(``uvip_ai/api/main.py``) yang memakai ``min(10, ...)`` sehingga skor
selalu mentok 10. Struktur JSON output TIDAK berubah - yang berubah hanya
isi 4 angka: ``beauty_score``, ``safety_score``, ``comfort_score``,
``uvi_score``.

SUMBER RUMUS
============
``Photogenic_Value_Index_Multi_Foto.ipynb`` (repo root), bagian
"11. COMPOSITE VISUAL INDICES" dan "11A. THEORETICAL WEIGHTING":

    BVI  Building Visibility       0.20
    VCI  Vegetation Coverage       0.15
    SOI  Sky Openness              0.10
    GAI  Ground Accessibility      0.10
    HAI  Human Activity            0.10
    VII  Vehicle Intensity         0.05
    TII  Traffic Infrastructure    0.05
    HDI  Heritage Dominance        0.25

    PVI = 0.20*BVI + 0.15*VCI + 0.10*SOI + 0.10*GAI
        + 0.10*HAI + 0.05*VII + 0.05*TII + 0.25*HDI

Bobot dijumlahkan 1.00. Kedelapan indeks dihitung di
``SegformerB5._compute_excel_metrics`` (mode ``excel_mode=True``).

PEMETAAN KE 4 FIELD KONTRAK
===========================
    beauty_score  <- PVI
    safety_score  <- 0.6*GAI + 0.4*TII
    comfort_score <- 0.5*VCI + 0.5*SOI
    uvi_score     <- 0.4*beauty + 0.3*safety + 0.3*comfort   (rumus lama)

Setiap skor dinormalisasi min-max ke 1..10 memakai ``CALIBRATION`` -
anchor persentil (p2/p98) dari segmentasi live dataset referensi
(431 foto). Lihat ``scripts/calibrate_pvi.py``.

KETERBATASAN (jujur)
====================
Skor ini BUKAN persepsi manusia. Masih turunan piksel - hanya formulasinya
mengikuti dokumen penelitian, diberi bobot bermakna, dan tidak jenuh.
Validasi terhadap panel penilaian manusia belum ada.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

__all__ = [
    "CALIBRATION",
    "PVI_WEIGHTS",
    "INDEX_KEYS",
    "SCORE_MIN",
    "SCORE_MAX",
    "SAFETY_WEIGHTS",
    "COMFORT_WEIGHTS",
    "UVI_WEIGHTS",
    "compute_pvi",
    "safety_index",
    "comfort_index",
    "compute_scores",
    "compute_scores_from_metrics",
    "indices_from_pct_by_class",
]

# --------------------------------------------------------------------------
# Bobot PVI - tabel "theoretical_weights" notebook penelitian.
# --------------------------------------------------------------------------
PVI_WEIGHTS: Dict[str, float] = {
    "building_visibility_index": 0.20,
    "vegetation_coverage_index": 0.15,
    "sky_openness_index": 0.10,
    "ground_accessibility_index": 0.10,
    "human_activity_index": 0.10,
    "vehicle_intensity_index": 0.05,
    "traffic_infrastructure_index": 0.05,
    "heritage_dominance_index": 0.25,
}

# --------------------------------------------------------------------------
# Anchor kalibrasi: persentil p2 / p98 dari segmentasi live 431 foto
# (data/extracted/photos). Bukan min/max absolut, supaya tidak ada satu
# foto outlier yang menggeser seluruh skala.
# Hitung ulang: python scripts/calibrate_pvi.py --photos data/extracted/photos --write
# --------------------------------------------------------------------------
CALIBRATION: Dict[str, Dict[str, float]] = {
    "pvi": {"min": 0.2633, "max": 0.3820},
    "safety_raw": {"min": 0.1738, "max": 0.3546},
    "comfort_raw": {"min": 0.1616, "max": 0.3918},
}

# Bobot campuran pembentuk safety/comfort/UVI. Default = rumus lama, tetapi
# bisa ditimpa per-request (lihat ``uvip_ai.scoring.profiles``).
SAFETY_WEIGHTS: Dict[str, float] = {
    "ground_accessibility_index": 0.6,
    "traffic_infrastructure_index": 0.4,
}
COMFORT_WEIGHTS: Dict[str, float] = {
    "vegetation_coverage_index": 0.5,
    "sky_openness_index": 0.5,
}
UVI_WEIGHTS: Dict[str, float] = {
    "beauty": 0.4,
    "safety": 0.3,
    "comfort": 0.3,
}

SCORE_MIN = 1.0
SCORE_MAX = 10.0

# Indeks yang wajib ada agar sebuah dict dianggap "metrics Excel" (bukan
# metrics 5-field jalur video). Dipakai build_unified_metrics untuk memutuskan
# jalur mana yang dipakai.
INDEX_KEYS = tuple(PVI_WEIGHTS)


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


def _normalize(value: float, anchor: Mapping[str, float],
               score_min: float = SCORE_MIN, score_max: float = SCORE_MAX) -> float:
    """Normalisasi min-max ke [SCORE_MIN, SCORE_MAX], lalu clamp.

    Anchor ``max == min`` (degenerate) -> kembalikan titik tengah.
    """
    span = float(anchor["max"]) - float(anchor["min"])
    if span <= 0 or score_max <= score_min:
        return (score_min + score_max) / 2.0
    ratio = (value - float(anchor["min"])) / span
    score = score_min + (score_max - score_min) * ratio
    return _clamp(score, score_min, score_max)


def compute_pvi(metrics: Mapping[str, Any],
                weights: Mapping[str, float] | None = None) -> float:
    """PVI - jumlah tertimbang 8 indeks (semua di-clamp ke 0..1)."""
    weights = weights or PVI_WEIGHTS
    return sum(
        float(weight) * _clamp(_get(metrics, key))
        for key, weight in weights.items()
    )


def safety_index(metrics: Mapping[str, Any],
                 weights: Mapping[str, float] | None = None) -> float:
    """Bahan ``safety_score``: akses pejalan kaki + infrastruktur lalu lintas."""
    weights = weights or SAFETY_WEIGHTS
    return sum(w * _clamp(_get(metrics, key)) for key, w in weights.items())


def comfort_index(metrics: Mapping[str, Any],
                  weights: Mapping[str, float] | None = None) -> float:
    """Bahan ``comfort_score``: tutupan vegetasi + keterbukaan langit."""
    weights = weights or COMFORT_WEIGHTS
    return sum(w * _clamp(_get(metrics, key)) for key, w in weights.items())


def compute_scores(metrics: Mapping[str, Any],
                   calibration: Mapping[str, Mapping[str, float]] | None = None,
                   weights: Mapping[str, float] | None = None,
                   blend: Mapping[str, Mapping[str, float]] | None = None,
                   scale: Mapping[str, float] | None = None) -> Dict[str, Any]:
    """8 indeks -> 4 skor siap-kontrak API.

    Parameters
    ----------
    calibration:
        Anchor pengganti ``CALIBRATION`` (dipakai ``scripts/calibrate_pvi.py``
        untuk menilai sebaran skor dengan anchor hasil hitung barunya, sebelum
        ditulis ke modul).
    weights:
        Bobot 8 indeks pengganti ``PVI_WEIGHTS``. Harus sudah berjumlah 1.0;
        normalisasi input pengguna dilakukan ``scoring.profiles``.
    blend:
        ``{"safety": {...}, "comfort": {...}, "uvi": {...}}`` pengganti bobot
        campuran default.
    scale:
        ``{"min": ..., "max": ...}`` batas skor (default 1..10). UVI dibagi
        ``max`` seperti implementasi lama, jadi default tidak berubah.

    Returns
    -------
    dict
        ``beauty_score`` / ``safety_score`` / ``comfort_score`` /
        ``uvi_score`` plus ``_raw`` untuk audit.

    Nilai mentah dihitung penuh (belum dibulatkan) sebelum dipakai
    menghitung ``uvi_score``, lalu semuanya dibulatkan.
    """
    if not metrics:
        metrics = {}
    anchors = calibration or CALIBRATION
    blend = blend or {}
    score_min = float((scale or {}).get("min", SCORE_MIN))
    score_max = float((scale or {}).get("max", SCORE_MAX))

    pvi = compute_pvi(metrics, weights)
    safety_raw = safety_index(metrics, blend.get("safety"))
    comfort_raw = comfort_index(metrics, blend.get("comfort"))

    beauty = _normalize(pvi, anchors["pvi"], score_min, score_max)
    safety = _normalize(safety_raw, anchors["safety_raw"], score_min, score_max)
    comfort = _normalize(comfort_raw, anchors["comfort_raw"], score_min, score_max)

    # UVI = rata-rata tertimbang ketiga skor. Pembagi ``score_max`` (default
    # 10) dipertahankan supaya hasil default identik dengan implementasi lama.
    uvi_weights = blend.get("uvi") or UVI_WEIGHTS
    uvi = (
        beauty * float(uvi_weights.get("beauty", 0.0))
        + safety * float(uvi_weights.get("safety", 0.0))
        + comfort * float(uvi_weights.get("comfort", 0.0))
    ) / score_max

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


def compute_scores_from_metrics(metrics: Mapping[str, Any],
                                calibration: Mapping[str, Mapping[str, float]] | None = None) -> Dict[str, float]:
    """Varian ringkas: hanya 4 field kontrak, tanpa ``_raw``."""
    scores = compute_scores(metrics, calibration)
    scores.pop("_raw", None)
    return scores


def indices_from_pct_by_class(pct_by_class: Mapping[str, Any]) -> Dict[str, float]:
    """Indeks rasio-piksel saja, untuk jalur yang tidak punya seg_map + piksel.

    Dipakai jalur VIDEO (``_run_video_task`` merata-ratakan ``pct_by_class``
    seluruh frame dan memanggil ``build_unified_metrics`` tanpa ``metrics``).

    Empat indeks murni rasio dihitung identik dengan jalur foto. Empat
    indeks yang butuh bentuk/warna/tekstur dihitung versi rasionya saja:

        BVI = building_ratio
        HDI = 0.40 * BVI          (suku solidity/warna/tekstur dihilangkan)

    Konsekuensinya skor video sistematis lebih rendah daripada skor foto
    untuk isi yang sama. Ini disengaja: jalur video tidak boleh menambah
    biaya per-frame dan tidak boleh diubah.
    """
    get = lambda key: _get(pct_by_class, key)  # noqa: E731
    ratio = lambda key: min(get(key) / 100.0, 1.0)  # noqa: E731

    bvi = ratio("building")
    return {
        "building_visibility_index": bvi,
        "vegetation_coverage_index": ratio("vegetation"),
        "sky_openness_index": ratio("sky"),
        "ground_accessibility_index": min(ratio("road") + ratio("sidewalk"), 1.0),
        "human_activity_index": ratio("pedestrian"),
        "vehicle_intensity_index": ratio("vehicle"),
        "traffic_infrastructure_index": ratio("other"),
        "heritage_dominance_index": 0.40 * bvi,
    }
