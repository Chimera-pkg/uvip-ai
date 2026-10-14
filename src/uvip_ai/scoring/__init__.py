"""Modul penilaian (scoring) UVIP AI.

Isi utama: `pvi_scorer` - mengubah 8 indeks penelitian (Excel-aligned)
menjadi 4 skor yang dikonsumsi kontrak API `perception_prediction`.

Modul ini BERDIRI SENDIRI (pure function, tanpa I/O, tanpa model ML)
sehingga bisa diuji tanpa memuat Segformer/XGBoost.
"""

from uvip_ai.scoring.pvi_scorer import (  # noqa: F401
    CALIBRATION,
    PVI_WEIGHTS,
    compute_pvi,
    compute_scores,
    compute_scores_from_metrics,
    indices_from_pct_by_class,
)

__all__ = [
    "CALIBRATION",
    "PVI_WEIGHTS",
    "compute_pvi",
    "compute_scores",
    "compute_scores_from_metrics",
    "indices_from_pct_by_class",
]
