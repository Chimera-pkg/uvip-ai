"""Kontrak ``uvip_ai.scoring.pvi_scorer``.

Yang dijaga di sini bukan angka kalibrasi (itu berubah tiap dataset
referensi berubah), melainkan sifat yang harus selalu benar:

    * skor tidak pernah keluar 1..10 / uvi 0..1
    * anchor min/max benar-benar memetakan ke ujung skala (anti-saturasi)
    * anchor degenerate tidak bikin ZeroDivisionError
    * sumbu metrik bergerak ke arah yang masuk akal
    * rasio piksel di-clamp sebelum jadi indeks
"""

from __future__ import annotations

import pytest

from uvip_ai.scoring.pvi_scorer import (
    SCORE_MAX,
    SCORE_MIN,
    compute_scores,
    compute_scores_from_metrics,
    indices_from_pct_by_class,
)

ANCHOR = {
    "pvi": {"min": 0.1, "max": 0.5},
    "safety_raw": {"min": 0.2, "max": 0.8},
    "comfort_raw": {"min": 0.1, "max": 0.9},
}


def _metrics(**overrides: float) -> dict:
    """Delapan indeks bernilai 0.5, bisa ditimpa per indeks."""
    base = {
        "building_visibility_index": 0.5,
        "vegetation_coverage_index": 0.5,
        "sky_openness_index": 0.5,
        "ground_accessibility_index": 0.5,
        "human_activity_index": 0.5,
        "vehicle_intensity_index": 0.5,
        "traffic_infrastructure_index": 0.5,
        "heritage_dominance_index": 0.5,
    }
    base.update(overrides)
    return base


def test_scores_stay_inside_contract_range():
    for raw in (0.0, 0.5, 1.0, 7.0, -3.0):
        scores = compute_scores_from_metrics(
            {k: raw for k in _metrics()}, calibration=ANCHOR
        )
        for key in ("beauty_score", "safety_score", "comfort_score"):
            assert SCORE_MIN <= scores[key] <= SCORE_MAX, (key, raw, scores)
        assert 0.0 <= scores["uvi_score"] <= 1.0


def test_anchor_endpoints_reach_both_ends_of_the_scale():
    low = compute_scores(_metrics(), calibration={
        "pvi": {"min": 1.0, "max": 2.0},
        "safety_raw": {"min": 1.0, "max": 2.0},
        "comfort_raw": {"min": 1.0, "max": 2.0},
    })
    high = compute_scores(_metrics(), calibration={
        "pvi": {"min": -1.0, "max": -0.5},
        "safety_raw": {"min": -1.0, "max": -0.5},
        "comfort_raw": {"min": -1.0, "max": -0.5},
    })
    assert (low["beauty_score"], low["safety_score"], low["comfort_score"]) == (
        SCORE_MIN, SCORE_MIN, SCORE_MIN
    )
    assert (high["beauty_score"], high["safety_score"], high["comfort_score"]) == (
        SCORE_MAX, SCORE_MAX, SCORE_MAX
    )


def test_degenerate_anchor_returns_midpoint():
    scores = compute_scores(_metrics(), calibration={
        "pvi": {"min": 0.3, "max": 0.3},
        "safety_raw": {"min": 0.3, "max": 0.3},
        "comfort_raw": {"min": 0.3, "max": 0.3},
    })
    mid = (SCORE_MIN + SCORE_MAX) / 2
    assert scores["beauty_score"] == scores["safety_score"] == scores["comfort_score"] == mid


def test_more_vegetation_never_lowers_comfort():
    sparse = compute_scores_from_metrics(
        _metrics(vegetation_coverage_index=0.1), calibration=ANCHOR
    )
    dense = compute_scores_from_metrics(
        _metrics(vegetation_coverage_index=0.9), calibration=ANCHOR
    )
    assert dense["comfort_score"] > sparse["comfort_score"]


@pytest.mark.parametrize("pct", [{"road": 90, "sidewalk": 40}, {"signage": 70, "street_furniture": 60}])
def test_pixel_ratios_are_clamped_to_one(pct):
    indices = indices_from_pct_by_class(pct)
    for key, value in indices.items():
        assert 0.0 <= value <= 1.0, (key, value)
    if "road" in pct:
        assert indices["ground_accessibility_index"] == pytest.approx(1.0)
    else:
        assert indices["traffic_infrastructure_index"] == pytest.approx(1.0)
