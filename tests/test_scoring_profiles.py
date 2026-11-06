"""Kontrak tuning skor: ``uvip_ai.scoring.profiles``.

Yang dijaga: input pengguna tidak bisa membuat skor keluar batas, bobot selalu
berjumlah 1.0 sebelum dipakai, dan profil tersimpan bisa dibaca ulang persis.
"""

from __future__ import annotations

import pytest

from uvip_ai.scoring import profiles

VEGETATION_ONLY = {
    "building_visibility_index": 0.0,
    "vegetation_coverage_index": 1.0,
    "sky_openness_index": 0.0,
    "ground_accessibility_index": 0.0,
    "human_activity_index": 0.0,
    "vehicle_intensity_index": 0.0,
    "traffic_infrastructure_index": 0.0,
    "heritage_dominance_index": 0.0,
}

def test_partial_weights_are_normalized_before_scoring():
    spec = profiles.resolve(tuning={"weights": {"vegetation_coverage_index": 0.5}})

    assert spec["normalized"] is True
    assert spec["warnings"]
    assert sum(spec["tuning"]["weights"].values()) == pytest.approx(1.0)

    tuned = profiles.score_indices(VEGETATION_ONLY, spec["tuning"])["perception_prediction"]
    default = profiles.score_indices(VEGETATION_ONLY, profiles.DEFAULT_TUNING)["perception_prediction"]
    assert tuned["beauty_score"] > default["beauty_score"]

def test_scale_bounds_are_respected():
    spec = profiles.resolve(tuning={"scale": {"min": 3, "max": 4}})
    scores = profiles.score_indices({"vegetation_coverage_index": 1.0}, spec["tuning"])["perception_prediction"]

    for field in profiles.SCORE_FIELDS:
        assert 3.0 <= scores[field] <= 4.0, (field, scores)
    assert 0.0 <= scores["uvi_score"] <= 1.0

def test_degenerate_anchor_span_is_rejected():
    with pytest.raises(profiles.TuningError):
        profiles.parse_tuning({"anchors": {"pvi": {"min": 0.3, "max": 0.301}}})

def test_unknown_weight_key_is_rejected():
    with pytest.raises(profiles.TuningError):
        profiles.parse_tuning({"weights": {"hdi": 0.2}})

def test_partial_indices_are_reported_not_fatal():
    result = profiles.score_indices({"vegetation_coverage_index": 0.4}, profiles.DEFAULT_TUNING)
    assert result["scoring"]["indices"]["vegetation_coverage_index"] == pytest.approx(0.4)
    assert any("tidak ada" in warning for warning in result["scoring"]["warnings"])

def test_every_slider_can_reach_its_default():
    """Slider yang max-nya di bawah default memaksa nilai berubah diam-diam."""
    for group in profiles.schema()["groups"]:
        for slider in group.get("sliders", []):
            assert slider["min"] <= slider["default"] <= slider["max"], (group["id"], slider["path"])
        for entry in group.get("ranges", []):
            assert entry["min"] <= entry["min_default"] <= entry["max"], (group["id"], entry["path"])
            assert entry["min"] <= entry["max_default"] <= entry["max"], (group["id"], entry["path"])

def test_profile_roundtrip_and_name_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(profiles, "PROFILES_DIR", tmp_path)

    saved = profiles.save_profile("uji_slider", {"scale": {"min": 2, "max": 9}})
    loaded = profiles.load_profile("uji_slider")

    assert loaded["profile_hash"] == saved["profile_hash"]
    assert loaded["tuning"]["scale"] == {"min": 2.0, "max": 9.0}
    assert [record["name"] for record in profiles.list_profiles()] == ["uji_slider"]

    assert profiles.delete_profile("uji_slider") is True
    assert profiles.delete_profile("uji_slider") is False
    with pytest.raises(KeyError):
        profiles.load_profile("uji_slider")

    with pytest.raises(profiles.TuningError):
        profiles.save_profile("../evil", {})
