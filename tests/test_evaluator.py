"""Unit tests untuk ``uvip_ai.pipeline.evaluator``."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from uvip_ai.pipeline.evaluator import (
    compare_evaluations,
    evaluate_image,
    extract_before_metrics,
)
from uvip_ai.scoring.pvi_scorer import INDEX_KEYS


def test_extract_before_metrics_from_scoring_block():
    target = {
        "perception_prediction": {
            "beauty_score": 6.5,
            "safety_score": 5.0,
            "comfort_score": 4.5,
            "uvi_score": 5.4,
        },
        "scoring": {
            "indices": {
                "building_visibility_index": 0.40,
                "vegetation_coverage_index": 0.15,
                "sky_openness_index": 0.20,
                "ground_accessibility_index": 0.30,
                "human_activity_index": 0.05,
                "vehicle_intensity_index": 0.10,
                "traffic_infrastructure_index": 0.05,
                "heritage_dominance_index": 0.25,
            }
        }
    }
    indices, scores = extract_before_metrics(target)
    assert scores["beauty_score"] == 6.5
    assert indices["vegetation_coverage_index"] == 0.15
    assert indices["building_visibility_index"] == 0.40
    assert set(indices.keys()) == set(INDEX_KEYS)


def test_extract_before_metrics_fallback_pct_by_class():
    target = {
        "perception_prediction": {"beauty_score": 5.0, "uvi_score": 4.0},
        "metrics_source": {
            "pct_by_class": {
                "building": 30.0,
                "vegetation": 20.0,
                "tree": 10.0,
                "sky": 15.0,
                "road": 20.0,
                "sidewalk": 5.0,
            }
        }
    }
    indices, scores = extract_before_metrics(target)
    assert scores["beauty_score"] == 5.0
    assert indices["vegetation_coverage_index"] > 0
    assert indices["building_visibility_index"] > 0
    assert len(indices) == len(INDEX_KEYS)


def test_compare_evaluations():
    before_indices = {k: 0.20 for k in INDEX_KEYS}
    after_indices = {k: 0.20 for k in INDEX_KEYS}
    after_indices["vegetation_coverage_index"] = 0.40  # naik 100%
    after_indices["vehicle_intensity_index"] = 0.10     # turun 50%

    before_scores = {"beauty_score": 5.0, "safety_score": 5.0, "comfort_score": 5.0, "uvi_score": 5.0}
    after_scores = {"beauty_score": 7.5, "safety_score": 6.0, "comfort_score": 8.0, "uvi_score": 7.2}

    res = compare_evaluations(before_indices, after_indices, before_scores, after_scores)
    assert "indices" in res
    assert "scores" in res
    assert "summary" in res

    veg_comp = res["indices"]["vegetation_coverage_index"]
    assert veg_comp["diff"] == 0.20
    assert veg_comp["percent_change"] == "+100.0%"

    veh_comp = res["indices"]["vehicle_intensity_index"]
    assert veh_comp["diff"] == -0.10
    assert veh_comp["percent_change"] == "-50.0%"

    uvi_comp = res["scores"]["uvi_score"]
    assert uvi_comp["diff"] == 2.2
    assert uvi_comp["formatted"] == "+2.20"


def test_evaluate_image_with_mock_model(tmp_path: Path):
    import cv2

    # Buat dummy image 100x100
    dummy_img_path = tmp_path / "test_generated.png"
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = (120, 180, 240)  # BGR
    cv2.imwrite(str(dummy_img_path), img)

    # Mock SegformerB5 model
    mock_model = MagicMock()
    mock_seg_map = np.ones((100, 100), dtype=np.int32)
    mock_metrics = {
        "building_visibility_index": 0.35,
        "vegetation_coverage_index": 0.40,
        "sky_openness_index": 0.15,
        "ground_accessibility_index": 0.25,
        "human_activity_index": 0.05,
        "vehicle_intensity_index": 0.05,
        "traffic_infrastructure_index": 0.05,
        "heritage_dominance_index": 0.20,
    }
    mock_pct = {
        "vegetation": 40.0,
        "building": 35.0,
        "sky": 15.0,
        "road": 5.0,
        "sidewalk": 5.0,
    }
    mock_model.infer.return_value = {
        "seg_map": mock_seg_map,
        "metrics": mock_metrics,
        "pct_by_class": mock_pct,
    }

    result = evaluate_image(
        image_path=dummy_img_path,
        stem="mock_test",
        save_overlay=True,
        seg_model=mock_model,
    )

    assert "indices" in result
    assert result["indices"]["vegetation_coverage_index"] == 0.40
    assert result["indices"]["building_visibility_index"] == 0.35
    assert "scores" in result
    assert "beauty_score" in result["scores"]
    assert "uvi_score" in result["scores"]
    assert "segmentation_results" in result
    assert result["segmentation_results"]["green_coverage_pct"] == 40.0
    assert result["overlay_url"] is not None
    assert result["overlay_path"] is not None


def test_api_evaluate_generated_endpoint(monkeypatch):
    from fastapi.testclient import TestClient
    from uvip_ai.api.main import app
    from uvip_ai.utils.storage import UPLOADS_DIR

    # Buat file dummy di uploads/generated
    gen_dir = UPLOADS_DIR / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    test_file = gen_dir / "test_api_gen_123.png"
    test_file.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    mock_eval = {
        "indices": {k: 0.3 for k in INDEX_KEYS},
        "scores": {"beauty_score": 7.0, "safety_score": 6.5, "comfort_score": 6.8, "uvi_score": 6.8},
        "segmentation_results": {
            "green_coverage_pct": 30.0,
            "building_coverage_pct": 35.0,
            "walkability_ratio": 0.4,
            "visual_clutter_index": 0.7,
            "sky_visibility_pct": 20.0,
        },
        "overlay_url": "/uploads/segmentation/overlay_gen_123.jpg",
        "elapsed_ms": 120,
    }
    import sys
    router_mod = sys.modules["uvip_ai.llm.router"]
    monkeypatch.setattr(router_mod, "evaluate_image", lambda *args, **kwargs: mock_eval)

    client = TestClient(app)
    try:
        res = client.post(
            "/llm/evaluate-generated",
            data={
                "generated_path": "uploads/generated/test_api_gen_123.png",
                "result": '{"perception_prediction": {"uvi_score": 5.0}, "scoring": {"indices": {"vegetation_coverage_index": 0.1}}}',
            }
        )
        assert res.status_code == 200
        data = res.json()
        assert "generated_metrics" in data
        assert "generated_scores" in data
        assert data["generated_scores"]["beauty_score"] == 7.0
        assert "comparison" in data
        assert "summary" in data["comparison"]
    finally:
        if test_file.exists():
            test_file.unlink()
