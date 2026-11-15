"""Evaluator gambar AI: segmentasi 8 indeks komposit & perbandingan Before vs After.

Modul ini menyediakan fungsi untuk:
1. Menjalankan model SegFormer-B5 pada gambar hasil generate AI (atau gambar apa pun di disk).
2. Menghasilkan 8 indeks komposit riset (BVI, VCI, SOI, GAI, HAI, VII, TII, HDI).
3. Menghitung 4 skor persepsi perkotaan (Beauty, Safety, Comfort, UVI).
4. Menyimpan segmentation map & overlay visual untuk gambar AI.
5. Membandingkan metrik foto awal (Before) vs gambar hasil redesain AI (After).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np
from PIL import Image as PILImage

from uvip_ai.pipeline.video_processor import CITYSCAPES_COLORS
from uvip_ai.scoring import profiles as scoring_profiles
from uvip_ai.scoring.pvi_scorer import (
    INDEX_KEYS,
    indices_from_pct_by_class,
)
from uvip_ai.utils.storage import (
    SEGMENTATION_DIR,
    output_path,
    rel_path,
    timestamp_ms,
    upload_url,
)

logger = logging.getLogger("uvip_ai.evaluator")

_seg_model_cache = None
_seg_model_lock = threading.Lock()


def get_loaded_model():
    """Kembalikan model jika sudah dimuat, atau None tanpa trigger lazy load."""
    return _seg_model_cache


def get_seg_model():
    """Get cached SegFormer model singleton (thread-safe, reuse across app)."""
    global _seg_model_cache
    if _seg_model_cache is None:
        with _seg_model_lock:
            if _seg_model_cache is None:
                logger.info("Loading SegFormer model...")
                from uvip_ai.segmentation.segformer import SegformerB5
                _seg_model_cache = SegformerB5()
                logger.info("Model loaded successfully")
    return _seg_model_cache


def evaluate_image(
    image_path: Path | str,
    stem: str = "gen",
    save_overlay: bool = True,
    scoring: Optional[dict] = None,
    seg_model=None,
) -> dict[str, Any]:
    """Jalankan inferensi SegFormer-B5 pada gambar di disk dan hitung 8 indeks + 4 skor.

    Args:
        image_path: Path lokal ke file gambar (JPG / PNG).
        stem: Stem nama file dasar untuk penamaan artefak overlay.
        save_overlay: Jika True, simpan visualisasi overlay segmentasi ke uploads/segmentation/.
        scoring: Tuning profil scoring (opsional).
        seg_model: Instance SegformerB5 (opsional, default memakai singleton get_seg_model()).

    Returns:
        Dict berisi indices (8 indeks), scores (4 persepsi), segmentation_results,
        overlay_url, overlay_path, dan pct_by_class.
    """
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File gambar tidak ditemukan: {path}")

    start_time = time.time()
    model = seg_model or get_seg_model()

    bgr_image = cv2.imread(str(path))
    if bgr_image is None:
        raise ValueError(f"Gagal membaca file gambar dengan OpenCV: {path}")

    # SegFormer dilatih pada RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    infer_res = model.infer(rgb_image, excel_mode=True)
    seg_map = infer_res["seg_map"]
    metrics = infer_res["metrics"]
    pct_by_class = infer_res["pct_by_class"]

    # Hitung skor persepsi terkalibrasi (Beauty, Safety, Comfort, UVI)
    scoring = scoring or {}
    scored = scoring_profiles.score_indices(
        metrics,
        scoring.get("tuning"),
        profile=scoring.get("profile", "default"),
        normalized=scoring.get("normalized", False),
        warnings=scoring.get("warnings"),
    )

    stamp = timestamp_ms()
    overlay_url_str = None
    overlay_rel_path = None
    seg_url_str = None

    if save_overlay:
        # Bangun gambar segmentasi warna Cityscapes
        seg_img = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        for class_id, color in CITYSCAPES_COLORS.items():
            mask = (seg_map == class_id)
            if np.any(mask):
                seg_img[mask] = color

        # Overlay blend (60% gambar asli, 40% warna segmentasi)
        overlay = cv2.addWeighted(rgb_image, 0.6, seg_img[..., ::-1], 0.4, 0)
        overlay_path = output_path(SEGMENTATION_DIR, "overlay_gen_", stem, ".jpg", stamp)
        pil_overlay = PILImage.fromarray(overlay)
        pil_overlay.save(str(overlay_path), quality=92)
        overlay_url_str = upload_url(overlay_path)
        overlay_rel_path = rel_path(overlay_path)

        seg_path = output_path(SEGMENTATION_DIR, "seg_gen_", stem, ".jpg", stamp)
        pil_seg = PILImage.fromarray(seg_img[..., ::-1])
        pil_seg.save(str(seg_path), quality=90)
        seg_url_str = upload_url(seg_path)

    # 5 metrik urban standar
    green_pct = pct_by_class.get("vegetation", 0) + pct_by_class.get("tree", 0)
    building_pct = pct_by_class.get("building", 0)
    road_pct = pct_by_class.get("road", 0)
    sky_pct = pct_by_class.get("sky", 0)
    sidewalk_pct = pct_by_class.get("sidewalk", 0)

    # 8 indeks komposit
    clean_indices = {
        k: round(float(metrics.get(k, 0.0)), 4)
        for k in INDEX_KEYS
    }

    elapsed_ms = int((time.time() - start_time) * 1000)

    return {
        "indices": clean_indices,
        "scores": scored["perception_prediction"],
        "segmentation_results": {
            "green_coverage_pct": round(green_pct, 4),
            "building_coverage_pct": round(building_pct, 4),
            "walkability_ratio": round(sidewalk_pct / (road_pct + sidewalk_pct + 0.01), 4),
            "visual_clutter_index": round(1.0 - (green_pct / 100.0), 4),
            "sky_visibility_pct": round(sky_pct, 4),
        },
        "pct_by_class": {k.replace(" ", "_"): round(v, 2) for k, v in pct_by_class.items()},
        "overlay_url": overlay_url_str,
        "overlay_path": overlay_rel_path,
        "segmentation_url": seg_url_str,
        "scoring": scored.get("scoring"),
        "elapsed_ms": elapsed_ms,
    }


def extract_before_metrics(target: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    """Ambil 8 indeks dan 4 skor persepsi dari hasil /ai/process original.

    Mendukung output baru (yang memiliki block `scoring.indices`) serta fallback
    dari `pct_by_class`.
    """
    scores = dict(target.get("perception_prediction") or {})

    scoring_block = target.get("scoring")
    indices = {}
    if isinstance(scoring_block, dict) and isinstance(scoring_block.get("indices"), dict):
        indices = dict(scoring_block["indices"])

    if not indices or len(indices) < len(INDEX_KEYS):
        metrics_source = target.get("metrics_source") or {}
        pct = metrics_source.get("pct_by_class") or {}
        if pct:
            indices = indices_from_pct_by_class(pct)

    clean_indices = {
        k: round(float(indices.get(k, 0.0)), 4)
        for k in INDEX_KEYS
    }
    return clean_indices, scores


def compare_evaluations(
    before_indices: Mapping[str, float],
    after_indices: Mapping[str, float],
    before_scores: Mapping[str, float],
    after_scores: Mapping[str, float],
) -> dict[str, Any]:
    """Hitung komparasi Before vs After untuk 8 indeks dan 4 skor persepsi."""
    delta_indices = {}
    for k in INDEX_KEYS:
        b = float(before_indices.get(k, 0.0))
        a = float(after_indices.get(k, 0.0))
        diff = a - b
        if abs(b) > 1e-5:
            pct_change = (diff / b) * 100.0
            pct_str = f"{pct_change:+.1f}%"
        else:
            pct_str = "+100.0%" if a > 0 else "0.0%"
        delta_indices[k] = {
            "before": round(b, 4),
            "after": round(a, 4),
            "diff": round(diff, 4),
            "percent_change": pct_str,
        }

    delta_scores = {}
    for k in ("beauty_score", "safety_score", "comfort_score", "uvi_score"):
        b = float(before_scores.get(k, 0.0))
        a = float(after_scores.get(k, 0.0))
        diff = a - b
        delta_scores[k] = {
            "before": round(b, 2),
            "after": round(a, 2),
            "diff": round(diff, 2),
            "formatted": f"{diff:+.2f}",
        }

    summary = {
        "vegetation_change": delta_indices.get("vegetation_coverage_index", {}).get("percent_change", "0.0%"),
        "ground_access_change": delta_indices.get("ground_accessibility_index", {}).get("percent_change", "0.0%"),
        "vehicle_intensity_change": delta_indices.get("vehicle_intensity_index", {}).get("percent_change", "0.0%"),
        "sky_openness_change": delta_indices.get("sky_openness_index", {}).get("percent_change", "0.0%"),
        "uvi_score_change": delta_scores.get("uvi_score", {}).get("formatted", "+0.00"),
    }

    return {
        "indices": delta_indices,
        "scores": delta_scores,
        "summary": summary,
    }
