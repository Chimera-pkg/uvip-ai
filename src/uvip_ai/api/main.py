"""
FastAPI Endpoint — Integrasi UVIP AI Pipeline (Step 8).

POST /ai/process → upload foto + GPS → return:
  - privacy_masked_url (path ke hasil blur)
  - segmentation_results (5 metrik urban)
  - perception_prediction (Beauty/Safety/Comfort/UVI)
  - shap_values (faktor pendorong Indonesia)

Endpoint ini dipanggil oleh WebSocket handler saat foto masuk dari mobile.
Target latency < 700ms (Step 9).
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import cv2
import fastapi
from fastapi import File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Setup logging — tampil di journalctl -u uvip -f
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("uvip_ai")

app = fastapi.FastAPI(title="UVIP-AI API", version="0.1.0")


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


class ProcessRequest(BaseModel):
    """Payload request (jika bukan multipart)."""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    is_offline_sync: bool = False


def save_upload(file: UploadFile) -> Path:
    path = Path("uploads/temp") / f"{file.filename}_{os.getpid()}"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = file.file.read()
    path.write_bytes(content)
    return path


def post_process(path: Path) -> dict:
    """Run full pipeline AI pada foto → return unified result."""
    # Step A: Privacy Guard — PrivacyGuard tidak punya low_vram_mode, gunakan default
    from uvip_ai.privacy.guard import PrivacyGuard
    guard = PrivacyGuard()
    guard_result = guard.process_image(str(path))
    masked_img = guard_result["blurred_image"]   # np.ndarray BGR
    boxes = guard_result.get("detections", [])

    # Step B: Segmentation
    from uvip_ai.segmentation.segformer import SegformerB5
    seg = SegformerB5(low_vram_mode=True)
    seg_res = seg.infer(masked_img)
    metrics = seg_res["metrics"]

    # Step C: Feature extraction (pakai model kecil untuk CPU)
    from uvip_ai.features.dinov2 import Dinov2Extractor
    from uvip_ai.config import settings
    feat_model = Dinov2Extractor(model_id=settings.dinov2_model, low_vram_mode=True)
    emb = feat_model.extract(str(path))
    logger.info("DINOv2 embedding: shape=%s", emb.shape)

    # Step D: XGBoost prediction
    from uvip_ai.training.xgboost_model import XGBoostPerceptionModel
    xgb_model = XGBoostPerceptionModel(model_path=settings.xgboost_model_path)
    predictions = xgb_model.predict(metrics, embedding=emb)
    logger.info("XGBoost predictions: %s", predictions)

    # Step E: SHAP explainability
    from uvip_ai.explain.shap_explain import ShapExplainer
    explainer = ShapExplainer(model_path=settings.xgboost_model_path)
    feature_names = [
        "green_coverage_pct",
        "building_coverage_pct",
        "walkability_ratio",
        "visual_clutter_index",
        "sky_visibility_pct",
    ]
    # Gabungkan metrik + embedding untuk SHAP
    features_for_shap = np.array([
        metrics["green_coverage_pct"],
        metrics["building_coverage_pct"],
        metrics["walkability_ratio"],
        metrics["visual_clutter_index"],
        metrics["sky_visibility_pct"],
    ])
    if emb is not None:
        features_for_shap = np.concatenate([features_for_shap, emb.flatten()])
    explanations = explainer.explain(features_for_shap, feature_names=feature_names)
    logger.info("SHAP explanations: %d features", len(explanations))

    # Cleanup
    seg.free_memory()
    feat_model.free_memory()
    xgb_model.free_memory()
    explainer.free_memory()

    # Save masked image
    out_dir = Path("uploads/masks")
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / f"mask_{path.name}"
    cv2.imwrite(str(mask_path), masked_img if isinstance(masked_img, np.ndarray) else cv2.imread(str(path)))

    return {
        "privacy_masked_url": str(mask_path.relative_to(Path.cwd())),
        "segmentation_results": {
            "green_coverage_pct": metrics["green_coverage_pct"],
            "building_coverage_pct": metrics["building_coverage_pct"],
            "walkability_ratio": metrics["walkability_ratio"],
            "visual_clutter_index": metrics["visual_clutter_index"],
            "sky_visibility_pct": metrics["sky_visibility_pct"],
        },
        "perception_prediction": predictions,
        "shap_values": explanations,
    }


async def _post_result_to_backend(photo_id: str, result: dict):
    """Kirim hasil segmentasi ke backend-uvip untuk disimpan ke DB."""
    from uvip_ai.config import settings
    if not settings.uvip_api_base_url or not photo_id:
        return

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(
                f"{settings.uvip_api_base_url}/segmentation-results/",
                json={"photo_id": photo_id, **result},
                headers={"Authorization": f"Bearer {settings.uvip_api_token}"} if settings.uvip_api_token else {},
            )
            logger.info("📤 Callback ke backend berhasil: photo_id=%s", photo_id)
    except Exception as e:
        logger.error("❌ Callback ke backend gagal: %s", e)


@app.post("/ai/process")
async def process_photo(
    file: UploadFile = File(...),
    photo_id: str = Form(""),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Process foto jalan → return masking, segmentasi, prediksi, explainability."""
    start = time.time()
    logger.info("📥 Foto masuk: %s (size: %s, photo_id: %s)", file.filename, file.size, photo_id)
    try:
        path = save_upload(file)
        logger.info("💾 Foto disimpan: %s", path)

        logger.info("⚙️  Memproses: %s ...", file.filename)
        result = post_process(path)

        elapsed = (time.time() - start) * 1000
        logger.info("✅ Selesai: %s — %.0fms", file.filename, elapsed)

        # Callback ke backend di background (tidak blocking response)
        if photo_id:
            background_tasks.add_task(_post_result_to_backend, photo_id, result)

        path.unlink(missing_ok=True)
        return JSONResponse(result)
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        logger.error("❌ Gagal: %s — %s (%.0fms)", file.filename, str(e), elapsed)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
