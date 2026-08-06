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

import os
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import fastapi
from fastapi import File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = fastapi.FastAPI(title="UVIP-AI API", version="0.1.0")


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


class ProcessRequest(fastapi.BaseModel):
    """Payload request (jika bukan multipart)."""
    latitude: float | None = None
    longitude: float | None = None
    is_offline_sync: bool = False


def save_upload(file: UploadFile) -> Path:
    path = Path("uploads/temp") / f"{file.filename}_{os.getpid()}"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = file.file.read()
    path.write_bytes(content)
    return path


def post_process(path: Path) -> dict:
    """Run full pipeline AI pada foto → return unified result."""
    # Step A: Privacy Guard
    from uvip_ai.privacy.guard import PrivacyGuard
    guard = PrivacyGuard(low_vram_mode=True)
    masked_img, boxes = guard.process(str(path))

    # Step B: Segmentation
    from uvip_ai.segmentation.segformer import SegformerB5
    seg = SegformerB5(low_vram_mode=True)
    seg_res = seg.infer(masked_img if hasattr(masked_img, 'convert') else path)
    metrics = seg_res["metrics"]

    # Step C: Feature extraction
    from uvip_ai.features.dinov2 import Dinov2Extractor
    feat_model = Dinov2Extractor(low_vram_mode=True)
    emb = feat_model.extract(masked_img if hasattr(masked_img, 'convert') else path)

    # Step D: XGBoost prediction (stub: use dummy model if not trained yet)
    # TODO: load model & predict
    predictions = {"beauty_score": 6.0, "safety_score": 6.5, "comfort_score": 6.2, "uvi_score": 6.3}

    # Step E: SHAP explainability
    from uvip_ai.explain.shap_explain import ShapExplainer
    # dummy features
    x = np.zeros(1029)
    explainer = ShapExplainer(None, [])  # will fail if no model; skip for stub
    explanations = []

    # Cleanup
    guard.free_memory(); seg.free_memory(); feat_model.free_memory()

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


@app.post("/ai/process")
async def process_photo(file: UploadFile = File(...)):
    """Process foto jalan → return masking, segmentasi, prediksi, explainability."""
    try:
        path = save_upload(file)
        result = post_process(path)
        path.unlink(missing_ok=True)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
