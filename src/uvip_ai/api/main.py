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
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import cv2
import fastapi
from fastapi import File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Setup logging — tampil di journalctl -u uvip -f
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("uvip_ai")

# ─── Background Task Storage (video processing) ─────────────────────────────
video_tasks: dict = {}
video_tasks_lock = threading.Lock()
TASK_CLEANUP_HOURS = 168  # 7 hari


def _cleanup_old_tasks():
    """Hapus task data yang sudah selesai lebih dari TASK_CLEANUP_HOURS."""
    now = time.time()
    cutoff = now - (TASK_CLEANUP_HOURS * 3600)
    with video_tasks_lock:
        expired = [
            tid for tid, t in video_tasks.items()
            if t.get("finished_at") and t["finished_at"] < cutoff
        ]
        for tid in expired:
            task = video_tasks.pop(tid)
            task_dir = Path("uploads/tasks") / tid
            shutil.rmtree(task_dir, ignore_errors=True)
            logger.info("Cleaned up old task: %s", tid)


app = fastapi.FastAPI(title="UVIP-AI API", version="0.1.0")

# Serve static files dari uploads/
uploads_dir = Path("uploads")
uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")


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
    seg_map = seg_res["seg_map"]

    # Generate segmentation visualizations
    from uvip_ai.pipeline.video_processor import CITYSCAPES_COLORS

    # 1. Raw segmentation map (color-coded classes)
    seg_img = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for class_id, color in CITYSCAPES_COLORS.items():
        mask = seg_map == class_id
        if np.any(mask):
            seg_img[mask] = color

    # 2. Overlay (segmentation blended with original)
    original_img = masked_img if isinstance(masked_img, np.ndarray) else cv2.imread(str(path))
    overlay = cv2.addWeighted(original_img, 0.5, seg_img, 0.5, 0)

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

    # Save segmentation visualizations
    seg_dir = Path("uploads/segmentation")
    seg_dir.mkdir(parents=True, exist_ok=True)

    # 1. Raw segmentation map
    seg_path = seg_dir / f"seg_{path.name}"
    cv2.imwrite(str(seg_path), seg_img)

    # 2. Overlay (segmentation + original)
    overlay_path = seg_dir / f"overlay_{path.name}"
    cv2.imwrite(str(overlay_path), overlay)

    return {
        "privacy_masked_url": str(mask_path),
        "segmentation_url": str(seg_path),
        "segmentation_overlay_url": str(overlay_path),
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


def _run_video_task(
    task_id: str,
    source_path: Path,
    target_fps: float,
    overlay_alpha: float,
    photo_id: Optional[str],
):
    """Background worker: process video frames + segmentation."""
    start = time.time()
    task_dir = Path("uploads/tasks") / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = task_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_out_dir = Path("uploads/videos")
    video_out_dir.mkdir(parents=True, exist_ok=True)

    with video_tasks_lock:
        video_tasks[task_id]["status"] = "processing"
        video_tasks[task_id]["phase"] = "extracting_frames"

    try:
        from uvip_ai.pipeline.video_processor import VideoProcessor
        from uvip_ai.segmentation.segformer import SegformerB5

        processor = VideoProcessor()
        video_info = processor.get_video_info(str(source_path))
        effective_fps = target_fps if target_fps else video_info["fps"]

        with video_tasks_lock:
            video_tasks[task_id]["video_info"] = video_info

        # Extract frames
        frame_paths = processor.extract_frames(str(source_path), fps=effective_fps)
        total_frames = len(frame_paths)
        logger.info("Task %s: extracted %d frames", task_id, total_frames)

        with video_tasks_lock:
            video_tasks[task_id]["total_frames"] = total_frames
            video_tasks[task_id]["phase"] = "segmentation"

        # Load model once
        seg = SegformerB5(low_vram_mode=True)

        temp_processed = []
        for i, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            seg_res = seg.infer(frame)
            seg_map = seg_res["seg_map"]
            overlay = processor.create_overlay(frame, seg_map, alpha=overlay_alpha)

            temp_path = frames_dir / f"processed_{i:06d}.jpg"
            cv2.imwrite(str(temp_path), overlay)
            temp_processed.append(temp_path)
            frame_path.unlink(missing_ok=True)

            with video_tasks_lock:
                video_tasks[task_id]["frames_processed"] = i + 1

            if (i + 1) % 10 == 0:
                logger.info("Task %s: %d/%d frames", task_id, i + 1, total_frames)

        seg.free_memory()

        with video_tasks_lock:
            video_tasks[task_id]["phase"] = "combining_video"

        output_filename = f"segmented_{task_id}.mp4"
        output_path = video_out_dir / output_filename
        processor.combine_frames_to_video(temp_processed, str(output_path), fps=effective_fps)

        for p in temp_processed:
            p.unlink(missing_ok=True)
        source_path.unlink(missing_ok=True)

        elapsed = (time.time() - start) * 1000
        logger.info("✅ Task %s done (%.0fms)", task_id, elapsed)

        result = {
            "video_url": str(output_path),
            "video_info": video_info,
            "frames_processed": len(temp_processed),
            "processing_time_ms": elapsed,
        }

        with video_tasks_lock:
            video_tasks[task_id].update({
                "status": "completed",
                "phase": "done",
                "result": result,
                "finished_at": time.time(),
            })

        if photo_id:
            try:
                import httpx
                with httpx.Client(timeout=10.0) as client:
                    client.post(
                        os.environ.get("BACKEND_URL", "http://localhost:8000")
                        + "/api/ai/video-result",
                        json={"photo_id": photo_id, **result},
                    )
            except Exception as cb_err:
                logger.warning("Callback failed for task %s: %s", task_id, cb_err)

    except Exception as e:
        elapsed = (time.time() - start) * 1000
        logger.error("❌ Task %s gagal: %s (%.0fms)", task_id, str(e), elapsed)
        for p in frames_dir.glob("*.jpg"):
            p.unlink(missing_ok=True)
        source_path.unlink(missing_ok=True)
        with video_tasks_lock:
            video_tasks[task_id].update({
                "status": "failed",
                "error": str(e),
                "finished_at": time.time(),
            })


@app.post("/ai/process-video")
async def process_video(
    file: UploadFile = File(...),
    photo_id: Optional[str] = Form(None),
    fps: Optional[float] = Form(None),
    overlay_alpha: float = Form(0.5),
    background_tasks: BackgroundTasks = None,
):
    """
    Process video → segmentation overlay (async).
    Return task_id immediately, poll /ai/process-video/status/{task_id} for progress.
    """
    import asyncio

    _cleanup_old_tasks()

    task_id = str(uuid.uuid4())[:8]
    path = await asyncio.to_thread(save_upload, file)

    with video_tasks_lock:
        video_tasks[task_id] = {
            "status": "queued",
            "phase": "queued",
            "filename": file.filename,
            "total_frames": None,
            "frames_processed": 0,
            "video_info": None,
            "result": None,
            "error": None,
            "created_at": time.time(),
            "finished_at": None,
        }

    background_tasks.add_task(
        _run_video_task, task_id, path, fps, overlay_alpha, photo_id
    )
    logger.info("Task %s queued: %s", task_id, file.filename)

    return JSONResponse({
        "task_id": task_id,
        "status": "queued",
        "status_url": f"/ai/process-video/status/{task_id}",
        "result_url": f"/ai/process-video/result/{task_id}",
    })


@app.get("/ai/process-video/status/{task_id}")
async def video_task_status(task_id: str):
    """Poll progress untuk task video."""
    with video_tasks_lock:
        task = video_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    progress = None
    if task["total_frames"] and task["total_frames"] > 0:
        progress = round(task["frames_processed"] / task["total_frames"] * 100, 1)

    return {
        "task_id": task_id,
        "status": task["status"],
        "phase": task["phase"],
        "filename": task["filename"],
        "total_frames": task["total_frames"],
        "frames_processed": task["frames_processed"],
        "progress_pct": progress,
        "error": task["error"],
    }


@app.get("/ai/process-video/result/{task_id}")
async def video_task_result(task_id: str):
    """Download hasil video setelah task selesai."""
    with video_tasks_lock:
        task = video_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] == "processing" or task["status"] == "queued":
        raise HTTPException(status_code=202, detail="Task still processing")
    if task["status"] == "failed":
        raise HTTPException(status_code=500, detail=task["error"])
    return JSONResponse(task["result"])


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
