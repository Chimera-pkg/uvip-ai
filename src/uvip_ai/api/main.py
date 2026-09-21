"""UVIP-AI API server - process foto/video dengan AI segmentation."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path for submodule imports  
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import os
import shutil
import time
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import cv2  # OpenCV for image processing
from fastapi import BackgroundTasks

import fastapi
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

# Setup logging — tampil di journalctl -u uvip -f
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger("uvip_ai")

# ─── Background Task Storage (video processing) ─────────────────────────────
video_tasks: dict = {}
video_tasks_lock = threading.Lock()
TASK_CLEANUP_HOURS = 168  # 7 hari
def get_seg_model():
    """Get cached SegFormer model. Load sekali, reuse semua task."""
    global _seg_model_cache
    if _seg_model_cache is None:
        logger.info("Loading SegFormer model...")
        from uvip_ai.segmentation.segformer import SegformerB5
        _seg_model_cache = SegformerB5()
        logger.info("Model loaded successfully")
    return _seg_model_cache


def _cleanup_old_tasks():
    """Hapus task data yang sudah selesai lebih dari TASK_CLEANUP_HOURS."""
    now = datetime.utcnow()
    to_remove = [
        tid for tid, tdata in video_tasks.items()
        if tdata["status"] in ["finished", "failed"]
        and (now - tdata["started_at"]) > timedelta(hours=TASK_CLEANUP_HOURS)
    ]
    for tid in to_remove:
        del video_tasks[tid]
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

    url: Optional[str] = None
    is_offline_sync: bool = False


def save_upload(file: UploadFile) -> Path:
    stem = Path(file.filename).stem
    path = uploads_dir / f"{stem}_{uuid.uuid4()}{Path(file.filename).suffix}"
    file.file.seek(0)
    with open(path, "wb") as f:
        content = file.file.read()
        f.write(content)
    return path


def post_process(path: Path) -> dict:
    """Run full pipeline AI pada foto → return unified result."""
    try:
        import numpy as np
        from uvip_ai.segmentation.segformer import SegformerB5
        
        # Load model
        seg = SegformerB5(low_vram_mode=True)
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        
        # Run inference
        result = seg.infer(image, excel_mode=True)
        
        # Prepare output
        metrics = result["metrics"]
        seg_map = result["seg_map"]
        
        logger.info("✅ Segmentation complete")
        
        return {
            "metrics": metrics,
            "seg_map_shape": seg_map.shape,
            "class_count": len(np.unique(seg_map)),
        }
    except Exception as e:
        logger.error("Processing error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


def _run_video_task(
    task_id: str, source_path: Path, target_fps: float, overlay_alpha: float, photo_id: Optional[str],
) -> None:
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

        # Get cached model (load sekali, reuse)
        seg = get_seg_model()

        # Process frames in memory (avoid disk I/O)
        processed_frames = []
        for i, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            seg_res = seg.infer(frame)
            seg_map = seg_res["seg_map"]
            overlay = processor.create_overlay(frame, seg_map, alpha=overlay_alpha)

            processed_frames.append(overlay)
            frame_path.unlink(missing_ok=True)

            with video_tasks_lock:
                video_tasks[task_id]["frames_processed"] = i + 1

            if (i + 1) % 10 == 0:
                logger.info("Task %s: %d/%d frames", task_id, i + 1, total_frames)

        # Don't free model - keep cached for next task

        with video_tasks_lock:
            video_tasks[task_id]["phase"] = "combining_video"

        # Write video directly from memory
        output_filename = f"segmented_{task_id}.mp4"
        output_path = video_out_dir / output_filename

        if processed_frames:
            h, w = processed_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_path), fourcc, effective_fps, (w, h))
            try:
                for frame in processed_frames:
                    out.write(frame)
            finally:
                out.release()

        source_path.unlink(missing_ok=True)

        elapsed = (time.time() - start) * 1000
        logger.info("✅ Task %s done (%.0fms)", task_id, elapsed)

        result = {
            "video_url": f"/uploads/videos/{output_filename}",
            "video_info": video_info,
            "frames_processed": len(processed_frames),
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
    """Process video → segmentation overlay (async). Return task_id immediately, poll /ai/process-video/status/{task_id} for progress."""
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

    background_tasks.add_task(_run_video_task, task_id, path, fps, overlay_alpha, photo_id)
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


@app.get("/ai/process-video/tasks")
async def list_video_tasks(status: Optional[str] = None):
    """List semua video tasks. Optional filter by status: queued, processing, completed, failed."""
    with video_tasks_lock:
        tasks_copy = video_tasks.copy()

    result = []
    for task_id, task in tasks_copy.items():
        if status and task["status"] != status:
            continue

        progress = None
        if task["total_frames"] and task["total_frames"] > 0:
            progress = round(task["frames_processed"] / task["total_frames"] * 100, 1)

        task_info = {
            "task_id": task_id,
            "status": task["status"],
            "phase": task["phase"],
            "filename": task["filename"],
            "total_frames": task["total_frames"],
            "frames_processed": task["frames_processed"],
            "progress_pct": progress,
            "error": task["error"],
            "created_at": task["created_at"],
            "finished_at": task.get("finished_at"),
        }

        if task["status"] == "completed" and task.get("result"):
            task_info["video_url"] = task["result"].get("video_url")
            task_info["processing_time_ms"] = task["result"].get("processing_time_ms")

        result.append(task_info)

    result.sort(key=lambda x: x["created_at"], reverse=True)

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
@app.post("/")
async def serverless_root(payload: dict = None):
    """Root handler for RunPod Queue Serverless"""
    if payload is None:
        return JSONResponse(status_code=400, content={"error": "No payload"})
    
    try:
        from src.uvip_ai.runpod_serverless import runpod_serverless
        return JSONResponse(content=runpod_serverless(payload))
    except Exception as e:
        logger.error(f"Serverless error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})                                                                                
                         

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
