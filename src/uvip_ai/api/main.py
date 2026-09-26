"""UVIP-AI API server - process foto/video dengan AI segmentation."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path for submodule imports  
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import json
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
_seg_model_cache = None  # Model cache (loaded once, reused for all tasks)

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
    """Run full pipeline AI pada foto → return unified result with saved files."""
    start_time = time.time()
    try:
        import numpy as np
        from uvip_ai.segmentation.segformer import SegformerB5
        from uvip_ai.pipeline.video_processor import CITYSCAPES_COLORS
        from PIL import Image as PILImage
        
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
        pct_by_class = result.get("pct_by_class", {})
        
        # Save segmentation visualization and mask
        seg_dir = Path("uploads") / "segmentation"
        seg_dir.mkdir(parents=True, exist_ok=True)
        masks_dir = Path("uploads") / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        file_stem = Path(path).stem
        
        # Color-coded segmentation map (BGR to RGB)
        seg_img = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        for class_id, color in CITYSCAPES_COLORS.items():
            mask = seg_map == class_id
            if np.any(mask):
                seg_img[mask] = color
        
        pil_seg = PILImage.fromarray(seg_img[..., ::-1])  # BGR to RGB
        seg_path = seg_dir / f"seg_{file_stem}_{timestamp}.jpg"
        pil_seg.save(str(seg_path))
        logger.info("💾 Segmentation saved: %s", seg_path)
        
        # Create privacy masked version (blur non-road areas)
        mask_indices = np.isin(seg_map, [0, 1, 2, 6, 7, 8])  # road, sidewalk, building, wall, fence, pole
        masked_image = image.copy()
        masked_image[~mask_indices] = cv2.GaussianBlur(image[~mask_indices], (5, 5), 0)
        mask_path = masks_dir / f"mask_{file_stem}_{timestamp}.jpg"
        cv2.imwrite(str(mask_path), masked_image)
        logger.info("🛡️ Privacy mask saved: %s", mask_path)
        
        # Create overlay (original + segmentation blend)
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original_rgb, 0.6, seg_img[..., ::-1], 0.4, 0)
        overlay_path = seg_dir / f"overlay_{file_stem}_{timestamp}.jpg"
        pil_overlay = PILImage.fromarray(overlay)
        pil_overlay.save(str(overlay_path))
        logger.info("🎨 Overlay saved: %s", overlay_path)
        
        # Calculate perception predictions
        green_pct = pct_by_class.get('vegetation', 0) + pct_by_class.get('tree', 0)
        building_pct = pct_by_class.get('building', 0)
        road_pct = pct_by_class.get('road', 0)
        sky_pct = pct_by_class.get('sky', 0)
        sidewalk_pct = pct_by_class.get('sidewalk', 0)
        
        beauty_score = min(10, max(0, (green_pct * 0.3) + (sky_pct * 0.4) + (2 if building_pct > 30 else 0)))
        safety_score = min(10, max(0, (sidewalk_pct * 2) + (road_pct * 0.5)))
        comfort_score = min(10, max(0, (green_pct * 0.5) + (sky_pct * 0.5)))
        uvi_score = (beauty_score / 10) * 0.4 + (safety_score / 10) * 0.3 + (comfort_score / 10) * 0.3
        
        # Build response matching expected backend format
        result_response = {
            "privacy_masked_url": f"/uploads/masks/{mask_path.name}",
            "segmentation_url": f"/uploads/segmentation/{seg_path.name}",
            "segmentation_overlay_url": f"/uploads/segmentation/{overlay_path.name}",
            "segmentation_results": {
                "green_coverage_pct": round(pct_by_class.get('vegetation', 0) + pct_by_class.get('tree', 0), 4),
                "building_coverage_pct": round(building_pct, 4),
                "walkability_ratio": round(sidewalk_pct / (road_pct + sidewalk_pct + 0.01), 4),
                "visual_clutter_index": round(1 - (green_pct / 100), 4),
                "sky_visibility_pct": round(sky_pct, 4),
            },
            "perception_prediction": {
                "beauty_score": round(beauty_score, 2),
                "safety_score": round(safety_score, 2),
                "comfort_score": round(comfort_score, 2),
                "uvi_score": round(uvi_score, 2),
            },
            "shap_values": [],
            "metrics_source": {
                "pct_by_class": {k.replace(" ", "_"): round(v, 2) for k, v in pct_by_class.items()},
                "seg_map_shape": list(seg_map.shape),
                "class_count": len(np.unique(seg_map)),
            },
            "_processing_time_ms": int((time.time() - start_time) * 1000),
        }
        
        # Save detailed results JSON
        json_dir = Path("uploads") / "results"
        json_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_dir / f"result_{file_stem}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(result_response, f, indent=2)
        logger.info("📄 Results saved: %s", json_path)
        
        logger.info("✅ Processing complete")
        
        return result_response
    except Exception as e:
        import traceback
        logger.error("Processing error: %s\n%s", str(e), traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
async def _run_video_task(task_id: str, video_path: Path, fps: Optional[float], overlay_alpha: float, photo_id: Optional[str]):
    """Background task untuk process video frame-by-frame dengan segmentation."""
    try:
        from uvip_ai.pipeline.video_processor import VideoProcessor
        # seg_model via get_seg_model() for caching and CUDA probe
        
        with video_tasks_lock:
            video_tasks[task_id]["status"] = "processing"
            video_tasks[task_id]["phase"] = "initializing"
        
        # Initialize models (lazy load) — use cached instance with CUDA probe
        logger.info("Loading models for video task %s...", task_id)
        seg_model = get_seg_model()
        processor = VideoProcessor()
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        target_fps = fps if fps else original_fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with video_tasks_lock:
            video_tasks[task_id]["total_frames"] = total_frames
            video_tasks[task_id]["video_info"] = {
                "original_fps": original_fps,
                "target_fps": target_fps,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
        
        # Process frames
        frame_idx = 0
        output_frames = []
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract and segment
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = seg_model.infer(image, max_resolution=768)
            seg_map = result["seg_map"]
            
            # Create overlay
            overlay = processor.create_overlay(frame, seg_map, alpha=overlay_alpha)
            output_frames.append(overlay)
            
            # Update progress
            video_tasks[task_id]["frames_processed"] = len(output_frames)
            
            if frame_idx % 10 == 0:
                with video_tasks_lock:
                    video_tasks[task_id]["phase"] = f"processing_{frame_idx}_{total_frames}"
            
            frame_idx += 1
        
        cap.release()
        
        # Combine frames to video
        with video_tasks_lock:
            video_tasks[task_id]["phase"] = "combining_frames"
        
        output_video_path = processor.output_dir / f"video_{task_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, target_fps, (output_frames[0].shape[1], output_frames[0].shape[0]))
        
        for frame in output_frames:
            out.write(frame)
        
        out.release()
        
        # Prepare result
        processing_time_ms = int((time.time() - start_time) * 1000)
        video_url = f"/uploads/videos/{output_video_path.name}"
        
        result_response = {
            "status": "completed",
            "task_id": task_id,
            "filename": video_path.name,
            "video_url": video_url,
            "total_frames": len(output_frames),
            "video_info": video_tasks[task_id]["video_info"],
            "processing_time_ms": processing_time_ms,
            "created_at": video_tasks[task_id]["created_at"],
        }
        
        with video_tasks_lock:
            video_tasks[task_id]["status"] = "completed"
            video_tasks[task_id]["result"] = result_response
            video_tasks[task_id]["finished_at"] = time.time()
        
        logger.info("✅ Video task %s completed: %s frames in %dms", 
                    task_id, len(output_frames), processing_time_ms)
        
        # Post to backend if photo_id provided
        if photo_id:
            await _post_result_to_backend(photo_id, result_response)
    
    except Exception as e:
        import traceback
        logger.error("❌ Video task %s failed: %s\n%s", task_id, str(e), traceback.format_exc())
        
        with video_tasks_lock:
            video_tasks[task_id]["status"] = "failed"
            video_tasks[task_id]["error"] = str(e)
            video_tasks[task_id]["finished_at"] = time.time()


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
