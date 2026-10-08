
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

# Status akhir sebuah task: tidak akan berubah lagi, aman untuk dibersihkan
TERMINAL_STATUSES = ("completed", "failed", "finished")


def _cleanup_old_tasks():
    """Hapus task data yang sudah selesai lebih dari TASK_CLEANUP_HOURS.

    Aman dipanggil kapan saja: task yang belum selesai / data lama yang tidak
    lengkap akan dilewati, bukan memunculkan KeyError.
    """
    now_ts = time.time()
    cutoff_seconds = TASK_CLEANUP_HOURS * 3600
    to_remove = []
    with video_tasks_lock:
        for tid, tdata in list(video_tasks.items()):
            if tdata.get("status") not in TERMINAL_STATUSES:
                continue
            # finished_at = kapan task selesai; created_at = fallback (task lama)
            ts = tdata.get("finished_at") or tdata.get("created_at")
            if not isinstance(ts, (int, float)):
                continue
            if (now_ts - ts) > cutoff_seconds:
                to_remove.append(tid)
        for tid in to_remove:
            del video_tasks[tid]
    for tid in to_remove:
        logger.info("Cleaned up old task: %s", tid)


app = fastapi.FastAPI(title="UVIP-AI API", version="0.1.0")

# Serve static files dari uploads/ — ABSOLUTE ke repo root.
# Dulu relatif (Path("uploads")) sehingga restart dari cwd berbeda bikin file
# "hilang" (tertulis di folder lain, diserve dari folder lain). Sekarang satu lokasi pasti.
REPO_ROOT = Path(__file__).resolve().parents[3]
uploads_dir = REPO_ROOT / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)


@app.get("/uploads/{rel_path:path}")
async def serve_upload(rel_path: str):
    """Serve file uploads/ TANPA cache + dengan range support.

    Cache-Control: no-store — browser/proxy selalu unduh ulang, tidak ada
    304. File video tidak pernah dioverwrite (nama unik video_<id>.mp4) dan
    tidak pernah dihapus, jadi yang diserve selalu file asli di disk.
    Catatan: 206 Partial Content akan TETAP muncul — itu bukan cache, itu
    cara browser streaming video (minta per potong supaya bisa seek).
    """
    from fastapi.responses import FileResponse
    safe = (uploads_dir / rel_path.lstrip("/")).resolve()
    if uploads_dir not in safe.parents and safe != uploads_dir:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not safe.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    suffix = safe.suffix.lower()
    media_type = {
        ".mp4": "video/mp4", ".webm": "video/webm", ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg", ".png": "image/png", ".json": "application/json",
    }.get(suffix, "application/octet-stream")
    return FileResponse(
        str(safe), media_type=media_type,
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Accept-Ranges": "bytes",
        },
    )


# Build stamp - bukti versi kode yang benar-benar jalan (lihat GET /health)
BUILD_STAMP = "jpg-h264-v15-libx264-limited-range"

# Binary ffmpeg yang sudah terbukti bisa encode (diisi oleh _encode_h264_frames)
_ffmpeg_exe_cache: Optional[str] = None

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat(),
            "build": BUILD_STAMP, "encode_path": "jpg_frames->libx264",
            "ffmpeg_in_use": _ffmpeg_exe_cache,
            "encode_selftest": _encode_selftest.get("status")}


@app.get("/health/ffmpeg")
async def health_ffmpeg():
    """Diagnosa: binary ffmpeg mana yang benar-benar bisa encode, dan lewat
    filter/encoder mana.

    Diuji pada frame genap DAN ganjil: frame video sumber bisa beresolusi
    ganjil dan itu salah satu penyebab libx264 gagal dibuka.
    """
    import tempfile
    import numpy as np
    rows = []
    with tempfile.TemporaryDirectory() as td:
        even = Path(td) / "even.jpg"
        odd = Path(td) / "odd.jpg"
        if not cv2.imwrite(str(even), np.zeros((64, 64, 3), dtype=np.uint8)):
            return {"error": "cv2.imwrite gagal - OpenCV bermasalah"}
        cv2.imwrite(str(odd), np.zeros((65, 63, 3), dtype=np.uint8))
        for exe in _ffmpeg_candidates():
            ok_even, det_even = _probe_ffmpeg(exe, even)
            ok_odd, det_odd = _probe_ffmpeg(exe, odd)
            h264_even, det_h264 = _probe_ffmpeg(exe, even, h264_only=True)
            rows.append({
                "path": exe, "version": _ffmpeg_version(exe),
                "encoders": _ffmpeg_encoders(exe),
                "encode_ok": ok_even, "detail": det_even,
                "h264_ok": h264_even, "h264_detail": det_h264,
                "odd_dims_ok": ok_odd, "odd_dims_detail": det_odd,
            })
    return {"build": BUILD_STAMP, "candidates": rows,
            "filter_chains": [n for n, _ in FILTER_CHAINS],
            "encode_attempts": [f"{e} {' '.join(o)}" for e, o in ENCODE_ATTEMPTS],
            "ffmpeg_in_use": _ffmpeg_exe_cache,
            "startup_selftest": _encode_selftest,
            "hint": "encode_ok=false di semua kandidat -> "
                    "apt-get update && apt-get install -y --reinstall ffmpeg. "
                    "h264_ok=false di semua kandidat berarti task video GAGAL "
                    "dengan jelas (mpeg4 SENGAJA tidak ditulis karena tidak "
                    "playable di Chrome/Firefox <video>); perbaiki dengan "
                    "ffmpeg distro (apt-get install -y ffmpeg) atau "
                    "pip install imageio-ffmpeg."}


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

def _ffmpeg_candidates() -> list:
    """Daftar binary ffmpeg yang mungkin dipakai, urut prioritas.

    Urutan: (1) binary yang sudah TERBUKTI bisa encode (hasil self-test/probe
    sebelumnya), (2) ffmpeg distro /usr/bin/ffmpeg, (3) binary static
    imageio-ffmpeg.

    ffmpeg distro didahulukan karena paket distro hampir selalu punya libx264
    yang sehat. Binary static imageio-ffmpeg bisa punya libx264 rusak
    (persis kasus di server: `h264_ok: false`) — dulu binary inilah yang
    dicoba pertama, sehingga setiap task membuang waktu di encoder rusak
    sebelum jatuh ke ffmpeg distro.
    """
    cands = []
    if _ffmpeg_exe_cache and Path(_ffmpeg_exe_cache).exists():
        cands.append(_ffmpeg_exe_cache)
    sys_ff = shutil.which("ffmpeg")
    if sys_ff and sys_ff not in cands:
        cands.append(sys_ff)
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        exe = get_ffmpeg_exe()
        if exe and Path(exe).exists() and exe not in cands:
            cands.append(exe)
    except Exception as exc:  # ImportError / RuntimeError / dll.
        logger.warning("imageio-ffmpeg tidak tersedia: %s", exc)
    return cands


# libx264 menolak dimensi ganjil ("width not divisible by 2"). Semua encode
# H.264 lewat filter ini supaya frame beresolusi ganjil tetap bisa diproses.
PAD_FILTER = "pad=ceil(iw/2)*2:ceil(ih/2)*2"

# Rantai filter yang dicoba berurutan. libx264 menolak dimensi ganjil
# ("width not divisible by 2"), jadi chain pertama memaksa dimensi genap.
# Subsampling yuv420p sendiri sudah dipasang lewat `-pix_fmt yuv420p` di
# command line, jadi tidak perlu diulang di setiap chain.
#
# `scale=out_range=tv` menormalkan rentang warna ke limited/TV range. Tanpa ini
# ffmpeg memilih yuvj420p (full-range) karena sumbernya JPG - format yang sudah
# deprecated di ffmpeg 7 dan ditandai "pc" di ffprobe. Limited range adalah
# bentuk standar untuk H.264. Chain kedua tetap ada kalau build tertentu tidak
# punya filter `scale` yang mendukung out_range.
FILTER_CHAINS = [
    ("pad+range+format", f"{PAD_FILTER},format=yuv420p,scale=out_range=tv"),
    ("pad+format", f"{PAD_FILTER},format=yuv420p"),
    ("none", None),
]

# Encoder H.264 saja. mpeg4 SENGAJA tidak ada di daftar: VLC muter mpeg4
# tapi Chrome/Firefox (<video>) tidak. Fallback mpeg4 = task "sukses" yang
# tidak bisa diplay di web. Kalau semua H.264 gagal, task GAGAL dengan jelas.
#
# Percobaan 1 = H.264 "normal": preset veryfast + CRF 23 + profil High (default
# libx264) + yuv420p. Ini kombinasi yang diterima Chrome, Firefox, Safari, Edge,
# Android, iOS, dan VLC. Percobaan 2 hanya jaring kalau CRF ditolak build
# tertentu. libopenh264 TIDAK dipakai: tidak ada di binary imageio-ffmpeg
# maupun ffmpeg distro, jadi hanya memperpanjang pesan error.
ENCODE_ATTEMPTS = [
    ("libx264", ["-preset", "veryfast", "-crf", "23"]),
    ("libx264", ["-preset", "veryfast", "-b:v", "4M"]),
]

# Bukti codec di dalam file hasil encode. Chrome/Firefox hanya mau H.264 (avc1)
# di dalam MP4; mpeg4 "sukses" di ffmpeg tapi mati di <video>. OpenCV melaporkan
# fourcc yang berbeda-beda antar versi/platform ('h264', 'avc1', 'X264'), jadi
# dicocokkan sebagai substring tanpa peduli huruf besar/kecil.
_H264_TAGS = ("h264", "avc1", "x264", "264")


def _is_h264(codec: str) -> bool:
    c = (codec or "").lower()
    return any(tag in c for tag in _H264_TAGS)


# Hasil self-test encoder saat startup (dilihat di GET /health).
_encode_selftest: dict = {"status": "pending"}


def _ffmpeg_version(exe: str) -> str:
    """Baris pertama `ffmpeg -version` (untuk diagnosa)."""
    import subprocess
    try:
        r = subprocess.run([exe, "-version"], capture_output=True, timeout=30)
        return (r.stdout.decode(errors="ignore").splitlines() or ["?"])[0]
    except Exception as exc:
        return f"gagal ambil versi: {exc}"


def _run_ffmpeg(cmd: list, timeout: int = 300) -> tuple:
    """Jalankan ffmpeg. Return (ok: bool, stderr_ringkas: str).

    stderr diringkas head+tail. Penyebab asli libx264 gagal dibuka
    ("height not divisible by 2", "Cannot allocate memory", dst.) ada di
    AWAL output; memotong ekor saja (err[-400:]) justru membuangnya dan
    menyisakan ringkasan generik "Generic error in an external library" -
    itu yang membuat kegagalan lama tidak bisa didiagnosa.
    """
    import subprocess
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timeout {timeout}s"
    err = r.stderr.decode(errors="ignore").strip()
    if r.returncode == 0:
        return True, err[-400:]
    if not err:
        return False, f"exit {r.returncode} tanpa stderr"
    if len(err) <= 900:
        return False, err
    return False, f"[HEAD] {err[:400]} ... [TAIL] {err[-400:]}"


def _ffmpeg_encoders(exe: str) -> list:
    """Encoder video yang tersedia di binary ini (untuk diagnosa libx264 hilang)."""
    import subprocess
    try:
        r = subprocess.run([exe, "-hide_banner", "-encoders"],
                           capture_output=True, timeout=30)
        text = r.stdout.decode(errors="ignore")
    except Exception as exc:
        return [f"gagal: {exc}"]
    wanted = ("libx264", "libopenh264", "h264", "mpeg4", "libx265")
    found = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            name = parts[1]
            if any(w in name for w in wanted):
                found.append(name)
    return found


def _probe_ffmpeg(exe: str, sample_frame: Path, fps: float = 25.0,
                  h264_only: bool = False) -> tuple:
    """Bukti binary ini BENAR-BENAR bisa encode dari 1 frame JPG.

    Return (ok: bool, detail: str). Beberapa build ffmpeg (mis. di container)
    punya libx264 yang gagal dibuka walau `-version` normal, jadi `-version`
    saja tidak cukup sebagai bukti.

    h264_only=True melewatkan jaring mpeg4, sehingga "ok" di sini berarti
    H.264 benar-benar tersedia (bukan sekadar bisa encode apa saja).

    Urutan filter/encoder di sini HARUS sama dengan encode asli, supaya
    encoder yang sehat tidak dianggap rusak hanya karena beda filter.
    """
    import tempfile
    last = ""
    attempts = [a for a in ENCODE_ATTEMPTS
                if a[0].startswith("lib")] if h264_only else ENCODE_ATTEMPTS
    with tempfile.TemporaryDirectory() as td:
        for chain_name, vf in FILTER_CHAINS:
            for enc, enc_opts in attempts:
                out = Path(td) / "probe.mp4"
                cmd = [exe, "-y", "-loglevel", "error", "-hide_banner",
                       "-framerate", f"{fps}", "-i", str(sample_frame)]
                if vf:
                    cmd += ["-vf", vf]
                cmd += ["-frames:v", "1", "-c:v", enc, "-pix_fmt", "yuv420p",
                        *enc_opts, str(out)]
                ok, detail = _run_ffmpeg(cmd, timeout=120)
                if ok and out.exists() and out.stat().st_size > 0:
                    return True, (f"ok enc={enc} filter={chain_name} "
                                  f"({out.stat().st_size} byte)")
                last = f"enc={enc} filter={chain_name}: {detail}"
    return False, last or "tidak ada kombinasi yang berhasil"


def _startup_encode_selftest() -> None:
    """Tes encode 1 frame saat startup; hasilnya di GET /health.

    Tujuannya memisahkan dua kelas kegagalan: "ffmpeg di server memang tidak
    bisa encode" (terlihat di sini sebelum ada upload) vs "gagal karena frame
    video tertentu" (baru terlihat saat task jalan).
    """
    import tempfile
    import numpy as np
    global _encode_selftest
    try:
        with tempfile.TemporaryDirectory() as td:
            probe = Path(td) / "selftest.jpg"
            if not cv2.imwrite(str(probe), np.zeros((64, 64, 3), dtype=np.uint8)):
                _encode_selftest = {"status": "error",
                                    "error": "cv2.imwrite gagal"}
                return
            results = {}
            winner = None
            for exe in _ffmpeg_candidates():
                ok, detail = _probe_ffmpeg(exe, probe, h264_only=True)
                results[exe] = {"ok": ok, "detail": detail,
                                "h264": ok,
                                "version": _ffmpeg_version(exe)}
                logger.info("Self-test encode H.264 %s: %s", exe,
                            detail if ok else f"GAGAL - {detail}")
                if ok and winner is None:
                    winner = exe
            # Kunci binary yang sudah terbukti sehat, supaya task video pertama
            # tidak membuang waktu di binary yang libx264-nya rusak.
            if winner:
                global _ffmpeg_exe_cache
                _ffmpeg_exe_cache = winner
                logger.info("ffmpeg dipakai: %s (lolos self-test H.264)", winner)
            _encode_selftest = {
                "status": "ok" if winner else "failed",
                "h264_available": bool(winner),
                "winner": winner,
                "results": results,
            }
            if _encode_selftest["status"] == "failed":
                logger.error(
                    "SEMUA ffmpeg gagal encode H.264 saat startup. Task video "
                    "akan GAGAL (bukan turun ke mpeg4) sampai ini diperbaiki: "
                    "apt-get update && apt-get install -y --reinstall ffmpeg, "
                    "atau pastikan imageio-ffmpeg terpasang "
                    "(pip install imageio-ffmpeg)")
    except Exception as exc:  # jangan sampai bikin app gagal start
        _encode_selftest = {"status": "error", "error": str(exc)}
        logger.warning("Self-test encode error: %s", exc)


@app.on_event("startup")
async def _on_startup() -> None:
    import threading
    # di thread terpisah: startup tidak boleh menunggu ffmpeg
    threading.Thread(target=_startup_encode_selftest, daemon=True).start()


def _encode_h264_frames(frames_dir: Path, fps: float, dst: Path) -> None:
    """Encode urutan JPG -> MP4 H.264 (browser-playable). TANPA fallback mpeg4.

    Keputusan sadar 2026-09-30: mpeg4 bisa diputar VLC tapi TIDAK oleh
    Chrome/Firefox <video> — fallback itu menghasilkan task "completed" yang
    tidak bisa diplay di kondisi normal. Jadi kalau seluruh H.264 gagal,
    fungsi ini raise RuntimeError (task jadi "failed" dengan pesan jelas)
    alih-alih menulis file mpeg4 yang menipu.
    """
    global _ffmpeg_exe_cache

    # fps invalid (0/NaN/inf) bikin libx264 gagal buka encoder - paksa ke rentang aman
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        fps = 0.0
    if not (0.1 <= fps <= 120):
        fps = 25.0

    frames = sorted(frames_dir.glob("f_*.jpg"))
    if not frames:
        raise RuntimeError(f"tidak ada frame JPG di {frames_dir}")
    n_frames = len(frames)
    pattern = str(frames_dir / "f_%06d.jpg")

    candidates = _ffmpeg_candidates()
    if not candidates:
        raise RuntimeError(
            "ffmpeg tidak ditemukan. Install: apt-get install -y ffmpeg "
            "atau pip install --force-reinstall imageio-ffmpeg")

    h264_attempts = ENCODE_ATTEMPTS  # semuanya libx264 (lihat ENCODE_ATTEMPTS)

    problems = []

    def _try(exe: str, enc: str, enc_opts: list, vf, chain_name: str) -> bool:
        global _ffmpeg_exe_cache
        cmd = [exe, "-y", "-loglevel", "error", "-hide_banner",
               "-framerate", f"{fps:.6f}", "-i", pattern]
        if vf:
            cmd += ["-vf", vf]
        cmd += ["-c:v", enc, "-pix_fmt", "yuv420p", "-r", f"{fps:.6f}",
                # Tag range/ruang warna eksplisit. Sumber JPG itu full-range
                # (yuvj420p), dan tanpa tag ini ffmpeg bisa menulis "unknown"
                # sehingga pemutar menebak rentang warna. H.264 yang benar
                # untuk web adalah limited/TV range + bt709.
                "-color_range", "tv", "-colorspace", "bt709",
                "-color_primaries", "bt709", "-color_trc", "bt709",
                *enc_opts, "-movflags", "+faststart", str(dst)]
        logger.info("ffmpeg: %s", " ".join(cmd))
        ok, detail = _run_ffmpeg(cmd, timeout=900)
        if ok and dst.exists() and dst.stat().st_size > 0:
            # Cek codec: ffmpeg bisa exit 0 tapi menulis codec lain. File non-H.264
            # = "sukses" palsu (mati di <video>), jadi jangan diterima.
            codec = _video_codec(dst)
            if not _is_h264(codec):
                detail = f"codec hasil '{codec}' bukan H.264"
                logger.warning("encode ditolak enc=%s filter=%s: %s",
                               enc, chain_name, detail)
                problems.append(f"{exe} [enc={enc},filter={chain_name}] {detail}")
                return False
            # cache hanya diisi setelah encode TERBUKTI berhasil
            _ffmpeg_exe_cache = exe
            logger.info("Encode sukses: %s enc=%s codec=%s filter=%s (%d frame, %d byte)",
                        exe, enc, codec, chain_name, n_frames, dst.stat().st_size)
            return True
        logger.warning("encode gagal enc=%s filter=%s: %s", enc, chain_name, detail)
        problems.append(
            f"{exe} [enc={enc},filter={chain_name}] "
            f"[{_ffmpeg_version(exe)}]: {detail}")
        return False

    # Tahap 1: H.264 di semua binary dulu.
    for exe in candidates:
        ok, detail = _probe_ffmpeg(exe, frames[0], fps)
        logger.info("Probe %s [%s]: %s", exe, _ffmpeg_version(exe),
                    detail if ok else f"GAGAL - {detail}")
        for chain_name, vf in FILTER_CHAINS:
            for enc, enc_opts in h264_attempts:
                if _try(exe, enc, enc_opts, vf, chain_name):
                    return

    # Tidak ada Tahap 2. mpeg4 SENGAJA tidak pernah ditulis: file mpeg4 bisa
    # "sukses" di ffmpeg dan tetap bisa dibuka OpenCV, tapi Chrome/Firefox
    # <video> menolaknya (MEDIA_ERR_SRC_NOT_SUPPORTED). Itu menghasilkan task
    # "completed" yang videonya mati di web — persis bug yang pernah terjadi.
    # Kalau semua H.264 gagal, lebih baik task "failed" dengan pesan jelas.
    raise RuntimeError(
        f"semua ffmpeg gagal encode H.264 (fps={fps}, frames={n_frames}). "
        + " || ".join(problems)
        + " || Cek GET /health/ffmpeg lalu: apt-get update && "
          "apt-get install -y --reinstall ffmpeg, atau pip install imageio-ffmpeg")


def _video_codec(path: Path) -> str:
    """Nama codec video di dalam file (mis. 'h264', 'mpeg4', 'vp90').

    Dipakai sebagai bukti nyata codec yang tertulis, karena `ffmpeg` bisa
    exit 0 sementara OpenCV tetap bisa membuka file non-H.264. Mengembalikan
    string kosong kalau tidak bisa dibaca.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return ""
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        return "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")
    finally:
        cap.release()


def _verify_playable(path: Path) -> tuple[int, int]:
    """Buka file hasil seperti browser/player. Return (w, h). Raise kalau tidak playable.

    Dua syarat, keduanya wajib:
    1. Bisa dibuka & dibaca sebagai video (file tidak corrupt/0-byte).
    2. Codec-nya H.264. Syarat kedua ini yang dulu hilang: `mpeg4` lolos
       syarat (1) karena OpenCV bisa decode mpeg4, padahal Chrome/Firefox
       <video> menolaknya. Itu sebabnya video "completed" tapi tidak bisa
       diputar di web.
    """
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"video output kosong: {path}")
    codec = _video_codec(path)
    if not _is_h264(codec):
        raise RuntimeError(
            f"video bukan H.264 (codec terbaca: '{codec or 'tidak terbaca'}') "
            f"-> tidak bisa diputar Chrome/Firefox: {path.name}")
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"video tidak bisa di-decode: {path.name}")
    return frame.shape[1], frame.shape[0]

def build_unified_metrics(pct_by_class: dict, seg_shape, processing_time_ms: int) -> dict:
    """Format metrics unified — sama persis untuk foto & video."""
    green_pct = pct_by_class.get('vegetation', 0) + pct_by_class.get('tree', 0)
    building_pct = pct_by_class.get('building', 0)
    road_pct = pct_by_class.get('road', 0)
    sky_pct = pct_by_class.get('sky', 0)
    sidewalk_pct = pct_by_class.get('sidewalk', 0)
    beauty_score = min(10, max(0, (green_pct * 0.3) + (sky_pct * 0.4) + (2 if building_pct > 30 else 0)))
    safety_score = min(10, max(0, (sidewalk_pct * 2) + (road_pct * 0.5)))
    comfort_score = min(10, max(0, (green_pct * 0.5) + (sky_pct * 0.5)))
    uvi_score = (beauty_score / 10) * 0.4 + (safety_score / 10) * 0.3 + (comfort_score / 10) * 0.3
    return {
        "segmentation_results": {
            "green_coverage_pct": round(green_pct, 4),
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
            "seg_map_shape": list(seg_shape),
            "class_count": len(pct_by_class),
        },
        "_processing_time_ms": processing_time_ms,
    }


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
        seg_map = result["seg_map"]
        metrics = result["metrics"]
        pct_by_class = result["pct_by_class"]
        # Save segmentation visualization and mask
        seg_dir = uploads_dir / "segmentation"
        seg_dir.mkdir(parents=True, exist_ok=True)
        masks_dir = uploads_dir / "masks"
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
        keep_classes = np.isin(seg_map, [0, 1, 2, 6, 7, 8])  # road, sidewalk, building, wall, fence, pole
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        masked_image = np.where(keep_classes[..., None], image, blurred)
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
        
        unified = build_unified_metrics(pct_by_class, seg_map.shape, int((time.time() - start_time) * 1000))
        result_response = {
            "privacy_masked_url": f"/uploads/masks/{mask_path.name}",
            "segmentation_url": f"/uploads/segmentation/{seg_path.name}",
            "segmentation_overlay_url": f"/uploads/segmentation/{overlay_path.name}",
            **unified,
        }

        json_dir = uploads_dir / "results"
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
def _run_video_task(task_id: str, video_path: Path, fps: Optional[float], overlay_alpha: float, photo_id: Optional[str]):
    # Sync (bukan async): Starlette menjalankannya di threadpool, jadi infer
    # PyTorch yang blocking tidak membekukan event loop (status/result 404/202
    # palsu). Jangan jadikan async lagi tanpa executor.
    """Background task untuk process video frame-by-frame dengan segmentation."""
    try:
        from uvip_ai.pipeline.video_processor import VideoProcessor
        # seg_model via get_seg_model() for caching and CUDA probe
        
        with video_tasks_lock:
            video_tasks[task_id]["status"] = "processing"
            video_tasks[task_id]["phase"] = "initializing"
        
        # Initialize models (lazy load) — use cached instance with CUDA probe
        logger.info("Loading models for video task %s...", task_id)
        t0 = time.time()
        seg_model = get_seg_model()
        logger.info("Model ready for task %s in %.0fs", task_id, time.time() - t0)
        processor = VideoProcessor()
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if not (isinstance(original_fps, (int, float)) and original_fps > 0 and original_fps < 1000):
            original_fps = 25.0
        target_fps = fps if (isinstance(fps, (int, float)) and fps > 0 and fps < 1000) else original_fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with video_tasks_lock:
            video_tasks[task_id]["total_frames"] = total_frames
            video_tasks[task_id]["phase"] = "processing_0"
            video_tasks[task_id]["video_info"] = {
                "original_fps": original_fps,
                "target_fps": target_fps,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
        
        # Process frames
        frame_idx = 0
        output_frames = []
        frame_pcts = []
        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = seg_model.infer(image, max_resolution=768)
            seg_map = result["seg_map"]
            frame_pcts.append(result.get("pct_by_class", {}))
            overlay = processor.create_overlay(frame, seg_map, alpha=overlay_alpha)
            output_frames.append(overlay)
            
            # Update progress
            with video_tasks_lock:
                video_tasks[task_id]["frames_processed"] = len(output_frames)

            if frame_idx % 10 == 0:
                logger.info("Task %s: frame %d/%s", task_id, frame_idx, total_frames)
                with video_tasks_lock:
                    video_tasks[task_id]["phase"] = f"processing_{frame_idx}_{total_frames}"

            frame_idx += 1
        
        cap.release()
        
        # Combine frames to video
        with video_tasks_lock:
            video_tasks[task_id]["phase"] = "combining_frames"
        
        if not output_frames:
            raise ValueError("No frames processed")
        output_video_path = processor.output_dir / f"video_{task_id}.mp4"
        # VideoWriter+mp4v unreliable di server (file 0-byte/raw corrupt -> ffmpeg
        # "received no packets"). Tulis frame ke disk, encode JPG->H264 langsung.
        frames_dir = processor.output_dir / f"frames_{task_id}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        try:
            for i, frame in enumerate(output_frames):
                if not cv2.imwrite(str(frames_dir / f"f_{i:06d}.jpg"), frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise RuntimeError(f"gagal tulis frame {i}")
            _encode_h264_frames(frames_dir, target_fps, output_video_path)
            vw, vh = _verify_playable(output_video_path)
            logger.info("Video %s playable: %dx%d", output_video_path.name, vw, vh)
        except Exception:
            # Jangan hapus frame kalau gagal - biar bisa diperiksa di server
            logger.error("Encode video gagal. Frame JPG disimpan di %s", frames_dir)
            raise
        else:
            shutil.rmtree(frames_dir, ignore_errors=True)
        
        # Prepare result
        processing_time_ms = int((time.time() - start_time) * 1000)
        video_url = f"/uploads/videos/{output_video_path.name}"
        video_codec = _video_codec(output_video_path)
        n = max(1, len(frame_pcts))
        class_keys = sorted({k for p in frame_pcts for k in p})
        avg_pct = {k: sum(p.get(k, 0.0) for p in frame_pcts) / n for k in class_keys}
        h, w = output_frames[0].shape[:2]
        unified = build_unified_metrics(avg_pct, (h, w), processing_time_ms)
        result_response = {
            "status": "completed",
            "task_id": task_id,
            "filename": video_path.name,
            "video_url": video_url,
            "video_overlay_url": video_url,
            "video_codec": video_codec,
            "video_playable_in_browser": _is_h264(video_codec),
            "total_frames": len(output_frames),
            "frames_processed": len(output_frames),
            "video_info": video_tasks[task_id]["video_info"],
            "created_at": video_tasks[task_id]["created_at"],
            "processing_time_ms": processing_time_ms,
            **unified,
        }
        json_dir = uploads_dir / "results"
        json_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_dir / f"result_video_{task_id}.json"
        with open(json_path, "w") as f:
            json.dump(result_response, f, indent=2)
        logger.info("📄 Video results saved: %s", json_path)
        with video_tasks_lock:
            video_tasks[task_id]["status"] = "completed"
            video_tasks[task_id]["result"] = result_response
            video_tasks[task_id]["finished_at"] = time.time()
        
        logger.info("✅ Video task %s completed: %s frames in %dms", 
                    task_id, len(output_frames), processing_time_ms)
        
        # Post to backend if photo_id provided (sync worker: jalankan coroutine via asyncio.run)
        if photo_id:
            import asyncio
            asyncio.run(_post_result_to_backend(photo_id, result_response))
    
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
    return result
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
