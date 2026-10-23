"""Penyimpanan artefak FLAT di ``uploads/`` — kompatibel dengan nama lama.

Nama file mengikuti skema lama supaya URL yang sudah tersimpan di DB/frontend
tetap cocok::

    uploads/originals/<nama_upload>_<uuid4>.jpg
    uploads/segmentation/seg_<nama_upload>_<uuid4>_<epoch_ms>.jpg
    uploads/segmentation/overlay_<nama_upload>_<uuid4>_<epoch_ms>.jpg
    uploads/masks/mask_<nama_upload>_<uuid4>_<epoch_ms>.jpg
    uploads/results/result_<nama_upload>_<uuid4>_<epoch_ms>.json

Video::

    uploads/originals/<nama_upload>_<uuid4>.mp4
    uploads/videos/video_<task_id>.mp4
    uploads/results/result_video_<task_id>.json

Hasil ``/llm/generate``::

    uploads/generated/img_<nama_upload>_<epoch_ms>.png

Semua artefak satu foto memakai ``<epoch_ms>`` yang sama, jadi pasangan
seg/overlay/mask/result bisa dipetakan hanya dari nama file.

Klien cukup mengirim ``result_url``/``result_path`` atau path artefak apa pun
(``/uploads/segmentation/overlay_...jpg``). :func:`resolve_result_path`
memetakan artefak ke ``result_*.json`` pasangannya, jadi payload tidak perlu
memuat JSON hasil yang besar.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from glob import escape as glob_escape
from pathlib import Path
from typing import Any, BinaryIO, Optional
from urllib.parse import unquote, urlparse

from uvip_ai.config import PROJECT_ROOT

UPLOADS_DIR = (PROJECT_ROOT / "uploads").resolve()
ORIGINALS_DIR = UPLOADS_DIR / "originals"
MASKS_DIR = UPLOADS_DIR / "masks"
SEGMENTATION_DIR = UPLOADS_DIR / "segmentation"
RESULTS_DIR = UPLOADS_DIR / "results"
VIDEOS_DIR = UPLOADS_DIR / "videos"
GENERATED_DIR = UPLOADS_DIR / "generated"
TEMP_DIR = UPLOADS_DIR / "temp"

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
_MIME_SUFFIXES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/gif": ".gif",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/x-matroska": ".mkv",
    "video/webm": ".webm",
}

# <prefix>_<stem>_<epoch_ms>[_N].<ext>; timestamp lama tetap didukung.
_TS = r"(?:\d{8}_\d{6}|\d{10,13})"
_ARTIFACT_RE = re.compile(
    r"^(?P<prefix>seg|overlay|mask|img|video|result_video|result)_"
    rf"(?P<stem>.+?)_(?P<ts>{_TS})(?P<n>_\d+)?"
    r"\.(?P<ext>jpe?g|png|webp|mp4|mov|webm|json)$",
    re.IGNORECASE,
)
_VIDEO_LEGACY_RE = re.compile(
    r"^video_(?P<task>[^.]+)\.(?:mp4|mov|avi|mkv|webm|m4v)$",
    re.IGNORECASE,
)

def timestamp_ms() -> str:
    """Timestamp nama file lama: epoch milidetik server."""
    return str(int(time.time() * 1000))

def slug_title(title: Optional[str], fallback: str = "photo") -> str:
    """Judul -> bagian nama file yang aman (tanpa spasi/simbol aneh)."""
    raw = (title or "").strip() or (fallback or "").strip() or "photo"
    raw = raw.replace("\\", "/").rsplit("/", 1)[-1]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_-")
    # Jangan potong UUID (36 char): batas 120 masih aman untuk nama file
    # setelah suffix uuid4 + epoch ditambahkan.
    return (stem or "photo")[:120]

def guess_suffix(filename: Optional[str], content_type: Optional[str],
                 kind: str) -> str:
    """Suffix dari nama file; fallback ke MIME lalu default per jenis."""
    suffix = Path(filename or "").suffix.lower()
    allowed = _IMAGE_SUFFIXES if kind == "photo" else _VIDEO_SUFFIXES
    if suffix in allowed:
        return suffix
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    by_mime = _MIME_SUFFIXES.get(mime)
    if by_mime and by_mime in allowed:
        return by_mime
    return ".jpg" if kind == "photo" else ".mp4"

def unique_path(path: Path) -> Path:
    """``path`` bila belum ada; selain itu ``_2``, ``_3``, ..."""
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"tidak menemukan nama unik untuk {path}")

def output_path(directory: Path, prefix: str, stem: Optional[str],
                suffix: str, stamp: Optional[str] = None) -> Path:
    """Path artefak baru: ``<dir>/<prefix><stem>_<epoch_ms><suffix>``."""
    name = f"{prefix}{slug_title(stem)}_{stamp or timestamp_ms()}{suffix}"
    directory.mkdir(parents=True, exist_ok=True)
    return unique_path(directory / name)

def save_upload(stream: BinaryIO, subdir: str, title: Optional[str],
                suffix: str) -> Path:
    """Stream upload ke ``uploads/<subdir>/<nama_upload>_<uuid4><suffix>``.

    UUID dipertahankan seperti skema lama supaya nama artefak turunan
    (``seg_<judul>_<uuid4>_<epoch>.jpg``) tetap kompatibel dengan URL yang
    sudah tersimpan di DB.
    """
    directory = UPLOADS_DIR / subdir
    directory.mkdir(parents=True, exist_ok=True)
    path = unique_path(directory / f"{slug_title(title)}_{uuid.uuid4()}{suffix}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
    with open(tmp, "wb") as out:
        shutil.copyfileobj(stream, out, 1024 * 1024)
    os.replace(tmp, path)
    return path

def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)

def rel_path(path: Path) -> str:
    """Path relatif repo (POSIX), mis. ``uploads/results/result_x.json``."""
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(UPLOADS_DIR)
    except ValueError:
        try:
            rel = resolved.relative_to(PROJECT_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"path di luar repo: {path}") from exc
    return f"uploads/{rel.as_posix()}" if rel.parts and rel.parts[0] != "uploads" else rel.as_posix()

def upload_url(path: Optional[Path]) -> Optional[str]:
    """URL publik ``/uploads/...``; None untuk file di luar uploads/."""
    if path is None:
        return None
    try:
        rel = path.resolve().relative_to(UPLOADS_DIR)
    except ValueError:
        return None
    return f"/uploads/{rel.as_posix()}"

def resolve_upload_path(ref: str) -> Path:
    """Referensi klien (URL/path relatif/absolut) -> path absolut di uploads/."""
    raw = (ref or "").strip()
    if not raw:
        raise ValueError("referensi kosong")
    if raw.startswith(("http://", "https://")):
        raw = unquote(urlparse(raw).path)
    if raw.startswith("/uploads/"):
        candidate = UPLOADS_DIR / raw[len("/uploads/"):]
    elif raw == "/uploads":
        candidate = UPLOADS_DIR
    elif raw.startswith("uploads/") or raw.startswith("uploads\\"):
        candidate = PROJECT_ROOT / raw
    else:
        path = Path(raw)
        candidate = path if path.is_absolute() else PROJECT_ROOT / path

    resolved = candidate.resolve()
    if resolved != UPLOADS_DIR and UPLOADS_DIR not in resolved.parents:
        raise PermissionError(f"referensi di luar uploads/: {ref}")
    return resolved

def _newest_result(stem: str, *, video: bool) -> Optional[Path]:
    safe_stem = glob_escape(stem)
    pattern = (f"result_video_{safe_stem}_*.json" if video
               else f"result_{safe_stem}_*.json")
    candidates = [p for p in RESULTS_DIR.glob(pattern) if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def artifact_result_path(path: Path) -> Optional[Path]:
    """Petakan artefak (seg/overlay/mask/video/img/original) ke result JSON."""
    name = path.name
    match = _ARTIFACT_RE.match(name)
    if match:
        prefix = match.group("prefix").lower()
        stem, ts, n = (match.group("stem"), match.group("ts"),
                       match.group("n") or "")
        if prefix == "result":
            return path if path.is_file() else None
        if prefix in ("result_video", "video"):
            candidate = RESULTS_DIR / f"result_video_{stem}_{ts}{n}.json"
            return candidate if candidate.is_file() else _newest_result(stem, video=True)
        candidate = RESULTS_DIR / f"result_{stem}_{ts}{n}.json"
        if candidate.is_file():
            return candidate
        # Proses lama menimpa result saat nama artefak bentrok dalam detik
        # yang sama, jadi ``_2`` sering hanya ada di artefak, bukan result.
        base = RESULTS_DIR / f"result_{stem}_{ts}.json"
        if base.is_file():
            return base
        return _newest_result(stem, video=False)

    # Video lama: video_<task_id>.mp4 -> result_video_<task_id>.json.
    legacy_video = _VIDEO_LEGACY_RE.match(name)
    if legacy_video:
        task = legacy_video.group("task")
        exact = RESULTS_DIR / f"result_video_{task}.json"
        return exact if exact.is_file() else _newest_result(task, video=True)

    # Original / nama lama tanpa prefix: cari result dengan stem yang sama.
    stem = path.stem
    for candidate in (RESULTS_DIR / f"result_{stem}.json",
                      RESULTS_DIR / f"result_video_{stem}.json"):
        if candidate.is_file():
            return candidate
    return (_newest_result(stem, video=False)
            or _newest_result(stem, video=True))

def resolve_result_path(ref: str) -> tuple[Path, str]:
    """Resolusi referensi -> ``(result_*.json, asal)``.

    Menerima ``result_url``/``result_path`` dari respons /ai/process, path
    artefak (mis. ``/uploads/segmentation/overlay_...jpg``), path JSON lain,
    atau file hasil video. Artefak yang sudah terhapus tetap bisa dipetakan
    ke result JSON pasangannya lewat nama file.
    """
    raw = (ref or "").strip()
    if not raw:
        raise ValueError("referensi kosong")
    if raw.startswith("{"):
        raise ValueError("JSON inline bukan path")

    path = resolve_upload_path(raw)
    if path.is_dir():
        raise ValueError(f"referensi berupa folder, bukan file: {raw}")
    if path.suffix.lower() == ".json":
        if not path.is_file():
            raise FileNotFoundError(f"file result tidak ditemukan: {raw}")
        return path, "path"

    result = artifact_result_path(path)
    if result is not None and result.is_file():
        return result, f"artifact:{path.name}"
    if not path.is_file():
        raise FileNotFoundError(
            f"file tidak ditemukan: {raw} dan tidak ada result JSON pasangannya")
    raise FileNotFoundError(
        f"tidak ada result JSON untuk artefak {path.name}; kirim "
        "result_url/result_path dari respons /ai/process atau file result_*.json"
    )
