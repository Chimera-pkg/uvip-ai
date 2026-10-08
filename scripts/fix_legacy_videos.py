#!/usr/bin/env python3
"""Transcode video lama yang codec-nya bukan H.264 (mis. mpeg4) jadi H.264.

Latar belakang
--------------
Sebelum build `jpg-h264-v14`, jalur encode punya fallback mpeg4 (MPEG-4 Part 2).
File itu bisa dibuka VLC/OpenCV tapi DITOLAK Chrome/Firefox `<video>`
(MEDIA_ERR_SRC_NOT_SUPPORTED). Akibatnya ada file lama di `uploads/videos/`
yang "selesai" tapi tidak bisa diputar di web.

Script ini memindai folder video, mendeteksi codec yang bukan H.264, lalu
menulis ulang dengan H.264 yang sama seperti jalur produksi:
    libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart

Pakai
-----
    # lihat dulu apa yang akan diubah (default, tidak menulis apa pun)
    python scripts/fix_legacy_videos.py

    # eksekusi
    python scripts/fix_legacy_videos.py --apply

    # folder lain
    python scripts/fix_legacy_videos.py --dir /path/ke/uploads/videos --apply

File asli disimpan sebagai `<nama>.mp4.bak` (tidak dihapus), dan hasil
transcode DIVERIFIKASI ulang sebelum menggantikan file asli.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Codec yang dianggap aman untuk Chrome/Firefox <video>
H264_TAGS = ("h264", "avc1", "x264", "264")
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm")


def _ffmpeg_candidates() -> list[str]:
    """Binary ffmpeg yang mungkin dipakai, urut prioritas (distro dulu)."""
    cands: list[str] = []
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        cands.append(sys_ff)
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        exe = get_ffmpeg_exe()
        if exe and Path(exe).exists() and exe not in cands:
            cands.append(exe)
    except Exception:
        pass
    return cands


def _codec_of(exe: str, path: Path) -> str:
    """Baca codec video lewat ffprobe (fallback: OpenCV)."""
    ffprobe = Path(exe).with_name("ffprobe")
    if not ffprobe.exists():
        ffprobe = shutil.which("ffprobe") or ""
    if ffprobe:
        try:
            r = subprocess.run(
                [str(ffprobe), "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name", "-of",
                 "default=nw=1:nk=1", str(path)],
                capture_output=True, timeout=60)
            if r.returncode == 0:
                return r.stdout.decode(errors="ignore").strip()
        except Exception:
            pass
    # Fallback: OpenCV (fourcc bisa beda-beda antar platform, cukup untuk
    # membedakan h264 vs mpeg4).
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                return ""
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            return "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")
        finally:
            cap.release()
    except Exception:
        return ""


def _is_h264(codec: str) -> bool:
    c = (codec or "").lower()
    return any(tag in c for tag in H264_TAGS)


def _transcode(exe: str, src: Path, dst: Path) -> None:
    """Encode ulang src -> dst sebagai H.264 MP4 yang playable di browser."""
    cmd = [
        exe, "-y", "-loglevel", "error", "-hide_banner",
        "-i", str(src),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p,scale=out_range=tv",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-color_range", "tv", "-colorspace", "bt709",
        "-color_primaries", "bt709", "-color_trc", "bt709",
        "-c:a", "aac", "-b:a", "128k",       # kalau ada audio
        "-movflags", "+faststart",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(
            f"ffmpeg gagal ({exe}): "
            f"{r.stderr.decode(errors='ignore').strip()[:600]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(ROOT / "uploads" / "videos"),
                    help="folder video (default: uploads/videos)")
    ap.add_argument("--apply", action="store_true",
                    help="tulis perubahan (tanpa ini hanya dry-run)")
    ap.add_argument("--keep-backup", action="store_true", default=True,
                    help="simpan file asli sebagai .bak (default: ya)")
    args = ap.parse_args()

    folder = Path(args.dir)
    if not folder.is_dir():
        print(f"[X] folder tidak ada: {folder}")
        return 1

    exes = _ffmpeg_candidates()
    if not exes:
        print("[X] ffmpeg tidak ditemukan. Install: apt-get install -y ffmpeg")
        return 1
    exe = exes[0]
    print(f"==> ffmpeg: {exe}")
    print(f"==> folder: {folder}")
    print(f"==> mode  : {'APPLY' if args.apply else 'DRY-RUN (pakai --apply untuk eksekusi)'}")
    print()

    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in VIDEO_EXTS
                   and not p.name.endswith(".bak"))
    if not files:
        print("Tidak ada file video di folder ini.")
        return 0

    fixed = skipped = failed = 0
    for src in files:
        codec = _codec_of(exe, src)
        if _is_h264(codec):
            print(f"  [OK]   {src.name}  (codec={codec or '?'})")
            skipped += 1
            continue

        size_mb = src.stat().st_size / 1e6
        print(f"  [FIX]  {src.name}  (codec={codec or '?'}, {size_mb:.1f} MB)")
        if not args.apply:
            fixed += 1
            continue

        # Nama sementara HARUS berakhiran .mp4: ffmpeg memilih muxer dari
        # ekstensi file, dan ".transcoding" membuatnya gagal dengan
        # "Unable to choose an output format".
        tmp = src.with_name(src.stem + ".transcoding.tmp.mp4")
        try:
            _transcode(exe, src, tmp)
            new_codec = _codec_of(exe, tmp)
            if not _is_h264(new_codec) or tmp.stat().st_size == 0:
                raise RuntimeError(
                    f"hasil transcode bukan H.264 (codec={new_codec!r})")
            backup = src.with_suffix(src.suffix + ".bak")
            if args.keep_backup:
                shutil.copy2(src, backup)
            shutil.move(str(tmp), str(src))
            print(f"         -> OK codec={new_codec}, "
                  f"{src.stat().st_size / 1e6:.1f} MB"
                  + (f", backup: {backup.name}" if args.keep_backup else ""))
            fixed += 1
        except Exception as exc:
            print(f"         -> GAGAL: {exc}")
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            failed += 1

    print()
    print(f"Ringkas: {fixed} {'akan diubah' if not args.apply else 'diubah'}, "
          f"{skipped} sudah H.264, {failed} gagal.")
    if not args.apply and fixed:
        print("Jalankan ulang dengan --apply untuk eksekusi.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
