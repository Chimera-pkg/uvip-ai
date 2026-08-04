#!/usr/bin/env python
"""
Ekstrak foto + metadata dari PDF dataset UVIP ("DATA LONG LAT DAN TABEL FOTO.pdf").

Struktur PDF (Canva): tiap halaman data berisi grid 6 foto street-level dengan
label titik (ST/SB/UT/UB/...) + koordinat lat/long. Ada 3 area:
Kayutangan, Alun-Alun Tugu, Alun-Alun Merdeka.

Output:
  data/extracted/photos/<AREA>/<AREA>_<LABEL>.jpg   (mis. KAYUTANGAN_ST-01.jpg)
  data/extracted/manifest.csv
  data/extracted/extraction_report.txt

Pemetaan: gambar ke-N pada halaman dipadankan dengan label & koordinat ke-N
(urutan kemunculan di teks halaman). Halaman dengan jumlah gambar != jumlah
label ditandai di report untuk pengecekan manual.
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import pypdf
from tqdm import tqdm

PDF_PATH = Path("DATA LONG LAT DAN TABEL FOTO.pdf")
OUTPUT_DIR = Path("data/extracted")
PHOTOS_DIR = OUTPUT_DIR / "photos"


def detect_area(page_text: str) -> str:
    txt = re.sub(r"\s+", "", page_text).upper()
    if "ALUN-ALUNTUGU" in txt:
        return "ALUN_ALUN_TUGU"
    if "ALUN-ALUNMERDEKA" in txt:
        return "ALUN_ALUN_MERDEKA"
    return "KAYUTANGAN"


def parse_labels_and_coords(page_text: str):
    """Kembalikan (labels, coords) sesuai urutan kemunculan di teks.

    labels: list of (type:str, num:int)  mis. ('ST', 1)
    coords: list of (lat:float, long:float)
    """
    compact = re.sub(r"\s+", "", page_text)
    # Include all 2-letter prefixes (ST, SB, UT, UB, TS, TU, BU, BS, TB, TT, etc.)
    labels = [(m.group(1), int(m.group(2))) for m in re.finditer(r"([A-Z]{2})(\d+)", compact)]
    coords = [(float(a), float(b))
              for a, b in re.findall(r"lat:(-?\d+\.\d+)long:(\d+\.\d+)", compact)]
    return labels, coords


def main() -> None:
    reader = pypdf.PdfReader(str(PDF_PATH))
    rows = []
    report_lines = []
    label_counter: Counter[str] = Counter()
    n_with_coords = 0

    for idx, page in enumerate(tqdm(reader.pages, desc="Ekstrak foto")):
        page_num = idx + 1
        try:
            images = list(page.images)
        except Exception as e:  # noqa: BLE001
            report_lines.append(f"p{page_num}: gagal baca gambar ({e})")
            continue
        if not images:
            continue

        text = page.extract_text() or ""
        area = detect_area(text)
        labels, coords = parse_labels_and_coords(text)

        # Peta halaman (PNG tunggal) atau halaman tanpa label → simpan sebagai aset
        if not labels:
            report_lines.append(
                f"p{page_num} [{area}]: {len(images)} gambar TANPA label "
                f"(kemungkinan peta/aset) → dilewati dari manifest"
            )
            continue

        if len(images) != len(labels):
            report_lines.append(
                f"p{page_num} [{area}]: MISMATCH {len(images)} gambar vs "
                f"{len(labels)} label — dipadankan sebanyak yang cocok"
            )

        n = min(len(images), len(labels))
        for i in range(n):
            image = images[i]
            ltype, lnum = labels[i]
            lat = coords[i][0] if i < len(coords) else None
            lng = coords[i][1] if i < len(coords) else None

            label = f"{ltype}-{lnum:02d}"
            ext = image.name.lower().rsplit(".", 1)[-1]
            if ext not in {"jpg", "jpeg", "png"}:
                ext = "jpg"
            filename = f"{area}_{label}.{ext}"

            out_dir = PHOTOS_DIR / area
            out_dir.mkdir(parents=True, exist_ok=True)
            # hindari overwrite bila label berulang di area yang sama
            out_file = out_dir / filename
            dup = 1
            while out_file.exists():
                dup += 1
                filename = f"{area}_{label}_{dup}.{ext}"
                out_file = out_dir / filename

            out_file.write_bytes(image.data)
            label_counter[area] += 1
            if lat is not None:
                n_with_coords += 1

            rows.append({
                "filename": filename,
                "relative_path": str(out_file.relative_to(OUTPUT_DIR)).replace("\\", "/"),
                "area": area,
                "point_id": label,
                "point_type": ltype,
                "point_number": lnum,
                "page": page_num,
                "lat": f"{lat:.8f}" if lat is not None else "",
                "long": f"{lng:.8f}" if lng is not None else "",
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "relative_path", "area", "point_id", "point_type",
            "point_number", "page", "lat", "long",
        ])
        writer.writeheader()
        writer.writerows(rows)

    report_path = OUTPUT_DIR / "extraction_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("UVIP-AI — Laporan Ekstraksi Foto dari PDF\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total foto ter-manifest : {len(rows)}\n")
        f.write(f"Foto dengan koordinat   : {n_with_coords}\n\n")
        f.write("Per area:\n")
        for area, c in label_counter.most_common():
            f.write(f"  {area:20s}: {c}\n")
        f.write("\nCatatan / anomali halaman:\n")
        for line in report_lines:
            f.write(f"  - {line}\n")

    print("\n" + "=" * 55)
    print(f"  Total foto      : {len(rows)}")
    print(f"  Dgn koordinat   : {n_with_coords}")
    for area, c in label_counter.most_common():
        print(f"    {area:20s}: {c}")
    print(f"  Manifest        : {manifest_path}")
    print(f"  Report          : {report_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
