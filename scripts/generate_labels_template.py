#!/usr/bin/env python
"""
Generate labels template dari manifest.csv.
Output: data/training/labels_template.csv yang siap diisi.

Cara pakai:
    python scripts/generate_labels_template.py

Setelah dijalankan, edit file labels_template.csv dan isi kolom label_* dengan skor 1-10.
"""
from pathlib import Path
import csv

def main():
    manifest_path = Path("data/extracted/manifest.csv")
    output_path = Path("data/training/labels_template.csv")

    if not manifest_path.exists():
        print(f"❌ File tidak ditemukan: {manifest_path}")
        print("   Pastikan sudah run: python scripts/extract_photos_from_pdf.py")
        return

    # Read manifest
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"✓ Found {len(rows)} photos in manifest")

    # Generate template
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "label_beauty", "label_safety", "label_comfort", "label_uvi"
        ])
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "filename": row["filename"],
                "label_beauty": "",  # Kosong, tinggal diisi
                "label_safety": "",
                "label_comfort": "",
                "label_uvi": "",
            })

    print(f"✓ Template generated: {output_path}")
    print(f"\n📝 Langkah selanjutnya:")
    print(f"   1. Buka file: {output_path}")
    print(f"   2. Isi kolom label_beauty, label_safety, label_comfort, label_uvi dengan skor 1-10")
    print(f"   3. Upload ke Kaggle sebagai input dataset")
    print(f"\n💡 Tips:")
    print(f"   - Bisa pakai Excel/Google Sheets untuk edit lebih cepat")
    print(f"   - Isi skor berdasarkan penilaian subjektif atau data survey")
    print(f"   - Kalau tidak ada data survey, bisa pakai estimasi visual")


if __name__ == "__main__":
    main()
