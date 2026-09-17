#!/usr/bin/env python3
"""
Quick start script untuk UVIP AI training pipeline.
Jalankan script ini untuk memulai proses training dari awal.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")

    result = subprocess.run(cmd, shell=True, capture_output=False)

    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed!")
        print(f"Exit code: {result.returncode}")
        sys.exit(1)

    print(f"\n✅ {description} completed successfully!")

def main():
    """Main quick start workflow."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║           🌆 UVIP AI - Quick Start Training              ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)

    # Check if labels.csv exists
    labels_path = Path("data/training/labels.csv")
    if not labels_path.exists():
        print("⚠️  labels.csv tidak ditemukan!")
        print("\n📋 Langkah pertama:")
        print("1. Copy template:")
        print("   cp data/templates/labels_template.csv data/training/labels.csv")
        print("\n2. Isi labels.csv dengan data survey kuesioner")
        print("   Format: filename,label_beauty,label_safety,label_comfort,label_uvi")
        print("   Contoh: KAYUTANGAN_ST-01.jpg,7.2,6.8,7.5,6.9")
        print("\n3. Jalankan kembali script ini setelah labels.csv siap")
        sys.exit(1)

    # Step 1: Extract features
    run_command(
        "python scripts/extract_features.py --input data/extracted/photos --output data/training/features.csv",
        "Step 1: Extract features dari foto (ini akan memakan waktu)"
    )

    # Step 2: Merge labels
    run_command(
        "python scripts/merge_labels.py --features data/training/features.csv --labels data/training/labels.csv --output data/training/dataset.csv",
        "Step 2: Merge features dengan labels"
    )

    # Step 3: Validate dataset
    run_command(
        "python scripts/validate_dataset.py --input data/training/dataset.csv",
        "Step 3: Validasi dataset"
    )

    # Step 4: Train models
    run_command(
        "python scripts/train_xgboost.py --input data/training/dataset.csv --output models/perception/ --n-folds 5",
        "Step 4: Train XGBoost models (4 target: Beauty, Safety, Comfort, UVI)"
    )

    # Step 5: Test inference
    test_image = "data/extracted/photos/KAYUTANGAN/KAYUTANGAN_ST-01.jpg"
    if Path(test_image).exists():
        run_command(
            f"python scripts/predict.py --image {test_image} --models models/perception/",
            "Step 5: Test inference dengan sample image"
        )

    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║                    🎉 Training Selesai!                   ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

✅ Models tersimpan di: models/perception/
   - beauty_xgb.pkl
   - safety_xgb.pkl
   - comfort_xgb.pkl
   - uvi_xgb.pkl

📊 Metrics: models/perception/metrics.json

🚀 Next steps:
   1. Deploy API: uvicorn src.uvip_ai.api.main:app --host 0.0.0.0 --port 8001
   2. Test endpoint: curl http://localhost:8001/health
   3. Predict: curl -X POST http://localhost:8001/predict -F "file=@photo.jpg"

📖 Dokumentasi lengkap: README.md
    """)

if __name__ == "__main__":
    main()
