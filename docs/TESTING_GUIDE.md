# 🧪 UVIP AI Testing Guide

Panduan lengkap untuk testing model UVIP AI pada gambar dan video.

## 📋 Overview

Ada 3 script testing yang bisa digunakan:

1. **`demo.py`** - Demo mode (tidak perlu trained models)
2. **`quick_test.py`** - Quick test dengan trained models
3. **`test_inference.py`** - Full inference testing dengan visualisasi

## 🚀 Quick Start

### 1. Demo Mode (Tanpa Trained Models)

Script ini menggunakan **mock predictions** untuk demonstrate pipeline. Cocok untuk testing awal.

```bash
# Test single image
python scripts/demo.py data/extracted/photos/KAYUTANGAN/KAYUTANGAN_ST-01.jpg

# Test video (sample 5 frames)
python scripts/demo.py test_video.mp4

# Test video (sample 10 frames)
python scripts/demo.py test_video.mp4 --frames 10
```

**Output yang dihasilkan:**
- `*_masked.jpg` - Image dengan privacy masking (blur wajah/plat)
- `*_segmentation.png` - Segmentation mask (colored)
- `*_overlay.png` - Original + mask blended
- `*_results.json` - Detailed results (metrics, predictions)

**Features yang didemo:**
- ✅ Privacy Guard (YOLOv8n) - blur wajah & plat nomor
- ✅ SegFormer-B0 - 5 urban metrics
- ✅ DINOv2-Base - 768-d embedding
- ⚠️ Mock predictions (bukan dari trained model)

### 2. Quick Test (Dengan Trained Models)

Setelah training di Kaggle, gunakan script ini untuk test dengan model yang sudah trained.

```bash
# Test single image
python scripts/quick_test.py data/extracted/photos/KAYUTANGAN/KAYUTANGAN_ST-01.jpg

# Test dengan custom models directory
python scripts/quick_test.py test_image.jpg --models models/perception

# Test video (sample 5 frames)
python scripts/quick_test.py test_video.mp4

# Test video (sample 10 frames)
python scripts/quick_test.py test_video.mp4 --video-frames 10
```

**Output yang dihasilkan:**
- `*_masked.jpg` - Image dengan privacy masking
- `*_segmentation.png` - Segmentation mask
- `*_results.json` - Results dengan real predictions

**Features yang ditest:**
- ✅ Privacy Guard (YOLOv8n)
- ✅ SegFormer-B0
- ✅ DINOv2-Base
- ✅ XGBoost models (beauty, safety, comfort, uvi)

### 3. Full Inference Testing

Script ini menyediakan visualisasi lengkap dengan matplotlib.

```bash
# Test single image dengan visualisasi
python scripts/test_inference.py --input test_image.jpg --visualize

# Test video
python scripts/test_inference.py --input test_video.mp4 --video-frames 30

# Custom output directory
python scripts/test_inference.py --input test_image.jpg --output my_results
```

**Output yang dihasilkan:**
- Semua output dari quick_test
- `*_visualization.png` - Visualisasi lengkap dengan matplotlib

## 📊 Understanding the Results

### Urban Metrics (dari SegFormer)

| Metric | Deskripsi | Range |
|--------|-----------|-------|
| `green_coverage_pct` | Persentase area hijau (vegetasi + sky) | 0-100% |
| `building_coverage_pct` | Persentase area bangunan | 0-100% |
| `walkability_ratio` | Rasio trotoar terhadap jalan | 0-1 |
| `visual_clutter_index` | Indeks kerumitan visual | 0-1 |
| `sky_visibility_pct` | Persentase langit terlihat | 0-100% |

### Perception Scores (dari XGBoost)

| Score | Deskripsi | Range |
|-------|-----------|-------|
| `beauty` | Skor keindahan visual | 0-10 |
| `safety` | Skor keamanan | 0-10 |
| `comfort` | Skor kenyamanan | 0-10 |
| `uvi` | Urban Visual Index (composite) | 0-10 |

## 🎬 Testing Video

Untuk testing video, script akan:
1. Sample N frames dari video
2. Process setiap frame secara independen
3. Calculate average metrics & predictions
4. Generate timeline results

**Tips:**
- Gunakan `--frames 5` untuk quick test
- Gunakan `--frames 30` untuk detailed analysis
- Video panjang (>1 menit) sebaiknya di-sample lebih banyak

## 🔍 Troubleshooting

### Error: "Cannot load image"
```bash
# Pastikan path file benar
ls data/extracted/photos/KAYUTANGAN/

# Gunakan absolute path
python scripts/demo.py /full/path/to/image.jpg
```

### Error: "Models directory not found"
```bash
# Download trained models dari Kaggle
# Upload ke VPS atau local directory
ls models/perception/
# Harus ada: beauty_xgb.pkl, safety_xgb.pkl, comfort_xgb.pkl, uvi_xgb.pkl
```

### Error: "CUDA out of memory"
```bash
# Gunakan low VRAM mode (sudah default)
python scripts/demo.py test_image.jpg

# Atau gunakan CPU mode (lebih lambat)
# Edit .env: UVIP_DEVICE=cpu
```

### Error: "Module not found"
```bash
# Pastikan dependencies terinstall
pip install -r requirements.txt

# Pastikan PYTHONPATH sudah di-set
export PYTHONPATH=/path/to/uvip-ai/src
```

## 📁 Output Structure

```
demo_outputs/
├── KAYUTANGAN_ST-01_masked.jpg         # Privacy masked
├── KAYUTANGAN_ST-01_segmentation.png   # Segmentation mask
├── KAYUTANGAN_ST-01_overlay.png        # Overlay visualization
├── KAYUTANGAN_ST-01_results.json       # Detailed results
└── video_summary.json                  # Video summary (jika test video)
```

## 🎯 Example Workflow

### 1. Test Pipeline (Demo Mode)
```bash
# Test dengan 1 foto sample
python scripts/demo.py data/extracted/photos/KAYUTANGAN/KAYUTANGAN_ST-01.jpg

# Lihat hasil
ls demo_outputs/
cat demo_outputs/KAYUTANGAN_ST-01_results.json
```

### 2. Test dengan Trained Models
```bash
# Download models dari Kaggle
# Upload ke models/perception/

# Test dengan real predictions
python scripts/quick_test.py data/extracted/photos/KAYUTANGAN/KAYUTANGAN_ST-01.jpg

# Compare results
cat test_outputs/KAYUTANGAN_ST-01_results.json
```

### 3. Batch Testing
```bash
# Test semua foto di satu area
for img in data/extracted/photos/KAYUTANGAN/*.jpg; do
    python scripts/demo.py "$img"
done

# Test video dengan multiple samples
python scripts/demo.py test_video.mp4 --frames 20
```

## 📊 Interpreting Results

### Good Urban Space (High Scores)
- 🌳 Green Coverage: >30%
- 🚶 Walkability: >0.4
- ☁️ Sky Visibility: >20%
- 💎 Beauty: >7.0
- 🛡️ Safety: >7.0
- 😌 Comfort: >7.0

### Poor Urban Space (Low Scores)
- 🏢 Building Coverage: >60%
- 📊 Visual Clutter: >0.5
- 💎 Beauty: <5.0
- 🛡️ Safety: <5.0
- 😌 Comfort: <5.0

## 🔗 Related Documentation

- [Kaggle Training Guide](KAGGLE_TRAINING.md) - Cara training di Kaggle
- [Deployment Guide](DEPLOYMENT.md) - Cara deploy ke production
- [Model Architecture](MODEL_ARCHITECTURE.md) - Detail arsitektur model

## 💡 Tips

1. **Start with demo mode** - Test pipeline dulu tanpa trained models
2. **Use small images** - Untuk quick testing, gunakan image <1024x1024
3. **Check GPU memory** - Monitor dengan `nvidia-smi` saat testing
4. **Save results** - Selalu save results untuk analisis lebih lanjut
5. **Compare metrics** - Bandingkan metrics antar lokasi untuk insight

## 🎉 Next Steps

1. ✅ Test pipeline dengan `demo.py`
2. ✅ Train models di Kaggle
3. ✅ Download trained models
4. ✅ Test dengan `quick_test.py`
5. ✅ Deploy ke VPS
6. ✅ Test API endpoint

---

**Need help?** Check [README.md](../README.md) atau buka issue di GitHub.
