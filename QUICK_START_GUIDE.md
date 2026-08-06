# 🚀 UVIP AI - Quick Start Guide

## ✅ Status: READY TO TRAIN

Semua kode sudah lengkap! Anda hanya perlu:
1. Isi data survey (labels.csv)
2. Jalankan training
3. Deploy API

---

## 📋 Step-by-Step Workflow

### Step 1: Isi Survey Labels (1-2 hari)

```bash
# Template sudah ada dengan 431 baris
cp data/templates/labels_template.csv data/training/labels.csv
```

**Edit `data/training/labels.csv`** dengan data survey kuesioner:
```csv
filename,label_beauty,label_safety,label_comfort,label_uvi
KAYUTANGAN_ST-01.jpg,7.2,6.8,7.5,6.9
KAYUTANGAN_ST-02.jpg,6.5,7.1,6.2,7.0
...
```

**Tips**:
- Gunakan skala 1-10 untuk setiap dimensi
- Kumpulkan dari 30-50 responden per foto
- Bisa pakai Google Forms atau spreadsheet

---

### Step 2: Jalankan Training (2-8 jam)

**Option A: One-Command (Recommended)**
```bash
python scripts/quick_start.py
```

**Option B: Step-by-Step**
```bash
# 1. Extract features (paling lama: 2-8 jam)
python scripts/extract_features.py \
  --input data/extracted/photos \
  --output data/training/features.csv

# 2. Merge dengan labels (< 1 menit)
python scripts/merge_labels.py \
  --features data/training/features.csv \
  --labels data/training/labels.csv \
  --output data/training/dataset.csv

# 3. Validasi dataset (< 1 menit)
python scripts/validate_dataset.py \
  --input data/training/dataset.csv

# 4. Train models (5-10 menit)
python scripts/train_xgboost.py \
  --input data/training/dataset.csv \
  --output models/perception/ \
  --n-folds 5

# 5. Test inference (< 1 menit)
python scripts/predict.py \
  --image data/extracted/photos/KAYUTANGAN/KAYUTANGAN_ST-01.jpg \
  --models models/perception/
```

---

### Step 3: Deploy API (< 5 menit)

**Local Development**
```bash
uvicorn src.uvip_ai.api.main:app --host 0.0.0.0 --port 8001
```

**Docker**
```bash
docker-compose up -d
```

**Test API**
```bash
# Health check
curl http://localhost:8001/health

# Predict
curl -X POST http://localhost:8001/predict \
  -F "file=@data/extracted/photos/KAYUTANGAN/KAYUTANGAN_ST-01.jpg"
```

---

## 💻 Hardware Options

### Local (Your Laptop - RTX 3060 6GB)
✅ **Pros**: Free, already set up
⚠️ **Cons**: Feature extraction lambat (8 jam)

```bash
# Use batch size 1 to avoid OOM
python scripts/extract_features.py --batch-size 1
```

### Cloud (Recommended - RunPod RTX 4090)
✅ **Pros**: Fast (2 jam), cheap ($2-3)
✅ **Cons**: Need to upload data

```bash
# 1. Rent RunPod GPU
# 2. Upload code & data
# 3. Run extraction
python scripts/extract_features.py --input data/extracted/photos
# 4. Download results
```

**See**: `docs/CLOUD_DEPLOYMENT.md` for detailed instructions

---

## 📊 Expected Results

### Training Output
```
models/perception/
├── beauty_xgb.pkl      # Beauty prediction model
├── safety_xgb.pkl      # Safety prediction model
├── comfort_xgb.pkl     # Comfort prediction model
├── uvi_xgb.pkl         # UVI prediction model
└── metrics.json        # R², MAE, RMSE scores
```

### Target Performance
- **R² ≥ 0.70** untuk semua 4 model
- Jika < 0.70: kumpulkan lebih banyak data survey

### API Response Example
```json
{
  "beauty_score": 7.2,
  "safety_score": 6.8,
  "comfort_score": 7.5,
  "uvi_score": 6.9,
  "segmentation_metrics": {
    "green_coverage_pct": 35.2,
    "building_coverage_pct": 28.1,
    "walkability_ratio": 0.42,
    "visual_clutter_index": 0.18,
    "sky_visibility_pct": 22.5
  },
  "shap_values": {
    "green_coverage_pct": 0.85,
    "building_coverage_pct": -0.32
  }
}
```

---

## 📖 Documentation

| File | Purpose |
|------|---------|
| **README.md** | Complete project guide |
| **PROJECT_SUMMARY.md** | Quick overview |
| **docs/STATUS.md** | Detailed status & checklist |
| **docs/CLOUD_DEPLOYMENT.md** | Cloud GPU guide |
| **docs/DEPLOYMENT_AND_SERVERS.md** | Server recommendations |

---

## 🐛 Common Issues

### CUDA Out of Memory
```bash
python scripts/extract_features.py --batch-size 1
```

### Slow Training
```bash
# Use cloud GPU (RunPod)
# See docs/CLOUD_DEPLOYMENT.md
```

### Low R² (< 0.70)
```bash
# 1. Collect more survey data
# 2. Tune hyperparameters
python scripts/train_xgboost.py --n-estimators 300 --max-depth 8
```

---

## 💰 Cost Estimate

| Scenario | Cost | Time |
|----------|------|------|
| **Local training** | $0 | 8 hours |
| **Cloud training (RunPod)** | $2-3 | 2 hours |
| **Production (monthly)** | $300-450 | Always-on |

---

## ✅ Final Checklist

- [ ] Fill `data/training/labels.csv` with survey data
- [ ] Run `python scripts/quick_start.py`
- [ ] Verify R² ≥ 0.70 in `models/perception/metrics.json`
- [ ] Test API: `curl http://localhost:8001/health`
- [ ] Deploy to production (optional)

---

## 🎉 You're Ready!

**Next action**:
```bash
# 1. Copy template
cp data/templates/labels_template.csv data/training/labels.csv

# 2. Edit labels.csv with your survey data

# 3. Run training
python scripts/quick_start.py
```

**Good luck! 🚀**
