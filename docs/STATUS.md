# UVIP AI - Status & Checklist

## ✅ Completed Components

### 1. Environment Setup
- [x] Virtual environment configured
- [x] PyTorch with CUDA installed
- [x] All dependencies in requirements.txt
- [x] GPU verification script (verify_gpu.py)
- [x] Configuration system (config.py)

### 2. Data Extraction
- [x] PDF photo extraction script (extract_photos_from_pdf.py)
- [x] Coordinate fixing script (fix_manifest_coords.py)
- [x] 434 photos extracted to data/extracted/photos/
- [x] Manifest CSV generated with GPS coordinates

### 3. AI Models
- [x] **Privacy Guard** (YOLOv8n) - src/uvip_ai/privacy/guard.py
  - Face & license plate detection
  - Gaussian blur masking
  - Batch processing support
  
- [x] **Segmentation** (SegFormer-B5) - src/uvip_ai/segmentation/segformer.py
  - 5 urban metrics extraction
  - Cityscapes pretrained model
  - GPU-optimized inference
  
- [x] **Feature Extraction** (DINOv2-Large) - src/uvip_ai/features/dinov2.py
  - 1024-d embedding generation
  - Batch processing
  - Memory-efficient loading
  
- [x] **Perception Prediction** (XGBoost) - src/uvip_ai/training/xgboost_model.py
  - 4 separate models (Beauty, Safety, Comfort, UVI)
  - K-Fold cross-validation
  - R² score evaluation
  - Model persistence (pickle)
  
- [x] **Explainability** (SHAP) - src/uvip_ai/explain/shap_explain.py
  - Feature importance analysis
  - Indonesian labels
  - Positive/negative impact visualization

### 4. Pipeline Scripts
- [x] extract_features.py - Extract features from all photos
- [x] merge_labels.py - Merge features with survey labels
- [x] train_xgboost.py - Train perception models
- [x] predict.py - Single image inference
- [x] validate_dataset.py - Dataset quality check

### 5. API & Deployment
- [x] FastAPI endpoint (api/main.py)
- [x] Dockerfile with CUDA support
- [x] docker-compose.yml
- [x] Model registry system

### 6. Documentation
- [x] README.md - Complete project documentation
- [x] DEPLOYMENT_AND_SERVERS.md - Server recommendations
- [x] CLOUD_DEPLOYMENT.md - Cloud deployment guide
- [x] This status document

---

## ⏳ Next Steps (User Action Required)

### Step 1: Prepare Survey Labels
**Priority: HIGH** | **Estimated Time: 1-2 days**

You need to collect survey data from respondents (Beauty, Safety, Comfort, UVI scores for each photo).

```bash
# 1. Copy template
cp data/templates/labels_template.csv data/training/labels.csv

# 2. Fill in labels.csv with survey data
# Format: filename,label_beauty,label_safety,label_comfort,label_uvi
# Example:
# KAYUTANGAN_ST-01.jpg,7.2,6.8,7.5,6.9
# KAYUTANGAN_ST-02.jpg,6.5,7.1,6.2,7.0
```

**Tips:**
- Use Likert scale 1-10 for each dimension
- Get at least 30-50 respondents per photo for statistical significance
- Can use Google Forms or similar survey tool

### Step 2: Extract Features (Cloud GPU Recommended)
**Priority: HIGH** | **Estimated Time: 2-3 hours** | **Cost: ~$1-2**

Your RTX 3060 (6GB VRAM) can handle this, but it will be slow. Cloud GPU is recommended.

**Option A: Local (Slow)**
```bash
python scripts/extract_features.py \
  --input data/extracted/photos \
  --output data/training/features.csv
```

**Option B: Cloud (Fast) - Recommended**
```bash
# 1. Rent RunPod GPU (RTX 4090)
# 2. Upload code & data
# 3. Run extraction
python scripts/extract_features.py \
  --input data/extracted/photos \
  --output data/training/features.csv
# 4. Download results
```

See [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md) for detailed instructions.

### Step 3: Merge Features & Labels
**Priority: HIGH** | **Estimated Time: < 1 minute**

```bash
python scripts/merge_labels.py \
  --features data/training/features.csv \
  --labels data/training/labels.csv \
  --output data/training/dataset.csv
```

### Step 4: Validate Dataset
**Priority: MEDIUM** | **Estimated Time: < 1 minute**

```bash
python scripts/validate_dataset.py \
  --input data/training/dataset.csv
```

Check for:
- Missing values
- Label distribution
- Outliers
- Minimum sample size (recommend ≥ 100 samples)

### Step 5: Train Models
**Priority: HIGH** | **Estimated Time: 5-10 minutes**

```bash
python scripts/train_xgboost.py \
  --input data/training/dataset.csv \
  --output models/perception/ \
  --n-folds 5
```

Target: R² ≥ 0.70 for all 4 models

If R² < 0.70:
- Collect more survey data
- Try different hyperparameters
- Feature engineering

### Step 6: Test Inference
**Priority: MEDIUM** | **Estimated Time: < 1 minute**

```bash
python scripts/predict.py \
  --image data/extracted/photos/KAYUTANGAN/KAYUTANGAN_ST-01.jpg \
  --models models/perception/
```

### Step 7: Deploy API
**Priority: LOW** | **Estimated Time: 30 minutes**

```bash
# Local
uvicorn src.uvip_ai.api.main:app --host 0.0.0.0 --port 8001

# Docker
docker-compose up -d
```

---

## 📊 Current Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Environment | ✅ Complete | Python 3.11, PyTorch CUDA, all deps |
| Data Extraction | ✅ Complete | 434 photos extracted |
| AI Models | ✅ Complete | All 5 models implemented |
| Training Pipeline | ✅ Complete | Ready to train (needs labels) |
| API | ✅ Complete | FastAPI endpoint ready |
| Documentation | ✅ Complete | All docs written |
| **Survey Labels** | ⏳ **Pending** | **User must collect** |
| **Feature Extraction** | ⏳ **Pending** | **User must run** |
| **Model Training** | ⏳ **Pending** | **User must run** |

---

## 💰 Cost Estimate

### Local Development (Your Laptop)
- **Hardware**: RTX 3060 (6GB) - Already owned
- **Electricity**: ~$5-10/month (if running 24/7)
- **Total**: $0 (one-time)

### Cloud Training (One-time)
- **RunPod RTX 4090**: ~$1-2 for feature extraction
- **RunPod RTX 4090**: ~$0.50 for model training
- **Total**: ~$2-3

### Production Deployment (Monthly)
- **RunPod RTX 4090**: ~$300/month (always-on)
- **GCP n1-standard-8 + T4**: ~$450/month
- **AWS g4dn.xlarge**: ~$400/month

**Recommendation**: Start with RunPod for training ($2-3), then decide on production deployment based on usage.

---

## 🎯 Quick Start Command Sequence

Copy-paste this sequence after you have labels.csv:

```bash
# 1. Extract features (local or cloud)
python scripts/extract_features.py \
  --input data/extracted/photos \
  --output data/training/features.csv

# 2. Merge with labels
python scripts/merge_labels.py \
  --features data/training/features.csv \
  --labels data/training/labels.csv \
  --output data/training/dataset.csv

# 3. Validate
python scripts/validate_dataset.py \
  --input data/training/dataset.csv

# 4. Train
python scripts/train_xgboost.py \
  --input data/training/dataset.csv \
  --output models/perception/

# 5. Test
python scripts/predict.py \
  --image data/extracted/photos/KAYUTANGAN/KAYUTANGAN_ST-01.jpg \
  --models models/perception/
```

---

## 📁 File Inventory

### Scripts (9 files)
```
scripts/
├── extract_photos_from_pdf.py  ✅
├── fix_manifest_coords.py      ✅
├── extract_features.py         ✅
├── merge_labels.py             ✅
├── train_xgboost.py            ✅
├── predict.py                  ✅
├── validate_dataset.py         ✅
├── verify_gpu.py               ✅
└── install_torch.sh            ✅
```

### Source Code (11 modules)
```
src/uvip_ai/
├── __init__.py                 ✅
├── config.py                   ✅
├── model_registry.py           ✅
├── privacy/
│   ├── __init__.py             ✅
│   └── guard.py                ✅
├── segmentation/
│   ├── __init__.py             ✅
│   └── segformer.py            ✅
├── features/
│   ├── __init__.py             ✅
│   └── dinov2.py               ✅
├── training/
│   ├── __init__.py             ✅
│   └── xgboost_model.py        ✅
├── explain/
│   ├── __init__.py             ✅
│   └── shap_explain.py         ✅
├── pipeline/
│   ├── __init__.py             ✅
│   └── build_dataset.py        ✅
├── api/
│   ├── __init__.py             ✅
│   └── main.py                 ✅
└── utils/
    ├── __init__.py             ✅
    └── device.py               ✅
```

### Data (434 photos extracted)
```
data/
├── extracted/
│   ├── photos/                 ✅ (434 images)
│   ├── manifest.csv            ✅
│   └── manifest_clean.csv      ✅
├── training/
│   └── (empty - user fills)
└── templates/
    └── labels_template.csv     ✅
```

### Documentation (4 files)
```
docs/
├── DEPLOYMENT_AND_SERVERS.md   ✅
├── CLOUD_DEPLOYMENT.md         ✅
├── STATUS.md                   ✅ (this file)
└── README.md                   ✅
```

### Configuration (5 files)
```
├── requirements.txt            ✅
├── Dockerfile                  ✅
├── docker-compose.yml          ✅
├── .gitignore                  ✅
└── .env.example                ✅
```

---

## 🐛 Known Issues & Solutions

### Issue 1: CUDA Out of Memory
**Problem**: RTX 3060 (6GB) runs out of VRAM during feature extraction

**Solution**:
```bash
# Use batch size 1
python scripts/extract_features.py --batch-size 1

# Or use cloud GPU
```

### Issue 2: Slow Feature Extraction
**Problem**: Local extraction takes too long

**Solution**:
- Use cloud GPU (RunPod RTX 4090): ~2 hours vs ~8 hours local
- Or reduce photo count for initial testing

### Issue 3: Low R² Score (< 0.70)
**Problem**: Models don't meet performance target

**Solution**:
1. Collect more survey data (aim for 200+ samples)
2. Check label quality (consistent ratings)
3. Try hyperparameter tuning:
   ```bash
   python scripts/train_xgboost.py --n-estimators 300 --max-depth 8
   ```

---

## 📞 Support

- **Documentation**: See README.md and docs/
- **Cloud Setup**: See CLOUD_DEPLOYMENT.md
- **Server Options**: See DEPLOYMENT_AND_SERVERS.md

---

## 🎉 Summary

**All code is complete and ready to use.**

You only need to:
1. Collect survey labels (Beauty, Safety, Comfort, UVI scores)
2. Run feature extraction (cloud recommended)
3. Train models
4. Deploy

**Estimated total time**: 2-3 days (including survey collection)
**Estimated cost**: $2-3 for cloud training

Good luck! 🚀
