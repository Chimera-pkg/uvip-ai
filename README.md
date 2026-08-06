# UVIP AI - Urban Visual Intelligence Platform

Sistem AI untuk menganalisis persepsi visual ruang kota menggunakan computer vision dan machine learning.

## 📋 Overview

UVIP AI memproses foto street-level untuk menghasilkan:
- **Privacy masking** (blur wajah & plat nomor)
- **Segmentation metrics** (5 metrik urban)
- **Feature embeddings** (1024-d DINOv2)
- **Perception predictions** (Beauty, Safety, Comfort, UVI)
- **Explainability** (SHAP values)

## 🏗️ Architecture

```
┌─────────────────┐
│   Input Photo   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Privacy Guard (YOLO)   │  → Blur wajah & plat nomor
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  SegFormer-B5           │  → 5 metrik urban:
│  (Segmentation)         │    - Green Coverage %
│                         │    - Building Coverage %
│                         │    - Walkability Ratio
│                         │    - Visual Clutter Index
│                         │    - Sky Visibility %
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  DINOv2-Large           │  → 1024-d embedding
│  (Feature Extraction)   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  XGBoost (4 models)     │  → Beauty Score
│  (Perception)           │  → Safety Score
│                         │  → Comfort Score
│                         │  → UVI Score
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  SHAP Explainability    │  → Feature importance
└─────────────────────────┘
```

## 📦 Project Structure

```
uvip-ai/
├── src/uvip_ai/
│   ├── privacy/
│   │   └── guard.py              # YOLOv8n privacy masking
│   ├── segmentation/
│   │   └── segformer.py          # SegFormer-B5 segmentation
│   ├── features/
│   │   └── dinov2.py             # DINOv2-Large feature extraction
│   ├── training/
│   │   └── xgboost_model.py      # XGBoost perception models
│   ├── explain/
│   │   └── shap_explain.py       # SHAP explainability
│   ├── pipeline/
│   │   └── build_dataset.py      # Dataset builder
│   ├── api/
│   │   └── main.py               # FastAPI endpoint
│   ├── model_registry.py         # Model versioning
│   ├── config.py                 # Configuration
│   └── utils/
│       └── device.py             # GPU utilities
├── scripts/
│   ├── extract_photos_from_pdf.py
│   ├── fix_manifest_coords.py
│   ├── extract_features.py       # Extract features from photos
│   ├── merge_labels.py           # Merge features with labels
│   ├── train_xgboost.py          # Train perception models
│   ├── predict.py                # Inference script
│   ├── verify_gpu.py             # GPU verification
│   └── install_torch.sh          # PyTorch installation
├── data/
│   ├── extracted/
│   │   ├── photos/               # Extracted photos
│   │   ├── manifest.csv          # Photo metadata
│   │   └── manifest_clean.csv    # Fixed coordinates
│   ├── training/
│   │   ├── features.csv          # Extracted features
│   │   ├── labels.csv            # Survey labels
│   │   └── dataset.csv           # Merged dataset
│   └── templates/
│       └── labels_template.csv   # Label template
├── models/
│   ├── perception/               # Trained XGBoost models
│   └── model_registry.json       # Model metadata
├── docs/
│   ├── DEPLOYMENT_AND_SERVERS.md # Server recommendations
│   └── CLOUD_DEPLOYMENT.md       # Cloud deployment guide
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repo-url>
cd uvip-ai

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Linux/Mac

# Install PyTorch with CUDA
bash scripts/install_torch.sh

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python scripts/verify_gpu.py
```

### 2. Extract Photos from PDF

```bash
# Extract 434 photos from dataset PDF
python scripts/extract_photos_from_pdf.py

# Fix coordinate issues
python scripts/fix_manifest_coords.py
```

### 3. Extract Features (Cloud GPU Recommended)

```bash
# Extract features from all photos
python scripts/extract_features.py \
  --input data/extracted/photos \
  --output data/training/features.csv
```

**Note:** Proses ini membutuhkan GPU dengan VRAM ≥6GB. Untuk laptop dengan RTX 3060 6GB, gunakan cloud GPU (lihat [CLOUD_DEPLOYMENT.md](docs/CLOUD_DEPLOYMENT.md)).

### 4. Prepare Labels

```bash
# Copy template
cp data/templates/labels_template.csv data/training/labels.csv

# Edit labels.csv dengan data survey kuesioner
# Format: filename,label_beauty,label_safety,label_comfort,label_uvi
```

### 5. Merge Features & Labels

```bash
python scripts/merge_labels.py \
  --features data/training/features.csv \
  --labels data/training/labels.csv \
  --output data/training/dataset.csv
```

### 6. Train Models

```bash
python scripts/train_xgboost.py \
  --input data/training/dataset.csv \
  --output models/perception/ \
  --n-folds 5
```

### 7. Test Inference

```bash
python scripts/predict.py \
  --image data/extracted/photos/KAYUTANGAN/KAYUTANGAN_ST-01.jpg \
  --models models/perception/
```

## 🔧 Scripts Reference

### Data Preparation

| Script | Fungsi |
|--------|--------|
| `extract_photos_from_pdf.py` | Ekstrak foto dari PDF dataset |
| `fix_manifest_coords.py` | Fix koordinat GPS yang tidak valid |
| `extract_features.py` | Extract features (segmentation + embedding) dari semua foto |
| `merge_labels.py` | Merge features dengan survey labels |

### Training & Inference

| Script | Fungsi |
|--------|--------|
| `train_xgboost.py` | Train 4 XGBoost models (Beauty, Safety, Comfort, UVI) |
| `predict.py` | Inference single image |
| `verify_gpu.py` | Verifikasi GPU & dependencies |

## 📊 Model Performance

Target performance (R² score):
- **Beauty**: ≥ 0.70
- **Safety**: ≥ 0.70
- **Comfort**: ≥ 0.70
- **UVI**: ≥ 0.70

Jika R² < 0.70, pertimbangkan:
1. Tambah data survey (lebih banyak responden)
2. Feature engineering (tambah metrik segmentasi)
3. Hyperparameter tuning (n_estimators, max_depth, learning_rate)

## 🖥️ Hardware Requirements

### Local Development (Minimal)
- **GPU**: NVIDIA RTX 3060 (6GB VRAM)
- **RAM**: 16GB
- **Storage**: 50GB SSD
- **OS**: Windows 10/11, Linux

### Production (Recommended)
- **GPU**: NVIDIA RTX 4090 (24GB) atau A100 (80GB)
- **RAM**: 32GB+
- **Storage**: 500GB NVMe SSD
- **OS**: Ubuntu 22.04 LTS

### Cloud Options
- **RunPod**: RTX 4090 @ $0.40/hour (recommended)
- **GCP**: n1-standard-8 + T4 @ $0.60/hour
- **AWS**: g4dn.xlarge @ $0.53/hour

Lihat [DEPLOYMENT_AND_SERVERS.md](docs/DEPLOYMENT_AND_SERVERS.md) untuk detail.

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t uvip-ai:latest .
```

### Run Container

```bash
docker run --gpus all -p 8001:8001 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  uvip-ai:latest
```

### Docker Compose

```bash
docker-compose up -d
```

API akan tersedia di `http://localhost:8001`

## 📡 API Endpoints

### Health Check
```bash
curl http://localhost:8001/health
```

### Predict Perception
```bash
curl -X POST http://localhost:8001/predict \
  -F "file=@photo.jpg" \
  -F "latitude=-7.976" \
  -F "longitude=112.630"
```

Response:
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
    "building_coverage_pct": -0.32,
    ...
  }
}
```

## 🌐 Cloud Deployment

### RunPod (Recommended)

```bash
# Install runpodctl
pip install runpodctl

# Create pod
runpodctl create pod \
  --name uvip-training \
  --gpuType "NVIDIA GeForce RTX 4090" \
  --volumeSize 100

# Connect & deploy
ssh root@<pod-ip>
git clone <repo-url>
cd uvip-ai
# ... follow training steps
```

### GCP

```bash
gcloud compute instances create uvip-gpu \
  --machine-type n1-standard-8 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --image-family pytorch-latest-gpu \
  --image-project deeplearning-platform-release
```

Lihat [CLOUD_DEPLOYMENT.md](docs/CLOUD_DEPLOYMENT.md) untuk panduan lengkap.

## 📝 Data Format

### Features CSV
```csv
filename,area,point_id,green_coverage_pct,building_coverage_pct,walkability_ratio,visual_clutter_index,sky_visibility_pct,emb_0,emb_1,...,emb_1023
KAYUTANGAN_ST-01.jpg,KAYUTANGAN,ST-01,35.2,28.1,0.42,0.18,22.5,0.123,0.456,...,0.789
```

### Labels CSV
```csv
filename,label_beauty,label_safety,label_comfort,label_uvi
KAYUTANGAN_ST-01.jpg,7.2,6.8,7.5,6.9
```

### Dataset CSV (Merged)
```csv
filename,area,point_id,green_coverage_pct,...,emb_0,...,label_beauty,label_safety,label_comfort,label_uvi
KAYUTANGAN_ST-01.jpg,KAYUTANGAN,ST-01,35.2,...,0.123,...,7.2,6.8,7.5,6.9
```

## 🔍 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python scripts/extract_features.py --batch-size 4

# Use fp16 mode
export UVIP_USE_FP16=true
```

### Model Not Loading
```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface

# Re-download models
python -c "from transformers import SegformerForSemanticSegmentation; SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-cityscapes-1024-1024')"
```

### Low R² Score
1. Check data quality: `python scripts/validate_dataset.py`
2. Increase training data (collect more surveys)
3. Tune hyperparameters: `python scripts/train_xgboost.py --n-estimators 300 --max-depth 8`

## 📚 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [SegFormer Paper](https://arxiv.org/abs/2105.15203)
- [DINOv2 Paper](https://arxiv.org/abs/2304.07193)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)

## 🤝 Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 👥 Authors

- UVIP AI Team

## 🙏 Acknowledgments

- NVIDIA for SegFormer models
- Meta AI for DINOv2
- Ultralytics for YOLOv8
- XGBoost developers
- SHAP library contributors
