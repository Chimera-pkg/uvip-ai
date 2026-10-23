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

# Install paket ini (editable) supaya `import uvip_ai` jalan
pip install -e .

# Verify GPU
python scripts/verify_gpu.py
```

### 1b. Jalankan API (lokal)

```bash
# Windows
.venv\Scripts\python.exe run_dev.py

# Linux/Mac
.venv/bin/python run_dev.py

# opsi: --host 0.0.0.0 --port 8010 --reload
```

Buka `http://127.0.0.1:8001/docs` (uji interaktif) atau cek `curl http://127.0.0.1:8001/health`.

> Alternatif tanpa `pip install -e .`: set `PYTHONPATH` ke folder `src`
> (`set PYTHONPATH=%CD%\src` di cmd, `$env:PYTHONPATH="$PWD\src"` di PowerShell),
> lalu `.venv\Scripts\python.exe -m uvicorn uvip_ai.api.main:app --port 8001`.

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

### Proses Foto
```bash
curl -X POST http://localhost:8001/ai/process \
  -F "file=@76117d88-42d0-4766-a2b8-97fbc4a80ecc.jpg" \
  -F "photo_id=76117d88-42d0-4766-a2b8-97fbc4a80ecc"
```

Semua artefak disimpan FLAT di folder per jenis (tanpa subfolder per aset)
dengan skema nama lama: original `<nama_upload>_<uuid4>.<ext>`, turunan
`<prefix>_<nama_upload>_<uuid4>_<epoch_ms>.<ext>`. `<nama_upload>` = stem
nama file upload; bila kosong dipakai `photo_id`. Satu foto memakai
`<epoch_ms>` yang sama untuk seg/mask/overlay/result. File lama yang memakai
timestamp `YYYYMMDD_HHMMSS` tetap bisa dibaca resolver.

Response (ringkas):
```json
{
  "photo_id": "76117d88-42d0-4766-a2b8-97fbc4a80ecc",
  "privacy_masked_url": "/uploads/masks/mask_76117d88-42d0-4766-a2b8-97fbc4a80ecc_1ee5f69f-62d9-45f7-9db0-96f5d8aac6a4_1792582425737.jpg",
  "segmentation_url": "/uploads/segmentation/seg_76117d88-42d0-4766-a2b8-97fbc4a80ecc_1ee5f69f-62d9-45f7-9db0-96f5d8aac6a4_1792582425737.jpg",
  "segmentation_overlay_url": "/uploads/segmentation/overlay_76117d88-42d0-4766-a2b8-97fbc4a80ecc_1ee5f69f-62d9-45f7-9db0-96f5d8aac6a4_1792582425737.jpg",
  "original_image_url": "/uploads/originals/76117d88-42d0-4766-a2b8-97fbc4a80ecc_1ee5f69f-62d9-45f7-9db0-96f5d8aac6a4.jpg",
  "result_url": "/uploads/results/result_76117d88-42d0-4766-a2b8-97fbc4a80ecc_1ee5f69f-62d9-45f7-9db0-96f5d8aac6a4_1792582425737.json",
  "result_path": "uploads/results/result_76117d88-42d0-4766-a2b8-97fbc4a80ecc_1ee5f69f-62d9-45f7-9db0-96f5d8aac6a4_1792582425737.json",
  "capture_location": {
    "latitude": -7.97685,
    "longitude": 112.6306111,
    "source": "exif_gps",
    "captured_at": "2026-10-22T08:31:09"
  },
  "segmentation_results": {
    "green_coverage_pct": 13.4722,
    "building_coverage_pct": 54.1653,
    "walkability_ratio": 0.347,
    "visual_clutter_index": 0.8653,
    "sky_visibility_pct": 1.1923
  },
  "perception_prediction": {
    "beauty_score": 9.69,
    "safety_score": 6.12,
    "comfort_score": 3.29,
    "uvi_score": 0.67
  },
  "metrics_source": {"indices_source": "segformer_excel", "pct_by_class": {"...": "..."}},
  "_processing_time_ms": 17046
}
```

### Rekomendasi + Redesign
```bash
# Segmentasi + skor + rekomendasi LLM (satu langkah)
curl -X POST http://localhost:8001/ai/advise \
  -F "file=@photo.jpg" \
  -F "location=Koridor Kayutangan, Malang"

# Hasil /ai/process -> gambar redesign (butuh OPENAI_API_KEY)
# "result" menerima result_url/result_path dari respons, path artefak
# (mask/seg/overlay) di uploads/, path file result_*.json, ATAU JSON inline.
curl -X POST http://localhost:8001/llm/generate \
  -F "result=/uploads/results/result_76117d88-42d0-4766-a2b8-97fbc4a80ecc_1ee5f69f-62d9-45f7-9db0-96f5d8aac6a4_1792582425737.json"
```

Skor dihitung `uvip_ai.scoring.pvi_scorer` (8 indeks SegFormer -> PVI, dinormalisasi
1..10 dengan anchor p2/p98 dari 431 foto), bukan model persepsi terlatih — lihat
[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md#-daftar-endpoint) untuk bentuk respons
lengkap dan batasannya.

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
# Mode hemat VRAM sudah default di skrip ekstraksi/builder (low_vram_mode=True).
# Kalau masih OOM: proses per area, bukan seluruh dataset sekaligus
python scripts/extract_features.py \
  --input data/extracted/photos/KAYUTANGAN \
  --output data/training/features_kayutangan.csv

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

### **Vast.ai (RECOMMENDED FOR BUDGET!)**
- **GTX 1070 @ $0.051/hour** ⚡ **SUPER CHEAP!**
- Perfect for inference and moderate batch processing
- See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for complete guide

Lihat [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) untuk detail Vast.ai deployment.


## 🐳 Docker Deployment

### Build Image
