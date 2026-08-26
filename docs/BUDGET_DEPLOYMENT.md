# 💰 Budget Deployment Guide - UVIP AI

Panduan deployment **murah** untuk UVIP AI. Strategi: backend terpisah dari AI, model lebih kecil, cloud GPU on-demand.

---

## 🎯 Strategi Budget-Friendly

### Prinsip Utama
1. **Backend (CPU only)**: VPS murah $5-10/bulan
2. **AI Training**: Cloud GPU on-demand, hanya bayar saat training (~$2-5)
3. **AI Inference**: Laptop lokal atau cloud GPU on-demand
4. **Model Smaller**: DINOv2-Base (768-d) instead of Large (1024-d)

---

## 📊 Kebutuhan Teknis (Revised)

### Backend (FastAPI) - CPU Only
- **CPU**: 2 vCPU
- **RAM**: 4-8 GB
- **Storage**: 50 GB SSD
- **GPU**: Tidak perlu

### AI Pipeline (GPU) - Optimized
| Model | Original | Budget Version | VRAM |
|---|---|---|---|
| YOLOv8n (privacy) | YOLOv8n | YOLOv8n (sudah nano) | ~1 GB |
| SegFormer (segmentation) | SegFormer-B5 (8GB) | **SegFormer-B0** (2GB) | ~2 GB |
| DINOv2 (embedding) | DINOv2-Large (12GB) | **DINOv2-Base** (768-d, 3GB) | ~3 GB |
| XGBoost (perception) | CPU | CPU | 0 GB |

**Total VRAM**: ~6 GB (bisa jalan di RTX 3060 6GB Anda!)

### Trade-offs
- **Embedding**: 768-d instead of 1024-d (25% lebih kecil, masih bagus)
- **Segmentation**: Lower resolution, tapi masih akurat untuk metrik urban
- **Latency**: Mungkin 1-2 detik instead of 700ms (acceptable untuk MVP)

---

## 💵 Opsi Deployment Budget

### Opsi 1: Ultra Budget - Local Only (~$0/bulan)

**Setup**:
- **Backend**: Jalankan di laptop Anda
- **AI Inference**: Laptop Anda (RTX 3060 6GB)
- **Training**: Cloud GPU on-demand (RunPod $0.20/jam)

**Cost Breakdown**:
| Item | Cost |
|------|------|
| Backend (local) | $0 |
| AI Inference (local) | $0 |
| Training (RunPod, 2 jam) | $0.40 |
| **Total** | **$0.40 one-time** |

**Limitations**:
- Laptop harus selalu on untuk backend
- Tidak bisa handle banyak concurrent users
- OK untuk development & demo

**How to Run**:
```bash
# Terminal 1: Backend
uvicorn src.uvip_ai.api.main:app --host 0.0.0.0 --port 8001

# Terminal 2: Frontend (sudah ada)
# ...
```

---

### Opsi 2: Budget Production - VPS + Cloud GPU (~$10-15/bulan)

**Setup**:
- **Backend**: VPS murah (DigitalOcean/Hetzner) $5-10/bulan
- **AI Training**: RunPod on-demand $0.20/jam
- **AI Inference**: VPS (CPU only, lambat) atau RunPod on-demand

**VPS Recommendations**:

| Provider | Specs | Price/Month | Link |
|----------|-------|-------------|------|
| **Hetzner** | 2 vCPU, 4GB RAM, 40GB SSD | €4.50 (~$5) | [hetzner.com](https://www.hetzner.com/cloud) |
| **DigitalOcean** | 2 vCPU, 4GB RAM, 80GB SSD | $12 | [digitalocean.com](https://www.digitalocean.com/products/droplets) |
| **Vultr** | 2 vCPU, 4GB RAM, 80GB SSD | $12 | [vultr.com](https://www.vultr.com/products/cloud-compute/) |
| **Linode** | 2 vCPU, 4GB RAM, 80GB SSD | $12 | [linode.com](https://www.linode.com/products/shared/) |

**Recommendation**: **Hetzner** (paling murah, performance bagus)

**Cost Breakdown**:
| Item | Cost |
|------|------|
| VPS (Hetzner) | $5/bulan |
| Training (RunPod, 2 jam) | $0.40 one-time |
| Inference (VPS CPU, lambat) | $0 (sudah termasuk VPS) |
| **Total** | **$5/bulan** |

**Architecture**:
```
┌─────────────────┐
│   Mobile App    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  VPS (Hetzner $5/mo)    │
│  - Backend FastAPI      │
│  - Database             │
│  - Photo storage        │
│  - CPU only             │
└────────┬────────────────┘
         │
         ▼ (saat perlu inference)
┌─────────────────────────┐
│  RunPod GPU (on-demand) │
│  - AI inference         │
│  - $0.20/jam            │
│  - Nyalakan saat perlu  │
└─────────────────────────┘
```

**How to Deploy**:
```bash
# 1. Setup VPS
ssh root@your-vps-ip
git clone <repo>
cd uvip-ai
pip install -r requirements.txt
uvicorn src.uvip_ai.api.main:app --host 0.0.0.0 --port 8001

# 2. Training (di RunPod)
# Rent GPU, upload data, train, download models
# See: docs/CLOUD_DEPLOYMENT.md

# 3. Inference (on-demand)
# Option A: Run inference di VPS (CPU, lambat ~5-10 detik)
# Option B: RunPod on-demand (cepat, $0.20/jam)
```

---

### Opsi 3: Google Colab Pro - All-in-One (~$10/bulan)

**Setup**:
- **Backend + AI**: Google Colab Pro (A100/L4 GPU)
- **Training + Inference**: Semua di Colab

**Cost**:
| Item | Cost |
|------|------|
| Google Colab Pro | $10/bulan |
| **Total** | **$10/bulan** |

**Benefits**:
- Dapat A100 (40GB VRAM) atau L4 (24GB)
- Bisa training & inference
- Jupyter notebook interface

**Limitations**:
- Session timeout (12 jam max)
- Tidak suitable untuk production 24/7
- OK untuk development & testing

**How to Use**:
```python
# Di Google Colab
!git clone <repo>
!cd uvip-ai && pip install -r requirements.txt
!uvicorn src.uvip_ai.api.main:app --host 0.0.0.0 --port 8001 &

# Access via ngrok
!ngrok http 8001
```

---

### Opsi 4: Kaggle Notebooks - FREE (~$0/bulan)

**Setup**:
- **Training + Inference**: Kaggle Notebooks (gratis 30 jam GPU/bulan)
- **Backend**: Local laptop

**Cost**:
| Item | Cost |
|------|------|
| Kaggle Notebooks | $0 (30 jam GPU/bulan) |
| Backend (local) | $0 |
| **Total** | **$0/bulan** |

**Benefits**:
- Gratis 30 jam GPU/bulan (T4/P100)
- Bisa training & inference
- Persistent storage 20GB

**Limitations**:
- 30 jam/bulan limit
- Session timeout (12 jam)
- Tidak suitable untuk production

**How to Use**:
```python
# Di Kaggle Notebook
!git clone <repo>
!cd uvip-ai && pip install -r requirements.txt

# Training
!python scripts/train_xgboost.py --input data/training/dataset.csv

# Inference
!python scripts/predict.py --image test.jpg --models models/perception/
```

---

## 🔧 Model Optimization untuk Budget

### 1. Gunakan DINOv2-Base instead of Large

**Config** (edit `.env`):
```bash
DINOV2_MODEL=facebook/dinov2-base  # 768-d, 3GB VRAM
# instead of
# DINOV2_MODEL=facebook/dinov2-large  # 1024-d, 12GB VRAM
```

**Impact**:
- VRAM: 12GB → 3GB (75% reduction)
- Embedding: 1024-d → 768-d (25% smaller)
- Quality: Masih bagus untuk perception prediction

### 2. Gunakan SegFormer-B0 instead of B5

**Config** (edit `.env`):
```bash
SEGFORMER_MODEL=nvidia/segformer-b0-finetuned-ade-512-512  # 2GB VRAM
# instead of
# SEGFORMER_MODEL=nvidia/segformer-b5-finetuned-cityscapes-1024-1024  # 8GB VRAM
```

**Impact**:
- VRAM: 8GB → 2GB (75% reduction)
- Resolution: 1024x1024 → 512x512 (lower, tapi masih OK)
- Quality: Slightly lower accuracy, tapi masih acceptable

### 3. Update Code untuk Support Model Smaller

Edit `src/uvip_ai/features/dinov2.py`:
```python
class Dinov2Extractor:
    DEFAULT_MODEL_ID = "facebook/dinov2-base"  # Changed from large
    EMBED_DIM = 768  # Changed from 1024
```

Edit `src/uvip_ai/segmentation/segformer.py`:
```python
class SegformerB5:
    DEFAULT_MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"  # Changed from b5
```

**Note**: Perlu retrain XGBoost models setelah ganti embedding size (768-d instead of 1024-d).

---

## 🚀 Recommended Budget Setup

### Untuk Development & Demo (Paling Murah)
**Opsi 1: Local Only** - $0.40 one-time
- Backend + AI di laptop
- Training di RunPod (2 jam, $0.40)
- OK untuk development & demo

### Untuk Production MVP
**Opsi 2: VPS + Cloud GPU** - $5/bulan
- Backend di Hetzner VPS ($5/bulan)
- Training di RunPod ($0.40 one-time)
- Inference di VPS (CPU, lambat) atau RunPod on-demand

### Untuk Testing & Experimentation
**Opsi 4: Kaggle Notebooks** - $0/bulan
- Semua di Kaggle (gratis 30 jam GPU/bulan)
- Backend di laptop
- OK untuk testing & experimentation

---

## 📊 Cost Comparison

| Setup | Monthly Cost | Best For |
|-------|--------------|----------|
| **Local Only** | $0 + $0.40 one-time | Development & demo |
| **VPS (Hetzner) + RunPod** | $5/bulan | Production MVP |
| **Google Colab Pro** | $10/bulan | Development & testing |
| **Kaggle Notebooks** | $0/bulan | Testing & experimentation |
| **RunPod Always-On** | $150-220/bulan | Production (budget) |
| **GCP g2-standard-4** | $400-500/bulan | Production (enterprise) |

---

## 🎯 My Recommendation

### Untuk Anda (Budget-Conscious)

**Phase 1: Development & Training** (Now)
- **Setup**: Local laptop + RunPod on-demand
- **Cost**: $0.40 one-time
- **Action**: 
  1. Collect survey labels
  2. Train di RunPod (2 jam, $0.40)
  3. Download trained models

**Phase 2: MVP Deployment** (1-2 bulan)
- **Setup**: Hetzner VPS + RunPod on-demand
- **Cost**: $5/bulan
- **Action**:
  1. Deploy backend di Hetzner VPS
  2. Inference di VPS (CPU, lambat tapi OK untuk MVP)
  3. Test dengan 10-20 users

**Phase 3: Production** (3-6 bulan, jika sukses)
- **Setup**: Upgrade ke RunPod always-on atau GCP
- **Cost**: $150-500/bulan
- **Action**:
  1. Scale up jika user base grows
  2. Migrate ke dedicated GPU server

---

## 📝 Quick Start (Budget Setup)

### Step 1: Train Models (RunPod, $0.40)
```bash
# 1. Rent RunPod GPU (RTX 4090, $0.40/jam)
# 2. Upload code & data
# 3. Run training
python scripts/quick_start.py
# 4. Download models (2 jam, $0.40)
```

### Step 2: Deploy Backend (Hetzner VPS, $5/bulan)
```bash
# 1. Rent Hetzner VPS (2 vCPU, 4GB RAM, €4.50/bulan)
# 2. Deploy backend
ssh root@your-vps-ip
git clone <repo>
cd uvip-ai
pip install -r requirements.txt
uvicorn src.uvip_ai.api.main:app --host 0.0.0.0 --port 8001
```

### Step 3: Test Inference
```bash
# Test di VPS (CPU, lambat ~5-10 detik)
curl -X POST http://your-vps-ip:8001/predict \
  -F "file=@test.jpg"
```

**Total Cost**: $5/bulan + $0.40 one-time = **~$5.40 pertama, $5/bulan seterusnya**

---

## 🔗 Links & Resources

### VPS Providers
- **Hetzner**: https://www.hetzner.com/cloud (€4.50/bulan, recommended)
- **DigitalOcean**: https://www.digitalocean.com/products/droplets ($12/bulan)
- **Vultr**: https://www.vultr.com/products/cloud-compute/ ($12/bulan)

### Cloud GPU (On-Demand)
- **RunPod**: https://www.runpod.io/ ($0.20-0.50/jam, recommended)
- **Vast.ai**: https://vast.ai/ ($0.15-0.40/jam, spot market)
- **Google Colab Pro**: https://colab.research.google.com/pro ($10/bulan)
- **Kaggle Notebooks**: https://www.kaggle.com/code (free, 30 jam/bulan)

### Model Optimization
- **DINOv2-Base**: https://huggingface.co/facebook/dinov2-base
- **SegFormer-B0**: https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512

---

## 📖 Documentation

- **README.md**: Complete project guide
- **CLOUD_DEPLOYMENT.md**: Cloud GPU guide (RunPod, GCP, AWS)
- **DEPLOYMENT_AND_SERVERS.md**: Server recommendations (original, mahal)
- **BUDGET_DEPLOYMENT.md**: This file (budget-friendly)

---

## ✅ Summary

**Budget Setup**:
- **Development**: $0.40 one-time (RunPod training)
- **MVP Production**: $5/bulan (Hetzner VPS)
- **Total**: ~$5.40 pertama, $5/bulan seterusnya

**Trade-offs**:
- Inference lambat (5-10 detik di VPS CPU)
- Model lebih kecil (768-d embedding, lower resolution segmentation)
- OK untuk MVP & 10-20 users

**Next Steps**:
1. Collect survey labels
2. Train di RunPod ($0.40)
3. Deploy backend di Hetzner VPS ($5/bulan)
4. Test dengan users
5. Scale up jika sukses

**Good luck! 🚀**
