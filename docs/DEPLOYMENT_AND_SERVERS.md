# UVIP-AI: Rekomendasi Server Produksi & Deployment

Dokumen ini berisi rekomendasi infrastruktur hardware, cloud providers, dan estimasi biaya untuk backend + AI pipeline UVIP dalam skala produksi (target 700ms latency).

---

## Ringkasan Kebutuhan Teknis

### Backend (FastAPI)
- API endpoint `POST /street-photos/` dan `POST /ai/process`
- Auth JWT, upload multipart files
- **CPU**: minimal 4 vCPU
- **RAM**: 16–32 GB
- **Storage**: SSD ≥ 50 GB (upload photos, masks)

### AI Pipeline (GPU Inference)
| Model | GPU Memory (fp16) | Catatan |
|---|---|---|
| YOLOv8n (privacy) | ~2 GB | Sangat ringan |
| SegFormer-B5 (segmentation) | ~8 GB | Heavy — batch 1 |
| DINOv2-Large (embedding) | ~12 GB | Terberat, bottleneck utama |
| XGBoost (perception) | CPU | Tidak perlu GPU |

**Total VRAM peak per request:** ~20–25 GB untuk load bertahap  
**Target latency:** < 700 ms end-to-end  
**Concurrent requests:** 5–10 awal → scale up later

### Dataset (untuk training)
- Input: 434 foto Kota Malang + label kuesioner (Beauty/Safety/Comfort/UVI)
- Output: model_registry.json + 4 XGBoost models
- **Training hardware:** GPU 1× A100/A10G / T4 cukup (XGBoost CPU anyway)

---

## Opsi 1: Cloud GPU Production (Rekomendasi Utama)

### 1. Google Cloud Platform (GCP) — **Paling Recommended**

#### Instance Type: `g2-standard-4` atau `a2-highgpu-1g`
- **CPU:** 4 vCPU
- **RAM:** 16 GB
- **GPU:** NVIDIA L4 24 GB **(best value untuk inference)**
- **Disk:** 50+ GB NVMe SSD

**Harga per bulan (est.):**
- GCP `g2-standard-4` (L4): ~$400–500/bulan (preemptible $150–200)
- GCP `a2-highgpu-1g` (A10G): ~$900–1,100/bulan

**Link:**
- [GCP Compute Engine](https://cloud.google.com/compute)
- [g2 instance details](https://cloud.google.com/compute/gpu-host-type#g2-standard-4)

**Keunggulan:**
- CUDA pre-installed, easy PyTorch setup
- Good network bandwidth untuk mobile sync
- Auto-scaling support via Kubernetes Engine (GKE)

---

### 2. AWS EC2

#### Instance Type: `g4dn.xlarge` (T4) atau `g5.xlarge` (A10G)
- **CPU:** 4 vCPU
- **RAM:** 16 GB
- **GPU:** NVIDIA T4 16 GB / A10G 24 GB
- **Disk:** EBS NVMe SSD 50+ GB

**Harga per bulan:**
- `g4dn.xlarge` (T4): ~$200–280/bulan
- `g5.xlarge` (A10G): ~$700–900/bulan

**Link:**
- [AWS EC2 GPU instances](https://aws.amazon.com/ec2/instance-types/#Get_a_GPU)
- [Deep Learning AMI](https://aws.amazon.com/marketplace/pp/prodview-zzrwrjqxwvq3o) (PyTorch ready)

**Keunggulan:**
- Mature ecosystem, many tutorials
- S3 untuk photo storage (integrate langsung)

---

### 3. RunPod / Vast.ai — **Termurah**

#### Community Cloud GPU (bisa dedicated)
- **CPU:** 4 vCPU
- **RAM:** 32 GB
- **GPU:** L4 24 GB / A10 24 GB
- **Disk:** 100+ GB NVMe

**Harga per jam:**
- L4: ~$0.20–0.30/hour → ~$150–220/bulan (24/7)
- A10: ~$0.35–0.50/hour → ~$250–350/bulan

**Link:**
- [RunPod.io](https://www.runpod.io/console/cloud-gpu-deployment)
- [Vast.ai](https://vast.ai/) (spot market)

**Keunggulan:**
- **50% lebih murah** dari GCP/AWS
- Docker-ready template tersedia
- Scaling mudah via CLI/API

---

## Opsi 2: Bare Metal Local Server (Jika Ada Budget Besar)

### Hardware Spec Production-Grade

| Komponen | Rekomendasi | Estimasi Harga |
|---|---|---|
| **GPU** | NVIDIA RTX 4090 24GB / L40S 48GB | Rp 50–80 jt |
| **CPU** | AMD Ryzen 9 7950X / Intel i9-14900K | Rp 25–30 jt |
| **RAM** | DDR5 64GB (2×32GB) | Rp 12–15 jt |
| **SSD** | NVMe Gen4 2TB Samsung 990 Pro | Rp 4–5 jt |
| **PSU** | 1000W Platinum Gold | Rp 3–4 jt |
| **Mobo** | B650E / Z790 chipset | Rp 10–15 jt |
| **Cooling** | AIO 360mm Liquid Cooler | Rp 5–7 jt |
| **Case** | High airflow (Fractal Meshify) | Rp 2–3 jt |

**Total:** ± **Rp 100–140 juta**

**Link pembelian Indonesia:**
- [Bhisma](https://bhisma.com/) (component PC)
- [NotebookShop](https://notebookshop.id/)
- [TomboyComputer](https://tomboycomputer.co.id/)

**Catatan:** Only recommended jika you plan multi-year use + have local ops team. Otherwise, cloud is more cost-effective.

---

## Opsi 3: Hybrid Approach (Development Local + Production Cloud)

Karena laptop kamu punya **RTX 3060 Laptop 6GB VRAM**:

| Phase | Hardware | Biaya |
|---|---|---|
| **Development & prototyping** | Laptop lokal | Rp 0 (sudah ada) |
| **Testing & validation** | RunPod L4 on-demand | $1–5/jam saat run |
| **Production** | GCP g2-standard-4 (always-on) | $400–500/bulan |

**Strategi:**
1. Train & test model di laptop (batch inference, low-vram mode)
2. Deploy ke RunPod/GCP via Docker image
3. CI/CD push model registry otomatis

---

## Stack Deployment Lengkap

### Docker Compose Template (production-ready)

```yaml
version: "3.9"

services:
  uvip-api:
    build: .
    ports:
      - "8001:8001"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./uploads:/app/uploads
      - ./models:/app/models
    env_file:
      - .env

  uvip-worker:
    extends: uvip-api
    command: python scripts/process_queue.py
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: always
```

---

## Estimsasi Biaya Total/Bulan (Scale Awal: 1000 fotos/bulan, 5 koncurrent req)

| Provider | Configuration | Monthly Cost |
|---|---|---|
| **GCP** | g2-standard-4 (L4) + Cloud Storage | **$450–550** |
| **AWS** | g4dn.xlarge (T4) + S3 | **$250–300** |
| **RunPod** | L4 community cluster (dedicated) | **$180–220** |
| **Vast.ai** | L4 spot market (on/off) | **$120–180** |

**Untuk scale 10k+/bulan:** Add 2× worker nodes, total ×1.5 multiplier.

---

## Quick Start Production (RunPod Example)

1. **Create pod:** https://runpod.io/console/cloud-gpu-deployment
   - Select template: "PyTorch Latest"
   - GPU: L4 24GB
   - Size: default (16GB RAM, 4 CPU)

2. **Deploy image:**
```bash
docker build -t uvip-ai .
docker tag uvip-ai:latest runpod.io/<YOUR_USER>/uvip-ai:latest
docker push runpod.io/<YOUR_USER>/uvip-ai:latest
```

3. **Attach pod & start:**
```bash
curl -X POST https://api.runpod.io/v2/pod/start \
  --header "Authorization: Bearer $RUNPOD_API_KEY" \
  --json '{"podTemplateId":"<TEMPLATE_ID>"}'
```

4. **Access API:** `https://<POD_IP>:8001`

---

## Checklist Setup Production

- [ ] Backup manifest.csv + semua extracted photos ke S3/GCS bucket
- [ ] Set `UVIP_LOW_VRAM_MODE=true` di .env (wajib untuk L4/T4)
- [ ] Load models via HuggingFace cache (`HF_HOME=/app/models/hf_cache`)
- [ ] Test end-to-end latency dengan postman: `curl http://<server>:8001/ai/process`
- [ ] Monitor GPU usage via `watch -n 1 nvidia-smi`

---

## Contact & Resources

- GitHub Issues: Report setup bugs di repo ini
- Discord UVIP: Komunitas developer (jika ada)
- Documentation: [Backend API](./BACKEND_API.md), [Model Registry](./MODEL_REGISTRY.md)

---

*Last updated: August 2026. Prices subject to change per provider policy.*
