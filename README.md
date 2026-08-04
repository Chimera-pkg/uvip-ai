# UVIP-AI — Urban Visual Perception (Pipeline AI)

Pipeline AI untuk menilai persepsi publik (**Beauty / Safety / Comfort / UVI**)
ruang kota dari foto street-level Kota Malang (Kayutangan, Alun-Alun Tugu,
Alun-Alun Merdeka). Backend & frontend UVIP sudah ada — repo ini mengerjakan
bagian **AI/ML** dan mengintegrasikannya ke API backend.

## Arsitektur pipeline
```
Foto → [YOLOv8n] blur wajah/plat → [SegFormer-B5] 5 metrik urban
     → [DINOv2] embedding 1024-d → [XGBoost] skor Beauty/Safety/Comfort/UVI
     → [SHAP] faktor pendorong → POST hasil ke backend UVIP
```

## Struktur folder
```
src/uvip_ai/
  config.py         # konfigurasi terpusat (.env)
  privacy/          # Step 2 — YOLOv8n masking
  segmentation/     # Step 3 — SegFormer-B5 + 5 metrik
  features/         # Step 4 — DINOv2 embedding
  training/         # Step 6 — XGBoost 4 model
  explain/          # Step 7 — SHAP
  pipeline/         # orkestrasi end-to-end
  api/              # Step 8 — FastAPI endpoint
  utils/
scripts/            # verify_gpu, install_torch, ekstraksi dataset
data/               # raw / extracted / masks / datasets (git-ignored)
models/             # weights + model_registry (git-ignored)
```

## Environment saat ini (terverifikasi)
- Python 3.11.9 ✅
- GPU: **RTX 3060 Laptop, 6 GB VRAM**, driver 546.80 ✅
- Karena VRAM 6 GB: `UVIP_USE_FP16=true` & `UVIP_LOW_VRAM_MODE=true` (default di `.env`).

## Setup (Step 1)

### 1. Virtual environment
```bash
python -m venv .venv
# Windows (Git Bash):
source .venv/Scripts/activate
# Windows (PowerShell):
#   .venv\Scripts\Activate.ps1
```

### 2. Install PyTorch (build CUDA) — WAJIB duluan
```bash
bash scripts/install_torch.sh
# atau manual:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install sisa dependency
```bash
pip install -r requirements.txt
```

### 4. Konfigurasi
```bash
cp .env.example .env      # lalu isi UVIP_API_TOKEN bila perlu
```

### 5. Verifikasi GPU & library
```bash
python scripts/verify_gpu.py
```
Harus muncul `CUDA tersedia` dan uji matmul GPU sukses.

## Menjalankan modul
Set `PYTHONPATH` ke `src` (atau `pip install -e .`):
```bash
export PYTHONPATH=src        # Windows PowerShell: $env:PYTHONPATH="src"
python scripts/verify_gpu.py
```

## Docker (Step 10)
```bash
docker build -t uvip-ai .
docker run --gpus all -p 8001:8001 --env-file .env uvip-ai
```

## Roadmap (lihat intro.txt)
| Step | Modul | Status |
|---|---|---|
| 1 Setup env & GPU | `scripts/verify_gpu.py` | ✅ selesai |
| 2 Privacy Guard (YOLOv8n) | `privacy/` | ⬜ |
| 3 Segmentasi (SegFormer-B5) | `segmentation/` | ⬜ |
| 4 Feature Extraction (DINOv2) | `features/` | ⬜ |
| 5 Siapkan dataset training | `pipeline/` | ⬜ |
| 6 Training XGBoost (R²≥0.7) | `training/` | ⬜ butuh label kuesioner |
| 7 SHAP | `explain/` | ⬜ |
| 8 FastAPI endpoint | `api/` | ⬜ |
| 9 Uji latensi <700ms | — | ⬜ |
| 10 model_registry + Docker | `Dockerfile` | ⬜ |

> ⚠️ **Bottleneck:** dataset kuesioner (skor persepsi dari responden) belum ada.
> PDF dataset saat ini hanya berisi 434 foto + koordinat GPS, tanpa label skor.
> Step 2–5 bisa jalan sekarang; Step 6 menunggu label.
