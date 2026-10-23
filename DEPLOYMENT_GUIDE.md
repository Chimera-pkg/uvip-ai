# 🚀 UVIP AI - Deployment Guide for Vast.ai GTX 1070

## 📋 Server Specifications (Your Server)
- **GPU:** NVIDIA GTX 1070 (8GB VRAM, CUDA 12.8) ✅
- **CPU:** Xeon E5-2620 v3 @ 2.4GHz (12 threads)
- **RAM:** Sufficient for inference
- **Storage:** 16GB NVMe SSD
- **Cost:** $0.051/hour ⚡ **SUPER CHEAP!**

---

## 🎯 Quick Start (5 Steps)

### Step 1: Setup Project on Server
```bash
# SSH into your server
ssh root@182.224.239.168

# Navigate to workspace
cd /workspace

# Run setup script
chmod +x scripts/setup_vast_ai.sh
./scripts/setup_vast_ai.sh
```

### Step 2: Upload Trained Models
```bash
# Upload models from local to server
python scripts/upload_models.py \
    --local-dir models/perception \
    --remote-path /app/models/perception \
    --key-file ~/.ssh/vastai_private_key
```

### Step 3: Deploy API Server
```bash
# Start the API server
chmod +x scripts/deploy_api.sh
./scripts/deploy_api.sh

# Server will run and you can CTRL+C to stop monitoring
# Server continues running in background
```

### Step 4: Test API
```bash
# Health check
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs

# LLM gateway (9router) — ikut ter-mount di app yang sama, prefix /llm
curl "http://localhost:8000/llm/health"
curl "http://localhost:8000/llm/ask?q=halo"
# Chat multi-turn (body JSON) — enak dites juga dari /docs
curl -X POST "http://localhost:8000/llm/chat" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Sebutkan 3 nama warung kopi."}]}'
# Analisa gambar (butuh model vision-capable; default NINE_ROUTER_VISION_MODEL)
curl -X POST "http://localhost:8000/llm/vision" \
    -F "file=@data/extracted/photos/KAYUTANGAN/KAYUTANGAN_BS-01.jpg" \
    -F "prompt=Jelaskan kondisi jalan pada gambar ini."
# Halaman uji (bisa di-embed via iframe): http://localhost:8000/llm/

# Foto + metrik UVI -> rekomendasi perbaikan (3 peran) + prompt gambar (JSON)
curl -X POST "http://localhost:8000/llm/advise" \
    -F "file=@data/extracted/photos/KAYUTANGAN/KAYUTANGAN_BS-01.jpg" \
    -F 'location=Alun-Alun Merdeka, Malang' \
    -F 'metrics={"vegetation_coverage_index":0.3477,"sky_openness_index":0.0223,"walkability_ratio":0.0238}' \
    -F 'scores={"beauty_score":4.2,"safety_score":3.1,"comfort_score":2.9,"uvi_score":0.36}'
# Balasan: field "advice" = objek terstruktur (diagnosis, masalah, rekomendasi,
# prompt_gambar, before_after, caveat). Semua parameter selain "file" opsional —
# tanpa metrics/scores, analisa dilakukan murni dari foto.

# Generate ulang foto + saran dari hasil /ai/process (OpenAI Images API)
curl -X POST "http://localhost:8000/llm/generate" \
    -F "result=/uploads/results/result_76117d88-42d0-4766-a2b8-97fbc4a80ecc_1ee5f69f-62d9-45f7-9db0-96f5d8aac6a4_1792582425737.json" \
    -F "size=1024x1024"
# "result" menerima result_url/result_path dari respons /ai/process,
# path artefak (mask/seg/overlay) di uploads/, path file result_*.json,
# ATAU JSON inline. Klien hanya kirim path - file sudah ada di server AI.
# Balasan: generated_image_url (PNG di uploads/generated/), raw_image_url,
# suggestions (kalimat saran, bukan LLM), prompt_image, elapsed_ms.
# Butuh OPENAI_API_KEY di .env. Model default gpt-image-1; size valid
# 1024x1024 | 1536x1024 | 1024x1536. Parameter "prompt" opsional untuk override.

# Segmentasi + skor (tanpa LLM)
curl -X POST "http://localhost:8000/ai/process" \
    -F "file=@data/testing/test_gps_exif.jpg"
# Balasan ringkas: privacy_masked_url, segmentation_url,
# segmentation_overlay_url, original_image_url, capture_location,
# segmentation_results, perception_prediction, metrics_source,
# _processing_time_ms.

# Satu langkah: segmentasi + skor + rekomendasi LLM
curl -X POST "http://localhost:8000/ai/advise" \
    -F "file=@data/testing/test_gps_exif.jpg" \
    -F "location=Koridor Kayutangan, Malang"
# Tambahan parameter: photo_id, latitude, longitude, context, model, detail.

# FeatureCollection GeoJSON dari hasil yang punya koordinat (mode lokal)
curl "http://localhost:8000/ai/geojson/results?limit=500"

# Arsip file original: /ai/process & /ai/process-video TIDAK lagi menghapus
# file upload. Semua artefak disimpan FLAT di uploads/<jenis>/ dengan skema
# nama lama yang kompatibel: <prefix>_<nama_upload>_<uuid4>_<epoch_ms>:
#   uploads/originals/, uploads/masks/, uploads/segmentation/,
#   uploads/results/, uploads/videos/, uploads/generated/
# Timestamp baru = epoch ms; file lama YYYYMMDD_HHMMSS tetap didukung resolver.
# Response memuat URL + result_path tiap artefak; tidak ada asset_id/folder aset.
# Backend menerimanya lewat callback dan menyimpannya di DB
# (segmentation_results.original_image_url / video_output_segmentations.original_video_url).
```

> **LLM gateway butuh API key.** Set `NINE_ROUTER_KEY` di `.env` (gitignored)
> atau environment service, lalu restart. Tanpa key, `/llm/health` tetap 200
> tapi `key_configured: false` dan `/llm/ask` akan gagal (upstream balas 401).
> Upstream membatasi **5 request/menit** per key — router meneruskan batas itu
> sebagai HTTP 503 berisi pesan aslinya.

### Step 5: Monitor Logs
```bash
# Access logs
tail -f logs/access.log

# Error logs  
tail -f logs/error.log

# Server output
cat api_server.log
```

---

## 🧪 Menjalankan Lokal (Windows) + Data Uji

Semua endpoint bisa dijalankan tanpa GPU cloud. Di workstation ini:

```bash
# 1. Aktifkan venv (python/python3 tidak ada di PATH)
.venv/Scripts/activate

# 2. Set PYTHONPATH ke folder src, lalu jalankan server
export PYTHONPATH=src
.venv/Scripts/python.exe -m uvicorn uvip_ai.api.main:app --host 127.0.0.1 --port 8000
# PowerShell: $env:PYTHONPATH="src"

# 3. Cek siap
curl http://127.0.0.1:8000/health
# Dokumen interaktif: http://127.0.0.1:8000/docs
```

Foto dataset **tidak menyimpan GPS** (diekspor tanpa metadata), jadi jalur EXIF
diuji dengan foto buatan. Skrip `scripts/make_test_photo.py` menempelkan GPS IFD
+ `DateTimeOriginal` ke sembarang JPEG:

```bash
.venv/Scripts/python.exe scripts/make_test_photo.py \
    --source data/extracted/photos/ALUN_ALUN_MERDEKA/ALUN_ALUN_MERDEKA_BU-03.jpg \
    --out data/testing/test_gps_exif.jpg \
    --lat -7.97685 --lon 112.6306111 --time "2026-10-22 08:31:09"
```

Hasilnya (186 KB, 807×605) dipakai di semua contoh `curl` pada panduan ini.
Tanpa file itu, `capture_location.source` bernilai `request` (koordinat dari
form klien) atau `none`.

---

## 📡 Daftar Endpoint

| Method | Path | Fungsi |
|---|---|---|
| GET | `/health` | status server + device |
| GET | `/health/ffmpeg` | cek ffmpeg + encoder video |
| POST | `/ai/process` | foto → masking + segmentasi + skor (tanpa LLM) |
| POST | `/ai/advise` | `/ai/process` + rekomendasi LLM dalam satu langkah |
| GET | `/ai/geojson/results` | `FeatureCollection` dari `uploads/results/result_*.json` |
| POST | `/ai/process-video` | video → segmentasi per frame (async, balikan `task_id`) |
| GET | `/ai/process-video/status/{task_id}` | progres task video |
| GET | `/ai/process-video/result/{task_id}` | hasil task video |
| GET | `/ai/process-video/tasks` | daftar task video |
| POST | `/llm/vision` | gambar → deskripsi/JSON (vision model) |
| POST | `/llm/advise` | gambar + metrik manual → rekomendasi |
| POST | `/llm/generate` | hasil `/ai/process` → gambar redesign + saran |
| POST | `/llm/chat` | chat multi-turn (JSON) |
| GET | `/llm/ask` | tanya satu kali (`?q=`) |
| GET/POST | `/llm/chat/completions` | kompatibel OpenAI chat |
| GET | `/llm/models`, `/llm/health` | daftar model & status upstream 9router |
| POST | `/` | handler RunPod Queue Serverless |
| GET | `/uploads/{path}` | file statis (mask, overlay, original, generated) |

### `POST /ai/process`

Multipart: `file` (wajib), `photo_id`, `latitude`, `longitude` (opsional).

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
    "captured_at": "2026-10-22T08:31:09",
    "altitude_m": 505.2,
    "direction_deg": 187.5
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
  "metrics_source": {
    "indices_source": "segformer_excel",
    "pct_by_class": {"road": 14.59, "sidewalk": 7.76, "building": 54.17,
                     "street_furniture": 8.36, "signage": 0.04,
                     "vegetation": 13.47, "other": 0.0, "sky": 1.19,
                     "pedestrian": 0.29, "vehicle": 0.14},
    "seg_map_shape": [605, 807],
    "class_count": 10
  },
  "shap_values": [],
  "_processing_time_ms": 17046
}
```

> Nama file memakai skema lama (kompatibel URL/DB yang sudah ada):
> `mask_<nama_upload>_<uuid4>_<epoch_ms>.jpg`,
> `seg_<nama_upload>_<uuid4>_<epoch_ms>.jpg`,
> `overlay_<nama_upload>_<uuid4>_<epoch_ms>.jpg`,
> `result_<nama_upload>_<uuid4>_<epoch_ms>.json`, dan original
> `<nama_upload>_<uuid4>.<ext>`. `<nama_upload>` = stem nama file upload;
> bila kosong dipakai `photo_id`. `<epoch_ms>` = `int(time.time() * 1000)`,
> satu nilai sama untuk semua artefak satu foto; bentrok milidetik diberi
> akhiran `_2`, `_3`. Video memakai `video_<task_id>` dan
> `result_video_<task_id>`; generate memakai `img_<stem>_<epoch_ms>` di
> `uploads/generated/`.

`capture_location.source` menjelaskan asal koordinat:

| Nilai | Arti |
|---|---|
| `exif_gps` | dibaca dari GPS IFD foto — posisi **saat foto diambil** |
| `request` | tidak ada EXIF GPS; dipakai `latitude`/`longitude` dari form klien |
| `none` | tidak ada keduanya; titik tidak muncul di GeoJSON |

**EXIF selalu menang** atas koordinat klien. Ini memperbaiki bug lama: aplikasi
melaporkan posisi pengunggah (bisa di kota lain), bukan lokasi pemotretan.
`captured_at` juga dari EXIF (`DateTimeOriginal`), bukan waktu upload.

### `POST /ai/advise`

Multipart: `file` (wajib), `photo_id`, `latitude`, `longitude`, `location`,
`context`, `model`, `detail` (`low|high`, default `high`).

Respons = **respons `/ai/process` + dua field tambahan**:

```json
{
  "advice": {
    "diagnosis": "Fasad arcade mendominasi bingkai (building 54.17%) dan beauty_score 9.69, tetapi comfort_score hanya 3.29 karena visual_clutter_index 0.8653 …",
    "masalah": [
      {
        "area": "kabel dan tiang",
        "temuan": "Kabel udara dan tiang utilitas di tepi kanan memotong pandangan serta zona pejalan.",
        "bukti_angka": "visual_clutter_index 0.8653; street_furniture 8.36",
        "tingkat": "tinggi"
      }
    ],
    "rekomendasi": [
      {
        "peran": "perencana_kota",
        "tindakan": "Tetapkan jalur pejalan menerus sisi arcade, lebar efektif minimum 2 m bebas hambatan, dan jadwalkan relokasi kabel udara pada segmen heritage.",
        "standar": "Permen PUPR 03/2014 tentang Pedoman Perencanaan, Penyediaan, dan Pemanfaatan Prasarana dan Sarana Jaringan Pejalan Kaki",
        "dampak": "menaikkan comfort_score",
        "prioritas": 1
      }
    ],
    "prompt_gambar": "Redesign this street, add more trees and greenery, …",
    "before_after": {"before": "…", "after": "…"},
    "caveat": "Analisa hanya dari 1 foto tanpa data 3D, topografi, volume lalu lintas, atau survei lapangan, sehingga rekomendasi masih hipotesis yang perlu verifikasi lapangan."
  },
  "advice_context": {
    "upstream": "https://9router-production-fc96.up.railway.app/v1",
    "model": "grok-4.6",
    "requested_model": "lo/glm-5.3-flash",
    "elapsed_ms": 60477,
    "parse_ok": true,
    "repair_used": false,
    "truncated": false,
    "image": {"filename": "test_gps_exif.jpg", "media_type": "image/jpeg",
              "bytes": 186198, "detail": "high"},
    "metrics_provided": ["building", "building_coverage_pct", "…"],
    "scores_provided": ["beauty_score", "comfort_score", "safety_score", "uvi_score"],
    "location": "Koridor Kayutangan, Malang"
  }
}
```

`advice_context.model` bisa berbeda dari `requested_model`: router memakai
fallback chain saat model yang diminta tidak tersedia. `parse_ok: false` berarti
balasan LLM tidak bisa diparse menjadi JSON setelah satu kali perbaikan —
`advice` tetap dikirim apa adanya (`{"raw": "..."}`).

Kalau `photo_id` diisi, callback ke backend **tidak** menyertakan `advice` /
`advice_context` — keduanya artefak tampilan, bukan data DB.

### `GET /ai/geojson/results`

`?limit=500` (default). Hanya hasil yang punya `capture_location` yang muncul —
tanpa koordinat titik tidak bisa ditempatkan di peta. `Content-Type:
application/geo+json`; urutan terbaru dulu.

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [112.6306111, -7.97685]},
      "properties": {
        "result_file": "result_76117d88-42d0-4766-a2b8-97fbc4a80ecc_1ee5f69f-62d9-45f7-9db0-96f5d8aac6a4_1792582425737.json",
        "result_path": "uploads/results/result_76117d88-42d0-4766-a2b8-97fbc4a80ecc_1ee5f69f-62d9-45f7-9db0-96f5d8aac6a4_1792582425737.json",
        "capture_source": "exif_gps",
        "captured_at": "2026-10-22T08:31:09",
        "beauty_score": 9.69,
        "safety_score": 6.12,
        "comfort_score": 3.29,
        "uvi_score": 0.67,
        "segmentation_overlay_url": "/uploads/segmentation/overlay_76117d88-42d0-4766-a2b8-97fbc4a80ecc_1ee5f69f-62d9-45f7-9db0-96f5d8aac6a4_1792582425737.jpg",
        "privacy_masked_url": "/uploads/masks/mask_76117d88-42d0-4766-a2b8-97fbc4a80ecc_1ee5f69f-62d9-45f7-9db0-96f5d8aac6a4_1792582425737.jpg",
        "original_image_url": "/uploads/originals/76117d88-42d0-4766-a2b8-97fbc4a80ecc_1ee5f69f-62d9-45f7-9db0-96f5d8aac6a4.jpg",
        "processing_time_ms": 17046
      }
    }
  ]
}
```

Untuk produksi pakai `GET /street-photos/geojson` di `backend-uvip` — sumbernya
PostGIS dan mencakup semua foto, termasuk yang belum diproses AI. Endpoint di
atas hanya untuk mode lokal tanpa DB.

### `POST /llm/generate`

Multipart: `result` (wajib — `result_url`/`result_path`, path artefak di
`uploads/`, atau JSON inline), `photo_path` (opsional), `size`, `prompt`,
`count`, `model`.

```json
{
  "generated_image_url": "/uploads/generated/img_76117d88-42d0-4766-a2b8-97fbc4a80ecc_1ee5f69f-62d9-45f7-9db0-96f5d8aac6a4_1792582500000.png",
  "generated_image_urls": ["/uploads/generated/img_76117d88-42d0-4766-a2b8-97fbc4a80ecc_1ee5f69f-62d9-45f7-9db0-96f5d8aac6a4_1792582500000.png"],
  "generated_paths": ["uploads/generated/img_76117d88-42d0-4766-a2b8-97fbc4a80ecc_1ee5f69f-62d9-45f7-9db0-96f5d8aac6a4_1792582500000.png"],
  "raw_image_url": "/uploads/originals/76117d88-42d0-4766-a2b8-97fbc4a80ecc_1ee5f69f-62d9-45f7-9db0-96f5d8aac6a4.jpg",
  "photo_path": null,
  "target": {
    "perception_prediction": {"beauty_score": 9.69, "uvi_score": 0.67, "…": "…"},
    "segmentation_results": {"green_coverage_pct": 13.4722, "…": "…"}
  },
  "suggestions": [
    "Tingkatkan tutupan vegetasi dari 13.5% dengan menambah pohon peneduh dan tanaman perdu di sepanjang koridor serta selasar bangunan.",
    "Buka pandangan langit dari 1.2% dengan merapikan kanopi pohon dan meniadakan papan iklan yang menghalangi.",
    "Perlebar dan sambungkan trotoar (saat ini 7.8% dari komposisi) agar aman dan nyaman bagi pejalan kaki."
  ],
  "prompt_image": "Redesign this street, add more trees and greenery, open up the sky view, …",
  "revised_prompt": null,
  "image_model": "gpt-image-1",
  "size": "1024x1024",
  "count": 1,
  "elapsed_ms": 68864,
  "usage": null
}
```

Nama file generated memakai timestamp (`img_<stem>_<epoch_ms>.png`), jadi
regenerate tidak menimpa hasil lama. `raw_image_url` diambil dari foto original
terbaru di aset yang sama, jadi klien tidak perlu mengirim `photo_path` lagi.

`suggestions` dihitung dari angka metrik (bukan LLM) supaya selalu konsisten
dengan `prompt_image`. Butuh `OPENAI_API_KEY`. `size`: `1024x1024` |
`1536x1024` | `1024x1536`.

### Arti skor

`perception_prediction` dihitung `uvip_ai.scoring.pvi_scorer`:

1. Segformer (mode `excel_mode`) menghasilkan 8 indeks penelitian.
2. `beauty_score` ← PVI (jumlah berbobot 8 indeks), `safety_score` ←
   `0.6*GAI + 0.4*TII`, `comfort_score` ← `0.5*VCI + 0.5*SOI`,
   `uvi_score` ← `0.4*beauty + 0.3*safety + 0.3*comfort` (rumus lama).
3. Tiap skor dinormalisasi min-max ke **1..10** memakai anchor p2/p98 dari
   segmentasi live 431 foto referensi (`scripts/calibrate_pvi.py`).

Sebaran pada 431 foto referensi (anchor dari himpunan yang sama):

| Skor | min | p2 | mean | p98 | max | std | mentok 10 |
|---|---|---|---|---|---|---|---|
| beauty | 1.00 | 1.01 | 5.66 | 9.98 | 10.00 | 2.51 | 2.1% |
| safety | 1.00 | 1.05 | 5.59 | 9.98 | 10.00 | 2.16 | 2.1% |
| comfort | 1.00 | 1.01 | 5.87 | 9.99 | 10.00 | 2.16 | 2.1% |
| uvi | 0.25 | 0.33 | 0.57 | 0.71 | 0.76 | 0.09 | — |

Nilai 2.1% di ujung skala adalah foto yang memang di luar p2/p98 — bukan
kejenuhan. Anchor bisa dihitung ulang:

```bash
.venv/Scripts/python.exe scripts/calibrate_pvi.py --write   # ~2-4 menit di GPU
```

**Keterbatasan yang harus disampaikan ke pengguna:** skor ini turunan piksel,
bukan penilaian manusia. Belum ada validasi panel penilai. Jangan sebut
"AI perception" tanpa catatan ini.


## 🔧 Complete Setup Commands

### Alternative: Manual Setup (if auto-script fails)

```bash
# 1. Verify environment
python --version
python -c "import torch; print(torch.__version__)"

# 2. Install dependencies
pip install -r requirements.txt
pip install uvicorn gunicorn

# 3. Create directories
mkdir -p models/perception data/raw data/extracted test_outputs logs

# 4. Upload models
# (Use SCP or upload_models.py)

# 5. Start server
gunicorn src.api.main:app \
    --bind 0.0.0.0:8000 \
    --workers 2 \
    --timeout 120 &
```

---

## 📦 Files Structure

```
/app/
├── scripts/
│   ├── setup_vast_ai.sh       # Initial setup
│   ├── deploy_api.sh          # Deploy API
│   └── upload_models.py       # Upload helper
├── models/
│   └── perception/
│       ├── beauty_model.pkl
│       ├── safety_model.pkl
│       ├── comfort_model.pkl
│       └── uvi_model.pkl
├── src/
│   └── api/
│       └── main.py           # FastAPI server
├── logs/
│   ├── access.log            # Request logs
│   └── error.log             # Error logs
├── test_outputs/             # Results storage
└── uploads/                   # Arsip upload (binary, diabaikan git)
    ├── originals/             # File ORIGINAL foto & video (tidak pernah dihapus)
    ├── masks/                 # Versi privacy-masked
    ├── segmentation/          # Peta segmentasi + overlay
    ├── videos/                # Video hasil segmentasi
    └── results/               # JSON hasil analisis (ikut di-version control)
```

---

## 🔐 Security Notes

1. **Never commit SSH keys** to git (already in .gitignore)
2. Use strong password for root (or disable root login)
3. Configure firewall if needed
4. Add API rate limiting in production

---

## 📊 Performance Expectations

With GTX 1070 (8GB VRAM):
- **Single image processing:** ~15-25 seconds
- **Batch size:** 4-8 images at once
- **Memory usage:** ~4-6 GB GPU RAM
- **Throughput:** ~10-20 predictions/minute

---

## 💰 Cost Calculation

Current rate: **$0.051/hour**

| Runtime | Cost |
|---------|------|
| 1 hour | $0.05 |
| 1 day | $1.22 |
| 1 week | $8.57 |
| 1 month | $36.72 |

**Recommendation:** Set up automatic billing and shut down when not in use.

---

## 🆘 Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size or process single images
```bash
# Edit predict_batch.py or adjust in config
```

### Issue: "Model not found"
**Solution:** Check models are uploaded correctly
```bash
ls -la /app/models/perception/
```

### Issue: "Port 8000 already in use"
**Solution:** Change port
```bash
gunicorn ... --bind 0.0.0.0:8001
```

### Issue: "Import error"
**Solution:** Re-run setup script
```bash
./scripts/setup_vast_ai.sh
```

---

## 🎓 Fine-tuning on Server

If you want to fine-tune SegFormer-B5 with Indonesian data:

```bash
# Prepare training data (upload to /app/data/fine_tuning/)
# Then run:
python scripts/fine_tune_segformer.py \
    --train-dir /app/data/fine_tuning/images \
    --mask-dir /app/data/fine_tuning/masks \
    --output-dir /app/models/finetuned_indonesia \
    --batch-size 8 \
    --num-epochs 15 \
    --learning-rate 1e-5
```

---

## 🌐 External Access

To access from outside:
1. Update security group rules on Vast.ai
2. Allow traffic on port 8000
3. Access at: `http://182.224.239.168:8000`

---

## 📞 Support

For issues:
1. Check logs first: `logs/error.log`
2. Review health status: `http://localhost:8000/health`
3. Consult this guide or contact support

---

## 🎉 Success Checklist

- [x] Server connected via SSH
- [x] Virtual environment activated
- [x] All dependencies installed
- [x] Trained models uploaded
- [x] API server running on port 8000
- [x] Health check passed
- [x] API docs accessible
- [x] Test prediction successful
- [x] Logs being captured
- [x] Auto-restart configured

**Congratulations! Your UVIP AI is deployed! 🚀**
