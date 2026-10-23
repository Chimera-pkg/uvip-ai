# Alur Pipeline UVIP-AI

Dokumen ini menjelaskan alur nyata di kode (`src/uvip_ai/api/main.py`), termasuk
titik-titik yang dulu bikin video gagal di server.

---

## 1. Alur Request (foto & video)

```mermaid
flowchart TD
    A[Client upload] --> B{Endpoint}
    B -->|POST /ai/process| C[post_process - sinkron]
    B -->|POST /ai/process-video| D[Buat task_id, status=queued<br/>balas 202 + task_id]

    D --> E[BackgroundTasks -> _run_video_task<br/>jalan di threadpool, bukan event loop]
    E --> F[Client polling GET /ai/process-video/status/task_id]

    C --> G[Load SegFormer-B5]
    F -.->|status=processing| F
    F -->|status=completed| H[GET /ai/process-video/result/task_id]

    G --> I[build_unified_metrics]
    E --> I
    I --> J[Response: segmentation_results +<br/>perception_prediction + metrics_source]
```

Catatan penting:
- `_run_video_task` **sync**, bukan `async`. Starlette menjalankannya di
  threadpool supaya infer PyTorch (blocking) tidak membekukan event loop.
  Kalau dijadikan `async` lagi, status/result bisa balas 404/202 palsu.
- Model di-cache lewat `get_seg_model()` (load sekali, reuse semua task).

---

## 2. Alur Video (perbaikan H.264 + anti-EAGAIN)

```mermaid
flowchart TD
    A[Video upload] --> B[Load model + VideoProcessor]
    B --> C[Cek sisa disk<br/>butuh ~400KB/frame + 200MB]
    C -->|disk kurang| X1[Gagal cepat<br/>pesan jelas]
    C -->|cukup| D[Loop frame]

    D --> E[infer -> seg_map + pct_by_class]
    E --> F[create_overlay]
    F --> G[Tulis JPG ke disk<br/>f_{i:06d}.jpg]
    G --> D

    D --> H[_encode_h264_frames]
    H --> I{Pass 1: default threads}
    I -->|gagal EAGAIN| J{Pass 2: -threads N -filter_threads 1}
    I -->|sukses| K[Verifikasi codec = H.264]
    J --> K
    K -->|bukan H.264| X2[Tolak - 'sukses palsu']
    K -->|H.264| L[Verifikasi playable<br/>_verify_playable]
    L --> M[Hapus folder frame]
    M --> N[Aggregate pct_by_class rata-rata]
    N --> O[build_unified_metrics]
```

### Kenapa frame ditulis ke disk satu per satu

Dulu semua overlay ditahan di list `output_frames` lalu ditulis setelah loop.
Untuk 408 frame 1080p itu ~2.5 GB per salinan (x2 dengan frame asli) di RAM.
Server kecil (Vast.ai, RAM ~9 GB) kehabisan memori, dan `pthread_create`
gagal -> ffmpeg melaporkan:

```
[Parsed_scale_2] Failed to configure output pad on Parsed_scale_2
Error reinitializing filters! Resource temporarily unavailable
Error while opening encoder - maybe incorrect parameters ...
```

Kunci diagnosis: **`filter=none` pun gagal**. Kalau filter yang salah, `none`
seharusnya jalan. Karena semuanya gagal, akarnya bukan filter tapi
**pembuatan thread gagal (EAGAIN)**.

### Tiga lapis penanganan

| Lapis | Apa | Di mana |
|---|---|---|
| 1 | Frame ditulis streaming ke disk, tidak ditahan di RAM | `_run_video_task` |
| 2 | Thread dibatasi sesuai RAM (`_resolve_encoder_threads`, ~40 MB/thread, maks 4) | pass 2 di `_encode_h264_frames` |
| 3 | Cek disk sebelum mulai, pesan error berisi detail | `_run_video_task`, `_run_ffmpeg` |

`_ffmpeg_version` + ringkasan stderr head+tail: penyebab asli libx264 ada di
**awal** output, jadi tidak boleh cuma ambil ekor (`err[-400:]`) - itu yang bikin
kegagalan lama cuma tampil "Generic error in an external library".

---

## 3. Pemilihan Device (fleksibel GTX / RTX / CPU)

```mermaid
flowchart TD
    A[SegformerB5.__init__] --> B{device arg?}
    B -->|diberi| C[Pakai itu]
    B -->|kosong| D{torch.cuda.is_available?}
    D -->|True| E[resolved = cuda]
    D -->|False| F[resolved = cpu<br/>'tidak ada GPU NVIDIA/CUDA']
    E --> G[Probe: tensor kecil di cuda + synchronize]
    G -->|sukses| H[Pakai CUDA<br/>log nama GPU + sm_XX]
    G -->|gagal| I[Turun ke CPU<br/>'GPU ada tapi tidak bisa dipakai']
```

Tidak ada pengecekan merek. Semua GPU NVIDIA lewat jalur yang sama:
GTX 10-series (Pascal `sm_61`), RTX 20/30/40 (Turing/Ampere/Ada), Tesla, dsb.
Yang membedakan hanya **build torch** harus punya kernel untuk `sm_XX` kartu
tersebut. Kalau tidak, probe gagal dan sistem otomatis jalan di CPU - tidak
crash. Cek hasil nyata di `GET /health`:

```json
{"gpu": "NVIDIA GeForce RTX 3060 Laptop GPU", "gpu_capability": "sm_86",
 "device_resolved": "cuda", "fp16": true,
 "ram_available_gb": 5.9, "encoder_threads": 4}
```

---

## 4. Dari Persentase ke Skor (rumus existing)

```mermaid
flowchart LR
    A[Frame / foto] --> B[SegFormer-B5<br/>19 kelas Cityscapes]
    B --> C[pct_by_class<br/>unit persen 0-100]
    C -->|video: rata-rata per frame| D[avg_pct]
    C --> E[build_unified_metrics]
    D --> E
    E --> F[beauty = clamp green*0.3 + sky*0.4<br/>+ 2 jika building > 30]
    E --> G[safety = clamp sidewalk*2 + road*0.5]
    E --> H[comfort = clamp green*0.5 + sky*0.5]
    F --> I[uvi = beauty/10*0.4<br/>+ safety/10*0.3 + comfort/10*0.3]
    G --> I
    H --> I
```

- `clamp` = `min(10, max(0, x))` -> skor 0-10.
- `green_pct` = `vegetation + tree`, `building_pct` = `building`.
- `uvi_score` rentangnya **0-1**, bukan 0-10.
- Video: `avg_pct[k] = sum(p[k] for p in frame_pcts) / n`.


---

## 4b. Arsip File: Flat, Kompatibel Nama Lama

```mermaid
flowchart LR
    A[Upload masuk] --> B["save_upload()<br/>uploads/originals/&lt;nama_upload&gt;_&lt;uuid4&gt;.&lt;ext&gt;"]
    B --> C[Proses AI<br/>mask / seg / overlay]
    C --> D["write_result() -> uploads/results/result_&lt;stem&gt;_&lt;epoch_ms&gt;.json"]
    D --> E[Response: URL + result_path tiap artefak]
    B -.->|file TIDAK dihapus| G[Arsip tetap di disk]
```

Satu foto memakai satu timestamp untuk artefak hasil proses. Semua file FLAT
per jenis (tanpa subfolder per aset), memakai skema nama lama:

```
uploads/originals/      <nama_upload>_<uuid4>.jpg
uploads/masks/          mask_<nama_upload>_<uuid4>_<epoch_ms>.jpg
uploads/segmentation/   seg_<nama_upload>_<uuid4>_<epoch_ms>.jpg
uploads/segmentation/   overlay_<nama_upload>_<uuid4>_<epoch_ms>.jpg
uploads/results/        result_<nama_upload>_<uuid4>_<epoch_ms>.json
uploads/generated/      img_<stem>_<epoch_ms>.png
uploads/videos/         video_<task_id>.mp4
uploads/results/        result_video_<task_id>.json
```

Aturan penamaan (`uvip_ai.utils.storage`):

- `<nama_upload>` = stem nama file upload (spasi/simbol → `_`); bila kosong
  dipakai `photo_id`, lalu `photo`/`video`.
- `<uuid4>` = uuid upload pada nama original; semua artefak turunan memakai
  stem lengkap original, jadi UUID tetap ada dan URL lama tetap cocok.
- `<epoch_ms>` = `int(time.time() * 1000)`; satu nilai sama untuk seg/mask/
  overlay/result satu foto. Bentrok pada milidetik yang sama diberi `_2`.
- Video memakai `task_id` delapan karakter seperti skema lama; generate `img_`.
- Resolver menerima timestamp lama `YYYYMMDD_HHMMSS` maupun `epoch_ms`.

Kontrak penting:

- File original **tidak pernah dihapus** setelah proses selesai.
- URL publik hanya untuk file di dalam `uploads/`; di luar itu `None`, bukan
  URL yang pasti 404.
- Klien tidak upload ulang: cukup kirim `result_url`/`result_path`, atau path
  artefak apa pun (`/uploads/segmentation/overlay_...jpg`). Resolver
  (`resolve_result_path`) memetakan artefak ke `result_*.json` pasangannya,
  jadi payload tidak perlu memuat JSON hasil yang besar. Artefak yang sudah
  terhapus tetap bisa dipetakan selama `result_*.json` pasangannya ada.
- Backend-uvip menerima path tersebut lewat callback dan menyimpannya di
  `segmentation_results.original_image_url` / `video_output_segmentations.original_video_url`.

---

## 5. Titik Diagnosa

| Endpoint | Untuk apa |
|---|---|
| `GET /health` | Build stamp, GPU/CPU, RAM, sisa disk, batas thread, hasil self-test |
| `GET /health/ffmpeg` | Binary mana yang benar-benar bisa encode H.264, lewat filter apa |
| `GET /ai/process-video/status/{id}` | Progres: `phase`, `frames_processed`, `total_frames` |
| `GET /ai/process-video/result/{id}` | Hasil akhir + `video_codec` + `video_playable_in_browser` |

Kalau encode gagal, folder frame JPG **tidak dihapus** supaya bisa diperiksa,
dan pesan error memuat detail ffmpeg yang gagal.
