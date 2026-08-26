# UVIP-AI Labeling Guidelines

## Cara Isi labels.csv

### Kolom yang Harus Diisi:
- `label_beauty`: Skor estetika visual (1-10)
- `label_safety`: Skor keamanan (1-10)
- `label_comfort`: Skor kenyamanan (1-10)
- `label_uvi`: Urban Vegetation Index (1-10)

---

## RUBRIK SCORING

### 🎨 BEAUTY (Estetika Visual)

**9-10 (Sangat Indah)**
- Komposisi visual sempurna
- Warna harmonis, kontras baik
- Elemen arsitektur/alam seimbang
- Tidak ada elemen mengganggu

**7-8 (Indah)**
- Komposisi bagus
- Warna menarik
- Ada elemen visual kuat
- Sedikit elemen kurang ideal

**5-6 (Cukup)**
- Komposisi standar
- Warna netral
- Tidak ada elemen standout
- Ada elemen kurang menarik

**3-4 (Kurang)**
- Komposisi lemah
- Warna monoton/kurang menarik
- Banyak elemen mengganggu
- Visual tidak menarik

**1-2 (Buruk)**
- Komposisi sangat buruk
- Warna tidak menarik
- Banyak elemen negatif (sampah, kerusakan)
- Visual sangat tidak menarik

---

### 🛡️ SAFETY (Keamanan)

**9-10 (Sangat Aman)**
- Penerangan sangat baik
- Trotoar lebar, kondisi baik
- Tidak ada hambatan
- Visibility sangat jelas
- Crosswalk/zebra cross ada

**7-8 (Aman)**
- Penerangan baik
- Trotoar cukup lebar
- Sedikit hambatan
- Visibility baik
- Ada fasilitas pejalan kaki

**5-6 (Cukup Aman)**
- Penerangan sedang
- Trotoar sempit tapi ada
- Ada hambatan minor
- Visibility sedang
- Fasilitas pejalan kaki minimal

**3-4 (Kurang Aman)**
- Penerangan buruk
- Trotoar sangat sempit/tidak ada
- Banyak hambatan
- Visibility buruk
- Tidak ada fasilitas pejalan kaki

**1-2 (Tidak Aman)**
- Gelap/tidak ada penerangan
- Tidak ada trotoar
- Banyak bahaya (lubang, konstruksi)
- Visibility sangat buruk
- Sangat berbahaya untuk pejalan kaki

---

### 🌿 COMFORT (Kenyamanan)

**9-10 (Sangat Nyaman)**
- Banyak vegetasi (pohon, taman)
- Shade/teduh alami
- Tidak panas/gerah
- Ada tempat duduk
- Atmosfer menyenangkan

**7-8 (Nyaman)**
- Cukup vegetasi
- Ada shade
- Suhu nyaman
- Ada fasilitas istirahat
- Atmosfer baik

**5-6 (Cukup Nyaman)**
- Sedikit vegetasi
- Shade minimal
- Suhu sedang
- Fasilitas terbatas
- Atmosfer netral

**3-4 (Kurang Nyaman)**
- Sangat sedikit vegetasi
- Tidak ada shade
- Panas/gerah
- Tidak ada fasilitas
- Atmosfer tidak nyaman

**1-2 (Tidak Nyaman)**
- Tidak ada vegetasi
- Exposure penuh matahari
- Sangat panas
- Tidak ada fasilitas sama sekali
- Atmosfer sangat tidak nyaman

---

### 🌳 UVI (Urban Vegetation Index)

**9-10 (Vegetasi Sangat Tinggi)**
- >70% area tertutup vegetasi
- Pohon besar, rindang
- Taman hijau luas
- Vegetasi sangat dominan

**7-8 (Vegetasi Tinggi)**
- 50-70% vegetasi
- Banyak pohon
- Area hijau signifikan
- Vegetasi dominan

**5-6 (Vegetasi Sedang)**
- 30-50% vegetasi
- Beberapa pohon
- Area hijau sedang
- Vegetasi dan built environment seimbang

**3-4 (Vegetasi Rendah)**
- 10-30% vegetasi
- Sedikit pohon
- Area hijau minimal
- Built environment dominan

**1-2 (Vegetasi Sangat Rendah)**
- <10% vegetasi
- Hampir tidak ada pohon
- Tidak ada area hijau
- Fully built environment

---

## TIPS LABELING

1. **Konsisten**: Gunakan kriteria yang sama untuk semua foto
2. **Relatif**: Bandingkan foto satu dengan yang lain
3. **Context**: Pertimbangkan fungsi area (jalan utama vs taman)
4. **Time of day**: Perhatikan waktu foto diambil (jika ada metadata)
5. **Weather**: Perhatikan kondisi cuaca (jika terlihat)

## TOOLS YANG MEMBANTU

### Labeling Tools:
1. **Label Studio** (open source, web-based)
   - Install: `pip install label-studio`
   - Run: `label-studio start`
   - Fitur: Image classification, custom labels, team collaboration

2. **CVAT** (Computer Vision Annotation Tool)
   - Web-based, open source
   - Good for image classification
   - Support team labeling

3. **Excel/Google Sheets**
   - Simple, no setup
   - Good untuk dataset kecil (<1000 images)
   - Template: `labels_template.csv`

### Data Collection:
1. **OpenStreetMap** - untuk metadata lokasi
2. **Google Street View** - untuk augmentasi data
3. **Drone footage** - untuk aerial view

### Quality Check:
1. **Inter-annotator agreement** - multiple labelers untuk same images
2. **Consistency check** - review outliers
3. **Statistical analysis** - cek distribusi skor

---

## CONTOH PENGISIAN

```csv
filename,area,point_id,label_beauty,label_safety,label_comfort,label_uvi,notes
ALUN_ALUN_MERDEKA_UB-01.jpg,ALUN_ALUN_MERDEKA,ALUN_ALUN_MERDEKA_UB-01,7,8,6,5,Taman kota dengan pohon rindang
KAYUTANGAN_ST-01.jpg,KAYUTANGAN,KAYUTANGAN_ST-01,5,4,3,2,Jalan utama dengan sedikit pohon
```

---

## ESTIMASI WAKTU

- **431 foto × 30 detik/foto = ~3.5 jam**
- Dengan Label Studio (lebih cepat): ~2 jam
- Dengan 2 labelers: ~1 jam

---

## NEXT STEPS

1. Pilih tool labeling (Label Studio recommended)
2. Import 431 foto
3. Label semua foto dengan rubrik di atas
4. Export ke CSV format
5. Upload ke Kaggle
6. Re-train model dengan real labels

Expected improvement: R² dari 0.2-0.3 → 0.6-0.8
