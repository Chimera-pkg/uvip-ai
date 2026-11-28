"""Infografis UVI: analisis regenerasi urban (master prompt) + render visual.

Fitur lanjutan setelah /ai/process. Alur:
1. Generate foto regenerasi urban (reuse /llm/generate -> gpt-image-1.5 edit).
2. LLM vision (deepseek-flash) menghasilkan JSON terstruktur: 10 intervensi
   (2 per 5 indikator UVI), penilaian regulasi, dan rekomendasi validasi.
3. Render infografis 2-kolom programatik (Pillow) -> PNG.

Modul ini hanya berisi PROMPT + skema + render (bukan endpoint HTTP);
endpoint /llm/infographic ada di router.py supaya bisa reuse helper yang ada.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Mapping, Optional

from PIL import Image, ImageDraw, ImageFont

# Lima indikator UVI sesuai master prompt, dipetakan ke metrik segmentasi.
# Urutan ini juga jadi urutan tampilan di infografis.
UVI_INDICATORS: tuple[tuple[str, str, str], ...] = (
    # (key JSON, label tampilan, sumber metrik di segmentation_results)
    ("building_visibility", "Building Visibility", "building_coverage_pct"),
    ("vegetation_coverage", "Vegetation Coverage", "green_coverage_pct"),
    ("sky_openness", "Sky Openness", "sky_visibility_pct"),
    ("area_accessibility", "Area Accessibility", "walkability_ratio"),
    ("visual_dominance", "Visual Dominance", "visual_clutter_index"),
)

INFOGRAPHIC_REQUIRED_KEYS: tuple[str, ...] = (
    "diagnosis", "urban_character", "heritage_status",
    "interventions", "regulatory_assessment", "validation_recommendation",
)

# System prompt: versi ramping master prompt "REGULATION-AWARE ADAPTIVE
# GENERATIVE AI FOR URBAN VISUAL INDEX (UVI) ENHANCEMENT". Ditulis bahasa
# Indonesia supaya konsisten dengan router /llm/advise. Output WAJIB JSON.
INFOGRAPHIC_SYSTEM = """Anda tim desain & perencanaan kota interdisipliner (arsitek-urban designer,
perencana tata ruang, arsitek lanskap, spesialis konservasi cagar budaya, perencana
infrastruktur pejalan & transportasi, spesialis aksesibilitas universal, spesialis
infrastruktur hijau, analis kepatuhan bangunan, dan analis kualitas visual berbasis AI).

TUGAS: analisa FOTO ruang publik eksisting + data UVI kuantitatifnya, identifikasi
defisiensi visual/spasial, lalu beri skenario regenerasi urban yang kontekstual,
realistis, dan sadar regulasi Indonesia.

ATURAN KETAT
1. Balas HANYA satu objek JSON valid. Tanpa kalimat pembuka, tanpa pagar markdown.
2. Setiap intervensi HARUS merepresentasikan modifikasi visual nyata (bukan sekadar
   "percantik"). Prioritaskan indikator dengan defisiensi TERBESAR; jangan intervensi
   indikator yang sudah baik; sebutkan trade-off antar indikator.
3. JANGAN mengarang angka, dimensi fisik, atau regulasi. Jika tidak yakin, tulis
   "Perlu verifikasi lapangan" / "Cannot Be Determined".
4. Untuk heritage: jangan klaim status legal pasti tanpa dokumen. Bedakan
   "Verified" vs "Indikasi visual" vs "Unknown".
5. Preservasi spasial WAJIB: posisi kamera, arah pandang, focal length, perspektif,
   geometri jalan, footprint & ketinggian bangunan, dan elemen heritage tetap
   dipertahankan. JANGAN menambah/mengubah awan, langit, cuaca, pencahayaan, atau
   bayangan; jangan menambah/menghapus/memindah elemen yang tidak disebut sebagai
   intervensi. Hasil harus realistis, bukan visualisasi fantasi.
6. Setiap intervensi memuat 5 field: existing_condition, proposed_intervention,
   technical_rationale, uvi_contribution, regulatory_consideration.
7. Penilaian regulasi memakai 4 status: "Verified Compliant" | "Potentially
   Compliant" | "Requires Design Revision" | "Cannot Be Determined".
8. RINGKAS: tiap teks maksimal 40 kata. Jangan ulang isi antar field.

SKEMA (semua field wajib ada):
{
  "diagnosis": "2-3 kalimat: defisiensi utama foto + kaitkan dengan angka UVI",
  "urban_character": "Historic | Commercial | Residential | Mixed-use | Civic | Institutional | Other",
  "heritage_status": "Protected Heritage Building | Heritage Area | Non-Heritage | Unknown",
  "interventions": {
    "building_visibility": [
      {"existing_condition": "...", "proposed_intervention": "...",
       "technical_rationale": "...", "uvi_contribution": "...",
       "regulatory_consideration": "..."},
      {"..."}   // TEPAT 2 item per indikator
    ],
    "vegetation_coverage": [2 item],
    "sky_openness": [2 item],
    "area_accessibility": [2 item],
    "visual_dominance": [2 item]
  },
  "regulatory_assessment": [
    {"aspect": "building | spatial_planning | pedestrian_accessibility |
               heritage_conservation | landscape_environment | road_safety",
     "status": "Verified Compliant | Potentially Compliant |
                Requires Design Revision | Cannot Be Determined",
     "note": "dukungan informasi / bukti yang hilang"}
  ],
  "validation_recommendation": "prosedur validasi UVI before/after kuantitatif"
}
"""

REGULATORY_ASPECTS = (
    "building", "spatial_planning", "pedestrian_accessibility",
    "heritage_conservation", "landscape_environment", "road_safety",
)


def infographic_schema_ok(obj: Mapping[str, Any]) -> tuple[bool, str]:
    """Cek objek infografis memenuhi skema minimum (10 intervensi, 5 indikator)."""
    missing = [k for k in INFOGRAPHIC_REQUIRED_KEYS if k not in obj]
    if missing:
        return False, f"kunci hilang: {', '.join(missing)}"
    if not isinstance(obj.get("diagnosis"), str) or not obj["diagnosis"].strip():
        return False, "'diagnosis' harus string tidak kosong"
    interventions = obj.get("interventions")
    if not isinstance(interventions, dict):
        return False, "'interventions' harus object"
    for key, _label, _src in UVI_INDICATORS:
        items = interventions.get(key)
        if not isinstance(items, list) or len(items) != 2:
            return False, f"interventions.{key} harus tepat 2 item"
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                return False, f"interventions.{key}[{i}] harus object"
            for field in ("existing_condition", "proposed_intervention",
                           "technical_rationale", "uvi_contribution",
                           "regulatory_consideration"):
                if not str(item.get(field) or "").strip():
                    return False, f"interventions.{key}[{i}].{field} wajib diisi"
    reg = obj.get("regulatory_assessment")
    if not isinstance(reg, list) or not reg:
        return False, "'regulatory_assessment' harus array tidak kosong"
    return True, ""


def build_infographic_prompt(*, metrics: Mapping[str, Any],
                             scores: Mapping[str, Any],
                             project_name: Optional[str],
                             photo_id: Optional[str],
                             location: Optional[str],
                             city: Optional[str],
                             lat: Optional[float],
                             lon: Optional[float],
                             urban_character: Optional[str],
                             heritage_status: Optional[str]) -> str:
    """Susun pesan user: metadata proyek + UVI + 5 indikator untuk LLM."""
    lines = ["Analisa foto ruang publik berikut dan susun skenario regenerasi "
             "urban (JSON sesuai skema system prompt)."]

    lines.append("\nA. INFO PROYEK")
    lines.append(f"  - Project Name: {project_name or 'Unknown'}")
    lines.append(f"  - Photo ID: {photo_id or 'Unknown'}")
    lines.append(f"  - Location: {location or 'Unknown'}")
    lines.append(f"  - City/Region: {city or 'Unknown'}")
    if lat is not None and lon is not None:
        lines.append(f"  - Coordinates: {lat:.6f}, {lon:.6f}")
    lines.append(f"  - Urban Character: {urban_character or 'Unknown (infer dari foto)'}")
    lines.append(f"  - Heritage Status: {heritage_status or 'Unknown (jangan klaim tanpa dokumen)'}")

    lines.append("\nB. SKOR UVI EKSISTING")
    uvi = scores.get("uvi_score")
    if isinstance(uvi, (int, float)):
        lines.append(f"  - UVI Score: {uvi:.2f} (0..1)")
    for k, v in scores.items():
        if k != "uvi_score" and isinstance(v, (int, float)):
            lines.append(f"  - {k}: {v:.2f}")
    if not any(isinstance(v, (int, float)) for v in scores.values()):
        lines.append("  - (UVI tidak tersedia - lakukan diagnosis visual kualitatif)")

    lines.append("\nC. LIMA INDIKATOR UVI")
    for key, label, src in UVI_INDICATORS:
        val = metrics.get(src)
        if isinstance(val, (int, float)):
            lines.append(f"  - {label}: {val:.2f}%")
        else:
            lines.append(f"  - {label}: tidak tersedia")

    lines.append("\nBalas HANYA objek JSON sesuai skema di system prompt.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Render infografis 2-kolom (Pillow)
# --------------------------------------------------------------------------

_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/calibri.ttf",
)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = _FONT_CANDIDATES if bold else (_FONT_CANDIDATES[1:] + _FONT_CANDIDATES[:1])
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_infographic_png(*, photo_path: Path, advice: Mapping[str, Any],
                           meta: Mapping[str, str],
                           uvi_score: Optional[float],
                           out_path: Path) -> Path:
    """Render layout 2-kolom: kiri foto (60%), kanan 10 intervensi (40%).

    Return path PNG yang sudah ditulis. Bukan AI-gambar: teks & angka 100% tajam.
    """
    W = 1440
    H = 960
    LEFT_W = int(W * 0.60)
    RIGHT_W = W - LEFT_W

    img = Image.new("RGB", (W, H), (18, 24, 38))
    d = ImageDraw.Draw(img)
    title_f = _font(30, bold=True)
    h_f = _font(20, bold=True)
    body_f = _font(15)
    small_f = _font(13)

    # --- Kolom kiri: foto + metadata ---
    try:
        photo = Image.open(photo_path).convert("RGB")
        photo.thumbnail((LEFT_W - 40, int(H * 0.62)))
        px = 20
        py = 20
        img.paste(photo, (px, py))
        photo_bottom = py + photo.height
    except OSError:
        photo_bottom = 20

    y = photo_bottom + 16
    for key, label in (("project_name", "Project"), ("photo_id", "Photo ID"),
                       ("location", "Lokasi"), ("city", "Kota")):
        val = meta.get(key) or "—"
        d.text((24, y), f"{label}: {val}", font=body_f, fill=(210, 218, 230))
        y += 24
    if uvi_score is not None:
        d.text((24, y), f"UVI: {uvi_score:.2f}", font=h_f, fill=(34, 197, 94))
        y += 28
    d.text((24, y), f"Urban Character: {meta.get('urban_character') or '—'}",
           font=body_f, fill=(210, 218, 230))
    y += 24
    d.text((24, y), f"Heritage: {meta.get('heritage_status') or '—'}",
           font=body_f, fill=(210, 218, 230))

    # --- Kolom kanan: 5 indikator + 10 intervensi ---
    rx = LEFT_W + 20
    ry = 20
    d.text((rx, ry), "INTERVENSI DESAIN", font=title_f, fill=(255, 255, 255))
    ry += 46
    if isinstance(advice.get("diagnosis"), str):
        for line in textwrap.wrap(advice["diagnosis"], width=46):
            d.text((rx, ry), line, font=small_f, fill=(160, 170, 185))
            ry += 17
        ry += 10

    interventions = advice.get("interventions") or {}
    for key, label, _src in UVI_INDICATORS:
        items = interventions.get(key) or []
        d.text((rx, ry), label, font=h_f, fill=(110, 165, 245))
        ry += 26
        for i, item in enumerate(items, 1):
            proposed = str(item.get("proposed_intervention") or "")[:150]
            for line in textwrap.wrap(f"{i}. {proposed}", width=42):
                d.text((rx + 10, ry), line, font=body_f, fill=(220, 226, 235))
                ry += 20
        ry += 8
        if ry > H - 200:
            break

    # --- Bawah: penilaian regulasi ---
    d.rectangle([20, H - 180, W - 20, H - 20], fill=(26, 34, 52))
    d.text((32, H - 168), "PENILAIAN REGULASI", font=h_f, fill=(255, 255, 255))
    by = H - 138
    reg = advice.get("regulatory_assessment") or []
    for item in reg[:6]:
        aspect = str(item.get("aspect") or "").replace("_", " ").title()
        status = str(item.get("status") or "")
        color = {"Verified Compliant": (34, 197, 94),
                 "Potentially Compliant": (234, 179, 8),
                 "Requires Design Revision": (239, 68, 68),
                 "Cannot Be Determined": (148, 163, 184)}.get(status, (200, 205, 215))
        d.text((32, by), f"• {aspect}", font=body_f, fill=(210, 218, 230))
        d.text((360, by), status, font=small_f, fill=color)
        by += 22

    img.save(str(out_path), quality=92)
    return out_path
