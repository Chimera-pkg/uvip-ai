"""Tuning skor PVI: bobot, campuran, anchor, dan skala - per request.

Modul ini adalah satu-satunya tempat parameter skor divalidasi dan
dinormalisasi sebelum masuk ``pvi_scorer``. Dipakai oleh:

* ``POST /ai/process`` dan ``POST /ai/advise`` (form ``tuning``/``profile``)
* ``POST /ai/score/preview`` (hitung ulang skor dari 8 indeks, tanpa GPU)
* ``GET/POST/DELETE /ai/scoring/profiles`` (profil bernama di disk)
* ``POST /ai/scoring/evaluate`` (sebaran skor pada dataset referensi)

Aturan penting:

* **Default identik dengan sebelum fitur ini ada.** ``DEFAULT_TUNING`` disalin
  dari konstanta ``pvi_scorer``; tanpa ``tuning``, output byte-per-byte sama
  (kecuali field baru ``scoring``/``perception_prediction_baseline``).
* **Tidak ada state global yang dimutasi.** Semua parameter diteruskan sebagai
  argumen, jadi dua request dengan tuning berbeda tidak saling mengganggu.
* **Input parsial di-merge di atas default** (atau di atas profil), lalu
  bobot dinormalisasi ke jumlah 1.0. Bobot efektif selalu dikembalikan.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from uvip_ai.scoring.pvi_scorer import (
    CALIBRATION,
    COMFORT_WEIGHTS,
    INDEX_KEYS,
    PVI_WEIGHTS,
    SAFETY_WEIGHTS,
    SCORE_MAX,
    SCORE_MIN,
    UVI_WEIGHTS,
    _clamp,
    compute_scores,
)
from uvip_ai.utils.storage import write_json_atomic

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROFILES_DIR = PROJECT_ROOT / "data" / "scoring" / "profiles"
REFERENCE_CSV = PROJECT_ROOT / "data" / "scoring" / "pvi_per_photo.csv"

SCORING_VERSION = "pvi-v1"
SCORE_FIELDS = ("beauty_score", "safety_score", "comfort_score")
MIN_ANCHOR_SPAN = 0.005
ANCHOR_RANGE = (-0.5, 1.5)
SCALE_RANGE = (0.0, 10.0)
MIN_SCALE_SPAN = 1.0
PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

DEFAULT_TUNING: dict[str, Any] = {
    "weights": dict(PVI_WEIGHTS),
    "blend": {
        "safety": dict(SAFETY_WEIGHTS),
        "comfort": dict(COMFORT_WEIGHTS),
        "uvi": dict(UVI_WEIGHTS),
    },
    "anchors": {key: dict(value) for key, value in CALIBRATION.items()},
    "scale": {"min": SCORE_MIN, "max": SCORE_MAX},
}

_BLEND_GROUPS: dict[str, tuple[Mapping[str, float], tuple[str, ...]]] = {
    "safety": (SAFETY_WEIGHTS, tuple(SAFETY_WEIGHTS)),
    "comfort": (COMFORT_WEIGHTS, tuple(COMFORT_WEIGHTS)),
    "uvi": (UVI_WEIGHTS, ("beauty", "safety", "comfort")),
}

_WEIGHT_LABELS = {
    "building_visibility_index": "Building Visibility (BVI)",
    "vegetation_coverage_index": "Vegetation Coverage (VCI)",
    "sky_openness_index": "Sky Openness (SOI)",
    "ground_accessibility_index": "Ground Accessibility (GAI)",
    "human_activity_index": "Human Activity (HAI)",
    "vehicle_intensity_index": "Vehicle Intensity (VII)",
    "traffic_infrastructure_index": "Traffic Infrastructure (TII)",
    "heritage_dominance_index": "Heritage Dominance (HDI)",
}

_BLEND_LABELS = {
    "safety": {
        "ground_accessibility_index": "GAI di safety",
        "traffic_infrastructure_index": "TII di safety",
    },
    "comfort": {
        "vegetation_coverage_index": "VCI di comfort",
        "sky_openness_index": "SOI di comfort",
    },
    "uvi": {"beauty": "Beauty di UVI", "safety": "Safety di UVI", "comfort": "Comfort di UVI"},
}

_ANCHOR_LABELS = {
    "pvi": "PVI (beauty)",
    "safety_raw": "Safety mentah",
    "comfort_raw": "Comfort mentah",
}


class TuningError(ValueError):
    """Input tuning tidak valid. Endpoint memetakan ini ke HTTP 422."""


# --------------------------------------------------------------------------
# Validasi + normalisasi
# --------------------------------------------------------------------------
def _num(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TuningError(f"{field} harus berupa angka, bukan {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise TuningError(f"{field} harus angka berhingga")
    return number


def _check_keys(obj: Mapping[str, Any], allowed: Mapping[str, Any] | tuple[str, ...],
                prefix: str) -> None:
    unknown = sorted(set(obj) - set(allowed))
    if unknown:
        raise TuningError(
            f"{prefix}: key tidak dikenal {unknown}; yang dikenal {sorted(allowed)}")


def _merge_weights(base: Mapping[str, float], raw: Any,
                   allowed: tuple[str, ...], prefix: str,
                   order: tuple[str, ...] | None = None) -> tuple[dict[str, float], bool]:
    """Gabungkan bobot parsial di atas ``base`` lalu normalisasi ke 1.0.

    ``raw`` boleh mapping (nama -> bobot) atau list sesuai ``order`` (dipakai
    bentuk ringkas ``[0.6, 0.4]``). Mengembalikan ``(bobot, dinormalisasi)``.
    """
    if raw is None:
        return dict(base), False
    if isinstance(raw, (list, tuple)):
        if order is None or len(raw) != len(order):
            raise TuningError(f"{prefix}: list harus berisi {len(order or ())} angka")
        raw = dict(zip(order, raw))
    if not isinstance(raw, Mapping):
        raise TuningError(f"{prefix} harus object atau list angka")

    merged = {key: float(base.get(key, 0.0)) for key in allowed}
    for key, value in raw.items():
        merged[key] = _num(value, f"{prefix}.{key}")
    _check_keys(raw, allowed, prefix)
    for key, value in merged.items():
        if not 0.0 <= value <= 1.0:
            raise TuningError(
                f"{prefix}.{key}={value} di luar 0..1 - kirim rasio, bukan persen")
    total = sum(merged.values())
    if total <= 0.0:
        raise TuningError(f"{prefix}: total bobot harus lebih besar dari 0")
    normalized = {key: round(value / total, 6) for key, value in merged.items()}
    # Tutup sisa pembulatan pada bobot terbesar supaya jumlahnya tepat 1.0.
    residual = round(1.0 - sum(normalized.values()), 6)
    if residual:
        largest = max(normalized, key=normalized.get)
        normalized[largest] = round(normalized[largest] + residual, 6)
    return normalized, abs(total - 1.0) > 1e-9


def _merge_blend(base: Mapping[str, Mapping[str, float]], raw: Any
                 ) -> tuple[dict[str, dict[str, float]], bool]:
    if raw is None:
        return copy.deepcopy(dict(base)), False
    if not isinstance(raw, Mapping):
        raise TuningError("blend harus object {safety, comfort, uvi}")
    _check_keys(raw, _BLEND_GROUPS, "blend")
    out = copy.deepcopy(dict(base))
    changed = False
    for group, subgroup in raw.items():
        allowed, order = _BLEND_GROUPS[group]
        out[group], normalized = _merge_weights(
            base.get(group, allowed), subgroup, tuple(allowed), f"blend.{group}", order)
        changed = changed or normalized
    return out, changed


def _merge_anchors(base: Mapping[str, Mapping[str, float]], raw: Any
                   ) -> tuple[dict[str, dict[str, float]], bool]:
    if raw is None:
        return copy.deepcopy(dict(base)), False
    if not isinstance(raw, Mapping):
        raise TuningError("anchors harus object {pvi, safety_raw, comfort_raw}")
    _check_keys(raw, base, "anchors")
    out = copy.deepcopy(dict(base))
    for group, pair in raw.items():
        if not isinstance(pair, Mapping):
            raise TuningError(f"anchors.{group} harus object {{min, max}}")
        _check_keys(pair, ("min", "max"), f"anchors.{group}")
        low = _num(pair.get("min", out[group]["min"]), f"anchors.{group}.min")
        high = _num(pair.get("max", out[group]["max"]), f"anchors.{group}.max")
        if not ANCHOR_RANGE[0] <= low < high <= ANCHOR_RANGE[1]:
            raise TuningError(
                f"anchors.{group}: butuh min < max di dalam rentang "
                f"{ANCHOR_RANGE[0]}..{ANCHOR_RANGE[1]} (min={low}, max={high})")
        if high - low < MIN_ANCHOR_SPAN:
            raise TuningError(
                f"anchors.{group}: rentang terlalu sempit ({high - low:.4f} < "
                f"{MIN_ANCHOR_SPAN}); skor akan mentok di kedua ujung")
        out[group] = {"min": low, "max": high}
    return out, True


def _merge_scale(base: Mapping[str, float], raw: Any
                 ) -> tuple[dict[str, float], bool]:
    if raw is None:
        return dict(base), False
    if not isinstance(raw, Mapping):
        raise TuningError("scale harus object {min, max}")
    _check_keys(raw, ("min", "max"), "scale")
    low = _num(raw.get("min", base["min"]), "scale.min")
    high = _num(raw.get("max", base["max"]), "scale.max")
    if not SCALE_RANGE[0] <= low < high <= SCALE_RANGE[1]:
        raise TuningError(
            f"scale: butuh {SCALE_RANGE[0]} <= min < max <= {SCALE_RANGE[1]}")
    if high - low < MIN_SCALE_SPAN:
        raise TuningError(
            f"scale: rentang minimal {MIN_SCALE_SPAN} (min={low}, max={high})")
    return {"min": low, "max": high}, True


def parse_tuning(raw: Any, *, base: Mapping[str, Any] | None = None
                 ) -> tuple[dict[str, Any], bool, list[str]]:
    """Validasi + normalisasi tuning.

    Returns ``(tuning_lengkap, dinormalisasi, warnings)``. ``base`` = tuning
    lengkap yang menjadi titik awal (dipakai saat menumpuk inline di atas
    profil tersimpan).
    """
    tuning = copy.deepcopy(dict(base or DEFAULT_TUNING))
    if raw is None:
        return tuning, False, []
    if not isinstance(raw, Mapping):
        raise TuningError("tuning harus object JSON")

    _check_keys(raw, ("weights", "blend", "anchors", "scale"), "tuning")
    warnings: list[str] = []
    normalized_any = False

    tuning["weights"], changed = _merge_weights(
        tuning["weights"], raw.get("weights"), INDEX_KEYS, "weights")
    normalized_any = normalized_any or changed
    if changed:
        warnings.append("bobot PVI tidak berjumlah 1.0 -> dinormalisasi proporsional")

    tuning["blend"], changed = _merge_blend(tuning["blend"], raw.get("blend"))
    normalized_any = normalized_any or changed
    if changed:
        warnings.append("bobot campuran tidak berjumlah 1.0 -> dinormalisasi proporsional")

    tuning["anchors"], _ = _merge_anchors(tuning["anchors"], raw.get("anchors"))
    tuning["scale"], _ = _merge_scale(tuning["scale"], raw.get("scale"))
    return tuning, normalized_any, warnings


def tuning_hash(tuning: Mapping[str, Any]) -> str:
    """Hash 8 karakter dari tuning efektif (audit: dua hasil sebanding?)."""
    canonical = json.dumps(tuning, sort_keys=True, separators=(",", ":"),
                           default=float)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


DEFAULT_TUNING_HASH = tuning_hash(DEFAULT_TUNING)


# --------------------------------------------------------------------------
# Skor dari 8 indeks
# --------------------------------------------------------------------------
def score_indices(indices: Mapping[str, Any] | None,
                  tuning: Mapping[str, Any] | None = None,
                  *,
                  profile: str = "default",
                  normalized: bool = False,
                  warnings: list[str] | None = None) -> dict[str, Any]:
    """8 indeks -> ``perception_prediction`` + blok audit ``scoring``.

    ``indices`` yang tidak lengkap diisi 0.0 dan dilaporkan di ``warnings``;
    nilai di luar 0..1 di-clamp dengan peringatan (bukan error) karena ini
    jalur hitung ulang, bukan input konfigurasi.
    """
    tuning = tuning or DEFAULT_TUNING
    notes = list(warnings or [])
    clean: dict[str, float] = {}
    for key in INDEX_KEYS:
        if not indices or key not in indices or indices.get(key) is None:
            clean[key] = 0.0
            notes.append(f"indeks '{key}' tidak ada -> 0.0")
            continue
        try:
            value = float(indices[key])
        except (TypeError, ValueError):
            raise TuningError(f"indices.{key} harus angka") from None
        if not math.isfinite(value):
            raise TuningError(f"indices.{key} harus angka berhingga")
        if not 0.0 <= value <= 1.0:
            notes.append(f"indeks '{key}'={value} di luar 0..1 -> di-clamp")
        clean[key] = _clamp(value)

    scores = compute_scores(
        clean,
        calibration=tuning["anchors"],
        weights=tuning["weights"],
        blend=tuning["blend"],
        scale=tuning["scale"],
    )
    raw = scores.pop("_raw")
    low = float(tuning["scale"]["min"])
    high = float(tuning["scale"]["max"])
    at_bound = [
        field for field in SCORE_FIELDS
        if scores[field] <= low + 1e-9 or scores[field] >= high - 1e-9
    ]

    block = {
        "version": SCORING_VERSION,
        "profile": profile,
        "profile_hash": tuning_hash(tuning),
        "normalized": bool(normalized),
        "weights": {key: round(float(value), 6) for key, value in tuning["weights"].items()},
        "blend": {
            group: {key: round(float(value), 6) for key, value in weights.items()}
            for group, weights in tuning["blend"].items()
        },
        "anchors": {
            group: {key: round(float(value), 6) for key, value in pair.items()}
            for group, pair in tuning["anchors"].items()
        },
        "scale": {"min": low, "max": high},
        "indices": {key: round(clean[key], 4) for key in INDEX_KEYS},
        "raw": raw,
        "at_bound": at_bound,
        "warnings": notes,
    }
    result = {"perception_prediction": scores, "scoring": block}
    if block["profile_hash"] != DEFAULT_TUNING_HASH:
        # Uji slider butuh pembanding: skor dengan default (anchor 1..10).
        baseline = compute_scores(clean)
        baseline.pop("_raw", None)
        result["perception_prediction_baseline"] = baseline
    return result


# --------------------------------------------------------------------------
# Profil bernama
# --------------------------------------------------------------------------
def _profile_path(name: str) -> Path:
    if not isinstance(name, str) or not PROFILE_NAME_RE.match(name):
        raise TuningError("nama profil harus 1-64 karakter [A-Za-z0-9_-]")
    return PROFILES_DIR / f"{name}.json"


def list_profiles() -> list[dict[str, Any]]:
    """Daftar profil tersimpan (yang rusak dilewati, bukan bikin 500)."""
    if not PROFILES_DIR.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(PROFILES_DIR.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        records.append({
            "name": record.get("name") or path.stem,
            "description": record.get("description", ""),
            "created_at": record.get("created_at"),
            "profile_hash": record.get("profile_hash"),
            "scoring_version": record.get("scoring_version"),
            "tuning": record.get("tuning"),
        })
    return records


def load_profile(name: str) -> dict[str, Any]:
    path = _profile_path(name)
    if not path.is_file():
        raise KeyError(name)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TuningError(f"profil '{name}' rusak: {exc}") from None
    if not isinstance(record, dict) or not isinstance(record.get("tuning"), dict):
        raise TuningError(f"profil '{name}' tidak memuat 'tuning'")
    tuning, normalized, warnings = parse_tuning(record["tuning"])
    return {
        "name": record.get("name") or name,
        "description": record.get("description", ""),
        "created_at": record.get("created_at"),
        "scoring_version": record.get("scoring_version", SCORING_VERSION),
        "profile_hash": tuning_hash(tuning),
        "tuning": tuning,
        "normalized": normalized,
        "warnings": warnings,
    }


def save_profile(name: str, tuning_raw: Any, description: str = "") -> dict[str, Any]:
    path = _profile_path(name)
    tuning, normalized, warnings = parse_tuning(tuning_raw)
    record = {
        "name": name,
        "description": (description or "").strip()[:280],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scoring_version": SCORING_VERSION,
        "profile_hash": tuning_hash(tuning),
        "tuning": tuning,
        "normalized": normalized,
        "warnings": warnings,
    }
    write_json_atomic(path, record)
    return record


def delete_profile(name: str) -> bool:
    path = _profile_path(name)
    if not path.is_file():
        return False
    path.unlink()
    return True


def resolve(*, tuning: Any = None, profile: str | None = None) -> dict[str, Any]:
    """Siapkan spesifikasi scorer dari ``profile`` + ``tuning`` inline.

    ``tuning`` boleh dict (endpoint JSON) atau string JSON (form multipart).
    Inline menimpa nilai profil, lalu seluruhnya divalidasi ulang.
    """
    if isinstance(tuning, str):
        text = tuning.strip()
        if text:
            try:
                tuning = json.loads(text)
            except json.JSONDecodeError as exc:
                raise TuningError(f"tuning bukan JSON valid: {exc.msg}") from None
        else:
            tuning = None
    if tuning is not None and not isinstance(tuning, Mapping):
        raise TuningError("tuning harus object JSON")

    base = None
    label = "default"
    if profile:
        record = load_profile(profile)
        base = record["tuning"]
        label = profile
    if tuning is not None:
        label = f"{profile}+inline" if profile else "inline"

    effective, normalized, warnings = parse_tuning(tuning, base=base)
    return {
        "tuning": effective,
        "profile": label,
        "normalized": normalized,
        "warnings": warnings,
        "profile_hash": tuning_hash(effective),
    }


# --------------------------------------------------------------------------
# Evaluasi sebaran pada dataset referensi
# --------------------------------------------------------------------------
def _percentile(ordered: list[float], pct: float) -> float:
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * (pct / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _stats(values: list[float], low: float, high: float) -> dict[str, float]:
    ordered = sorted(values)
    count = len(ordered) or 1
    mean = sum(ordered) / count
    variance = sum((value - mean) ** 2 for value in ordered) / count
    return {
        "mean": round(mean, 3),
        "std": round(math.sqrt(variance), 3),
        "p2": round(_percentile(ordered, 2), 3),
        "p98": round(_percentile(ordered, 98), 3),
        "min": round(ordered[0], 3) if ordered else 0.0,
        "max": round(ordered[-1], 3) if ordered else 0.0,
        "at_ceiling_pct": round(
            100.0 * sum(1 for value in ordered if value >= high - 1e-9) / count, 2),
        "at_floor_pct": round(
            100.0 * sum(1 for value in ordered if value <= low + 1e-9) / count, 2),
    }


def evaluate_reference(tuning: Mapping[str, Any] | None = None,
                       csv_path: str | Path | None = None) -> dict[str, Any]:
    """Sebaran skor seluruh dataset referensi untuk satu tuning (tanpa GPU).

    Dataset default ``data/scoring/pvi_per_photo.csv`` menyimpan 8 indeks per
    foto hasil segmentasi live, jadi mengubah slider tidak perlu infer ulang.
    """
    path = Path(csv_path) if csv_path else REFERENCE_CSV
    if not path.is_file():
        raise FileNotFoundError(f"dataset referensi tidak ada: {path}")

    tuning = tuning or DEFAULT_TUNING
    low = float(tuning["scale"]["min"])
    high = float(tuning["scale"]["max"])

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [key for key in INDEX_KEYS if key not in (reader.fieldnames or ())]
        if missing:
            raise TuningError(f"dataset referensi tidak punya kolom indeks: {missing}")
        rows = list(reader)
    if not rows:
        raise TuningError(f"dataset referensi kosong: {path}")

    buckets: dict[str, list[float]] = {field: [] for field in SCORE_FIELDS}
    buckets["uvi_score"] = []
    by_area: dict[str, list[float]] = {}
    for row in rows:
        indices = {}
        for key in INDEX_KEYS:
            try:
                indices[key] = float(row[key])
            except (TypeError, ValueError):
                raise TuningError(f"nilai indeks tidak valid di {path.name}: {key}") from None
        scored = compute_scores(
            indices,
            calibration=tuning["anchors"],
            weights=tuning["weights"],
            blend=tuning["blend"],
            scale=tuning["scale"],
        )
        for field in buckets:
            buckets[field].append(float(scored[field]))
        area = (row.get("area") or "TANPA_AREA").strip() or "TANPA_AREA"
        by_area.setdefault(area, []).append(float(scored["beauty_score"]))

    scores = {
        "beauty": _stats(buckets["beauty_score"], low, high),
        "safety": _stats(buckets["safety_score"], low, high),
        "comfort": _stats(buckets["comfort_score"], low, high),
        "uvi": _stats(buckets["uvi_score"], 0.0, 1.0),
    }
    warnings: list[str] = []
    for name, stats in scores.items():
        if stats["at_ceiling_pct"] > 10.0:
            warnings.append(
                f"{name}: {stats['at_ceiling_pct']}% foto mentok batas atas "
                f"({high:g}) - anchor terlalu sempit")
        if stats["at_floor_pct"] > 10.0:
            warnings.append(
                f"{name}: {stats['at_floor_pct']}% foto mentok batas bawah "
                f"({low:g}) - anchor terlalu sempit")
        if stats["std"] < 0.5 and name != "uvi":
            warnings.append(f"{name}: sebaran terlalu sempit (std {stats['std']})")

    return {
        "n_photos": len(rows),
        "source": path.name,
        "scores": scores,
        "by_area": [
            {"area": area, "n": len(values),
             "beauty_mean": round(sum(values) / len(values), 3)}
            for area, values in sorted(by_area.items())
        ],
        "warnings": warnings,
    }


# --------------------------------------------------------------------------
# Skema slider untuk UI
# --------------------------------------------------------------------------
def schema() -> dict[str, Any]:
    """Definisi slider: satu sumber kebenaran untuk demo/dashboard."""
    default = DEFAULT_TUNING

    def weight_slider(path: str, label: str, value: float) -> dict[str, Any]:
        return {"path": path, "label": label, "default": round(float(value), 6),
                "min": 0.0, "max": 1.0, "step": 0.01, "unit": "rasio"}

    weight_sliders = [
        weight_slider(f"weights.{key}", _WEIGHT_LABELS.get(key, key),
                      default["weights"][key])
        for key in INDEX_KEYS
    ]
    blend_sliders = []
    for group in ("safety", "comfort", "uvi"):
        for key, value in default["blend"][group].items():
            blend_sliders.append(weight_slider(
                f"blend.{group}.{key}",
                _BLEND_LABELS[group].get(key, f"{key} di {group}"), value))

    anchor_groups = [
        {
            "group": group,
            "label": _ANCHOR_LABELS.get(group, group),
            "path": f"anchors.{group}",
            "min_default": pair["min"],
            "max_default": pair["max"],
            "min": ANCHOR_RANGE[0],
            "max": ANCHOR_RANGE[1],
            "min_span": MIN_ANCHOR_SPAN,
            "step": 0.001,
        }
        for group, pair in default["anchors"].items()
    ]

    return {
        "version": SCORING_VERSION,
        "defaults": default,
        "score_fields": list(SCORE_FIELDS) + ["uvi_score"],
        "groups": [
            {
                "id": "weights",
                "label": "Bobot PVI",
                "kind": "weights",
                "normalized": True,
                "help": "Mengubah peringkat foto relatif satu sama lain.",
                "sliders": weight_sliders,
            },
            {
                "id": "blend",
                "label": "Campuran skor",
                "kind": "weights",
                "normalized": True,
                "help": "Bagaimana indeks digabung menjadi safety/comfort/UVI.",
                "sliders": blend_sliders,
            },
            {
                "id": "anchors",
                "label": "Anchor kalibrasi (p2/p98)",
                "kind": "ranges",
                "help": "Meregangkan skala tanpa mengubah urutan.",
                "ranges": anchor_groups,
            },
            {
                "id": "scale",
                "label": "Batas skor",
                "kind": "scale",
                "help": "Batas bawah/atas skor 0..10. UVI dibagi nilai maksimum.",
                "sliders": [
                    {"path": "scale.min", "label": "Skor minimum",
                     "default": default["scale"]["min"],
                     "min": SCALE_RANGE[0], "max": SCALE_RANGE[1] - MIN_SCALE_SPAN,
                     "step": 0.5},
                    {"path": "scale.max", "label": "Skor maksimum",
                     "default": default["scale"]["max"],
                     "min": SCALE_RANGE[0] + MIN_SCALE_SPAN, "max": SCALE_RANGE[1],
                     "step": 0.5},
                ],
            },
        ],
        "presets": [
            {"name": record["name"], "description": record["description"]}
            for record in list_profiles()
        ],
    }
